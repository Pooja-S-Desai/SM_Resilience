# Switch_migration_optimizer_MCF_.py
# Path-based MCF (precomputed K paths) + load-based objectives
# Migration-cost RT term uses GLOBAL mean RT increase:
#   ΔmeanRT_pos = max(0, meanRT_final - meanRT_init)
# Propagation RT per switch is MAX cost among used K-paths (non-zero flow).

from __future__ import annotations

import os
import time
import math
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from link_checks import extract_mcf_solution_bundle
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from helpers import (
    CAPACITY_THRESHOLD as GLOBAL_THRESHOLD,
    RESULTS_FOLDER,
    TIME_LIMIT as time_limit
)
from rt_metrics import build_paths_sc_from_switch_paths
# -----------------------------
# utilities
# -----------------------------
def _undir(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _path_cost_and_edges(G, nodes):
    edges = [_undir(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    cost = 0.0
    for u, v in edges:
        if G.has_edge(u, v):
            cost += float(G[u][v].get("temp_cost", G[u][v].get("weight", 1.0)))
        elif G.has_edge(v, u):
            cost += float(G[v][u].get("temp_cost", G[v][u].get("weight", 1.0)))
        else:
            cost += 1.0
    return cost, edges


def run_migration_optimizer_integrated_mcf(
    G,
    switches: List[int],
    controllers: List[int],
    loads: Dict[int, float],
    capacities: Dict[int, float],
    init_assign: Dict[int, int],
    *,
    objective_type: str = "variance",
    topology_name: Optional[str] = None,

    # --- Migration-cost inputs ---
    Dcc: Optional[dict] = None,          # controller-to-controller distance (same structure you already use)
    sync_per_ctrl_ms: float = 0.0,       # scalar sync penalty added to each switch RT (ms) via assigned controller
    w_mig: float = 1.0,                  # weight on #migrations
    w_cc: float = 1.0,                   # weight on controller-to-controller transfer
    w_rt: float = 1.0,                   # weight on ΔmeanRT_pos (ms)

    # --- Link caps / demand params ---
    edge_caps_e: Optional[dict] = None,  # {(min(u,v),max(u,v)): cap_bits}
    msg_bits: int = 128,

    # --- K-path cache inputs ---
    sc_kpaths: Optional[dict] = None,          # {(s,c): [path0, path1, ...]}  (path entries can be dict or list)
    sc_kpath_cost_ms: Optional[dict] = None,   # {(s,c,k): cost_ms}  (your cached RTT/prop ms)
    sc_kpath_edges: Optional[dict] = None,     # {(s,c,k): [(u,v),...]} (undirected edges along path)

    allow_path_splitting: bool = False,

    # --- RT / queue model knobs ---
    init_mean_rt_ms: Optional[float] = None,   # MUST be computed in main once per topo/run using same RT definition
    init_rt_ms_by_switch: Optional[Dict[int, float]] = None,  #
    rho_max: float = 0.95,                     # keep λ within rho_max * μ
    pwl_segments: int = 12,                    # PWL segments for W(λ)
    eta_path: float = 1e-9,                    # tiny tie-breaker: prefers using shorter paths when feasible
    # --- NEW: Path-based MCF inputs (like Switch_migration_optimizer_MCF_.py) ---

    alpha:float,
    beta:float,
  

):
    """
    Variables:
      y[s,c]      ∈ {0,1}   assignment
      f[s,c,k]    ∈ [0,1]   path split fractions (or {0,1} if not splitting)
      u[s,c,k]    ∈ {0,1}   path-used indicator (u=1 iff f>0)
      z[s]        ∈ {0,1}   migration indicator

      lam[c]      ≥ 0       controller load (req/s)
      W_ms[c]     ≥ 0       queueing/system time (ms), PWL approx of 1/(μ-λ)
      prop_ms[s]  ≥ 0       propagation RTT for switch s = max(cost_ms of used paths)
      T_ms[s]     ≥ 0       total RT for switch s = prop_ms[s] + W_ms[assigned c] + sync_ms

      mean_T_ms         mean RT across all switches (ms)
      delta_mean_rt_pos max(0, mean_T_ms - init_mean_rt_ms)

    Objective:
      base_obj(load-based) + migration_cost
    where migration_cost includes:
      w_mig * (#migrations) + w_cc * (cc_transfer) + w_rt * (ΔmeanRT_pos)
    """

    # -----------------------------
    # validation
    # -----------------------------
    if sc_kpaths is None or sc_kpath_cost_ms is None:
        raise ValueError("MCF path-based optimizer requires sc_kpaths and sc_kpath_cost_ms from preprocessing.")

    if edge_caps_e is None:
        edge_caps_e = {}

    # undirected edge list for link caps
    E = [_undir(u, v) for (u, v) in G.edges()]
    E = list(dict.fromkeys(E))  # unique preserve or
    # demand bits/s per switch
    demand_bits = {s: float(loads.get(s, 0.0)) * float(msg_bits) for s in switches}

    # Ensure every reachable switch-controller pair has at least one path.
    # The precomputed K-path cache can be sparse after filtering; MCF still
    # needs a complete assignment domain.
    for s in switches:
        for c in controllers:
            if s == c:
                continue
            plist = sc_kpaths.setdefault((s, c), [])
            has_cost = any((s, c, k) in sc_kpath_cost_ms for k in range(len(plist)))
            if has_cost:
                continue
            try:
                nodes = list(nx.shortest_path(G, s, c, weight="weight"))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            cost, edges = _path_cost_and_edges(G, nodes)
            k = len(plist)
            plist.append({"nodes": nodes, "edges": edges, "cost": cost})
            sc_kpath_cost_ms[(s, c, k)] = cost
            if sc_kpath_edges is not None:
                sc_kpath_edges[(s, c, k)] = edges

    # allowed (s,c): must have >=1 cached/fallback path if s!=c; allow (c,c)
    allowed_pairs = []
    for s in switches:
        for c in controllers:
            if s == c:
                allowed_pairs.append((s, c))
            else:
                plist = sc_kpaths.get((s, c), [])
                if plist:
                    # also require at least one cost entry
                    any_cost = any(((s, c, k) in sc_kpath_cost_ms) for k in range(len(plist)))
                    if any_cost:
                        allowed_pairs.append((s, c))

    # if a switch has no reachable controller, return a clean failure
    for s in switches:
        if not any(ss == s for (ss, _) in allowed_pairs):
            return (
                {},
                {},
                None,
                0,
                {},
                {},
                None,
                {},
                {},
                f"NO_FEASIBLE_ALLOWED_CONTROLLER_MCF_PATH_SWITCH_{s}"
            )

    # Kset and path_edges and per-pair K indices
    Kset: List[Tuple[int, int, int]] = []
    path_edges: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}
    Kset_by_pair: Dict[Tuple[int, int], List[int]] = {}

    for (s, c) in allowed_pairs:
        if s == c:
            continue

        plist = sc_kpaths.get((s, c), [])
        # sort k indices by increasing cached cost (enforces “ascending order” preference structurally)
        ks_all = [k for k in range(len(plist)) if (s, c, k) in sc_kpath_cost_ms]
        ks_all.sort(key=lambda kk: float(sc_kpath_cost_ms[(s, c, kk)]))

        for k in ks_all:
            # edges
            if sc_kpath_edges is not None and (s, c, k) in sc_kpath_edges:
                eds = [(_undir(u, v)) for (u, v) in sc_kpath_edges[(s, c, k)]]
            else:
                # derive from node list
                p = plist[k]
                nodes = p.get("nodes", []) if isinstance(p, dict) else p
                if not nodes or len(nodes) < 2:
                    continue
                eds = [_undir(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

            Kset.append((s, c, k))
            path_edges[(s, c, k)] = eds
            Kset_by_pair.setdefault((s, c), []).append(k)

    # -----------------------------
    # model
    # -----------------------------
    m = gp.Model("MCF_Kpaths_LoadObj_MeanRTCost")
    m.setParam("OutputFlag", 0)
    if time_limit and time_limit > 0:
        m.setParam("TimeLimit", float(time_limit))

    # assignment + migration
    y = m.addVars(allowed_pairs, vtype=GRB.BINARY, name="y")
    z = m.addVars(switches, vtype=GRB.BINARY, name="z")

    # path fractions
    f_vtype = GRB.CONTINUOUS if allow_path_splitting else GRB.BINARY
    f = m.addVars(Kset, vtype=f_vtype, lb=0.0, ub=1.0, name="f")

    # path-used indicators (to implement MAX among used paths)
    u = m.addVars(Kset, vtype=GRB.BINARY, name="u")

    # -----------------------------
    # constraints
    # -----------------------------
    # (1) exactly one controller per switch
    for s in switches:
        Cs = [c for (ss, c) in allowed_pairs if ss == s]
        m.addConstr(gp.quicksum(y[s, c] for c in Cs) == 1, name=f"one_ctrl_{s}")

    # (2) optional self-host if controller is also a node and (c,c) exists
    for c in controllers:
        if c in switches and (c, c) in y:
            m.addConstr(y[c, c] == 1, name=f"self_host_{c}")

    # (3) link fractions to assignment: sum_k f[s,c,k] == y[s,c]
    for (s, c) in allowed_pairs:
        if s == c:
            continue
        ks = Kset_by_pair.get((s, c), [])
        if not ks:
            m.addConstr(y[s, c] == 0, name=f"no_paths_force0_{s}_{c}")
        else:
            m.addConstr(gp.quicksum(f[s, c, k] for k in ks) == y[s, c],
                        name=f"frac_sum_{s}_{c}")

    # (4) define “used path” u from f (u=1 iff f>0)
    EPS_USE = 1e-6
    # (4) define “used path” u from f
    if allow_path_splitting:
        EPS_USE = 1e-6
        for (s, c, k) in Kset:
            m.addConstr(f[s, c, k] <= u[s, c, k], name=f"f_le_u_{s}_{c}_{k}")
            m.addConstr(f[s, c, k] >= EPS_USE * u[s, c, k], name=f"f_ge_epsu_{s}_{c}_{k}")
    else:
        # binary single-path mode: u and f are the same indicator
        for (s, c, k) in Kset:
            m.addConstr(u[s, c, k] == f[s, c, k], name=f"u_eq_f_{s}_{c}_{k}")


    # (5) migration detector z[s] from y vs init assignment
    x0_allowed = {(s, c): int(init_assign.get(s) == c) for (s, c) in allowed_pairs}
    for s in switches:
        prev = init_assign.get(s, None)
        if prev is None or (s, prev) not in y:
            m.addConstr(z[s] == 1, name=f"mig_forced_{s}")
        else:
            for c in [cc for (ss, cc) in allowed_pairs if ss == s]:
                m.addConstr(z[s] >= y[s, c] - x0_allowed[(s, c)], name=f"mig_pos_{s}_{c}")
                m.addConstr(z[s] >= x0_allowed[(s, c)] - y[s, c], name=f"mig_neg_{s}_{c}")

    # (6) controller loads lam[c] (req/s) and usable capacity guard
    load_expr = {
        c: gp.quicksum(float(loads.get(s, 0.0)) * y[s, c]
                       for (s, cc) in allowed_pairs if cc == c)
        for c in controllers
    }
    lam = m.addVars(controllers, lb=0.0, name="lambda")
    for c in controllers:
        m.addConstr(lam[c] == load_expr[c], name=f"lambda_def_{c}")
        m.addConstr(lam[c] <= GLOBAL_THRESHOLD * float(capacities[c]), name=f"cap_guard_{c}")

    # (7) link capacity with path incidence
    edge_to_kset = defaultdict(list)
    for (s, c, k) in Kset:
        for e in path_edges[(s, c, k)]:
            edge_to_kset[e].append((s, c, k))

    for e in E:
        cap = float(edge_caps_e.get(e, float("inf")))
        if cap < float("inf"):
            lhs = gp.quicksum(demand_bits[s] * f[s, c, k] for (s, c, k) in edge_to_kset.get(e, []))
            m.addConstr(lhs <= cap, name=f"linkcap_{e[0]}_{e[1]}")

    # -----------------------------*
    # RT model (final mean RT)
    # -----------------------------
    # queueing time via PWL: W(λ) ≈ 1/(μ-λ), with μ = GLOBAL_THRESHOLD * capacity
    mu_eff = {c: float(GLOBAL_THRESHOLD) * float(capacities[c]) for c in controllers}
    # keep λ within PWL domain [0, rho_max*μ]
    for c in controllers:
        m.addConstr(lam[c] <= float(mu_eff[c]) * float(rho_max), name=f"lambda_domain_{c}")

    W_ms = m.addVars(controllers, lb=0.0, name="W_ms")
    for c in controllers:
        muc = float(mu_eff[c])
        if muc <= 0.0:
            m.addConstr(W_ms[c] >= 1e9, name=f"W_dead_{c}")
            continue

        xs = [rho * muc for rho in (i * (float(rho_max) / pwl_segments) for i in range(pwl_segments + 1))]
        ys_ms = [1000.0 / max(1e-9, (muc - x)) for x in xs]  # sec -> ms
        m.addGenConstrPWL(lam[c], W_ms[c], xs, ys_ms, name=f"W_pwl_ms_{c}")

    # propagation for each switch: MAX cost among used paths (u=1)
    prop_ms = {s: m.addVar(lb=0.0, name=f"prop_ms_{s}") for s in switches}
    for (s, c, k) in Kset:
        cost = float(sc_kpath_cost_ms[(s, c, k)])
        m.addConstr(prop_ms[s] >= cost * u[s, c, k], name=f"prop_max_{s}_{c}_{k}")

    # total RT per switch: link to assigned controller via big-M
    T_ms = m.addVars(switches, lb=0.0, name="T_ms")
    BIGM_T = 1e9
    S_ms = {c: float(sync_per_ctrl_ms) for c in controllers}

    for s in switches:
        Cs = [c for (ss, c) in allowed_pairs if ss == s]
        for c in Cs:
            m.addConstr(T_ms[s] - (prop_ms[s] + W_ms[c] + S_ms[c]) <= BIGM_T * (1 - y[s, c]),
                        name=f"T_up_{s}_{c}")
            m.addConstr((prop_ms[s] + W_ms[c] + S_ms[c]) - T_ms[s] <= BIGM_T * (1 - y[s, c]),
                        name=f"T_lo_{s}_{c}")

    # mean RT
    nS = max(1, len(switches))
    mean_T_ms = m.addVar(lb=0.0, name="mean_T_ms")
    m.addConstr(gp.quicksum(T_ms[s] for s in switches) == float(nS) * mean_T_ms, name="mean_link")

    # ΔmeanRT_pos = max(0, mean_T_ms - init_mean_rt_ms)
    if init_mean_rt_ms is None:
        init_mean_rt_ms = 0.0
    delta_mean_rt_pos = m.addVar(lb=0.0, name="delta_mean_rt_pos")
    m.addConstr(delta_mean_rt_pos >= mean_T_ms - float(init_mean_rt_ms), name="delta_mean_pos_def")

    # -----------------------------
    # base load objective (load-based)
    # -----------------------------
    obj = str(objective_type).lower()
    alias = {
        "maxmin": "maxmin",
        "min_max_util": "min_max_util",
        "min_sum_util": "min_sum_util",
        "variance": "variance",
        "min_dev": "min_dev",
    }
    obj = alias.get(obj, obj)

    # small tie-breaker to prefer short paths if multiple solutions
    total_path_cost_bits_ms = gp.quicksum(
        float(sc_kpath_cost_ms[(s, c, k)]) * demand_bits[s] * f[s, c, k]
        for (s, c, k) in Kset
    )

    if obj == "min_max_util":
        U = m.addVar(lb=0.0, name="Umax")
        for c in controllers:
            # util <= U  => load <= U * cap
            m.addConstr(load_expr[c] <= U * float(capacities[c]), name=f"utilcap_{c}")
        base_obj = U + float(eta_path) * total_path_cost_bits_ms

    elif obj == "min_sum_util":
        base_obj = gp.quicksum(load_expr[c] / float(capacities[c]) for c in controllers) + float(eta_path) * total_path_cost_bits_ms

    elif obj == "variance":
        total_load = sum(float(loads.get(s, 0.0)) for s in switches)
        kctrl = max(1, len(controllers))
        muL = total_load / kctrl
        base_obj = gp.quicksum((load_expr[c] - muL) * (load_expr[c] - muL) for c in controllers) + float(eta_path) * total_path_cost_bits_ms

    elif obj == "min_dev":
        total_load = sum(float(loads.get(s, 0.0)) for s in switches)
        kctrl = max(1, len(controllers))
        muL = total_load / kctrl
        d = {c: m.addVar(lb=0.0, name=f"abs_dev_{c}") for c in controllers}
        for c in controllers:
            m.addConstr(load_expr[c] - muL <= d[c], name=f"absdev_pos_{c}")
            m.addConstr(muL - load_expr[c] <= d[c], name=f"absdev_neg_{c}")
        base_obj = gp.quicksum(d[c] for c in controllers) + float(eta_path) * total_path_cost_bits_ms

    else:  # maxmin
        Lmax = m.addVar(lb=0.0, name="Lmax")
        Lmin = m.addVar(lb=0.0, name="Lmin")
        for c in controllers:
            m.addConstr(load_expr[c] <= Lmax, name=f"L_le_{c}")
            m.addConstr(load_expr[c] >= Lmin, name=f"L_ge_{c}")
        base_obj = (Lmax - Lmin) + float(eta_path) * total_path_cost_bits_ms

    # -----------------------------
    # migration cost pack
    # -----------------------------
    num_mig = gp.quicksum(z[s] for s in switches)

    # controller-to-controller transfer cost (linear):
    # cc_transfer = Σ_s Σ_{c != c0(s)} Dcc[c0,c] * y[s,c]
    cc_transfer = 0.0
    cc_expr_by_s = None  # store per-switch expression so we can extract later

    if Dcc is not None:
        cc_expr_by_s = {}
        for s in switches:
            c0 = init_assign.get(s, None)
            if c0 is None:
                cc_expr_by_s[s] = 0.0
                continue

            terms_s = []
            for c in controllers:
                if (s, c) not in y:
                    continue
                if c == c0:
                    continue

                # accept dict-of-dicts or flat
                if isinstance(Dcc.get(c0, {}), dict):
                    dval = float(Dcc.get(c0, {}).get(c, 0.0))
                else:
                    dval = float(Dcc.get((c0, c), 0.0))

                terms_s.append(dval * y[s, c])

            cc_expr_by_s[s] = gp.quicksum(terms_s) if terms_s else 0.0

        cc_transfer = gp.quicksum(cc_expr_by_s[s] for s in switches)

    migration_cost_expr = (
        float(w_mig) * num_mig
        + float(w_cc) * cc_transfer
        + float(w_rt) * delta_mean_rt_pos
    )


    m.setObjective((alpha * base_obj )+ (beta* migration_cost_expr), GRB.MINIMIZE)


    # -----------------------------
    # solve
    # -----------------------------
    # ============================================================
    # SOLVE + STATUS HANDLING (PATH MCF)
    # ============================================================

    m.optimize()
    # -----------------------------
    # MIP GAP SAFE INITIALIZATION
    # -----------------------------
    mip_gap_solved = None
    if m.SolCount > 0:
        mip_gap_solved = float(m.MIPGap)
    # -----------------------------
    # INFEASIBLE → IIS
    # -----------------------------
    if m.Status == GRB.INFEASIBLE:

        iis_dir = os.path.join(RESULTS_FOLDER, "iis_reports")
        os.makedirs(iis_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")

        try:
            m.setParam(GRB.Param.Presolve, 0)
            m.setParam(GRB.Param.IISMethod, 1)
            m.computeIIS()

            ilp_file = os.path.join(
                iis_dir,
                f"{topology_name or 'topo'}_path_mcf_{ts}.ilp"
            )
            m.write(ilp_file)

            print("IIS written to:", ilp_file)
            status_msg = f"INFEASIBLE_MCF_PATH (IIS:{ilp_file})"

        except:
            status_msg = "INFEASIBLE_MCF_PATH (IIS_FAILED)"

        return (
            {},
            {},
            None,
            0,
            {},
            {},
            None,
            {},
            {},
            status_msg
        )

    # -----------------------------
    # INF_OR_UNBD
    # -----------------------------
    if m.Status == GRB.INF_OR_UNBD:
        return (
            {},
            {},
            None,
            0,
            {},
            {},
            None,
            {},
            {},
            "INF_OR_UNBOUNDED_MCF_PATH"
        )

    # -----------------------------
    # TIME LIMIT (no solution)
    # -----------------------------
    if m.Status == GRB.TIME_LIMIT and m.SolCount == 0:
        return (
            {},
            {},
            None,
            0,
            {},
            {},
            None,
            {},
            {},
            "TIME_LIMIT_NO_SOLUTION_MCF_PATH"
        )

    # -----------------------------
    # NO FEASIBLE SOLUTION
    # -----------------------------
    if m.SolCount == 0:

        print(f"No feasible solution. Status = {m.Status}")

        return (
            {},
            {},
            None,
            0,
            {},
            {},
            None,
            {},
            {},
            f"NO_FEASIBLE_SOLUTION_STATUS_{m.Status}_PATH"
        )

    # ============================================================
    # NORMAL CASE
    # ============================================================

    status_msg = "OPTIMAL" if m.Status == GRB.OPTIMAL else f"FEASIBLE_STATUS_{m.Status}"
    # -----------------------------
    # extract solution
    # -----------------------------
    final_assign: Dict[int, int] = {}
    for s in switches:
        chosen = None
        for c in [cc for (ss, cc) in allowed_pairs if ss == s]:
            if y[s, c].X > 0.5:
                chosen = c
                break
        if chosen is None:
            chosen = init_assign.get(s, None)
        if chosen is not None:
            final_assign[s] = chosen

    missing_assign = [s for s in switches if s not in final_assign]
    if missing_assign:
        return (
            {},
            {},
            None,
            0,
            {},
            {},
            mip_gap_solved,
            {},
            {},
            f"NO_FEASIBLE_COMPLETE_ASSIGNMENT_MCF_PATH_MISSING_{len(missing_assign)}"
        )

    final_loads = {c: float(lam[c].X) for c in controllers}
    mig_count = sum(1 for s in switches if final_assign.get(s) != init_assign.get(s))
    # -----------------------------
    # chosen single k per switch (only meaningful when allow_path_splitting=False)
    # -----------------------------
    chosen_k_final = {}
    for s in switches:
        c = final_assign.get(s, None)
        if c is None or s == c:
            chosen_k_final[s] = None
            continue

        ks = Kset_by_pair.get((s, c), [])
        kstar = None
        for k in ks:
            if (s, c, k) in f and f[s, c, k].X > 0.5:
                kstar = int(k)
                break
        chosen_k_final[s] = kstar

    # per-switch cc delay (ms) for logging
    cc_delay_ms_by_switch = {s: 0.0 for s in switches}
    if Dcc is not None:
        for s in switches:
            c0 = init_assign.get(s, None)
            c1 = final_assign.get(s, None)
            if c0 is None or c1 is None or c0 == c1:
                continue

            if isinstance(Dcc.get(c0, {}), dict):
                cc_delay_ms_by_switch[s] = float(Dcc.get(c0, {}).get(c1, 0.0))
            else:
                cc_delay_ms_by_switch[s] = float(Dcc.get((c0, c1), 0.0))


    bundle = extract_mcf_solution_bundle(
        switches=switches,
        final_assign=final_assign,
        f=f,
        Kset_by_pair=Kset_by_pair,
        sc_kpaths=sc_kpaths,
        sc_kpath_cost_ms=sc_kpath_cost_ms,
        path_edges=path_edges,
        demand_bits=demand_bits,
        allow_path_splitting=allow_path_splitting,
    )

    paths_used_by_switch = bundle["paths_used_by_switch"]
    paths_by_switch_final = bundle["paths_by_switch"]
    usage_e = bundle["usage_e"]
    # -----------------------------
    # ✅ RT delta logging (GLOBAL mean)
    # -----------------------------
    init_mean_rt_ms_f = float(init_mean_rt_ms or 0.0)
    mean_rt_ms_final = float(mean_T_ms.X)
    delta_mean_rt_ms = mean_rt_ms_final - init_mean_rt_ms_f
    delta_mean_rt_pos_ms = float(delta_mean_rt_pos.X)  # already the positive-part variable
    if init_rt_ms_by_switch is None:
        T_ms_init_by_switch = {int(s): 0.0 for s in switches}
    else:
        # normalize keys to int and provide default 0 for any missing switch
        T_ms_init_by_switch = {int(s): float(init_rt_ms_by_switch.get(s, 0.0)) for s in switches}

    delta_rt_ms_by_switch = {
        s: float(T_ms[s].X) - float(T_ms_init_by_switch.get(s, 0.0))
        for s in switches
    }
    delta_rt_pos_ms_by_switch = {s: max(0.0, v) for s, v in delta_rt_ms_by_switch.items()}

    print("\n=== DEBUG: W_exact vs W_solver (PATH) ===")

    for c in controllers:
        mu = float(GLOBAL_THRESHOLD) * float(capacities[c])
        lam_val = float(lam[c].X)

        if mu - lam_val <= 1e-9:
            print(f"C{c}: near saturation, skip")
            continue

        W_exact = 1000.0 / (mu - lam_val)
        W_solver = float(W_ms[c].X)

        print(
            f"C{c} | λ={lam_val:.6f} μ={mu:.6f} | "
            f"W_exact={W_exact:.6f} ms | "
            f"W_solver={W_solver:.6f} ms | "
            f"Δ={W_solver - W_exact:.6f}"
        )


    rt_metrics = {
        "mean_rt_ms_init": init_mean_rt_ms_f,
        "mean_rt_ms_final": mean_rt_ms_final,
        "delta_mean_rt_ms": delta_mean_rt_ms,           # signed
        "delta_mean_rt_pos_ms": delta_mean_rt_pos_ms,   # positive-part used in objective
        "lambda_by_ctrl": {c: float(lam[c].X) for c in controllers},
        "W_ms_by_ctrl": {c: float(W_ms[c].X) for c in controllers},
        "S_ms_steiner": {c: float(S_ms[c]) for c in controllers},
        "prop_ms_by_switch": {s: float(prop_ms[s].X) for s in switches},
        "T_ms_by_switch": {s: float(T_ms[s].X) for s in switches},
        "paths_used_by_switch": paths_used_by_switch,
        "chosen_k_final": chosen_k_final,
        "cc_delay_ms_by_switch": cc_delay_ms_by_switch,
        "delta_total_rt_ms_by_switch": delta_rt_ms_by_switch,
        "delta_total_rt_pos_ms_by_switch": delta_rt_pos_ms_by_switch,
    }



    # -----------------------------
    # ✅ Migration cost decomposition (for JSON logging)
    # -----------------------------
    num_mig_val = float(num_mig.getValue())
    cc_transfer_val = float(cc_transfer.getValue()) if Dcc is not None else 0.0

    mig_cost = {
        "w_mig": float(w_mig),
        "w_cc": float(w_cc),
        "w_rt": float(w_rt),
        "count": int(mig_count),
        "num_mig_expr": num_mig_val,
        "cc_transfer": cc_transfer_val,
        "rt_mean_init_ms": init_mean_rt_ms_f,
        "rt_mean_final_ms": mean_rt_ms_final,
        "delta_rt": float(delta_mean_rt_ms),
        "delta_rt_pos": float(delta_mean_rt_pos_ms),
        "migration_cost_total": float(migration_cost_expr.getValue()),
    }
    paths_new = build_paths_sc_from_switch_paths(final_assign, paths_by_switch_final)
    obj_val = float(m.objVal) if m.SolCount > 0 else None

    return final_assign, final_loads, obj_val, mig_count, usage_e, rt_metrics, mip_gap_solved, mig_cost, paths_new,status_msg
