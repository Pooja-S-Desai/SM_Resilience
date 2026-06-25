from __future__ import annotations

import os
import time
from typing import Dict
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from link_checks import extract_mcf_solution_bundle
from helpers import (
    CAPACITY_THRESHOLD as GLOBAL_THRESHOLD,   # usable fraction for controller capacity guard
    CAPACITY_THRESHOLD,                   # kept for reporting helpers if you want
    LINK_BUDGET_BITS,
    RESULTS_FOLDER,
    FIBER_SEC_PER_KM,
    TIME_LIMIT as time_limit
)
from experiment_logger_100runs import  _write_resilience_debug_log
from rt_metrics import build_paths_sc_from_switch_paths
from plotting import plot_final_vs_recovery_assignment
from rt_metrics import flows_to_usage_and_rtt
from collections import defaultdict
def _compute_usage_from_solution(f, commodity_pairs, arcs, edge_caps_e):
    """
    Uses solved f[...] (tupledict of gurobi vars) to compute undirected per-link usage.
    Returns:
      usage_e: {(u,v): used_bits_per_sec}
      util_e:  {(u,v): used/cap} for finite caps
    """
    usage_e = {}
    util_e  = {}

    # build undirected edge set from arcs
    undirected_edges = set()
    for (u, v) in arcs:
        undirected_edges.add(_undir((u, v)))

    for e in undirected_edges:
        u, v = e
        used = 0.0
        for (s, c) in commodity_pairs:
            # add both directions if exist
            if (s, c, u, v) in f:
                used += float(f[s, c, u, v].X)
            if (s, c, v, u) in f:
                used += float(f[s, c, v, u].X)

        usage_e[e] = used

        cap = float(edge_caps_e.get(e, float("inf")))
        if cap < float("inf") and cap > 0:
            util_e[e] = used / cap

    return usage_e, util_e

# ---------- small utilities ----------
def _undir(e):
    u, v = e
    return (u, v) if u < v else (v, u)

def _arc_latency_sec(G, u, v):
    d = G[u][v]
    if "latency_sec" in d:
        return float(d["latency_sec"])
    if "weight" in d:
        # Your pipeline convention: weight is "meters-like"; convert using fiber sec/km
        return float(d["weight"]) * float(FIBER_SEC_PER_KM / 1000.0)
    return 1.0 * float(FIBER_SEC_PER_KM / 1000.0)

def _default_edge_caps(G, per_link_bits: float) -> dict[tuple[int, int], float]:
    caps = {}
    for u, v in G.edges():
        e = _undir((u, v))
        caps[e] = float(G[u][v].get("cap_bits", per_link_bits))
    return caps

def _max_shortest_rtt_bound(G, switches, controllers, round_trip: bool) -> float:
    """
    Conservative BIG-M bound based on shortest latencies (only for M sizing).
    """
    GG = G.to_undirected() if G.is_directed() else G

    def w(u, v, d):
        if "latency_sec" in d:
            return float(d["latency_sec"])
        if "weight" in d:
            return float(d["weight"]) * float(FIBER_SEC_PER_KM / 1000.0)
        return 1.0 * float(FIBER_SEC_PER_KM / 1000.0)

    rtt_max = 0.0
    for s in switches:
        for c in controllers:
            if not nx.has_path(GG, s, c):
                continue
            try:
                length = nx.shortest_path_length(GG, s, c, weight=lambda uu, vv, dd: w(uu, vv, dd))
            except Exception:
                continue
            if round_trip:
                length *= 2.0
            rtt_max = max(rtt_max, float(length))
    return max(rtt_max, 1e-3)

def _extract_path_nodes_from_b_used(s, c, b_used, arcs, *, max_hops=10000):
    # Build adjacency for this commodity from arcs with b_used==1
    nxt = {}
    for (u, v) in arcs:
        key = (s, c, u, v)
        if key in b_used and b_used[key].X > 0.5:
            nxt[u] = v

    # Walk from s until c (or stop)
    path = [s]
    cur = s
    seen = set([s])
    for _ in range(max_hops):
        if cur == c:
            break
        if cur not in nxt:
            break
        cur = nxt[cur]
        if cur in seen:     # cycle safety
            break
        seen.add(cur)
        path.append(cur)

    # Only accept if it truly reaches c
    if path and path[-1] == c:
        return path
    return []

def run_migration_optimizer_integrated_mcf_arc(
    G,
    switches,
    controllers,
    loads,
    capacities,
    init_assign,
    *,
    objective_type: str = "maxmin",     # one of: maxmin, min_max_util, min_sum_util, min_dev, variance
    topology_name: str | None = None,

    # RT knobs (already in your arc-based file)
    round_trip: bool = True,
    rho_max: float = 0.95,
    pwl_segments: int = 12,

    # Practical anti-cycle regularizer (recommended small positive like 1e-12..1e-8)
    eta: float = 0.0,
    rho_max_local=0.95,  
    # Migration-cost inputs
    Dcc: dict | None = None,             # controller↔controller shortest path cost (you decide units)
    sync_per_ctrl_ms: float = 0.0,       # steiner sync penalty added to RT via controller (ms)
    w_mig: float = 1.0,
    w_cc: float = 1.0,
    w_rt: float = 1.0,
    w_steiner: float = 0.0,        # NEW: accept steiner weight (default 0 if you want optional)
    cost_mode: str = "weight",     # NEW: accept caller arg (may be unused)

    # This MUST be computed in main using same RT definition on init assignment
    init_mean_rt_ms: float | None = None,
    init_rt_ms_by_switch: dict[int, float] | None = None,

    # Link caps / demand params
    edge_caps_e: dict | None = None,
    msg_bits: int = 128,



    allow_path_splitting=True,
    alpha:float,
    beta:float,
    gamma_res: float = 1.0 ,
    run_index: int = 0,
    resilience_log_dir: str | None = None,

    # plotting from MCF-ARC only
    plot_recovery: bool = False,
    plot_pos: dict | None = None,
    plot_save_dir: str | None = None,
    plot_topology_name: str | None = None,
    plot_file_tag: str | None = None,
    node_capacities: dict | None = None,
):
    """
    Arc-based integrated assignment + MCF + RT variables.
    Base objective is load-based (5 options).
    Migration cost includes:
      (#migrations) + (C-C transfer gated by migration) + ΔmeanRT_pos.

    NOTE:
    - No DAG/acyclic constraints are added (as requested).
    - If you want to discourage cycles, set eta > 0 (tiny).
    """

    # ---------- arcs ----------
    arcs = []
    for u, v in G.edges():
        arcs.append((u, v))
        arcs.append((v, u))

    UG = G.to_undirected()

    # reachability gating
    allowed_pairs = [(s, c) for s in switches for c in controllers if nx.has_path(UG, s, c)]
    for s in switches:
        if not any(ss == s for (ss, _) in allowed_pairs):
            return (
                {}, {}, None, 0, {}, {}, None,
                {},
                {},
                f"NO_FEASIBLE_ALLOWED_CONTROLLER_MCF_ARC_SWITCH_{s}"
            )

    commodity_pairs = [(s, c) for (s, c) in allowed_pairs if s != c]
    x0_allowed = {(s, c): int(init_assign.get(s) == c) for (s, c) in allowed_pairs}

    # edge capacities (undirected)
    if edge_caps_e is None:
        edge_caps_e = _default_edge_caps(G, per_link_bits=float(LINK_BUDGET_BITS))

    # demand in bits/s
    demand_bits = {s: float(loads.get(s, 0.0)) * float(msg_bits) for s in switches}

    # Big-M sizing helpers
    rtt_M = _max_shortest_rtt_bound(G, switches, controllers, round_trip=round_trip)
    mu_eff = {c: float(GLOBAL_THRESHOLD) * float(capacities[c]) for c in controllers}

    # W max bound
    Wmax_overall = 0.0
    for c, muc in mu_eff.items():
        if muc <= 0:
            Wmax_overall = max(Wmax_overall, 1e6)
        else:
            eps = max(1e-9, 1.0 - float(rho_max))
            Wmax_overall = max(Wmax_overall, 1.0 / (muc * eps))

    # Steiner sync in seconds and ms
    S_ms = {c: float(sync_per_ctrl_ms) for c in controllers}
    s_const_sec = float(sync_per_ctrl_ms) / 1000.0
    S_sec = {c: s_const_sec for c in controllers}

    BIG_M = 1e9

    # ---------- model ----------
    m = gp.Model("arc_mcf_loadobj_rt_migcost")
    m.setParam("OutputFlag", 0)
    if time_limit and time_limit > 0:
        m.setParam("TimeLimit", float(time_limit))


    # ---------- variables ----------
    y = m.addVars(allowed_pairs, vtype=GRB.BINARY, name="y_assign")
    z = m.addVars(switches, vtype=GRB.BINARY, name="z_mig")

    # flow f[s,c,u,v] in bits/s
    f = m.addVars([(s, c, u, v) for (s, c) in commodity_pairs for (u, v) in arcs],
                  lb=0.0, name="f")

    # controller load and queue
    lam = m.addVars(controllers, lb=0.0, name="lambda")   # req/s
    W_sec = m.addVars(controllers, lb=0.0, name="W_sec")  # seconds

    # RT in seconds and ms
    RTT_flow_sec = m.addVars(commodity_pairs, lb=0.0, name="RTT_flow_sec")
    T_sec = m.addVars(switches, lb=0.0, name="T_sec")
    T_ms = m.addVars(switches, lb=0.0, name="T_ms")       # convenience ms mirror



    # ============================================================
    # WORST-USED-PATH (max over used paths) helpers (Option-2)
    # ============================================================
    # b[s,c,u,v] = 1 iff commodity (s,c) uses directed arc (u,v) with non-zero flow
    EPS_FLOW = 1e-9  # bits/s threshold for "non-zero"
    b_used = m.addVars([(s, c, u, v) for (s, c) in commodity_pairs for (u, v) in arcs],
                    vtype=GRB.BINARY, name="b_used")
    # One-way propagation along the chosen path (seconds)
    P_oneway_sec = m.addVars(commodity_pairs, lb=0.0, name="P_oneway_sec")

    # pi[s,c,n] = potential/label (one-way propagation seconds) at node n for commodity (s,c)
    pi = m.addVars([(s, c, n) for (s, c) in commodity_pairs for n in UG.nodes()],
                lb=0.0, name="pi")

    # Pworst_oneway_sec[s,c] = worst one-way path propagation cost among used paths
    Pworst_oneway_sec = m.addVars(commodity_pairs, lb=0.0, name="Pworst_oneway_sec")

    # arc one-way delays (sec) for directed arcs
    arc_delay_sec = {(u, v): float(_arc_latency_sec(G, u, v)) for (u, v) in arcs}

    # BIG-M for potentials: conservative upper bound in seconds
    # Use something safely above any reasonable end-to-end latency.
    BIGM_PI = float(rtt_M) + 10.0

    # ---------- adjacency bookkeeping ----------
    out_arcs = {n: [] for n in G.nodes()}
    in_arcs = {n: [] for n in G.nodes()}
    for (u, v) in arcs:
        out_arcs[u].append((u, v))
        in_arcs[v].append((u, v))

    # ---------- constraints ----------
    # (1) exactly one controller per switch
    for s in switches:
        Cs = [c for (ss, c) in allowed_pairs if ss == s]
        m.addConstr(gp.quicksum(y[s, c] for c in Cs) == 1, name=f"one_ctrl_{s}")

    # (2) self-host if (c,c) allowed and c is a switch-node
    for c in controllers:
        if c in switches and (c, c) in y:
            m.addConstr(y[c, c] == 1, name=f"self_host_{c}")

    # (3) migration detector z[s]
    for s in switches:
        prev = init_assign.get(s)
        if (s, prev) in y:
            for c in controllers:
                if (s, c) in y:
                    m.addConstr(z[s] >= y[s, c] - x0_allowed[(s, c)], name=f"mig_pos_{s}_{c}")
                    m.addConstr(z[s] >= x0_allowed[(s, c)] - y[s, c], name=f"mig_neg_{s}_{c}")
        else:
            m.addConstr(z[s] == 1, name=f"mig_forced_{s}")

    # (4) controller load definitions: load_expr and lam
    load_expr = {
        c: gp.quicksum(float(loads.get(s, 0.0)) * y[s, c] for s in switches if (s, c) in y)
        for c in controllers
    }
    for c in controllers:
        m.addConstr(lam[c] == load_expr[c], name=f"lambda_def_{c}")
        # usable capacity guard
        m.addConstr(lam[c] <= float(GLOBAL_THRESHOLD) * float(capacities[c]), name=f"cap_guard_{c}")
        # keep inside PWL domain
        m.addConstr(lam[c] <= float(mu_eff[c]) * float(rho_max), name=f"lambda_domain_{c}")


    if allow_path_splitting:
        # ============================================================
        # SPLITTABLE (fractional MCF): your original logic
        # ============================================================

        # (5) activate flows: f <= demand_bits[s] * y[s,c]
        for (s, c) in commodity_pairs:
            d_bits = float(demand_bits[s])
            for (u, v) in arcs:
                m.addConstr(f[s, c, u, v] <= d_bits * y[s, c],
                            name=f"activate_{s}_{c}_{u}_{v}")

        # (5b) define b_used from flow (needed for worst-path RTT via pi)
        for (s, c) in commodity_pairs:
            d_bits = float(demand_bits[s])
            for (u, v) in arcs:
                # if b_used=0 => f=0
                m.addConstr(f[s, c, u, v] <= d_bits * b_used[s, c, u, v],
                            name=f"f_le_dbUsed_{s}_{c}_{u}_{v}")

                # if b_used=1 => enforce tiny positive flow to keep b_used honest
                m.addConstr(f[s, c, u, v] >= EPS_FLOW * b_used[s, c, u, v],
                            name=f"f_ge_epsbUsed_{s}_{c}_{u}_{v}")

                # arc use only if commodity is active (assignment chosen)
                m.addConstr(b_used[s, c, u, v] <= y[s, c],
                            name=f"bUsed_le_y_{s}_{c}_{u}_{v}")

        # (6) flow conservation for each commodity (s,c)
        for (s, c) in commodity_pairs:
            d_bits = float(demand_bits[s])
            for n in G.nodes():
                out_sum = gp.quicksum(f[s, c, uu, vv] for (uu, vv) in out_arcs[n])
                in_sum  = gp.quicksum(f[s, c, uu, vv] for (uu, vv) in in_arcs[n])

                if n == s:
                    m.addConstr(out_sum - in_sum == d_bits * y[s, c],
                                name=f"flow_src_{s}_{c}_{n}")
                elif n == c:
                    m.addConstr(in_sum - out_sum == d_bits * y[s, c],
                                name=f"flow_dst_{s}_{c}_{n}")
                else:
                    m.addConstr(out_sum - in_sum == 0.0,
                                name=f"flow_bal_{s}_{c}_{n}")

    else:
        # ============================================================
        # UNSPLITTABLE (binary single-path per commodity):
        # b_used defines the unique path; f is forced to full-demand on chosen arcs
        # ============================================================

        # (5') Link flow to chosen arcs: if arc chosen -> carries full demand, else 0
        for (s, c) in commodity_pairs:
            d_bits = float(demand_bits[s])
            for (u, v) in arcs:
                m.addConstr(f[s, c, u, v] == d_bits * b_used[s, c, u, v],
                            name=f"f_eq_d_b_{s}_{c}_{u}_{v}")
                m.addConstr(b_used[s, c, u, v] <= y[s, c],
                            name=f"b_le_y_{s}_{c}_{u}_{v}")

        # (6') Single directed s->c path constraints on b_used (no splitting/branching)
        for (s, c) in commodity_pairs:
            for n in G.nodes():
                outb = gp.quicksum(b_used[s, c, uu, vv] for (uu, vv) in out_arcs[n])
                inb  = gp.quicksum(b_used[s, c, uu, vv] for (uu, vv) in in_arcs[n])

                if n == s:
                    # source: exactly 1 outgoing if active; 0 incoming
                    m.addConstr(outb == y[s, c], name=f"b_src_out_{s}_{c}_{n}")
                    m.addConstr(inb  == 0,       name=f"b_src_in_{s}_{c}_{n}")

                elif n == c:
                    # sink: exactly 1 incoming if active; 0 outgoing
                    m.addConstr(inb  == y[s, c], name=f"b_sink_in_{s}_{c}_{n}")
                    m.addConstr(outb == 0,       name=f"b_sink_out_{s}_{c}_{n}")

                else:
                    # intermediate: either unused (0,0) or used (1,1), and degree <= 1 prevents splitting
                    m.addConstr(outb == inb, name=f"b_bal_{s}_{c}_{n}")
                    m.addConstr(outb <= 1,  name=f"b_outdeg1_{s}_{c}_{n}")
                    m.addConstr(inb  <= 1,  name=f"b_indeg1_{s}_{c}_{n}")


    # # (6b) potentials define WORST used path cost (max over used paths) and avoid cycles
    # # pi[s,c,s] = 0
    # # if arc (u,v) is used then pi[v] >= pi[u] + delay(u,v)
    # # worst one-way path cost is pi at sink: Pworst_oneway_sec[s,c] = pi[s,c,c]
    # for (s, c) in commodity_pairs:
    #     m.addConstr(pi[s, c, s] == 0.0, name=f"pi_src_{s}_{c}")

    #     for (u, v) in arcs:
    #         m.addConstr(
    #             pi[s, c, v] >= pi[s, c, u] + arc_delay_sec[(u, v)] - BIGM_PI * (1 - b_used[s, c, u, v]),
    #             name=f"pi_inc_{s}_{c}_{u}_{v}"
    #         )

    #     m.addConstr(Pworst_oneway_sec[s, c] == pi[s, c, c], name=f"Pworst_def_{s}_{c}")



    # (7) per-link bandwidth caps (undirected)
    for u, v in G.edges():
        e = _undir((u, v))
        cap = float(edge_caps_e.get(e, float("inf")))
        if cap < float("inf"):
            lhs = gp.quicksum(f[s, c, u, v] + f[s, c, v, u] for (s, c) in commodity_pairs)
            m.addConstr(lhs <= cap, name=f"linkcap_{e[0]}_{e[1]}")

    # (8) PWL for W_sec[c] ≈ 1/(μ-λ)
    for c in controllers:
        muc = float(mu_eff[c])
        if muc <= 0.0:
            m.addConstr(W_sec[c] >= 1e6, name=f"W_dead_{c}")
            continue

        xs = [rho * muc for rho in (i * (float(rho_max) / pwl_segments) for i in range(pwl_segments + 1))]
        ys = [1.0 / max(1e-9, muc - x) for x in xs]
        m.addGenConstrPWL(lam[c], W_sec[c], xs, ys, name=f"W_pwl_{c}")

    # (9) Exact RTT for UNSPLITTABLE: RTT_flow_sec[s,c] = rtfactor * sum(delay * b_used)
    for (s, c) in commodity_pairs:
        rtfactor = 2.0 if round_trip else 1.0

        m.addConstr(
            P_oneway_sec[s, c] ==
            gp.quicksum(arc_delay_sec[(u, v)] * b_used[s, c, u, v] for (u, v) in arcs),
            name=f"P_oneway_def_{s}_{c}"
        )

        m.addConstr(
            RTT_flow_sec[s, c] == rtfactor * P_oneway_sec[s, c],
            name=f"rtt_exact_{s}_{c}"
        )


    # (10) Switch RT: T_sec[s] = RTT_flow_sec + W_sec[assigned] + Steiner_sec
    for s in switches:
        Cs = [c for (ss, c) in allowed_pairs if ss == s]
        for c in Cs:
            if s == c:
                # self-host: no RTT term
                m.addConstr(T_sec[s] - (W_sec[c] + S_sec[c]) <= BIG_M * (1 - y[s, c]), name=f"T_up_self_{s}_{c}")
                m.addConstr((W_sec[c] + S_sec[c]) - T_sec[s] <= BIG_M * (1 - y[s, c]), name=f"T_lo_self_{s}_{c}")
            else:
                m.addConstr(T_sec[s] - (RTT_flow_sec[s, c] + W_sec[c] + S_sec[c]) <= BIG_M * (1 - y[s, c]),
                            name=f"T_up_{s}_{c}")
                m.addConstr((RTT_flow_sec[s, c] + W_sec[c] + S_sec[c]) - T_sec[s] <= BIG_M * (1 - y[s, c]),
                            name=f"T_lo_{s}_{c}")

    # Mirror in ms for objective/migration-cost usage
    for s in switches:
        m.addConstr(T_ms[s] == 1000.0 * T_sec[s], name=f"Tms_link_{s}")

    # ---------- practical cycle discouragement ----------
    # Tiny regularizer on total flow distance/cost (helps avoid cyclic flows)
    # Set eta small positive in main (e.g., 1e-12..1e-8).
    if float(eta) != 0.0:
        total_flow_cost = gp.quicksum(
            float(_arc_latency_sec(G, u, v)) * f[s, c, u, v]
            for (s, c) in commodity_pairs
            for (u, v) in arcs
        )
    else:
        total_flow_cost = 0.0
    usage_e = {}
    rt_metrics = {}

    # ============================================================
    # OBJECTIVES (LOAD-BASED 5) + MIGRATION COST (ΔmeanRT_pos)
    # ============================================================

    # Mean RT and delta mean RT (positive part)
    nS = max(1, len(switches))
    mean_T_ms = m.addVar(lb=0.0, name="mean_T_ms")
    m.addConstr(gp.quicksum(T_ms[s] for s in switches) == float(nS) * mean_T_ms, name="mean_rt_link")

    if init_mean_rt_ms is None:
        init_mean_rt_ms = 0.0

    delta_mean_rt_pos = m.addVar(lb=0.0, name="delta_mean_rt_pos_ms")
    m.addConstr(delta_mean_rt_pos >= mean_T_ms - float(init_mean_rt_ms), name="delta_mean_rt_pos_def")

    # -----------------------------
    # Base load objective (5)
    # -----------------------------
    obj = str(objective_type).lower()

    if obj == "maxmin":
        Lmax = m.addVar(lb=0.0, name="Lmax")
        Lmin = m.addVar(lb=0.0, name="Lmin")
        for c in controllers:
            m.addConstr(load_expr[c] <= Lmax, name=f"L_le_{c}")
            m.addConstr(load_expr[c] >= Lmin, name=f"L_ge_{c}")
        base_obj = (Lmax - Lmin)

    elif obj == "min_max_util":
        U = m.addVar(lb=0.0, name="Umax")
        for c in controllers:
            m.addConstr(load_expr[c] <= U * float(capacities[c]), name=f"utilcap_{c}")
        base_obj = U

    elif obj == "min_sum_util":
        base_obj = gp.quicksum(load_expr[c] / float(capacities[c]) for c in controllers)

    elif obj == "min_dev":
        total_load = sum(float(loads.get(s, 0.0)) for s in switches)
        kctrl = max(1, len(controllers))
        muL = float(total_load) / float(kctrl)

        d = m.addVars(controllers, lb=0.0, name="abs_dev_load")
        for c in controllers:
            m.addConstr(load_expr[c] - muL <= d[c], name=f"absdev_pos_{c}")
            m.addConstr(muL - load_expr[c] <= d[c], name=f"absdev_neg_{c}")
        base_obj = gp.quicksum(d[c] for c in controllers)

    elif obj == "variance":
        total_load = sum(float(loads.get(s, 0.0)) for s in switches)
        kctrl = max(1, len(controllers))
        muL = float(total_load) / float(kctrl)
        base_obj = gp.quicksum((load_expr[c] - muL) * (load_expr[c] - muL) for c in controllers)

    else:
        raise ValueError(f"Unknown objective_type={objective_type}. Use: maxmin, min_max_util, min_sum_util, min_dev, variance")

    # Add tiny anti-cycle flow regularizer if enabled
    base_obj = base_obj + float(eta) * total_flow_cost

    # -----------------------------
    # Migration-cost pack (exactly your definition)
    # migrations + C-C transfer (gated by migration) + ΔmeanRT_pos
    # -----------------------------
    num_mig = gp.quicksum(z[s] for s in switches)

    cc_transfer = 0.0
    if Dcc is not None:
        terms = []
        for s in switches:
            c0 = init_assign.get(s, None)
            if c0 is None:
                continue
            for c in controllers:
                if (s, c) not in y:
                    continue
                if c0 == c:
                    continue

                # dict-of-dicts or flat dict
                if isinstance(Dcc.get(c0, {}), dict):
                    dval = float(Dcc.get(c0, {}).get(c, 0.0))
                else:
                    dval = float(Dcc.get((c0, c), 0.0))

                # IMPORTANT: gate by y AND by migration z[s]
                terms.append(dval * y[s, c] * z[s])

        cc_transfer = gp.quicksum(terms) if terms else 0.0

    migration_cost_expr = (
        float(w_mig) * num_mig
        + float(w_cc) * cc_transfer
        + float(w_rt) * delta_mean_rt_pos
    )



    # ============================================================
    # PROACTIVE RESILIENCY PLANNING
    # Same optimization:
    # - each controller j is considered as a possible failed controller
    # - switches primarily assigned to j are assigned to backup controller k
    # - if no existing controller can absorb them, residual[s,j] = 1
    # - headroom is flushed per failure case because constraints are per j
    # ============================================================

    backup_pairs = [
        (s, j, k)
        for s in switches
        for j in controllers
        for k in controllers
        if k != j and (s, j) in y
    ]

    bkp = m.addVars(backup_pairs, vtype=GRB.BINARY, name="backup")

    residual = m.addVars(
        [(s, j) for s in switches for j in controllers if (s, j) in y],
        vtype=GRB.BINARY,
        name="residual_backup"
    )
    # ---------------------------------------------------------
    # Candidate residual controllers (new controller placement)
    # ---------------------------------------------------------
    residual_candidates = [v for v in G.nodes() if v not in controllers]

    old_node_capacity = {
        v: float(node_capacities.get(v, 0.0))
        for v in residual_candidates
    }

    rctrl_pairs = [
        (j, v)
        for j in controllers
        for v in residual_candidates
    ]

    rctrl = m.addVars(
        rctrl_pairs,
        vtype=GRB.BINARY,
        name="residual_controller"
    )
    # If s is assigned to primary controller j:
    # either choose one existing backup controller k,
    # or mark switch s as residual for new-controller placement.
    for s in switches:
        for j in controllers:
            if (s, j) in y:
                m.addConstr(
                    gp.quicksum(bkp[s, j, k] for k in controllers if k != j)
                    + residual[s, j]
                    == y[s, j],
                    name=f"backup_or_residual_{s}_{j}"
                )

    # Capacity check for every single-controller failure case.
    # For each failed controller j, surviving controller k can receive
    # some of j's switches only if k remains within usable capacity.
    for j in controllers:
        for k in controllers:
            if k == j:
                continue

            recovered_load_j_to_k = gp.quicksum(
                float(loads[s]) * bkp[s, j, k]
                for s in switches
                if (s, j, k) in bkp
            )

            m.addConstr(
                load_expr[k] + recovered_load_j_to_k
                <= float(GLOBAL_THRESHOLD) * float(capacities[k]),
                name=f"backup_cap_fail_{j}_to_{k}"
            )

    # ============================================================
    # POST-FAILURE LOAD BALANCE OBJECTIVE
    # For each failed controller j, after its orphan switches are
    # reassigned to surviving controllers, keep survivor loads balanced.
    # ============================================================

    postfail_Lmax = {}
    postfail_Lmin = {}

    for j in controllers:
        postfail_Lmax[j] = m.addVar(lb=0.0, name=f"postfail_Lmax_fail_{j}")
        postfail_Lmin[j] = m.addVar(lb=0.0, name=f"postfail_Lmin_fail_{j}")

        for k in controllers:
            if k == j:
                continue

            recovered_load_j_to_k = gp.quicksum(
                float(loads[s]) * bkp[s, j, k]
                for s in switches
                if (s, j, k) in bkp
            )

            post_failure_load_k = load_expr[k] + recovered_load_j_to_k

            m.addConstr(
                post_failure_load_k <= postfail_Lmax[j],
                name=f"postfail_Lmax_fail_{j}_survivor_{k}"
            )

            m.addConstr(
                post_failure_load_k >= postfail_Lmin[j],
                name=f"postfail_Lmin_fail_{j}_survivor_{k}"
            )

    post_failure_balance_obj = gp.quicksum(
        postfail_Lmax[j] - postfail_Lmin[j]
        for j in controllers
    )


    # residual_load[j] is total load of switches of failed controller j
    # that could not be backed up by existing controllers.
    # Exact switch identities are stored by residual[s,j].
    residual_load = {}
    for j in controllers:
        residual_load[j] = m.addVar(lb=0.0, name=f"residual_load_fail_{j}")

        m.addConstr(
            residual_load[j] ==
            gp.quicksum(
                float(loads[s]) * residual[s, j]
                for s in switches
                if (s, j) in residual
            ),
            name=f"residual_load_def_{j}"
        )
    for j in controllers:

        # At most one new controller if controller j fails
        m.addConstr(
            gp.quicksum(rctrl[j, v] for v in residual_candidates) <= 1,
            name=f"one_residual_controller_{j}"
        )

        # Residual switches imply a residual controller exists
        for s in switches:
            if (s, j) in residual:
                m.addConstr(
                    residual[s, j] <= gp.quicksum(
                        rctrl[j, v] for v in residual_candidates
                    ),
                    name=f"residual_requires_controller_{s}_{j}"
                )

        # Capacity of selected residual controller
        m.addConstr(
            residual_load[j] <=
            GLOBAL_THRESHOLD *
            gp.quicksum(
                old_node_capacity[v] * rctrl[j, v]
                for v in residual_candidates
            ),
            name=f"residual_capacity_{j}"
        )
    # Backup cost: amount of load assigned to existing backups.
    backup_cost_expr = gp.quicksum(
        float(loads[s]) * bkp[s, j, k]
        for (s, j, k) in backup_pairs
    )

    # Strong penalty: residual should happen only when existing controllers
    # cannot absorb the switch under the capacity constraints above.
    BIG_RES_PENALTY = 1e6

    BIG_RES_LOAD = 1e6      # penalize residual load
    BIG_RES_NODE = 1e5      # penalize opening a new controller

    residual_cost_expr = (
        BIG_RES_LOAD * gp.quicksum(residual_load[j] for j in controllers)
        +
        BIG_RES_NODE * gp.quicksum(rctrl[j, v] for (j, v) in rctrl_pairs)
    )

    resiliency_obj = (
        backup_cost_expr
        + residual_cost_expr
        + post_failure_balance_obj
    )



    a = max(0.0, float(alpha))
    b = max(0.0, float(beta))
    g = max(0.0, float(gamma_res))

    s_abg = a + b + g
    if s_abg == 0.0:
        a, b, g = 1.0, 0.0, 0.0
    else:
        a, b, g = a / s_abg, b / s_abg, g / s_abg

    m.setObjective(
        a * base_obj
        + b * migration_cost_expr
        + g * resiliency_obj,
        GRB.MINIMIZE
    )


    # ============================================================
    # POST-SOLVE HANDLING (WITH STATUS PROPAGATION)
    # ============================================================

    m.optimize()

    # -----------------------------
    # IIS dump for infeasible
    # -----------------------------
    if m.Status == GRB.INFEASIBLE:
        iis_dir = os.path.join(RESULTS_FOLDER, "iis_reports")
        os.makedirs(iis_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")

        try:
            m.setParam(GRB.Param.Presolve, 0)
            m.computeIIS()
            fname = os.path.join(
                iis_dir,
                f"{topology_name or 'topo'}_arc_mcf_{ts}.ilp"
            )
            m.write(fname)
            print("IIS written to:", fname)
            status_msg = f"INFEASIBLE_MCF_ARC (IIS:{fname})"
        except:
            status_msg = "INFEASIBLE_MCF_ARC (IIS_FAILED)"

        return (
            {}, {}, None, 0, {}, {}, None,
            {
                "num_mig": 0,
                "cc_transfer": 0.0,
                "delta_rt": 0.0,
                "total": 0.0,
            },
            {},
            status_msg
        )

    # -----------------------------
    # INF_OR_UNBD
    # -----------------------------
    if m.Status == GRB.INF_OR_UNBD:
        return (
            {}, {}, None, 0, {}, {}, None,
            {},
            {},
            "INF_OR_UNBOUNDED_MCF_ARC"
        )

    # -----------------------------
    # TIME LIMIT (no solution)
    # -----------------------------
    if m.Status == GRB.TIME_LIMIT and m.SolCount == 0:
        return (
            {}, {}, None, 0, {}, {}, None,
            {},
            {},
            "TIME_LIMIT_NO_SOLUTION_MCF_ARC"
        )

    # -----------------------------
    # No feasible solution
    # -----------------------------
    if m.SolCount == 0:
        return (
            {}, {}, None, 0, {}, {}, None,
            {},
            {},
            f"NO_FEASIBLE_SOLUTION_STATUS_{m.Status}"
        )

    # ============================================================
    # NORMAL EXTRACTION (OPTIMAL / FEASIBLE)
    # ============================================================

    status_msg = "OPTIMAL" if m.Status == GRB.OPTIMAL else f"FEASIBLE_STATUS_{m.Status}"

    final_assign: Dict[int, int] = {}
    final_loads = {c: float(lam[c].X) for c in controllers}

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
            {}, {}, None, 0, {}, {}, None,
            {},
            {},
            f"NO_FEASIBLE_COMPLETE_ASSIGNMENT_MCF_ARC_MISSING_{len(missing_assign)}"
        )

    # ----------------------------
    # Extract bundle
    # ----------------------------
    bundle = extract_mcf_solution_bundle(
        switches=switches,
        final_assign=final_assign,
        f=f,
        G=G,
        loads=loads,
        msg_bits=msg_bits,
        allow_path_splitting=allow_path_splitting,
    )

    paths_by_switch = bundle["paths_by_switch"]
    paths_used_by_switch = bundle["paths_used_by_switch"]
    usage_e = bundle["usage_e"]

    util_e = {
        e: (usage_e[e] / edge_caps_e[e])
        for e in usage_e
        if edge_caps_e.get(e, float("inf")) < float("inf") and edge_caps_e[e] > 0
    }

    # ----------------------------
    # RT metrics
    # ----------------------------
    resp_ms_by_switch = {s: float(T_ms[s].X) for s in switches}
    queue_ms_by_ctrl = {c: 1000.0 * float(W_sec[c].X) for c in controllers}

    prop_ms_by_switch = {}
    for s in switches:
        c = final_assign.get(s)
        if c is None:
            continue
        steiner_ms = float(sync_per_ctrl_ms) if c in controllers else 0.0
        prop_ms_by_switch[s] = max(
            0.0,
            resp_ms_by_switch.get(s, 0.0)
            - queue_ms_by_ctrl.get(c, 0.0)
            - steiner_ms
        )

    # ----------------------------
    # Migration stats
    # ----------------------------
    mig_count = sum(1 for s in switches if z[s].X > 0.5)

    final_mean_rt_ms = float(mean_T_ms.X)
    init_mean_rt_ms_f = float(init_mean_rt_ms or 0.0)

    delta_rt_total = final_mean_rt_ms - init_mean_rt_ms_f
    delta_rt_pos = max(0.0, delta_rt_total)

    mig_cost = {
        "count": int(mig_count),
        "cc_transfer": 0.0,
        "delta_rt": float(delta_rt_total),
        "delta_rt_pos": float(delta_rt_pos),
    }

    obj_val = float(m.ObjVal)
    mip_gap = float(m.MIPGap) if m.IsMIP else None

    rt_metrics = {
        "T_ms_by_switch": resp_ms_by_switch,
        "prop_ms_by_switch": prop_ms_by_switch,
        "queue_ms_by_ctrl": queue_ms_by_ctrl,
        "mean_rt_ms_final": final_mean_rt_ms,
        "mean_rt_ms_init": init_mean_rt_ms_f,
        "delta_mean_rt_ms": delta_rt_total,
        "delta_mean_rt_pos_ms": delta_rt_pos,
    }
    selected_residual_controller = {
        j: v
        for (j, v), var in rctrl.items()
        if var.X > 0.5
    }
    paths_new = build_paths_sc_from_switch_paths(final_assign, paths_by_switch)
    if resilience_log_dir is not None:
        safe_topo = str(topology_name or "topology").replace("/", "_").replace(" ", "_")

        resilience_log_file = os.path.join(
            resilience_log_dir,
            safe_topo,
            f"{safe_topo}_run{int(run_index):03d}_resilience_debug.log"
        )

        _write_resilience_debug_log(
            log_file=resilience_log_file,
            topology_name=topology_name or "topology",
            run_index=run_index,
            controllers=controllers,
            switches=switches,
            loads=loads,
            capacities=capacities,
            final_loads=final_loads,
            bkp=bkp,
            residual=residual,
            backup_cost_expr=backup_cost_expr,
            residual_cost_expr=residual_cost_expr,
            post_failure_balance_obj=post_failure_balance_obj,
            resiliency_obj=resiliency_obj,
            gamma_res=gamma_res,
        )
 
    # ============================================================
    # EXTRA RESILIENCE PLOTS — ONLY FROM MCF_ARC
    # One image per failed controller
    # ============================================================

    if plot_recovery and plot_pos is not None and plot_save_dir is not None:

        recovery_plot_dir = os.path.join(plot_save_dir, "recovery_plots")
        os.makedirs(recovery_plot_dir, exist_ok=True)

        for failed_c in controllers:

            residual_c = selected_residual_controller.get(failed_c)

            orphan_switches = [
                s for s in switches
                if final_assign.get(s) == failed_c
            ]

            if not orphan_switches:
                continue

            recovery_assign = dict(final_assign)

            # remove failed controller assignment
            for s in orphan_switches:
                assigned = False

                # first try optimizer-selected existing backup controllers
                for k in controllers:
                    if k == failed_c:
                        continue

                    if (s, failed_c, k) in bkp and bkp[s, failed_c, k].X > 0.5:
                        recovery_assign[s] = k
                        assigned = True
                        break

                # if optimizer marked this switch as residual,
                # assign it to optimizer-selected residual controller
                if not assigned:
                    if residual_c is not None and (s, failed_c) in residual:
                        if residual[s, failed_c].X > 0.5:
                            recovery_assign[s] = residual_c
                            assigned = True

                # safety fallback: keep it away from failed controller
                if not assigned:
                    recovery_assign[s] = residual_c if residual_c is not None else failed_c

            recovery_loads = defaultdict(float)
            for s, c in recovery_assign.items():
                if c == failed_c:
                    continue
                recovery_loads[c] += float(loads.get(s, 0.0))

            recovery_loads = dict(recovery_loads)

            # controllers shown in recovery figure:
            # remove failed controller, add residual controller if selected
            plot_controllers = [c for c in controllers if c != failed_c]

            if residual_c is not None and residual_c not in plot_controllers:
                plot_controllers.append(residual_c)

            # capacities shown in recovery figure
            plot_controller_capacity = dict(capacities)

            backup_capacity = None
            if residual_c is not None:
                backup_capacity = float(node_capacities.get(residual_c, 0.0))
                plot_controller_capacity[residual_c] = backup_capacity

            plot_final_vs_recovery_assignment(
                G=G,
                pos=plot_pos,
                switches=switches,
                controllers=plot_controllers,

                final_assign=final_assign,
                recovery_assign=recovery_assign,

                loads=loads,
                final_loads=final_loads,
                recovery_loads=recovery_loads,

                topology_name=plot_topology_name or topology_name or "topology",
                save_dir=recovery_plot_dir,

                controller_capacity=plot_controller_capacity,
                failed_controller=failed_c,
                backup_controller=residual_c,
                backup_capacity=backup_capacity,

                file_tag=plot_file_tag
            )
    return (
        final_assign,
        final_loads,
        obj_val,
        int(mig_count),
        usage_e,
        rt_metrics,
        mip_gap,
        mig_cost,
        paths_new,
        status_msg
    )
