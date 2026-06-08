"""
Baseline-1 On Load Balancing via Switch Migration in Software-Defined Networking (Strict Paper MILP): Load-Deviation + Migration-Cost (Paper-Style)

This module implements a separate optimization routine that matches the paper-style MILP
(migration-only binary variables x[s,c], normalized controller utilization ellhat,
mean ellbar, absolute deviation sigma, and migration cost theta).

Design intent:
- Keeping topology-derived inputs from your pipeline unchanged:
  * init_assign: initial switch->controller mapping
  * loads: switch packet_in rates (req/s)
  * capacities: controller service capacities (req/s)
  * dij: switch->controller latency/distance matrix (same unit you already use)
  * Dcc: controller->controller latency/distance matrix (same unit as dij)
- Change ONLY the constraints/objective to the paper’s MILP structure.

Outputs:
- final_assign: dict switch -> controller
- final_loads: dict controller -> total load (req/s)
- meta: dict with objective and components, migrations, solver status
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Tuple, Any, Optional
from rt_metrics import build_paths_sc_from_switch_paths
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:  # pragma: no cover
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = e
else:
    _GUROBI_IMPORT_ERROR = None

import networkx as nx
def _get_v_cc(Dcc: Optional[dict], c0: Hashable, c: Hashable) -> float:
    """
    Fetch controller-to-controller value v(c0,c) from Dcc, supporting common shapes:
    - nested dict: Dcc[c0][c]
    - flat dict:   Dcc[(c0,c)]
    """
    if c0 == c:
        return 0.0
    if Dcc is None:
        return 0.0
    try:
        row = Dcc.get(c0, None)
        if isinstance(row, dict):
            return float(row.get(c, 0.0))
    except Exception:
        pass
    try:
        return float(Dcc.get((c0, c), 0.0))
    except Exception:
        return 0.0


def run_baseline1_paper_milp(
    G: nx.Graph,
    switches: Iterable[Hashable],
    controllers: Iterable[Hashable],
    init_assign: Dict[Hashable, Hashable],
    loads: Dict[Hashable, float],
    capacities: Dict[Hashable, float],
    dij: Dict[Tuple[Hashable, Hashable], float],
    Dcc: Optional[dict] = None,
    L_frac: float = 0.8,
    vartheta: float = 1.0,
    time_limit: float = 300.0,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[Dict[Hashable, Hashable], Dict[Hashable, float], Dict[str, Any]]:
    """
    Solve the strict paper-style baseline MILP.

    Parameters
    ----------
    switches, controllers : iterables
        Node IDs matching your pipeline.
    init_assign : dict
        Initial mapping s -> c0. Assumed complete (every switch assigned).
    loads : dict
        Switch load r[s] in req/s (packet_in/s).
    capacities : dict
        Controller capacity gamma[c] in req/s.
    dij : dict
        Switch-to-controller latency/distance d[s,c] (unit must be consistent with Dcc).
    Dcc : dict, optional
        Controller-to-controller latency/distance v[c0,c] (same unit as dij).
    L_frac : float
        Max allowed normalized load ellhat[c] (paper uses 0.95).
    vartheta : float
        Weight applied to v[c0,c] in theta (paper constant).
    time_limit : float
        Gurobi TimeLimit (seconds).
    mip_gap : float, optional
        Gurobi MIPGap, e.g., 0.01 for 1%.
    verbose : bool
        If False, suppress solver output.

    Returns
    -------
    final_assign : dict
    final_loads : dict
    meta : dict
        Includes objective, components, migrations, and status.
    """
    if gp is None:
        raise ImportError(f"gurobipy not available: {_GUROBI_IMPORT_ERROR}")

    switches = list(switches)
    controllers = list(controllers)

    missing = [s for s in switches if s not in init_assign]
    if missing:
        raise ValueError(f"init_assign missing switches: {missing[:10]} (total {len(missing)})")

    r = {s: float(loads.get(s, 0.0)) for s in switches}
    gamma = {c: float(capacities[c]) for c in controllers}

    # J_of[c] = switches initially assigned to controller c
    J_of = {c: [] for c in controllers}
    for s in switches:
        c0 = init_assign[s]
        if c0 not in J_of:
            raise ValueError(f"init_assign maps switch {s} to controller {c0} not in controllers list")
        J_of[c0].append(s)

    m = gp.Model("baseline1_paper_milp")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", float(time_limit))
    if mip_gap is not None:
        m.setParam("MIPGap", float(mip_gap))

    # Decision: x[s,c] = 1 if switch s migrates from its initial c0 to destination controller c (c != c0)
    x: Dict[Tuple[Hashable, Hashable], Any] = {}
    for s in switches:
        c0 = init_assign[s]
        for c in controllers:
            if c != c0:
                x[(s, c)] = m.addVar(vtype=GRB.BINARY, name=f"x[{s},{c}]")

    # Continuous variables: ellhat[c] (normalized load), ellbar (mean), sigma[c] (abs dev)
    ellhat = {c: m.addVar(lb=0.0, ub=float(L_frac), vtype=GRB.CONTINUOUS, name=f"ellhat[{c}]")
              for c in controllers}
    ellbar = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="ellbar")
    sigma = {c: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"sigma[{c}]")
             for c in controllers}

    m.update()

    # (1) Each switch migrates to at most one destination controller
    for s in switches:
        c0 = init_assign[s]
        m.addConstr(
            gp.quicksum(x[(s, c)] for c in controllers if c != c0) <= 1,
            name=f"at_most_one_mig[{s}]"
        )

    # (2) Define ellhat[c] using outgoing and incoming migrations (paper structure)
    for c in controllers:
        expr = gp.LinExpr()

        # baseline load from initial switches at c, minus those migrated away
        for s in J_of[c]:
            expr += (r[s] / gamma[c])
            expr -= (r[s] / gamma[c]) * gp.quicksum(
                x[(s, c2)] for c2 in controllers if c2 != c
            )

        # incoming migrations from other controllers to c
        for c_src in controllers:
            if c_src == c:
                continue
            for s in J_of[c_src]:
                expr += (r[s] / gamma[c]) * x[(s, c)]

        m.addConstr(ellhat[c] == expr, name=f"ellhat_def[{c}]")

    # (3) Mean load ellbar
    k = max(1, len(controllers))
    m.addConstr(k * ellbar == gp.quicksum(ellhat[c] for c in controllers), name="mean_load_def")

    # (4) Absolute deviation linearization
    for c in controllers:
        m.addConstr(ellhat[c] - ellbar <= sigma[c], name=f"dev_pos[{c}]")
        m.addConstr(ellbar - ellhat[c] <= sigma[c], name=f"dev_neg[{c}]")

    # (5) Migration cost theta(s:c0->c) (paper form)
    theta: Dict[Tuple[Hashable, Hashable], float] = {}
    for s in switches:
        c0 = init_assign[s]
        for c in controllers:
            if c == c0:
                continue
            d_sc0 = float(dij.get((s, c0), 0.0))
            d_sc = float(dij.get((s, c), 0.0))
            v_c0c = _get_v_cc(Dcc, c0, c)
            theta[(s, c)] = (
                (1.0 - (r[s] / gamma[c0]) * d_sc0) +
                ((r[s] / gamma[c]) * d_sc) +
                (float(vartheta) * v_c0c)
            )

    obj_load_dev = gp.quicksum(sigma[c] for c in controllers)
    obj_mig_cost = gp.quicksum(theta[(s, c)] * x[(s, c)] for (s, c) in x.keys())
    m.setObjective(obj_load_dev + obj_mig_cost, GRB.MINIMIZE)

   # ============================================================
    # SOLVE + STATUS HANDLING (BASELINE1 MILP)
    # ============================================================

    m.optimize()

    # -----------------------------
    # INFEASIBLE → IIS
    # -----------------------------
    if m.Status == GRB.INFEASIBLE:

        import os, time
        from helpers import RESULTS_FOLDER

        iis_dir = os.path.join(RESULTS_FOLDER, "iis_reports")
        os.makedirs(iis_dir, exist_ok=True)

        try:
            m.setParam(GRB.Param.Presolve, 0)
            m.computeIIS()

            ts = time.strftime("%Y%m%d_%H%M%S")
            iis_path = os.path.join(
                iis_dir,
                f"baseline1_milp_{ts}.ilp"
            )

            m.write(iis_path)

            print("IIS written to:", iis_path)
            status_msg = f"INFEASIBLE_B1 (IIS:{iis_path})"

        except Exception as e:
            print("IIS failed:", e)
            status_msg = "INFEASIBLE_B1 (IIS_FAILED)"

        return (
            {},
            {},
            {},
            {},
            None,
            None,
            status_msg
        )

    # -----------------------------
    # INF_OR_UNBD
    # -----------------------------
    if m.Status == GRB.INF_OR_UNBD:
        return (
            {},
            {},
            {},
            {},
            None,
            None,
            "INF_OR_UNBOUNDED_B1"
        )

    # -----------------------------
    # TIME LIMIT (no solution)
    # -----------------------------
    if m.Status == GRB.TIME_LIMIT and m.SolCount == 0:
        return (
            {},
            {},
            {},
            {},
            None,
            None,
            "TIME_LIMIT_NO_SOLUTION_B1"
        )

    # -----------------------------
    # NO SOLUTION
    # -----------------------------
    if m.SolCount == 0:
        return (
            {},
            {},
            {},
            {},
            None,
            None,
            f"NO_FEASIBLE_SOLUTION_STATUS_{m.Status}_B1"
        )

    # -----------------------------
    # NORMAL CASE
    # -----------------------------
    status_msg = "OPTIMAL" if m.Status == GRB.OPTIMAL else f"FEASIBLE_STATUS_{m.Status}"

    mip_gap_solved = float(m.MIPGap) if m.SolCount > 0 else None
    obj_val = float(m.ObjVal)

    final_assign = dict(init_assign)
    if m.SolCount > 0:
        for s in switches:
            c0 = init_assign[s]
            chosen = None
            for c in controllers:
                if c == c0:
                    continue
                if x[(s, c)].X > 0.5:
                    chosen = c
                    break
            if chosen is not None:
                final_assign[s] = chosen

    final_loads = {c: 0.0 for c in controllers}
    for s, c in final_assign.items():
        final_loads[c] += float(loads.get(s, 0.0))

    mig_count = sum(1 for s in switches if final_assign[s] != init_assign[s])

    meta = {
        "status": status_msg,
        "objective": float(m.ObjVal) if m.SolCount > 0 else None,
        "obj_load_dev": float(obj_load_dev.getValue()) if m.SolCount > 0 else None,
        "obj_mig_cost": float(obj_mig_cost.getValue()) if m.SolCount > 0 else None,
        "migrations": int(mig_count),
        "L_frac": float(L_frac),
        "vartheta": float(vartheta),
        "time_limit": float(time_limit),
        "mip_gap": mip_gap_solved
    }

    paths = {}

    for s in switches:
        c = final_assign[s]   # ✅ already computed correctly

        try:
            paths[s] = nx.shortest_path(G, source=s, target=c, weight="weight")
        except nx.NetworkXNoPath:
            paths[s] = None
            break

    paths_sc = build_paths_sc_from_switch_paths(final_assign, paths)

    return final_assign, paths_sc,final_loads, meta,obj_val,mip_gap_solved,status_msg