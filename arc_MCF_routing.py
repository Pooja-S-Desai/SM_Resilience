from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Any, Optional
import math
import gurobipy as gp
from gurobipy import GRB
from rt_metrics import compute_response_metrics
from link_checks import usage_on_paths_undirected
from iis_logger import write_iis_with_context

def _undir(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)

def _build_arcs_from_undirected_edges(edges_undirected: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    arcs = []
    for (u,v) in edges_undirected:
        u = int(u); v = int(v)
        if u == v: 
            continue
        arcs.append((u,v))
        arcs.append((v,u))
    return arcs

def _extract_path_from_binary_arcs(arcs_used: Dict[Tuple[int,int], int], s: int, t: int) -> List[int]:
    # arcs_used has 1 on used directed arcs for a SINGLE commodity
    # Follow from s until reaching t; includes simple cycle guard.
    nxt = {}
    for (u,v), val in arcs_used.items():
        if val == 1:
            if u in nxt:
                # should not happen if model is correct (outdegree=1 on path nodes)
                pass
            nxt[u] = v

    path = [s]
    cur = s
    seen = {s}
    for _ in range(100000):
        if cur == t:
            break
        if cur not in nxt:
            raise ValueError(f"Path extraction failed: dead-end at {cur} for commodity {s}->{t}")
        cur = nxt[cur]
        if cur in seen:
            raise ValueError(f"Path extraction failed: cycle detected for commodity {s}->{t}")
        seen.add(cur)
        path.append(cur)

    if path[-1] != t:
        raise ValueError(f"Path extraction failed: did not reach t={t} from s={s}")
    return path


def solve_binary_mcf_routing_and_init_rt(
    *,
    G,
    switches: List[int],
    assignment: Dict[int,int],              # fixed init assignment (s->c)
    loads_req_s: Dict[int,float],           # req/s per switch
    capacities_req_s: Dict[int,float],      # req/s capacity per controller
    edge_caps_bits: Dict[Tuple[int,int], float],  # undirected (u,v)->cap bits/s
    msg_bits: int,
    global_threshold: float,
    rho_max: float,
    sync_per_ctrl_ms: float = 0.0,
    round_trip: bool = True,
    fiber_sec_per_km: float = 5e-6,
    cost_mode: str = "weight",
    time_limit: float = 60.0,
    output_flag: int = 0,
) -> Tuple[
    Dict[Tuple[int,int], List[int]],        # paths_sc: (s,c)->[nodes]
    Dict[Tuple[int,int], float],            # usage_e: undirected edge->bits/s
    Dict[str, Any],                         # init_rt_path dict (like you already expect)
]:
    """
    Single call that:
      1) solves binary unsplittable MCF routing for fixed assignment (only commodities s->assigned c, skip s==c)
      2) extracts one path for each routed pair
      3) computes link usage (bits/s)
      4) computes init RT using extracted path propagation + M/M/1 W + sync term
    """

    # ----- build commodities (ONLY assignment pairs) -----
    commodities = []
    for s, c in assignment.items():
        s = int(s); c = int(c)
        if s == c:
            continue
        commodities.append((s, c))

    # ----- arcs from undirected edges -----
    undirected_edges = list(edge_caps_bits.keys())
    arcs = _build_arcs_from_undirected_edges(undirected_edges)

    # demand bits/sec per commodity = load(req/s)*msg_bits
    demand_bits_s = {int(s): float(loads_req_s.get(int(s), 0.0)) * float(msg_bits) for s in switches}

    m = gp.Model("init_fixed_assignment_binary_mcf")
    m.Params.OutputFlag = int(output_flag)
    m.Params.TimeLimit = float(time_limit)

    # f[s,t,u,v] binary: commodity (s->t) uses directed arc (u,v)
    f = m.addVars(
        [(s, t, u, v) for (s,t) in commodities for (u,v) in arcs],
        vtype=GRB.BINARY,
        name="f"
    )

    # -------------------------
    # (1) Link capacity constraint (UNDIRECTED)
    # sum_i f_i(u,v)*d_i + f_i(v,u)*d_i <= cap(u,v)
    # -------------------------
    for (u,v), cap in edge_caps_bits.items():
        u = int(u); v = int(v)
        expr = gp.LinExpr()
        for (s,t) in commodities:
            d = demand_bits_s[int(s)]
            expr += d * f[s,t,u,v]
            expr += d * f[s,t,v,u]
        m.addConstr(expr <= float(cap), name=f"cap_{u}_{v}")

    # -------------------------
    # (2)(3)(4) Flow conservation (directed), per commodity
    # For all nodes x:
    #   out(x)-in(x) =  1 if x==s
    #                = -1 if x==t
    #                =  0 otherwise
    # -------------------------
    # precompute incident arcs per node for speed
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for (u,v) in arcs:
        out_arcs[int(u)].append((u,v))
        in_arcs[int(v)].append((u,v))

    for (s,t) in commodities:
        s = int(s); t = int(t)
        for x in G.nodes():
            x = int(x)
            out_expr = gp.quicksum(f[s,t,u,v] for (u,v) in out_arcs[x])
            in_expr  = gp.quicksum(f[s,t,u,v] for (u,v) in in_arcs[x])

            rhs = 0
            if x == s:
                rhs = 1
            elif x == t:
                rhs = -1

            m.addConstr(out_expr - in_expr == rhs, name=f"flow_{s}_{t}_{x}")

    # Objective: feasibility only (or small minimize total arcs)
    m.setObjective(gp.quicksum(f[s,t,u,v] for (s,t) in commodities for (u,v) in arcs), GRB.MINIMIZE)

    m.optimize()
    if m.Status == GRB.INFEASIBLE:

        print("Model is INFEASIBLE → computing IIS")

        status_msg = f"INFEASIBLE_INIT_ROUTING_MCF_ARC_{m.Status}"

        try:
            m.setParam(GRB.Param.Presolve, 0)
            m.setParam(GRB.Param.IISMethod, 1)
            m.computeIIS()

            write_iis_with_context(
                m,
                base_dir="./iis_logs/mcf_arc_initial_milp",
                topo_name=str(G.graph.get("name", "unknown")),
                run_index=-1,
                stage="mcf_arc_initial_milp",
            )
        except Exception as e:
            print("IIS failed:", e)
            status_msg += "_IIS_FAILED"

        # ✅ RETURN CLEAN FAILURE (NO CRASH)
        return {}, {}, {"status_msg": status_msg}

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):

        status_msg = f"FAILED_INIT_ROUTING_STATUS_{m.Status}"

        return {}, {}, {"status_msg": status_msg}
    # ----- extract paths_sc -----
    paths_sc: Dict[Tuple[int,int], List[int]] = {}

    for (s,t) in commodities:
        arcs_used = {}
        for (u,v) in arcs:
            val = f[s,t,u,v].X
            arcs_used[(u,v)] = 1 if val > 0.5 else 0
        paths_sc[(s,t)] = _extract_path_from_binary_arcs(arcs_used, s, t)

    # self-assigned: trivial path (no edges)
    for s, c in assignment.items():
        s = int(s); c = int(c)
        if s == c:
            paths_sc[(s,c)] = [s]



    usage_shortest_init, util_e = usage_on_paths_undirected(
        G,
        paths_sc, #shortest_paths
        assignment,
        loads_req_s,
        msg_bits,
        edge_caps=edge_caps_bits  # stressed bandwidth
    )
    rt_metrics = compute_response_metrics(
        G=G,
        assignment=assignment,
        loads=loads_req_s,
        capacities=capacities_req_s,                       
        paths=paths_sc,
        round_trip=round_trip,
        per_ctrl_ms=sync_per_ctrl_ms,
    )
   

    return dict(paths_sc), usage_shortest_init, rt_metrics, "OPTIMAL"
