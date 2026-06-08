
# Path_initial_routing
import gurobipy as gp
from gurobipy import GRB
import math
OVERLOAD_FACTOR = 1


def choose_single_kpath_for_assignment(
    *,
    switches,
    assignment,              # {s: c}
    loads,                   # {s: req/s}
    msg_bits,                # int
    edge_caps_e,             # {(u,v): cap_bits/s} undirected keys
    sc_kpaths,               # {(s,c): [path0, path1, ...]} nodes-list or {"nodes":[...]}
    sc_kpath_edges=None,     # {(s,c,k): [(u,v),...]} optional
    sc_kpath_cost_ms=None,   # {(s,c,k): cost_ms} optional objective
    time_limit=60.0,
    controllers,
    capacities,
    global_threshold,
    rho_max = 0.95,
    Dcc,        # controller-to-controller distance (same structure you already use)
    sync_per_ctrl_ms: float = 0.0, 
):
    """
    For a FIXED switch->controller assignment, choose exactly ONE k-path per switch
    such that all link-capacity constraints are satisfied.

    Returns:
      chosen_k: {s: k}
      chosen_path_nodes: {(s,c): [n0,n1,...]}
      usage_e: {(u,v): used_bits/s}
    """
    def _undir(u, v):
        return (u, v) if u < v else (v, u)

    demand_bits = {s: float(loads.get(s, 0.0)) * float(msg_bits) for s in switches}

    # Build candidate Kset only for the assigned controller of each switch
    Kset = []              # (s,k)
    path_edges = {}        # (s,k) -> [undirected edges]
    path_nodes = {}        # (s,k) -> [nodes]
    for s in switches:
        c = assignment[s]
        if s == c:
            # trivial "path": no edges; still treat as a single option k=0
            Kset.append((s, 0))
            path_edges[(s, 0)] = []
            path_nodes[(s, 0)] = [s]
            continue

        plist = sc_kpaths.get((s, c), [])
        if not plist:
            raise ValueError(f"No cached paths for (s={s}, c={c}).")

        # keep only ks that exist in cost dict if provided
        ks = list(range(len(plist)))
        if sc_kpath_cost_ms is not None:
            ks = [k for k in ks if (s, c, k) in sc_kpath_cost_ms]
        if not ks:
            raise ValueError(f"No usable k for (s={s}, c={c}) after filtering.")

        for k in ks:
            if sc_kpath_edges is not None and (s, c, k) in sc_kpath_edges:
                eds = [_undir(u, v) for (u, v) in sc_kpath_edges[(s, c, k)]]
                nodes = [s]  # optional; keep nodes from plist below if you want
                p = plist[k]
                nodes = p.get("nodes", []) if isinstance(p, dict) else p
            else:
                p = plist[k]
                nodes = p.get("nodes", []) if isinstance(p, dict) else p
                eds = [_undir(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]

            Kset.append((s, k))
            path_edges[(s, k)] = eds
            path_nodes[(s, k)] = nodes

    # Build edge universe from capacities
    E = list(edge_caps_e.keys())

    m = gp.Model("fixed_assignment_single_path_choice")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", float(time_limit))

    # x[s,k] = 1 if switch s uses k-th path to its assigned controller
    x = m.addVars(Kset, vtype=GRB.BINARY, name="x")

    # Exactly one path per switch
    for s in switches:
        ks = [k for (ss, k) in Kset if ss == s]
        m.addConstr(gp.quicksum(x[s, k] for k in ks) == 1, name=f"one_path_{s}")

    # Link capacity constraints: sum_{s,k using e} demand_bits[s] * x[s,k] <= cap_e
    for e in E:
        cap = float(edge_caps_e[e])
        expr = gp.LinExpr()
        for (s, k) in Kset:
            if e in path_edges[(s, k)]:
                expr += demand_bits[s] * x[s, k]
        

        m.addConstr(expr <= OVERLOAD_FACTOR * cap, name=f"cap_{e[0]}_{e[1]}")


        # m.addConstr(expr <= cap, name=f"cap_{e[0]}_{e[1]}")

    # Objective: optional (prefer min RTT/cost if cost dict available)
    if sc_kpath_cost_ms is not None:
        obj = gp.quicksum(float(sc_kpath_cost_ms[(s, assignment[s], k)]) * demand_bits[s] * x[s, k] for (s, k) in Kset if s != assignment[s])
        m.setObjective(obj, GRB.MINIMIZE)
    else:
        m.setObjective(0.0, GRB.MINIMIZE)

    m.optimize()
   

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("path_choice_initial.ilp")
        print("IIS written to path_choice.ilp")

    # 🔥 CRITICAL FIX
    if m.SolCount == 0:
        print("⚠️ No feasible INIT path solution → fallback to shortest path")

        chosen_k = {}
        chosen_path_nodes = {}
        usage_e = {e: 0.0 for e in E}

        for s in switches:
            c = assignment[s]
            plist = sc_kpaths.get((s, c), [])

            if not plist:
                continue

            kstar = 0   # fallback: first path
            chosen_k[s] = kstar

            p = plist[kstar]
            nodes = p.get("nodes", []) if isinstance(p, dict) else p
            chosen_path_nodes[(s, c)] = nodes

            # compute usage
            edges = [_undir(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
            for e in edges:
                usage_e[e] = usage_e.get(e, 0.0) + demand_bits[s]

        return chosen_k, chosen_path_nodes, usage_e, None


    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("path_choice_initial.ilp")
        print("IIS written to path_choice.ilp")

    chosen_k = {}
    chosen_path_nodes = {}
    usage_e = {e: 0.0 for e in E}

    for s in switches:
        c = assignment[s]
        ks = [k for (ss, k) in Kset if ss == s]
        kstar = max(ks, key=lambda k: x[s, k].X)
        chosen_k[s] = int(kstar)
        chosen_path_nodes[(s, c)] = path_nodes[(s, kstar)]

        # accumulate usage
        for e in path_edges[(s, kstar)]:
            usage_e[e] = usage_e.get(e, 0.0) + demand_bits[s]

    # ---------------------------
    # NEW: init RT computed from EXACT chosen k
    # ---------------------------
    init_rt_bundle = None
    if (controllers is not None and capacities is not None and global_threshold is not None and sc_kpath_cost_ms is not None):
        controllers = [int(c) for c in controllers]
        lam = {c: 0.0 for c in controllers}
        for s in switches:
            c0 = assignment.get(s)
            if c0 in lam:
                lam[c0] += float(loads.get(s, 0.0))

        mu = {c: float(capacities[c]) * float(global_threshold) for c in controllers}

        W_ms = {}
        for c in controllers:
            muc = mu[c]
            lamc = min(lam[c], float(rho_max) * muc)
            denom = max(1e-9, muc - lamc)
            W_ms[c] = 1000.0 / denom

        prop_ms = {}
        T_init = {}
        for s in switches:
            c0 = assignment.get(s)
            if s == c0:
                prop_ms[s] = 0.0
            else:
                kstar = chosen_k[s]
                prop_ms[s] = float(sc_kpath_cost_ms[(s, c0, kstar)])

            T_init[s] = float(prop_ms[s]) + float(W_ms.get(c0, 0.0)) + float(sync_per_ctrl_ms)

        finite_vals = [v for v in T_init.values() if math.isfinite(v)]
        mean_init = (sum(finite_vals) / max(1, len(finite_vals))) if finite_vals else math.inf

        init_rt_bundle = {
            "T_init_ms_by_switch": T_init,
            "prop_init_ms_by_switch": prop_ms,
            "W_init_ms_by_ctrl": W_ms,
            "init_mean_rt_ms": mean_init,
            "lambda_by_ctrl": lam,
            "mu_by_ctrl": mu,
        }

    return chosen_k, chosen_path_nodes, usage_e, init_rt_bundle
