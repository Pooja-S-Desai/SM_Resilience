# Switch_migration_optimizer.py

import os
import math
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from helpers import (
    RESULTS_FOLDER,
    CAPACITY_THRESHOLD as GLOBAL_THRESHOLD,
    TIME_LIMIT
)
from rt_metrics import build_paths_sc_from_switch_paths
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def run_migration_optimizer(
    G,
    switches,
    controllers,
    dij,
    init_assign,
    loads,
    capacities,
    topology_name=None,
    *,
    # Objectives:
    # "migration_cost" | "min_max_util" | "min_sum_util" | "variance" | "min_dev" | "maxmin"
    objective_type,
    # --- NEW for migration-cost objective ---
    Dcc: dict | None = None,             # controller↔controller shortest path distances
    sync_per_ctrl_ms: float = 0.0,   # total Steiner backbone weight (scalar)
    w_mig: float = 1.0,                  # weight for number of migrations
    w_delta_path: float = 1.0,           # weight for S→C positive path-cost increase
    w_cc: float = 1.0,                   # weight for C↔C transfer distance
    w_steiner: float = 1.0,              # weight for Steiner broadcast per migration
    # --- Optional shortest-path bandwidth caps (kept as in your file) ---
    edge_caps_e: dict | None = None,     # {(min(u,v),max(u,v)): cap_bits}
    msg_bits: int = 128,
    cost_mode: str = "weight",           # "weight" (geo) or "hops"
    alpha:float,
    beta:float,
    
):
    """
    Optimize switch→controller assignment with classic fairness objectives
    and a migration-cost-only objective when requested.

    Constraints:
      - Each switch assigned to exactly one controller
      - Controller usable capacity: sum_s loads[s] * y[s,c] ≤ GLOBAL_THRESHOLD * capacities[c]
      - Self-host: y[c,c] == 1 if c is a switch
      - (commented, optional) Per-switch path-cost guard: post ≤ 1.20 × baseline
      - (commented, optional) Migration budget: Σ_s z[s] ≤ ceil(MIG_CONTROL_THRESHOLD * |switches|)
      - z[s] equals “did s migrate?” exactly (tight equality)

    Objectives (choose via `objective_type`):
      - "migration_cost" : Minimize Σ components defined in the header comment
      - "min_max_util"   : Minimize max utilization across controllers
      - "min_sum_util"   : Minimize sum of utilizations
      - "variance"       : Minimize variance of loads (QP)
      - "min_dev"        : Minimize L1 deviation from mean utilization
      - "maxmin"         : Minimize (Lmax - Lmin) over controller loads
    """
    model = gp.Model("switch_migration")
    model.setParam('OutputFlag', 0)
    model.setParam("Time_Limit", float(TIME_LIMIT))

    # ---------- Variables ----------
    # y[s,c] = 1 if switch s is assigned to controller c
    y = model.addVars(switches, controllers, vtype=GRB.BINARY, name="assign")
    # z[s] = 1 if switch s changes controller (migration)
    z = model.addVars(switches, vtype=GRB.BINARY, name="migrated")

    # Initial assignment indicator
    x0 = {(s, c): int(init_assign[s] == c) for s in switches for c in controllers}


    # Usable capacity per controller
    cap_lim = {c: GLOBAL_THRESHOLD * capacities[c] for c in controllers}

    # ---------- Constraints ----------
    # 1) One controller per switch
    for s in switches:
        model.addConstr(gp.quicksum(y[s, c] for c in controllers) == 1, name=f"assign_{s}")

    # 2) Capacity limit (and build load_expr for objectives)
    load_expr = {
        c: gp.quicksum(loads[s] * y[s, c] for s in switches)
        for c in controllers
    }
    for c in controllers:
        model.addConstr(load_expr[c] <= cap_lim[c], name=f"capacity_{c}")

    # 3) Migration detector (tight equality: z[s] == 1 iff new controller != old controller)
    for s in switches:
        c0 = init_assign.get(s, None)
        if c0 in controllers:
            model.addConstr(
                z[s] == gp.quicksum(y[s, c] for c in controllers if c != c0),
                name=f"z_equals_migrate_{s}"
            )
        else:
            # Policy: count first assignment as migration (change to ==0 if you don't want that)
            model.addConstr(
                z[s] == gp.quicksum(y[s, c] for c in controllers),
                name=f"z_equals_first_assign_{s}"
            )

    # 4) Self-host: if controller node is also a switch, it must serve itself
    for c in controllers:
        if c in switches:
            model.addConstr(y[c, c] == 1, name=f"controller_serves_itself_{c}")

   
    base_to_old = {}
    for s in switches:
        c0 = init_assign.get(s, None)
        base_to_old[s] = float(dij.get((s, c0), 0.0)) if c0 in controllers else 0.0

    delta_pos = {}
    for s in switches:
        for c in controllers:
            d_new = float(dij.get((s, c), 0.0))
            inc   = d_new - base_to_old[s]
            delta_pos[(s, c)] = inc if inc > 0.0 else 0.0

    dcc_pair = {}
    for s in switches:
        c0 = init_assign.get(s, None)
        for c in controllers:
            if c0 in controllers and c != c0 and Dcc is not None:
                if isinstance(Dcc.get(c0, {}), dict):
                    dcc_pair[(s, c)] = float(Dcc.get(c0, {}).get(c, 0.0))
                else:
                    dcc_pair[(s, c)] = float(Dcc.get((c0, c), 0.0))
            else:
                dcc_pair[(s, c)] = 0.0

    num_migrations = gp.quicksum(z[s] for s in switches)
    path_increase  = gp.quicksum(delta_pos[(s, c)] * y[s, c] for s in switches for c in controllers)
    cc_transfer    = gp.quicksum(dcc_pair[(s, c)] * y[s, c] for s in switches for c in controllers)
    steiner_bcast  = float(sync_per_ctrl_ms) * num_migrations

    migration_cost_expr = (
          float(w_mig)        * num_migrations
        + float(w_delta_path) * path_increase
        + float(w_cc)         * cc_transfer
        + float(w_steiner)    * steiner_bcast
    )
    # =======================================================================

       # ---------- Objective selection (multi-objective with ALPHA/BETA) ----------
    # Build the base objective expression (or 0 if you choose "migration_cost")
    base_obj = 0.0

    if objective_type == "min_max_util":
        U = model.addVar(lb=0.0, name="Umax")
        for c in controllers:
            model.addConstr(load_expr[c] <= U * float(capacities[c]), name=f"util_cap_{c}")
        base_obj = U

    elif objective_type == "min_sum_util":
        base_obj = gp.quicksum(load_expr[c] / float(capacities[c]) for c in controllers)

    elif objective_type == "variance":
        total = sum(float(loads[s]) for s in switches)
        k = max(1, len(controllers))
        mu = total / k
        base_obj = gp.quicksum((load_expr[c] - mu) * (load_expr[c] - mu) for c in controllers)  # QP

    elif objective_type == "min_dev":
        total_load = sum(float(loads[s]) for s in switches)
        k = max(1, len(controllers))
        mu = total_load / k                     # mean LOAD per controller

        d = {c: model.addVar(lb=0.0, name=f"abs_dev_{c}") for c in controllers}

        for c in controllers:
            Lc = load_expr[c]                   # controller LOAD
            model.addConstr(Lc - mu <= d[c])
            model.addConstr(mu - Lc <= d[c])

        base_obj = gp.quicksum(d[c] for c in controllers)
        
    elif objective_type == "maxmin":
        Lmax = model.addVar(name="L_max")
        Lmin = model.addVar(name="L_min")
        for c in controllers:
            model.addConstr(load_expr[c] <= Lmax)
            model.addConstr(load_expr[c] >= Lmin)
        base_obj = (Lmax - Lmin)

    elif objective_type == "migration_cost":
        # No separate base term; combine will use only the migration part via BETA.
        base_obj = 0.0

    else:
        raise ValueError(f"Unknown objective_type: {objective_type}")

    # Combine with ALPHA/BETA (normalize; if both zero, default to ALPHA=1, BETA=0)
    a = float(alpha)
    b = float(beta)

    # Final multi-objective
    model.setObjective(a * base_obj + b * migration_cost_expr, GRB.MINIMIZE)


    # ---------- Solve ----------
    total_load = sum(float(loads[s]) for s in switches)
    total_cap  = sum(float(cap_lim[c]) for c in controllers)
    if total_load > total_cap:
        print(f"⚠️ Warning: Load {total_load:.2f} exceeds total usable capacity {total_cap:.2f}")
        print("👉 Proceeding anyway to allow IIS if infeasible...")


    # ============================================================
    # SOLVE + STATUS HANDLING (SHORTEST PATH OPTIMIZER)
    # ============================================================

    model.optimize()

    # -----------------------------
    # INFEASIBLE → IIS
    # -----------------------------
    if model.Status == GRB.INFEASIBLE:

        iis_dir = os.path.join(RESULTS_FOLDER, "iis_reports")
        os.makedirs(iis_dir, exist_ok=True)

        try:
            model.setParam(GRB.Param.Presolve, 0)
            model.setParam(GRB.Param.IISMethod, 1)
            model.computeIIS()

            iis_path = os.path.join(
                iis_dir,
                f"{topology_name or 'topo'}_sp_optimizer.iis"
            )
            model.write(iis_path)

            print("IIS written to:", iis_path)
            status_msg = f"INFEASIBLE_SP (IIS:{iis_path})"

        except Exception as e:
            print("IIS failed:", e)
            status_msg = "INFEASIBLE_SP (IIS_FAILED)"

        return (
            {},
            {},
            {},
            None,
            0,
            None,
            status_msg
        )

    # -----------------------------
    # INF_OR_UNBD
    # -----------------------------
    if model.Status == GRB.INF_OR_UNBD:
        return (
            {},
            {},
            {},
            None,
            0,
            None,
            "INF_OR_UNBOUNDED_SP"
        )

    # -----------------------------
    # TIME LIMIT (no solution)
    # -----------------------------
    if model.Status == GRB.TIME_LIMIT and model.SolCount == 0:
        return (
            {},
            {},
            {},
            None,
            0,
            None,
            "TIME_LIMIT_NO_SOLUTION_SP"
        )

    # -----------------------------
    # NO SOLUTION
    # -----------------------------
    if model.SolCount == 0:
        return (
            {},
            {},
            {},
            None,
            0,
            None,
            f"NO_FEASIBLE_SOLUTION_STATUS_{model.Status}_SP"
        )

    # -----------------------------
    # NORMAL CASE
    # -----------------------------
    status_msg = "OPTIMAL" if model.Status == GRB.OPTIMAL else f"FEASIBLE_STATUS_{model.Status}"

    mip_gap_solved = float(model.MIPGap) if model.SolCount > 0 else None
    # ---------- Extract solution ----------
    final_assign = {}
    for s in switches:
        for c in controllers:
            if y[s, c].X > 0.5:
                final_assign[s] = c
                break

    final_loads = {c: float(load_expr[c].getValue()) for c in controllers}

    # Simple report vs usable capacity
    for c in sorted(controllers):
        thr = float(cap_lim[c])
        stat = "Overloaded" if final_loads[c] > thr else "Underloaded"
        print(f"Controller {c:>2}: Load = {final_loads[c]:>8.2f} | Usable = {thr:>8.2f} | {stat}")

    migration_count = sum(1 for s in switches if final_assign[s] != init_assign[s])
    # ---------- Extract solution ----------
    final_assign = {}
    paths = {}   # NEW

    for s in switches:
        for c in controllers:
            if y[s, c].X > 0.5:
                final_assign[s] = c

                # 🔹 Extract shortest path from s to c
                try:
                    path = nx.shortest_path(G, source=s, target=c, weight="weight")
                except nx.NetworkXNoPath:
                    path = None

                paths[s] = path
                break

    paths_sc = build_paths_sc_from_switch_paths(final_assign, paths)


    obj_val = float(model.objVal) if model.SolCount > 0 else None
    return final_assign, final_loads, paths_sc,obj_val, migration_count, mip_gap_solved,status_msg