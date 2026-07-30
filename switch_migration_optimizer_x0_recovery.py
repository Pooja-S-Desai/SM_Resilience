# Switch_migration_optimizer.py

import os
import math
import time
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from helpers import (
    RESULTS_FOLDER,
    CAPACITY_THRESHOLD as GLOBAL_THRESHOLD,
    OVERLOAD_THRESHOLD,
    TIME_LIMIT
)
from rt_metrics import build_paths_sc_from_switch_paths, path_latency_ms
from failure_scenario_handler import process_failure_scenario
from plotting import plot_final_vs_recovery_assignment
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def run_migration_optimizer_x0_recovery(
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
    w_rt: float = 1.0,                   # weight for positive mean response-time increase
    w_cc: float = 1.0,                   # weight for C↔C transfer distance
    w_steiner: float = 1.0,              # weight for Steiner broadcast per migration
    # --- Optional shortest-path bandwidth caps (kept as in your file) ---
    edge_caps_e: dict | None = None,     # {(min(u,v),max(u,v)): cap_bits}
    msg_bits: int = 128,
    cost_mode: str = "weight",           # "weight" (geo) or "hops"
    alpha:float,
    beta:float,
    gamma_res: float = 1.0,
    init_mean_rt_ms: float | None = None,
    run_index: int = 0,
    plot_recovery: bool = False,
    plot_pos: dict | None = None,
    plot_save_dir: str | None = None,
    master_seed: int | None = None,
    switch_seed: int | None = None,
    run_number: int | None = None,
    comparison_csv_file: str | None = None,
    
):
    """
    Optimize switch→controller assignment with classic fairness objectives
    and a migration-cost-only objective when requested.

    Constraints:
      - Each switch assigned to exactly one controller
      - Controller usable capacity: sum_s loads[s] * y[s,c] ≤ GLOBAL_THRESHOLD * capacities[c]
      - Self-host: y[c,c] == 1 if c is a switch
      - Migration cost penalizes positive mean response-time increase
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
    model = gp.Model("switch_migration_x0_recovery")
    model.setParam('OutputFlag', 0)
    model.setParam("Time_Limit", float(TIME_LIMIT))

    # ---------- Variables ----------
    # y[s,c] = 1 if switch s is assigned to controller c
    y = model.addVars(switches, controllers, vtype=GRB.BINARY, name="assign")
    # z[s] = 1 if switch s changes controller (migration)
    z = model.addVars(switches, vtype=GRB.BINARY, name="migrated")

    # Initial assignment indicator
    x0 = {(s, c): int(init_assign[s] == c) for s in switches for c in controllers}


    # Load-balancing target: the shortest model must bring every controller
    # to at most the configured overload threshold (80%).
    cap_lim = {c: OVERLOAD_THRESHOLD * capacities[c] for c in controllers}

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
    cc_transfer    = gp.quicksum(dcc_pair[(s, c)] * y[s, c] for s in switches for c in controllers)
    steiner_bcast  = float(sync_per_ctrl_ms) * num_migrations

    rho_max = 0.95
    pwl_segments = 20
    lam_rt = model.addVars(controllers, lb=0.0, name="lambda_rt")
    W_ms = model.addVars(controllers, lb=0.0, name="W_ms")
    for c in controllers:
        mu_eff = float(GLOBAL_THRESHOLD) * float(capacities[c])
        xs = [i * (rho_max * mu_eff / pwl_segments) for i in range(pwl_segments + 1)]
        ys = [1000.0 / max(1e-9, mu_eff - value) for value in xs]
        model.addConstr(lam_rt[c] == load_expr[c], name=f"lambda_rt_link_{c}")
        model.addConstr(lam_rt[c] <= rho_max * mu_eff, name=f"rt_domain_{c}")
        model.addGenConstrPWL(lam_rt[c], W_ms[c], xs, ys, name=f"W_pwl_ms_{c}")

    propagation_ms = {}
    for s in switches:
        for c in controllers:
            try:
                path = nx.shortest_path(G, source=s, target=c, weight="weight")
                propagation_ms[s, c] = 2.0 * path_latency_ms(G, path)
            except nx.NetworkXNoPath:
                propagation_ms[s, c] = 1e9

    T_ms = model.addVars(switches, lb=0.0, name="T_ms")
    big_m_rt = 1e9
    for s in switches:
        for c in controllers:
            candidate = propagation_ms[s, c] + W_ms[c] + float(sync_per_ctrl_ms)
            model.addConstr(T_ms[s] - candidate <= big_m_rt * (1 - y[s, c]))
            model.addConstr(candidate - T_ms[s] <= big_m_rt * (1 - y[s, c]))

    mean_T_ms = model.addVar(lb=0.0, name="mean_T_ms")
    model.addConstr(
        gp.quicksum(T_ms[s] for s in switches)
        == float(max(1, len(switches))) * mean_T_ms,
        name="mean_rt_link",
    )
    delta_mean_rt_pos = model.addVar(lb=0.0, name="delta_mean_rt_pos_ms")
    model.addConstr(
        delta_mean_rt_pos >= mean_T_ms - float(init_mean_rt_ms or 0.0),
        name="delta_mean_rt_pos_def",
    )

    migration_cost_expr = (
          float(w_mig)        * num_migrations
        + float(w_rt)         * delta_mean_rt_pos
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

    # ---------- Proactive single-controller-failure recovery ----------
    # ABLATION: this recovery plan is tied to x0 (the pre-migration assignment).
    # It is intentionally used to measure the consequence of replacing y by x0.
    # Keep the shortest-path load-balancing model above unchanged and add the
    # same recovery decisions used by MCF-ARC.
    backup_pairs = [
        (s, failed, target)
        for s in switches
        for failed in controllers
        for target in controllers
        if target != failed
    ]
    bkp = model.addVars(backup_pairs, vtype=GRB.BINARY, name="backup")
    residual_pairs = [(s, failed) for s in switches for failed in controllers]
    residual = model.addVars(residual_pairs, vtype=GRB.BINARY, name="residual_backup")

    residual_candidates = [int(v) for v in G.nodes() if v not in controllers]
    rctrl_pairs = [(failed, v) for failed in controllers for v in residual_candidates]
    rctrl = model.addVars(rctrl_pairs, vtype=GRB.BINARY, name="residual_controller")
    residual_location_pairs = [
        (s, failed, v)
        for s in switches
        for failed in controllers
        for v in residual_candidates
    ]
    residual_at = model.addVars(
        residual_location_pairs, vtype=GRB.BINARY, name="residual_at"
    )

    # Experimental x0-coupled recovery variant.
    # Recovery is planned for the INITIAL assignment x0, not optimized assignment y.
    initial_load_const = {
        c: sum(float(loads[s]) * x0[(s, c)] for s in switches)
        for c in controllers
    }

    for s in switches:
        for failed in controllers:
            model.addConstr(
                gp.quicksum(bkp[s, failed, target] for target in controllers if target != failed)
                + residual[s, failed]
                == x0[(s, failed)],
                name=f"backup_or_residual_{s}_{failed}",
            )

            if residual_candidates:
                model.addConstr(
                    gp.quicksum(residual_at[s, failed, v] for v in residual_candidates)
                    == residual[s, failed],
                    name=f"locate_residual_{s}_{failed}",
                )
                for v in residual_candidates:
                    model.addConstr(
                        residual_at[s, failed, v] <= rctrl[failed, v],
                        name=f"residual_uses_controller_{s}_{failed}_{v}",
                    )
            else:
                model.addConstr(residual[s, failed] == 0, name=f"no_residual_site_{s}_{failed}")

    for failed in controllers:
        model.addConstr(
            gp.quicksum(rctrl[failed, v] for v in residual_candidates) <= 1,
            name=f"one_residual_controller_{failed}",
        )
        for target in controllers:
            if target == failed:
                continue
            recovered = gp.quicksum(
                float(loads[s]) * bkp[s, failed, target] for s in switches
            )
            model.addConstr(
                initial_load_const[target] + recovered
                <= GLOBAL_THRESHOLD * float(capacities[target]),
                name=f"backup_cap_fail_{failed}_to_{target}",
            )

    residual_load = {
        failed: gp.quicksum(float(loads[s]) * residual[s, failed] for s in switches)
        for failed in controllers
    }
    postfail_lmax = {}
    postfail_lmin = {}
    big_load = sum(float(loads[s]) for s in switches)
    for failed in controllers:
        postfail_lmax[failed] = model.addVar(lb=0.0, name=f"postfail_Lmax_{failed}")
        postfail_lmin[failed] = model.addVar(lb=0.0, name=f"postfail_Lmin_{failed}")
        for target in controllers:
            if target == failed:
                continue
            recovered = gp.quicksum(
                float(loads[s]) * bkp[s, failed, target] for s in switches
            )
            post_load = initial_load_const[target] + recovered
            model.addConstr(post_load <= postfail_lmax[failed])
            model.addConstr(post_load >= postfail_lmin[failed])
        if residual_candidates:
            opened = gp.quicksum(rctrl[failed, v] for v in residual_candidates)
            model.addConstr(residual_load[failed] <= postfail_lmax[failed] + big_load * (1 - opened))
            model.addConstr(residual_load[failed] >= postfail_lmin[failed] - big_load * (1 - opened))

    backup_load_cost = gp.quicksum(
        float(loads[s]) * bkp[s, failed, target]
        for s, failed, target in backup_pairs
    )
    residual_cost = (
        1e6 * gp.quicksum(residual_load[failed] for failed in controllers)
        + 1e5 * gp.quicksum(rctrl[failed, v] for failed, v in rctrl_pairs)
    )
    backup_distance_cost = gp.quicksum(
        float(dij.get((s, target), 0.0)) * bkp[s, failed, target]
        for s, failed, target in backup_pairs
    )
    residual_distance_cost = gp.LinExpr()
    graph_for_distance = G.to_undirected() if G.is_directed() else G
    for s, failed, v in residual_location_pairs:
        try:
            distance = float(nx.shortest_path_length(graph_for_distance, s, v, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distance = big_load
        residual_distance_cost += distance * residual_at[s, failed, v]

    post_failure_balance = gp.quicksum(
        postfail_lmax[failed] - postfail_lmin[failed] for failed in controllers
    )
    resiliency_obj = (
        backup_load_cost + residual_cost + backup_distance_cost
        + residual_distance_cost + post_failure_balance
    )

    # Combine base load balancing, migration cost, and resilience.
    a = float(alpha)
    b = float(beta)
    g = float(gamma_res)
    weight_sum = max(0.0, a) + max(0.0, b) + max(0.0, g)
    if weight_sum <= 0.0:
        a, b, g = 1.0, 0.0, 0.0
    else:
        a, b, g = max(0.0, a) / weight_sum, max(0.0, b) / weight_sum, max(0.0, g) / weight_sum

    # Final multi-objective
    model.setObjective(a * base_obj + b * migration_cost_expr + g * resiliency_obj, GRB.MINIMIZE)


    # ---------- Solve ----------
    total_load = sum(float(loads[s]) for s in switches)
    total_cap  = sum(float(cap_lim[c]) for c in controllers)
    if total_load > total_cap:
        print(f"⚠️ Warning: Load {total_load:.2f} exceeds total usable capacity {total_cap:.2f}")
        print("👉 Proceeding anyway to allow IIS if infeasible...")


    # ============================================================
    # SOLVE + STATUS HANDLING (SHORTEST PATH OPTIMIZER)
    # ============================================================

    solve_started = time.perf_counter()
    model.optimize()
    solve_time_sec = time.perf_counter() - solve_started

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
                f"{topology_name or 'topo'}_sp_x0_recovery_optimizer.iis"
            )
            model.write(iis_path)

            print("IIS written to:", iis_path)
            status_msg = f"INFEASIBLE_SP_X0_RECOVERY (IIS:{iis_path})"

        except Exception as e:
            print("IIS failed:", e)
            status_msg = "INFEASIBLE_SP_X0_RECOVERY (IIS_FAILED)"

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
            "INF_OR_UNBOUNDED_SP_X0_RECOVERY"
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
            "TIME_LIMIT_NO_SOLUTION_SP_X0_RECOVERY"
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
            f"NO_FEASIBLE_SOLUTION_STATUS_{model.Status}_SP_X0_RECOVERY"
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
    selected_residual_controller = {
        int(failed): int(v)
        for (failed, v), variable in rctrl.items()
        if variable.X > 0.5
    }

    output_root = plot_save_dir or os.path.join(RESULTS_FOLDER, "SHORTEST_X0_RESILIENT")
    for failed in controllers:
        failed = int(failed)
        recovery_assignment = dict(init_assign)
        residual_by_switch = {}
        orphan_switches = [
            int(s) for s in switches if int(init_assign.get(s, -1)) == failed
        ]
        residual_controller = selected_residual_controller.get(failed)

        for s in orphan_switches:
            assigned = False
            for target in controllers:
                if int(target) == failed:
                    continue
                if bkp[s, failed, target].X > 0.5:
                    recovery_assignment[s] = int(target)
                    assigned = True
                    break
            if not assigned and residual[s, failed].X > 0.5:
                if residual_controller is not None:
                    recovery_assignment[s] = residual_controller
                else:
                    residual_by_switch[s] = {
                        "residual_fraction": 1.0,
                        "residual_load": float(loads[s]),
                    }

        backup_capacity = None
        if residual_controller is not None:
            backup_capacity = 1.20 * sum(
                float(loads[s]) for s in orphan_switches
                if residual[s, failed].X > 0.5
            )

        process_failure_scenario(
            algorithm="SHORTEST_X0_RESILIENT",
            topology_name=topology_name or "topology",
            run_index=run_index,
            failed_controller=failed,
            G=G,
            pos=plot_pos,
            switches=switches,
            controllers=controllers,
            loads=loads,
            capacities=capacities,
            usable_threshold=float(GLOBAL_THRESHOLD),
            overload_threshold=float(OVERLOAD_THRESHOLD),
            initial_assignment=init_assign,
            recovery_assignment=recovery_assignment,
            fractional_assignment={},
            residual_by_switch=residual_by_switch,
            status=status_msg,
            solve_time_sec=solve_time_sec,
            objective_value=obj_val,
            mip_gap=mip_gap_solved,
            backup_controller=residual_controller,
            backup_capacity=backup_capacity,
            output_root=output_root,
            comparison_csv_file=comparison_csv_file,
            make_plot=plot_recovery,
            file_tag=f"SHORTEST_X0_RESILIENT_run{run_index:03d}_failC{failed}",
            plot_left_title="Initial Association",
            master_seed=master_seed,
            switch_seed=switch_seed,
            run_number=run_number,
            reassignment_scope="orphan_only",
            sync_delay_ms=sync_per_ctrl_ms,
            edge_caps=edge_caps_e,
            msg_bits=msg_bits,
            link_utilization_threshold=0.90,
        )

        # X0 recovery is planned from the initial assignment, so the common
        # handler above produces Initial vs Post-Recovery.  Also retain the
        # optimized final assignment as a separate comparison, as requested.
        if plot_recovery and plot_pos is not None:
            recovery_loads = {
                int(c): 0.0
                for c in set(controllers) | set(recovery_assignment.values())
                if int(c) != failed
            }
            for s, c in recovery_assignment.items():
                if int(c) != failed:
                    recovery_loads[int(c)] = (
                        recovery_loads.get(int(c), 0.0) + float(loads[s])
                    )

            plot_controllers = [int(c) for c in controllers]
            plot_capacities = {int(c): float(v) for c, v in capacities.items()}
            if residual_controller is not None:
                if residual_controller not in plot_controllers:
                    plot_controllers.append(residual_controller)
                if backup_capacity is not None:
                    plot_capacities[residual_controller] = float(backup_capacity)

            plot_final_vs_recovery_assignment(
                G=G,
                pos=plot_pos,
                switches=switches,
                controllers=plot_controllers,
                final_assign=final_assign,
                recovery_assign=recovery_assignment,
                loads=loads,
                final_loads=final_loads,
                recovery_loads=recovery_loads,
                topology_name=topology_name or "topology",
                save_dir=os.path.join(output_root, "failure_plots"),
                controller_capacity=plot_capacities,
                failed_controller=failed,
                backup_controller=residual_controller,
                backup_capacity=backup_capacity,
                migration_count=sum(
                    1 for s in switches
                    if final_assign.get(s) != recovery_assignment.get(s)
                ),
                capacity_threshold=float(OVERLOAD_THRESHOLD),
                file_tag=(
                    f"SHORTEST_X0_RESILIENT_run{run_index:03d}"
                    "_final_vs_recovery"
                ),
                left_panel_title="Final Association",
            )

    return final_assign, final_loads, paths_sc,obj_val, migration_count, mip_gap_solved,status_msg
