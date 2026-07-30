# Switch_migration_optimizer_sequential.py
#
# Two-stage sequential counterpart of the joint shortest-path resilient model.
#
# Stage 1: optimize normal-state load balancing and migration cost only.
# Stage 2: fix the Stage-1 assignment and optimize proactive recovery plans.
#
# This model is intended for a controlled comparison with the joint model:
#   - Same normal-state variables/constraints
#   - Same resilience variables/constraints/objective
#   - Only the optimization structure changes from joint to sequential

import os
import time
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

from helpers import (
    RESULTS_FOLDER,
    CAPACITY_THRESHOLD as GLOBAL_THRESHOLD,
    OVERLOAD_THRESHOLD,
    TIME_LIMIT,
)
from rt_metrics import build_paths_sc_from_switch_paths, path_latency_ms
from failure_scenario_handler import process_failure_scenario

os.makedirs(RESULTS_FOLDER, exist_ok=True)


def _safe_mip_gap(model):
    return float(model.MIPGap) if model.SolCount > 0 else None


def _status_message(model, stage_name):
    if model.Status == GRB.OPTIMAL:
        return f"OPTIMAL_{stage_name}"
    if model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        return f"TIME_LIMIT_FEASIBLE_{stage_name}"
    if model.SolCount > 0:
        return f"FEASIBLE_STATUS_{model.Status}_{stage_name}"
    return f"NO_SOLUTION_STATUS_{model.Status}_{stage_name}"


def _handle_infeasible(model, topology_name, suffix):
    iis_dir = os.path.join(RESULTS_FOLDER, "iis_reports")
    os.makedirs(iis_dir, exist_ok=True)
    try:
        model.setParam(GRB.Param.Presolve, 0)
        model.setParam(GRB.Param.IISMethod, 1)
        model.computeIIS()
        path = os.path.join(
            iis_dir,
            f"{topology_name or 'topo'}_{suffix}.iis",
        )
        model.write(path)
        return f"INFEASIBLE_{suffix.upper()} (IIS:{path})"
    except Exception:
        return f"INFEASIBLE_{suffix.upper()} (IIS_FAILED)"


def run_migration_optimizer_sequential(
    G,
    switches,
    controllers,
    dij,
    init_assign,
    loads,
    capacities,
    topology_name=None,
    *,
    objective_type,
    Dcc: dict | None = None,
    sync_per_ctrl_ms: float = 0.0,
    w_mig: float = 1.0,
    w_rt: float = 1.0,
    w_cc: float = 1.0,
    w_steiner: float = 1.0,
    edge_caps_e: dict | None = None,
    msg_bits: int = 128,
    cost_mode: str = "weight",
    alpha: float,
    beta: float,
    gamma_res: float = 1.0,  # retained for call compatibility; Stage 2 optimizes resilience alone
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
    Sequential shortest-path resilient optimization.

    Stage 1:
        Optimize y and z using only normal-state load balancing and
        migration-cost terms.

    Stage 2:
        Fix y to the Stage-1 assignment and optimize bkp, residual,
        residual-controller placement, recovery distance, and
        post-failure balance.

    Return signature matches the joint shortest-path optimizer:
        final_assign, final_loads, paths_sc, objective_value,
        migration_count, mip_gap, status_msg
    """

    switches = list(switches)
    controllers = list(controllers)

    # ============================================================
    # STAGE 1: NORMAL-STATE LOAD BALANCING + MIGRATION
    # ============================================================
    lb_model = gp.Model("shortest_sequential_stage1_load_balancing")
    lb_model.setParam("OutputFlag", 0)
    lb_model.setParam("TimeLimit", float(TIME_LIMIT))

    y = lb_model.addVars(switches, controllers, vtype=GRB.BINARY, name="assign")
    z = lb_model.addVars(switches, vtype=GRB.BINARY, name="migrated")

    cap_lim = {
        c: float(OVERLOAD_THRESHOLD) * float(capacities[c])
        for c in controllers
    }

    for s in switches:
        lb_model.addConstr(
            gp.quicksum(y[s, c] for c in controllers) == 1,
            name=f"assign_{s}",
        )

    load_expr = {
        c: gp.quicksum(float(loads[s]) * y[s, c] for s in switches)
        for c in controllers
    }
    for c in controllers:
        lb_model.addConstr(
            load_expr[c] <= cap_lim[c],
            name=f"capacity_{c}",
        )

    for s in switches:
        c0 = init_assign.get(s)
        if c0 in controllers:
            lb_model.addConstr(
                z[s] == gp.quicksum(
                    y[s, c] for c in controllers if c != c0
                ),
                name=f"z_equals_migrate_{s}",
            )
        else:
            lb_model.addConstr(
                z[s] == gp.quicksum(y[s, c] for c in controllers),
                name=f"z_equals_first_assign_{s}",
            )

    for c in controllers:
        if c in switches:
            lb_model.addConstr(
                y[c, c] == 1,
                name=f"controller_serves_itself_{c}",
            )

    dcc_pair = {}
    for s in switches:
        c0 = init_assign.get(s)
        for c in controllers:
            if c0 in controllers and c != c0 and Dcc is not None:
                if isinstance(Dcc.get(c0, {}), dict):
                    dcc_pair[(s, c)] = float(
                        Dcc.get(c0, {}).get(c, 0.0)
                    )
                else:
                    dcc_pair[(s, c)] = float(
                        Dcc.get((c0, c), 0.0)
                    )
            else:
                dcc_pair[(s, c)] = 0.0

    num_migrations = gp.quicksum(z[s] for s in switches)
    cc_transfer = gp.quicksum(
        dcc_pair[(s, c)] * y[s, c]
        for s in switches
        for c in controllers
    )
    steiner_bcast = float(sync_per_ctrl_ms) * num_migrations

    rho_max = 0.95
    pwl_segments = 20
    lam_rt = lb_model.addVars(controllers, lb=0.0, name="lambda_rt")
    W_ms = lb_model.addVars(controllers, lb=0.0, name="W_ms")
    for c in controllers:
        mu_eff = float(GLOBAL_THRESHOLD) * float(capacities[c])
        xs = [i * (rho_max * mu_eff / pwl_segments) for i in range(pwl_segments + 1)]
        ys = [1000.0 / max(1e-9, mu_eff - value) for value in xs]
        lb_model.addConstr(lam_rt[c] == load_expr[c], name=f"lambda_rt_link_{c}")
        lb_model.addConstr(lam_rt[c] <= rho_max * mu_eff, name=f"rt_domain_{c}")
        lb_model.addGenConstrPWL(lam_rt[c], W_ms[c], xs, ys, name=f"W_pwl_ms_{c}")

    propagation_ms = {}
    for s in switches:
        for c in controllers:
            try:
                path = nx.shortest_path(G, source=s, target=c, weight="weight")
                propagation_ms[s, c] = 2.0 * path_latency_ms(G, path)
            except nx.NetworkXNoPath:
                propagation_ms[s, c] = 1e9

    T_ms = lb_model.addVars(switches, lb=0.0, name="T_ms")
    big_m_rt = 1e9
    for s in switches:
        for c in controllers:
            candidate = propagation_ms[s, c] + W_ms[c] + float(sync_per_ctrl_ms)
            lb_model.addConstr(T_ms[s] - candidate <= big_m_rt * (1 - y[s, c]))
            lb_model.addConstr(candidate - T_ms[s] <= big_m_rt * (1 - y[s, c]))

    mean_T_ms = lb_model.addVar(lb=0.0, name="mean_T_ms")
    lb_model.addConstr(
        gp.quicksum(T_ms[s] for s in switches)
        == float(max(1, len(switches))) * mean_T_ms,
        name="mean_rt_link",
    )
    delta_mean_rt_pos = lb_model.addVar(lb=0.0, name="delta_mean_rt_pos_ms")
    lb_model.addConstr(
        delta_mean_rt_pos >= mean_T_ms - float(init_mean_rt_ms or 0.0),
        name="delta_mean_rt_pos_def",
    )

    migration_cost_expr = (
        float(w_mig) * num_migrations
        + float(w_rt) * delta_mean_rt_pos
        + float(w_cc) * cc_transfer
        + float(w_steiner) * steiner_bcast
    )

    base_obj = 0.0

    if objective_type == "min_max_util":
        umax = lb_model.addVar(lb=0.0, name="Umax")
        for c in controllers:
            lb_model.addConstr(
                load_expr[c] <= umax * float(capacities[c]),
                name=f"util_cap_{c}",
            )
        base_obj = umax

    elif objective_type == "min_sum_util":
        base_obj = gp.quicksum(
            load_expr[c] / float(capacities[c])
            for c in controllers
        )

    elif objective_type == "variance":
        total = sum(float(loads[s]) for s in switches)
        mean = total / max(1, len(controllers))
        base_obj = gp.quicksum(
            (load_expr[c] - mean) * (load_expr[c] - mean)
            for c in controllers
        )

    elif objective_type == "min_dev":
        total = sum(float(loads[s]) for s in switches)
        mean = total / max(1, len(controllers))
        dev = {
            c: lb_model.addVar(lb=0.0, name=f"abs_dev_{c}")
            for c in controllers
        }
        for c in controllers:
            lb_model.addConstr(load_expr[c] - mean <= dev[c])
            lb_model.addConstr(mean - load_expr[c] <= dev[c])
        base_obj = gp.quicksum(dev[c] for c in controllers)

    elif objective_type == "maxmin":
        lmax = lb_model.addVar(lb=0.0, name="L_max")
        lmin = lb_model.addVar(lb=0.0, name="L_min")
        for c in controllers:
            lb_model.addConstr(load_expr[c] <= lmax)
            lb_model.addConstr(load_expr[c] >= lmin)
        base_obj = lmax - lmin

    elif objective_type == "migration_cost":
        base_obj = 0.0

    else:
        raise ValueError(f"Unknown objective_type: {objective_type}")

    # Normalize only alpha and beta in Stage 1.
    a = max(0.0, float(alpha))
    b = max(0.0, float(beta))
    denom = a + b
    if denom <= 0.0:
        a, b = 1.0, 0.0
    else:
        a, b = a / denom, b / denom

    lb_model.setObjective(
        a * base_obj + b * migration_cost_expr,
        GRB.MINIMIZE,
    )

    total_load = sum(float(loads[s]) for s in switches)
    total_usable_capacity = sum(cap_lim.values())
    if total_load > total_usable_capacity:
        print(
            f"Warning: load {total_load:.2f} exceeds Stage-1 usable "
            f"capacity {total_usable_capacity:.2f}"
        )

    stage1_start = time.perf_counter()
    lb_model.optimize()
    stage1_time = time.perf_counter() - stage1_start

    if lb_model.Status == GRB.INFEASIBLE:
        status = _handle_infeasible(
            lb_model,
            topology_name,
            "shortest_sequential_stage1",
        )
        return {}, {}, {}, None, 0, None, status

    if lb_model.Status == GRB.INF_OR_UNBD:
        return {}, {}, {}, None, 0, None, "INF_OR_UNBOUNDED_STAGE1"

    if lb_model.SolCount == 0:
        return (
            {},
            {},
            {},
            None,
            0,
            None,
            _status_message(lb_model, "STAGE1"),
        )

    final_assign = {}
    for s in switches:
        for c in controllers:
            if y[s, c].X > 0.5:
                final_assign[s] = c
                break

    final_loads = {
        c: sum(
            float(loads[s])
            for s in switches
            if final_assign.get(s) == c
        )
        for c in controllers
    }

    migration_count = sum(
        1
        for s in switches
        if final_assign.get(s) != init_assign.get(s)
    )

    paths = {}
    for s in switches:
        c = final_assign[s]
        try:
            paths[s] = nx.shortest_path(
                G,
                source=s,
                target=c,
                weight="weight",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths[s] = None

    paths_sc = build_paths_sc_from_switch_paths(final_assign, paths)

    stage1_obj = float(lb_model.ObjVal)
    stage1_gap = _safe_mip_gap(lb_model)
    stage1_status = _status_message(lb_model, "STAGE1")

    # ============================================================
    # STAGE 2: RECOVERY OPTIMIZATION WITH y FIXED
    # ============================================================
    rec_model = gp.Model("shortest_sequential_stage2_recovery")
    rec_model.setParam("OutputFlag", 0)
    rec_model.setParam("TimeLimit", float(TIME_LIMIT))

    # Fixed assignment indicator derived from Stage 1.
    y_fixed = {
        (s, c): int(final_assign.get(s) == c)
        for s in switches
        for c in controllers
    }

    fixed_load = {
        c: sum(
            float(loads[s]) * y_fixed[(s, c)]
            for s in switches
        )
        for c in controllers
    }

    backup_pairs = [
        (s, failed, target)
        for s in switches
        for failed in controllers
        for target in controllers
        if target != failed
    ]
    bkp = rec_model.addVars(
        backup_pairs,
        vtype=GRB.BINARY,
        name="backup",
    )

    residual_pairs = [
        (s, failed)
        for s in switches
        for failed in controllers
    ]
    residual = rec_model.addVars(
        residual_pairs,
        vtype=GRB.BINARY,
        name="residual_backup",
    )

    residual_candidates = [
        int(v)
        for v in G.nodes()
        if v not in controllers
    ]
    rctrl_pairs = [
        (failed, v)
        for failed in controllers
        for v in residual_candidates
    ]
    rctrl = rec_model.addVars(
        rctrl_pairs,
        vtype=GRB.BINARY,
        name="residual_controller",
    )

    residual_location_pairs = [
        (s, failed, v)
        for s in switches
        for failed in controllers
        for v in residual_candidates
    ]
    residual_at = rec_model.addVars(
        residual_location_pairs,
        vtype=GRB.BINARY,
        name="residual_at",
    )

    # Same backup-or-residual constraint as the joint model, but RHS is fixed.
    for s in switches:
        for failed in controllers:
            rec_model.addConstr(
                gp.quicksum(
                    bkp[s, failed, target]
                    for target in controllers
                    if target != failed
                )
                + residual[s, failed]
                == y_fixed[(s, failed)],
                name=f"backup_or_residual_{s}_{failed}",
            )

            if residual_candidates:
                rec_model.addConstr(
                    gp.quicksum(
                        residual_at[s, failed, v]
                        for v in residual_candidates
                    )
                    == residual[s, failed],
                    name=f"locate_residual_{s}_{failed}",
                )
                for v in residual_candidates:
                    rec_model.addConstr(
                        residual_at[s, failed, v]
                        <= rctrl[failed, v],
                        name=f"residual_uses_controller_{s}_{failed}_{v}",
                    )
            else:
                rec_model.addConstr(
                    residual[s, failed] == 0,
                    name=f"no_residual_site_{s}_{failed}",
                )

    # Same survivor capacity constraints as the joint model.
    for failed in controllers:
        if residual_candidates:
            rec_model.addConstr(
                gp.quicksum(
                    rctrl[failed, v]
                    for v in residual_candidates
                )
                <= 1,
                name=f"one_residual_controller_{failed}",
            )

        for target in controllers:
            if target == failed:
                continue

            recovered = gp.quicksum(
                float(loads[s]) * bkp[s, failed, target]
                for s in switches
            )

            rec_model.addConstr(
                fixed_load[target] + recovered
                <= float(GLOBAL_THRESHOLD) * float(capacities[target]),
                name=f"backup_cap_fail_{failed}_to_{target}",
            )

    residual_load = {
        failed: gp.quicksum(
            float(loads[s]) * residual[s, failed]
            for s in switches
        )
        for failed in controllers
    }

    postfail_lmax = {}
    postfail_lmin = {}
    big_load = sum(float(loads[s]) for s in switches)

    for failed in controllers:
        postfail_lmax[failed] = rec_model.addVar(
            lb=0.0,
            name=f"postfail_Lmax_{failed}",
        )
        postfail_lmin[failed] = rec_model.addVar(
            lb=0.0,
            name=f"postfail_Lmin_{failed}",
        )

        for target in controllers:
            if target == failed:
                continue

            recovered = gp.quicksum(
                float(loads[s]) * bkp[s, failed, target]
                for s in switches
            )
            post_load = fixed_load[target] + recovered

            rec_model.addConstr(
                post_load <= postfail_lmax[failed],
                name=f"post_lmax_{failed}_{target}",
            )
            rec_model.addConstr(
                post_load >= postfail_lmin[failed],
                name=f"post_lmin_{failed}_{target}",
            )

        if residual_candidates:
            opened = gp.quicksum(
                rctrl[failed, v]
                for v in residual_candidates
            )
            rec_model.addConstr(
                residual_load[failed]
                <= postfail_lmax[failed] + big_load * (1 - opened),
                name=f"residual_lmax_{failed}",
            )
            rec_model.addConstr(
                residual_load[failed]
                >= postfail_lmin[failed] - big_load * (1 - opened),
                name=f"residual_lmin_{failed}",
            )

    backup_load_cost = gp.quicksum(
        float(loads[s]) * bkp[s, failed, target]
        for s, failed, target in backup_pairs
    )

    residual_cost = (
        1e6
        * gp.quicksum(
            residual_load[failed]
            for failed in controllers
        )
        + 1e5
        * gp.quicksum(
            rctrl[failed, v]
            for failed, v in rctrl_pairs
        )
    )

    backup_distance_cost = gp.quicksum(
        float(dij.get((s, target), 0.0))
        * bkp[s, failed, target]
        for s, failed, target in backup_pairs
    )

    residual_distance_cost = gp.LinExpr()
    graph_for_distance = (
        G.to_undirected()
        if G.is_directed()
        else G
    )

    for s, failed, v in residual_location_pairs:
        try:
            distance = float(
                nx.shortest_path_length(
                    graph_for_distance,
                    s,
                    v,
                    weight="weight",
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distance = big_load

        residual_distance_cost += (
            distance * residual_at[s, failed, v]
        )

    post_failure_balance = gp.quicksum(
        postfail_lmax[failed] - postfail_lmin[failed]
        for failed in controllers
    )

    resiliency_obj = (
        backup_load_cost
        + residual_cost
        + backup_distance_cost
        + residual_distance_cost
        + post_failure_balance
    )

    rec_model.setObjective(resiliency_obj, GRB.MINIMIZE)

    stage2_start = time.perf_counter()
    rec_model.optimize()
    stage2_time = time.perf_counter() - stage2_start

    if rec_model.Status == GRB.INFEASIBLE:
        status = _handle_infeasible(
            rec_model,
            topology_name,
            "shortest_sequential_stage2",
        )
        return (
            final_assign,
            final_loads,
            paths_sc,
            stage1_obj,
            migration_count,
            stage1_gap,
            f"{stage1_status};{status}",
        )

    if rec_model.Status == GRB.INF_OR_UNBD:
        return (
            final_assign,
            final_loads,
            paths_sc,
            stage1_obj,
            migration_count,
            stage1_gap,
            f"{stage1_status};INF_OR_UNBOUNDED_STAGE2",
        )

    if rec_model.SolCount == 0:
        return (
            final_assign,
            final_loads,
            paths_sc,
            stage1_obj,
            migration_count,
            stage1_gap,
            f"{stage1_status};{_status_message(rec_model, 'STAGE2')}",
        )

    stage2_obj = float(rec_model.ObjVal)
    stage2_gap = _safe_mip_gap(rec_model)
    total_solve_time = stage1_time + stage2_time

    selected_residual_controller = {
        int(failed): int(v)
        for (failed, v), variable in rctrl.items()
        if variable.X > 0.5
    }

    output_root = (
        plot_save_dir
        or os.path.join(
            RESULTS_FOLDER,
            "SHORTEST_SEQUENTIAL_RESILIENT",
        )
    )

    for failed in controllers:
        failed = int(failed)
        recovery_assignment = dict(final_assign)
        residual_by_switch = {}

        orphan_switches = [
            int(s)
            for s in switches
            if int(final_assign.get(s, -1)) == failed
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
                float(loads[s])
                for s in orphan_switches
                if residual[s, failed].X > 0.5
            )

        process_failure_scenario(
            algorithm="SHORTEST_SEQUENTIAL_RESILIENT",
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
            initial_assignment=final_assign,
            recovery_assignment=recovery_assignment,
            fractional_assignment={},
            residual_by_switch=residual_by_switch,
            status=(
                f"{stage1_status};"
                f"{_status_message(rec_model, 'STAGE2')}"
            ),
            solve_time_sec=total_solve_time,
            objective_value=stage2_obj,
            mip_gap=stage2_gap,
            backup_controller=residual_controller,
            backup_capacity=backup_capacity,
            output_root=output_root,
            comparison_csv_file=comparison_csv_file,
            make_plot=plot_recovery,
            file_tag=(
                f"SHORTEST_SEQUENTIAL_RESILIENT_"
                f"run{run_index:03d}_failC{failed}"
            ),
            master_seed=master_seed,
            switch_seed=switch_seed,
            run_number=run_number,
            reassignment_scope="orphan_only",
            sync_delay_ms=sync_per_ctrl_ms,
            edge_caps=edge_caps_e,
            msg_bits=msg_bits,
            link_utilization_threshold=0.90,
        )

    # For compatibility, return Stage-2 recovery objective as obj_val.
    # The two stages have different objectives, so do not sum their values.
    combined_status = (
        f"{stage1_status};"
        f"{_status_message(rec_model, 'STAGE2')};"
        f"STAGE1_TIME={stage1_time:.6f};"
        f"STAGE2_TIME={stage2_time:.6f}"
    )

    return (
        final_assign,
        final_loads,
        paths_sc,
        stage2_obj,
        migration_count,
        stage2_gap,
        combined_status,
    )
