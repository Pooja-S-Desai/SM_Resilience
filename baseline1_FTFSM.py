
from __future__ import annotations

import os
import time
import json
from collections import defaultdict
from typing import Dict, List, Optional
from failure_scenario_handler import process_failure_scenario
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

from plotting import plot_final_vs_recovery_assignment


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _build_shortest_paths(G, final_assign, cost_mode="weight"):
    paths_pair = {}
    paths_sw = {}

    UG = G.to_undirected()
    weight = None if cost_mode == "hops" else "weight"

    for s, c in final_assign.items():
        if s == c:
            p = [s]
        else:
            try:
                p = nx.shortest_path(UG, s, c, weight=weight)
            except Exception:
                p = []
        paths_pair[(s, c)] = p
        paths_sw[s] = p

    return paths_pair, paths_sw


def _loads_by_controller(assign, loads, controllers=None):
    out = defaultdict(float)
    if controllers is not None:
        for c in controllers:
            out[int(c)] = 0.0

    for s, c in assign.items():
        out[int(c)] += float(loads.get(int(s), 0.0))

    return dict(out)


def run_baseline1_FTFSM(
    G,
    switches: List[int],
    controllers: List[int],
    init_assign: Dict[int, int],
    loads: Dict[int, float],
    capacities: Dict[int, float],
    dij: Optional[dict] = None,
    Dcc: Optional[dict] = None,
    *,
    usable_threshold: float = 0.95,
    overload_threshold: float = 0.90,
    rule_install_cost: float = 0.0,
    epsilon: float = 1.0,
    omega: Optional[List[int]] = None,
    time_limit: float = 300.0,
    verbose: bool = False,
    cost_mode: str = "weight",
    topology_name: str | None = None,
    run_index: int = 0,
    plot_recovery: bool = False,
    plot_pos: dict | None = None,
    plot_save_dir: str | None = None,
    plot_file_tag: str | None = None,
    master_seed: int | None = None,
    switch_seed: int | None = None,
    run_number: int | None = None,
    comparison_csv_file: str | None = None,
    edge_caps: dict | None = None,
    msg_bits: float = 128.0,
):
    switches = list(map(int, switches))
    controllers = list(map(int, controllers))
    init_assign = {int(s): int(c) for s, c in dict(init_assign).items()}
    loads = {int(s): float(v) for s, v in dict(loads).items()}
    capacities = {int(c): float(v) for c, v in dict(capacities).items()}

    missing_assignments = [i for i in switches if i not in init_assign]
    if missing_assignments:
        preview = ", ".join(map(str, missing_assignments[:10]))
        suffix = "..." if len(missing_assignments) > 10 else ""
        raise ValueError(
            "FTFSM requires one initial controller assignment per switch; "
            f"missing {len(missing_assignments)} switch(es): {preview}{suffix}"
        )

    invalid_controllers = {
        i: init_assign[i]
        for i in switches
        if init_assign[i] not in controllers
    }
    if invalid_controllers:
        raise ValueError(
            "FTFSM initial assignments reference controllers outside the "
            f"controller set: {invalid_controllers}"
        )


    if omega is None:
        omega = list(controllers)
    else:
        omega = list(map(int, omega))

    normal_mode = (len(omega) == 0)
    if normal_mode:
        omega = [-1]   # dummy no-failure scenario
    else:
            omega = omega
    # ============================================================
    # MULTI-SCENARIO WRAPPER
    # Solve one failed-controller scenario at a time.
    # ============================================================
    if len(omega) > 1:

        all_results = {}
        first_success = None
        total_solve_time = 0.0

        for failed_c in omega:

            print(f"\n[FTFSM] Solving separate failure scenario: C{failed_c} fails")

            (
                fa_tmp,
                paths_tmp,
                fl_tmp,
                meta_tmp,
                obj_tmp,
                mip_tmp,
                status_tmp,
            ) = run_baseline1_FTFSM(
                G=G,
                switches=switches,
                controllers=controllers,
                init_assign=init_assign,
                loads=loads,
                capacities=capacities,
                dij=dij,
                Dcc=Dcc,
                usable_threshold=usable_threshold,
                overload_threshold=overload_threshold,
                rule_install_cost=rule_install_cost,
                epsilon=epsilon,
                omega=[failed_c],
                time_limit=time_limit,
                verbose=verbose,
                cost_mode=cost_mode,
                topology_name=topology_name,
                run_index=run_index,
                # Each single-failure solve uses the common fractional plotter.
                plot_recovery=plot_recovery,
                plot_pos=plot_pos,
                plot_save_dir=plot_save_dir,
                plot_file_tag=plot_file_tag,
                master_seed=master_seed,
                switch_seed=switch_seed,
                run_number=run_number,
                comparison_csv_file=comparison_csv_file,
                edge_caps=edge_caps,
                msg_bits=msg_bits,
            )

            total_solve_time += float((meta_tmp or {}).get("solve_time_sec", 0.0))

            all_results[int(failed_c)] = {
                "status": status_tmp,
                "final_assign": fa_tmp,
                "final_loads": fl_tmp,
                "meta": meta_tmp,
                "obj": obj_tmp,
                "mip_gap": mip_tmp,
            }

            if first_success is None and (
                status_tmp == "SUCCESS"
                or str(status_tmp).startswith("FEASIBLE_STATUS")
            ):
                first_success = (
                    fa_tmp,
                    paths_tmp,
                    fl_tmp,
                    meta_tmp,
                    obj_tmp,
                    mip_tmp,
                    status_tmp,
                )

        # ------------------------------------------------------------
        # Write combined scenario log once
        # ------------------------------------------------------------
        if plot_save_dir is not None:
            multi_dir = os.path.join(plot_save_dir, "FTFSM_all_failure_scenarios")
            os.makedirs(multi_dir, exist_ok=True)

            tag = f"_{plot_file_tag}" if plot_file_tag else ""
            out_file = os.path.join(
                multi_dir,
                f"{topology_name or 'topology'}{tag}_run{run_index:03d}_all_ftfsm_scenarios.json"
            )

            with open(out_file, "w") as fh:
                json.dump(_json_safe(all_results), fh, indent=2)

            print(f"[FTFSM ALL SCENARIOS] written → {out_file}")

        # Per-controller plots are produced by process_failure_scenario using
        # the fractional traffic representation. No dominant-controller plot
        # is generated here because it would hide traffic splitting.

        if first_success is not None:
            fa_ret, paths_ret, fl_ret, meta_ret, obj_ret, mip_ret, status_ret = first_success
            meta_ret["all_failure_scenarios"] = _json_safe(all_results)
            meta_ret["total_solve_time_all_scenarios"] = total_solve_time
            return fa_ret, paths_ret, fl_ret, meta_ret, obj_ret, mip_ret, status_ret

        meta_fail = {
            "status": "INFEASIBLE_ALL_FTFSM_SCENARIOS",
            "solve_time_sec": total_solve_time,
            "eta": None,
            "mip_gap": None,
            "num_migrations": None,
            "recovery_plan": {},
            "recovery_assignments": {},
            "recovery_loads_by_failure": {},
            "all_failure_scenarios": _json_safe(all_results),
        }

        return {}, {}, {}, meta_fail, None, None, "INFEASIBLE_ALL_FTFSM_SCENARIOS"

    # ============================================================
    # SINGLE FAILURE SCENARIO MODEL
    # Only reaches here when omega = [failed_controller]
    # ============================================================

    model = gp.Model("baseline1_FTFSM_exact")
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("TimeLimit", float(time_limit))

    p = {
        (i, j): float(loads.get(i, 0.0))
        for i in switches
        for j in controllers
    }

    alpha = {
        j: float(capacities[j]) * float(usable_threshold)
        for j in controllers
    }

    m_active = {
        (j, w): 1 if normal_mode else (0 if j == w else 1)
        for j in controllers
        for w in omega
    }

    BIG_M_LOAD = sum(float(loads.get(i, 0.0)) for i in switches) * 10.0 + 1.0

    # -----------------------------
    # Variables
    # -----------------------------
    x = model.addVars(switches, controllers, vtype=GRB.BINARY, name="x")

    b = model.addVars(
        switches,
        controllers,
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="b"
    )

    f = model.addVars(
        [
            (i, j, jp, w)
            for i in switches
            for j in controllers
            for jp in controllers
            for w in omega
            if jp != j
        ],
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="f"
    )

    z = model.addVars(
        [
            (i, j, jp, w)
            for i in switches
            for j in controllers
            for jp in controllers
            for w in omega
            if jp != j
        ],
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="z"
    )

    g = model.addVars(
        [
            (i, j, w)
            for i in switches
            for j in controllers
            for w in omega
        ],
        lb=0.0,
        ub=1.0,
        vtype=GRB.CONTINUOUS,
        name="g"
    )

    L = model.addVars(controllers, omega, lb=0.0, name="L")
    k = model.addVars(controllers, omega, lb=0.0, name="k")
    q = model.addVars(omega, vtype=GRB.BINARY, name="q")
    eta = model.addVar(lb=0.0, name="eta")

    # -----------------------------
    # Eq. (3b): x_i^j <= m_j^w
    # -----------------------------
    for i in switches:
        for j in controllers:
            for w in omega:
                model.addConstr(
                    x[i, j] <= m_active[j, w],
                    name=f"eq3b_x_active_{i}_{j}_{w}"
                )

    # -----------------------------
    # Eq. (3c): sum_j b_i^j = 1
    # -----------------------------
    for i in switches:
        model.addConstr(
            gp.quicksum(b[i, j] for j in controllers) == 1.0,
            name=f"eq3c_b_sum_{i}"
        )

    # -----------------------------
    # Eq. (3d): b_i^j <= x_i^j
    # -----------------------------
    for i in switches:
        for j in controllers:
            model.addConstr(
                b[i, j] <= x[i, j],
                name=f"eq3d_b_le_x_{i}_{j}"
            )

    # Global fractional recovery: after a controller failure, traffic from
    # every switch may be split among any surviving controllers.  This is the
    # flexibility represented by b_i^j; restricting surviving switches to
    # their original controller would turn FTFSM into orphan-only recovery
    # even though its reported reassignment scope is global.

    # -----------------------------
    # Eq. (3e): f_i^{j,j'}(w) <= m_{j'}^w.  The destination
    # controller must be active; the source may be the failed controller.
    # -----------------------------
    for i in switches:
        for j in controllers:
            for jp in controllers:
                if jp == j:
                    continue
                for w in omega:
                    model.addConstr(
                        f[i, j, jp, w] <= m_active[jp, w],
                        name=f"eq3e_f_active_{i}_{j}_{jp}_{w}"
                    )

    # -----------------------------
    # Eq. (4a): z = f AND x
    # -----------------------------
    for i in switches:
        for j in controllers:
            for jp in controllers:
                if jp == j:
                    continue
                for w in omega:
                    model.addConstr(
                        z[i, j, jp, w] <= f[i, j, jp, w],
                        name=f"eq4a_z_le_f_{i}_{j}_{jp}_{w}"
                    )
                    model.addConstr(
                        z[i, j, jp, w] <= x[i, j],
                        name=f"eq4a_z_le_x_{i}_{j}_{jp}_{w}"
                    )
                    model.addConstr(
                        z[i, j, jp, w] >= f[i, j, jp, w] + x[i, j] - 1.0,
                        name=f"eq4a_z_ge_fx_{i}_{j}_{jp}_{w}"
                    )

    # -----------------------------
    # Eq. (4c): g = b AND m
    # -----------------------------
    for i in switches:
        for j in controllers:
            for w in omega:
                model.addConstr(
                    g[i, j, w] <= b[i, j],
                    name=f"eq4c_g_le_b_{i}_{j}_{w}"
                )
                model.addConstr(
                    g[i, j, w] <= m_active[j, w],
                    name=f"eq4c_g_le_m_{i}_{j}_{w}"
                )
                model.addConstr(
                    g[i, j, w] >= b[i, j] + m_active[j, w] - 1.0,
                    name=f"eq4c_g_ge_bm_{i}_{j}_{w}"
                )

    # -----------------------------
    # Post-failure controller load.
    #
    # b_i^j is explicitly defined by the paper as the fraction of switch i's
    # traffic distributed to controller j.  Computing L from binary x (as the
    # previous implementation did) charges the full switch load to every
    # controller in the support and makes fractional solutions unattractive.
    # The load must therefore be the fraction-weighted traffic, including the
    # configured rule-installation overhead.
    # -----------------------------
    for j in controllers:
        for w in omega:

            model.addConstr(
                L[j, w] == gp.quicksum(
                    (1.0 + float(rule_install_cost))
                    * float(p[i, j])
                    * b[i, j]
                    for i in switches
                ),
                name=f"eq6c_L_{j}_{w}"
            )

    # -----------------------------
    # Eq. (4b): k = m AND L
    # -----------------------------
    for j in controllers:
        for w in omega:
            mjw = m_active[j, w]

            model.addConstr(
                k[j, w] <= L[j, w],
                name=f"eq4b_k_le_L_{j}_{w}"
            )
            model.addConstr(
                k[j, w] <= BIG_M_LOAD * mjw,
                name=f"eq4b_k_le_Mm_{j}_{w}"
            )
            model.addConstr(
                k[j, w] >= L[j, w] - BIG_M_LOAD * (1 - mjw),
                name=f"eq4b_k_ge_LminusM_{j}_{w}"
            )

    # -----------------------------
    # Eq. (6d): k_j(w) <= eta * alpha_j
    # -----------------------------
    for j in controllers:
        for w in omega:
            model.addConstr(
                k[j, w] <= eta * alpha[j],
                name=f"eq6d_eta_cap_{j}_{w}"
            )

    # -----------------------------
    # Eq. (6e): paper-exact linearized Eq. (3h)
    # -----------------------------
    for i in switches:
        for j in controllers:
            for w in omega:
                model.addConstr(
                    gp.quicksum(
                        f[i, jp, j, w]
                        for jp in controllers
                        if jp != j
                    )
                    == b[i, j] - g[i, j, w],
                    name=f"eq6e_recovery_flow_{i}_{j}_{w}"
                )
    # -----------------------------
    # Eq. (5a)-(5c): q linearization
    # -----------------------------
    for w in omega:
        for j in controllers:
            model.addConstr(
                q[w] >= 1 - m_active[j, w],
                name=f"eq5a_q_lb_{j}_{w}"
            )

        model.addConstr(
            q[w] <= gp.quicksum(
                1 - m_active[j, w]
                for j in controllers
            ),
            name=f"eq5b_q_ub_{w}"
        )

    # -----------------------------
    # Eq. (3i): epsilon survival constraint
    # -----------------------------
    for i in switches:
        for w in omega:
            model.addConstr(
                gp.quicksum(b[i, j] for j in controllers)
                >= float(epsilon) * q[w],
                name=f"eq3i_epsilon_{i}_{w}"
            )

    model.setObjective(eta, GRB.MINIMIZE)

    start = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - start

    # ============================================================
    # No solution handling
    # ============================================================
    if model.SolCount == 0:

        if model.Status == GRB.INFEASIBLE:
            status = "INFEASIBLE_FTFSM"
        elif model.Status == GRB.INF_OR_UNBD:
            status = "INF_OR_UNBD_FTFSM"
        elif model.Status == GRB.TIME_LIMIT:
            status = "TIME_LIMIT_NO_SOLUTION_FTFSM"
        else:
            status = f"NO_SOLUTION_FTFSM_{model.Status}"

        try:
            if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                os.makedirs("./iis_logs/ftfsm", exist_ok=True)
                model.computeIIS()
                scenario_name = "normal" if normal_mode else f"failC{omega[0]}"

                model.write(
                    f"./iis_logs/ftfsm/"
                    f"{topology_name or 'topology'}_run{run_index:03d}_{scenario_name}_ftfsm.ilp"
                )
        except Exception:
            pass

        meta = {
            "status": status,
            "solve_time_sec": solve_time,
            "eta": None,
            "mip_gap": None,
            "num_migrations": None,
            "recovery_plan": {},
            "recovery_assignments": {},
            "recovery_loads_by_failure": {},
        }

        return {}, {}, {}, meta, None, None, status

    # ============================================================
    # Solution extraction
    # ============================================================
    status = "SUCCESS" if model.Status == GRB.OPTIMAL else f"FEASIBLE_STATUS_{model.Status}"

    # b_i^j is the paper's actual traffic distribution.  Use its largest
    # fraction only where an integral compatibility assignment is required.
    final_assign = {
        int(i): int(max(controllers, key=lambda j: b[i, j].X))
        for i in switches
    }

    final_loads = _loads_by_controller(
        final_assign,
        loads,
        controllers=controllers
    )

    paths_pair, paths_by_switch = _build_shortest_paths(
        G,
        final_assign,
        cost_mode=cost_mode
    )

    mig_count = sum(
        1 for i in switches
        if int(init_assign.get(i)) != int(final_assign.get(i))
    )
    failed_c = int(omega[0])

    migrated_switches = {}
    for i in switches:
        original = int(init_assign[i])
        fractions = {
            int(j): float(b[i, j].X)
            for j in controllers
            if b[i, j].X > 1e-6
        }
        moved_fraction = max(0.0, 1.0 - fractions.get(original, 0.0))
        if moved_fraction > 1e-6:
            migrated_switches[int(i)] = {
                "from": original,
                "to_fractional": fractions,
                "dominant_to": int(final_assign[i]),
                "load": float(loads[i]),
                "moved_fraction": moved_fraction,
                "moved_load": float(loads[i]) * moved_fraction,
                "is_orphan": original == failed_c,
            }


    mig_count = len(migrated_switches)

    orphan_migration_count = sum(
        1
        for info in migrated_switches.values()
        if info["is_orphan"]
    )

    non_orphan_migration_count = sum(
        1
        for info in migrated_switches.values()
        if not info["is_orphan"]
    )

    total_migrated_load = sum(
        float(info["moved_load"])
        for info in migrated_switches.values()
    )

    orphan_migrated_load = sum(
        float(info["moved_load"])
        for info in migrated_switches.values()
        if info["is_orphan"]
    )

    non_orphan_migrated_load = sum(
        float(info["moved_load"])
        for info in migrated_switches.values()
        if not info["is_orphan"]
    )
    failed_c = int(omega[0])

    obj_val = float(model.ObjVal)
    mip_gap = float(model.MIPGap) if model.IsMIP else None

    # Store b and f values also, so you can inspect fractional FT-FSM variables
    b_values = {
        int(i): {
            int(j): float(b[i, j].X)
            for j in controllers
            if b[i, j].X > 1e-6
        }
        for i in switches
    }

    if normal_mode:
        f_values = {}
    else:
        f_values = {}
        for i in switches:
            f_values[int(i)] = {}
            for j in controllers:
                for jp in controllers:
                    if jp == j:
                        continue
                    for w in omega:
                        key = (i, j, jp, w)
                        if key in f and f[key].X > 1e-6:
                            f_values[int(i)][f"{j}->{jp}|fail{w}"] = float(f[key].X)

    if normal_mode:
        recovery_plan = {}
        recovery_assignments = {}
        recovery_loads_by_failure = {}
    else:
            # existing recovery loop here
        recovery_plan = {}
        recovery_assignments = {}
        recovery_loads_by_failure = {}

        for failed_c in omega:
            failed_c = int(failed_c)

            recovery_plan[failed_c] = {}
            recovery_assign = dict(final_assign)

            for i in switches:
                targets = dict(b_values.get(int(i), {}))
                original = int(init_assign[i])
                unchanged = (
                    len(targets) == 1
                    and original in targets
                    and abs(float(targets[original]) - 1.0) <= 1e-6
                )
                if unchanged:
                    continue
                recovery_plan[failed_c][int(i)] = targets
                if targets:
                    recovery_assign[int(i)] = int(max(targets, key=targets.get))

            recovery_assignments[failed_c] = recovery_assign

            rec_loads = defaultdict(float)
            for i in switches:
                targets = b_values.get(int(i), {})
                for c, fraction in targets.items():
                    if int(c) != failed_c:
                        rec_loads[int(c)] += (
                            float(loads.get(int(i), 0.0)) * float(fraction)
                        )

            recovery_loads_by_failure[failed_c] = dict(rec_loads)

    # ------------------------------------------------------------
    # Write per-scenario plan JSON
    # ------------------------------------------------------------
    if plot_save_dir is not None:
        ftfsm_dir = os.path.join(plot_save_dir, "FTFSM_recovery_plans")
        os.makedirs(ftfsm_dir, exist_ok=True)

        tag = f"_{plot_file_tag}" if plot_file_tag else ""
        if normal_mode:
            fname = f"{topology_name or 'topology'}{tag}_run{run_index:03d}_ftfsm_normal_assignment.json"
        else:
            fname = (
                f"{topology_name or 'topology'}{tag}_run{run_index:03d}_"
                f"failC{int(omega[0])}_ftfsm_recovery_plan.json"
            )

        plan_file = os.path.join(ftfsm_dir, fname)


        with open(plan_file, "w") as fh:
            json.dump(
                _json_safe(
                    {
                        "topology": topology_name or "topology",
                        "run_index": int(run_index),
                        "status": status,
                        "eta": float(eta.X),
                        "failed_controller": None if normal_mode else int(omega[0]),
                        "final_assign": final_assign,
                        "final_loads": final_loads,
                        "b_fractional_assignment": b_values,
                        "f_fractional_migration": f_values,
                        "recovery_plan_fractional": recovery_plan,
                        "recovery_assignments_largest_fraction": recovery_assignments,
                        "recovery_loads_by_failure": recovery_loads_by_failure,
                        "network_wide_migrated_switches": migrated_switches,
                        "total_network_migrations": int(mig_count),
                        "orphan_migration_count": int(orphan_migration_count),
                        "non_orphan_migration_count": int(non_orphan_migration_count),
                        "total_migrated_load": float(total_migrated_load),
                        "orphan_migrated_load": float(orphan_migrated_load),
                        "non_orphan_migrated_load": float(non_orphan_migrated_load),
                    }
                ),
                fh,
                indent=2
            )

        print(f"[FTFSM PLAN] written → {plan_file}")

    scenario_records = {}
    if not normal_mode and omega and plot_save_dir is not None:
        failed_c = int(omega[0])
        frac_plan = recovery_plan.get(failed_c, {})
        residual_map = {}
        for sw in [s for s, c in init_assign.items() if int(c) == failed_c]:
            assigned_fraction = sum(float(v) for v in frac_plan.get(int(sw), {}).values())
            residual_fraction = max(0.0, 1.0 - assigned_fraction)
            if residual_fraction > 1e-8:
                residual_map[int(sw)] = {
                    "residual_fraction": residual_fraction,
                    "residual_load": float(loads.get(int(sw), 0.0)) * residual_fraction,
                }
        scenario_records[failed_c] = process_failure_scenario(
            algorithm="FTFSM",
            topology_name=topology_name or "topology",
            run_index=run_index,
            failed_controller=failed_c,
            G=G, pos=plot_pos, switches=switches, controllers=controllers,
            loads=loads, capacities=capacities, usable_threshold=usable_threshold,
            overload_threshold=overload_threshold,
            initial_assignment=init_assign,
            recovery_assignment=recovery_assignments.get(failed_c, {}),
            fractional_assignment=frac_plan,
            residual_by_switch=residual_map,
            status=status, solve_time_sec=solve_time, objective_value=obj_val, mip_gap=mip_gap,
            output_root=plot_save_dir,
            comparison_csv_file=comparison_csv_file,
            make_plot=bool(plot_recovery and plot_pos is not None),
            file_tag=f"FTFSM_run{run_index:03d}_failC{failed_c}",
            switch_seed=switch_seed, master_seed=master_seed, run_number=run_number,
            reassignment_scope="global",
            edge_caps=edge_caps,
            msg_bits=msg_bits,
            link_utilization_threshold=0.90,
        )

    meta = {
        "status": status,
        "solve_time_sec": solve_time,
        "eta": float(eta.X),
        "mip_gap": mip_gap,

        "num_migrations": int(mig_count),
        "total_network_migrations": int(mig_count),
        "orphan_migration_count": int(orphan_migration_count),
        "non_orphan_migration_count": int(non_orphan_migration_count),

        "total_migrated_load": float(total_migrated_load),
        "orphan_migrated_load": float(orphan_migrated_load),
        "non_orphan_migrated_load": float(non_orphan_migrated_load),

        "migrated_switches": migrated_switches,

        "recovery_plan": recovery_plan,
        "recovery_assignments": recovery_assignments,
        "recovery_loads_by_failure": recovery_loads_by_failure,
        "b_fractional_assignment": b_values,
        "f_fractional_migration": f_values,
        "reassignment_scope": "global",
        "paths_pair": paths_pair,
        "failure_scenarios": scenario_records,
    }

    return (
        final_assign,
        paths_by_switch,
        final_loads,
        meta,
        obj_val,
        mip_gap,
        status,
    )
