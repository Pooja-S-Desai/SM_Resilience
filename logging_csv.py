import os, csv,math
from collections import defaultdict

def write_edge_usage_csv(
    out_csv,
    topology,
    run_index,
    phase,
    algo,
    G,

    usage_routed,
    edge_caps_uniform,

    edge_caps_before_stress=None,
    edge_caps_after_stress=None,

    paths_by_switch=None,

    usage_shortest_init=None,
    usage_routed_init=None,

    ebc=None,
    stressed_edges=None,
    stress_threshold=0.8,

    # ✅ ALL NEW PARAMS MUST HAVE DEFAULTS
    alpha=None,
    beta=None,
    link_sens=None,
    k_paths=None,
    load_sens=None,
    controller_sens=None,
    
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    file_exists = os.path.exists(out_csv)
    mode = "a" if file_exists else "w"

    # -------- Build edge → switches map --------
    edge_to_switches = defaultdict(set)
    edge_to_paths = defaultdict(list)

    for s, path in (paths_by_switch or {}).items():
        if not path or len(path) < 2:
            continue

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = (u, v) if (u, v) in usage_routed else (v, u)

            edge_to_switches[e].add(s)
            edge_to_paths[e].append(f"{s}:" + "->".join(map(str, path)))

    # stressed lookup
    stressed_dict = {}
    if stressed_edges:
        stressed_dict = {e: util for e, util in stressed_edges}

    with open(out_csv, mode, newline="") as f:
        w = csv.writer(f)

        if not file_exists:
            w.writerow([
                "topology","run","phase","algo",
                "u","v",

                "cap_uniform",
                "cap_before_stress",
                "cap_after_stress",

                "usage_routed",
                "util_routed",

                "usage_shortest_init",
                "usage_routed_init",

                "util_shortest_init",
                "util_routed_init",

                "violated",

                "num_switches",
                "switches",
                "paths",

                "ebc",
                "is_stressed",
                "stress_threshold",
                "cap_change_pct",
                "alpha","beta","link_sens","k_paths","load_sens","controller_sens",
                
            ])

        for (u, v) in G.edges():

            e = (u, v) if (u, v) in usage_routed else (v, u)

            # -------- CAPACITY --------
            cap_uniform = float(edge_caps_uniform.get(e, 0.0))
            cap_before  = float((edge_caps_before_stress or {}).get(e, cap_uniform))
            cap_after   = float((edge_caps_after_stress or {}).get(e, cap_uniform))

            cap_final = cap_after

            # -------- USAGE --------
            usage_final = float(usage_routed.get(e, 0.0))
            util_final = usage_final / cap_final if cap_final > 0 else 0.0

            usage_sp = float((usage_shortest_init or {}).get(e, 0.0))
            usage_rt_init = float((usage_routed_init or {}).get(e, 0.0))

            util_sp = usage_sp / cap_final if cap_final > 0 else 0.0
            util_rt_init = usage_rt_init / cap_final if cap_final > 0 else 0.0

            violated = int(util_final > 1.0)

            # -------- STRESS --------
            is_stressed = 1 if e in stressed_dict else 0

            # -------- CAP CHANGE --------
            if cap_before > 0:
                cap_change_pct = 100 * (cap_after - cap_before) / cap_before
            else:
                cap_change_pct = 0.0

            # -------- SWITCH / PATH --------
            sws = edge_to_switches.get(e, set())
            paths = edge_to_paths.get(e, [])

            # -------- EBC --------
            ebc_val = float((ebc or {}).get(e, 0.0))

            w.writerow([
                topology,
                run_index,
                phase,
                algo,
                u, v,

                round(cap_uniform, 3),
                round(cap_before, 3),
                round(cap_after, 3),

                round(usage_final, 3),
                round(util_final, 4),

                round(usage_sp, 3),
                round(usage_rt_init, 3),

                round(util_sp, 4),
                round(util_rt_init, 4),

                violated,

                len(sws),
                "|".join(map(str, sws)),
                " || ".join(paths),

                round(ebc_val, 6),
                is_stressed,
                stress_threshold,
                round(cap_change_pct, 2),
                alpha,
                beta,
                link_sens,
                k_paths,
                load_sens,
                controller_sens,
            ])

    print("✅ Edge CSV written:", out_csv)

def _d(a, b):
    """Safe difference for CSV deltas; returns None on None/NaN/TypeError."""
    try:
        if a is None or b is None:
                return None
        if isinstance(a, float) and math.isnan(a):
                return None
        if isinstance(b, float) and math.isnan(b):
                return None
        return float(a) - float(b)
    except Exception:
            return None


def write_switch_rt_csv(
    out_csv,
    topology,
    run_index,
    phase,
    algo,
    assign,
    paths_by_switch,
    init_paths_by_switch,   # NEW
    rt_data,
    *,
    init_loads_by_switch=None,
    scaled_loads_by_switch=None,
    init_assign_by_switch=None,
    load_scale_alpha=None,
    alpha=None,
    beta=None,
    link_sens=None,
    k_paths=None,
    load_sens=None,
    controller_sens=None,
):
    import os, csv

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rt_data = rt_data or {}
    init_loads_by_switch = init_loads_by_switch or {}
    scaled_loads_by_switch = scaled_loads_by_switch or {}
    init_assign_by_switch = init_assign_by_switch or {}

    # =========================
    # FINAL (Solver Outputs)
    # =========================
    final_rt_by_s      = rt_data.get("resp_ms_by_switch", {}) or {}
    final_prop_by_s    = rt_data.get("prop_ms_by_switch", {}) or {}
    final_queue_by_c   = rt_data.get("solver_queue_ms_by_ctrl", {}) or {}
    final_real_queue_c = rt_data.get("queue_ms_by_ctrl", {}) or {}
    steiner_by_c       = rt_data.get("Steiner", {}) or {}
    cc_delay_by_s      = rt_data.get("cc_delay_ms_by_switch", {}) or {}

    # =========================
    # INIT (Baseline)
    # =========================
    init_rt_by_s        = rt_data.get("init_resp_ms_by_switch", {}) or {}
    init_prop_by_s      = rt_data.get("init_prop_ms_by_switch", {}) or {}
    init_queue_by_c     = rt_data.get("init_solver_queue_ms_by_ctrl", {}) or {}
    init_real_queue_by_c= rt_data.get("init_real_queue_ms_by_ctrl", {}) or {}

    # =========================
    # Helper
    # =========================
    def _path_to_str(p):
        if not p:
            return ""
        return "->".join(map(str, p))

    file_exists = os.path.exists(out_csv)
    mode = "a" if file_exists else "w"

    with open(out_csv, mode, newline="") as f:
        writer = csv.writer(f)

        # ================= HEADER =================
        if not file_exists:
            writer.writerow([
                "topology","algo","run","phase","switch",

                "init_assigned_controller",
                "assigned_controller",
                "is_migrated",

                # PATHS
                "init_path",
                "final_path",

                # INIT (NO recomputation)
                "init_solver_prop_ms",
                "init_solver_total_rt_ms",
                "init_real_queue_ms",
                "init_switch_load",
                "init_scaled_switch_load",

                # FINAL (NO recomputation)
                "solver_prop_ms",
                "solver_queue_ms",
                "solver_total_rt_ms",
                "real_queue_ms",
                "delta_rt_if_migrated",

                # Additional
                "cc_transfer_ms",
                "steiner_ms",
                "load_scale_alpha",
                "scaled_switch_load",
                "alpha","beta","link_sens","k_paths","load_sens","controller_sens",
            ])

        # ================= ROWS =================
        for s in sorted(assign.keys()):

            c_final = assign.get(s)
            c_init  = init_assign_by_switch.get(s)

            migrated = int(c_init is not None and c_final != c_init)

            # -------- PATHS --------
            init_path_str  = _path_to_str(init_paths_by_switch.get(s))
            final_path_str = _path_to_str(paths_by_switch.get(s))

            # -------- INIT --------
            init_prop = float(init_prop_by_s.get(s, 0.0))
            init_total = float(init_rt_by_s.get(s, 0.0))
            init_q_real = float(init_real_queue_by_c.get(c_init, 0.0))

            # -------- FINAL --------
            final_prop = float(final_prop_by_s.get(s, 0.0))
            final_q_solver = float(final_queue_by_c.get(c_final, 0.0))
            final_q_real = float(final_real_queue_c.get(c_final, 0.0))
            final_total = float(final_rt_by_s.get(s, 0.0))

            # -------- DELTA RT (ONLY IF MIGRATED) --------
            delta_rt = (final_total - init_total) if (migrated and final_total > init_total) else 0.0

            # -------- OTHER --------
            cc_ms = float(cc_delay_by_s.get(s, 0.0)) / 200000.0
            steiner = float(steiner_by_c.get(c_final, 0.0))

            init_load = init_loads_by_switch.get(s, "")
            scaled_load = scaled_loads_by_switch.get(s, "")

            # -------- WRITE --------
            writer.writerow([
                topology,
                algo,
                run_index,
                phase,
                s,

                c_init,
                c_final,
                migrated,

                init_path_str,
                final_path_str,

                # INIT
                round(init_prop, 6),
                round(init_total, 6),
                round(init_q_real, 6),
                init_load,
                init_load,

                # FINAL
                round(final_prop, 6),
                round(final_q_solver, 6),
                round(final_total, 6),
                round(final_q_real, 6),
                round(delta_rt, 6),

                # Extras
                round(cc_ms, 6),
                round(steiner, 6),
                load_scale_alpha,
                scaled_load,
                alpha,
                beta,
                link_sens,
                k_paths,
                load_sens,
                controller_sens,
            ])

    print("✅ Clean switch-level RT + path + migration CSV written:", out_csv)
def compute_migration_cost_components(
    init_assign: dict,
    final_assign: dict,
    Dcc: dict,
    rt_init: dict,
    rt_final: dict,
):
    """
    Migration cost components:

    total = num_mig + cc_transfer + delta_rt

    where:
      num_mig     = number of switches that changed controller
      cc_transfer = controller-to-controller transfer cost
      delta_rt    = sum over migrated switches of max(0, final_rt - init_rt)

    Notes:
    - Only TRUE migrations are considered (c0 != c1)
    - Only RT increases are penalized
    - RT keys are handled robustly
    """

    num_mig = 0
    cc_transfer = 0.0
    delta_rt = 0.0

    # --- Resolve RT maps robustly ---
    rt_init_map = (
        rt_init.get("resp_by_switch")
        or rt_init.get("T_ms_by_switch")
        or rt_init.get("resp_ms_by_switch")
        or {}
    )

    rt_final_map = (
        rt_final.get("resp_by_switch")
        or rt_final.get("T_ms_by_switch")
        or rt_final.get("resp_ms_by_switch")
        or {}
    )

    for s, c1 in final_assign.items():
        c0 = init_assign.get(s, None)

        # Only actual migrations
        if c0 is not None and c1 != c0:
            num_mig += 1

            # --- Controller-to-controller cost ---
            if Dcc is not None:
                if isinstance(Dcc.get(c0, {}), dict):
                    cc_transfer += float(Dcc.get(c0, {}).get(c1, 0.0))
                else:
                    cc_transfer += float(Dcc.get((c0, c1), 0.0))

            # --- RT penalty (only if both values exist) ---
            if s in rt_init_map and s in rt_final_map:
                t0 = float(rt_init_map[s])
                t1 = float(rt_final_map[s])

                if t1 > t0:
                    delta_rt += (t1 - t0)

    total = num_mig + cc_transfer + delta_rt

    return {
        "total": total,
        "num_mig": num_mig,
        "cc_transfer": cc_transfer,
        "delta_rt": delta_rt,
    }
def ensure_dir(path: str):
    """Create directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



def write_controller_csv_reuse_switch_logs(
    out_csv,
    topology,
    run_index,
    algo,
    phase,
    controllers,
    capacities_init,
    capacities_final,
    loads_by_ctrl,
    switch_to_ctrl,
    switch_W_init,
    switch_W_final,
    switch_loads,
    capacity_threshold,
    alpha,
    beta,
    link_sens,
    k_paths,
    load_sens,
    controller_sens,
):

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    file_exists = os.path.exists(out_csv)
    mode = "a" if file_exists else "w"

    with open(out_csv, mode, newline="") as f:
        w = csv.writer(f)

        if not file_exists:
            w.writerow([
                "topology","run","algo","phase","controller",
                "lambda",
                "mu_init","mu_final",
                "scale_factor",
                "rho_init","rho_final",
                "W_init","W_final",
                "stable",
                "alpha","beta","link_sens","k_paths","load_sens","controller_sens",
            ])

        for c in controllers:

            lam = float(loads_by_ctrl.get(c, 0.0))

            mu_init = float(capacity_threshold * capacities_init[c])
            mu_final = float(capacity_threshold * capacities_final[c])

            # ✅ per-controller scale factor
            if capacities_init[c] > 0:
                scale_factor = capacities_final[c] / capacities_init[c]
            else:
                scale_factor = 0.0

            # utilization
            rho_init = lam / mu_init if mu_init > 0 else float("inf")
            rho_final = lam / mu_final if mu_final > 0 else float("inf")

            # aggregate W from switches
            switches_c = [s for s, cc in switch_to_ctrl.items() if cc == c]

            if switches_c:
                W_init = sum(switch_W_init.get(s, 0.0) for s in switches_c) / len(switches_c)
                W_final = sum(switch_W_final.get(s, 0.0) for s in switches_c) / len(switches_c)
            else:
                W_init, W_final = 0.0, 0.0

            stable = int(rho_final < 1.0)

            w.writerow([
                topology,
                run_index,
                algo,
                phase,
                c,
                round(lam, 4),
                round(mu_init, 4),
                round(mu_final, 4),
                round(scale_factor, 4),
                round(rho_init, 4),
                round(rho_final, 4),
                round(W_init, 4),
                round(W_final, 4),
                stable,
                alpha,
                beta,
                link_sens,
                k_paths,
                load_sens,
                controller_sens,
            ])

    print("✅ Controller CSV updated:", out_csv)
