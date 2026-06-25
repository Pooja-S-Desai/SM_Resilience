
import os, json
from collections import defaultdict
import os, csv, time
import numpy as np
from helpers import RESULTS_FOLDER
from helpers import ( Objective )
def _undir(u, v):
    return (u, v) if u < v else (v, u)



import os, csv, time
import numpy as np

# ---------------- SAFE HELPERS ----------------
def _safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else np.nan

def _safe_max(x):
    return float(np.max(x)) if len(x) > 0 else np.nan

def _safe_min(x):
    return float(np.min(x)) if len(x) > 0 else np.nan

def _safe_p95(x):
    return float(np.percentile(x, 95)) if len(x) > 0 else np.nan


def log_run_to_csv(
    logs_dir,
    *,
    run_index,
    topo,
    algo,
    nodes,
    loads_by_switch,
    controller_set,
    controller_caps,
    init_assign,
    final_assign,
    init_loads_by_ctrl,
    final_loads_by_ctrl,
    init_dev,
    final_dev,

    obj_value,
    solve_time_sec,
    mip_gap,
    status_msg,

    rt_mean_init_ms,
    rt_mean_final_ms,
    rt_max_init_ms,
    rt_max_final_ms,
    rt_p95_ms_init,
    rt_p95_ms_final,
    delta_rt_p95_ms,

    ctrl_util_mean_init,
    ctrl_util_mean_final,
    ctrl_util_max_init,
    ctrl_util_max_final,
    delta_ctrl_util_max,
    ctrl_util_p95_init,
    ctrl_util_p95_final,
    ctrl_util_std_final,
    ctrl_util_cov_final,
    jain_load_init,
    jain_load_final,

    ctrl_headroom_p50_final,
    ctrl_headroom_p95_final,

    prop_mean_ms_final,
    W_mean_ms_final,
    unstable_ctrls_final,

    link_util_mean_used_final,
    link_util_max_final,
    link_util_p95_final,
    violated_links_final,
    excess_viol_final,

    rebalanced_load_total,

    alpha,
    beta,
    k_paths,
    link_sens,

    mig_cost_components,
    mig_dist_mean,
    mig_dist_p95,

    rt_init_map,
    rt_final_map,

    prop_delay_max_ms,
    queue_delay_max_ms,
    sync_delay_ms,

    ctrl_rt_mean_ms,
    ctrl_rt_max_ms,
    ctrl_rt_p95_ms,

    ctrl_headroom_mean_final
):
    os.makedirs(logs_dir, exist_ok=True)
    file_path = os.path.join(logs_dir, "run.csv")

    write_header = not os.path.exists(file_path)

    BASE_COLUMNS = [
        "batch_ts","run_index","topology","algo","nodes","num_switches","num_controllers",
        "objective","alpha","beta","k_paths","link_sens",
        "obj_value","solve_time_sec","mip_gap","status",

        "ctrl_load_mean_init","ctrl_load_mean_final",
        "ctrl_load_max_init","ctrl_load_max_final",
        "ctrl_capacity_mean","ctrl_usable_capacity_mean",

        "mig_total","mig_count","mig_cc","mig_rt_penalty",
        "load_dev_init","load_dev_final",

        "ctrl_util_mean_init","ctrl_util_mean_final",
        "ctrl_util_max_init","ctrl_util_max_final","delta_ctrl_util_max",
        "ctrl_util_p95_init","ctrl_util_p95_final",
        "ctrl_util_std_final","ctrl_util_cov_final",
        "jain_load_init","jain_load_final",

        "ctrl_headroom_mean_final","ctrl_headroom_p50_final","ctrl_headroom_p95_final",

        "rt_mean_init_ms","rt_mean_final_ms","delta_rt_mean_ms",
        "rt_max_init_ms","rt_max_final_ms",
        "rt_p95_init_ms","rt_p95_final_ms","delta_rt_p95_ms",

        "prop_delay_mean_ms","prop_delay_max_ms",
        "queue_delay_mean_ms","queue_delay_max_ms",
        "sync_delay_ms",

        "ctrl_rt_mean_ms","ctrl_rt_p95_ms","ctrl_rt_max_ms",
        "unstable_controllers",

        "link_util_mean","link_util_max","link_util_p95",
        "violated_links_count","excess_util_total",

        "rebalanced_load_total",
    ]

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)

        if write_header:
            writer.writeheader()

        batch_ts = time.strftime("%Y%m%d_%H%M%S")

        # ---------------- SAFE LISTS ----------------
        init_ctrl_vals = list(init_loads_by_ctrl.values()) if init_loads_by_ctrl else []
        final_ctrl_vals = list(final_loads_by_ctrl.values()) if final_loads_by_ctrl else []
        cap_vals = list(controller_caps.values()) if controller_caps else []

        # ---------------- MIGRATION SAFE ----------------
        moved = [s for s in init_assign if s in final_assign and init_assign[s] != final_assign[s]]

        loads_moved = [loads_by_switch.get(s, np.nan) for s in moved]

        rt_deltas = [
            rt_final_map[s] - rt_init_map[s]
            for s in moved
            if s in rt_init_map and s in rt_final_map
        ]

        # ---------------- ROW ----------------
        row = {
            "batch_ts": batch_ts,
            "run_index": run_index,
            "topology": topo,
            "nodes": nodes,
            "num_switches": len(loads_by_switch) if loads_by_switch else 0,
            "num_controllers": len(controller_set) if controller_set else 0,

            "objective": Objective,
            "alpha": alpha,
            "beta": beta,
            "k_paths": k_paths,
            "link_sens": link_sens,

            "obj_value": obj_value,
            "solve_time_sec": solve_time_sec,
            "mip_gap": mip_gap,
            "status": status_msg,

            # -------- SAFE CONTROLLER STATS --------
            "ctrl_load_mean_init": _safe_mean(init_ctrl_vals),
            "ctrl_load_mean_final": _safe_mean(final_ctrl_vals),
            "ctrl_load_max_init": _safe_max(init_ctrl_vals),
            "ctrl_load_max_final": _safe_max(final_ctrl_vals),
            "ctrl_capacity_mean": _safe_mean(cap_vals),
            "ctrl_usable_capacity_mean": _safe_mean([0.8*c for c in cap_vals]),

            # -------- MIGRATION --------
            "mig_total": mig_cost_components.get("total", np.nan),
            "mig_count": mig_cost_components.get("num_mig", 0),
            "mig_cc": mig_cost_components.get("cc_transfer", np.nan),
            "mig_rt_penalty": mig_cost_components.get("delta_rt", np.nan),

            "load_dev_init": init_dev,
            "load_dev_final": final_dev,

            # -------- UTIL --------
            "ctrl_util_mean_init": ctrl_util_mean_init,
            "ctrl_util_mean_final": ctrl_util_mean_final,
            "ctrl_util_max_init": ctrl_util_max_init,
            "ctrl_util_max_final": ctrl_util_max_final,
            "delta_ctrl_util_max": delta_ctrl_util_max,
            "ctrl_util_p95_init": ctrl_util_p95_init,
            "ctrl_util_p95_final": ctrl_util_p95_final,
            "ctrl_util_std_final": ctrl_util_std_final,
            "ctrl_util_cov_final": ctrl_util_cov_final,
            "jain_load_init": jain_load_init,
            "jain_load_final": jain_load_final,

            # -------- HEADROOM --------
            "ctrl_headroom_mean_final": ctrl_headroom_mean_final,
            "ctrl_headroom_p50_final": ctrl_headroom_p50_final,
            "ctrl_headroom_p95_final": ctrl_headroom_p95_final,

            # -------- RT --------
            "rt_mean_init_ms": rt_mean_init_ms,
            "rt_mean_final_ms": rt_mean_final_ms,
            "delta_rt_mean_ms": (
                rt_mean_final_ms - rt_mean_init_ms
                if (rt_mean_final_ms is not None and rt_mean_init_ms is not None)
                else np.nan
            ),

            "rt_max_init_ms": rt_max_init_ms,
            "rt_max_final_ms": rt_max_final_ms,
            "rt_p95_init_ms": rt_p95_ms_init,
            "rt_p95_final_ms": rt_p95_ms_final,
            "delta_rt_p95_ms": delta_rt_p95_ms,

            # -------- DELAYS --------
            "prop_delay_mean_ms": prop_mean_ms_final,
            "prop_delay_max_ms": prop_delay_max_ms,
            "queue_delay_mean_ms": W_mean_ms_final,
            "queue_delay_max_ms": queue_delay_max_ms,
            "sync_delay_ms": sync_delay_ms,

            # -------- CTRL RT --------
            "ctrl_rt_mean_ms": ctrl_rt_mean_ms,
            "ctrl_rt_p95_ms": ctrl_rt_p95_ms,
            "ctrl_rt_max_ms": ctrl_rt_max_ms,
            "unstable_controllers": unstable_ctrls_final,

            # -------- LINK --------
            "link_util_mean": link_util_mean_used_final,
            "link_util_max": link_util_max_final,
            "link_util_p95": link_util_p95_final,
            "violated_links_count": violated_links_final,
            "excess_util_total": excess_viol_final,

            "rebalanced_load_total": rebalanced_load_total,
        }

        writer.writerow(row)






def write_detailed_mcf_run_log_json(
    *,
    out_dir: str,
    topology_name: str,
    run_tag: str,                    # e.g., "run_014" or timestamp
    G,
    switches,
    controllers,
    loads,
    capacities,
    init_assign,
    final_assign,
    msg_bits: int,
    edge_caps_e: dict,               # {(u,v): cap_bits/s}
    init_usage_e: dict,              # {(u,v): init_bits/s}
    final_usage_e: dict,             # {(u,v): final_bits/s} from optimizer usage_e
    T_init_ms_by_switch: dict,       # {s: init_rt_ms}
    rt_metrics: dict,                # includes T_ms_by_switch and paths_used_by_switch
):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{topology_name}__{run_tag}__MCF_detailed__{ts}.json"
    fpath = os.path.join(out_dir, fname)

    # ---------- edge records ----------
    edges = []
    for (u, v) in G.edges():
        e = _undir(u, v)

        cap = edge_caps_e.get(e, float("inf"))
        init_used = float(init_usage_e.get(e, 0.0))
        final_used = float(final_usage_e.get(e, 0.0))

        # optional "cost" fields (store whatever exists, do not assume)
        attr = G[u][v]
        edge_cost = attr.get("weight", None)
        latency_sec = attr.get("latency_sec", None)

        edges.append({
            "edge": [int(e[0]), int(e[1])],
            "cap_bits_per_sec": None if cap == float("inf") else float(cap),
            "init_used_bits_per_sec": init_used,
            "final_used_bits_per_sec": final_used,
            "util_init": None if cap == float("inf") else (init_used / cap if cap > 0 else None),
            "util_final": None if cap == float("inf") else (final_used / cap if cap > 0 else None),
            "weight": None if edge_cost is None else float(edge_cost),
            "latency_sec": None if latency_sec is None else float(latency_sec),
        })

    # ---------- switch records ----------
    T_final = rt_metrics.get("T_ms_by_switch", {}) if rt_metrics else {}
    paths_used = rt_metrics.get("paths_used_by_switch", {}) if rt_metrics else {}
    T_final = (
        (rt_metrics or {}).get("T_ms_by_switch")
        or (rt_metrics or {}).get("resp_by_switch")
        or (rt_metrics or {}).get("resp_ms_by_switch")
        or {}
    )
    prop_final = (
        (rt_metrics or {}).get("prop_by_switch")
        or (rt_metrics or {}).get("prop_ms_by_switch")
        or {}
    )

    switches_out = []
    for s in switches:
        c0 = init_assign.get(s, None)
        cf = final_assign.get(s, None)

        switches_out.append({
            "switch": int(s),
            "init_controller": None if c0 is None else int(c0),
            "final_controller": None if cf is None else int(cf),
            "load_req_per_sec": float(loads.get(s, 0.0)),
            "demand_bits_per_sec": float(loads.get(s, 0.0)) * float(msg_bits),
            "init_rt_ms": float(T_init_ms_by_switch.get(s, 0.0)),
            "final_rt_ms": float(T_final.get(s, 0.0)),
            "delta_rt_ms": float(T_final.get(s, 0.0)) - float(T_init_ms_by_switch.get(s, 0.0)),
            # list of used paths (already contains k, frac, cost_ms, edges)
            "paths_used": paths_used.get(s, []),
        })

    payload = {
        "topology": topology_name,
        "run_tag": run_tag,
        "timestamp": ts,
        "counts": {
            "num_switches": int(len(switches)),
            "num_controllers": int(len(controllers)),
            "num_edges": int(G.number_of_edges()),
        },
        "edges": edges,
        "switches": switches_out,
        "rt_summary": {
            "mean_rt_ms_final": rt_metrics.get("mean_rt_ms_final", None) if rt_metrics else None,
            "delta_mean_rt_pos_ms": rt_metrics.get("delta_mean_rt_pos_ms", None) if rt_metrics else None,
        },
    }
    

    with open(fpath, "w") as fp:
        json.dump(payload, fp, indent=2)

    return fpath
def log_failure(algo, status, run_index, topo_name, G_run, alpha, beta, k_path_count, sens):
    log_run_to_csv(
        logs_dir=RESULTS_FOLDER,
        algo=algo,
        run_index=run_index,
        topo=topo_name,
        nodes=G_run.number_of_nodes(),
        loads_by_switch={},
        controller_set=[],
        controller_caps={},
        init_assign={},
        final_assign={},
        init_loads_by_ctrl={},
        final_loads_by_ctrl={},
        init_dev=0.0,
        final_dev=0.0,
        obj_value=None,
        solve_time_sec=0.0,
        mip_gap=None,
        status_msg=status,

        # everything zero
        rt_mean_init_ms=0.0, rt_mean_final_ms=0.0,
        rt_max_init_ms=0.0, rt_max_final_ms=0.0,
        rt_p95_ms_init=0.0, rt_p95_ms_final=0.0, delta_rt_p95_ms=0.0,
        ctrl_util_mean_init=0.0, ctrl_util_mean_final=0.0,
        ctrl_util_max_init=0.0, ctrl_util_max_final=0.0, delta_ctrl_util_max=0.0,
        ctrl_util_p95_init=0.0, ctrl_util_p95_final=0.0,
        ctrl_util_std_final=0.0, ctrl_util_cov_final=0.0,
        jain_load_init=0.0, jain_load_final=0.0,
        ctrl_headroom_p50_final=0.0, ctrl_headroom_p95_final=0.0, ctrl_headroom_mean_final=0.0,
        prop_mean_ms_final=0.0, W_mean_ms_final=0.0,
        unstable_ctrls_final=0,
        link_util_mean_used_final=0.0, link_util_max_final=0.0, link_util_p95_final=0.0,
        violated_links_final=0, excess_viol_final=0.0,
        rebalanced_load_total=0.0,
        alpha=alpha, beta=beta, k_paths=k_path_count, link_sens=sens,
        mig_cost_components={}, mig_dist_mean=0.0, mig_dist_p95=0.0,
        rt_init_map={}, rt_final_map={},
        prop_delay_max_ms=0.0, queue_delay_max_ms=0.0, sync_delay_ms=0.0,
        ctrl_rt_mean_ms=0.0, ctrl_rt_max_ms=0.0, ctrl_rt_p95_ms=0.0,
    )


def _write_resilience_debug_log(
    *,
    log_file,
    topology_name,
    run_index,
    controllers,
    switches,
    loads,
    capacities,
    final_loads,
    bkp,
    residual,
    backup_cost_expr,
    residual_cost_expr,
    post_failure_balance_obj,
    resiliency_obj,
    gamma_res,
):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as fh:

        def w(line=""):
            fh.write(str(line) + "\n")

        w("\n" + "="*90)
        w(f"TOPOLOGY={topology_name} | RUN={run_index}")
        w("RESILIENCY PLAN")
        w("="*90)

        for j in controllers:

            w(f"\nFAILED CONTROLLER {j}")

            backup_load = 0.0
            residual_total = 0.0
            survivor_loads = {}

            for k in controllers:
                if k != j:
                    survivor_loads[k] = final_loads[k]

            w("BACKUPS:")
            found_backup = False

            for s in switches:
                for k in controllers:
                    if k == j:
                        continue

                    if (s, j, k) in bkp and bkp[s, j, k].X > 0.5:
                        w(f"  S{s} (load={loads[s]:.2f}) --> C{k}")
                        backup_load += float(loads[s])
                        survivor_loads[k] += float(loads[s])
                        found_backup = True

            if not found_backup:
                w("  None")

            w(f"Total Backup Load = {backup_load:.2f}")

            w("RESIDUALS:")
            residual_switches = []

            for s in switches:
                if (s, j) in residual and residual[s, j].X > 0.5:
                    residual_switches.append(s)
                    residual_total += float(loads[s])
                    w(f"  S{s} (load={loads[s]:.2f})")

            if not residual_switches:
                w("  None")

            w(f"Residual Load = {residual_total:.2f}")

            w("POST-FAILURE LOADS:")

            for k in controllers:
                if k == j:
                    continue

                util = survivor_loads[k] / float(capacities[k])

                w(
                    f"  C{k}: "
                    f"load={survivor_loads[k]:.2f} "
                    f"cap={capacities[k]:.2f} "
                    f"util={util:.3f}"
                )

            if survivor_loads:
                mx = max(survivor_loads.values())
                mn = min(survivor_loads.values())
                mean_load = sum(survivor_loads.values()) / len(survivor_loads)

                w(
                    f"POSTFAIL MAX={mx:.2f} "
                    f"MIN={mn:.2f} "
                    f"MAX-MIN={mx-mn:.2f} "
                    f"MEAN={mean_load:.2f}"
                )

        w("\n" + "="*90)
        w("RESILIENCY OBJECTIVES")
        w("="*90)
        w(f"Backup Cost           = {backup_cost_expr.getValue():.4f}")
        w(f"Residual Cost         = {residual_cost_expr.getValue():.4f}")
        w(f"Post-Failure Balance  = {post_failure_balance_obj.getValue():.4f}")
        w(f"Total Resiliency Obj  = {resiliency_obj.getValue():.4f}")
        w(f"gamma_res             = {gamma_res}")
        w("="*90)