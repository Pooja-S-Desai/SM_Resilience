# main.py
from __future__ import annotations
import shutil


# =============================
# Standard Library
# =============================
import os
import sys
import json
import argparse
import math
import time
import csv
import traceback
from collections import defaultdict
from arc_MCF_routing import solve_binary_mcf_routing_and_init_rt
# =============================
# Third Party
# =============================
import networkx as nx

# =============================
# Project Core Modules
# =============================
import helpers
from experiment_logger_100runs import log_run_to_csv,log_failure
from experiment_logger_100runs import  write_detailed_mcf_run_log_json
from controller_selection_module import get_min_controllers_and_assignment

# Routing / Path
from path_MCF_routing import choose_single_kpath_for_assignment
from arc_MCF_routing import solve_binary_mcf_routing_and_init_rt

# Optimizers
from Switch_migration_optimizer_MCF_path import run_migration_optimizer_integrated_mcf
from Switch_Migration_MCF_arc import run_migration_optimizer_integrated_mcf_arc
from switch_migration_optimizer_shortest import run_migration_optimizer

# Baselines / Variants
from baseline1_paper_milp import run_baseline1_paper_milp
from baseline2_EASM import run_baseline2_easm_exact
# Steiner / Sync
from steiner_opt import run_steiner_constant_penalty

# Cache
from cache_init_only import compute_or_load_init_only

# Link / Network checks
from link_checks import (
    design_uniform_link_budgets,
    usage_on_paths_undirected,
    summarize_link_usage,
    design_bandwidths_demand_calibrated,
)

# =============================
# Helpers (clean + complete)
# =============================

from helpers import (
    # ---- Seeds ----
    get_run_seeds,

    # ---- Paths ----
    TOPOLOGY_FOLDER,
    TOPOLOGY_LIST_FILE,
    RESULTS_FOLDER,

    # ---- Load / Traffic ----
    MANUAL_LOAD_MEAN,
    MSG_BITS_PER_REQ,
    LINK_BUDGET_BITS,

    # ---- Capacity / Queue ----
    CAPACITY_THRESHOLD,

    # ---- Routing ----
    ROUTING_MODE,
    PATH_CACHE,
    PENALTY_PATH_ALPHA,
    # ---- Objectives ----
    SP_OBJECTIVE,
    MCF_OBJECTIVE,
    RT_OBJECTIVE,

    # ---- Latency / Sync ----
    LATENCY_VARIANT,
    SYNC_MODE,
    SYNC_PHI,
    Objective,
    # ---- Migration weights ----
    # MIG_W_MIG,
    # MIG_W_DELTA,
    # MIG_W_CC,
    # MIG_W_STEINER,
    # MIG_W_RT,

    # ---- Stress settings ----
    LOAD_SCALES,
    REVISED_BW_FRACTION,
    CORE_EDGE_FRACTION,
    CONTROLLER_SCALE_FACTOR,
    CORE_EXPERIMENT_MODE,
    CORE_UTIL_THRESHOLD,

    # ---- Experiment grid ----
    K_VALUES,
    LINK_SENS_VALUES,
    ALPHA_VALUES,
    experiment_configs,
    _rt_pstats,
    _mean_prop_ms,
    _mean_W_ms_over_switches,
    _link_stats,
    _ctrl_lb,
    _rebalanced_total,
    _mig_stats,
)

# =============================
# Preprocessing
# =============================
from preprocessing import (
    list_topologies_by_size,
    get_topology_path_from_list,
    load_topology,
    assign_geographical_weights,
    assign_random_loads,
    precompute_shortest_paths_for_topology,
    derive_sc_cc_from_shortest,
    derive_sc_kpaths_for_mcf,

)

# =================================
# Stress.py
from stress import (
    stressed_switches_from_edges,
    is_feasible,
    find_stressed_edges,scale_core_edge_caps,
    apply_controlled_controller_reduction,
    
)
# ===================================
# =============================
# Plotting / Metrics
# =============================
from plotting import plot_assignments, get_geographical_pos
from rt_metrics import compute_response_metrics, compute_controller_side_rt_only

from logging_csv import (
    write_edge_usage_csv,
    _d,
    ensure_dir,
    compute_migration_cost_components,
    write_switch_rt_csv,
    write_controller_csv_reuse_switch_logs
)
# ==================================================================================


# i have already ensured all capocities are correcct in inital optimizations now i want
#  one control of controller_scale_factor which i can change as variable through which i will 
# try to educe the controller capacites of only overloaded controllers by some percentage may be
#  0 or 10 or 15 etc after reduction also it should try to recheck that is 
# still system feasuble that is all loads still be accomodated within 80% of totoal controller capacities





# Confirm instances are dictionaries not floats
def _assert_dict(name, obj):
    assert isinstance(obj, dict), f"{name} must be dict, got {type(obj)}"



RUN_BASELINE1 = True   # set False to skip
RUN_BASELINE2 = True
# =========================================================================================================================================
# Main Function
# =========================================================================================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--topo-count", type=int, default=10, help="number of topologies to run from the list (start at smallest)")
    ap.add_argument("--topo-idx", type=int, default=0, dest="topo_idx", help="0-based index into the topology list file")
    ap.add_argument("--master-seed", type=int, default=helpers._MASTER_SEED, help="master seed for reproducibility")
    ap.add_argument("--log_sync_artifacts", type=str, default="none")
    args = ap.parse_args()


    # ensure topology list exists
    if not os.path.exists(TOPOLOGY_LIST_FILE):
        list_topologies_by_size(TOPOLOGY_FOLDER, TOPOLOGY_LIST_FILE)
        print(f"📄 Topology list generated at {TOPOLOGY_LIST_FILE}")

    # -------------------------------------
    # ALWAYS EXECUTE BELOW
    # -------------------------------------

    # load topology names
    with open(TOPOLOGY_LIST_FILE, "r") as _f:
        _topo_names = [ln.strip() for ln in _f if ln.strip()]

    _max_topos = min(args.topo_count, len(_topo_names))

    # -------------------------------------
    # Experiment loop (PER CONFIG)
    # -------------------------------------
    for EXP in experiment_configs():

        alpha = round(EXP["alpha"], 1)
        beta  = round(EXP["beta"], 1)
        sens  = EXP["sens"]
        k_path_count = EXP["k"]

        CORE_EDGE_EXTRA_STRESS_FACTOR = sens

        # ----------------------------------------
        # Config folder
        # ----------------------------------------
        exp_tag = f"K{k_path_count}_S{sens}_A{alpha:.1f}"
        EXP_FOLDER = os.path.join(RESULTS_FOLDER, exp_tag)
        ensure_dir(EXP_FOLDER)

        # ✅ KEEP SAME VARIABLE NAMES (but reassign paths)
        GLOBAL_RUN_CSV        = os.path.join(RESULTS_FOLDER, "run.csv")
        switch_csv_file     = os.path.join(RESULTS_FOLDER, "switch.csv")
        link_csv_file       = os.path.join(RESULTS_FOLDER, "link.csv")
        controller_csv_file = os.path.join(RESULTS_FOLDER, "controller.csv")

        # ----------------------------------------
        # Logger PER CONFIG
        # ----------------------------------------
        EXP_LOG_DIR = os.path.join(EXP_FOLDER, "logs")
        ensure_dir(EXP_LOG_DIR)

        # logger = ExperimentLogger(
        #     logs_dir=RESULTS_FOLDER,
        #     alpha=alpha,
        #     beta=beta,
        #     master_seed=args.master_seed,
        # )

        # ----------------------------------------
        # Run root (plots etc.)
        # ----------------------------------------
        RUN_ROOT = os.path.join(EXP_FOLDER, "runs")
        ensure_dir(RUN_ROOT)

        print("\n==============================")
        print(f"K paths      : {k_path_count}")
        print(f"Link stress  : {sens}")
        print(f"Alpha        : {alpha}")
        print(f"Beta         : {beta}")
        print("==============================\n")

        # topology loop
        for idx in range(_max_topos):
            current_algo = "TOPOLOGY_SETUP"

            topo_path = get_topology_path_from_list(
                idx,
                TOPOLOGY_LIST_FILE,
                TOPOLOGY_FOLDER
            )

            topo_name = os.path.splitext(os.path.basename(topo_path))[0]

            try:
                
                G, geo_dup = load_topology(topo_path)

                assign_geographical_weights(G)

                # ---- position check ----
                pos = get_geographical_pos(G)
                if len(pos) != G.number_of_nodes():
                    print(f"⛔ {topo_name}: Missing coordinates. Skipping.")
                    continue

                print(f"\n=== Topology: {topo_name} ===")
                G_pre = G.copy()

                # ---- precompute ----
                dist_all, spath_all, ebc, precompute_time_for_shortest_paths, cache_file = precompute_shortest_paths_for_topology(
                    G=G_pre,
                    topo_name=f"{topo_name}",
                    cost_mode=ROUTING_MODE,
                    use_cache=bool(PATH_CACHE),
                )

                print(f"[PATH_CACHE] {cache_file}")
                print(f"[SP PRECOMPUTE TIME] {precompute_time_for_shortest_paths:.4f} sec")
                TOPO_RUN_ROOT = os.path.join(RUN_ROOT, topo_name)
                ensure_dir(TOPO_RUN_ROOT)

                # ---- run loop ----
                for RUN_INDEX in range(args.runs):
                    current_algo = "RUN_SETUP"

                    SEEDS = get_run_seeds(RUN_INDEX)
                    RUN_DIR = os.path.join(RUN_ROOT, topo_name, f"run_{RUN_INDEX:03d}")
                    ensure_dir(RUN_DIR)
                    def ALG_DIR(name: str):
                        d = os.path.join(RUN_DIR, name)
                        os.makedirs(d, exist_ok=True)
                        return d


                    directed = "Yes" if G.is_directed() else "No"
                    connected = "Yes" if nx.is_connected(G) else "No"

                    G_run = G.copy()

                    loads = assign_random_loads(
                        G_run,
                        seed=SEEDS["loads"],
                        mean=180.0,
                        std=25.0
                    )
                    switch_W_final={}
                    switches = list(G_run.nodes())
                    # quick peek at loads (first 10 switches)
                    print("Sample loads (req/s):", dict(list(loads.items())[:10]))

                    # from cache file check if inital assignemnt exists else call inital assignemnt controller selection optimizer from cache.py
                    #  whatever is the current init just return here and proceed
                    controllers, capacities, init_assign_cs, cachefile, loads, steiner_data,status_cs = compute_or_load_init_only(
                        G=G_run,
                        topo_name=topo_name,
                        master_seed=getattr(helpers, "_MASTER_SEED", args.master_seed),
                        seeds=SEEDS,                   # keep per-run uniqueness if you want distinct inits
                        cost_mode=ROUTING_MODE,
                        get_min_controllers_and_assignment_fn=get_min_controllers_and_assignment,
                        loads=loads,                   # freshly sampled; may be overridden by cached if prefer_cached_loads=True
                        use_cache=True,               # recompute init so stale infeasible caches cannot drive MCF failures
                        prefer_cached_loads=True,      # True → reuse cached loads on cache hit
                    )
                    if status_cs != "OPTIMAL":
                        log_failure("INIT_ASSIGN", status_cs, RUN_INDEX, topo_name, G_run, alpha, beta, k_path_count, sens)
                        continue

                    missing = [s for s in switches if s not in init_assign_cs]
                    if missing:
                        raise ValueError(f"{topo_name}: init_assign_cs missing {len(missing)} switches, e.g. {missing[:5]}")

                    print(f"[CACHE_INIT_ONLY] {cachefile}")
                    print(f"[CTRL] k={len(controllers)} controllers={controllers}")
                    print(f"[CAPS] {capacities}")
                    BASE_CAPACITIES = dict(capacities)
                    SYNC_DELAY_MS = steiner_data["const_ms"]
                    tree_total = steiner_data["tree_total"]
                    tree_edges = steiner_data["tree_edges"]
                    steiner_nodes = steiner_data["steiner_nodes"]
                    per_ctrl_ms_used= SYNC_DELAY_MS

                    edge_caps_uniform = design_uniform_link_budgets(
                        G_run.to_undirected(),
                        per_link_bits=helpers.LINK_BUDGET_BITS)


                    # ------------------------------------------------------------
                    # NEW: Derive SC + CC shortest-path matrices from precomputed all-pairs shortest
                    # Pcc is controller to controller shortest paths
                    # ------------------------------------------------------------
                    dij, paths, Dcc, Pcc = derive_sc_cc_from_shortest(
                        dist_all=dist_all,
                        spath_all=spath_all,
                        switches=switches,
                        controllers=controllers,
                    )


# ===============================================================================================================================================
                     # --------------------------------------------------
                    # Use stressed caps only if stress factor is active
                    # --------------------------------------------------
                    # initial routing usage
                    usage_init = usage_on_paths_undirected(
                        G_run,
                        paths,
                        init_assign_cs,
                        loads,
                        MSG_BITS_PER_REQ
                    )

                    chosen_caps = None
                    chosen_frac = None
                    BASE_LOADS = dict(loads)
                    for f in REVISED_BW_FRACTION:
                        print(f"\nTesting fraction = {f}")
                        #  Here we try to take the uniform bandwidth capapcities and try to change those such that 
                        # %f of the links should becomes overloaded that is above 80% and rest of them should be as in fucntion where we say that non-overloaded link bw 
                        # must be in range of 40-80% util and overloaded must be 81 to 98%util
                        caps_try = design_bandwidths_demand_calibrated(
                            G_run,
                            loads,
                            MSG_BITS_PER_REQ,
                            paths,
                            init_assign_cs,
                            frac=f,
                            seed=SEEDS["link_caps"]
                        )

                        ok = is_feasible(usage_init, caps_try)

                        # compute max util for logging
                        max_util = max(
                            (usage_init[e] / caps_try.get(e,1))
                            for e in usage_init
                            if caps_try.get(e,0)>0
                        )

                        print(f"Max util = {max_util:.3f}")

                        if ok:
                            print("✅ Feasible")
                            chosen_caps = caps_try
                            chosen_frac = f
                            break
                        else:
                            print("❌ Infeasible")

                    # fallback
                    if chosen_caps is None:
                        print("⚠️ No feasible fraction — using largest")
                        chosen_caps = caps_try
                        chosen_frac = REVISED_BW_FRACTION[-1]

                    edge_caps = chosen_caps
                    for (u,v),cap in edge_caps.items():
                        if G_run.has_edge(u,v):
                            G_run[u][v]["bandwidth"] = cap
                        else:
                            G_run[v][u]["bandwidth"] = cap
                    print(f"Selected fraction = {chosen_frac}")




                    usage_shortest_init, util_e = usage_on_paths_undirected(
                        G_run,
                        paths, #shortest_paths
                        init_assign_cs,
                        loads,
                        MSG_BITS_PER_REQ,
                        edge_caps=edge_caps  # stressed bandwidth
                    )

                    # ==================================================
                    # CORE EDGE SELECTION (Two experiment modes)
                    # ==================================================
                    stressed_edges = find_stressed_edges(usage_shortest_init, edge_caps, 0.8)
                    if CORE_EXPERIMENT_MODE == "congestion":
                        core_edges = [
                            e for e, u in stressed_edges
                            if u >= CORE_UTIL_THRESHOLD
                        ]

                        print(f"[MODE: CONGESTION]")
                        print(f"Core edges = util ≥ {CORE_UTIL_THRESHOLD}")
                        print(f"Selected {len(core_edges)} congested edges")


                    elif CORE_EXPERIMENT_MODE == "structural":

                        # Sort edges by EBC (descending)
                        sorted_ebc = sorted(ebc.items(), key=lambda x: -x[1])

                        k = max(1, int(CORE_EDGE_FRACTION * len(sorted_ebc)))

                        core_edges = [e for e, _ in sorted_ebc[:k]]

                        print(f"[MODE: STRUCTURAL]")
                        print(f"Core edges = top {CORE_EDGE_FRACTION*100:.1f}% by EBC")
                        print(f"Selected {len(core_edges)} structural edges")

                    else:
                        raise ValueError(f"Unknown CORE_EXPERIMENT_MODE: {CORE_EXPERIMENT_Mrun_steiner_constant_penaltyODE}")        



                    chosen_paths_init_sc, usage_init_routed, init_rt,status_routing = solve_binary_mcf_routing_and_init_rt(
                        G=G_run,
                        switches=switches,
                        assignment=init_assign_cs,
                        loads_req_s=loads,
                        capacities_req_s=capacities,
                        edge_caps_bits=edge_caps,
                        msg_bits=MSG_BITS_PER_REQ,
                        global_threshold=CAPACITY_THRESHOLD,
                        rho_max=0.95,
                        sync_per_ctrl_ms=float(SYNC_DELAY_MS or 0.0),
                        round_trip=True,
                        fiber_sec_per_km=5e-6,
                        cost_mode=ROUTING_MODE,
                        time_limit=300,
                        output_flag=0,
                    )

                    if "INFEASIBLE" in status_routing or "FAILED" in status_routing:
                        log_failure("INIT_ROUTING", status_routing, RUN_INDEX, topo_name, G_run, alpha, beta, k_path_count, sens)
                        continue    
                    # --- paths (s,c) → s ---
                    chosen_paths_init = {
                        s: chosen_paths_init_sc.get((s, init_assign_cs[s]), [])
                        for s in switches
                    }
                    # --- extract ALL RT metrics (NO recompute) ---
                    T_init = init_rt["T_final_ms_by_switch"]
                    prop_init = init_rt["prop_by_switch"]
                    W_init = init_rt["Wsys_by_ctrl"]

                    lam_init = init_rt["lambda_by_ctrl"]
                    mu_init = init_rt["mu_by_ctrl"]
                    rho_init = init_rt["rho_by_ctrl"]

                    unstable_ctrls_init = init_rt["unstable_controllers"]

                    init_mean_rt = init_rt["init_mean_rt_ms"]
                    # init_p95_rt = init_rt["init_p95_rt_ms"]


                    # --------------------------------------------------------------------------------------
                    # After inital assignemnt impose extra stress or link sensitivity test on inital setup to run the algorithms
                    # =================================================================================================
                    if CORE_EDGE_EXTRA_STRESS_FACTOR > 0:
                        edge_caps_stressed =scale_core_edge_caps(edge_caps, core_edges, CORE_EDGE_EXTRA_STRESS_FACTOR) #Edgecaps caliberated with 10% overloads
                        edge_caps = edge_caps_stressed
                        print("✅ Using stressed edge capacities for optimization")
                    else:
                        edge_caps_stressed = edge_caps.copy()
                        print("ℹ️ No edge stress applied — using calibrated capacities")

# ====================================================================================================================
# After inital routing and applying link sens recalcualte the usage and genrate the paths for mcf path in stressed env
# ====================================================================================================================

                    derive_start = time.perf_counter()
                    sc_kpaths, sc_kpath_cost_ms, sc_kpath_edges = derive_sc_kpaths_for_mcf(
                        G_run,
                        switches=switches,
                        controllers=controllers,
                        K=k_path_count,
                        edge_caps=edge_caps_stressed,
                        usage_e=usage_init_routed,          # usage after inital assignemnt and routing so that this can be used for mcf_path of switch migration.-9
                        alpha=PENALTY_PATH_ALPHA,
                    )
                    derive_time = time.perf_counter() - derive_start
                    kpath_preparation_time = derive_time

                   
                    # Switches linked to core edges are identified and only their loads are increased
                    # e.g., 1.2 → 20% load increase

                    chosen_scale = None
                    core_switches = {n for e in core_edges for n in e}

                    for scale_factor in LOAD_SCALES:

                        print(f"\n--- Testing scale={scale_factor} ---")

                        test_loads = {
                            s: (scale_factor * BASE_LOADS[s] if s in core_switches else BASE_LOADS[s])
                            for s in BASE_LOADS
                        }

                        lambda_total_scaled = sum(test_loads.values())

                        mu_total = sum(
                            CAPACITY_THRESHOLD * cap
                            for cap in capacities.values()
                        )

                        if lambda_total_scaled <= mu_total:

                            print(
                                f"✅ Feasible scale={scale_factor} | "
                                f"load={lambda_total_scaled:.1f} | "
                                f"usable_cap={mu_total:.1f}"
                            )

                            chosen_scale = scale_factor
                            loads = test_loads
                            break

                        else:

                            print(
                                f"❌ Infeasible scale={scale_factor} | "
                                f"load={lambda_total_scaled:.1f} > usable_cap={mu_total:.1f}"
                            )

                    # ============================
                    # SYSTEM FEASIBILITY CHECK
                    # afrer switchea re scaled we have to check 
                    # if all those switch loads are feasible in 80% of overall controller 
                    # capapcities which means can we accomodate all these swicth loads without any controller overload 
                    # ============================

                    lambda_total_scaled = sum(loads.values())

                    mu_total = sum(
                        CAPACITY_THRESHOLD * capacities[c]
                        for c in controllers
                    )

                    if lambda_total_scaled > mu_total:
                        print(
                            f"⚠️ Infeasible scenario: "
                            f"load={lambda_total_scaled:.1f} > usable_cap={mu_total:.1f}"
                        )
                        continue

                    rho_sys = lambda_total_scaled / mu_total

                    print(
                        f"✅ System feasible | "
                        f"total_load={lambda_total_scaled:.1f} | "
                        f"usable_cap={mu_total:.1f} | "
                        f"ρ_sys={rho_sys:.3f}"
                    )


                    # ------------------------------------------------------------
                    # META (FOR LOGGING / REPRODUCIBILITY)
                    # ------------------------------------------------------------
                    seeds_blob = {
                        "run": SEEDS["run"],
                        "loads": SEEDS["loads"],
                        "caps_menu": SEEDS["caps_menu"],
                        "caps_nodes": SEEDS["caps_nodes"],
                    }

                    edge_caps_calibrated = edge_caps.copy()

                    write_edge_usage_csv(
                        out_csv=link_csv_file,
                        topology=topo_name,
                        run_index=RUN_INDEX,
                        phase="INIT",
                        algo="MCF_PATH",
                        G=G_run,
                        usage_routed=usage_init_routed,
                        edge_caps_uniform=edge_caps_uniform,
                        edge_caps_before_stress=edge_caps_calibrated,
                        edge_caps_after_stress=edge_caps_stressed,
                        paths_by_switch=chosen_paths_init,
                        usage_shortest_init=usage_shortest_init,
                        usage_routed_init=usage_init_routed,
                        ebc=ebc,
                        stressed_edges=stressed_edges,
                        alpha=alpha,
                        beta=beta,
                        link_sens=sens,
                        k_paths = k_path_count,
                        load_sens =  0.0,
                        controller_sens = 0.0,

                    )
                    init_loads_by_ctrl = defaultdict(float)
                    for s, c in init_assign_cs.items():
                        init_loads_by_ctrl[c] += float(loads.get(s, 0.0))
                    init_dev = (max(init_loads_by_ctrl.values()) - min(init_loads_by_ctrl.values())) if init_loads_by_ctrl else 0.0



                    if(CONTROLLER_SCALE_FACTOR > 0):
                        capacities, overloaded_ctrls, feasible, applied = apply_controlled_controller_reduction(
                                capacities=capacities,
                                loads_by_ctrl=init_loads_by_ctrl,  # ← FIXED
                                loads=loads,
                                controllers=controllers,
                                controller_scale_factor=CONTROLLER_SCALE_FACTOR,
                                threshold=CAPACITY_THRESHOLD,
                            )

                        print(f"[CTRL REDUCTION] applied={applied}, feasible={feasible}")
                        print(f"[OVERLOADED CTRLS] {overloaded_ctrls}")
                        if not feasible:
                            print("⚠️ Skipping scenario due to infeasible capacity reduction")
                            continue


                    # ============================================================
                    # BASELINE (INIT) METRICS — REUSABLE ACROSS ALGORITHMS
                    # ============================================================

                    # --- extract ALL RT metrics (NO recompute) ---
                    T_init = init_rt["T_final_ms_by_switch"]
                    prop_init = init_rt["prop_by_switch"]
                    W_init = init_rt["Wsys_by_ctrl"]

                    lam_init = init_rt["lambda_by_ctrl"]
                    mu_init = init_rt["mu_by_ctrl"]
                    rho_init = init_rt["rho_by_ctrl"]

                    unstable_ctrls_init = init_rt["unstable_controllers"]

                    init_mean_ms_rt = init_rt["init_mean_rt_ms"]
                    # init_p95_rt = init_rt["init_p95_rt_ms"]
                    rt_max_ms_init = init_rt["max_resp"]

                    # ------------------------------------------------------------
                    # Load balancing lower bound
                    # ------------------------------------------------------------
                    lb_init = _ctrl_lb(lam_init, capacities, usable_frac=1.0)


                    # ------------------------------------------------------------
                    # Response-time statistics (already in ms)
                    # ------------------------------------------------------------
                    rtp_init = _rt_pstats(init_rt)
                    rt_p95_ms_init  = (
                        float(rtp_init["p95"]) if rtp_init["p95"] == rtp_init["p95"] else float("nan")
                    )


                    # ------------------------------------------------------------
                    # RT COMPONENT BREAKDOWN
                    # ------------------------------------------------------------
                    prop_mean_ms_init = _mean_prop_ms(init_rt)                  # network delay
                    queue_mean_ms_init = _mean_W_ms_over_switches(init_rt, init_assign_cs)  # queue delay

                    # Steiner delay (constant per request)
                    steiner_ms_init = SYNC_DELAY_MS

                    # Fraction of RT due to queueing
                    queue_share_init = (
                        queue_mean_ms_init / max(1e-9, init_mean_ms_rt)
                    ) if math.isfinite(init_mean_ms_rt) else float("nan")


                    # ------------------------------------------------------------
                    # CONTROLLER-SIDE METRICS (QUEUE ONLY)
                    # ------------------------------------------------------------
                    # ctrl_rt_mean_ms_init = float(ctrl_only_init["gamma_bar_ctrl"])

                    # ctrl_rt_by_ctrl_ms_init = {
                    #     c: v for c, v in ctrl_only_init["gamma_by_ctrl"].items()
                    # }

                    # ctrl_rt_unstable_init = len(ctrl_only_init["unstable_controllers"])
                    link_sum_init = summarize_link_usage(usage_init_routed, edge_caps)
                    link_init_stats = _link_stats(usage_init_routed, edge_caps)

                    # Optional artifact file path (image or JSON you might save)
                    artifact_path = ""
                    if args.log_sync_artifacts == "full":
                        os.makedirs(os.path.join(EXP_LOG_DIR, "sync_artifacts"), exist_ok=True)
                        artifact_path = os.path.join(
                            EXP_LOG_DIR, "sync_artifacts",
                            f"{topo_name}_run{RUN_INDEX:03d}_sync.json"
                            )
                        with open(artifact_path, "w") as fh:
                            json.dump(
                                {
                                    "controllers": controllers,
                                    "metric": "constant steiner weight",
                                    "sync_mode": "steiner",
                                    "tree_edges": tree_edges,      # log the weighted triples
                                    "steiner_nodes": steiner_nodes,
                                    "sync_phi": SYNC_PHI,
                                    "per_controller_ms": SYNC_DELAY_MS,
                                },
                                fh, indent=2)
# ==========================================================================================================================
                        # ===========BASELINE-1: optimization-only (load-based objective)==============
# ============================================================================================================================                  
                    if RUN_BASELINE1:
                        print(f"🚀 ENTERING BASELINE1 | run={RUN_INDEX}")

                        solve_start = time.perf_counter()
                        fa_b1, paths_b1,fl_b1, meta_b1,obj_val_b1,mip_b1,status_b1 = run_baseline1_paper_milp(
                            G=G_run,
                            switches=switches,
                            controllers=controllers,
                            init_assign=init_assign_cs,
                            loads=loads,
                            capacities=capacities,
                            dij=dij,
                            Dcc=Dcc,
                            verbose=True
                        )
                        solve_time_b1 = time.perf_counter() - solve_start

                        if "INFEASIBLE" in status_b1 or "NO_FEASIBLE" in status_b1 or "TIME_LIMIT" in status_b1:
                            log_failure("B1", "B1_INFEASIBLE", RUN_INDEX, topo_name, G_run, alpha, beta, k_path_count, sens)
                            continue    
                        # ----- evaluation (IDENTICAL to SP) -----
                        rt_b1 = compute_response_metrics(
                            G_run, fa_b1, loads, capacities, paths_b1, round_trip=True,
                            per_ctrl_ms=SYNC_DELAY_MS,
                        )
                        usage_b1 = usage_on_paths_undirected(G_run, paths, fa_b1, loads, MSG_BITS_PER_REQ)
                        link_b1_stats = _link_stats(usage_b1, edge_caps)
                        lb_b1 = _ctrl_lb(fl_b1, capacities, usable_frac=1.0)

                        rtp_b1 = _rt_pstats(rt_b1)

                        rt_mean_ms_b1 =  float(rt_b1.get("mean_resp", float("nan")))
                        rt_p95_ms_b1  =  float(rtp_b1["p95"]) if rtp_b1["p95"] == rtp_b1["p95"] else float("nan")
                        rt_max_ms_b1 = float(rt_b1.get("max_resp") or float("nan"))

                        prop_mean_ms_b1 = _mean_prop_ms(rt_b1)
                        W_mean_ms_b1    = _mean_W_ms_over_switches(rt_b1, fa_b1)
                        queue_share_b1  = (W_mean_ms_b1 / max(1e-9, rt_mean_ms_b1)) if math.isfinite(rt_mean_ms_b1) else float("nan")

                        mig_stats_b1 = _mig_stats(G_run, init_assign_cs, fa_b1, ROUTING_MODE, rt_init=init_rt, rt_final=rt_b1)

                        # ---- define SP-consistent fields for logging ----
                        mig_b1 = int(mig_stats_b1.get("count", 0))  
                        obj_b1 = "BASELINE 1"
                        # ---- migration-cost breakdown: DO IT ALWAYS if weights are active ----
                        mig_cost_comp_b1 = {}

                        mig_cost_comp_b1 = compute_migration_cost_components(
                                init_assign=init_assign_cs,
                                final_assign=fa_b1,
                                Dcc=Dcc,
                                rt_init=init_rt,
                                rt_final=rt_b1
                            )
                        # ---- plot (same as SP) ----
                        plot_assignments(
                            G_run, pos, switches, controllers,
                            init_assign_cs, fa_b1, loads, fl_b1,
                            topo_name, ALG_DIR("B1"), capacities,
                            extra_title=f"B1 objective=paper_milp",
                            file_tag=f"B1_run{RUN_INDEX:03d}_topo{idx:02d}"
                        )

                        _assert_dict("link_summary_init", link_sum_init)
                        _assert_dict("link_summary_final_B1", summarize_link_usage(usage_b1, edge_caps))

                        # =========================
                        # BUILD controller loads (final)
                        # =========================
                        final_loads_b1 = defaultdict(float)
                        for s, c in fa_b1.items():
                            final_loads_b1[c] += float(loads.get(s, 0.0))

                        final_dev_b1 = (
                            max(final_loads_b1.values()) - min(final_loads_b1.values())
                        ) if final_loads_b1 else 0.0

                        # =========================
                        # SAFE migration cost dict
                        # =========================
                        mig_cost_comp_b1 = mig_cost_comp_b1 or {}

                        # =========================
                        # CALL LOGGER
                        # =========================
                        log_run_to_csv(
                            logs_dir=RESULTS_FOLDER,

                            algo="B1",   # ✅ IMPORTANT

                            run_index=RUN_INDEX,
                            topo=topo_name,
                            nodes=G_run.number_of_nodes(),

                            loads_by_switch=loads,
                            controller_set=controllers,
                            controller_caps=capacities,

                            init_assign=init_assign_cs,
                            final_assign=fa_b1,

                            init_loads_by_ctrl={},
                            final_loads_by_ctrl=final_loads_b1,

                            init_dev=init_dev,
                            final_dev=final_dev_b1,

                            obj_value=obj_val_b1,
                            solve_time_sec=solve_time_b1,
                            mip_gap=mip_b1,
                            status_msg="SUCCESS",

                            # ---------------- RT ----------------
                            rt_mean_init_ms=init_mean_ms_rt,
                            rt_mean_final_ms=rt_mean_ms_b1,

                            rt_max_init_ms=rt_max_ms_init,
                            rt_max_final_ms=rt_max_ms_b1,

                            rt_p95_ms_init=rt_p95_ms_init,
                            rt_p95_ms_final=rt_p95_ms_b1,
                            delta_rt_p95_ms=rt_p95_ms_b1 - rt_p95_ms_init,

                            # ---------------- CTRL ----------------
                            ctrl_util_mean_init=lb_init["util_mean"],
                            ctrl_util_mean_final=lb_b1["util_mean"],

                            ctrl_util_max_init=lb_init["util_max"],
                            ctrl_util_max_final=lb_b1["util_max"],
                            delta_ctrl_util_max=_d(lb_b1["util_max"], lb_init["util_max"]),

                            ctrl_util_p95_init=lb_init["util_p95"],
                            ctrl_util_p95_final=lb_b1["util_p95"],

                            ctrl_util_std_final=lb_b1["util_std"],
                            ctrl_util_cov_final=lb_b1["util_cov"],

                            jain_load_init=lb_init["jain"],
                            jain_load_final=lb_b1["jain"],

                            ctrl_headroom_p50_final=lb_b1["head_p50"],
                            ctrl_headroom_p95_final=lb_b1["head_p95"],
                            ctrl_headroom_mean_final=lb_b1["head_mean"],

                            # ---------------- DELAYS ----------------
                            prop_mean_ms_final=prop_mean_ms_b1,
                            W_mean_ms_final=W_mean_ms_b1,

                            unstable_ctrls_final=len(rt_b1.get("unstable_controllers", [])),

                            # ---------------- LINK ----------------
                            link_util_mean_used_final=link_b1_stats["mean_used"],
                            link_util_max_final=link_b1_stats["max"],
                            link_util_p95_final=link_b1_stats["p95"],
                            violated_links_final=link_b1_stats["viol"],
                            excess_viol_final=link_b1_stats["excess"],

                            # ---------------- LOAD ----------------
                            rebalanced_load_total=_rebalanced_total(lam_init, final_loads_b1),

                            # ---------------- CONFIG ----------------
                            alpha=alpha,
                            beta=beta,
                            k_paths=k_path_count,
                            link_sens=sens,

                            # ---------------- MIG ----------------
                            mig_cost_components=mig_cost_comp_b1,
                            mig_dist_mean=mig_stats_b1["dist_mean"],
                            mig_dist_p95=mig_stats_b1["dist_p95"],

                            # ---------------- RT MAP ----------------
                            rt_init_map=T_init,
                            rt_final_map=rt_b1["T_final_ms_by_switch"],

                            # ---------------- DELAY BREAKDOWN ----------------
                            prop_delay_max_ms=rt_b1["prop_max_ms"],
                            queue_delay_max_ms = max(v for v in rt_b1["Wsys_by_ctrl"].values()
                                                    if math.isfinite(v)),
                            sync_delay_ms=SYNC_DELAY_MS,

                            # ---------------- CTRL RT ----------------
                            ctrl_rt_mean_ms=0.0,  # optional (fill later if needed)
                            ctrl_rt_max_ms=0.0,
                            ctrl_rt_p95_ms=0.0,
                        )

                        write_edge_usage_csv(
                            out_csv=link_csv_file,
                            topology=topo_name,
                            run_index=RUN_INDEX,
                            phase="SM",
                            algo="B1",   

                            G=G_run,
                            usage_routed=usage_b1,

                            edge_caps_uniform=edge_caps_uniform,
                            edge_caps_before_stress=edge_caps_calibrated,
                            edge_caps_after_stress=edge_caps_stressed,

                            paths_by_switch=paths_b1,   

                            usage_shortest_init=usage_shortest_init,
                            usage_routed_init=usage_init_routed,

                            ebc=ebc,
                            stressed_edges=stressed_edges,

                            alpha=alpha,
                            beta=beta,
                            link_sens=sens,
                            k_paths=k_path_count,
                            load_sens=0.0,
                            controller_sens=0.0,
                        )
                        write_switch_rt_csv(
                            switch_csv_file,
                            topo_name,
                            RUN_INDEX,
                            "B1",
                            "SM",

                            # FINAL assignment + paths
                            fa_b1,
                            paths_b1,
                            chosen_paths_init,   # ✅ ADD THIS

                            # RT DATA (B1 uses SP-style metrics)
                            {
                                # FINAL
                                "resp_ms_by_switch": rt_b1["T_final_ms_by_switch"],
                                "prop_ms_by_switch": rt_b1["prop_by_switch"],
                                "solver_queue_ms_by_ctrl": rt_b1["Wsys_by_ctrl"],
                                "queue_ms_by_ctrl": rt_b1["Wsys_by_ctrl"],

                                # INIT
                                "init_resp_ms_by_switch": T_init,
                                "init_prop_ms_by_switch": prop_init,
                                "init_solver_queue_ms_by_ctrl": W_init,
                            },

                            init_loads_by_switch=BASE_LOADS,
                            scaled_loads_by_switch=loads,
                            init_assign_by_switch=init_assign_cs,

                            load_scale_alpha="0",

                            alpha=alpha,
                            beta=beta,
                            link_sens=sens,
                            k_paths=k_path_count,
                            load_sens=0.0,
                            controller_sens=0.0,
                        )

                        # =========================
                        # BUILD switch-level queue delay (FINAL)
                        # =========================
                        W_b1_by_ctrl = rt_b1.get("Wsys_by_ctrl", {})

                        switch_W_final = {
                            s: W_b1_by_ctrl.get(fa_b1[s], 0.0)
                            for s in fa_b1
                        }
                        write_controller_csv_reuse_switch_logs(
                            out_csv=controller_csv_file,
                            topology=topo_name,
                            run_index=RUN_INDEX,
                            algo="B1",
                            phase="SM",
                            controllers=controllers,

                            capacities_init=BASE_CAPACITIES,
                            capacities_final=capacities,

                            loads_by_ctrl=fl_b1,
                            switch_to_ctrl=fa_b1,

                            switch_W_init=W_init,
                            switch_W_final=switch_W_final,

                            switch_loads=loads,
                            capacity_threshold=CAPACITY_THRESHOLD,
                            alpha=alpha,
                            beta=beta,
                            link_sens=sens,
                            k_paths = k_path_count,
                            load_sens =  0.0,
                            controller_sens = 0.0,
                        )


# ==========================================================================================================================
# ===========BASELINE-3: EASM-"Efficiency-aware Switch Migration for Balancing Controller Loads in SDN"=====================
# ============================================================================================================================

                    if RUN_BASELINE2:
                        print(f"🚀 ENTERING BASELINE2 EASM | run={RUN_INDEX}")

                        solve_start = time.perf_counter()

                        (
                            fa_easm,
                            paths_easm_pair,
                            paths_easm_switch,
                            fl_easm,
                            obj_val_easm,
                            mig_easm,
                            usage_easm,
                            mip_easm,
                            status_easm,
                            meta_easm,
                        ) = run_baseline2_easm_exact(
                            G=G_run,
                            switches=switches,
                            controllers=controllers,
                            loads=loads,
                            capacities=capacities,
                            init_assign=init_assign_cs,
                            dij=dij,
                            paths_sc=paths,
                            msg_bits=MSG_BITS_PER_REQ,
                            usable_threshold=1.0,
                            overload_threshold=0.8,
                            verbose=False,
                        )

                        solve_time_easm = time.perf_counter() - solve_start

                        # ======================================================
                        # MAP EASM OUTPUTS → B3 VARIABLE NAMES
                        # ======================================================

                        fa_b3 = fa_easm
                        paths_b3 = paths_easm_pair
                        paths_b3_switch = paths_easm_switch
                        fl_b3 = fl_easm
                        obj_val_b3 = obj_val_easm
                        mig_b3 = mig_easm
                        usage_b3 = usage_easm
                        mip_b3 = mip_easm
                        status_b3 = status_easm
                        solve_time_b3 = solve_time_easm

                        # ======================================================
                        # RESPONSE-TIME EVALUATION
                        # ======================================================

                        rt_b3 = compute_response_metrics(
                            G_run,
                            fa_b3,
                            loads,
                            capacities,
                            paths_b3,
                            round_trip=True,
                            per_ctrl_ms=SYNC_DELAY_MS,
                        )

                        if "INFEASIBLE" in status_b3 or "TIME_LIMIT" in status_b3:
                            log_failure("B3", "B3_INFEASIBLE", RUN_INDEX, topo_name, G_run, alpha, beta, k_path_count, sens)
                        else:
                            usage_b3 = usage_on_paths_undirected(
                                G_run, paths, fa_b3, loads, MSG_BITS_PER_REQ
                            )

                            link_b3_stats = _link_stats(usage_b3, edge_caps)
                            lb_b3 = _ctrl_lb(fl_b3, capacities, usable_frac=1.0)

                            rtp_b3 = _rt_pstats(rt_b3)

                            rt_mean_ms_b3 = float(rt_b3.get("mean_resp", float("nan")))
                            rt_p95_ms_b3  = float(rtp_b3["p95"]) if rtp_b3["p95"] == rtp_b3["p95"] else float("nan")
                            rt_max_ms_b3  = float(rt_b3.get("max_resp" or float("nan")))

                            prop_mean_ms_b3 = _mean_prop_ms(rt_b3)
                            W_mean_ms_b3    = _mean_W_ms_over_switches(rt_b3, fa_b3)

                            mig_stats_b3 = _mig_stats(
                                G_run,
                                init_assign_cs,
                                fa_b3,
                                ROUTING_MODE,
                                rt_init=init_rt,
                                rt_final=rt_b3
                            )

                            mig_b3 = int(mig_stats_b3.get("count", 0))
                            obj_b3 = obj_val_b3

                            mig_cost_comp_b3 = compute_migration_cost_components(
                                init_assign=init_assign_cs,
                                final_assign=fa_b3,
                                Dcc=Dcc,
                                rt_init=init_rt,
                                rt_final=rt_b3
                            ) or {}

                            # ======================================================
                            # PLOT
                            # ======================================================

                            plot_assignments(
                                G_run, pos, switches, controllers,
                                init_assign_cs, fa_b3, loads, fl_b3,
                                topo_name, ALG_DIR("B3"), capacities,
                                extra_title=f"B3 objective=EASM efficiency-aware heuristic",
                                file_tag=f"B3_run{RUN_INDEX:03d}_topo{idx:02d}"
                            )

                            _assert_dict("link_summary_init", link_sum_init)
                            _assert_dict("link_summary_final_B3", summarize_link_usage(usage_b3, edge_caps))

                            # ======================================================
                            # FINAL CONTROLLER LOADS
                            # ======================================================

                            final_loads_b3 = defaultdict(float)
                            for s, c in fa_b3.items():
                                final_loads_b3[c] += float(loads.get(s, 0.0))

                            final_dev_b3 = (
                                max(final_loads_b3.values()) - min(final_loads_b3.values())
                            ) if final_loads_b3 else 0.0

                            # ======================================================
                            # MAIN CSV LOG
                            # ======================================================

                            log_run_to_csv(
                                logs_dir=RESULTS_FOLDER,

                                algo="B3",

                                run_index=RUN_INDEX,
                                topo=topo_name,
                                nodes=G_run.number_of_nodes(),

                                loads_by_switch=loads,
                                controller_set=controllers,
                                controller_caps=capacities,

                                init_assign=init_assign_cs,
                                final_assign=fa_b3,

                                init_loads_by_ctrl=dict(init_loads_by_ctrl),
                                final_loads_by_ctrl=final_loads_b3,

                                init_dev=init_dev,
                                final_dev=final_dev_b3,

                                obj_value=obj_val_b3,
                                solve_time_sec=solve_time_b3,
                                mip_gap=mip_b3,
                                status_msg="SUCCESS",

                                # ---------------- RT ----------------
                                rt_mean_init_ms=init_mean_ms_rt,
                                rt_mean_final_ms=rt_mean_ms_b3,

                                rt_max_init_ms=rt_max_ms_init,
                                rt_max_final_ms=rt_max_ms_b3,

                                rt_p95_ms_init=rt_p95_ms_init,
                                rt_p95_ms_final=rt_p95_ms_b3,
                                delta_rt_p95_ms=rt_p95_ms_b3 - rt_p95_ms_init,

                                # ---------------- CTRL ----------------
                                ctrl_util_mean_init=lb_init["util_mean"],
                                ctrl_util_mean_final=lb_b3["util_mean"],

                                ctrl_util_max_init=lb_init["util_max"],
                                ctrl_util_max_final=lb_b3["util_max"],
                                delta_ctrl_util_max=_d(lb_b3["util_max"], lb_init["util_max"]),

                                ctrl_util_p95_init=lb_init["util_p95"],
                                ctrl_util_p95_final=lb_b3["util_p95"],

                                ctrl_util_std_final=lb_b3["util_std"],
                                ctrl_util_cov_final=lb_b3["util_cov"],

                                jain_load_init=lb_init["jain"],
                                jain_load_final=lb_b3["jain"],

                                ctrl_headroom_p50_final=lb_b3["head_p50"],
                                ctrl_headroom_p95_final=lb_b3["head_p95"],
                                ctrl_headroom_mean_final=lb_b3["head_mean"],

                                # ---------------- DELAYS ----------------
                                prop_mean_ms_final=prop_mean_ms_b3,
                                W_mean_ms_final=W_mean_ms_b3,

                                unstable_ctrls_final=len(rt_b3.get("unstable_controllers", [])),

                                # ---------------- LINK ----------------
                                link_util_mean_used_final=link_b3_stats["mean_used"],
                                link_util_max_final=link_b3_stats["max"],
                                link_util_p95_final=link_b3_stats["p95"],
                                violated_links_final=link_b3_stats["viol"],
                                excess_viol_final=link_b3_stats["excess"],

                                # ---------------- LOAD ----------------
                                rebalanced_load_total=_rebalanced_total(lam_init, final_loads_b3),

                                # ---------------- CONFIG ----------------
                                alpha=alpha,
                                beta=beta,
                                k_paths=k_path_count,
                                link_sens=sens,

                                # ---------------- MIG ----------------
                                mig_cost_components=mig_cost_comp_b3,
                                mig_dist_mean=mig_stats_b3["dist_mean"],
                                mig_dist_p95=mig_stats_b3["dist_p95"],

                                # ---------------- RT MAP ----------------
                                rt_init_map=T_init,
                                rt_final_map=rt_b3["T_final_ms_by_switch"],

                                # ---------------- DELAY BREAKDOWN ----------------
                                prop_delay_max_ms=rt_b3["prop_max_ms"],
                                queue_delay_max_ms=max(
                                    v for v in rt_b3["Wsys_by_ctrl"].values()
                                    if math.isfinite(v)
                                ),
                                sync_delay_ms=SYNC_DELAY_MS,

                                # ---------------- CTRL RT ----------------
                                ctrl_rt_mean_ms=0.0,
                                ctrl_rt_max_ms=0.0,
                                ctrl_rt_p95_ms=0.0,
                            )

                            # ======================================================
                            # LINK CSV
                            # ======================================================

                            write_edge_usage_csv(
                                out_csv=link_csv_file,
                                topology=topo_name,
                                run_index=RUN_INDEX,
                                phase="SM",
                                algo="B3",

                                G=G_run,
                                usage_routed=usage_b3,

                                edge_caps_uniform=edge_caps_uniform,
                                edge_caps_before_stress=edge_caps_calibrated,
                                edge_caps_after_stress=edge_caps_stressed,

                                paths_by_switch=paths_b3_switch,

                                usage_shortest_init=usage_shortest_init,
                                usage_routed_init=usage_init_routed,

                                ebc=ebc,
                                stressed_edges=stressed_edges,

                                alpha=alpha,
                                beta=beta,
                                link_sens=sens,
                                k_paths=k_path_count,
                                load_sens=0.0,
                                controller_sens=0.0,
                            )

                            # ======================================================
                            # SWITCH RT CSV
                            # ======================================================

                            write_switch_rt_csv(
                                switch_csv_file,
                                topo_name,
                                RUN_INDEX,
                                "B3",
                                "SM",

                                fa_b3,
                                paths_b3_switch,
                                chosen_paths_init,

                                {
                                    "resp_ms_by_switch": rt_b3["T_final_ms_by_switch"],
                                    "prop_ms_by_switch": rt_b3["prop_by_switch"],
                                    "solver_queue_ms_by_ctrl": rt_b3["Wsys_by_ctrl"],
                                    "queue_ms_by_ctrl": rt_b3["Wsys_by_ctrl"],

                                    "init_resp_ms_by_switch": T_init,
                                    "init_prop_ms_by_switch": prop_init,
                                    "init_solver_queue_ms_by_ctrl": W_init,
                                },

                                init_loads_by_switch=BASE_LOADS,
                                scaled_loads_by_switch=loads,
                                init_assign_by_switch=init_assign_cs,

                                load_scale_alpha="0",

                                alpha=alpha,
                                beta=beta,
                                link_sens=sens,
                                k_paths=k_path_count,
                                load_sens=0.0,
                                controller_sens=0.0,
                            )

                            # ======================================================
                            # CONTROLLER CSV
                            # ======================================================

                            W_b3_by_ctrl = rt_b3.get("Wsys_by_ctrl", {})

                            switch_W_final_b3 = {
                                s: W_b3_by_ctrl.get(fa_b3[s], 0.0)
                                for s in fa_b3
                            }

                            write_controller_csv_reuse_switch_logs(
                                out_csv=controller_csv_file,
                                topology=topo_name,
                                run_index=RUN_INDEX,
                                algo="B3",
                                phase="SM",
                                controllers=controllers,

                                capacities_init=BASE_CAPACITIES,
                                capacities_final=capacities,

                                loads_by_ctrl=fl_b3,
                                switch_to_ctrl=fa_b3,

                                switch_W_init=W_init,
                                switch_W_final=switch_W_final_b3,

                                switch_loads=loads,
                                capacity_threshold=CAPACITY_THRESHOLD,

                                alpha=alpha,
                                beta=beta,
                                link_sens=sens,
                                k_paths=k_path_count,
                                load_sens=0.0,
                                controller_sens=0.0,
                            )

# ==========================================================================================================================
                        # ===========Multi-commodity Flow-ARC==============
# ============================================================================================================================                        

                    current_algo = "MCF_ARC"
                    solve_start_MCF_arc = time.perf_counter()
                    print(f"🚀 ENTERING MCF ARC | run={RUN_INDEX}")

                    (
                        fa_mcf_arc,                  # final assignment
                        fl_mcf_arc,                  # loads per controller
                        obj_mcf_arc,                 # solver objective value
                        mig_mcf_arc,                 # migration count
                        usage_mcf_arc,               # ✅ FLOW-based link usage
                        rt_mcf_arc_solver,           # ✅ solver RT (flow-based)
                        mip_MCF_arc,                 # mip gap
                        mig_cost_arc,                # migration cost breakdown
                        paths_by_switch_final_arc,    # ✅ reconstructed paths for RT
                        status_arc
                    ) = run_migration_optimizer_integrated_mcf_arc(
                        G=G_run,
                        switches=switches,
                        controllers=controllers,
                        loads=loads,
                        capacities=capacities,
                        init_assign=init_assign_cs,

                        edge_caps_e=edge_caps,
                        msg_bits=MSG_BITS_PER_REQ,

                        objective_type=Objective,
                        topology_name=topo_name,

                        round_trip=True,
                        eta=1e-6,

                        # migration cost
                        Dcc=Dcc,
                        sync_per_ctrl_ms=SYNC_DELAY_MS,
                        cost_mode=ROUTING_MODE,

                        # RT baseline (same as PATH)
                        init_mean_rt_ms=init_mean_ms_rt,

                        rho_max=0.95,
                        pwl_segments=12,
                        allow_path_splitting=True,

                        alpha=alpha,
                        beta=beta,
                    )

                    solve_time_mcf_arc = time.perf_counter() - solve_start_MCF_arc
                    status_arc = str(status_arc)
                    missing_fa_mcf_arc = [s for s in switches if s not in (fa_mcf_arc or {})]
                    arc_failed = (
                        status_arc != "OPTIMAL"
                        and not status_arc.startswith("FEASIBLE_STATUS")
                    )
                    if arc_failed:
                        log_failure("MCF_ARC", status_arc, RUN_INDEX, topo_name, G_run, alpha, beta, k_path_count, sens)
                    elif missing_fa_mcf_arc:
                        log_failure(
                            "MCF_ARC",
                            f"INCOMPLETE_ASSIGNMENT_MCF_ARC_MISSING_{len(missing_fa_mcf_arc)}",
                            RUN_INDEX, topo_name, G_run, alpha, beta, k_path_count, sens
                        )
                    else:    
                        paths_mcf_arc = {
                            s: paths_by_switch_final_arc.get((s, fa_mcf_arc[s]), [])
                            for s in switches
                        }
                        # paths_mcf_arc_sc = {
                        #     (s, fa_mcf_arc[s]): paths_by_switch_final_arc.get((s, fa_mcf_arc[s]), [])
                        #     for s in switches
                        # }
                        # =========================
                        # STANDARD RT (UNIFORM)
                        # =========================
                        rt_mcf_std_arc = compute_response_metrics(
                            G_run,
                            fa_mcf_arc,
                            loads,
                            capacities,
                            paths_by_switch_final_arc,   # must come from ARC extractor
                            round_trip=True,
                            per_ctrl_ms=SYNC_DELAY_MS,
                        )

                        # =========================
                        # FLOW RT (SOLVER)
                        # =========================
                        rt_mcf_flow_arc = rt_mcf_arc_solver or {}
                        T_flow_arc = rt_mcf_flow_arc.get("T_ms_by_switch", {})

                        rt_pack_flow_arc = {"resp_by_switch": T_flow_arc}
                        rtp_flow_arc = _rt_pstats(rt_pack_flow_arc)
                        rt_max_ms_mcf_flow_arc  = max(T_flow_arc.values())                    
                        rt_mean_ms_mcf_flow_arc = (
                                        sum(T_flow_arc.values()) / len(T_flow_arc)
                                    ) if T_flow_arc else float("nan")
                        rt_p95_ms_mcf_flow_arc  = float(rtp_flow_arc["p95"]) if rtp_flow_arc["p95"] == rtp_flow_arc["p95"] else float("nan")
                        
                        # =========================
                        # LINK + LOAD
                        # =========================

                        link_mcf_arc_stats = _link_stats(usage_mcf_arc, edge_caps)
                        lb_mcf_arc = _ctrl_lb(fl_mcf_arc, capacities, usable_frac=1.0)

                        # =========================
                        # RT STATS (STANDARD)
                        # =========================
                        rtp_mcf_arc = _rt_pstats(rt_mcf_std_arc)

                        rt_mean_ms_mcf_arc = float(rt_mcf_std_arc.get("mean_resp", float("nan")))
                        rt_p95_ms_mcf_arc  = float(rtp_mcf_arc["p95"]) if rtp_mcf_arc["p95"] == rtp_mcf_arc["p95"] else float("nan")
                        rt_max_ms_mcf_arc  = float(rt_mcf_std_arc.get("max_resp", float("nan")))

                        prop_mean_ms_mcf_arc = _mean_prop_ms(rt_mcf_std_arc)
                        W_mean_ms_mcf_arc    = _mean_W_ms_over_switches(rt_mcf_std_arc, fa_mcf_arc)

                        queue_share_mcf_arc = (
                            W_mean_ms_mcf_arc / max(1e-9, rt_mean_ms_mcf_arc)
                        ) if math.isfinite(rt_mean_ms_mcf_arc) else float("nan")

                        # =========================
                        # MIGRATION STATS
                        # =========================
                        mig_stats_mcf_arc = _mig_stats(
                            G_run, init_assign_cs, fa_mcf_arc,
                            ROUTING_MODE,
                            rt_init=init_rt,
                            rt_final=rt_mcf_std_arc
                        )

                        mig_mcf_arc = int(mig_stats_mcf_arc.get("count", 0))

                        # =========================
                        # MIGRATION COST
                        # =========================

                        mig_cost_comp_mcf_arc = compute_migration_cost_components(
                            init_assign=init_assign_cs,
                            final_assign=fa_mcf_arc,
                            Dcc=Dcc,
                            rt_init=init_rt,
                            rt_final=rt_mcf_std_arc
                        )
                        print("MIG DEBUG:",
                        "num_mig=", mig_cost_comp_mcf_arc["num_mig"],
                        "cc=", mig_cost_comp_mcf_arc["cc_transfer"],
                        "delta_rt=", mig_cost_comp_mcf_arc["delta_rt"],
                        "total=", mig_cost_comp_mcf_arc["total"])
    
                        # =========================
                        # PLOT
                        # =========================
                        plot_assignments(
                            G_run, pos, switches, controllers,
                            init_assign_cs, fa_mcf_arc, loads, fl_mcf_arc,
                            topo_name, ALG_DIR("MCF_ARC"), capacities,
                            extra_title=f"MCF_ARC objective={Objective}",
                            file_tag=f"MCF_ARC_run{RUN_INDEX:03d}_topo{idx:02d}"
                        )

                        _assert_dict("link_summary_init", link_sum_init)
                        _assert_dict("link_summary_final_MCF_ARC", summarize_link_usage(usage_mcf_arc, edge_caps))



                        log_run_to_csv(
                            logs_dir=RESULTS_FOLDER,

                            algo="MCF_ARC",

                            run_index=RUN_INDEX,
                            topo=topo_name,
                            nodes=G_run.number_of_nodes(),

                            loads_by_switch=loads,
                            controller_set=controllers,
                            controller_caps=capacities,

                            init_assign=init_assign_cs,
                            final_assign=fa_mcf_arc,

                            init_loads_by_ctrl=dict(init_loads_by_ctrl),
                            final_loads_by_ctrl=fl_mcf_arc,

                            init_dev=init_dev,
                            final_dev=max(fl_mcf_arc.values()) - min(fl_mcf_arc.values()) if fl_mcf_arc else 0.0,

                            obj_value=obj_mcf_arc,
                            solve_time_sec=solve_time_mcf_arc,
                            mip_gap=mip_MCF_arc,
                            status_msg="SUCCESS",

                            # ---------------- RT (STANDARD) ----------------
                            rt_mean_init_ms=init_mean_ms_rt,
                            rt_mean_final_ms=rt_mean_ms_mcf_arc,

                            rt_max_init_ms=rt_max_ms_init,
                            rt_max_final_ms=rt_max_ms_mcf_arc,

                            rt_p95_ms_init=rt_p95_ms_init,
                            rt_p95_ms_final=rt_p95_ms_mcf_arc,
                            delta_rt_p95_ms=rt_p95_ms_mcf_arc - rt_p95_ms_init,

                            # ---------------- CTRL ----------------
                            ctrl_util_mean_init=lb_init["util_mean"],
                            ctrl_util_mean_final=lb_mcf_arc["util_mean"],

                            ctrl_util_max_init=lb_init["util_max"],
                            ctrl_util_max_final=lb_mcf_arc["util_max"],
                            delta_ctrl_util_max=_d(lb_mcf_arc["util_max"], lb_init["util_max"]),

                            ctrl_util_p95_init=lb_init["util_p95"],
                            ctrl_util_p95_final=lb_mcf_arc["util_p95"],

                            ctrl_util_std_final=lb_mcf_arc["util_std"],
                            ctrl_util_cov_final=lb_mcf_arc["util_cov"],

                            jain_load_init=lb_init["jain"],
                            jain_load_final=lb_mcf_arc["jain"],

                            ctrl_headroom_p50_final=lb_mcf_arc["head_p50"],
                            ctrl_headroom_p95_final=lb_mcf_arc["head_p95"],
                            ctrl_headroom_mean_final=lb_mcf_arc["head_mean"],

                            # ---------------- DELAYS ----------------
                            prop_mean_ms_final=prop_mean_ms_mcf_arc,
                            W_mean_ms_final=W_mean_ms_mcf_arc,

                            unstable_ctrls_final=len(rt_mcf_std_arc.get("unstable_controllers", [])),

                            # ---------------- LINK ----------------
                            link_util_mean_used_final=link_mcf_arc_stats["mean_used"],
                            link_util_max_final=link_mcf_arc_stats["max"],
                            link_util_p95_final=link_mcf_arc_stats["p95"],
                            violated_links_final=link_mcf_arc_stats["viol"],
                            excess_viol_final=link_mcf_arc_stats["excess"],

                            # ---------------- LOAD ----------------
                            rebalanced_load_total=_rebalanced_total(lam_init, fl_mcf_arc),

                            # ---------------- CONFIG ----------------
                            alpha=alpha,
                            beta=beta,
                            k_paths=k_path_count,
                            link_sens=sens,

                            # ---------------- MIG ----------------
                            mig_cost_components=mig_cost_comp_mcf_arc,
                            mig_dist_mean=mig_stats_mcf_arc["dist_mean"],
                            mig_dist_p95=mig_stats_mcf_arc["dist_p95"],

                            # ---------------- RT MAP ----------------
                            rt_init_map=T_init,
                            rt_final_map=rt_mcf_std_arc["T_final_ms_by_switch"],

                            # ---------------- DELAY BREAKDOWN ----------------
                            prop_delay_max_ms=rt_mcf_std_arc["prop_max_ms"],
                            queue_delay_max_ms=max(
                                v for v in rt_mcf_std_arc["Wsys_by_ctrl"].values()
                                if math.isfinite(v)
                            ),
                            sync_delay_ms=SYNC_DELAY_MS,

                            # ---------------- CTRL RT ----------------
                            ctrl_rt_mean_ms=rt_mean_ms_mcf_flow_arc,   # 🔥 FLOW RT
                            ctrl_rt_max_ms=rt_max_ms_mcf_flow_arc,
                            ctrl_rt_p95_ms=rt_p95_ms_mcf_flow_arc,
                        )

                        # ============================================================
                        # EDGE USAGE CSV — MCF ARC
                        # ============================================================
                        write_edge_usage_csv(
                            out_csv=link_csv_file,
                            topology=topo_name,
                            run_index=RUN_INDEX,
                            phase="SM",
                            algo="MCF_ARC",

                            G=G_run,
                            usage_routed=usage_mcf_arc,   # ✅ from optimizer

                            edge_caps_uniform=edge_caps_uniform,
                            edge_caps_before_stress=edge_caps_calibrated,
                            edge_caps_after_stress=edge_caps_stressed,

                            paths_by_switch=paths_mcf_arc,

                            usage_shortest_init=usage_shortest_init,
                            usage_routed_init=usage_init_routed,

                            ebc=ebc,
                            stressed_edges=stressed_edges,

                            alpha=alpha,
                            beta=beta,
                            link_sens=sens,
                            k_paths=k_path_count,
                            load_sens=0.0,
                            controller_sens=0.0,
                        )

                        # ============================================================
                        # SWITCH RT CSV — MCF ARC
                        # ============================================================
                        write_switch_rt_csv(
                            switch_csv_file,
                            topo_name,
                            RUN_INDEX,
                            "MCF_ARC",
                            "SM",

                            # FINAL assignment + paths
                            fa_mcf_arc,
                            paths_mcf_arc,
                            chosen_paths_init,   # ✅ ADD THIS

                            # RT DATA (ARC STANDARD RT)
                            {
                                # FINAL
                                "resp_ms_by_switch": rt_mcf_std_arc["T_final_ms_by_switch"],
                                "prop_ms_by_switch": rt_mcf_std_arc["prop_by_switch"],
                                "solver_queue_ms_by_ctrl": rt_mcf_arc_solver["queue_ms_by_ctrl"],
                                "queue_ms_by_ctrl": rt_mcf_std_arc["Wsys_by_ctrl"],

                                # INIT
                                "init_resp_ms_by_switch": T_init,
                                "init_prop_ms_by_switch": prop_init,
                                "init_solver_queue_ms_by_ctrl": W_init,
                            },

                            init_loads_by_switch=BASE_LOADS,
                            scaled_loads_by_switch=loads,
                            init_assign_by_switch=init_assign_cs,

                            load_scale_alpha="0",

                            alpha=alpha,
                            beta=beta,
                            link_sens=sens,
                            k_paths=k_path_count,
                            load_sens=0.0,
                            controller_sens=0.0,
                        )

                        # ============================================================
                        # CONTROLLER CSV — MCF ARC
                        # ============================================================
                        W_mcf_arc_by_ctrl = rt_mcf_std_arc.get("Wsys_by_ctrl", {})

                        switch_W_final_mcf_arc = {
                            s: W_mcf_arc_by_ctrl.get(fa_mcf_arc[s], 0.0)
                            for s in fa_mcf_arc
                        }

                        write_controller_csv_reuse_switch_logs(
                            out_csv=controller_csv_file,
                            topology=topo_name,
                            run_index=RUN_INDEX,
                            algo="MCF_ARC",
                            phase="SM",
                            controllers=controllers,

                            capacities_init=BASE_CAPACITIES,
                            capacities_final=capacities,

                            loads_by_ctrl=fl_mcf_arc,
                            switch_to_ctrl=fa_mcf_arc,

                            switch_W_init=W_init,
                            switch_W_final=switch_W_final_mcf_arc,

                            switch_loads=loads,
                            capacity_threshold=CAPACITY_THRESHOLD,

                            alpha=alpha,
                            beta=beta,
                            link_sens=sens,
                            k_paths=k_path_count,
                            load_sens=0.0,
                            controller_sens=0.0,
                        )


                # ✅ correct
            except Exception as e:
                print(f"❌ Error on run={RUN_INDEX} topo_idx={idx}: {type(e).__name__}({e!r})")
                traceback.print_exc()
                if "G_run" in locals() and "RUN_INDEX" in locals():
                    log_failure(
                        current_algo,
                        f"EXCEPTION_{type(e).__name__}_{str(e)[:80]}",
                        RUN_INDEX,
                        topo_name,
                        G_run,
                        alpha,
                        beta,
                        k_path_count,
                        sens,
                    )
                continue




    print("✅ Done.")

if __name__ == "__main__":
    main()
