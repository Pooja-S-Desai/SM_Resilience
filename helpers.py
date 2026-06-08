

# helpers.py
import os
import random
from typing import Dict
import time
import math
import networkx as nx
import hashlib




# ----- Configure master list of run seeds -----
# =========================================================================================================

_MASTER_SEED = 40       # change if you want a different seed list
_NUM_RUNS    = 5

random.seed(_MASTER_SEED)
_SEED_LIST = [random.randint(1, 10**9) for _ in range(_NUM_RUNS)]

def get_seed(run_idx: int) -> int:
    """Return the canonical seed for the k-th run (0..N-1)."""
    return _SEED_LIST[run_idx % len(_SEED_LIST)]

def subseed(run_seed: int, label: str) -> int:
    h = hashlib.blake2b(f"{label}|{run_seed}".encode(), digest_size=8).hexdigest()
    return (int(h, 16) & 0x7fffffff) or 1

# # ----- Deterministic sub-seeds per run for independent streams -----
# def subseed(run_seed: int, label: str) -> int:
#     """Derive a stable sub-seed from a run seed and a label ('loads','caps',...)."""
#     return (hash((label, run_seed)) & 0x7fffffff) or 1  # avoid 0

def get_run_seeds(run_idx: int) -> Dict[str, int]:
    """
    Convenience: return a dict of common sub-seeds for this run.
    You can add more labels if needed (e.g., 'plot', 'shuffle', ...).
    """
    s = get_seed(run_idx)
    return {
        "run":   s,
        "loads": subseed(s, "loads"),
        "caps_menu": subseed(s, "caps_menu"),
        "caps_nodes": subseed(s, "caps_nodes"),
        "link_caps": subseed(s, "link_caps"),   # ← ADD THIS
    }

# =====================================================================================================





# =============================
# CORE SYSTEM PATHS
# =============================

TOPOLOGY_FOLDER = './topologyzoo-master/'
TOPOLOGY_LIST_FILE = './topology_list.txt'
RESULTS_FOLDER = './results'
PATH_CACHE = True
# =============================
# SYSTEM CONSTANTS
# =============================

PROPAGATION_SPEED = 200_000_000   # meters/sec
LATENCY_THRESHOLD_SEC = 0.003     # 3 ms

CONTROLLER_CAPACITY = 3000
TIME_LIMIT = 500                  # solver timeout (seconds)
PENALTY = 1e6

# =============================
# LOAD / TRAFFIC
# =============================

MAX_LOAD = 200
MIN_LOAD = 100
MANUAL_LOAD_MEAN = 150

MSG_BITS_PER_REQ = 128
LINK_BUDGET_BITS = 100600  # ✅ choose and keep ONLY one
CONTROL_SHARE = 0.10

# =============================
# CAPACITY / QUEUE MODEL
# =============================

CAPACITY_THRESHOLD = 0.8   # M/M/1 stability threshold

# =============================
# SYNC MODEL
# =============================

SYNC_MSG_BITS = 64000.0
SYNC_UPDATES_PER_SEC = 1.0

# =============================
# ROUTING / PATH SETTINGS
# =============================

ROUTING_MODE = "weight"   # "hops" or "weight"
PATH_CACHE = 1
PENALTY_PATH_ALPHA = 2.0
# =============================
# OBJECTIVES
# =============================

SP_OBJECTIVE = "maxmin"
MCF_OBJECTIVE = "maxmin"
RT_OBJECTIVE = "maxmin"

# =============================
# LATENCY MODEL
# =============================

LATENCY_VARIANT = "prop_mm1_sync"   # "prop", "prop_mm1", "prop_mm1_sync"
SYNC_MODE = "steiner"               # "none", "mst", "steiner"
SYNC_PHI = 0.02

# =============================
# MIGRATION WEIGHTS
# =============================
INITIAL_ASSUMED_CONTROLLERS_FOR_CAP = 0.10
CAPACITY_THRESHOLD_INITIAL=0.8

MIG_W_MIG = 1.0
MIG_W_DELTA = 1.0
MIG_W_CC = 1.0
MIG_W_STEINER = 1.0
MIG_W_RT = 1.0

# =============================
# STRESS SETTINGS
# =============================

# Load scaling (for stressed switches)
LOAD_SCALES = [1.0]

# Fraction of links pushed near overload
REVISED_BW_FRACTION = [0.10]

# Core edge selection (EBC-based)
CORE_EDGE_FRACTION = 0.10

#Objective
Objective="maxmin" #choices=["maxmin","min_max_util","min_sum_util","variance","min_dev", "migration_cost"])

# Controller degradation factor
CONTROLLER_SCALE_FACTOR = 0.0

# Stress identification mode
CORE_EXPERIMENT_MODE = "structural"   # "structural" or "congestion"

# Threshold for congestion detection
CORE_UTIL_THRESHOLD = 0.8
FIBER_SEC_PER_KM = 5e-6  # ≈ 2e5 km/s
PROPAGATION_SPEED = 200_000_000 
# =============================
# EXPERIMENT GRID
# =============================
CAPACITY_THRESHOLD_mm1=0.8
K_VALUES = [5]

LINK_SENS_VALUES = [0.00, 0.05, 0.10]

ALPHA_VALUES = [0.5]
def experiment_configs():

    for k in K_VALUES:
        for sens in LINK_SENS_VALUES:
            for alpha in ALPHA_VALUES:

                yield {
                    "k": k,
                    "sens": sens,
                    "alpha": alpha,
                    "beta": 1.0 - alpha
                }

# # Capacity reduction on core edges
# CORE_EDGE_EXTRA_STRESS_FACTOR = 0.10
# =============================================================================================

def ensure_dir(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def get_topology_name_from_path(path):
    """Extracts topology name without extension from file path."""
    return os.path.splitext(os.path.basename(path))[0]

def print_controller_stats(capacities, loads_init=None, loads_final=None, threshold=CAPACITY_THRESHOLD):
    """
    Prints controller capacity, initial and final loads, utilization %, and overload status for both phases.
    """
    print("\n--- Controller Stats ---")
    print(f"{'Controller':>10} | {'Capacity':>8} | {'Init Load':>10} | {'Init %':>7} | {'Init Stat':>11} | {'Final Load':>11} | {'Final %':>8} | {'Final Stat':>12}")
    print("-" * 95)

    for c in sorted(capacities):
        cap = capacities[c]
        init = loads_init.get(c, 0) if loads_init else 0
        final = loads_final.get(c, 0) if loads_final else 0

        init_util = (init / cap) * 100 if cap > 0 else 0
        final_util = (final / cap) * 100 if cap > 0 else 0

        init_status = "Overloaded" if init > threshold * cap else "Underloaded"
        final_status = "Overloaded" if final > threshold * cap else "Underloaded"

        print(f"{c:>10} | {cap:>8} | {init:>10.1f} | {init_util:>7.2f}% | {init_status:>11} | {final:>11.1f} | {final_util:>8.2f}% | {final_status:>12}")


def set_controller_capacity(value):
    global CONTROLLER_CAPACITY
    CONTROLLER_CAPACITY = int(value)

def _percentile(xs, p):
    cleaned = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
            if math.isfinite(xf):
                cleaned.append(xf)
        except:
            continue

    if not cleaned:
        return float("nan")

    cleaned.sort()

    k = (len(cleaned) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)

    if f == c:
        return cleaned[int(k)]

    return cleaned[f] * (c - k) + cleaned[c] * (k - f)


def _rt_pstats(rt_dict):
    """
    Return {p50, p95} in MILLISECONDS from resp_by_switch.
    NOTE: compute_response_metrics() already outputs ms.
    """
    if not isinstance(rt_dict, dict):
        return {"p50": float("nan"), "p95": float("nan")}

    raw_vals = rt_dict.get("resp_by_switch", {})

    vals = [
        float(v) for v in raw_vals.values()
        if v is not None and math.isfinite(float(v))
    ]

    if not vals:
        return {"p50": float("nan"), "p95": float("nan")}

    return {
        "p50": _percentile(vals, 50),
        "p95": _percentile(vals, 95),
    }


    

def _mean_prop_ms(rt_dict):
    """
    Mean propagation delay in MILLISECONDS.
    NOTE: rt_metrics already outputs prop_by_switch in ms.
    """
    vals = rt_dict.get("prop_by_switch", {}) if isinstance(rt_dict, dict) else {}
    if not vals:
        return 0.0
    return (sum(vals.values()) / max(1, len(vals)))


def _mean_W_ms_over_switches(rt_dict, assignment):
    """
    Average controller queueing (Wsys) seen by switches, in MILLISECONDS.
    NOTE: rt_metrics already outputs Wsys_by_ctrl in ms.
    """
    Wc = rt_dict.get("Wsys_by_ctrl", {}) if isinstance(rt_dict, dict) else {}
    if not Wc or not assignment:
        return 0.0
    ws = [Wc.get(assignment[s], float("nan")) for s in assignment]
    ws = [w for w in ws if w is not None and not math.isinf(float(w))]
    return (sum(ws) / max(1, len(ws))) if ws else 0.0

def _link_stats(usage_e, budget_e):
    """Return p95, max, mean_used, violations, excess, used, total, frac_used."""
    utils, used_utils = [], []
    violations, excess, used = 0, 0.0, 0
    for e, used_bits in (usage_e or {}).items():
        cap = float(budget_e.get(e, 0.0))
        if cap <= 0: 
            continue
        u = float(used_bits) / cap
        utils.append(u)
        if used_bits > 0:
            used += 1
            used_utils.append(u)
        if u > 0.8 + 1e-12:
            violations += 1
            excess += (u - 0.8)
    total = len(budget_e)
    return {
        "p95": _percentile(utils, 95) if utils else 0.0,
        "max": max(utils) if utils else 0.0,
        "mean_used": (sum(used_utils)/len(used_utils)) if used_utils else 0.0,
        "viol": violations, "excess": excess,
        "used": used, "total": total, "frac_used": (used/total) if total else 0.0
    }

def _ctrl_lb(lam_by_ctrl, capacities, usable_frac):
    """Controller load-balance stats + rho/mu dicts."""
    mu = {c: float(capacities.get(c, 0.0)) * float(usable_frac) for c in capacities}
    rho = {}
    for c in capacities:
        mc = mu[c]
        rho[c] = (float(lam_by_ctrl.get(c, 0.0)) / mc) if mc > 0 else float("inf")

    lam_vals = [float(lam_by_ctrl.get(c, 0.0)) for c in capacities]
    rho_vals = [r for r in rho.values() if math.isfinite(r)]

    def _jain(vals):
        if not vals:
            return 0.0
        s = sum(vals); s2 = sum(v*v for v in vals); n = len(vals)
        return (s*s)/(n*s2) if (n and s2 > 0) else 0.0

    util_max  = max(rho_vals) if rho_vals else 0.0
    util_mean = (sum(rho_vals)/len(rho_vals)) if rho_vals else 0.0
    util_std  = (sum((r - util_mean)**2 for r in rho_vals)/len(rho_vals))**0.5 if rho_vals else 0.0
    util_cov  = (util_std / util_mean) if util_mean > 0 else 0.0
    headroom  = [1.0 - r for r in rho_vals] if rho_vals else []
    inband    = [0.6 <= r <= 0.9 for r in rho_vals] if rho_vals else []

    stats = {
        "util_max": util_max,
        "util_p95": _percentile(rho_vals, 95) if rho_vals else 0.0,
        "util_mean": util_mean,
        "util_std": util_std,
        "util_cov": util_cov,
        "jain": _jain(lam_vals),
        "final_dev_load": (max(lam_vals) - min(lam_vals)) if lam_vals else 0.0,
        "final_dev_util": (max(rho_vals) - min(rho_vals)) if rho_vals else 0.0,
        "head_p50": _percentile(headroom, 50) if headroom else 0.0,
        "head_mean": (sum(headroom) / len(headroom)) if headroom else 0.0,
        "head_p95": _percentile(headroom, 95) if headroom else 0.0,
        "frac_in_band": (sum(1 for b in inband if b)/len(inband)) if inband else 0.0,
        "mu": mu,
        "rho": rho,
    }
    return stats


# util_max / util_p95 / util_mean / util_std / util_cov: how busy controllers are.

# jain: fairness on controller loads (not utilization).

# final_dev_load: (max load − min load).

# final_dev_util: (max util − min util).

# head_p50 / head_p95: med/95th percentile headroom.

# frac_in_band: fraction of controllers in [0.6, 0.9] utilization.

# mu: effective capacities per controller.

# rho: utilizations per controller.


def _rebalanced_total(lam_init, lam_final):
    keys = set(lam_init) | set(lam_final)
    return 0.5 * sum(abs(float(lam_final.get(k,0.0)) - float(lam_init.get(k,0.0))) for k in keys)

def _mig_stats(G, init_assign, final_assign, cost_mode, rt_init=None, rt_final=None):
    """Distances between old/new controllers & ΔRT for migrated switches only."""
    migrated = [s for s in init_assign if final_assign.get(s) != init_assign.get(s)]
    weight_key = None if cost_mode == "hops" else "weight"
    dists = []
    for s in migrated:
        c0, c1 = init_assign[s], final_assign[s]
        try:
            d = nx.shortest_path_length(G.to_undirected(), c0, c1, weight=weight_key)
        except Exception:
            d = 0.0
        dists.append(float(d))

    p95_d = _percentile(dists, 95) if dists else 0.0
    mean_d = (sum(dists)/len(dists)) if dists else 0.0

    p95_drt_ms, mean_drt_ms = 0.0, 0.0
    if rt_init and rt_final:
        deltas = []
        for s in migrated:
            # NOTE: resp_by_switch is already in MILLISECONDS
            t0 = float(rt_init.get("resp_by_switch", {}).get(s, 0.0))
            t1 = float(rt_final.get("resp_by_switch", {}).get(s, 0.0))
            d  = max(0.0, t1 - t0)   # <-- NO *1e3
            deltas.append(d)
        if deltas:
            p95_drt_ms = _percentile(deltas, 95)
            mean_drt_ms = sum(deltas)/len(deltas)

    return {
        "migrations": len(migrated),
        "dist_mean": mean_d,
        "dist_p95": p95_d,
        "delta_rt_p95_ms": p95_drt_ms,
        "delta_rt_mean_ms": mean_drt_ms,
    }

#migrations: # of switches that changed controller.

# dist_mean / dist_p95: how far controllers changed (in hops or weight).

# # delta_rt_mean_ms / delta_rt_p95_ms: how much worse RT got (if at all) for migrated switches. 
# def compute_init_rt_ms_by_switch_from_kpaths(
#     switches, controllers, init_assign, loads, capacities,
#     sc_kpath_cost_ms, Kset_by_pair,
#     *,
#     global_threshold: float,
#     sync_per_ctrl_ms: float = 0.0,
#     rho_max: float = 0.95,
# ):
#     # controller loads under init mapping
#     lam = {c: 0.0 for c in controllers}
#     for s in switches:
#         c0 = init_assign.get(s)
#         if c0 in lam:
#             lam[c0] += float(loads.get(s, 0.0))

#     # service rates
#     mu = {c: float(capacities[c]) * float(global_threshold) for c in controllers}

#     # queueing W (ms) under init mapping
#     W_ms = {}
#     for c in controllers:
#         muc = mu[c]
#         lamc = min(lam[c], rho_max * muc)
#         denom = max(1e-9, muc - lamc)
#         W_ms[c] = 1000.0 / denom

#     # propagation per switch for init: min path cost to its init controller
#     prop_ms = {}
#     for s in switches:
#         c0 = init_assign.get(s)
#         ks = Kset_by_pair.get((s, c0), [])
#         if not ks:
#             prop_ms[s] = 0.0
#         else:
#             prop_ms[s] = min(float(sc_kpath_cost_ms[(s, c0, k)]) for k in ks)

#     # total RT per switch (ms)
#     T_init = {}
#     for s in switches:
#         c0 = init_assign.get(s)
#         T_init[s] = float(prop_ms.get(s, 0.0)) + float(W_ms.get(c0, 0.0)) + float(sync_per_ctrl_ms)

#     mean_init = sum(T_init.values()) / max(1, len(T_init))
#     return T_init, prop_ms, W_ms, mean_init


def build_capacity_menu_and_node_caps(
    G,
    *,
    pivot_capacity: int,
    seed_menu: int,
    seed_nodes: int,
    spread: float = 0.2,
    num_options: int = 10,
):
    # ---- capacity menu (local RNG, deterministic) ----
    rnd_menu = random.Random(int(seed_menu))
    lower = int(pivot_capacity)
    upper = int(pivot_capacity * (1 + float(spread)))

    capacity_options = sorted(set(rnd_menu.randint(lower, upper) for _ in range(int(num_options))))
    if not capacity_options:
        capacity_options = [lower]

    # ---- per-node capacity assignment (local RNG, deterministic) ----
    rnd_nodes = random.Random(int(seed_nodes))
    node_list = sorted(G.nodes())          # CRITICAL: stable ordering across both versions

    node_capacities = {v: rnd_nodes.choice(capacity_options) for v in node_list}

    return capacity_options, node_capacities
