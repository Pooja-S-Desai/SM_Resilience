# =========================
# RT METRICS (ALL RETURNS IN MILLISECONDS; NO SECONDS ANYWHERE)
# =========================
from typing import Dict, Tuple, List, Optional
import math
from helpers import CAPACITY_THRESHOLD_mm1, FIBER_SEC_PER_KM


# -------------------------------------------------
# Core conversion policy (meters -> ms)
# -------------------------------------------------
# If FIBER_SEC_PER_KM is seconds per km, then:
#   1 meter = 0.001 km
#   time_sec_per_meter = FIBER_SEC_PER_KM / 1000
#   time_ms_per_meter  = (FIBER_SEC_PER_KM / 1000) * 1000 = FIBER_SEC_PER_KM
#
# So numerically:
MS_PER_METER = float(FIBER_SEC_PER_KM)  # correct given your constant definition


# -------------------------------------------------
# Propagation helpers (ALL IN MILLISECONDS)
# -------------------------------------------------
def edge_latency_ms(G, u, v) -> float:
    """
    One-way propagation latency (ms) for edge (u,v).

    Priority:
      1) edge['latency_sec'] -> convert to ms
      2) edge['weight'] interpreted as METERS -> ms using MS_PER_METER
      3) fallback: 1 meter-equivalent
    """
    data = G[u][v]
    if "latency_sec" in data:
        return float(data["latency_sec"]) * 1000.0
    if "weight" in data:
        return float(data["weight"]) * MS_PER_METER
    return 1.0 * MS_PER_METER


def path_latency_ms(G, path: List[int]) -> float:
    """Sum one-way propagation latency (ms) over the path."""
    if not path or len(path) < 2:
        return 0.0
    return sum(edge_latency_ms(G, path[i], path[i + 1]) for i in range(len(path) - 1))


# -------------------------------------------------
# RT Metrics (Shortest-path based) — RETURNS MILLISECONDS
# -------------------------------------------------
def compute_response_metrics(
    G,
    assignment: Dict[int, int],               # switch -> controller
    loads: Dict[int, float],                  # λ_s (req/s)
    capacities: Dict[int, float],             # μ_c raw (req/s)
    paths: Dict[Tuple[int, int], List[int]],  # shortest paths
    round_trip: bool = True,
    per_ctrl_ms: Optional[object] = None,     # ms (scalar or dict), additive constant delay
):
    """
    Total response time per switch:
      RT_ms = (2-way propagation_ms) + (M/M/1 queueing_ms) + (per_ctrl_ms)

    Returns ALL TIME values in MILLISECONDS.
    Non-time metrics keep natural units:
      - lambda_by_ctrl: req/s
      - mu_by_ctrl: req/s
      - rho_by_ctrl: unitless
    """

    # (1) Aggregate arrivals per controller: λ_c
    lam = {c: 0.0 for c in capacities}
    for s, c in assignment.items():
        lam[c] += float(loads.get(s, 0.0))

    # (2) Effective service rate: μ'_c
    mu = {c: float(capacities[c]) * float(CAPACITY_THRESHOLD_mm1) for c in capacities}

    # (3) M/M/1 system time (in ms): W_ms = 1000 / (μ - λ)
    Wc_ms = {}
    rho = {}
    unstable = set()

    for c in capacities:
        denom = mu[c] - lam[c]
        if denom <= 0.0 or mu[c] <= 0.0:
            Wc_ms[c] = math.inf
            rho[c] = math.inf
            unstable.add(c)
        else:
            rho[c] = lam[c] / mu[c]
            Wc_ms[c] = 1000.0 / denom



    # (5) Per-switch propagation + response (ms)
    prop_ms_by_switch = {}
    resp_ms_by_switch = {}
    rtfactor = 2.0 if round_trip else 1.0

    for s, c in assignment.items():
        p = paths.get((s, c), [])
        one_way_ms = path_latency_ms(G, p) if p else 0.0
        prop_ms = rtfactor * one_way_ms

        prop_ms_by_switch[s] = prop_ms
        extra_ms = per_ctrl_ms
        resp_ms_by_switch[s] = prop_ms + Wc_ms.get(c, math.inf) + extra_ms

    # (6) Aggregate stats (ms)
    vals = list(resp_ms_by_switch.values())
    if not vals:
        mean_T = mad_T = max_T = 0.0
    else:
        finite_vals = [v for v in vals if math.isfinite(v)]
        if finite_vals:
            mean_T = sum(finite_vals) / len(finite_vals)
            mad_T = sum(abs(x - mean_T) for x in finite_vals) / len(finite_vals)
            max_T = max(vals)  # keep inf if present
        else:
            mean_T = mad_T = max_T = math.inf
    # (6A) Max propagation delay across switches
    if prop_ms_by_switch:
        max_prop_ms = max(prop_ms_by_switch.values())
    else:
        max_prop_ms = 0.0
    return {
        "lambda_by_ctrl": lam,                 # req/s
        "mu_by_ctrl": mu,                      # req/s
        "rho_by_ctrl": rho,                    # unitless
        "Wsys_by_ctrl": Wc_ms,                 # ms
        "prop_by_switch": prop_ms_by_switch,   # ms
        "T_final_ms_by_switch": resp_ms_by_switch,   # ms
        "init_mean_rt_ms": mean_T,                   # ms
        "mad_resp": mad_T,                     # ms
        "max_resp": max_T, 
        "prop_max_ms": max_prop_ms,                    # ms
        "unstable_controllers": sorted(unstable),
    }


# -------------------------------------------------
# Controller-side RT only (M/M/1) — RETURNS MILLISECONDS
# -------------------------------------------------
def compute_controller_side_rt_only(
    lambda_by_ctrl: Dict[int, float],     # λ_i req/s
    mu_by_ctrl: Dict[int, float],         # μ_i req/s (effective)
):
    """
    Controller-side (queueing-only) metrics in MILLISECONDS.

    Per controller mean system time:
      gamma_i_ms = 1000 / (μ_i - λ_i)

    tau_i = λ_i / (μ_i - λ_i)  (dimensionless)

    Network-wide mean controller-side RT (request-weighted):
      gamma_bar_ctrl_ms = 1000 * (Σ tau_i) / (Σ λ_i)
    """
    eps = 1e-9
    gamma_by_ctrl_ms = {}
    tau_by_ctrl = {}
    unstable = []

    total_lambda = sum(float(v) for v in lambda_by_ctrl.values())

    DENOM_FLOOR = 1e-2  # req/s floor for reporting safety

    for c in mu_by_ctrl:
        lam = float(lambda_by_ctrl.get(c, 0.0))
        mu = float(mu_by_ctrl.get(c, 0.0))

        denom = mu - lam
        if mu <= 0.0 or denom <= DENOM_FLOOR:
            denom_eff = DENOM_FLOOR
            unstable.append(c)
        else:
            denom_eff = denom

        gamma_by_ctrl_ms[c] = 1000.0 / denom_eff
        tau_by_ctrl[c] = lam / denom_eff  # dimensionless

    sum_tau = sum(float(v) for v in tau_by_ctrl.values())
    gamma_bar_ctrl_ms = 1000.0 * sum_tau / max(eps, total_lambda)

    return {
        "gamma_by_ctrl": gamma_by_ctrl_ms,     # ms
        "tau_by_ctrl": tau_by_ctrl,            # dimensionless
        "gamma_bar_ctrl": gamma_bar_ctrl_ms,   # ms
        "total_lambda": total_lambda,
        "unstable_controllers": sorted(unstable),
    }


# -------------------------------------------------
# Flow-based usage + RT (propagation from flows) — RETURNS MILLISECONDS
# -------------------------------------------------
def flows_to_usage_and_rtt(
    G,
    f,                                # f[s,c,u,v] = bits/s on arc (u,v)
    commodity_pairs,                  # list of (s,c) with flow vars (usually s != c)
    loads, capacities,                # λ_s and μ_c (req/s)
    msg_bits: float,                  # bits per request
    round_trip: bool = True,
    *,
    assignment_pairs=None,            # list of ALL (s,c) assignments
    rho_max: float = 0.95,            # soft clamp for reporting
    per_ctrl_ms=None,                 # ms (scalar or dict), additive constant delay
):
    """
    Returns:
      usage_e: undirected edge -> aggregate flow (bits/s)
      rt_metrics: ALL TIME values in MILLISECONDS (propagation derived from flows)
    """

    def _undir(u, v):
        return (u, v) if u < v else (v, u)

    UG = G.to_undirected()

    # (A) Edge usage (bits/s), sum both directions
    usage_e = {}
    for u, v in UG.edges():
        e = _undir(u, v)
        total = 0.0
        for (s, c) in commodity_pairs:
            if (s, c, u, v) in f:
                total += f[s, c, u, v].X
            if (s, c, v, u) in f:
                total += f[s, c, v, u].X
        usage_e[e] = total

    # (B) Controller arrivals λ_c (req/s)
    lam = {c: 0.0 for c in capacities}
    if assignment_pairs is None:
        assignment_pairs = commodity_pairs[:]
    for (s, c) in assignment_pairs:
        lam[c] += float(loads.get(s, 0.0))

    # (C) Effective service μ'_c (req/s)
    mu = {c: float(capacities[c]) * float(CAPACITY_THRESHOLD_mm1) for c in capacities}

    # Clamp λ for reporting
    lam_clamped = {}
    for c in capacities:
        lam_clamped[c] = min(lam[c], rho_max * mu[c]) if mu[c] > 0.0 else lam[c]

    # Queueing in ms: W_ms = 1000/(μ - λ_clamped)
    Wc_ms = {}
    rho = {}
    unstable = set()
    for c in capacities:
        if mu[c] <= 0.0 or lam[c] >= mu[c]:
            unstable.add(c)

        denom = mu[c] - lam_clamped[c]
        if denom <= 0.0 or mu[c] <= 0.0:
            Wc_ms[c] = float("inf")
            rho[c] = float("inf")
        else:
            rho[c] = lam_clamped[c] / mu[c]
            Wc_ms[c] = 1000.0 / denom

    # (D) Arc propagation latency (ms)
    def _arc_latency_ms(G, u, v):
        d = G[u][v]
        if "latency_sec" in d:
            return float(d["latency_sec"]) * 1000.0
        if "weight" in d:
            return float(d["weight"]) * MS_PER_METER
        return 1.0 * MS_PER_METER


    prop_ms_by_switch = {}
    resp_ms_by_switch = {}

    commodity_set = set(commodity_pairs)
    rtfactor = 2.0 if round_trip else 1.0

    # Pre-list edges once (avoid rebuilding each loop)
    edges_list = list(UG.edges())

    for (s, c) in assignment_pairs:
        d_bits = float(loads.get(s, 0.0)) * float(msg_bits)  # bits/s

        if d_bits <= 0.0:
            prop_ms = 0.0
        elif (s, c) not in commodity_set:
            prop_ms = 0.0  # self-hosted or not modeled as commodity
        else:
            # Flow-weighted one-way propagation in ms:
            # one_way_ms = (Σ latency_ms(u,v) * flow(u,v)) / d_bits
            numer = 0.0
            for (u, v) in edges_list:
                if (s, c, u, v) in f:
                    numer += _arc_latency_ms(G, u, v) * f[s, c, u, v].X
                if (s, c, v, u) in f:
                    numer += _arc_latency_ms(G, v, u) * f[s, c, v, u].X

            one_way_ms = numer / d_bits
            prop_ms = rtfactor * one_way_ms

        extra_ms = per_ctrl_ms
        W_ms = Wc_ms.get(c, float("inf"))

        prop_ms_by_switch[s] = prop_ms
        resp_ms_by_switch[s] = prop_ms + W_ms + extra_ms

    # (E) Aggregate stats (ms)
    vals = list(resp_ms_by_switch.values())
    if not vals:
        mean_T = mad_T = max_T = 0.0
    else:
        finite_vals = [v for v in vals if math.isfinite(v)]
        if finite_vals:
            mean_T = sum(finite_vals) / len(finite_vals)
            mad_T = sum(abs(x - mean_T) for x in finite_vals) / len(finite_vals)
            max_T = max(vals)
        else:
            mean_T = mad_T = max_T = float("inf")

    rt_metrics = {
        "lambda_by_ctrl": lam,                 # req/s
        "lambda_clamped": lam_clamped,         # req/s
        "mu_by_ctrl": mu,                      # req/s
        "rho_by_ctrl": rho,                    # unitless
        "Wsys_by_ctrl": Wc_ms,                 # ms
        "prop_by_switch": prop_ms_by_switch,   # ms
        "resp_by_switch": resp_ms_by_switch,   # ms
        "mean_resp": mean_T,                   # ms
        "mad_resp": mad_T,                     # ms
        "max_resp": max_T,                     # ms
        "unstable_controllers": sorted(unstable),
    }

    return usage_e, rt_metrics
    
def build_paths_sc_from_switch_paths(assign, paths_by_switch):
    paths_sc = {}

    for s, c in assign.items():

        # ✅ CASE 1: self-controller
        if s == c:
            paths_sc[(s, c)] = [s]   # or [] if you prefer
            continue

        # ✅ CASE 2: normal path
        p = paths_by_switch.get(s)

        if not p:
            raise ValueError(f"❌ Missing path for switch {s} → controller {c}")

        paths_sc[(s, c)] = p

    return paths_sc