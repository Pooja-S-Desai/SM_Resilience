# baseline3_ddm_exact.py
# ------------------------------------------------------------
# Baseline-3: Exact DDM-style switch migration baseline
# Paper: "A Distributed Decision Mechanism for Controller Load Balancing
# Based on Switch Migration in SDN", China Communications, 2018.
#
# This version follows the paper algorithm structure:
#   Stage 1: construct MDF around each overloaded controller
#   Stage 2: select migrating switch using rho probability/ranking
#            select target controller using Algorithm 1 greedy pruning
#   Stage 3: perform migration and repeat until controller load is balanced
#
# Notes for your pipeline:
# - Controllers are fixed. No controller placement is performed.
# - Initial mapping comes from main.py as init_assign.
# - Routing comes from main.py as paths_sc = shortest paths {(s,c): path}.
# - This baseline does NOT optimize MCF/link capacity. It only migrates assignment.
# ------------------------------------------------------------

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import math
import networkx as nx


def _safe_dist(dmat: Optional[dict], a: int, b: int, default: float = 0.0) -> float:
    """Supports both nested dict dmat[a][b] and flat dict dmat[(a,b)]."""
    if a == b:
        return 0.0
    if dmat is None:
        return default
    try:
        row = dmat.get(a, None)
        if isinstance(row, dict):
            v = row.get(b, default)
        else:
            v = dmat.get((a, b), default)
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _controller_loads(
    assign: Dict[int, int],
    loads: Dict[int, float],
    controllers: List[int],
) -> Dict[int, float]:
    out = {c: 0.0 for c in controllers}
    for s, c in assign.items():
        if c in out:
            out[c] += float(loads.get(s, 0.0))
    return out


def _build_paths(
    final_assign: Dict[int, int],
    paths_sc: Optional[dict],
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]:
    """Returns pair-paths for RT and switch-paths for CSV logging."""
    pair, sw = {}, {}
    for s, c in final_assign.items():
        if s == c:
            p = [s]
        else:
            p = list((paths_sc or {}).get((s, c), []))
        pair[(s, c)] = p
        sw[s] = p
    return pair, sw


def _compute_usage_shortest(
    assign: Dict[int, int],
    paths_pair: Dict[Tuple[int, int], List[int]],
    loads: Dict[int, float],
    msg_bits: int,
) -> Dict[Tuple[int, int], float]:
    usage = defaultdict(float)
    for s, c in assign.items():
        p = paths_pair.get((s, c), [])
        if not p or len(p) < 2:
            continue
        bits = float(loads.get(s, 0.0)) * float(msg_bits)
        for i in range(len(p) - 1):
            u, v = p[i], p[i + 1]
            e = (u, v) if u < v else (v, u)
            usage[e] += bits
    return dict(usage)


def _subdomain_neighbors(
    G: nx.Graph,
    assign: Dict[int, int],
    cr: int,
    controllers: List[int],
    Dcc: Optional[dict],
) -> List[int]:
    """
    DDM MDF is built from an overloaded subdomain and eligible neighbor subdomains.
    In this pipeline, subdomains are induced by current switch-controller assignment.
    We treat ck as neighbor of cr if any physical graph edge crosses the two subdomains.
    If no such neighbor is found, we fall back to all reachable controllers by Dcc.
    """
    domain_cr = {s for s, c in assign.items() if c == cr}
    neigh = set()

    for u, v in G.edges():
        cu = assign.get(int(u))
        cv = assign.get(int(v))
        if cu == cr and cv in controllers and cv != cr:
            neigh.add(cv)
        if cv == cr and cu in controllers and cu != cr:
            neigh.add(cu)

    if not neigh:
        for ck in controllers:
            if ck != cr and math.isfinite(_safe_dist(Dcc, cr, ck, float("inf"))):
                neigh.add(ck)

    return sorted(neigh)


def run_baseline3_ddm_exact(
    G: nx.Graph,
    switches: List[int],
    controllers: List[int],
    loads: Dict[int, float],
    capacities: Dict[int, float],
    init_assign: Dict[int, int],
    *,
    dij: Optional[dict] = None,
    Dcc: Optional[dict] = None,
    paths_sc: Optional[dict] = None,
    msg_bits: int = 128,

    # Paper thresholds/constants.
    overload_threshold: float = 0.8,       # Eq. 14: 0.9 <= gamma <= 1 => overload
    usable_threshold: float = 1.0,         # Paper gamma = load / omega. Use 1.0 for exact paper.
                                           # Use CAPACITY_THRESHOLD only if you want pipeline-safe capacity.
    tau_data: float = 1.0,                 # tau1
    tau_move: float = 1.0,                 # tau2
    tau_sync: float = 1.0,                 # tau3
    v_data: float = 60.0,                  # paper: v_cr = 60 KB/s
    delta_rule: float = 30.0,              # paper: delta_rule = 30 Byte
    eps_comm: float = 15.0,                # paper: epsilon = 15 KB/s
    mu_sync: float = 3.0,                  # paper: mu = 3 KB/s

    max_rounds: int = 10_000,
    max_migrations: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[
    Dict[int, int],                         # final_assign
    Dict[Tuple[int, int], List[int]],       # paths_pair for RT
    Dict[int, List[int]],                   # paths_by_switch for logging
    Dict[int, float],                       # final_loads
    float,                                  # total DDM objective cost
    int,                                    # migration count
    Dict[Tuple[int, int], float],           # usage_e on shortest paths
    None,                                   # mip_gap
    str,                                    # status
    Dict[str, Any],                         # meta
]:
    switches = list(map(int, switches))
    controllers = list(map(int, controllers))
    assign = {int(s): int(c) for s, c in dict(init_assign).items()}
    max_migrations = len(switches) if max_migrations is None else int(max_migrations)

    omega = {c: float(capacities.get(c, 0.0)) * float(usable_threshold) for c in controllers}

    def gamma(loads_by_ctrl: Dict[int, float], c: int) -> float:
        return float("inf") if omega.get(c, 0.0) <= 0 else float(loads_by_ctrl.get(c, 0.0)) / omega[c]

    def build_mdf(cr: int, loads_by_ctrl: Dict[int, float], used_controllers: Set[int]) -> List[int]:
        """
        Stage 1 MDF:
        - cr must be overloaded.
        - eligible neighbor controllers must be normal/underloaded.
        - no MDF intersection: avoid controllers already used in another MDF during same outer pass.
        """
        gammas = {c: gamma(loads_by_ctrl, c) for c in controllers}
        avg_gamma = sum(g for g in gammas.values() if math.isfinite(g)) / max(1, sum(math.isfinite(g) for g in gammas.values()))

        nbrs = _subdomain_neighbors(G, assign, cr, controllers, Dcc)
        accepted = [cr]
        for ck in nbrs:
            if ck in used_controllers:
                continue
            # Eq. 15 interpreted from DDM text: neighbor agrees if it is below average and not overloaded.
            if gammas[ck] < avg_gamma and gammas[ck] < overload_threshold:
                accepted.append(ck)

        return accepted

    def select_switch_by_rho(cr: int, Src: List[int]) -> Tuple[int, Dict[int, float]]:
        """
        Stage 2 migrating switch:
        Paper selects switch with max rho_ir, considering high occupied resource eta_ir
        and long distance d_ir. We implement rho as normalized exp(eta_norm + d_norm),
        which preserves exactly the stated selection behavior: larger load/resource share
        and larger distance => larger migration probability.
        """
        if not Src:
            raise ValueError("Empty source switch set")

        max_load = max(float(loads.get(s, 0.0)) for s in Src) or 1.0
        max_dist = max(_safe_dist(dij, s, cr, 0.0) for s in Src) or 1.0

        raw = {}
        for s in Src:
            eta = float(loads.get(s, 0.0)) / max_load
            dnorm = _safe_dist(dij, s, cr, 0.0) / max_dist
            raw[s] = math.exp(eta + dnorm)

        denom = sum(raw.values()) or 1.0
        rho = {s: raw[s] / denom for s in Src}
        return max(Src, key=lambda s: rho[s]), rho

    def cost_terms(cr: int, ck: int, s_star: int, current_loads: Dict[int, float]) -> Tuple[float, float, float, float]:
        """
        Exact paper cost structure:
          Pdata = sum_{i in S_cr} d_ir * v_cr * x_ir
          Pmove = Prule + Pcom + Preq
          Prule = delta_rule * d_ir * x_ir
          Pcom  = epsilon * [sum_{i in S_cr}(x_ir*d_ir) + sum_{j in S_ck}(x_jk*d_jk)]
          Preq  = min_dik * epsilon * [sum x_ir + sum x_jk]
          Psyn  = mu * d_rk * [sum_{i in S_cr} g_ir + sum_{j in S_ck} g_jk]
        Pipeline mapping:
          g_ir/g_jk are represented by switch request loads.
          x is implicit from current assignment.
        """
        S_cr = [s for s in switches if assign.get(s) == cr]
        S_ck = [s for s in switches if assign.get(s) == ck]

        d_ir_star = _safe_dist(dij, s_star, cr, 0.0)
        d_ik_star = _safe_dist(dij, s_star, ck, 0.0)
        d_rk = _safe_dist(Dcc, cr, ck, 0.0)

        Pdata = sum(_safe_dist(dij, s, cr, 0.0) * v_data for s in S_cr)

        Prule = delta_rule * d_ir_star

        Pcom = eps_comm * (
            sum(_safe_dist(dij, s, cr, 0.0) for s in S_cr)
            + sum(_safe_dist(dij, s, ck, 0.0) for s in S_ck)
        )

        Preq = d_ik_star * eps_comm * (len(S_cr) + len(S_ck))

        Pmove = Prule + Pcom + Preq

        Psyn = mu_sync * d_rk * (
            sum(float(loads.get(s, 0.0)) for s in S_cr)
            + sum(float(loads.get(s, 0.0)) for s in S_ck)
        )

        Pobject = tau_data * Pdata + tau_move * Pmove + tau_sync * Psyn
        return Pdata, Pmove, Psyn, Pobject

    total_cost = 0.0
    migrations = 0
    moved_switches = set()
    step_log = []
    status = "SUCCESS"

    for rnd in range(int(max_rounds)):
        loads_by_ctrl = _controller_loads(assign, loads, controllers)
        overloaded = [
            c for c in controllers
            if gamma(loads_by_ctrl, c) >= overload_threshold
        ]

        if not overloaded:
            status = "SUCCESS"
            break

        if migrations >= max_migrations:
            status = "STOPPED_MAX_MIGRATIONS"
            break

        used_in_mdf = set()
        did_move = False

        # Multiple MDFs can exist simultaneously; process by highest gamma first.
        for cr in sorted(overloaded, key=lambda c: gamma(loads_by_ctrl, c), reverse=True):
            if cr in used_in_mdf:
                continue

            Fr = build_mdf(cr, loads_by_ctrl, used_in_mdf)
            used_in_mdf.update(Fr)

            if len(Fr) <= 1:
                continue
                

            Src = [s for s in switches if assign.get(s) == cr and s != cr and s not in moved_switches]
            if not Src:
                continue

            s_star, rho = select_switch_by_rho(cr, Src)
            r_s = float(loads.get(s_star, 0.0))

            # Algorithm 1 target selection.
            Nr = [c for c in Fr if c != cr]
            target_candidates = []
            stored = {"Pdata": [], "Pmove": [], "Psyn": []}

            # Lines 1-10: nearest-controller round robin + remove if gamma_k* > 0.9.
            Nr_sorted = sorted(Nr, key=lambda ck: _safe_dist(dij, s_star, ck, float("inf")))
            for ck in Nr_sorted:
                cand_loads = dict(loads_by_ctrl)
                cand_loads[cr] = cand_loads.get(cr, 0.0) - r_s
                cand_loads[ck] = cand_loads.get(ck, 0.0) + r_s
                if gamma(cand_loads, ck) > overload_threshold:
                    continue
                target_candidates.append(ck)

            if not target_candidates:
                continue

            costs = {}
            for ck in target_candidates:
                Pdata, Pmove, Psyn, Pobject = cost_terms(cr, ck, s_star, loads_by_ctrl)
                costs[ck] = {
                    "Pdata": Pdata,
                    "Pmove": Pmove,
                    "Psyn": Psyn,
                    "Pobject": Pobject,
                }

            # Lines 11-18: for each cost term remove max-cost controller and store min-cost controller.
            remaining = set(target_candidates)
            for term in ("Pdata", "Pmove", "Psyn"):
                if not remaining:
                    break
                max_c = max(remaining, key=lambda c: costs[c][term])
                remaining.discard(max_c)
                if remaining:
                    min_c = min(remaining, key=lambda c: costs[c][term])
                    stored[term].append(min_c)

            # Lines 19-22: compute Pobject for stored controllers and choose min Pobject.
            final_pool = set()
            for vals in stored.values():
                final_pool.update(vals)
            if not final_pool:
                final_pool = set(target_candidates)

            ctar = min(final_pool, key=lambda c: costs[c]["Pobject"])

            assign[s_star] = ctar
            moved_switches.add(s_star)
            migrations += 1
            total_cost += float(costs[ctar]["Pobject"])
            did_move = True

            step_log.append({
                "round": rnd,
                "MDF": list(Fr),
                "source_controller": cr,
                "target_controller": ctar,
                "switch": s_star,
                "rho": {int(k): float(v) for k, v in rho.items()},
                **{k: float(v) for k, v in costs[ctar].items()},
            })

            if verbose:
                print(f"[DDM-EXACT] round={rnd} MDF={Fr} migrate s={s_star}: C{cr}->C{ctar}, cost={costs[ctar]['Pobject']:.3f}")

            if migrations >= max_migrations:
                break

        if not did_move:
            status = "NO_FEASIBLE_MDF_TARGET"
            break
    else:
        status = "STOPPED_MAX_ROUNDS"

    final_loads = _controller_loads(assign, loads, controllers)
    paths_pair, paths_sw = _build_paths(assign, paths_sc)
    usage_e = _compute_usage_shortest(assign, paths_pair, loads, msg_bits)

    meta = {
        "paper": "DDM_2018_exact_algorithm_structure",
        "heuristic": True,
        "note": "Exact DDM algorithm structure; link routing remains shortest-path as in the paper's pre-optimized routing assumption.",
        "migrations": migrations,
        "total_ddm_cost": float(total_cost),
        "steps": step_log,
        "final_gamma_by_ctrl": {c: gamma(final_loads, c) for c in controllers},
        "status": status,
        "constants": {
            "overload_threshold": overload_threshold,
            "tau_data": tau_data,
            "tau_move": tau_move,
            "tau_sync": tau_sync,
            "v_data": v_data,
            "delta_rule": delta_rule,
            "eps_comm": eps_comm,
            "mu_sync": mu_sync,
        },
    }

    return (
        assign,
        paths_pair,
        paths_sw,
        final_loads,
        float(total_cost),
        int(migrations),
        usage_e,
        None,
        status,
        meta,
    )