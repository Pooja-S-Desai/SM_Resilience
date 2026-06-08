# baseline3_easm_exact.py
# ------------------------------------------------------------
# Baseline-3: EASM exact algorithm-structure implementation
# Paper: "EASM: Efficiency-aware switch migration for balancing
# controller loads in software-defined networking"
#
# Implements the EASM flow:
#   EASM-1: load imbalance detection using load-difference matrix
#           and trigger factor.
#   EASM-2: migration-object selection:
#           emigration controller, migrating switch, immigration controller.
#   EASM-3: migration triplet execution and repeated re-detection.
#
# Pipeline adaptations:
# - Controllers are fixed.
# - Initial switch-controller assignment comes from init_assign.
# - Controller load is represented by your pipeline's switch request load.
# - Routing is not optimized here; final paths are shortest-path paths_sc.
# ------------------------------------------------------------

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import math
import networkx as nx


def _safe_dist(dmat: Optional[dict], a, b, default: float = 0.0) -> float:
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


def _controller_loads(assign, loads, controllers):
    out = {c: 0.0 for c in controllers}
    for s, c in assign.items():
        if c in out:
            out[c] += float(loads.get(s, 0.0))
    return out


def _mean_load(loads_by_ctrl, controllers):
    if not controllers:
        return 0.0
    return sum(float(loads_by_ctrl.get(c, 0.0)) for c in controllers) / len(controllers)


def _eta_load_variance(loads_by_ctrl, controllers):
    if not controllers:
        return 0.0
    Lbar = _mean_load(loads_by_ctrl, controllers)
    return sum((float(loads_by_ctrl.get(c, 0.0)) - Lbar) ** 2 for c in controllers) / len(controllers)


def _build_paths(final_assign, paths_sc):
    pair = {}
    sw = {}
    for s, c in final_assign.items():
        if s == c:
            p = [s]
        else:
            p = list((paths_sc or {}).get((s, c), []))
        pair[(s, c)] = p
        sw[s] = p
    return pair, sw


def _compute_usage_shortest(assign, paths_pair, loads, msg_bits):
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


def _load_difference_matrix(ctrl_loads, controllers):
    D = {}
    for cm in controllers:
        for cn in controllers:
            denom = float(ctrl_loads.get(cn, 0.0))
            if denom <= 0:
                D[(cm, cn)] = float("inf") if ctrl_loads.get(cm, 0.0) > 0 else 1.0
            else:
                D[(cm, cn)] = float(ctrl_loads.get(cm, 0.0)) / denom
    return D


def _trigger_factor_set(ctrl_loads, controllers):
    D = _load_difference_matrix(ctrl_loads, controllers)
    finite_vals = [v for v in D.values() if math.isfinite(v)]
    if not finite_vals:
        return [], D, 0.0

    maxD = max(finite_vals)
    minD = min(finite_vals)
    Lambda = 0.0 if maxD <= 0 else (maxD - minD) / maxD

    TF = []
    seen = set()
    for cm in controllers:
        for cn in controllers:
            if cm == cn:
                continue
            dmn = D.get((cm, cn), 1.0)
            dnm = D.get((cn, cm), 1.0)
            if not (math.isfinite(dmn) and math.isfinite(dnm)):
                continue
            delta = abs(dmn - dnm)
            if delta > Lambda:
                heavy, light = (cm, cn) if ctrl_loads.get(cm, 0.0) > ctrl_loads.get(cn, 0.0) else (cn, cm)
                if heavy != light and (heavy, light) not in seen:
                    TF.append((heavy, light))
                    seen.add((heavy, light))

    return TF, D, Lambda


def run_baseline3_easm_exact(
    G: nx.Graph,
    switches: List[int],
    controllers: List[int],
    loads: Dict[int, float],
    capacities: Dict[int, float],
    init_assign: Dict[int, int],
    *,
    dij: Optional[dict] = None,
    paths_sc: Optional[dict] = None,
    msg_bits: int = 128,
    usable_threshold: float = 0.8,
    overload_threshold: float = 0.9,
    packet_in_cost: float = 1.0,
    gamma_weight: float = 0.5,
    max_rounds: int = 10_000,
    max_migrations: Optional[int] = None,
    verbose: bool = False,
):
    switches = list(map(int, switches))
    controllers = list(map(int, controllers))
    assign = {int(s): int(c) for s, c in dict(init_assign).items()}

    max_migrations = len(switches) if max_migrations is None else int(max_migrations)
    Omega = {c: float(capacities.get(c, 0.0)) * float(usable_threshold) for c in controllers}

    def util(ctrl_loads, c):
        cap = float(Omega.get(c, 0.0))
        if cap <= 0:
            return float("inf")
        return float(ctrl_loads.get(c, 0.0)) / cap

    def feasible_target(cand_loads, cn):
        return util(cand_loads, cn) <= 1.0

    def migration_cost(s, cm, cn):
        alpha_s = float(loads.get(s, 0.0))
        h_im = _safe_dist(dij, s, cm, 0.0)
        h_in = _safe_dist(dij, s, cn, float("inf"))
        if not math.isfinite(h_in):
            return float("inf"), h_im, h_in
        mc = float(packet_in_cost) + alpha_s * abs(h_in - h_im)
        return max(mc, 1e-9), h_im, h_in

    total_migration_cost = 0.0
    total_tau = 0.0
    migrations = 0
    moved: Set[int] = set()
    step_log = []
    status = "SUCCESS"

    for rnd in range(int(max_rounds)):
        ctrl_loads = _controller_loads(assign, loads, controllers)
        TF, Dmat, Lambda = _trigger_factor_set(ctrl_loads, controllers)

        CEM = []
        mean_load = _mean_load(ctrl_loads, controllers)
        for cm, cn in TF:
            if util(ctrl_loads, cm) >= overload_threshold or ctrl_loads[cm] > mean_load:
                CEM.append(cm)
        CEM = sorted(set(CEM), key=lambda c: ctrl_loads.get(c, 0.0), reverse=True)

        if not CEM:
            status = "SUCCESS"
            break

        if migrations >= max_migrations:
            status = "STOPPED_MAX_MIGRATIONS"
            break

        did_move = False
        eta_before = _eta_load_variance(ctrl_loads, controllers)
        Lbar = _mean_load(ctrl_loads, controllers)

        for cm in CEM:
            Gamma_cm = [s for s in switches if assign.get(s) == cm and s not in moved and s != cm]
            if not Gamma_cm:
                continue

            switch_records = {}
            max_h = max((_safe_dist(dij, s, cm, 0.0) for s in Gamma_cm), default=0.0)
            exp_max_h = math.exp(min(max_h, 700))

            for s in Gamma_cm:
                alpha_s = float(loads.get(s, 0.0))
                target_records = {}
                best_tau_for_s = None
                best_target_for_tau = None

                for cn in controllers:
                    if cn == cm:
                        continue

                    cand_loads = dict(ctrl_loads)
                    cand_loads[cm] = cand_loads.get(cm, 0.0) - alpha_s
                    cand_loads[cn] = cand_loads.get(cn, 0.0) + alpha_s

                    if not feasible_target(cand_loads, cn):
                        continue

                    mc, h_im, h_in = migration_cost(s, cm, cn)
                    if not math.isfinite(mc):
                        continue

                    eta_after = _eta_load_variance(cand_loads, controllers)
                    tau = abs(eta_after - eta_before) / mc

                    target_records[cn] = {
                        "cand_loads": cand_loads,
                        "eta_after": eta_after,
                        "migration_cost": mc,
                        "tau": tau,
                        "h_old": h_im,
                        "h_new": h_in,
                    }

                    if best_tau_for_s is None or tau > best_tau_for_s:
                        best_tau_for_s = tau
                        best_target_for_tau = cn

                if best_tau_for_s is None:
                    continue

                post_source_load = ctrl_loads[cm] - alpha_s
                h_im = _safe_dist(dij, s, cm, 0.0)
                distance_factor = math.exp(min(h_im, 700)) / exp_max_h if exp_max_h > 0 else 1.0
                balance_factor = 1.0 / (1.0 + abs(Lbar - post_source_load))
                rho = best_tau_for_s * balance_factor * distance_factor

                switch_records[s] = {
                    "rho": rho,
                    "target_records": target_records,
                    "best_tau_target": best_target_for_tau,
                }

            if not switch_records:
                continue

            si = max(switch_records, key=lambda s: switch_records[s]["rho"])
            target_records = switch_records[si]["target_records"]

            best_cn = None
            best_phi = None
            for cn, rec in target_records.items():
                cand_loads = rec["cand_loads"]
                remaining = max(0.0, Omega[cn] - cand_loads[cn])
                phi = float(gamma_weight) * remaining + (1.0 - float(gamma_weight)) * rec["tau"]
                if best_phi is None or phi > best_phi:
                    best_phi = phi
                    best_cn = cn

            if best_cn is None:
                continue

            rec = target_records[best_cn]
            assign[si] = best_cn
            moved.add(si)

            migrations += 1
            total_migration_cost += float(rec["migration_cost"])
            total_tau += float(rec["tau"])
            did_move = True

            step_log.append({
                "round": rnd,
                "triplet": [cm, si, best_cn],
                "source_controller": cm,
                "switch": si,
                "target_controller": best_cn,
                "lambda_threshold": float(Lambda),
                "rho_switch": float(switch_records[si]["rho"]),
                "phi_target": float(best_phi),
                "eta_before": float(eta_before),
                "eta_after": float(rec["eta_after"]),
                "migration_cost": float(rec["migration_cost"]),
                "migration_efficiency_tau": float(rec["tau"]),
                "old_hop": float(rec["h_old"]),
                "new_hop": float(rec["h_new"]),
            })

            if verbose:
                print(
                    f"[EASM-EXACT] round={rnd} triplet=[C{cm},s{si},C{best_cn}] "
                    f"rho={switch_records[si]['rho']:.6g}, phi={best_phi:.6g}, "
                    f"tau={rec['tau']:.6g}, MC={rec['migration_cost']:.3f}"
                )

            if migrations >= max_migrations:
                break

        if not did_move:
            # EASM heuristic found no beneficial/local feasible migration.
            # This is not global infeasibility; keep current assignment.
            status = "SUCCESS_NO_EASM_MOVE"
            break

    else:
        status = "STOPPED_MAX_ROUNDS"

    final_loads = _controller_loads(assign, loads, controllers)
    paths_pair, paths_sw = _build_paths(assign, paths_sc)
    usage_e = _compute_usage_shortest(assign, paths_pair, loads, msg_bits)

    obj_val = float(total_migration_cost - total_tau)

    meta = {
        "paper": "EASM_exact_algorithm_structure",
        "heuristic": True,
        "status": status,
        "migrations": int(migrations),
        "total_migration_cost": float(total_migration_cost),
        "total_migration_efficiency_tau": float(total_tau),
        "objective_proxy": obj_val,
        "steps": step_log,
        "final_util_by_ctrl": {c: util(final_loads, c) for c in controllers},
        "constants": {
            "usable_threshold": usable_threshold,
            "overload_threshold": overload_threshold,
            "packet_in_cost": packet_in_cost,
            "gamma_weight": gamma_weight,
        },
    }

    return (
        assign,
        paths_pair,
        paths_sw,
        final_loads,
        obj_val,
        int(migrations),
        usage_e,
        None,
        status,
        meta,
    )
