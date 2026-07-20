from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

import networkx as nx

from plotting import (
    plot_final_vs_recovery_assignment,
    plot_fractional_traffic_recovery_comparison,
)


TOL = 1e-8


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return obj


def _normalize_fractional_plan(
    fractional_assignment: Optional[Dict],
) -> Dict[int, Dict[int, float]]:
    """
    Converts keys and values to:

        {
            switch: {
                controller: fraction
            }
        }
    """
    normalized: Dict[int, Dict[int, float]] = {}

    for switch, targets in (fractional_assignment or {}).items():
        switch = int(switch)

        if not isinstance(targets, dict):
            continue

        normalized_targets = {}

        for controller, value in targets.items():
            try:
                controller = int(controller)
            except (TypeError, ValueError):
                continue

            if isinstance(value, dict):
                fraction = float(
                    value.get(
                        "fraction",
                        value.get("weight", value.get("value", 0.0)),
                    )
                )
            else:
                fraction = float(value)

            if fraction > TOL:
                normalized_targets[controller] = fraction

        # Empty switch maps are not fractional recovery. Keeping them made old
        # failed runs appear as uses_fractional_recovery=YES while containing
        # no actual controller allocation.
        if normalized_targets:
            normalized[switch] = normalized_targets

    return normalized


def _normalize_residuals(
    residual_by_switch: Optional[Dict],
    loads: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    normalized = {}

    for switch, info in (residual_by_switch or {}).items():
        switch = int(switch)
        switch_load = float(loads.get(switch, 0.0))

        if isinstance(info, dict):
            residual_fraction = float(
                info.get(
                    "residual_fraction",
                    info.get("fraction", 0.0),
                )
            )

            residual_load = float(
                info.get(
                    "residual_load",
                    switch_load * residual_fraction,
                )
            )
        else:
            residual_load = float(info)
            residual_fraction = (
                residual_load / switch_load
                if switch_load > 0
                else 0.0
            )

        if residual_load > TOL:
            normalized[switch] = {
                "switch_load": switch_load,
                "residual_fraction": residual_fraction,
                "residual_load": residual_load,
            }

    return normalized


def _integral_post_failure_loads(
    recovery_assignment: Dict[int, int],
    loads: Dict[int, float],
    controllers: Iterable[int],
    failed_controller: int,
) -> Dict[int, float]:
    result = {
        int(c): 0.0
        for c in controllers
        if int(c) != int(failed_controller)
    }

    for switch, controller in recovery_assignment.items():
        switch = int(switch)
        controller = int(controller)

        if controller == failed_controller:
            continue

        result[controller] = (
            result.get(controller, 0.0)
            + float(loads.get(switch, 0.0))
        )

    return result


def _fractional_post_failure_loads(
    *,
    initial_assignment: Dict[int, int],
    fractional_assignment: Dict[int, Dict[int, float]],
    residual_by_switch: Dict[int, Dict[str, float]],
    loads: Dict[int, float],
    controllers: Iterable[int],
    failed_controller: int,
) -> Dict[int, float]:
    """
    For every switch:

    - If it has fractional assignments, use those fractions.
    - Otherwise, retain it at its original controller when that controller
      survives.
    """
    result = {
        int(c): 0.0
        for c in controllers
        if int(c) != int(failed_controller)
    }

    for switch, original_controller in initial_assignment.items():
        switch = int(switch)
        original_controller = int(original_controller)
        switch_load = float(loads.get(switch, 0.0))

        targets = fractional_assignment.get(switch, {})

        if targets:
            for target, fraction in targets.items():
                target = int(target)

                if target == failed_controller:
                    continue

                result[target] = (
                    result.get(target, 0.0)
                    + switch_load * float(fraction)
                )

        elif original_controller != failed_controller:
            result[original_controller] = (
                result.get(original_controller, 0.0)
                + switch_load
            )

        # Residual load is intentionally not added to any controller.

    return result


def _dominant_assignment_from_fractional(
    *,
    initial_assignment: Dict[int, int],
    fractional_assignment: Dict[int, Dict[int, float]],
    failed_controller: int,
) -> Dict[int, int]:
    """
    Used only for compatibility with integral plotting/evaluation.

    The complete fractional plan remains preserved separately.
    """
    dominant = {
        int(s): int(c)
        for s, c in initial_assignment.items()
    }

    for switch, targets in fractional_assignment.items():
        valid_targets = {
            int(c): float(frac)
            for c, frac in targets.items()
            if int(c) != int(failed_controller)
            and float(frac) > TOL
        }

        if valid_targets:
            dominant[int(switch)] = max(
                valid_targets,
                key=valid_targets.get,
            )

    return dominant


def _migration_statistics(
    *,
    initial_assignment: Dict[int, int],
    recovery_assignment: Dict[int, int],
    fractional_assignment: Dict[int, Dict[int, float]],
    residual_by_switch: Dict[int, Dict[str, float]],
    loads: Dict[int, float],
    failed_controller: int,
) -> Dict[str, Any]:
    migrated_switches = {}
    total_fractionally_migrated_load = 0.0

    for switch, original_controller in initial_assignment.items():
        switch = int(switch)
        original_controller = int(original_controller)
        switch_load = float(loads.get(switch, 0.0))

        targets = fractional_assignment.get(switch, {})

        if targets:
            retained_fraction = (
                float(targets.get(original_controller, 0.0))
                if original_controller != failed_controller
                else 0.0
            )

            residual_fraction = float(
                residual_by_switch
                .get(switch, {})
                .get("residual_fraction", 0.0)
            )

            moved_fraction = max(
                0.0,
                1.0 - retained_fraction - residual_fraction,
            )

            moved_load = switch_load * moved_fraction

            if moved_fraction > TOL:
                migrated_switches[switch] = {
                    "from": original_controller,
                    "to_fractional": targets,
                    "load": switch_load,
                    "moved_fraction": moved_fraction,
                    "moved_load": moved_load,
                    "is_orphan": (
                        original_controller == failed_controller
                    ),
                }

                total_fractionally_migrated_load += moved_load

        else:
            new_controller = recovery_assignment.get(switch)

            if new_controller is None:
                continue

            new_controller = int(new_controller)

            if new_controller != original_controller:
                migrated_switches[switch] = {
                    "from": original_controller,
                    "to": new_controller,
                    "load": switch_load,
                    "moved_fraction": 1.0,
                    "moved_load": switch_load,
                    "is_orphan": (
                        original_controller == failed_controller
                    ),
                }

                total_fractionally_migrated_load += switch_load

    orphan_migrations = sum(
        1
        for value in migrated_switches.values()
        if value["is_orphan"]
    )

    non_orphan_migrations = sum(
        1
        for value in migrated_switches.values()
        if not value["is_orphan"]
    )

    return {
        "migrated_switches": migrated_switches,
        "total_migrations": len(migrated_switches),
        "orphan_migrations": orphan_migrations,
        "non_orphan_migrations": non_orphan_migrations,
        "total_migrated_load": total_fractionally_migrated_load,
    }


def _append_failure_record(
    *,
    output_file: str,
    topology_name: str,
    run_index: int,
    controllers,
    algorithm: str,
    failed_controller: int,
    record: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as fh:
                root = json.load(fh)
        except (OSError, json.JSONDecodeError):
            root = {}
    else:
        root = {}

    root.setdefault("topology", str(topology_name))
    root.setdefault("run_index", int(run_index))
    root.setdefault(
        "controllers",
        [int(c) for c in controllers],
    )
    root.setdefault("algorithms", {})

    root["algorithms"].setdefault(str(algorithm), {})
    root["algorithms"][str(algorithm)][
        str(int(failed_controller))
    ] = _json_safe(record)

    temporary_file = output_file + ".tmp"

    with open(temporary_file, "w") as fh:
        json.dump(_json_safe(root), fh, indent=2)

    os.replace(temporary_file, output_file)



RECOVERY_CSV_COLUMNS = [
    "method_model", "switch_seed", "master_seed", "run_index", "run_number",
    "topology", "total_number_of_switches", "total_number_of_controllers",
    "status", "objective_value", "switch_loads", "controller_capacities",
    "initial_switch_assignment", "recovery_assignment_dominant",
    "controller_loads_initial",
    "max_minus_min_controller_loads_initial",
    "max_minus_min_controller_loads_final",
    "max_controller_load_initial", "max_controller_load_final",
    "min_controller_load_initial", "min_controller_load_final",
    "variance_controller_loads_initial",
    "solve_time_sec", "migration_cost", "number_of_migrations",
    "number_of_switches_migrated", "migrated_switches", "orphan_migrations",
    "non_orphan_migrations", "failed_controller_number", "orphan_switches",
    "orphan_count", "post_failure_variance", "pre_failure_response_time_ms",
    "post_failure_mean_switch_response_time_ms",
    "post_failure_max_switch_response_time_ms", "controller_loads_post_migration",
    "controller_utilization_post_migration", "post_failure_switch_assignment_with_loads",
    "uses_fractional_recovery", "fractional_switch_count",
    "fractional_controller_allocations", "fractional_load_allocations",
    "fractional_assignment_details",
    "residual_by_switch", "overloaded_controllers",
    "total_orphan_load", "total_orphan_load_reassigned", "total_orphan_load_residual",
    "total_orphan_load_unaccommodated", "new_controller_residual_by_switch",
    "new_controller_planned", "new_controller_id", "new_controller_capacity",
    "total_controller_capacity_after_plan", "new_controller_total_assigned_load",
    "new_controller_utilization", "max_link_utilization_post_migration",
    "violated_links_post_migration", "violated_link_count_post_migration",
    "link_utilization_threshold", "reassignment_scope",
    "global_total_switch_load_unaccommodated", "feasibility", "mip_gap",
]


def _variance(values) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def _json_cell(value) -> str:
    return json.dumps(_json_safe(value), separators=(",", ":"), sort_keys=True)


def _edge_key(u, v):
    u = int(u)
    v = int(v)
    return (u, v) if u < v else (v, u)


def _normalize_edge_map(edge_map: Optional[Dict]) -> Dict[tuple, float]:
    normalized = {}
    for edge, value in (edge_map or {}).items():
        try:
            u, v = edge
            normalized[_edge_key(u, v)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _shortest_path_link_usage(
    *,
    G,
    assignment: Dict[int, int],
    loads: Dict[int, float],
    msg_bits: float,
    failed_controller: Optional[int] = None,
    residual_by_switch: Optional[Dict] = None,
) -> Dict[tuple, float]:
    usage = defaultdict(float)
    residual_switches = {int(s) for s in (residual_by_switch or {})}
    graph = G.to_undirected() if G.is_directed() else G

    for switch, controller in assignment.items():
        switch = int(switch)
        controller = int(controller)

        if failed_controller is not None and controller == int(failed_controller):
            continue
        if switch in residual_switches:
            continue

        try:
            path = (
                [switch]
                if switch == controller
                else list(nx.shortest_path(graph, switch, controller, weight="weight"))
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        traffic = float(loads.get(switch, 0.0)) * float(msg_bits)
        for u, v in zip(path[:-1], path[1:]):
            usage[_edge_key(u, v)] += traffic

    return dict(usage)


def _link_utilization_statistics(
    *,
    link_usage: Optional[Dict],
    edge_caps: Optional[Dict],
    threshold: float,
) -> Dict[str, Any]:
    capacities = _normalize_edge_map(edge_caps)
    usage = _normalize_edge_map(link_usage)

    max_util = 0.0
    violated_links = {}

    for edge, capacity in capacities.items():
        if capacity <= 0.0:
            continue

        used = float(usage.get(edge, 0.0))
        utilization = used / capacity
        max_util = max(max_util, utilization)

        if utilization > float(threshold) + TOL:
            violated_links[edge] = {
                "used_bits_per_sec": used,
                "capacity_bits_per_sec": capacity,
                "utilization": utilization,
                "excess_over_threshold": utilization - float(threshold),
            }

    return {
        "max_link_utilization": max_util,
        "violated_links": violated_links,
        "violated_link_count": len(violated_links),
    }


def _append_recovery_csv(output_file: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    normalized = {name: row.get(name, "") for name in RECOVERY_CSV_COLUMNS}
    existing_rows = []
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, newline="") as fh:
            existing_rows = list(csv.DictReader(fh))

    # Re-running an experiment replaces the same scenario instead of silently
    # duplicating it.  Different models, runs and controller failures coexist.
    key_columns = (
        "method_model", "master_seed", "run_index", "topology",
        "failed_controller_number",
    )
    new_key = tuple(str(normalized.get(name, "")) for name in key_columns)
    existing_rows = [
        old for old in existing_rows
        if tuple(str(old.get(name, "")) for name in key_columns) != new_key
    ]
    existing_rows.append(normalized)

    temporary_file = output_file + ".tmp"
    with open(temporary_file, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=RECOVERY_CSV_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(existing_rows)
    os.replace(temporary_file, output_file)


def _shortest_path_response_metrics(
    *,
    G,
    assignment,
    loads,
    capacities,
    sync_delay_ms=0.0,
    usable_threshold=0.95,
    failed_controller=None,
    residual_by_switch=None,
):
    """Evaluate post-failure switch RT consistently using shortest paths and M/M/1."""
    try:
        from rt_metrics import compute_response_metrics
        paths = {}
        UG = G.to_undirected() if G.is_directed() else G
        residual_switches = {
            int(s) for s in (residual_by_switch or {})
        }
        rt_assignment = {}
        for s, c in assignment.items():
            s = int(s)
            c = int(c)
            if failed_controller is not None and c == int(failed_controller):
                continue
            if s in residual_switches:
                continue
            if c not in capacities:
                continue
            rt_assignment[s] = c

        for s, c in rt_assignment.items():
            if s == c:
                paths[(s, c)] = [s]
            else:
                try:
                    paths[(s, c)] = list(nx.shortest_path(UG, s, c, weight="weight"))
                except Exception:
                    paths[(s, c)] = []
        # Use the same capacity threshold that this recovery record reports.
        rt = compute_response_metrics(
            G, rt_assignment, loads, capacities, paths,
            round_trip=True, per_ctrl_ms=float(sync_delay_ms or 0.0),
            capacity_threshold=float(usable_threshold),
        )
        return rt
    except Exception:
        return {}


def process_failure_scenario(
    *,
    algorithm: str,
    topology_name: str,
    run_index: int,
    failed_controller: int,

    G,
    pos,
    switches,
    controllers,
    loads: Dict[int, float],
    capacities: Dict[int, float],
    usable_threshold: float,
    overload_threshold: float = 0.90,

    initial_assignment: Dict[int, int],

    # Use this for integral recovery algorithms
    recovery_assignment: Optional[Dict[int, int]] = None,

    # Use this for FT-FSM or another fractional algorithm
    fractional_assignment: Optional[
        Dict[int, Dict[int, float]]
    ] = None,

    residual_by_switch: Optional[Dict] = None,

    status: str = "SUCCESS",
    solve_time_sec: float = 0.0,
    objective_value=None,
    mip_gap=None,

    backup_controller: Optional[int] = None,
    backup_capacity: Optional[float] = None,

    output_root: str,
    comparison_csv_file: Optional[str] = None,
    make_plot: bool = True,
    file_tag: Optional[str] = None,

    # Common comparison/CSV metadata
    master_seed: Optional[int] = None,
    switch_seed: Optional[int] = None,
    run_number: Optional[int] = None,
    reassignment_scope: str = "orphan_only",
    pre_failure_response_time_ms: Optional[float] = None,
    post_failure_mean_switch_response_time_ms: Optional[float] = None,
    post_failure_max_switch_response_time_ms: Optional[float] = None,
    sync_delay_ms: float = 0.0,
    edge_caps: Optional[Dict] = None,
    link_usage: Optional[Dict] = None,
    msg_bits: float = 128.0,
    link_utilization_threshold: float = 0.90,
    write_csv: bool = True,
) -> Dict[str, Any]:
    """
    Common handler for one failed-controller scenario.

    Supports:
      - integral network-wide reassignment;
      - fractional switch-flow reassignment;
      - residual/unassigned load;
      - common metrics;
      - one common JSON;
      - integral or fractional recovery plots.
    """
    failed_controller = int(failed_controller)

    initial_assignment = {
        int(s): int(c)
        for s, c in initial_assignment.items()
    }

    recovery_assignment = {
        int(s): int(c)
        for s, c in (recovery_assignment or {}).items()
    }

    fractional_assignment = _normalize_fractional_plan(
        fractional_assignment
    )

    residual_by_switch = _normalize_residuals(
        residual_by_switch,
        loads,
    )

    is_fractional = bool(fractional_assignment)

    if is_fractional:
        dominant_recovery_assignment = (
            _dominant_assignment_from_fractional(
                initial_assignment=initial_assignment,
                fractional_assignment=fractional_assignment,
                failed_controller=failed_controller,
            )
        )

        post_failure_loads = _fractional_post_failure_loads(
            initial_assignment=initial_assignment,
            fractional_assignment=fractional_assignment,
            residual_by_switch=residual_by_switch,
            loads=loads,
            controllers=controllers,
            failed_controller=failed_controller,
        )

    else:
        dominant_recovery_assignment = (
            recovery_assignment
            if recovery_assignment
            else dict(initial_assignment)
        )

        post_failure_loads = _integral_post_failure_loads(
            dominant_recovery_assignment,
            loads,
            controllers,
            failed_controller,
        )

    # Complete initial load vector: every controller is represented.
    initial_loads = {int(c): 0.0 for c in controllers}
    for switch, controller in initial_assignment.items():
        initial_loads[int(controller)] = initial_loads.get(int(controller), 0.0) + float(
            loads.get(int(switch), 0.0)
        )

    orphan_switches = [
        int(s) for s, c in initial_assignment.items()
        if int(c) == failed_controller
    ]
    orphan_switch_set = set(orphan_switches)
    orphan_load = sum(float(loads.get(s, 0.0)) for s in orphan_switches)

    migration_stats = _migration_statistics(
        initial_assignment=initial_assignment,
        recovery_assignment=dominant_recovery_assignment,
        fractional_assignment=fractional_assignment,
        residual_by_switch=residual_by_switch,
        loads=loads,
        failed_controller=failed_controller,
    )

    # Every surviving controller must appear, including zero-load controllers.
    surviving_controllers = [
        int(c) for c in controllers if int(c) != failed_controller
    ]
    if backup_controller is not None:
        backup_controller = int(backup_controller)
        if backup_controller not in surviving_controllers:
            surviving_controllers.append(backup_controller)

    complete_post_failure_loads = {
        int(c): float(post_failure_loads.get(int(c), 0.0))
        for c in surviving_controllers
    }

    complete_capacities = {int(c): float(capacities.get(int(c), 0.0)) for c in surviving_controllers}
    if backup_controller is not None and backup_capacity is not None:
        complete_capacities[int(backup_controller)] = float(backup_capacity)

    # Requested utilization definition: actual load / raw controller capacity.
    utilization = {}
    for controller in surviving_controllers:
        cap = float(complete_capacities.get(controller, 0.0))
        utilization[controller] = (
            float(complete_post_failure_loads.get(controller, 0.0)) / cap
            if cap > 0.0 else float("inf")
        )

    overloaded_controllers = [
        int(c) for c, value in utilization.items()
        if value > float(overload_threshold) + TOL
    ]

    total_residual_load = sum(
        float(info["residual_load"]) for info in residual_by_switch.values()
    )
    total_orphan_load_reassigned = max(0.0, float(orphan_load) - float(total_residual_load))

    # Complete switch -> post-controller/load representation.
    post_assignment_with_loads = {}
    for switch in sorted(int(s) for s in switches):
        original = initial_assignment.get(switch)
        residual_load = float(residual_by_switch.get(switch, {}).get("residual_load", 0.0))
        entry = {
            "pre_controller": original,
            "post_controller": dominant_recovery_assignment.get(switch),
            "switch_load": float(loads.get(switch, 0.0)),
            "is_orphan": switch in orphan_switch_set,
            "residual_load": residual_load,
        }
        if switch in fractional_assignment:
            entry["post_fractional_controllers"] = fractional_assignment[switch]
        post_assignment_with_loads[switch] = entry

    # Post-failure RT if the caller did not supply it.
    post_caps_for_rt = dict(capacities)
    if backup_controller is not None and backup_capacity is not None:
        post_caps_for_rt[int(backup_controller)] = float(backup_capacity)
    if (
        post_failure_mean_switch_response_time_ms is None
        or post_failure_max_switch_response_time_ms is None
    ):
        rt_post = _shortest_path_response_metrics(
            G=G,
            assignment=dominant_recovery_assignment,
            loads=loads,
            capacities=post_caps_for_rt,
            sync_delay_ms=sync_delay_ms,
            usable_threshold=usable_threshold,
            failed_controller=failed_controller,
            residual_by_switch=residual_by_switch,
        )
        if post_failure_mean_switch_response_time_ms is None:
            post_failure_mean_switch_response_time_ms = rt_post.get(
                "mean_resp",
                rt_post.get("init_mean_rt_ms"),
            )
        if post_failure_max_switch_response_time_ms is None:
            post_failure_max_switch_response_time_ms = rt_post.get("max_resp")

    total_survivor_usable_capacity = sum(
        float(usable_threshold) * float(complete_capacities.get(c, 0.0))
        for c in surviving_controllers
    )
    total_switch_load = sum(float(loads.get(int(s), 0.0)) for s in switches)
    global_unaccommodated = (
        max(0.0, total_switch_load - total_survivor_usable_capacity)
        if str(reassignment_scope).lower() == "global" else 0.0
    )

    recovery_feasible = (
        "INFEASIBLE" not in str(status).upper()
        and "NO_SOLUTION" not in str(status).upper()
        and total_residual_load <= TOL
        and global_unaccommodated <= TOL
    )

    backup_assigned_load = (
        float(complete_post_failure_loads.get(int(backup_controller), 0.0))
        if backup_controller is not None else 0.0
    )
    backup_util = (
        backup_assigned_load / float(backup_capacity)
        if backup_controller is not None and backup_capacity not in (None, 0) else 0.0
    )

    new_controller_residual_by_switch = {}
    if backup_controller is not None:
        backup_controller_int = int(backup_controller)
        for switch in sorted(int(s) for s in switches):
            if (
                int(initial_assignment.get(switch, -1)) == failed_controller
                and int(dominant_recovery_assignment.get(switch, -1)) == backup_controller_int
            ):
                switch_load = float(loads.get(switch, 0.0))
                new_controller_residual_by_switch[switch] = {
                    "switch_load": switch_load,
                    "residual_fraction": 1.0,
                    "residual_load": switch_load,
                    "assigned_controller": backup_controller_int,
                }
    new_controller_residual_load = sum(
        float(info["residual_load"])
        for info in new_controller_residual_by_switch.values()
    )

    if link_usage is None and edge_caps:
        link_usage = _shortest_path_link_usage(
            G=G,
            assignment=dominant_recovery_assignment,
            loads=loads,
            msg_bits=float(msg_bits),
            failed_controller=failed_controller,
            residual_by_switch=residual_by_switch,
        )
    link_stats = _link_utilization_statistics(
        link_usage=link_usage,
        edge_caps=edge_caps,
        threshold=float(link_utilization_threshold),
    )

    record = {
        "algorithm": str(algorithm),
        "failed_controller": failed_controller,
        "status": str(status),
        "solve_time_sec": float(solve_time_sec),
        "objective_value": objective_value,
        "mip_gap": mip_gap,
        "initial_assignment": initial_assignment,
        "recovery_assignment_dominant": dominant_recovery_assignment,
        "fractional_assignment": fractional_assignment,
        "uses_fractional_recovery": is_fractional,
        "orphan_switches": orphan_switches,
        "orphan_switch_count": len(orphan_switches),
        "orphan_load": float(orphan_load),
        **migration_stats,
        "migration_cost": migration_stats["total_migrations"],
        "post_failure_loads": complete_post_failure_loads,
        "post_failure_utilization": utilization,
        "post_failure_switch_assignment_with_loads": post_assignment_with_loads,
        "max_post_failure_utilization": max(utilization.values()) if utilization else 0.0,
        "post_failure_variance": _variance(complete_post_failure_loads.values()),
        "overloaded_controllers": overloaded_controllers,
        "residual_by_switch": residual_by_switch,
        "residual_switch_count": len(residual_by_switch),
        "total_residual_load": float(total_residual_load),
        "total_unaccommodated_residual_load": float(total_residual_load),
        "new_controller_residual_by_switch": new_controller_residual_by_switch,
        "new_controller_residual_load": float(new_controller_residual_load),
        "total_orphan_load_reassigned": total_orphan_load_reassigned,
        "backup_controller": backup_controller,
        "backup_capacity": backup_capacity,
        "reassignment_scope": str(reassignment_scope),
        "recovery_feasible": bool(recovery_feasible),
        "pre_failure_response_time_ms": pre_failure_response_time_ms,
        "post_failure_mean_switch_response_time_ms": post_failure_mean_switch_response_time_ms,
        "post_failure_max_switch_response_time_ms": post_failure_max_switch_response_time_ms,
        "link_usage_post_migration": _normalize_edge_map(link_usage),
        "edge_capacities": _normalize_edge_map(edge_caps),
        "max_link_utilization_post_migration": link_stats["max_link_utilization"],
        "violated_links_post_migration": link_stats["violated_links"],
        "violated_link_count_post_migration": link_stats["violated_link_count"],
        "link_utilization_threshold": float(link_utilization_threshold),
    }

    if write_csv:
        os.makedirs(output_root, exist_ok=True)

        # A caller may route every model/run/failure into one experiment-level
        # comparison file while keeping plots and JSON artifacts per run/model.
        csv_file = comparison_csv_file or os.path.join(
            output_root, "comparison_recovery.csv"
        )
        csv_row = {
            "method_model": str(algorithm),
            "switch_seed": switch_seed,
            "master_seed": master_seed,
            "run_index": int(run_index),
            "run_number": int(run_number if run_number is not None else run_index + 1),
            "topology": str(topology_name),
            "status": str(status),
            "objective_value": objective_value,
            "total_number_of_switches": len(switches),
            "total_number_of_controllers": len(controllers),
            "switch_loads": _json_cell({int(s): float(loads.get(int(s), 0.0)) for s in switches}),
            "controller_capacities": _json_cell({int(c): float(capacities.get(int(c), 0.0)) for c in controllers}),
            "initial_switch_assignment": _json_cell(initial_assignment),
            "recovery_assignment_dominant": _json_cell(dominant_recovery_assignment),
            "controller_loads_initial": _json_cell(initial_loads),
            "max_minus_min_controller_loads_initial": (
                max(initial_loads.values()) - min(initial_loads.values()) if initial_loads else 0.0
            ),
            "max_minus_min_controller_loads_final": (
                max(complete_post_failure_loads.values())
                - min(complete_post_failure_loads.values())
                if complete_post_failure_loads else 0.0
            ),
            "max_controller_load_initial": max(initial_loads.values()) if initial_loads else 0.0,
            "max_controller_load_final": (
                max(complete_post_failure_loads.values())
                if complete_post_failure_loads else 0.0
            ),
            "min_controller_load_initial": min(initial_loads.values()) if initial_loads else 0.0,
            "min_controller_load_final": (
                min(complete_post_failure_loads.values())
                if complete_post_failure_loads else 0.0
            ),
            "variance_controller_loads_initial": _variance(initial_loads.values()),
            "solve_time_sec": float(solve_time_sec),
            "migration_cost": migration_stats["total_migrations"],
            "number_of_migrations": migration_stats["total_migrations"],
            "number_of_switches_migrated": migration_stats["total_migrations"],
            "migrated_switches": _json_cell(migration_stats["migrated_switches"]),
            "orphan_migrations": migration_stats["orphan_migrations"],
            "non_orphan_migrations": migration_stats["non_orphan_migrations"],
            "failed_controller_number": failed_controller,
            "orphan_switches": _json_cell(orphan_switches),
            "orphan_count": len(orphan_switches),
            "post_failure_variance": _variance(complete_post_failure_loads.values()),
            "pre_failure_response_time_ms": pre_failure_response_time_ms,
            "post_failure_mean_switch_response_time_ms": post_failure_mean_switch_response_time_ms,
            "post_failure_max_switch_response_time_ms": post_failure_max_switch_response_time_ms,
            "controller_loads_post_migration": _json_cell(complete_post_failure_loads),
            "controller_utilization_post_migration": _json_cell(utilization),
            "post_failure_switch_assignment_with_loads": _json_cell(post_assignment_with_loads),
            "uses_fractional_recovery": "YES" if is_fractional else "NO",
            "fractional_switch_count": len(fractional_assignment),
            "fractional_controller_allocations": _json_cell(fractional_assignment),
            "fractional_load_allocations": _json_cell({
                int(s): {
                    int(c): float(loads.get(int(s), 0.0)) * float(fraction)
                    for c, fraction in targets.items()
                }
                for s, targets in fractional_assignment.items()
            }),
            "fractional_assignment_details": _json_cell({
                int(s): {
                    "switch_load": float(loads.get(int(s), 0.0)),
                    "controller_fractions": {
                        int(c): float(fraction) for c, fraction in targets.items()
                    },
                    "controller_load_allocations": {
                        int(c): float(loads.get(int(s), 0.0)) * float(fraction)
                        for c, fraction in targets.items()
                    },
                    "assigned_fraction": sum(float(fraction) for fraction in targets.values()),
                    "assigned_load": sum(
                        float(loads.get(int(s), 0.0)) * float(fraction)
                        for fraction in targets.values()
                    ),
                }
                for s, targets in fractional_assignment.items()
            }),
            "residual_by_switch": _json_cell(residual_by_switch),
            "overloaded_controllers": _json_cell(overloaded_controllers),
            "total_orphan_load": float(orphan_load),
            "total_orphan_load_reassigned": total_orphan_load_reassigned,
            "total_orphan_load_residual": float(total_residual_load + new_controller_residual_load),
            "total_orphan_load_unaccommodated": float(total_residual_load),
            "new_controller_residual_by_switch": _json_cell(new_controller_residual_by_switch),
            "new_controller_planned": "YES" if backup_controller is not None else "NO",
            "new_controller_id": backup_controller,
            "new_controller_capacity": backup_capacity if backup_capacity is not None else 0.0,
            "total_controller_capacity_after_plan": sum(complete_capacities.values()),
            "new_controller_total_assigned_load": backup_assigned_load,
            "new_controller_utilization": backup_util,
            "max_link_utilization_post_migration": link_stats["max_link_utilization"],
            "violated_links_post_migration": _json_cell(link_stats["violated_links"]),
            "violated_link_count_post_migration": link_stats["violated_link_count"],
            "link_utilization_threshold": float(link_utilization_threshold),
            "reassignment_scope": str(reassignment_scope),
            "global_total_switch_load_unaccommodated": global_unaccommodated,
            "feasibility": "FEASIBLE" if recovery_feasible else "INFEASIBLE",
            "mip_gap": mip_gap,
        }
        _append_recovery_csv(csv_file, csv_row)
        record["recovery_csv"] = csv_file

    failure_log_file = os.path.join(
        output_root,
        "failure_analysis.json",
    )

    _append_failure_record(
        output_file=failure_log_file,
        topology_name=topology_name,
        run_index=run_index,
        controllers=controllers,
        algorithm=algorithm,
        failed_controller=failed_controller,
        record=record,
    )

    if make_plot and pos is not None:
        # output_root is already the model artifact directory.  Do not append
        # the model name again (MCF_ARC/MCF_ARC/failure_plots was accidental).
        plot_directory = os.path.join(output_root, "failure_plots")
        os.makedirs(plot_directory, exist_ok=True)

        # Keep the failed controller in the left (pre-failure) panel.  The
        # common plotting helper removes it only from the recovery panel.
        plot_controllers = [int(c) for c in controllers]

        plot_capacities = {
            int(c): float(v)
            for c, v in capacities.items()
        }

        if backup_controller is not None:
            backup_controller = int(backup_controller)

            if backup_controller not in plot_controllers:
                plot_controllers.append(backup_controller)

            if backup_capacity is not None:
                plot_capacities[backup_controller] = float(
                    backup_capacity
                )

        common_tag = (
            file_tag
            or (
                f"{algorithm}_run{int(run_index):03d}_"
                f"failC{failed_controller}"
            )
        )

        if is_fractional:
            plot_path = plot_fractional_traffic_recovery_comparison(
                G=G,
                pos=pos,
                switches=switches,
                controllers=plot_controllers,

                final_assign=initial_assignment,
                fractional_recovery=fractional_assignment,
                residual_by_switch=residual_by_switch,

                loads=loads,
                final_loads=initial_loads,
                recovery_loads=post_failure_loads,

                topology_name=topology_name,
                save_dir=plot_directory,
                controller_capacity=plot_capacities,

                failed_controller=failed_controller,
                backup_controller=backup_controller,
                backup_capacity=backup_capacity,
                migration_count=migration_stats["total_migrations"],
                capacity_threshold=overload_threshold,

                file_tag=common_tag,
            )
            record["plot_path"] = plot_path

        else:
            integral_tag = common_tag.rsplit("_failC", 1)[0]
            plot_final_vs_recovery_assignment(
                G=G,
                pos=pos,
                switches=switches,
                controllers=plot_controllers,

                final_assign=initial_assignment,
                recovery_assign=dominant_recovery_assignment,

                loads=loads,
                final_loads=initial_loads,
                recovery_loads=post_failure_loads,

                topology_name=topology_name,
                save_dir=plot_directory,
                controller_capacity=plot_capacities,

                failed_controller=failed_controller,
                backup_controller=backup_controller,
                backup_capacity=backup_capacity,
                migration_count=migration_stats["total_migrations"],
                capacity_threshold=overload_threshold,

                file_tag=integral_tag,
            )
            # The integral plotting helper writes JPG/PDF/SVG variants.
            record["plot_directory"] = plot_directory

    return record
