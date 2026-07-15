from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from plotting import (
    plot_final_vs_recovery_assignment,
    plot_final_vs_fractional_recovery_assignment,
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

        normalized[switch] = {}

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
                normalized[switch][controller] = fraction

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
    make_plot: bool = True,
    file_tag: Optional[str] = None,
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

    initial_loads = defaultdict(float)

    for switch, controller in initial_assignment.items():
        initial_loads[int(controller)] += float(
            loads.get(int(switch), 0.0)
        )

    initial_loads = dict(initial_loads)

    orphan_switches = [
        int(s)
        for s, c in initial_assignment.items()
        if int(c) == failed_controller
    ]

    orphan_load = sum(
        float(loads.get(s, 0.0))
        for s in orphan_switches
    )

    migration_stats = _migration_statistics(
        initial_assignment=initial_assignment,
        recovery_assignment=dominant_recovery_assignment,
        fractional_assignment=fractional_assignment,
        residual_by_switch=residual_by_switch,
        loads=loads,
        failed_controller=failed_controller,
    )

    utilization = {}

    for controller, load_value in post_failure_loads.items():
        capacity = float(capacities.get(controller, 0.0))

        if (
            backup_controller is not None
            and int(controller) == int(backup_controller)
            and backup_capacity is not None
        ):
            capacity = float(backup_capacity)

        usable_capacity = capacity * float(usable_threshold)

        utilization[int(controller)] = (
            float(load_value) / usable_capacity
            if usable_capacity > 0
            else float("inf")
        )

    overloaded_controllers = [
        int(c)
        for c, value in utilization.items()
        if value > 1.0 + TOL
    ]

    total_residual_load = sum(
        float(info["residual_load"])
        for info in residual_by_switch.values()
    )

    recovery_feasible = (
        "INFEASIBLE" not in str(status).upper()
        and "NO_SOLUTION" not in str(status).upper()
        and not overloaded_controllers
        and total_residual_load <= TOL
    )

    record = {
        "algorithm": str(algorithm),
        "failed_controller": failed_controller,
        "status": str(status),

        "solve_time_sec": float(solve_time_sec),
        "objective_value": objective_value,
        "mip_gap": mip_gap,

        "initial_assignment": initial_assignment,
        "recovery_assignment_dominant":
            dominant_recovery_assignment,

        "fractional_assignment": fractional_assignment,
        "uses_fractional_recovery": is_fractional,

        "orphan_switches": orphan_switches,
        "orphan_switch_count": len(orphan_switches),
        "orphan_load": float(orphan_load),

        **migration_stats,

        "post_failure_loads": post_failure_loads,
        "post_failure_utilization": utilization,
        "max_post_failure_utilization": (
            max(utilization.values())
            if utilization
            else 0.0
        ),
        "overloaded_controllers": overloaded_controllers,

        "residual_by_switch": residual_by_switch,
        "residual_switch_count": len(residual_by_switch),
        "total_residual_load": float(total_residual_load),

        "backup_controller": backup_controller,
        "backup_capacity": backup_capacity,

        "recovery_feasible": bool(recovery_feasible),
    }

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
        plot_directory = os.path.join(
            output_root,
            str(algorithm),
            "failure_plots",
        )
        os.makedirs(plot_directory, exist_ok=True)

        plot_controllers = [
            int(c)
            for c in controllers
            if int(c) != failed_controller
        ]

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
            plot_final_vs_fractional_recovery_assignment(
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

                file_tag=common_tag,
            )

        else:
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

                file_tag=common_tag,
            )

    return record