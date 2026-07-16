from __future__ import annotations

"""
Baseline 3: Fast and Load-aware Controller Failover (FLCF), Fang et al. (2016).

Implementation policy
---------------------
* A separate proactive recovery plan is computed for each single-controller
  failure.
* Only switches controlled by the failed controller are genes/migration
  targets. Assignments of all surviving switches remain fixed.
* The chromosome fitness follows paper Eq. (1): the sum of population-
  normalized average switch-controller delay and population-normalized
  controller-load standard deviation. Lower is better.
* Paper Eq. (2) is used during crossover to choose between the two parent
  controller genes using normalized switch-controller delay plus normalized
  current controller load. Lower is better.
* Initial population size follows Eq. (3), with the paper's lower bound k=50.
* SUBSET size follows Eq. (4): largest d with d + C(d,2) <= k.
* Tournament selection, all-pairs crossover, conditional mutation, elitist
  refill, and mutation probability 0.4 follow the paper's flowchart and
  Algorithms 1-2.

The wrapper deliberately matches baseline2_PREF's ten-value pipeline return
and delegates CSV/JSON/plot generation to process_failure_scenario so its
post-optimization outputs have the same schema and MCF-ARC-style plots.
"""

import inspect
import json
import math
import os
import random
import time
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx

from failure_scenario_handler import process_failure_scenario

TOL = 1e-12


# ---------------------------------------------------------------------------
# Common metrics
# ---------------------------------------------------------------------------
def compute_controller_loads(
    assignment: Mapping[int, int],
    controllers: Iterable[int],
    loads: Mapping[int, float],
) -> Dict[int, float]:
    result = {int(c): 0.0 for c in controllers}
    for switch, controller in assignment.items():
        controller = int(controller)
        if controller in result:
            result[controller] += float(loads.get(int(switch), 0.0))
    return result


def load_std(controller_loads: Mapping[int, float]) -> float:
    values = [float(v) for v in controller_loads.values()]
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def load_deviation(controller_loads: Mapping[int, float]) -> float:
    values = [float(v) for v in controller_loads.values()]
    return max(values) - min(values) if values else 0.0


def _safe_normalize(value: float, minimum: float, maximum: float) -> float:
    denominator = float(maximum) - float(minimum)
    if abs(denominator) <= TOL:
        return 0.0
    return (float(value) - float(minimum)) / denominator


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Delay model used by paper Eq. (1) and Eq. (2)
# ---------------------------------------------------------------------------
def _pair_delay(
    G: nx.Graph,
    switch: int,
    controller: int,
    *,
    dij: Optional[Mapping] = None,
    paths_sc: Optional[Mapping] = None,
) -> float:
    """Return one-way switch-controller delay/cost in the caller's units."""
    switch = int(switch)
    controller = int(controller)

    if switch == controller:
        return 0.0

    if dij is not None:
        if (switch, controller) in dij:
            return float(dij[(switch, controller)])
        nested = dij.get(switch) if hasattr(dij, "get") else None
        if isinstance(nested, Mapping) and controller in nested:
            return float(nested[controller])

    path = None
    if paths_sc is not None:
        path = paths_sc.get((switch, controller))

    if not path:
        try:
            path = nx.shortest_path(
                G.to_undirected(),
                source=switch,
                target=controller,
                weight="weight",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G[u][v]
        if "latency_sec" in data:
            total += float(data["latency_sec"]) * 1000.0
        else:
            total += float(data.get("weight", 1.0))
    return total


# ---------------------------------------------------------------------------
# Paper population/subset sizes
# ---------------------------------------------------------------------------
def flcf_population_size(num_orphans: int, num_survivors: int) -> int:
    """Paper Eq. (3): k=max(50, ceil(log(n^m)))=max(50,ceil(m log n))."""
    m = max(0, int(num_orphans))
    n = max(1, int(num_survivors))
    log_search_space = 0.0 if m == 0 or n == 1 else m * math.log(n)
    return max(50, int(math.ceil(log_search_space)))


def flcf_subset_size(population_size: int) -> int:
    """Paper Eq. (4): largest d satisfying d + C(d,2) <= k."""
    k = max(1, int(population_size))
    d = 1
    while (d + 1) + ((d + 1) * d) // 2 <= k:
        d += 1
    return d


# ---------------------------------------------------------------------------
# One failed-controller scenario
# ---------------------------------------------------------------------------
def run_flcf_single_failure(
    G,
    switches,
    controllers,
    loads,
    capacities,
    final_assign,
    failed_controller,
    *,
    dij=None,
    paths_sc=None,
    generations: int = 150,
    population_size: Optional[int] = None,
    mutation_rate: float = 0.40,
    tournament_size: int = 3,
    usable_threshold: float = 0.90,
    enforce_capacity: bool = False,
    capacity_penalty: float = 1e9,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute FLCF's proactive recovery plan for one controller failure."""
    rng = random.Random(int(seed))

    switches = [int(s) for s in switches]
    controllers = [int(c) for c in controllers]
    failed_controller = int(failed_controller)
    final_assign = {int(s): int(c) for s, c in final_assign.items()}
    loads = {int(s): float(v) for s, v in loads.items()}
    capacities = {int(c): float(v) for c, v in capacities.items()}

    survivors = [c for c in controllers if c != failed_controller]
    if not survivors:
        return {
            "failed_controller": failed_controller,
            "status": "NO_SURVIVING_CONTROLLER",
            "failed_switches": [],
            "surviving_controllers": [],
            "recovery_plan": {},
            "recovery_assign": {},
            "recovery_loads": {},
            "average_delay": float("inf"),
            "sigma": float("inf"),
            "objective": float("inf"),
            "fitness_with_penalty": float("inf"),
            "reassignment_cost": 0,
            "capacity_violation": float("inf"),
            "population_size": 0,
            "subset_size": 0,
        }

    # FLCF migration targets: ONLY switches controlled by failed controller.
    failed_switches = [
        s for s in switches if final_assign.get(s) == failed_controller
    ]
    fixed_assign = {
        s: c for s, c in final_assign.items() if c != failed_controller
    }

    def decode(chromosome: Mapping[int, int]) -> Dict[int, int]:
        recovered = dict(fixed_assign)
        recovered.update({int(s): int(c) for s, c in chromosome.items()})
        return recovered

    delay_table = {
        (s, c): _pair_delay(
            G,
            s,
            c,
            dij=dij,
            paths_sc=paths_sc,
        )
        for s in failed_switches
        for c in survivors
    }

    def make_chromosome() -> Dict[int, int]:
        return {s: rng.choice(survivors) for s in failed_switches}

    def chromosome_metrics(chromosome: Mapping[int, int]) -> Tuple[float, float]:
        recovered = decode(chromosome)
        orphan_delays = [delay_table[(s, int(chromosome[s]))] for s in failed_switches]
        average_delay = (
            sum(orphan_delays) / len(orphan_delays)
            if orphan_delays
            else 0.0
        )
        sigma = load_std(compute_controller_loads(recovered, survivors, loads))
        return float(average_delay), float(sigma)

    def capacity_violation(chromosome: Mapping[int, int]) -> float:
        recovered = decode(chromosome)
        controller_loads = compute_controller_loads(recovered, survivors, loads)
        return sum(
            max(
                0.0,
                controller_loads[c]
                - float(usable_threshold) * float(capacities.get(c, 0.0)),
            )
            for c in survivors
        )

    def evaluate_population(population: List[Dict[int, int]]) -> List[Dict[str, float]]:
        raw = [chromosome_metrics(chromosome) for chromosome in population]
        delays = [value[0] for value in raw]
        sigmas = [value[1] for value in raw]
        delay_min, delay_max = min(delays), max(delays)
        sigma_min, sigma_max = min(sigmas), max(sigmas)

        evaluated = []
        for chromosome, (average_delay, sigma) in zip(population, raw):
            # Paper Eq. (1), lower is better.
            objective = (
                _safe_normalize(average_delay, delay_min, delay_max)
                + _safe_normalize(sigma, sigma_min, sigma_max)
            )
            cap_violation = capacity_violation(chromosome)
            fitness = objective
            if enforce_capacity:
                fitness += float(capacity_penalty) * cap_violation
            evaluated.append(
                {
                    "average_delay": average_delay,
                    "sigma": sigma,
                    "objective": objective,
                    "capacity_violation": cap_violation,
                    "fitness": fitness,
                }
            )
        return evaluated

    if not failed_switches:
        recovery_assign = dict(final_assign)
        recovery_loads = compute_controller_loads(recovery_assign, survivors, loads)
        return {
            "failed_controller": failed_controller,
            "status": "NO_AFFECTED_SWITCHES",
            "failed_switches": [],
            "surviving_controllers": survivors,
            "recovery_plan": {},
            "recovery_assign": recovery_assign,
            "recovery_loads": recovery_loads,
            "average_delay": 0.0,
            "sigma": load_std(recovery_loads),
            "load_deviation": load_deviation(recovery_loads),
            "objective": 0.0,
            "fitness_with_penalty": 0.0,
            "reassignment_cost": 0,
            "capacity_violation": 0.0,
            "population_size": 0,
            "subset_size": 0,
        }

    k = (
        flcf_population_size(len(failed_switches), len(survivors))
        if population_size is None
        else max(2, int(population_size))
    )
    subset_size = flcf_subset_size(k)
    population = [make_chromosome() for _ in range(k)]

    best_chromosome: Optional[Dict[int, int]] = None
    best_raw_key = (float("inf"), float("inf"), float("inf"))
    best_evaluation: Dict[str, float] = {}

    def update_best(pop: List[Dict[int, int]], evaluations: List[Dict[str, float]]) -> None:
        nonlocal best_chromosome, best_raw_key, best_evaluation
        for chromosome, evaluation in zip(pop, evaluations):
            key = (
                float(evaluation["fitness"]),
                float(evaluation["average_delay"]),
                float(evaluation["sigma"]),
            )
            if key < best_raw_key:
                best_raw_key = key
                best_chromosome = dict(chromosome)
                best_evaluation = dict(evaluation)

    def tournament_select(
        pop: List[Dict[int, int]],
        evaluations: List[Dict[str, float]],
    ) -> int:
        indices = rng.sample(range(len(pop)), k=min(tournament_size, len(pop)))
        return min(indices, key=lambda idx: evaluations[idx]["fitness"])

    def crossover(
        parent_a: Dict[int, int],
        parent_b: Dict[int, int],
    ) -> Dict[int, int]:
        """Paper Algorithm 1 using gene evaluation from Eq. (2)."""
        child: Dict[int, int] = {}

        # Current loads start with all surviving switches fixed.
        current_loads = compute_controller_loads(fixed_assign, survivors, loads)
        all_pair_delays = [delay_table[(s, c)] for s in failed_switches for c in survivors]
        delay_min = min(all_pair_delays) if all_pair_delays else 0.0
        delay_max = max(all_pair_delays) if all_pair_delays else 0.0

        for switch in failed_switches:
            controller_a = int(parent_a[switch])
            controller_b = int(parent_b[switch])

            candidate_controllers = [controller_a, controller_b]
            current_values = [current_loads.get(c, 0.0) for c in survivors]
            load_min = min(current_values) if current_values else 0.0
            load_max = max(current_values) if current_values else 0.0

            def gene_evaluation(controller: int) -> float:
                # Paper Eq. (2), lower is better.
                return (
                    _safe_normalize(
                        delay_table[(switch, controller)],
                        delay_min,
                        delay_max,
                    )
                    + _safe_normalize(
                        current_loads.get(controller, 0.0),
                        load_min,
                        load_max,
                    )
                )

            chosen = min(candidate_controllers, key=gene_evaluation)
            child[switch] = chosen
            current_loads[chosen] = (
                current_loads.get(chosen, 0.0) + float(loads.get(switch, 0.0))
            )

        return child

    def mutate(chromosome: Dict[int, int]) -> Dict[int, int]:
        mutated = dict(chromosome)
        for switch in failed_switches:
            if rng.random() < float(mutation_rate):
                mutated[switch] = rng.choice(survivors)
        return mutated

    for generation in range(max(1, int(generations))):
        evaluations = evaluate_population(population)
        update_best(population, evaluations)

        # Step 4: tournament-selected SUBSET of size delta(k).
        subset_indices: List[int] = []
        while len(subset_indices) < subset_size:
            idx = tournament_select(population, evaluations)
            if idx not in subset_indices:
                subset_indices.append(idx)
        subset = [dict(population[idx]) for idx in subset_indices]

        # Step 5: one child for every unordered pair in SUBSET.
        children: List[Dict[int, int]] = []
        for parent_a, parent_b in combinations(subset, 2):
            child = crossover(parent_a, parent_b)

            local_population = [parent_a, parent_b, child]
            local_eval = evaluate_population(local_population)
            child_better = local_eval[2]["fitness"] < min(
                local_eval[0]["fitness"], local_eval[1]["fitness"]
            )
            if not child_better:
                child = mutate(child)
            children.append(child)

        # Steps 6-7: SUBSET + C&MSET, then best PRESET members to size k.
        next_population: List[Dict[int, int]] = subset + children
        ranked_indices = sorted(
            range(len(population)),
            key=lambda idx: evaluations[idx]["fitness"],
        )
        for idx in ranked_indices:
            if len(next_population) >= k:
                break
            next_population.append(dict(population[idx]))

        # Defensive fill only if duplicate/edge cases make it short.
        while len(next_population) < k:
            next_population.append(make_chromosome())

        population = next_population[:k]

        if verbose and generation % 20 == 0:
            print(
                f"[FLCF-GA] failed=C{failed_controller} "
                f"generation={generation} best={best_raw_key[0]:.6f}"
            )

    final_evaluations = evaluate_population(population)
    update_best(population, final_evaluations)
    assert best_chromosome is not None

    recovery_assign = decode(best_chromosome)
    recovery_loads = compute_controller_loads(recovery_assign, survivors, loads)
    cap_violation = capacity_violation(best_chromosome)

    status = "SUCCESS"
    if enforce_capacity and cap_violation > 1e-9:
        status = "CAPACITY_VIOLATED"

    return {
        "failed_controller": failed_controller,
        "status": status,
        "failed_switches": failed_switches,
        "surviving_controllers": survivors,
        "recovery_plan": best_chromosome,
        "recovery_assign": recovery_assign,
        "recovery_loads": recovery_loads,
        "average_delay": best_evaluation.get("average_delay", 0.0),
        "sigma": load_std(recovery_loads),
        "load_deviation": load_deviation(recovery_loads),
        "objective": best_evaluation.get("objective", 0.0),
        "fitness_with_penalty": best_evaluation.get("fitness", 0.0),
        "reassignment_cost": sum(
            1
            for s in switches
            if final_assign.get(s) != recovery_assign.get(s)
        ),
        "capacity_violation": cap_violation,
        "population_size": k,
        "subset_size": subset_size,
        "mutation_rate": mutation_rate,
    }


# ---------------------------------------------------------------------------
# All single-controller failures
# ---------------------------------------------------------------------------
def run_flcf_all_failures(
    G,
    switches,
    controllers,
    loads,
    capacities,
    final_assign,
    *,
    dij=None,
    paths_sc=None,
    generations=150,
    population_size=None,
    mutation_rate=0.40,
    tournament_size=3,
    usable_threshold=0.90,
    enforce_capacity=False,
    seed=42,
    verbose=False,
) -> Dict[int, Dict[str, Any]]:
    results = {}
    for index, failed_controller in enumerate(controllers):
        results[int(failed_controller)] = run_flcf_single_failure(
            G=G,
            switches=switches,
            controllers=controllers,
            loads=loads,
            capacities=capacities,
            final_assign=final_assign,
            failed_controller=failed_controller,
            dij=dij,
            paths_sc=paths_sc,
            generations=generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            usable_threshold=usable_threshold,
            enforce_capacity=enforce_capacity,
            seed=int(seed) + index,
            verbose=verbose,
        )
    return results


# ---------------------------------------------------------------------------
# Compatibility call to common post-processing handler
# ---------------------------------------------------------------------------
def _call_common_handler(**kwargs):
    """Pass only arguments supported by the installed handler version."""
    signature = inspect.signature(process_failure_scenario)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return process_failure_scenario(**accepted)


def _build_shortest_paths(G, assignment, cost_mode="weight"):
    paths_pair = {}
    paths_switch = {}
    weight = None if cost_mode == "hops" else "weight"
    graph = G.to_undirected()
    for switch, controller in assignment.items():
        try:
            path = (
                [int(switch)]
                if int(switch) == int(controller)
                else nx.shortest_path(graph, int(switch), int(controller), weight=weight)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = []
        paths_pair[(int(switch), int(controller))] = path
        paths_switch[int(switch)] = path
    return paths_pair, paths_switch


# ---------------------------------------------------------------------------
# Drop-in pipeline wrapper (same ten-value structure as baseline2 PREF)
# ---------------------------------------------------------------------------
def run_baseline3_flcf_exact(
    G,
    switches,
    controllers,
    loads,
    capacities,
    init_assign,
    dij=None,
    paths_sc=None,
    msg_bits=None,
    usable_threshold=0.90,
    overload_threshold=0.90,
    alpha=0.5,  # accepted only for call compatibility; not used by FLCF Eq. (1)
    gamma=0.8,  # accepted only for call compatibility; FLCF has no migration budget
    population_size=None,
    generations=150,
    mutation_rate=0.40,
    default_link_failure_prob=0.01,  # accepted for call compatibility; unused
    default_node_failure_prob=0.01,  # accepted for call compatibility; unused
    probability_aggregate="max",    # accepted for call compatibility; unused
    enforce_capacity=False,
    output_dir=None,
    seed=42,
    verbose=False,
    topology_name=None,
    run_index=0,
    plot_recovery=True,
    plot_pos=None,
    master_seed=None,
    switch_seed=None,
    run_number=None,
    pre_failure_response_time_ms=None,
    sync_delay_ms=0.0,
    comparison_csv_file=None,
    cost_mode="weight",
):
    """Run FLCF for all single-controller failures and match PREF outputs."""
    start = time.perf_counter()
    final_assign = {int(s): int(c) for s, c in init_assign.items()}
    final_loads = compute_controller_loads(final_assign, controllers, loads)

    results = run_flcf_all_failures(
        G=G,
        switches=switches,
        controllers=controllers,
        loads=loads,
        capacities=capacities,
        final_assign=final_assign,
        dij=dij,
        paths_sc=paths_sc,
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        usable_threshold=overload_threshold,
        enforce_capacity=enforce_capacity,
        seed=seed,
        verbose=verbose,
    )
    solve_time = time.perf_counter() - start

    scenario_records: Dict[int, Dict[str, Any]] = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "flcf_ga_full_log.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(results), handle, indent=2)

        for failed_controller, result in results.items():
            scenario_records[int(failed_controller)] = _call_common_handler(
                algorithm="FLCF_GA_2016",
                topology_name=topology_name or "topology",
                run_index=int(run_index),
                failed_controller=int(failed_controller),
                G=G,
                pos=plot_pos,
                switches=switches,
                controllers=controllers,
                loads=loads,
                capacities=capacities,
                usable_threshold=float(overload_threshold),
                initial_assignment=final_assign,
                recovery_assignment=result.get("recovery_assign", {}),
                residual_by_switch={},
                status=result.get("status", "UNKNOWN"),
                solve_time_sec=float(solve_time),
                objective_value=result.get("objective"),
                mip_gap=None,
                output_root=output_dir,
                comparison_csv_file=comparison_csv_file,
                make_plot=bool(plot_recovery),
                file_tag=(
                    f"FLCF_run{int(run_index):03d}_"
                    f"failC{int(failed_controller)}"
                ),
                master_seed=master_seed,
                switch_seed=switch_seed if switch_seed is not None else seed,
                run_number=run_number,
                reassignment_scope="orphan_only",
                pre_failure_response_time_ms=pre_failure_response_time_ms,
                sync_delay_ms=sync_delay_ms,
                write_csv=True,
            )
        json_log = json_path
    else:
        json_log = None

    objectives = [float(result.get("objective", 0.0)) for result in results.values()]
    objective_value = sum(objectives) / len(objectives) if objectives else 0.0

    migration_counts = [
        int(result.get("reassignment_cost", 0)) for result in results.values()
    ]
    migration_count = (
        sum(migration_counts) / len(migration_counts)
        if migration_counts
        else 0.0
    )

    usage = {
        int(c): (
            float(final_loads.get(int(c), 0.0)) / float(capacities[int(c)])
            if float(capacities.get(int(c), 0.0)) > 0.0
            else 0.0
        )
        for c in controllers
    }

    statuses = [result.get("status", "UNKNOWN") for result in results.values()]
    if all(status in ("SUCCESS", "NO_AFFECTED_SWITCHES") for status in statuses):
        status = "SUCCESS"
    elif any(status == "CAPACITY_VIOLATED" for status in statuses):
        status = "CAPACITY_VIOLATED"
    else:
        status = "PARTIAL"

    paths_pair, paths_switch = _build_shortest_paths(
        G,
        final_assign,
        cost_mode=cost_mode,
    )

    meta = {
        "algorithm": "FLCF_GA_2016",
        "paper": "A Fast and Load-aware Controller Failover Mechanism for Software-Defined Networks",
        "reassignment_scope": "orphan_only",
        "fitness": "normalized_average_delay + normalized_controller_load_std",
        "gene_evaluation": "normalized_delay + normalized_current_controller_load",
        "generations": generations,
        "population_size_override": population_size,
        "mutation_rate": mutation_rate,
        "enforce_capacity": enforce_capacity,
        "solve_time_sec": solve_time,
        "all_failure_results": results,
        "failure_scenarios": scenario_records,
        "json_log": json_log,
        "status_per_failure": statuses,
    }

    return (
        final_assign,
        paths_pair,
        paths_switch,
        final_loads,
        objective_value,
        migration_count,
        usage,
        None,  # GA has no MIP gap
        status,
        meta,
    )


# Convenient aliases.
run_baseline_flcf_exact = run_baseline3_flcf_exact
run_baseline3_FLCF_exact = run_baseline3_flcf_exact


__all__ = [
    "run_flcf_single_failure",
    "run_flcf_all_failures",
    "run_baseline3_flcf_exact",
    "run_baseline_flcf_exact",
    "run_baseline3_FLCF_exact",
]
