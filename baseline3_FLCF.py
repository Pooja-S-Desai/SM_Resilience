from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from failure_scenario_handler import process_failure_scenario


def _controller_loads(assign: Dict[int, int], controllers, loads) -> Dict[int, float]:
    out = {int(c): 0.0 for c in controllers}
    for s, c in assign.items():
        if int(c) in out:
            out[int(c)] += float(loads.get(int(s), 0.0))
    return out


def _load_std(loads_by_controller: Dict[int, float]) -> float:
    values = list(loads_by_controller.values())
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _safe_normalized(value: float, minimum: float, maximum: float) -> float:
    denominator = maximum - minimum
    if abs(denominator) <= 1e-12:
        return 0.0
    return (value - minimum) / denominator


def _build_shortest_paths(G, assignment: Dict[int, int], cost_mode: str = "weight"):
    weight = None if cost_mode == "hops" else "weight"
    UG = G.to_undirected()
    pair_paths = {}
    switch_paths = {}
    for s, c in assignment.items():
        if int(s) == int(c):
            path = [int(s)]
        else:
            try:
                path = nx.shortest_path(UG, int(s), int(c), weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = []
        pair_paths[(int(s), int(c))] = path
        switch_paths[int(s)] = path
    return pair_paths, switch_paths


def _distance_lookup(G, s: int, c: int, dij=None, cost_mode: str = "weight") -> float:
    if dij is not None:
        if isinstance(dij.get(s), dict):
            value = dij.get(s, {}).get(c)
        else:
            value = dij.get((s, c))
        if value is not None:
            return float(value)
    try:
        return float(nx.shortest_path_length(
            G.to_undirected(), s, c,
            weight=None if cost_mode == "hops" else "weight",
        ))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def _solve_one_failure(
    *,
    G,
    switches,
    controllers,
    loads,
    capacities,
    final_assign,
    failed_controller,
    dij=None,
    usable_threshold=0.8,
    population_size=50,
    generations=150,
    mutation_rate=0.4,
    tournament_size=3,
    seed=42,
    cost_mode="weight",
):
    """FLCF recovery GA for one failed controller.

    A chromosome maps only the orphan switches of the failed controller to
    surviving controllers. Surviving switches remain fixed. Fitness follows
    the paper's Eq. (1): normalized average switch-controller delay plus
    normalized post-recovery controller-load standard deviation.
    """
    failed_controller = int(failed_controller)
    survivors = [int(c) for c in controllers if int(c) != failed_controller]
    orphan_switches = [
        int(s) for s in switches
        if int(final_assign.get(int(s))) == failed_controller
    ]

    if not survivors:
        return {
            "status": "NO_SURVIVING_CONTROLLERS",
            "recovery_assign": dict(final_assign),
            "orphan_switches": orphan_switches,
            "objective": float("inf"),
            "average_delay": float("inf"),
            "load_std": float("inf"),
            "capacity_violation": float("inf"),
            "generations": 0,
        }

    if not orphan_switches:
        recovered = dict(final_assign)
        survivor_loads = _controller_loads(recovered, survivors, loads)
        return {
            "status": "NO_AFFECTED_SWITCHES",
            "recovery_assign": recovered,
            "orphan_switches": [],
            "objective": 0.0,
            "average_delay": 0.0,
            "load_std": _load_std(survivor_loads),
            "capacity_violation": 0.0,
            "generations": 0,
        }

    rng = random.Random(int(seed) + failed_controller * 100003)
    fixed = {
        int(s): int(c) for s, c in final_assign.items()
        if int(c) != failed_controller
    }

    feasible_targets = {}
    for s in orphan_switches:
        targets = [c for c in survivors if math.isfinite(_distance_lookup(G, s, c, dij, cost_mode))]
        feasible_targets[s] = targets or list(survivors)

    def random_chromosome():
        return {s: rng.choice(feasible_targets[s]) for s in orphan_switches}

    def decode(chromosome):
        result = dict(fixed)
        result.update({int(s): int(c) for s, c in chromosome.items()})
        return result

    def raw_metrics(chromosome):
        recovered = decode(chromosome)
        distances = [_distance_lookup(G, s, recovered[s], dij, cost_mode) for s in orphan_switches]
        avg_delay = sum(distances) / len(distances) if distances else 0.0
        post_loads = _controller_loads(recovered, survivors, loads)
        sigma = _load_std(post_loads)
        violation = sum(
            max(0.0, post_loads[c] - float(usable_threshold) * float(capacities[c]))
            for c in survivors
        )
        return avg_delay, sigma, violation, post_loads

    population_size = max(2, int(population_size))
    population = [random_chromosome() for _ in range(population_size)]

    def evaluated_population(pop):
        raw = [(chrom, *raw_metrics(chrom)) for chrom in pop]
        delays = [row[1] for row in raw]
        sigmas = [row[2] for row in raw]
        dmin, dmax = min(delays), max(delays)
        smin, smax = min(sigmas), max(sigmas)
        evaluated = []
        for chrom, delay, sigma, violation, post_loads in raw:
            # Paper Eq. (1), with a hard capacity penalty so infeasible plans
            # cannot win merely due to good delay/load balance.
            objective = (
                _safe_normalized(delay, dmin, dmax)
                + _safe_normalized(sigma, smin, smax)
                + 1e6 * violation
            )
            evaluated.append((objective, chrom, delay, sigma, violation, post_loads))
        evaluated.sort(key=lambda row: row[0])
        return evaluated

    best_row = None
    for _generation in range(max(1, int(generations))):
        evaluated = evaluated_population(population)
        if best_row is None or evaluated[0][0] < best_row[0]:
            best_row = evaluated[0]

        # Paper-inspired subset: tournament selection, crossover, mutation,
        # and elitist refill from the previous population.
        subset_size = max(2, int(math.sqrt(population_size)))
        subset = []
        while len(subset) < subset_size:
            contestants = rng.sample(evaluated, k=min(tournament_size, len(evaluated)))
            subset.append(min(contestants, key=lambda row: row[0])[1])

        next_population = [dict(evaluated[i][1]) for i in range(min(subset_size, len(evaluated)))]
        while len(next_population) < population_size:
            parent1 = rng.choice(subset)
            parent2 = rng.choice(subset)
            child = {}
            for s in orphan_switches:
                c1, c2 = parent1[s], parent2[s]
                if c1 == c2:
                    child[s] = c1
                    continue

                # Paper Eq. (2): choose the better gene using normalized
                # switch-controller delay and current destination load.
                trial_loads = _controller_loads(decode(child), survivors, loads)
                delays = [_distance_lookup(G, s, c, dij, cost_mode) for c in (c1, c2)]
                loads_now = [trial_loads.get(c, 0.0) for c in (c1, c2)]
                score1 = _safe_normalized(delays[0], min(delays), max(delays)) + _safe_normalized(loads_now[0], min(loads_now), max(loads_now))
                score2 = _safe_normalized(delays[1], min(delays), max(delays)) + _safe_normalized(loads_now[1], min(loads_now), max(loads_now))
                child[s] = c1 if score1 <= score2 else c2

            if rng.random() < float(mutation_rate):
                mutated_switch = rng.choice(orphan_switches)
                child[mutated_switch] = rng.choice(feasible_targets[mutated_switch])
            next_population.append(child)

        population = next_population[:population_size]

    final_eval = evaluated_population(population)
    if best_row is None or final_eval[0][0] < best_row[0]:
        best_row = final_eval[0]

    objective, best_chromosome, avg_delay, sigma, violation, post_loads = best_row
    recovered = decode(best_chromosome)
    status = "SUCCESS" if violation <= 1e-8 else "CAPACITY_VIOLATED"

    return {
        "status": status,
        "recovery_assign": recovered,
        "recovery_plan": {int(s): int(best_chromosome[s]) for s in orphan_switches},
        "orphan_switches": orphan_switches,
        "objective": float(objective),
        "average_delay": float(avg_delay),
        "load_std": float(sigma),
        "capacity_violation": float(violation),
        "post_failure_loads": post_loads,
        "generations": int(generations),
    }


def run_baseline3_flcf_exact(
    G,
    switches: List[int],
    controllers: List[int],
    loads: Dict[int, float],
    capacities: Dict[int, float],
    init_assign: Dict[int, int],
    dij=None,
    paths_sc=None,
    msg_bits: int = 128,
    usable_threshold: float = 0.8,
    overload_threshold: float = 0.8,
    population_size: int = 50,
    generations: int = 150,
    mutation_rate: float = 0.4,
    tournament_size: int = 3,
    output_dir: Optional[str] = None,
    topology_name: Optional[str] = None,
    run_index: int = 0,
    plot_recovery: bool = True,
    plot_pos: Optional[dict] = None,
    cost_mode: str = "weight",
    seed: int = 42,
    verbose: bool = False,
):
    """Drop-in FLCF baseline with the same 10-value return convention.

    IMPORTANT: ``init_assign`` is the current balanced network snapshot. In your
    pipeline pass ``fa_mcf_arc`` here, not ``init_assign_cs``.
    """
    del paths_sc, msg_bits, overload_threshold, verbose

    solve_start = time.perf_counter()
    final_assign = {int(s): int(c) for s, c in init_assign.items()}
    final_loads = _controller_loads(final_assign, controllers, loads)
    failure_records = {}
    scenario_results = {}

    for failed_c in controllers:
        scenario_start = time.perf_counter()
        result = _solve_one_failure(
            G=G,
            switches=switches,
            controllers=controllers,
            loads=loads,
            capacities=capacities,
            final_assign=final_assign,
            failed_controller=failed_c,
            dij=dij,
            usable_threshold=usable_threshold,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            seed=seed,
            cost_mode=cost_mode,
        )
        scenario_time = time.perf_counter() - scenario_start
        scenario_results[int(failed_c)] = result

        if output_dir is not None:
            failure_records[int(failed_c)] = process_failure_scenario(
                algorithm="FLCF",
                topology_name=topology_name or "topology",
                run_index=run_index,
                failed_controller=int(failed_c),
                G=G,
                pos=plot_pos,
                switches=switches,
                controllers=controllers,
                loads=loads,
                capacities=capacities,
                usable_threshold=usable_threshold,
                initial_assignment=final_assign,
                recovery_assignment=result["recovery_assign"],
                fractional_assignment={},
                residual_by_switch={},
                status=result["status"],
                solve_time_sec=scenario_time,
                objective_value=result["objective"],
                mip_gap=None,
                backup_controller=None,
                backup_capacity=None,
                output_root=output_dir,
                make_plot=bool(plot_recovery and plot_pos is not None),
                file_tag=f"FLCF_run{run_index:03d}_failC{int(failed_c)}",
            )

    solve_time = time.perf_counter() - solve_start
    objectives = [float(r["objective"]) for r in scenario_results.values() if math.isfinite(float(r["objective"]))]
    migration_counts = [len(r.get("recovery_plan", {})) for r in scenario_results.values()]
    statuses = [r["status"] for r in scenario_results.values()]

    status = "SUCCESS" if all(s in ("SUCCESS", "NO_AFFECTED_SWITCHES") for s in statuses) else "PARTIAL"
    obj_val = sum(objectives) / len(objectives) if objectives else float("inf")
    migration_count = sum(migration_counts) / len(migration_counts) if migration_counts else 0.0
    usage = {
        int(c): final_loads[int(c)] / float(capacities[int(c)])
        if float(capacities.get(int(c), 0.0)) > 0 else float("inf")
        for c in controllers
    }
    paths_pair, paths_switch = _build_shortest_paths(G, final_assign, cost_mode)

    meta = {
        "algorithm": "FLCF",
        "paper": "A Fast and Load-aware Controller Failover Mechanism for Software-Defined Networks (2016)",
        "normal_state_assignment": final_assign,
        "all_failure_results": scenario_results,
        "failure_scenarios": failure_records,
        "solve_time_sec": solve_time,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "usable_threshold": usable_threshold,
        "status_per_failure": statuses,
    }

    return (
        final_assign,
        paths_pair,
        paths_switch,
        final_loads,
        obj_val,
        migration_count,
        usage,
        None,
        status,
        meta,
    )