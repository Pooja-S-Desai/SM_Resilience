# baseline_pref_cp_ga.py

import os
import json
import math
import time
import random
from typing import Dict, List

import networkx as nx
from failure_scenario_handler import process_failure_scenario


# ============================================================
# BASIC METRICS
# ============================================================

def compute_controller_loads(assign, controllers, loads):
    ctrl_loads = {c: 0.0 for c in controllers}

    for s, c in assign.items():
        if c in ctrl_loads:
            ctrl_loads[c] += float(loads[s])

    return ctrl_loads


def load_std(ctrl_loads):
    values = list(ctrl_loads.values())

    if not values:
        return float("inf")

    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)

    return math.sqrt(variance)


def load_deviation(ctrl_loads):
    values = list(ctrl_loads.values())

    if not values:
        return 0.0

    return max(values) - min(values)


# ============================================================
# PAPER EQ. (1): CONNECTIVITY FAILURE PROBABILITY
# ============================================================

def path_failure_probability(
    G,
    controller,
    switch,
    link_failure_prob=None,
    node_failure_prob=None,
    default_link_failure_prob=0.01,
    default_node_failure_prob=0.01,
):
    link_failure_prob = link_failure_prob or {}
    node_failure_prob = node_failure_prob or {}

    if controller == switch:
        return 0.0

    try:
        path = nx.shortest_path(G, source=controller, target=switch)
    except nx.NetworkXNoPath:
        return 1.0

    survival_prob = 1.0

    for u, v in zip(path[:-1], path[1:]):
        pe = link_failure_prob.get(
            (u, v),
            link_failure_prob.get((v, u), default_link_failure_prob)
        )
        survival_prob *= (1.0 - float(pe))

    # Exclude controller, include destination switch.
    for node in path[1:]:
        pv = node_failure_prob.get(node, default_node_failure_prob)
        survival_prob *= (1.0 - float(pv))

    return 1.0 - survival_prob


def assignment_probability_failure(
    G,
    assign,
    switches,
    link_failure_prob=None,
    node_failure_prob=None,
    default_link_failure_prob=0.01,
    default_node_failure_prob=0.01,
    aggregate="max",
):
    probs = []

    for s in switches:
        c = assign.get(s)

        if c is None:
            probs.append(1.0)
            continue

        probs.append(
            path_failure_probability(
                G=G,
                controller=c,
                switch=s,
                link_failure_prob=link_failure_prob,
                node_failure_prob=node_failure_prob,
                default_link_failure_prob=default_link_failure_prob,
                default_node_failure_prob=default_node_failure_prob,
            )
        )

    if not probs:
        return 0.0

    if aggregate == "max":
        return max(probs)

    if aggregate == "mean":
        return sum(probs) / len(probs)

    if aggregate == "sum":
        return sum(probs)

    raise ValueError("aggregate must be one of: max, mean, sum")


# ============================================================
# PREF-CP GA FOR ONE FAILED CONTROLLER
# ============================================================

def run_pref_cp_ga_single_failure(
    G,
    switches,
    controllers,
    loads,
    capacities,
    final_assign,
    failed_controller,
    alpha=0.5,
    gamma=0.8,
    usable_threshold=0.8,
    population_size=50,
    generations=150,
    mutation_rate=0.10,
    tournament_size=3,
    elite_fraction=0.20,
    link_failure_prob=None,
    node_failure_prob=None,
    default_link_failure_prob=0.01,
    default_node_failure_prob=0.01,
    probability_aggregate="max",
    enforce_capacity=False,
    penalty_weight=1e9,
    seed=42,
    verbose=False,
):
    """
    Paper-faithful PREF-CP recovery GA for one failed controller.

    Important modelling decisions:
    1. A chromosome contains the post-failure controller assignment of EVERY
       switch, not only switches orphaned by the failed controller.
    2. The paper's reassignment constraint is enforced as a hard constraint:

           R_tilde <= gamma * |S|

       Every generated, crossed, and mutated chromosome is repaired to satisfy
       this limit. Reassignment violations are therefore never accepted into
       the population.
    3. A switch previously assigned to the failed controller must move to a
       surviving controller. These are mandatory migrations.
    """
    rng = random.Random(seed)

    switches = [int(s) for s in switches]
    controllers = [int(c) for c in controllers]
    final_assign = {int(s): int(c) for s, c in final_assign.items()}

    link_failure_prob = link_failure_prob or {}
    node_failure_prob = node_failure_prob or {}

    survivors = [c for c in controllers if c != failed_controller]
    if not survivors:
        raise ValueError("No surviving controllers available.")

    failed_switches = [
        s for s in switches
        if final_assign.get(s) == failed_controller
    ]
    failed_switch_set = set(failed_switches)

    # Equation (7): R_tilde <= gamma * |S|.
    # Since R_tilde is integer, the largest permitted migration count is floor(...).
    max_reassignments = int(math.floor(float(gamma) * len(switches) + 1e-12))
    mandatory_reassignments = len(failed_switches)

    # Every orphan switch must move. If even these mandatory migrations exceed
    # the paper's migration budget, this failure scenario is infeasible.
    if mandatory_reassignments > max_reassignments:
        return {
            "failed_controller": failed_controller,
            "status": "INFEASIBLE_REASSIGNMENT_LIMIT",
            "failed_switches": failed_switches,
            "surviving_controllers": survivors,
            "recovery_plan": {},
            "recovery_assign": {},
            "recovery_loads": {},
            "sigma": None,
            "load_deviation": None,
            "p_tilde": None,
            "objective": None,
            "fitness_with_penalty": None,
            "reassignment_cost": mandatory_reassignments,
            "max_reassignments": max_reassignments,
            "mandatory_reassignments": mandatory_reassignments,
            "capacity_violation": None,
            "reassignment_violation": (
                mandatory_reassignments - max_reassignments
            ),
        }

    optional_switches = [
        s for s in switches
        if s not in failed_switch_set
    ]
    optional_budget = max_reassignments - mandatory_reassignments

    def choose_different_controller(current_controller):
        """Choose a surviving controller different from the current one."""
        choices = [c for c in survivors if c != current_controller]
        if not choices:
            return current_controller
        return rng.choice(choices)

    def reassignment_cost(chromosome):
        """Paper reassignment cost: number of old/new assignment differences."""
        return sum(
            1
            for s in switches
            if final_assign.get(s) != chromosome.get(s)
        )

    def is_reassignment_feasible(chromosome):
        """Exact Equation (7) feasibility test."""
        return reassignment_cost(chromosome) <= max_reassignments

    def repair_chromosome(chromosome):
        """
        Enforce the failed-controller domain and the exact reassignment limit.

        Mandatory migrations are never reverted. If crossover or mutation moves
        too many healthy switches, excess optional migrations are reverted to
        their original controllers until Equation (7) is satisfied.
        """
        repaired = dict(chromosome)

        # Ensure every switch has one surviving controller.
        for s in switches:
            old_c = final_assign[s]

            if s not in repaired:
                repaired[s] = (
                    rng.choice(survivors)
                    if old_c == failed_controller
                    else old_c
                )

            if repaired[s] == failed_controller or repaired[s] not in survivors:
                repaired[s] = rng.choice(survivors)

        # Mandatory orphan migrations cannot be reverted.
        for s in failed_switches:
            if repaired[s] == failed_controller:
                repaired[s] = rng.choice(survivors)

        moved_optional = [
            s for s in optional_switches
            if repaired[s] != final_assign[s]
        ]

        excess = max(
            0,
            mandatory_reassignments + len(moved_optional) - max_reassignments,
        )

        if excess > 0:
            rng.shuffle(moved_optional)
            for s in moved_optional[:excess]:
                repaired[s] = final_assign[s]

        assert is_reassignment_feasible(repaired)
        assert all(repaired[s] in survivors for s in switches)
        return repaired

    def make_chromosome():
        """
        Build a complete post-failure assignment for all switches.

        - Orphan switches are mandatorily reassigned.
        - A random number of healthy switches may additionally migrate, but the
          total never exceeds the paper's gamma|S| limit.
        """
        chromosome = {}

        for s in switches:
            old_c = final_assign[s]
            chromosome[s] = (
                rng.choice(survivors)
                if old_c == failed_controller
                else old_c
            )

        if optional_budget > 0 and optional_switches:
            extra_count = rng.randint(
                0,
                min(optional_budget, len(optional_switches)),
            )
            for s in rng.sample(optional_switches, extra_count):
                chromosome[s] = choose_different_controller(final_assign[s])

        return repair_chromosome(chromosome)

    def decode(chromosome):
        """The chromosome itself is the complete post-failure assignment."""
        return dict(chromosome)

    def reassignment_violation(chromosome):
        """
        Kept only for reporting. It is always zero for population members because
        Equation (7) is enforced exactly by construction and repair.
        """
        return max(
            0,
            reassignment_cost(chromosome) - max_reassignments,
        )

    def capacity_violation(assign):
        ctrl_loads = compute_controller_loads(assign, survivors, loads)
        violation = 0.0

        for c in survivors:
            # Common usable controller-capacity interpretation used throughout
            # the comparison pipeline.
            allowed = float(usable_threshold) * float(capacities[c])
            violation += max(0.0, ctrl_loads[c] - allowed)

        return violation

    def compute_objective(chromosome):
        assign = decode(chromosome)

        ctrl_loads = compute_controller_loads(assign, survivors, loads)
        sigma = load_std(ctrl_loads)

        p_tilde = assignment_probability_failure(
            G=G,
            assign=assign,
            switches=switches,
            link_failure_prob=link_failure_prob,
            node_failure_prob=node_failure_prob,
            default_link_failure_prob=default_link_failure_prob,
            default_node_failure_prob=default_node_failure_prob,
            aggregate=probability_aggregate,
        )

        return alpha * sigma + (1.0 - alpha) * p_tilde

    def compute_fitness(chromosome):
        # Equation (7) is a hard constraint, not a penalty.
        if not is_reassignment_feasible(chromosome):
            return float("inf")

        assign = decode(chromosome)
        penalty = 0.0

        # Capacity handling is left as the existing optional penalty mechanism.
        if enforce_capacity:
            penalty += penalty_weight * capacity_violation(assign)

        return compute_objective(chromosome) + penalty

    def select_parent(population):
        candidates = rng.sample(
            population,
            k=min(tournament_size, len(population)),
        )
        return min(candidates, key=compute_fitness)

    def crossover(parent1, parent2):
        """Uniform crossover over all switch genes, followed by exact repair."""
        child = {
            s: (parent1[s] if rng.random() < 0.5 else parent2[s])
            for s in switches
        }
        return repair_chromosome(child)

    def mutate(chromosome):
        """Mutate any switch gene, then restore exact Equation (7) feasibility."""
        mutated = dict(chromosome)

        for s in switches:
            if rng.random() >= mutation_rate:
                continue

            old_c = final_assign[s]

            if s in failed_switch_set:
                # Orphan switch must remain assigned to a survivor.
                mutated[s] = choose_different_controller(mutated[s])
            else:
                # A healthy switch may either move to another survivor or return
                # to its original controller. This lets the GA explore different
                # migration sets while respecting the hard budget after repair.
                candidate_values = list(survivors)
                if old_c in survivors and old_c not in candidate_values:
                    candidate_values.append(old_c)
                mutated[s] = rng.choice(candidate_values)

        return repair_chromosome(mutated)

    # Every chromosome is feasible with respect to Equation (7).
    population = [make_chromosome() for _ in range(population_size)]

    best = min(population, key=compute_fitness)
    best_fit = compute_fitness(best)

    elite_count = max(1, int(elite_fraction * population_size))

    for gen in range(generations):
        population = sorted(population, key=compute_fitness)
        new_population = population[:elite_count]

        while len(new_population) < population_size:
            p1 = select_parent(population)
            p2 = select_parent(population)

            child = crossover(p1, p2)
            child = mutate(child)

            # Defensive check: violating children never enter the population.
            if is_reassignment_feasible(child):
                new_population.append(child)

        population = new_population

        current_best = min(population, key=compute_fitness)
        current_fit = compute_fitness(current_best)

        if current_fit < best_fit:
            best = dict(current_best)
            best_fit = current_fit

        if verbose and gen % 20 == 0:
            print(
                f"[PREF-CP-GA] failed={failed_controller}, "
                f"generation={gen}, fitness={best_fit:.6f}, "
                f"migrations={reassignment_cost(best)}/{max_reassignments}"
            )

    recovery_assign = decode(best)
    recovery_loads = compute_controller_loads(
        recovery_assign,
        survivors,
        loads,
    )

    sigma = load_std(recovery_loads)
    p_tilde = assignment_probability_failure(
        G=G,
        assign=recovery_assign,
        switches=switches,
        link_failure_prob=link_failure_prob,
        node_failure_prob=node_failure_prob,
        default_link_failure_prob=default_link_failure_prob,
        default_node_failure_prob=default_node_failure_prob,
        aggregate=probability_aggregate,
    )
    objective = alpha * sigma + (1.0 - alpha) * p_tilde

    cap_v = capacity_violation(recovery_assign)
    reassignment_count = reassignment_cost(best)
    reass_v = reassignment_violation(best)

    status = "SUCCESS"
    if enforce_capacity and cap_v > 1e-9:
        status = "CAPACITY_VIOLATED"

    migrated_switches = {
        s: {
            "from": final_assign[s],
            "to": recovery_assign[s],
            "is_orphan": s in failed_switch_set,
        }
        for s in switches
        if final_assign[s] != recovery_assign[s]
    }

    return {
        "failed_controller": failed_controller,
        "status": status,
        "failed_switches": failed_switches,
        "surviving_controllers": survivors,
        "recovery_plan": dict(best),
        "recovery_assign": recovery_assign,
        "recovery_loads": recovery_loads,
        "migrated_switches": migrated_switches,
        "sigma": sigma,
        "load_deviation": load_deviation(recovery_loads),
        "p_tilde": p_tilde,
        "objective": objective,
        "fitness_with_penalty": best_fit,
        "reassignment_cost": reassignment_count,
        "max_reassignments": max_reassignments,
        "mandatory_reassignments": mandatory_reassignments,
        "optional_reassignments": (
            reassignment_count - mandatory_reassignments
        ),
        "capacity_violation": cap_v,
        "reassignment_violation": reass_v,
    }


# ============================================================
# ALL CONTROLLER FAILURES IN ONE PREF-CP RUN
# ============================================================

def run_pref_cp_ga_all_failures(
    G,
    switches,
    controllers,
    loads,
    capacities,
    final_assign,
    alpha=0.5,
    gamma=0.8,
    usable_threshold=0.8,
    population_size=50,
    generations=150,
    mutation_rate=0.10,
    link_failure_prob=None,
    node_failure_prob=None,
    default_link_failure_prob=0.01,
    default_node_failure_prob=0.01,
    probability_aggregate="max",
    enforce_capacity=False,
    seed=42,
    verbose=False,
):
    results = {}

    for idx, failed_c in enumerate(controllers):
        results[failed_c] = run_pref_cp_ga_single_failure(
            G=G,
            switches=switches,
            controllers=controllers,
            loads=loads,
            capacities=capacities,
            final_assign=final_assign,
            failed_controller=failed_c,
            alpha=alpha,
            gamma=gamma,
            usable_threshold=usable_threshold,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            link_failure_prob=link_failure_prob,
            node_failure_prob=node_failure_prob,
            default_link_failure_prob=default_link_failure_prob,
            default_node_failure_prob=default_node_failure_prob,
            probability_aggregate=probability_aggregate,
            enforce_capacity=enforce_capacity,
            seed=seed + idx,
            verbose=verbose,
        )

    return results


# ============================================================
# LOGGING
# ============================================================

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    return obj


def save_pref_cp_logs(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "pref_cp_ga_full_log.json")
    csv_path = os.path.join(output_dir, "pref_cp_ga_summary.csv")

    with open(json_path, "w") as f:
        json.dump(json_safe(results), f, indent=2)

    with open(csv_path, "w") as f:
        f.write(
            "failed_controller,status,num_failed_switches,"
            "reassignment_cost,sigma,load_deviation,p_tilde,"
            "objective,fitness_with_penalty,capacity_violation,"
            "reassignment_violation\n"
        )

        for failed_c, r in results.items():
            f.write(
                f"{failed_c},{r['status']},{len(r['failed_switches'])},"
                f"{r['reassignment_cost']},{r['sigma']},"
                f"{r['load_deviation']},{r['p_tilde']},"
                f"{r['objective']},{r['fitness_with_penalty']},"
                f"{r['capacity_violation']},{r['reassignment_violation']}\n"
            )

    return json_path, csv_path


# ============================================================
# PLOTS
# ============================================================


def run_baseline_pref_cp_ga_exact(
    G,
    switches,
    controllers,
    loads,
    capacities,
    init_assign,
    dij=None,
    paths_sc=None,
    msg_bits=None,
    usable_threshold=0.8,
    overload_threshold=0.8,
    alpha=0.5,
    gamma=0.8,
    population_size=50,
    generations=150,
    mutation_rate=0.10,
    default_link_failure_prob=0.01,
    default_node_failure_prob=0.01,
    probability_aggregate="max",
    enforce_capacity=True,
    output_dir=None,
    seed=42,
    verbose=False,
    *,
    topology_name=None,
    run_index=0,
    plot_recovery=False,
    plot_pos=None,
    plot_save_dir=None,
    plot_file_tag=None,
):
    """Run PREF-CP-GA from the supplied current network assignment.

    In the comparison pipeline, pass the MCF-ARC balanced assignment as
    ``init_assign``. For every failed-controller scenario, the result is sent
    through the same ``process_failure_scenario`` function used by MCF-ARC.
    """
    solve_start = time.perf_counter()

    # This is the common pre-failure snapshot. In main, pass fa_mcf_arc here.
    final_assign = {int(s): int(c) for s, c in dict(init_assign).items()}
    final_loads = compute_controller_loads(final_assign, controllers, loads)

    results = run_pref_cp_ga_all_failures(
        G=G,
        switches=switches,
        controllers=controllers,
        loads=loads,
        capacities=capacities,
        final_assign=final_assign,
        alpha=alpha,
        gamma=gamma,
        usable_threshold=usable_threshold,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        default_link_failure_prob=default_link_failure_prob,
        default_node_failure_prob=default_node_failure_prob,
        probability_aggregate=probability_aggregate,
        enforce_capacity=enforce_capacity,
        seed=seed,
        verbose=verbose,
    )

    output_root = plot_save_dir or output_dir
    if output_root is not None:
        os.makedirs(output_root, exist_ok=True)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_log, csv_log = save_pref_cp_logs(results, output_dir)
    else:
        json_log = None
        csv_log = None

    # ------------------------------------------------------------------
    # SAME COMMON POST-FAILURE HANDLER AS MCF-ARC
    # One record and one final-vs-recovery plot per failed controller.
    # ------------------------------------------------------------------
    failure_records = {}
    if output_root is not None:
        for failed_c, result in results.items():
            failed_c = int(failed_c)
            recovery_assign = {
                int(s): int(c)
                for s, c in (result.get("recovery_assign") or {}).items()
            }

            # A reassignment-limit infeasibility may intentionally have no
            # valid recovery assignment. The handler still writes the scenario
            # JSON; plotting is enabled only when a complete assignment exists.
            complete_recovery = (
                len(recovery_assign) == len(switches)
                and all(s in recovery_assign for s in switches)
            )

            tag_prefix = plot_file_tag or "PREF_CP_GA"
            record = process_failure_scenario(
                algorithm="PREF_CP_GA",
                topology_name=topology_name or "topology",
                run_index=int(run_index),
                failed_controller=failed_c,
                G=G,
                pos=plot_pos,
                switches=switches,
                controllers=controllers,
                loads=loads,
                capacities=capacities,
                usable_threshold=float(usable_threshold),
                initial_assignment=final_assign,
                recovery_assignment=recovery_assign,
                fractional_assignment={},
                residual_by_switch={},
                status=str(result.get("status", "UNKNOWN")),
                solve_time_sec=float(result.get("solve_time_sec", 0.0)),
                objective_value=result.get("objective"),
                mip_gap=None,
                backup_controller=None,
                backup_capacity=None,
                output_root=output_root,
                make_plot=bool(
                    plot_recovery
                    and plot_pos is not None
                    and complete_recovery
                ),
                file_tag=(
                    f"{tag_prefix}_run{int(run_index):03d}_failC{failed_c}"
                ),
            )
            failure_records[failed_c] = record

    solve_time = time.perf_counter() - solve_start

    valid_results = [
        r for r in results.values()
        if r.get("objective") is not None
    ]
    obj_values = [float(r["objective"]) for r in valid_results]
    obj_val = sum(obj_values) / len(obj_values) if obj_values else 0.0

    mig_values = [
        int(r.get("reassignment_cost", 0))
        for r in valid_results
    ]
    migration_count = (
        sum(mig_values) / len(mig_values)
        if mig_values else 0.0
    )

    # Normal-state controller utilization; failure-specific utilization is in
    # failure_records and all_failure_results.
    usage = {
        c: (
            float(final_loads.get(c, 0.0)) / float(capacities[c])
            if float(capacities.get(c, 0.0)) > 0.0
            else 0.0
        )
        for c in controllers
    }

    statuses = [str(r.get("status", "UNKNOWN")) for r in results.values()]
    if all(s in ("SUCCESS", "NO_AFFECTED_SWITCHES") for s in statuses):
        status = "SUCCESS"
    elif any(s == "INFEASIBLE_REASSIGNMENT_LIMIT" for s in statuses):
        status = "INFEASIBLE_REASSIGNMENT_LIMIT"
    elif any("CAPACITY" in s for s in statuses):
        status = "CAPACITY_VIOLATED"
    else:
        status = "PARTIAL"

    paths_pair = {}
    paths_switch = {}
    mip = None

    meta = {
        "algorithm": "PREF-CP-GA",
        "paper": "Proactive Controller Assignment Schemes in SDN For Fast Recovery",
        "alpha": alpha,
        "gamma": gamma,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "probability_aggregate": probability_aggregate,
        "default_link_failure_prob": default_link_failure_prob,
        "default_node_failure_prob": default_node_failure_prob,
        "enforce_capacity": enforce_capacity,
        "usable_threshold": usable_threshold,
        "solve_time_sec": solve_time,
        "all_failure_results": results,
        "failure_scenarios": failure_records,
        "json_log": json_log,
        "csv_log": csv_log,
        "status_per_failure": statuses,
        "normal_assignment_source": "caller supplied current assignment",
    }

    return (
        final_assign,
        paths_pair,
        paths_switch,
        final_loads,
        obj_val,
        migration_count,
        usage,
        mip,
        status,
        meta,
    )

