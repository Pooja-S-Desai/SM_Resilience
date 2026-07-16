# baseline_pref_cp_ga.py

import os
import json
import math
import time
import random
from typing import Any, Dict, List

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
    rng = random.Random(seed)

    link_failure_prob = link_failure_prob or {}
    node_failure_prob = node_failure_prob or {}

    survivors = [c for c in controllers if c != failed_controller]

    if not survivors:
        raise ValueError("No surviving controllers available.")

    failed_switches = [
        s for s in switches
        if final_assign.get(s) == failed_controller
    ]

    fixed_assign = {
        s: c for s, c in final_assign.items()
        if c != failed_controller
    }

    def make_chromosome():
        return {s: rng.choice(survivors) for s in failed_switches}

    def decode(chromosome):
        recovered = dict(fixed_assign)
        recovered.update(chromosome)
        return recovered

    def reassignment_cost(chromosome):
        recovered = decode(chromosome)
        return sum(
            1 for s in switches
            if final_assign.get(s) != recovered.get(s)
        )

    def reassignment_violation(chromosome):
        return max(0.0, reassignment_cost(chromosome) - gamma * len(switches))

    def capacity_violation(assign):
        ctrl_loads = compute_controller_loads(assign, survivors, loads)
        violation = 0.0

        for c in survivors:
            allowed = (1.0 - alpha) * float(capacities[c])
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
        assign = decode(chromosome)

        penalty = penalty_weight * reassignment_violation(chromosome)

        if enforce_capacity:
            penalty += penalty_weight * capacity_violation(assign)

        return compute_objective(chromosome) + penalty

    def select_parent(population):
        candidates = rng.sample(
            population,
            k=min(tournament_size, len(population))
        )
        return min(candidates, key=compute_fitness)

    def crossover(parent1, parent2):
        child = {}

        for s in failed_switches:
            trial1 = dict(child)
            trial2 = dict(child)

            trial1[s] = parent1[s]
            trial2[s] = parent2[s]

            for rem in failed_switches:
                if rem not in trial1:
                    trial1[rem] = rng.choice(survivors)
                if rem not in trial2:
                    trial2[rem] = rng.choice(survivors)

            child[s] = parent1[s] if compute_fitness(trial1) <= compute_fitness(trial2) else parent2[s]

        return child

    def mutate(chromosome):
        mutated = dict(chromosome)

        for s in failed_switches:
            if rng.random() < mutation_rate:
                mutated[s] = rng.choice(survivors)

        return mutated

    if not failed_switches:
        recovery_assign = dict(final_assign)
        recovery_loads = compute_controller_loads(recovery_assign, survivors, loads)

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

        return {
            "failed_controller": failed_controller,
            "status": "NO_AFFECTED_SWITCHES",
            "failed_switches": [],
            "surviving_controllers": survivors,
            "recovery_plan": {},
            "recovery_assign": recovery_assign,
            "recovery_loads": recovery_loads,
            "sigma": sigma,
            "load_deviation": load_deviation(recovery_loads),
            "p_tilde": p_tilde,
            "objective": objective,
            "fitness_with_penalty": objective,
            "reassignment_cost": 0,
            "capacity_violation": 0.0,
            "reassignment_violation": 0.0,
        }

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

            new_population.append(child)

        population = new_population

        current_best = min(population, key=compute_fitness)
        current_fit = compute_fitness(current_best)

        if current_fit < best_fit:
            best = current_best
            best_fit = current_fit

        if verbose and gen % 20 == 0:
            print(
                f"[PREF-CP-GA] failed={failed_controller}, "
                f"generation={gen}, fitness={best_fit:.6f}"
            )

    recovery_assign = decode(best)
    recovery_loads = compute_controller_loads(recovery_assign, survivors, loads)

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
    reass_v = reassignment_violation(best)

    status = "SUCCESS"

    if enforce_capacity and cap_v > 1e-9:
        status = "CAPACITY_VIOLATED"

    if reass_v > 1e-9:
        status = "REASSIGNMENT_LIMIT_VIOLATED"

    return {
        "failed_controller": failed_controller,
        "status": status,
        "failed_switches": failed_switches,
        "surviving_controllers": survivors,
        "recovery_plan": best,
        "recovery_assign": recovery_assign,
        "recovery_loads": recovery_loads,
        "sigma": sigma,
        "load_deviation": load_deviation(recovery_loads),
        "p_tilde": p_tilde,
        "objective": objective,
        "fitness_with_penalty": best_fit,
        "reassignment_cost": reassignment_cost(best),
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

def draw_assignment_subplot(
    ax,
    G,
    pos,
    switches,
    controllers,
    assign,
    title,
    failed_controller=None,
):
    ax.set_title(title, fontsize=9)

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="lightgray",
        width=0.8,
        alpha=0.7,
    )

    switch_nodes = [s for s in switches if s not in controllers]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=switch_nodes,
        node_size=100,
        node_color="white",
        edgecolors="black",
        linewidths=0.8,
        ax=ax,
    )

    active_controllers = [c for c in controllers if c != failed_controller]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=active_controllers,
        node_size=260,
        node_color="lightgray",
        edgecolors="black",
        linewidths=1.2,
        ax=ax,
    )

    if failed_controller is not None and failed_controller in G.nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[failed_controller],
            node_size=300,
            node_color="white",
            edgecolors="red",
            linewidths=2.0,
            ax=ax,
        )

    for s, c in assign.items():
        if s in pos and c in pos:
            if failed_controller is not None and c == failed_controller:
                continue

            ax.plot(
                [pos[s][0], pos[c][0]],
                [pos[s][1], pos[c][1]],
                linestyle="--",
                linewidth=0.8,
                color="black",
                alpha=0.45,
            )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: str(n) for n in G.nodes()},
        font_size=6,
        ax=ax,
    )

    ax.axis("off")


def plot_pref_cp_failure(
    G,
    switches,
    controllers,
    loads,
    final_assign,
    result,
    output_dir,
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    failed_c = result["failed_controller"]
    pos = nx.spring_layout(G, seed=seed)

    normal_loads = compute_controller_loads(final_assign, controllers, loads)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    draw_assignment_subplot(
        axes[0],
        G,
        pos,
        switches,
        controllers,
        final_assign,
        title=(
            "Normal final assignment\n"
            f"dev={load_deviation(normal_loads):.2f}"
        ),
        failed_controller=None,
    )

    draw_assignment_subplot(
        axes[1],
        G,
        pos,
        switches,
        controllers,
        result["recovery_assign"],
        title=(
            f"PREF-CP-GA recovery: failed {failed_c}\n"
            f"σ={result['sigma']:.2f}, "
            f"P̃={result['p_tilde']:.4f}, "
            f"Obj={result['objective']:.4f}"
        ),
        failed_controller=failed_c,
    )

    plt.tight_layout()

    path = os.path.join(output_dir, f"pref_cp_ga_failure_{failed_c}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_pref_cp_all_failures(
    G,
    switches,
    controllers,
    loads,
    final_assign,
    results,
    output_dir,
    seed=42,
):
    plot_paths = {}

    for failed_c, result in results.items():
        plot_paths[failed_c] = plot_pref_cp_failure(
            G=G,
            switches=switches,
            controllers=controllers,
            loads=loads,
            final_assign=final_assign,
            result=result,
            output_dir=output_dir,
            seed=seed,
        )

    return plot_paths


# ============================================================
# DROP-IN PIPELINE FUNCTION — KEEP THIS CALL CONSISTENT
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
    usable_threshold=0.90,
    overload_threshold=0.90,
    alpha=0.5,
    gamma=0.8,
    population_size=50,
    generations=150,
    mutation_rate=0.10,
    default_link_failure_prob=0.01,
    default_node_failure_prob=0.01,
    probability_aggregate="max",
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
):
    solve_start = time.perf_counter()

    final_assign = dict(init_assign)

    final_loads = compute_controller_loads(
        final_assign,
        controllers,
        loads
    )

    results = run_pref_cp_ga_all_failures(
        G=G,
        switches=switches,
        controllers=controllers,
        loads=loads,
        capacities=capacities,
        final_assign=final_assign,
        alpha=alpha,
        gamma=gamma,
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

    solve_time = time.perf_counter() - solve_start

    scenario_records = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        # The common recovery CSV/JSON below supersedes the old PREF-only
        # summary files, which had a different schema and duplicated records.
        json_log, csv_log = None, None

        # Use the exact same common plotting routine and recovery CSV writer as MCF-ARC.
        # The old PREF-only spring-layout plot is intentionally not called.
        common_root = output_dir
        for failed_c, result in results.items():
            scenario_records[int(failed_c)] = process_failure_scenario(
                algorithm="PREF_CP_GA",
                topology_name=topology_name or "topology",
                run_index=int(run_index),
                failed_controller=int(failed_c),
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
                output_root=common_root,
                comparison_csv_file=comparison_csv_file,
                make_plot=bool(plot_recovery),
                file_tag=f"PREF_run{int(run_index):03d}_failC{int(failed_c)}",
                master_seed=master_seed,
                switch_seed=switch_seed if switch_seed is not None else seed,
                run_number=run_number,
                reassignment_scope="orphan_only",
                pre_failure_response_time_ms=pre_failure_response_time_ms,
                sync_delay_ms=sync_delay_ms,
                write_csv=True,
            )
        plot_paths = {
            int(fc): rec.get("plot_path")
            for fc, rec in scenario_records.items()
            if rec.get("plot_path")
        }
    else:
        json_log = None
        csv_log = None
        plot_paths = {}


    obj_values = [
        float(r.get("objective", 0.0))
        for r in results.values()
    ]

    obj_val = sum(obj_values) / len(obj_values) if obj_values else 0.0

    mig_values = [
        int(r.get("reassignment_cost", 0))
        for r in results.values()
    ]

    migration_count = (
        sum(mig_values) / len(mig_values)
        if mig_values else 0
    )

    usage = {
        c: (
            float(final_loads[c]) / float(capacities[c])
            if float(capacities.get(c, 0.0)) > 0.0
            else 0.0
        )
        for c in controllers
    }

    statuses = [r.get("status", "UNKNOWN") for r in results.values()]

    if all(s in ("SUCCESS", "NO_AFFECTED_SWITCHES") for s in statuses):
        status = "SUCCESS"
    elif any(s == "REASSIGNMENT_LIMIT_VIOLATED" for s in statuses):
        status = "REASSIGNMENT_LIMIT_VIOLATED"
    elif any(s == "CAPACITY_VIOLATED" for s in statuses):
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
        "solve_time_sec": solve_time,
        "all_failure_results": results,
        "failure_scenarios": scenario_records,
        "plot_paths": plot_paths,
        "json_log": json_log,
        "csv_log": csv_log,
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
        mip,
        status,
        meta,
    )
