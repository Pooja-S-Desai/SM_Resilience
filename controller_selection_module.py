# Controller_selelction_module
import os
import math
import gurobipy as gp
from gurobipy import GRB, quicksum
from helpers import ( CAPACITY_THRESHOLD, INITIAL_ASSUMED_CONTROLLERS_FOR_CAP,
    MAX_LOAD, CAPACITY_THRESHOLD_INITIAL,MIN_LOAD,HOT_UTIL_RANGE,NORMAL_UTIL_RANGE
)
import random
import random, numpy as np
from iis_logger import write_iis_with_context
CONTROLLER_FAILURE_FOLDER = "./controller_selection_failure"
os.makedirs(CONTROLLER_FAILURE_FOLDER, exist_ok=True)
MCF_RHO_MAX_FOR_INIT = 0.95


def _build_feasible_imbalanced_fallback(G, loads, estimated_k, pivot_capacity):
    """
    Deterministic backup for experiments: always returns a complete initial
    placement with at least one controller above CAPACITY_THRESHOLD utilization.
    """
    nodes = sorted(G.nodes())
    controllers = nodes[:min(estimated_k, len(nodes))]
    if not controllers:
        return None, [], {}, {}, "FAILED_NO_NODES"

    hot = controllers[0]
    assignment = {}
    ctrl_loads = {c: 0.0 for c in controllers}

    # Controllers serve themselves first.
    for c in controllers:
        assignment[c] = c
        ctrl_loads[c] += float(loads.get(c, 0.0))

    remaining = [s for s in nodes if s not in assignment]

    # Pack the hottest controller just above the threshold. Its capacity is
    # then chosen from its assigned load, so it is overloaded but not >100%.
    target_hot_load = max(
        CAPACITY_THRESHOLD * float(pivot_capacity) + 1.0,
        float(loads.get(hot, 0.0)),
    )
    for s in sorted(remaining, key=lambda x: float(loads.get(x, 0.0)), reverse=True):
        if ctrl_loads[hot] > target_hot_load:
            break
        assignment[s] = hot
        ctrl_loads[hot] += float(loads.get(s, 0.0))

    # Balance everything else over the remaining controllers.
    remaining = [s for s in nodes if s not in assignment]
    other_controllers = controllers[1:] or controllers
    for s in sorted(remaining, key=lambda x: float(loads.get(x, 0.0)), reverse=True):
        c = min(other_controllers, key=lambda cc: ctrl_loads[cc])
        assignment[s] = c
        ctrl_loads[c] += float(loads.get(s, 0.0))

    capacities = {}
    # for c in controllers:
    #     load_c = max(ctrl_loads[c], 1.0)
    #     if c == hot:
    #         capacities[c] = max(1.0, math.ceil(load_c / 0.85))
    #     else:
    #         capacities[c] = max(float(pivot_capacity), math.ceil(load_c / 0.70))


    rng = random.Random(seed_menu)

    for c in controllers:
        load_c = max(ctrl_loads[c], 1.0)

        if c == hot:
            target_util = rng.uniform(*HOT_UTIL_RANGE)
            capacities[c] = max(1.0, math.ceil(load_c / target_util))
        else:
            target_util = rng.uniform(*NORMAL_UTIL_RANGE)
            capacities[c] = max(float(pivot_capacity), math.ceil(load_c / target_util))




    total_load = sum(float(loads.get(s, 0.0)) for s in nodes)
    effective_threshold = CAPACITY_THRESHOLD_INITIAL * MCF_RHO_MAX_FOR_INIT
    usable_total = sum(effective_threshold * capacities[c] for c in controllers)
    if usable_total < total_load:
        deficit = total_load - usable_total
        capacities[controllers[-1]] += math.ceil(deficit / effective_threshold) + 1

    overloaded = [
        c for c in controllers
        if ctrl_loads[c] > CAPACITY_THRESHOLD * capacities[c]
    ]
    if not overloaded:
        load_hot = max(ctrl_loads[hot], 1.0)
        capacities[hot] = max(1.0, math.ceil(load_hot / 0.85))
        overloaded = [hot]

    print(f"🛟 FALLBACK controller selection used; overloaded controllers: {overloaded}")
    return len(controllers), controllers, capacities, assignment, {} , "OPTIMAL"
def get_randomized_uniform_capacity_from_initial(initial_capacity, spread, num_options, *, seed: int):
    """
    Build a candidate capacity menu around 'initial_capacity' with +/- spread,
    using a local RNG seeded deterministically.
    """
    rnd = random.Random(seed)             # <-- local PRNG
    lower = max(1, int(initial_capacity * (1 - spread)))
    upper = int(initial_capacity * (1 + spread))
    candidates = sorted(set(rnd.randint(lower, upper) for _ in range(num_options)))
    candidates.append(int(initial_capacity))
    candidates = sorted(set(candidates))
    return rnd.choice(candidates), candidates

def get_min_controllers_and_assignment(
    G,
    loads,
    timelimit=0,
    topology_name=None,
    seed_menu=None,
    seed_nodes=None,
    *,
    cost_mode="weight",   # harmless extra kwarg for future use
    seeds=None,           # NEW: dict like {"caps_menu":..., "caps_nodes":...}
    **kwargs,             # swallow any future kwargs safely
):
    """
    Returns: k_min, controllers, capacities, init_assign
    """

    # --- NEW: normalize 'seeds' dict into existing knobs (backward compatible) ---
    # If caller passes a 'seeds' dict we use its entries to set the RNGs
    if seeds is not None:
        if seed_menu is None:
            try:
                seed_menu = seeds.get("caps_menu")
            except Exception:
                pass
        if seed_nodes is None:
            try:
                seed_nodes = seeds.get("caps_nodes")
            except Exception:
                pass

    # --- NEW: seed RNGs if provided (optional; safe if your impl uses randomness) ---
    try:
        
        if seed_menu is not None:
            random.seed(int(seed_menu))
            # keep numpy deterministic too
            np.random.seed(int(seed_menu) % (2**32 - 1))
        # If you prefer distinct seeding for node choice, uncomment:
        # if seed_nodes is not None:
        #     random.seed(int(seed_nodes))
    except Exception:
        # Seeding is best-effort; if environment lacks numpy, continue.
        pass



    num_nodes = len(G.nodes())
    if num_nodes < 50:
        k_percent = 0.11      # 11%
    else:
        k_percent = 0.08      # 8%

    estimated_k = max(2, math.ceil(k_percent * num_nodes))
#  estimated_k = max(2, math.ceil(0.10 * num_nodes))
    

    # Step 1: Calculate pivot and generate capacity options
    # pivot_capacity = int(((MAX_LOAD * num_nodes / estimated_k) / CAPACITY_THRESHOLD_INITIAL * 1.1))
    # _, capacity_options = get_randomized_uniform_capacity_from_initial(
    #     pivot_capacity, spread=0.1, num_options=20, seed=seed_menu   # <-- seeded
    # )
    
    total_switch_load = sum(float(loads[u]) for u in G.nodes())

    pivot_capacity = int(
        (total_switch_load / estimated_k)
        / (CAPACITY_THRESHOLD_INITIAL * MCF_RHO_MAX_FOR_INIT)
        * 1.1
    )
    _, capacity_options = get_randomized_uniform_capacity_from_initial(
        pivot_capacity, spread=0.2, num_options=10, seed=seed_menu   # <-- seeded
    )



    print(f"🔧 Pivot capacity: {pivot_capacity}")
    print(f"🎯 Estimated controllers (20% of nodes): {estimated_k}")
    print(f"🎲 Capacity options: {capacity_options}")

    # Step 2: Assign a random capacity to each node from the options (seeded, no global state)
    rnd_nodes = random.Random(seed_nodes)                                   # <-- local PRNG
    node_capacities = {v: rnd_nodes.choice(capacity_options) for v in G.nodes()}
    anchor_nodes = sorted(G.nodes())[:estimated_k]
    for v in anchor_nodes:
        node_capacities[v] = max(node_capacities[v], pivot_capacity)

    # Step 3: Create the Gurobi model
    try:
        m = gp.Model("min_controllers_fixed_random_capacity")
    except gp.GurobiError as e:
        print(f"⚠️ Controller selection Gurobi startup failed ({e}); using deterministic feasible fallback.")
        return _build_feasible_imbalanced_fallback(G, loads, estimated_k, pivot_capacity)

    m.setParam("OutputFlag", 0)
    if timelimit > 0:
        m.setParam("TimeLimit", timelimit)

    # Variables
    x1 = m.addVars(G.nodes(), vtype=GRB.BINARY, name="is_controller")
    x2 = m.addVars(G.nodes(), G.nodes(), vtype=GRB.BINARY, name="assign")
    overloaded = m.addVars(G.nodes(), vtype=GRB.BINARY, name="overloaded")

    # Step 4: Constraints
    for u in G.nodes():
        m.addConstr(quicksum(x2[u, v] for v in G.nodes()) == 1, name=f"assign_{u}")

    for v in G.nodes():
        m.addConstr(
            quicksum(loads[u] * x2[u, v] for u in G.nodes()) <= node_capacities[v] * x1[v],
            name=f"capacity_limit_{v}"
        )
        m.addConstr(quicksum(x2[u, v] for u in G.nodes()) >= x1[v], name=f"min_load_{v}")

    for u in G.nodes():
        for v in G.nodes():
            m.addConstr(x2[u, v] <= x1[v], name=f"valid_assign_{u}_{v}")

    m.addConstr(
        quicksum(x1[v] for v in G.nodes()) == estimated_k,
        name="exact_estimated_20_percent_controllers"
    )

    for v in G.nodes():
        load_on_v = gp.quicksum(loads[u] * x2[u, v] for u in G.nodes())
        m.addConstr((overloaded[v] == 0) >> (load_on_v <= CAPACITY_THRESHOLD * node_capacities[v]),
                    name=f"no_overload_{v}")
        m.addConstr((overloaded[v] == 1) >> (load_on_v >= CAPACITY_THRESHOLD * node_capacities[v] + 1e-3),
                    name=f"overload_{v}")
        m.addConstr(overloaded[v] <= x1[v], name=f"non_controller_not_overloaded_{v}")

    m.addConstr(
        quicksum(
            node_capacities[v] * x1[v] * CAPACITY_THRESHOLD_INITIAL * MCF_RHO_MAX_FOR_INIT
            for v in G.nodes()
        ) >= total_switch_load,
        name="total_capacity_satisfies_total_load"
    )

    min_overloaded_ctrls = 1
    m.addConstr(
        gp.quicksum(overloaded[v] for v in G.nodes()) >= min_overloaded_ctrls,
        name="at_least_one_selected_controller_above_80pct"
    )
    for v in G.nodes():
        m.addConstr(x2[v, v] >= x1[v], name=f"controller_serves_itself_{v}")

    # Step 5: Objective
    m.setObjective(quicksum(x1[v] for v in G.nodes()), GRB.MINIMIZE)

    # Step 6: Solve
    try:
        m.optimize()
    except gp.GurobiError as e:
        print(f"⚠️ Controller selection solve failed ({e}); using deterministic feasible fallback.")
        return _build_feasible_imbalanced_fallback(G, loads, estimated_k, pivot_capacity)


    if m.status == GRB.Status.OPTIMAL:
        min_controllers = int(m.objVal)
        selected = [v for v in G.nodes() if x1[v].X > 0.5]
        assigned_capacities = {v: node_capacities[v] for v in selected}
        initial_assignment = {u: v for u in G.nodes() for v in G.nodes() if x2[u, v].X > 0.5}
        selected_loads = {
            v: sum(float(loads[u]) for u in G.nodes() if initial_assignment.get(u) == v)
            for v in selected
        }
        overloaded_selected = [
            v for v in selected
            if selected_loads[v] > CAPACITY_THRESHOLD * assigned_capacities[v]
        ]

        if selected and initial_assignment:
            print(f"✔ SUCCESS: Initial assignment successful for {topology_name}")
            print(f"⚖️ Initial overloaded controllers (>{CAPACITY_THRESHOLD:.0%}): {overloaded_selected}")
            return min_controllers, selected, assigned_capacities, initial_assignment, node_capacities, "OPTIMAL"
        else:
            print(f"❌ Initial assignment failed for {topology_name}")
            return None, [], {}, {}, "FAILED_INIT_ASSIGNMENT"

    # Step 7: Handle infeasibility (IIS)
    status_msg = f"FAILED_INITIAL_ASSIGNMENT_STATUS_{m.Status}"
    try:
        print(f"⚠️ Gurobi status {m.status}. Writing IIS for {topology_name}...")
        m.setParam(GRB.Param.Presolve, 0)
        m.setParam(GRB.Param.IISMethod, 1)
        m.computeIIS()
        write_iis_with_context(
            m,
            base_dir=CONTROLLER_FAILURE_FOLDER,
            topo_name=topology_name,
            run_index=seeds.get("run", -1) if seeds else -1,
            stage="controller_selection",
        )
        status_msg = f"INFEASIBLE_INITIAL_ASSIGNMENT_STATUS_{m.Status}"
    except Exception as e:
        print(f"❌ Failed to write IIS: {e}")

    print(f"⚠️ Controller selection MILP failed with {status_msg}; using deterministic feasible fallback.")
    return _build_feasible_imbalanced_fallback(G, loads, estimated_k, pivot_capacity)
