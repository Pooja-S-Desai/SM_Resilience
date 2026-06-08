# Controller_selelction_module
import os
import gurobipy as gp
from gurobipy import GRB, quicksum
from helpers import ( CAPACITY_THRESHOLD, INITIAL_ASSUMED_CONTROLLERS_FOR_CAP,
    MAX_LOAD, CAPACITY_THRESHOLD_INITIAL,MIN_LOAD
)
import random
import random, numpy as np
from iis_logger import write_iis_with_context
CONTROLLER_FAILURE_FOLDER = "./controller_selection_failure"
os.makedirs(CONTROLLER_FAILURE_FOLDER, exist_ok=True)

def get_randomized_uniform_capacity_from_initial(initial_capacity, spread, num_options, *, seed: int):
    """
    Build a candidate capacity menu around 'initial_capacity' with +/- 'spread',
    using a local RNG seeded deterministically.
    """
    rnd = random.Random(seed)             # <-- local PRNG
    lower = initial_capacity
    upper = int(initial_capacity * (1 + spread))
    candidates = sorted(set(rnd.randint(lower, upper) for _ in range(num_options)))
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
    total_load = num_nodes * MAX_LOAD
    # estimated_k = max(2, int(INITIAL_ASSUMED_CONTROLLERS_FOR_CAP * num_nodes))
    

    if num_nodes > 40:
        estimated_k = 5
    elif num_nodes > 30 :
        estimated_k = 4
    else:
        estimated_k = 3

    # Step 1: Calculate pivot and generate capacity options
    # pivot_capacity = int(((MAX_LOAD * num_nodes / estimated_k) / CAPACITY_THRESHOLD_INITIAL * 1.1))
    # _, capacity_options = get_randomized_uniform_capacity_from_initial(
    #     pivot_capacity, spread=0.1, num_options=20, seed=seed_menu   # <-- seeded
    # )
    
    # total_load = sum(loads.values())
    pivot_capacity = int((total_load / estimated_k) / CAPACITY_THRESHOLD_INITIAL * 1.1)
    _, capacity_options = get_randomized_uniform_capacity_from_initial(
        pivot_capacity, spread=0.1, num_options=20, seed=seed_menu   # <-- seeded
    )



    print(f"🔧 Pivot capacity: {pivot_capacity}")
    print(f"🎲 Capacity options: {capacity_options}")

    # Step 2: Assign a random capacity to each node from the options (seeded, no global state)
    rnd_nodes = random.Random(seed_nodes)                                   # <-- local PRNG
    node_capacities = {v: rnd_nodes.choice(capacity_options) for v in G.nodes()}

    # Step 3: Create the Gurobi model
    m = gp.Model("min_controllers_fixed_random_capacity")
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

    if num_nodes >30:
         m.addConstr(quicksum(x1[v] for v in G.nodes()) >= 6, name="min_5_controllers")
    elif num_nodes > 20 :

        m.addConstr(quicksum(x1[v] for v in G.nodes()) >= 5, name="min_4_controllers")
    else:
        m.addConstr(quicksum(x1[v] for v in G.nodes()) >= 4, name="min_3_controllers")

    for v in G.nodes():
        load_on_v = gp.quicksum(loads[u] * x2[u, v] for u in G.nodes())
        m.addConstr((overloaded[v] == 0) >> (load_on_v <= CAPACITY_THRESHOLD * node_capacities[v] - 1e-3),
                    name=f"no_overload_{v}")
        m.addConstr((overloaded[v] == 1) >> (load_on_v >= CAPACITY_THRESHOLD * node_capacities[v]),
                    name=f"overload_{v}")
        m.addConstr(overloaded[v] <= x1[v], name=f"non_controller_not_overloaded_{v}")

    total_switch_load = sum(loads.values())
    m.addConstr(
        quicksum(node_capacities[v] * x1[v] * CAPACITY_THRESHOLD_INITIAL for v in G.nodes()) >= total_switch_load,
        name="total_capacity_satisfies_total_load"
    )

    # m.addConstr(
    #     gp.quicksum(overloaded[v] for v in G.nodes()) >= 0.3 * gp.quicksum(x1[v] for v in G.nodes()),
    #     name="min_30_percent_overloaded"
    # )

    for v in G.nodes():
        m.addConstr(x2[v, v] >= x1[v], name=f"controller_serves_itself_{v}")

    # Step 5: Objective
    # m.setObjective(quicksum(x1[v] for v in G.nodes()), GRB.MINIMIZE)
    a = 0.8
    b = 0.2   # small weight

    m.setObjective(
        a * quicksum(x1[v] for v in G.nodes()) 
        - b * quicksum(overloaded[v] for v in G.nodes()),
        GRB.MINIMIZE
    )

    # Step 6: Solve
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        min_controllers = int(m.objVal)
        selected = [v for v in G.nodes() if x1[v].X > 0.5]
        assigned_capacities = {v: node_capacities[v] for v in selected}
        initial_assignment = {u: v for u in G.nodes() for v in G.nodes() if x2[u, v].X > 0.5}

        if selected and initial_assignment:
            print(f"✔ SUCCESS: Initial assignment successful for {topology_name}")
            return min_controllers, selected, assigned_capacities, initial_assignment, "OPTIMAL"
        else:
            print(f"❌ Initial assignment failed for {topology_name}")
            return None, [], {}, {}, "FAILED_INIT_ASSIGNMENT"

    # Step 7: Handle infeasibility (IIS)
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

    return None, [], {}, {}, status_msg