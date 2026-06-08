from collections import defaultdict



# --- your modules ---
def stressed_switches_from_edges(stressed_edges):
    sw_stress = defaultdict(int)

    for (u,v), util in stressed_edges:
        sw_stress[u] += 1
        sw_stress[v] += 1

    return sorted(sw_stress.items(), key=lambda x: -x[1])

def is_feasible(usage_e, edge_caps):

    for e, used in usage_e.items():

        cap = edge_caps.get(e, 0)

        if cap <= 0:
            return False

        if used / cap > 1.0:
            return False

    return True

def find_stressed_edges(usage_e, edge_caps, thresh=0.8):
    stressed = []

    for e, used in usage_e.items():
        cap = edge_caps.get(e, 0)
        if cap <= 0:
            continue

        util = used / cap

        if util >= thresh:
            stressed.append((e, util))

    stressed.sort(key=lambda x: -x[1])  # highest stress first
    return stressed



def scale_core_edge_caps(edge_caps, core_edges, stress_factor):
    new_caps = edge_caps.copy()

    # no change case
    if stress_factor == 0:
        return new_caps

    if not (0 < stress_factor < 1):
        raise ValueError(
            f"stress_factor must be in (0,1), got {stress_factor}"
        )

    for e in core_edges:
        if e in new_caps:
            new_caps[e] *= (1.0 - stress_factor)

    return new_caps



def apply_controlled_controller_reduction(
    capacities,
    loads_by_ctrl,
    loads,
    controllers,
    controller_scale_factor,
    threshold=0.8,
    revert_if_infeasible=True,
):
    """
    Controlled degradation of controller capacities.

    Steps:
    1. Identify overloaded controllers (rho >= threshold)
    2. Reduce ONLY their capacities by controller_scale_factor
    3. Check global feasibility:
           total_load <= threshold * total_capacity
    4. If infeasible → revert (optional)

    Returns:
        new_capacities,
        overloaded_controllers,
        feasible (True/False),
        scale_applied (True/False)
    """

    # --- Copy to avoid in-place mutation ---
    base_caps = dict(capacities)
    capacities = dict(base_caps)   # reset controller capacities
    new_caps = dict(capacities)

    # --- Edge case: no scaling ---
    if controller_scale_factor == 0:
        return base_caps, [], True, False

    if not (0 < controller_scale_factor < 1):
        raise ValueError("controller_scale_factor must be in (0,1)")

    # --- Step 1: detect overloaded controllers ---
    overloaded = []
    for c in controllers:
        lam = loads_by_ctrl.get(c, 0.0)
        cap = base_caps.get(c, 0.0)

        if cap <= 0:
            continue

        rho = lam / cap

        if rho >= threshold:
            overloaded.append(c)

    # --- Step 2: reduce only overloaded ---
    for c in overloaded:
        new_caps[c] = base_caps[c] * (1.0 - controller_scale_factor)

    # --- Step 3: global feasibility check ---
    total_load = sum(loads.values())
    total_cap  = sum(new_caps[c] for c in controllers)

    feasible = total_load <= threshold * total_cap

    # --- Step 4: revert if needed ---
    if not feasible and revert_if_infeasible:
        return base_caps, overloaded, False, False

    return new_caps, overloaded, feasible, True






