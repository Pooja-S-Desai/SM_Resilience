import collections,random
from typing import Dict, List, Tuple, Iterable, Any
import math  # used by compute_umax
from helpers import MAX_LOAD

# ----------------------------
# Type aliases for readability
# ----------------------------
Edge = Tuple[int, int]   # undirected edge represented as a pair (min(u,v), max(u,v))
Path = List[int]         # a path is a list of node IDs [n0, n1, ..., nk]


def _e_undirected(u: Any, v: Any) -> Edge:
    """
    Return a canonical (sorted) undirected edge tuple.
    - We cast endpoints to int (in case nodes are strings)
    - We sort so that (3,5) and (5,3) become the same key: (3,5)
    This avoids double-counting edges when accumulating usage.
    """
    return tuple(sorted((int(u), int(v))))


def design_uniform_link_budgets(G, per_link_bits: float) -> Dict[Edge, float]:
    """
    Build a per-edge capacity/budget dictionary where EVERY edge has the same capacity.
    Useful as a simple baseline link budget model.

    Args:
      G               : networkx Graph
      per_link_bits   : capacity in bits/s to assign to each undirected edge

    Returns:
      budget_e: dict mapping (min(u,v), max(u,v)) -> capacity (bits/s)
    """
    budget_e: Dict[Edge, float] = {}
    for u, v in G.edges():
        # canonicalize to undirected edge key
        e = (u, v) if u < v else (v, u)
        budget_e[e] = float(per_link_bits)
    return budget_e


def compute_link_usage_for_assignment(
    assignment: Dict[int, int],
    loads: Dict[int, float],
    paths: Dict[Tuple[int, int], Path],
    msg_bits: float = 128.0
) -> Dict[Edge, float]:
    """
    Compute per-link CONTROL-PLANE traffic (bits/s) induced by an assignment,
    assuming each switch forwards 'loads[s]' requests per second to its controller,
    and each request message has 'msg_bits' bits.

    Args:
      assignment : {switch -> controller} chosen mapping
      loads      : {switch -> req/s} arrival rate at each switch
      paths      : {(s,c) -> [s, ..., c]} precomputed shortest path node lists
      msg_bits   : bits per one control-plane request (OpenFlow PacketIn + overhead)

    Returns:
      usage_e : dict mapping undirected edge -> aggregate bits/s carried
    """
    usage_e: Dict[Edge, float] = collections.defaultdict(float)

    for s, c in assignment.items():
        p = paths.get((s, c))
        if not p or len(p) < 2:
            # No path or trivial path → contributes nothing
            continue

        # Data rate of control traffic from switch s toward controller c (bits/s)
        # load[s] is req/s; multiply by msg size (bits) to get bits/s
        d_ctrl = float(loads[s]) * float(msg_bits)

        # Add that traffic to each undirected edge along the path
        for i in range(len(p) - 1):
            e = _e_undirected(p[i], p[i + 1])
            usage_e[e] += d_ctrl

    return dict(usage_e)


from collections import defaultdict

def usage_on_paths_undirected(
    G,
    paths,
    assignment,
    loads,
    msg_bits,
    edge_caps=None
):

    usage = defaultdict(float)

    # ---------------------------------------
    # Compute link usage (bits/s)
    # ---------------------------------------
    for s, c in assignment.items():

        p = paths.get((s, c))
        if not p or len(p) < 2:
            continue

        d = float(loads.get(s, 0.0)) * float(msg_bits)

        for i in range(len(p)-1):

            u, v = p[i], p[i+1]
            e = (u, v) if u < v else (v, u)

            usage[e] += d

    usage = dict(usage)

    # ---------------------------------------
    # If capacities provided → compute util
    # ---------------------------------------
    if edge_caps is None:
        return usage

    util = {}

    for e, u in usage.items():

        cap = float(edge_caps.get(e, 0.0))

        if cap <= 0:
            util[e] = float("inf")
        else:
            util[e] = u / cap

    return usage, util

def summarize_link_usage(usage_e: Dict[Edge, float], budget_e: Dict[Edge, float]) -> Dict[str, float]:
    """
    Turn raw link usage (bits/s) into a few summary stats using edge budgets (bits/s).

    Definitions:
      util(e) = usage_e[e] / budget_e[e]
      A link is counted as "violated" if util(e) > 0.8 (80%) — configurable threshold.

    Args:
      usage_e  : {(u,v) -> bits/s used}
      budget_e : {(u,v) -> bits/s capacity}  (must be for undirected edges; same keying)

    Returns:
      {
        "violated_links": number of links with util > 0.8,
        "max_util"     : max utilization over all edges (0..inf),
        "mean_util"    : mean utilization over edges that have usage entries,
        "links"        : number of edges that had a finite budget and non-trivial usage
      }
    """
    if not budget_e:
        # No budgets -> can't compute utilization meaningfully
        return {
            "violated_links": 0.0,
            "max_util": 0.0,
            "mean_util": 0.0,
            "links": 0.0
        }

    utils: List[float] = []
    viol = 0

    # Only iterate over edges that appear in usage (others assumed 0 usage)
    for e, used in usage_e.items():
        # Get the capacity; default 0 if missing (will count as violation)
        b = float(budget_e.get(e, 0.0))
        if b <= 0.0:
            # No or zero capacity → count as a violation and skip utilization calc
            viol += 1
            continue

        u = float(used) / b  # utilization fraction
        utils.append(u)

        # Threshold (80%) can be adjusted if needed
        if u > 0.8 + 1e-9:
            viol += 1

    if not utils:
        # No edges with valid budgets were used
        return {
            "violated_links": float(viol),
            "max_util": 0.0,
            "mean_util": 0.0,
            "links": 0.0
        }

    return {
        "violated_links": float(viol),
        "max_util": max(utils),
        "mean_util": sum(utils) / len(utils),
        "links": float(len(utils))
    }


def compute_umax(usage_e: Dict[Edge, float], edge_caps_e: Dict[Edge, float]):
    """
    Compute the worst (maximum) link utilization over all edges with a finite, positive capacity.

    Args:
      usage_e     : {(u,v) -> bits/s used}, edges keyed as undirected tuples
      edge_caps_e : {(u,v) -> bits/s capacity}, undirected

    Returns:
      umax        : maximum utilization (usage/cap) over eligible edges (float)
      worst_e     : edge tuple with the highest utilization (or None)
      worst_usage : bits/s on that worst edge
      worst_cap   : bits/s capacity on that worst edge

    Notes:
      - Edges with cap None, inf, or <= 0 are ignored (treated as ineligible).
      - If an edge doesn’t appear in usage_e, it’s treated as 0 usage.
      - If all edges are ineligible, returns (0.0, None, 0.0, inf).
    """
    umax = -math.inf
    worst_e = None
    worst_usage = 0.0
    worst_cap = float("inf")

    for e, cap in edge_caps_e.items():
        # Skip edges with invalid or infinite capacity
        if cap is None or cap == float("inf") or cap <= 0:
            continue

        # Usage defaults to 0 if not present
        use = float(usage_e.get(e, 0.0))
        u = use / float(cap)  # utilization

        # Track the max
        if u > umax:
            umax = u
            worst_e = e
            worst_usage = use
            worst_cap = float(cap)

    # If no eligible edges were considered, standardize the return
    if umax == -math.inf:
        return 0.0, None, 0.0, float("inf")

    return umax, worst_e, worst_usage, worst_cap



"""
Demand-based bandwidth sizing.

1) Computes traffic on each link from switch→controller flows.
2) Marks top `frac` busiest links as "stressed".
3) Assigns target utilization:
      - stressed links: ~90–98% (near congestion)
      - normal links: ~40–80%
4) Capacity is back-calculated as:
      capacity = traffic / target_utilization

So bandwidth is derived from demand, not fixed.
Higher traffic ⇒ larger capacity.
"""


def design_bandwidths_demand_calibrated(
    G,
    loads,
    msg_bits,
    paths,
    assignment,
    frac,   # % of edges to stress
    seed
):

    import random
    rng = random.Random(int(seed))

    # -------------------------
    # Step 1: traffic per edge
    # -------------------------
    usage_e = collections.defaultdict(float)

    for s,c in assignment.items():
        p = paths.get((s,c), [])
        if len(p) < 2:
            continue

        flow = loads[s] * msg_bits

        for i in range(len(p)-1):
            u,v = p[i], p[i+1]
            e = (u,v) if u<v else (v,u)
            usage_e[e] += flow

    # -------------------------
    # Step 2: choose stressed edges
    # -------------------------
    edges_sorted = sorted(
        usage_e.items(),
        key=lambda x: x[1],
        reverse=True
    )
    k = max(1, int(len(G.edges()) * frac))

   
    stressed_edges = {e for e,_ in edges_sorted[:k]}

    # -------------------------
    # Step 3: capacity sizing
    # -------------------------
    edge_caps = {}


    for u,v in G.edges():

        e = (u,v) if u<v else (v,u)
        used = usage_e.get(e, 0.0)

        if used <= 0:

            # assume small nominal demand
            nominal_used = 1*MAX_LOAD* msg_bits   # or any small flow

            util = random.uniform(0.40, 0.80)

            edge_caps[e] = nominal_used / util
            continue


        if e in stressed_edges:
            util = rng.uniform(0.81, 0.98)
        else:
            util = rng.uniform(0.40, 0.80)

        edge_caps[e] = used / util




    # -------------------------
    # Diagnostics
    # -------------------------
    utils = [usage_e[e]/cap for e,cap in edge_caps.items() if cap>0]

    print("\n--- Traffic-Driven Bandwidth Design ---")
    print(f"Stressed edges = {len(stressed_edges)}")
    print(f"Mean util = {sum(utils)/len(utils):.3f}")
    print(f"Max util = {max(utils):.3f}")
    print(f"Min util = {min(utils):.3f}")

    return edge_caps
def extract_mcf_solution_bundle(
    switches,
    final_assign,
    f,
    *,
    # PATH MODE inputs
    Kset_by_pair=None,
    sc_kpaths=None,
    sc_kpath_cost_ms=None,
    path_edges=None,
    demand_bits=None,

    # ARC MODE inputs
    G=None,
    loads=None,
    msg_bits=None,

    allow_path_splitting=False,
):
    """
    Works for BOTH:
    - PATH MCF (k-path based)
    - ARC MCF (arc-flow based)
    """

    # =========================================================
    # MODE DETECTION
    # =========================================================
    is_path_mode = Kset_by_pair is not None

    paths_used_by_switch = {s: [] for s in switches}
    paths_by_switch = {}
    usage_e = {}

    # =========================================================
    # 🔵 PATH MODE (your existing logic)
    # =========================================================
    if is_path_mode:

        for s in switches:
            c = final_assign.get(s)
            if c is None or s == c:
                paths_by_switch[s] = []
                continue

            ks = Kset_by_pair.get((s, c), [])

            used = []
            for k in ks:
                if (s, c, k) not in f:
                    continue

                frac = float(f[s, c, k].X)
                if frac > 1e-9:
                    used.append({
                        "k": k,
                        "frac": frac,
                        "cost": sc_kpath_cost_ms[(s, c, k)],
                        "edges": path_edges[(s, c, k)],
                    })

                    # usage
                    bits = demand_bits[s] * frac
                    for e in path_edges[(s, c, k)]:
                        usage_e[e] = usage_e.get(e, 0.0) + bits

            paths_used_by_switch[s] = sorted(used, key=lambda x: x["cost"])

            # choose representative path (max cost as you want)
            if used:
                best = max(used, key=lambda x: x["cost"])
                k = best["k"]
                plist = sc_kpaths.get((s, c), [])
                p = plist[k]
                nodes = p.get("nodes", []) if isinstance(p, dict) else p
                paths_by_switch[s] = nodes
            else:
                paths_by_switch[s] = []

        return {
            "paths_used_by_switch": paths_used_by_switch,
            "paths_by_switch": paths_by_switch,
            "usage_e": usage_e,
        }

    # =========================================================
    # 🔴 ARC MODE (NEW unified logic)
    # =========================================================
    else:

        def _undir(u, v):
            return (u, v) if u < v else (v, u)

        for s in switches:
            c = final_assign.get(s)
            if c is None or s == c:
                paths_by_switch[s] = []
                continue

            # reconstruct path from f
            path = [s]
            current = s
            visited = set([s])

            while current != c:
                found = False
                for (ss, cc, u, v) in f.keys():
                    if ss == s and cc == c and u == current and f[ss, cc, u, v].X > 1e-9:
                        if v in visited:
                            break
                        path.append(v)
                        visited.add(v)
                        current = v
                        found = True
                        break

                if not found:
                    break

            paths_by_switch[s] = path

            # usage aggregation
            d_bits = float(loads.get(s, 0.0)) * float(msg_bits)
            for (ss, cc, u, v) in f.keys():
                if ss == s and cc == c:
                    flow = float(f[ss, cc, u, v].X)
                    if flow > 1e-9:
                        e = _undir(u, v)
                        usage_e[e] = usage_e.get(e, 0.0) + flow

        return {
            "paths_used_by_switch": {},  # ARC doesn't track k paths
            "paths_by_switch": paths_by_switch,
            "usage_e": usage_e,
        }