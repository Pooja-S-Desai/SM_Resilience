# steiner_core.py
from __future__ import annotations
from typing import List, Tuple, Optional
import networkx as nx
from helpers import PROPAGATION_SPEED as PROP_M_PER_SEC  # e.g., 2.0e8 m/s

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError("Gurobi is required for the exact Steiner solver.") from e


def _undir(u: int, v: int):
    return (u, v) if u < v else (v, u)


def _edge_cost(G: nx.Graph, u: int, v: int, cost_mode: str) -> float:
    """
    Optimization objective cost (NOT necessarily meters).
    - cost_mode="weight": use edge weight
    - cost_mode="hops":  all edges cost 1
    """
    if cost_mode == "weight" and "weight" in G[u][v]:
        return float(G[u][v]["weight"])
    return 1.0


def solve_steiner_tree_exact(
    G: nx.Graph,
    controllers: List[int],
    *,
    cost_mode: str = "weight",
    timelimit: Optional[float] = None,
    quiet: bool = True
) -> Tuple[List[Tuple[int, int]], float, List[int], float]:
    """
    Returns:
      chosen_edges: list of undirected edges in Steiner tree
      total_cost:   objective cost (depends on cost_mode)
      steiner_nodes: intermediate nodes used
      total_meters: physical length in meters (sum of edge['weight'] if present)
    """
    UG = G.to_undirected()
    S = list(dict.fromkeys(int(x) for x in controllers))
    if len(S) <= 1:
        return [], 0.0, [], 0.0
    if any(t not in UG for t in S):
        raise ValueError("Terminal not in graph.")
    comp = nx.node_connected_component(UG, S[0])
    if any(t not in comp for t in S[1:]):
        raise RuntimeError("Controllers disconnected.")

    r, K = S[0], len(S) - 1
    nodes = list(UG.nodes())
    undirected = sorted({_undir(int(u), int(v)) for u, v in UG.edges()})
    arcs = [(u, v) for (u, v) in undirected] + [(v, u) for (u, v) in undirected]

    c = {e: _edge_cost(UG, e[0], e[1], cost_mode) for e in undirected}

    m = gp.Model("steiner_scflow")
    if quiet:
        m.setParam("OutputFlag", 0)
    if timelimit and timelimit > 0:
        m.setParam("TimeLimit", float(timelimit))

    y = m.addVars(undirected, vtype=GRB.BINARY, name="y")
    f = m.addVars(arcs, lb=0.0, name="f")

    m.setObjective(gp.quicksum(c[e] * y[e] for e in undirected), GRB.MINIMIZE)

    sup = {v: 0.0 for v in nodes}
    sup[r] = float(K)
    for t in S[1:]:
        sup[int(t)] = -1.0

    outA = {n: [] for n in nodes}
    inA = {n: [] for n in nodes}
    for (u, v) in arcs:
        outA[u].append((u, v))
        inA[v].append((u, v))

    for n in nodes:
        m.addConstr(
            gp.quicksum(f[a] for a in outA[n]) - gp.quicksum(f[a] for a in inA[n]) == sup.get(n, 0.0)
        )

    for (u, v) in undirected:
        m.addConstr(f[u, v] + f[v, u] <= float(K) * y[u, v])

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi status {m.Status}")

    chosen = [(u, v) for (u, v) in undirected if y[u, v].X > 0.5]
    total_cost = float(sum(c[e] for e in chosen))

    # Build subgraph
    H = nx.Graph()
    H.add_nodes_from(UG.nodes(data=True))
    for (u, v) in chosen:
        H.add_edge(u, v, **(UG[u][v] if UG.has_edge(u, v) else {}))

    steiner_nodes = [n for n in sorted(set(H.nodes()) - set(S)) if H.degree[n] > 0]

    # PHYSICAL LENGTH IN METERS (always)
    total_meters = 0.0
    for (u, v) in chosen:
        if "weight" in UG[u][v]:
            total_meters += float(UG[u][v]["weight"])
        else:
            total_meters += 1.0  # safe fallback

    return chosen, total_cost, steiner_nodes, float(total_meters)


def run_steiner_constant_penalty(G, controllers, cost_mode="weight", two_way=True):
    # Solve exact Steiner
    tree_edges, total_cost, steiner_nodes, total_meters = solve_steiner_tree_exact(
        G, controllers, cost_mode=cost_mode
    )

    # Convert meters -> ms using physics
    ms_per_meter = 1000.0 / float(PROP_M_PER_SEC)
    const_ms = total_meters * ms_per_meter * (2.0 if two_way else 1.0)

    # Return same items you already consume + keep total_meters in case you want it
    return tree_edges, total_cost, steiner_nodes, const_ms
