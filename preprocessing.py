# preprocessing.py
import os
import math
import pickle
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time 
import networkx as nx
from geopy.distance import distance as geo_distance
from helpers import FIBER_SEC_PER_KM  # seconds per km in fiber (your constant)
import csv
LOG_FOLDER = "./logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

PENALTY_FACTOR = 2.0





def create_log_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_FOLDER, f"topology_log_{timestamp}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Topology", "Nodes", "Edges", "Directed", "Connected",
            "GeoDuplicates", "LoadAssigned"
        ])
    return path

def append_log(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def list_topologies_by_size(topology_folder, output_file):
    """
    List all .graphml and .gml files sorted by ascending file size and
    write their base names (without extension) to a text file.
    """
    files = [
        (f, os.path.getsize(os.path.join(topology_folder, f)))
        for f in os.listdir(topology_folder)
        if f.endswith('.graphml') or f.endswith('.gml')
    ]
    files.sort(key=lambda x: x[1])  # Sort by file size

    with open(output_file, 'w') as f:
        for filename, _ in files:
            base = os.path.splitext(filename)[0]
            f.write(base + '\n')




def get_topology_path_from_list(index, list_file, topology_folder):
    """
    Get full file path for the topology file based on index and extension check.
    """
    with open(list_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    if index < 0 or index >= len(names):
        raise IndexError("Index out of range")

    basename = names[index]
    for ext in ['.graphml', '.gml']:
        full_path = os.path.join(topology_folder, basename + ext)
        if os.path.exists(full_path):
            return full_path

    raise FileNotFoundError(f"No file found for {basename} with .graphml or .gml")


# =========================
# PATH CACHE
# =========================
PATH_CACHE_DIR = "./cache_paths"
os.makedirs(PATH_CACHE_DIR, exist_ok=True)

def _path_cache_file(topo_name: str, cost_mode: str, k_paths: int) -> str:
    safe = topo_name.replace("/", "_").replace(" ", "_")
    return os.path.join(PATH_CACHE_DIR, f"{safe}__{cost_mode}__K{k_paths}.pkl")


# =========================
# TOPOLOGY LOADING
# =========================
def load_topology(file_path: str) -> Tuple[nx.Graph, str]:
    """
    Robustly loads a topology from .graphml or .gml.

    Guarantees:
      - Undirected simple Graph (no parallel edges)
      - Connected (else raises ValueError)
      - Nodes relabeled to consecutive integers starting at 1
      - Returns (G, geo_duplicates_flag "Yes"/"No")

    GML fallback:
      - If GML parse fails due to duplicate labels, rewrite label lines uniquely, reparse.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    try:
        if ext == ".graphml":
            G_raw = nx.read_graphml(file_path)
        elif ext == ".gml":
            G_raw = nx.read_gml(file_path, label="id")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    except Exception as e:
        # fallback only for .gml
        if ext != ".gml":
            raise

        print(f"⚠️ Raw GML read failed: {e}. Trying fallback label renaming...")

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        label_counter = {}

        for line in lines:
            if "label" in line:
                parts = line.strip().split()
                label = parts[-1].strip('"')

                if label in label_counter:
                    label_counter[label] += 1
                    label = f"{label}_{label_counter[label]}"
                else:
                    label_counter[label] = 0

                new_lines.append(f'    label "{label}"\n')
            else:
                new_lines.append(line)

        temp_path = file_path + ".temp"
        with open(temp_path, "w") as f:
            f.writelines(new_lines)

        G_raw = nx.read_gml(temp_path, label="label")
        os.remove(temp_path)

    # Deduplicate edges: read into MultiGraph then rebuild simple Graph
    G_multi = nx.MultiGraph(G_raw)
    G = nx.Graph()
    G.add_nodes_from(G_multi.nodes(data=True))

    seen_edges = set()
    for u, v, data in G_multi.edges(data=True):
        key = frozenset([str(u), str(v)])
        if key not in seen_edges:
            G.add_edge(u, v, **data)
            seen_edges.add(key)

    # Relabel nodes to ints 1..N
    G = nx.convert_node_labels_to_integers(G, first_label=1)

    # Validate
    if G.is_directed():
        raise ValueError("Graph is directed")
    if not nx.is_connected(G):
        raise ValueError("Graph is not connected")

    # Duplicate geo coordinate check (optional signal)
    geo_duplicates = "No"
    seen_coords = set()
    for n, data in G.nodes(data=True):
        lat = data.get("Latitude")
        lon = data.get("Longitude")
        if lat is not None and lon is not None:
            coord = (round(float(lat), 6), round(float(lon), 6))
            if coord in seen_coords:
                geo_duplicates = "Yes"
                break
            seen_coords.add(coord)

    return G, geo_duplicates



# =========================
# EDGE WEIGHTS / LATENCY
# =========================
def assign_geographical_weights(G: nx.Graph, fallback_meters: float = 1000.0) -> None:
    """
    Sets edge attribute 'weight' in meters using GPS coordinates if available.
    If missing, assigns fallback weight (default 1000m).
    """
    for u, v in G.edges():
        try:
            coord_u = (G.nodes[u]["Latitude"], G.nodes[u]["Longitude"])
            coord_v = (G.nodes[v]["Latitude"], G.nodes[v]["Longitude"])
            if None in coord_u or None in coord_v:
                raise ValueError
            G[u][v]["weight"] = geo_distance(coord_u, coord_v).meters
        except (KeyError, ValueError):
            G[u][v]["weight"] = float(fallback_meters)

# ===========================================================================
# Propagation delay
# =============================================================================
def edge_one_way_latency_sec(G: nx.Graph, u: int, v: int) -> float:
    """
    Uses edge['latency_sec'] if present; else converts edge['weight'] (meters)
    to seconds using FIBER_SEC_PER_KM.
    """
    data = G[u][v]
    if "latency_sec" in data:
        return float(data["latency_sec"])
    if "weight" in data:
        meters = float(data["weight"])
        return meters * (FIBER_SEC_PER_KM / 1000.0)
    # conservative fallback
    return 1.0 * (FIBER_SEC_PER_KM / 1000.0)


def path_latency_sec(G: nx.Graph, p: List[int]) -> float:
    if not p or len(p) < 2:
        return 0.0
    return sum(edge_one_way_latency_sec(G, p[i], p[i + 1]) for i in range(len(p) - 1))

# ========================================================================================


def compute_all_pairs_k_paths(
    G: nx.Graph,
    k_paths: int = 3,
    cost_mode: str = "weight",
) -> dict:
    """
    All-pairs candidate paths: kpaths[u][v] = [cand0, cand1, ..., cand<=K-1]
    Robust: keeps fewer-than-K paths instead of wiping to [].
    """
    UG = G.to_undirected()
    weight_key = "weight"

    nodes = [int(n) for n in UG.nodes()]
    kpaths = {u: {} for u in nodes}

    for u in nodes:
        for v in nodes:
            if u == v:
                kpaths[u][v] = [{
                    "nodes": [u],
                    "edges": [],
                    "lat_sec": 0.0,
                    "ow_ms": 0.0,
                    "rtt_ms": 0.0,
                    "weight_sum": 0.0,
                    "hops": 0,
                }]
                continue

            try:
                gen = nx.shortest_simple_paths(UG, source=u, target=v, weight=weight_key)
                cand = []
                for p in gen:
                    p = [int(x) for x in p]
                    edges = [(min(p[i], p[i+1]), max(p[i], p[i+1])) for i in range(len(p)-1)]
                    lat = path_latency_sec(G, p)
                    cand.append({
                        "nodes": p,
                        "edges": edges,
                        "lat_sec": lat,
                        "ow_ms": 1000.0 * lat,
                        "rtt_ms": 2.0 * 1000.0 * lat,
                        "weight_sum": sum(float(G[p[i]][p[i+1]].get("weight", 1.0)) for i in range(len(p)-1)),
                        "hops": len(p) - 1,
                    })
                    if len(cand) >= k_paths:
                        break

                kpaths[u][v] = cand   # cand may be size 1..K (but not empty unless truly no path)


            except nx.NetworkXNoPath:
                kpaths[u][v] = []
            except Exception as e:
                # keep info for debugging instead of silently wiping
                kpaths[u][v] = []
                # optionally print once in a while:
                # print(f"[KPATHS] failed u={u} v={v} err={e}")

    return kpaths
# ==================================================================================================
# Precomputed shortest path once per topology
# =====================================================================================================

def compute_all_pairs_shortest(
    G: nx.Graph,
    cost_mode: str = "weight",
):
    UG = G.to_undirected()
    weight_key = None if cost_mode == "hops" else "weight"

    dist = {}
    spath = {}

    for src, (dist_map, path_map) in nx.all_pairs_dijkstra(UG, weight=weight_key):
        src = int(src)
        dist[src] = {int(k): float(v) for k, v in dist_map.items()}
        spath[src] = {int(k): [int(x) for x in p] for k, p in path_map.items()}

    return dist, spath

def precompute_shortest_paths_for_topology(
    G: nx.Graph,
    topo_name: str,
    cost_mode: str = "weight",
    use_cache: bool = True,
):
    """
    Precompute and cache once per topology:
      - all-pairs shortest distances
      - one shortest path per pair
      - edge betweenness centrality (EBC)
    """

    cache_file = _path_cache_file(topo_name, cost_mode, k_paths=0)

    # --------------------------------------------------
    # Load from cache
    # --------------------------------------------------
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            blob = pickle.load(f)

        compute_time = blob.get("meta", {}).get("compute_time", 0.0)

        return (
            blob["dist"],
            blob["spath"],
            blob["ebc"],            # ✅ added
            compute_time,
            cache_file,
        )

    # --------------------------------------------------
    # Compute shortest paths
    # --------------------------------------------------
    start = time.perf_counter()

    dist, spath = compute_all_pairs_shortest(G, cost_mode=cost_mode)

    compute_time = time.perf_counter() - start

    # --------------------------------------------------
    # Compute EBC
    # --------------------------------------------------
    ebc_raw = nx.edge_betweenness_centrality(G, weight="weight")

    ebc = {
        tuple(sorted((u, v))): val
        for (u, v), val in ebc_raw.items()
    }

    # --------------------------------------------------
    # Save cache
    # --------------------------------------------------
    blob = {
        "dist": dist,
        "spath": spath,
        "ebc": ebc,   # ✅ added
        "meta": {
            "topo": topo_name,
            "cost_mode": cost_mode,
            "compute_time": compute_time,
            "ts": datetime.now().isoformat(),
        },
    }

    with open(cache_file, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dist, spath, ebc, compute_time, cache_file

def derive_sc_cc_from_shortest(
    dist_all: Dict[int, Dict[int, float]],
    spath_all: Dict[int, Dict[int, List[int]]],
    switches: List[int],
    controllers: List[int],
):
    """
    Extract only required shortest-path data from precomputed all-pairs.

    Returns:
        dij[(s,c)]   : distance
        paths[(s,c)] : node path
        Dcc[a][b]    : controller distance
        Pcc[a][b]    : controller path
    """

    dij = {}
    paths = {}

    Dcc = {c: {} for c in controllers}
    Pcc = {c: {} for c in controllers}

    # --------------------------
    # Switch → Controller
    # --------------------------
    for s in switches:
        for c in controllers:
            d = dist_all.get(s, {}).get(c, float("inf"))
            p = spath_all.get(s, {}).get(c, [])

            dij[(s, c)] = float(d)
            paths[(s, c)] = list(p) if p else []

    # --------------------------
    # Controller ↔ Controller
    # --------------------------
    for a in controllers:
        for b in controllers:
            if a == b:
                Dcc[a][b] = 0.0
                Pcc[a][b] = [a]
            else:
                d = dist_all.get(a, {}).get(b, float("inf"))
                p = spath_all.get(a, {}).get(b, [])

                Dcc[a][b] = float(d)
                Pcc[a][b] = list(p) if p else []

    return dij, paths, Dcc, Pcc

def assign_random_loads(G, seed: int, mean: float = 200.0, std: float = 30.0):
    """
    Assign per-node loads ~ N(mean, std^2).
    Loads are rounded to ints and clipped at 0 (no negatives).
    Reproducible via `seed`.
    """
    rnd = random.Random(seed)
    loads = {}
    for node in G.nodes():
        x = rnd.gauss(mean, std)      # draw from a normal distribution
        load = max(0, int(round(x)))  # no negative loads
        G.nodes[node]['load'] = load
        loads[node] = load
    return loads


# ==========================================================================================
# Compute Bhandari and yen 
# =============================================================================================


def derive_sc_kpaths_for_mcf(
    G: nx.Graph,
    switches: List[int],
    controllers: List[int],
    K: int,
    edge_caps: Dict[Tuple[int, int], float],
    usage_e: Dict[Tuple[int, int], float],
    alpha: float,
) -> Tuple[
    Dict[Tuple[int, int], List[dict]],
    Dict[Tuple[int, int, int], float],
    Dict[Tuple[int, int, int], List[Tuple[int, int]]],
]:
    """
    Generate K candidate paths per (switch, controller) using:

    cost(e) = distance / capacity * (1 + alpha * utilization)

    Steps:
      1. Generate weighted disjoint paths (greedy removal)
      2. Fill remaining using Yen shortest paths
      3. Ensure uniqueness
    """

    switches = [int(s) for s in switches]
    controllers = [int(c) for c in controllers]

    sc_kpaths = {}
    sc_kpath_cost = {}
    sc_kpath_edges = {}

    # ============================================================
    # STEP 1: Assign congestion-aware temporary cost
    # ============================================================
    for u, v, data in G.edges(data=True):

        e = (u, v) if u < v else (v, u)

        dist = float(data.get("weight", 1.0))
        bw = float(edge_caps.get(e, 1.0))
        usage = float(usage_e.get(e, 0.0))

        if bw <= 0:
            bw = 1e-9

        util = min(usage / bw, 1.5)

        data["temp_cost"] = (dist / bw) * (1 + alpha * util)

    # ============================================================
    # STEP 2: Path generation
    # ============================================================
    for s in switches:
        for c in controllers:

            cand_paths = []
            G_work = G.copy()
            seen = set()
            # -------------------------------
            # A) Weighted disjoint paths
            # -------------------------------
            for _ in range(K):
                try:
                    p = nx.shortest_path(G_work, s, c, weight="temp_cost")
                    p = list(p)

                    t = tuple(p)
                    if t not in seen:
                        cand_paths.append(p)
                        seen.add(t)

                    for i in range(len(p) - 1):
                        u1, v1 = p[i], p[i + 1]

                        if G_work.has_edge(u1, v1):
                            data = G_work[u1][v1]

                            # apply multiplicative penalty
                            data["temp_cost"] *= (1.0 + PENALTY_FACTOR)

                except nx.NetworkXNoPath:
                    break

            # -------------------------------
            # B) Yen fallback (if needed)
            # -------------------------------
            if len(cand_paths) < K:
                try:
                    gen = nx.shortest_simple_paths(G_work, s, c, weight="temp_cost")

                    for p in gen:
                        p = list(p)

                        if p not in cand_paths:
                            cand_paths.append(p)

                        if len(cand_paths) >= K:
                            break

                except nx.NetworkXNoPath:
                    pass

            # -------------------------------
            # C) Convert to structured output
            # -------------------------------
            cand_list = []

            for k, p in enumerate(cand_paths):

                edges = [
                    (min(p[i], p[i + 1]), max(p[i], p[i + 1]))
                    for i in range(len(p) - 1)
                ]

                cost = sum(float(G_work[u][v]["temp_cost"]) for (u, v) in edges)

                cand = {
                    "nodes": p,
                    "edges": edges,
                    "cost": cost,
                }

                cand_list.append(cand)

                sc_kpath_cost[(s, c, k)] = cost
                sc_kpath_edges[(s, c, k)] = edges

            sc_kpaths[(s, c)] = cand_list

    # ============================================================
    # STEP 3: Cleanup temp weights
    # ============================================================
    for _, _, data in G.edges(data=True):
        data.pop("temp_cost", None)

    return sc_kpaths, sc_kpath_cost, sc_kpath_edges        