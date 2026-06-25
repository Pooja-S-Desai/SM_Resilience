# cache_init_only.py
from __future__ import annotations
import os, json, hashlib
from typing import Dict, Tuple, List, Callable
import networkx as nx
from steiner_opt import run_steiner_constant_penalty

# Use your helpers' RESULTS_FOLDER + ensure_dir; fall back if helpers missing.
try:
    from helpers import RESULTS_FOLDER, ensure_dir
except Exception:
    RESULTS_FOLDER = "./results"
    def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def compute_or_load_init_only(
    G: nx.Graph,
    topo_name: str,
    master_seed: int,
    seeds: Dict[str, int],
    cost_mode: str,
    get_min_controllers_and_assignment_fn: Callable[..., Tuple[int, List[int], Dict[int, float], Dict[int, int], Dict[int, float], str]],
    loads: Dict[int, float],
    *,
    use_cache: bool = True,
    prefer_cached_loads: bool = True,
):
    """
    Cache ONLY:
      - controller capacities (per controller)
      - initial assignment (switch -> controller)
      - switch loads (per switch)

    No Steiner, no sync, no residual capacities.

    Returns:
      controllers: List[int]
      capacities:  Dict[int, float]
      init_assign: Dict[int, int]
      cachefile:   str (full path in RESULTS_FOLDER/init_cache)
      loads_used:  Dict[int, float]
    """
    # ---------- where caches live ----------
    cache_root = os.path.join(RESULTS_FOLDER, "init_cache")
    ensure_dir(cache_root)

    # ---------- cache key ----------
    key = {
        "topo": str(topo_name),
        "master_seed": int(master_seed),
        "seeds": {k: int(v) for k, v in (seeds or {}).items()},
        "cost_mode": str(cost_mode),
    }
    keyhash = hashlib.blake2b(
        json.dumps(key, sort_keys=True).encode(), digest_size=16
    ).hexdigest()

    cachefile = os.path.join(cache_root, f"{topo_name}_{keyhash}_init_only.json")

    # ---------- FAST PATH: try cache ----------
    if use_cache:
        try:
            with open(cachefile, "r") as fh:
                data = json.load(fh)

            if data.get("_keyhash") == keyhash:
                controllers = [int(x) for x in data["controllers"]]
                capacities  = {int(k): float(v) for k, v in data["capacities"].items()}
                init_assign = {int(k): int(v) for k, v in data["init_assign"].items()}
                cached_loads = {int(k): float(v) for k, v in data.get("loads", {}).items()}

                steiner_data = data.get("steiner")

                if steiner_data is None:
                    raise ValueError(f"{topo_name}: cache missing steiner → delete old cache file")

                nodes = {int(n) for n in G.nodes()}
                controller_set = set(controllers)
                if not controller_set:
                    raise ValueError(f"{topo_name}: cached init has no controllers")
                missing_assign = nodes - set(init_assign)
                bad_assign = {
                    s: c for s, c in init_assign.items()
                    if s in nodes and c not in controller_set
                }
                if missing_assign or bad_assign:
                    raise ValueError(
                        f"{topo_name}: cached init is incomplete/invalid "
                        f"(missing={len(missing_assign)}, bad={len(bad_assign)})"
                    )

                loads_used = cached_loads if (prefer_cached_loads and cached_loads) else loads

                node_capacities = {
                    int(k): float(v)
                    for k, v in data.get("node_capacities", capacities).items()
                }

                return controllers, capacities, init_assign, cachefile, loads_used, steiner_data, "OPTIMAL", node_capacities
        except FileNotFoundError:
            pass
        except Exception:
            # stale/invalid cache → recompute fresh below
            pass

    # ---------- SLOW PATH: compute fresh (expects 4-tuple everywhere) ----------
    try:
        kmin, controllers, capacities, init_assign,node_capacities,status_cs = get_min_controllers_and_assignment_fn(
            G=G,
            loads=loads,
            topology_name=topo_name,
            cost_mode=cost_mode,
            seed_menu=(seeds or {}).get("caps_menu"),
            seed_nodes=(seeds or {}).get("caps_nodes"),
            seeds=seeds,  # ok if callee ignores
        )
        if status_cs != "OPTIMAL":
            return controllers, capacities, init_assign, cachefile, loads, {}, status_cs,node_capacities
        tree_edges, tree_total, steiner_nodes, const_ms = run_steiner_constant_penalty(
            G,
            controllers,
            cost_mode=cost_mode,            # "weight" or "hops"
            two_way=True 
            )
        print(f"[STEINER] edges={len(tree_edges)} total_w={tree_total:.3f} const_ms={const_ms:.3f}")

        steiner_data = {
            "tree_edges": tree_edges,
            "tree_total": float(tree_total),
            "steiner_nodes": list(map(int, steiner_nodes)),
            "const_ms": float(const_ms),
        }   
         
    except TypeError as e:
        raise RuntimeError(
            "get_min_controllers_and_assignment must return "
            "(kmin, controllers, capacities, init_assign, node_capacities, status_cs). "
            "Update controller_selection_module.py accordingly."
        ) from e

    # ---------- write cache (init + loads only) ----------
    if use_cache:
        obj = {
            "_keyhash": keyhash,
            "controllers": [int(x) for x in controllers],
            "capacities": {int(k): float(v) for k, v in (capacities or {}).items()},
            "init_assign": {int(k): int(v) for k, v in (init_assign or {}).items()},
            "loads": {int(k): float(v) for k, v in (loads or {}).items()},
            "steiner": steiner_data,
            "node_capacities": {int(k): float(v) for k, v in (node_capacities or {}).items()},
        }
        with open(cachefile, "w") as fh:
            json.dump(obj, fh, indent=2)

    return controllers, capacities, init_assign, cachefile, loads, steiner_data,status_cs,node_capacities
