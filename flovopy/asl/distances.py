# distances.py
from __future__ import annotations
import os, pickle
from typing import Dict, Tuple, Optional, Sequence, Iterable
import numpy as np
from obspy.core.inventory import Inventory
from obspy import Stream
from obspy.geodetics.base import locations2degrees, degrees2kilometers

# import your grid + hasher
from flovopy.asl.grid import Grid, station_ids_from_inventory, station_ids_from_stream, make_hash

def _station_ids(inv: Optional[Inventory], st: Optional[Stream]) -> Tuple[str, ...]:
    if st is not None:
        return station_ids_from_stream(st)
    if inv is not None:
        return station_ids_from_inventory(inv)
    raise ValueError("Provide at least one of (inventory, stream).")

def _cache_path(cache_dir: str, grid: Grid, station_ids: Sequence[str]) -> str:
    key = make_hash("distances", grid.id, tuple(sorted(station_ids)))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"Distances_{key}.pkl")

def compute_distances(
    grid: Grid,
    *,
    inventory: Inventory,
    stream: Optional[Stream] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Compute node→station distances (km) and station coords for a set of seed IDs."""
    # which seed IDs to use?
    ids = station_ids_from_stream(stream) if stream is not None else station_ids_from_inventory(inventory)

    gridlat = grid.gridlat.reshape(-1)
    gridlon = grid.gridlon.reshape(-1)
    nodelat = np.asarray(gridlat, float)
    nodelon = np.asarray(gridlon, float)
    nnodes = nodelat.size

    distances: Dict[str, np.ndarray] = {}
    coords_map: Dict[str, Dict[str, float]] = {}

    for sid in ids:
        try:
            c = inventory.get_coordinates(sid)
            stalat = float(c["latitude"])
            stalon = float(c["longitude"])
            coords_map[sid] = {
                "latitude": stalat,
                "longitude": stalon,
                "elevation": float(c.get("elevation", 0.0)),
            }
        except Exception as e:
            print(f"[WARN] Skipping {sid}: {e}")
            continue

        # vectorized distance calc (faster than Python loop)
        # locations2degrees isn’t vectorized, so do a small loop over nodes once
        deg = np.array([locations2degrees(nla, nlo, stalat, stalon) for nla, nlo in zip(nodelat, nodelon)], dtype=float)
        km = degrees2kilometers(deg)
        if km.size != nnodes:
            raise RuntimeError(f"distance vector length {km.size} != grid nodes {nnodes}")
        distances[sid] = km

    return distances, coords_map

def compute_or_load_distances(
    grid: Grid,
    *,
    inventory: Inventory,
    stream: Optional[Stream] = None,
    cache_dir: str = "asl_cache",
    force_recompute: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict]:
    """
    Load node→station distances from cache if available, else compute and cache.
    Cache key is derived from grid.id and the sorted station IDs.
    """
    ids = _station_ids(inventory, stream)
    path = _cache_path(cache_dir, grid, ids)

    if os.path.exists(path) and not force_recompute:
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            # basic validation
            nnodes = grid.gridlat.size
            for sid, vec in bundle["distances"].items():
                if np.asarray(vec).size != nnodes:
                    raise ValueError("Cached distances do not match current grid node count.")
            meta = bundle.get("meta", {})
            meta.update({"from_cache": True, "cache_path": path})
            return bundle["distances"], bundle["coords"], meta
        except Exception as e:
            print(f"[WARN] Failed to load distances cache ({e}); recomputing.")

    # compute fresh
    distances, coords = compute_distances(grid, inventory=inventory, stream=stream)
    bundle = {
        "distances": distances,
        "coords": coords,
        "meta": {
            "grid_id": grid.id,
            "n_nodes": grid.gridlat.size,
            "station_ids": ids,
        },
    }
    try:
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
    except Exception as e:
        print(f"[WARN] Could not save distances cache: {e}")

    meta = bundle["meta"] | {"from_cache": False, "cache_path": path}
    return distances, coords, meta