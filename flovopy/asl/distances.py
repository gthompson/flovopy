# distances.py
from __future__ import annotations

import os
import math
import pickle
from typing import Dict, Tuple, Optional, Sequence, Iterable, Any

import numpy as np
from obspy.core.inventory import Inventory
from obspy import Stream
from obspy.geodetics.base import gps2dist_azimuth

# Only used to *list* station ids; the distance engine does not depend on Grid/NodeGrid internals.
from flovopy.asl.grid import (
    station_ids_from_inventory,
    station_ids_from_stream,
)

# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def _hash16(*parts: Iterable[Any]) -> str:
    """Stable compact hash (16 hex chars)."""
    import hashlib, json
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _station_ids(inv: Optional[Inventory], st: Optional[Stream]) -> Tuple[str, ...]:
    """Prefer Stream ids (subset) else Inventory ids."""
    if st is not None:
        return station_ids_from_stream(st)
    if inv is not None:
        return station_ids_from_inventory(inv)
    raise ValueError("Provide at least one of (inventory, stream).")


def _grid_nodes_lonlat_elev(gridobj) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Return (lon, lat, elev?) 1-D arrays for either:
      - Grid: gridlon/gridlat/(optional)node_elev_m
      - NodeGrid: node_lon/node_lat/(optional)node_elev_m
    """
    # Sparse NodeGrid
    if hasattr(gridobj, "node_lon") and hasattr(gridobj, "node_lat"):
        lon = np.asarray(gridobj.node_lon, float).ravel()
        lat = np.asarray(gridobj.node_lat, float).ravel()
        elev = None
        if getattr(gridobj, "node_elev_m", None) is not None:
            elev = np.asarray(gridobj.node_elev_m, float).ravel()
        return lon, lat, elev

    # Rectangular Grid
    if hasattr(gridobj, "gridlon") and hasattr(gridobj, "gridlat"):
        lon = np.asarray(gridobj.gridlon, float).ravel()
        lat = np.asarray(gridobj.gridlat, float).ravel()
        elev = None
        if getattr(gridobj, "node_elev_m", None) is not None:
            elev = np.asarray(gridobj.node_elev_m, float).ravel()
        return lon, lat, elev

    raise TypeError("gridobj must be a Grid (gridlon/gridlat) or NodeGrid (node_lon/node_lat).")


def _grid_signature_tuple(gridobj) -> Tuple:
    """
    Stable tuple describing the grid for caching. If the object provides
    .signature().as_tuple(), we use it; otherwise fallback to bbox + count.
    """
    if hasattr(gridobj, "signature"):
        sig = gridobj.signature()
        if hasattr(sig, "as_tuple"):
            return tuple(sig.as_tuple())
        try:
            return tuple(sig.__dict__.items())
        except Exception:
            pass

    lon, lat, elev = _grid_nodes_lonlat_elev(gridobj)
    return (
        float(np.nanmin(lat)), float(np.nanmax(lat)),
        float(np.nanmin(lon)), float(np.nanmax(lon)),
        int(lat.size),
        bool(elev is not None),
    )


def _station_coords_from_inventory(inv: Inventory) -> Dict[str, Dict[str, float]]:
    """
    seed_id -> {latitude, longitude, elevation}
    Elevation from channel if present, else station; meters above sea level.
    """
    out: Dict[str, Dict[str, float]] = {}
    for net in inv:
        ncode = net.code
        for sta in net:
            scode = sta.code
            sta_elev = float(sta.elevation or 0.0)
            for cha in sta:
                sid = f"{ncode}.{scode}.{cha.location_code}.{cha.code}"
                elev = float(cha.elevation) if cha.elevation is not None else sta_elev
                out[sid] = {
                    "latitude":  float(cha.latitude),
                    "longitude": float(cha.longitude),
                    "elevation": float(elev),
                }
    return out


def _cache_path(cache_dir: str, grid_sig: Tuple, station_ids: Sequence[str], use_3d: bool) -> str:
    key = _hash16(("Distances", grid_sig), tuple(sorted(station_ids)), bool(use_3d))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"Distances_{key}.pkl")


def _compute_horiz_distances_km(
    nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    st_lon: float, st_lat: float,
) -> np.ndarray:
    """Horizontal great-circle (km) using gps2dist_azimuth."""
    out = np.empty(nodes_lon.size, dtype=float)
    for i, (lo, la) in enumerate(zip(nodes_lon, nodes_lat)):
        dm, _, _ = gps2dist_azimuth(la, lo, st_lat, st_lon)  # gps2… expects (lat, lon)
        out[i] = dm / 1000.0
    return out

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def compute_distances(
    grid,
    *,
    inventory: Inventory,
    stream: Optional[Stream] = None,
    use_elevation: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Compute node→station distances (km) and station coords.

    - Works for both Grid (rectangular) and NodeGrid (sparse).
    - If the grid provides node_elevations *and* use_elevation=True → 3-D:
        d = sqrt(d_horiz^2 + dz^2), with dz in meters.
      otherwise horizontal-only.

    Returns
    -------
    distances_km : dict[seed_id] -> np.ndarray (N_nodes,)
    coords_map   : dict[seed_id] -> {latitude, longitude, elevation}
    """
    ids = station_ids_from_stream(stream) if stream is not None else station_ids_from_inventory(inventory)
    node_lon, node_lat, node_elev = _grid_nodes_lonlat_elev(grid)
    n_nodes = int(node_lon.size)
    want_3d = bool(use_elevation and (node_elev is not None))

    coords_map = _station_coords_from_inventory(inventory)
    distances: Dict[str, np.ndarray] = {}

    for sid in ids:
        if sid not in coords_map:
            continue
        sc = coords_map[sid]
        st_lat = float(sc["latitude"])
        st_lon = float(sc["longitude"])
        st_elev = float(sc.get("elevation", 0.0) or 0.0)

        d_horiz_km = _compute_horiz_distances_km(node_lon, node_lat, st_lon, st_lat)

        if want_3d:
            dz_m = st_elev - np.asarray(node_elev, float)
            d_km = np.sqrt((d_horiz_km * 1000.0) ** 2 + dz_m ** 2) / 1000.0
        else:
            d_km = d_horiz_km

        if d_km.size != n_nodes:
            raise RuntimeError(f"distance vector length {d_km.size} != grid nodes {n_nodes}")
        distances[sid] = d_km

    return distances, coords_map


def compute_or_load_distances(
    grid,
    *,
    inventory: Inventory,
    stream: Optional[Stream] = None,
    cache_dir: str = os.environ.get("FLOVOPY_CACHE", "asl_cache"),
    force_recompute: bool = False,
    use_elevation: bool = True,   # default 3-D if grid provides elevations
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict]:
    """
    Load node→station distances from cache if available, else compute and cache.

    Cache key derives from (grid signature, sorted station IDs, 3-D flag).
    Compatible with Grid and NodeGrid.
    """
    ids = _station_ids(inventory, stream)
    grid_sig = _grid_signature_tuple(grid)
    _, _, node_elev = _grid_nodes_lonlat_elev(grid)
    want_3d = bool(use_elevation and (node_elev is not None))

    path = _cache_path(cache_dir, grid_sig, ids, want_3d)

    if os.path.exists(path) and not force_recompute:
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            dists = bundle["distances"]
            coords = bundle["coords"]
            # ---- MIGRATE/normalize coords (older caches had "elevation_m") ----
            for sid, sc in list(coords.items()):
                if "elevation" not in sc:
                    sc["elevation"] = float(sc.get("elevation_m", 0.0))
            meta = bundle.get("meta", {})
            # sanity: node count must match
            cur_n_nodes = int(_grid_nodes_lonlat_elev(grid)[0].size)
            any_vec = next(iter(dists.values()), np.empty(0, float))
            if int(np.asarray(any_vec).size) != cur_n_nodes:
                raise ValueError("Cached distances do not match current grid node count.")
            meta.update({"from_cache": True, "cache_path": path, "version": "v2"})
            # write-back migrated coords so future loads are clean (best effort)
            try:
                with open(path, "wb") as f:
                    pickle.dump({"distances": dists, "coords": coords, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass
            return dists, coords, meta
        except Exception as e:
            print(f"[DIST:WARN] Failed to load distances cache ({e}); recomputing…")

    distances, coords = compute_distances(
        grid, inventory=inventory, stream=stream, use_elevation=use_elevation
    )
    node_z = getattr(grid, "node_elev_m", None)
    meta = {
        "grid_signature": grid_sig,
        "n_nodes": int(_grid_nodes_lonlat_elev(grid)[0].size),
        "n_stations": len(distances),
        "station_ids": tuple(sorted(distances.keys())),
        "used_3d": want_3d,
        "has_node_elevations": node_z is not None,
        "station_elev_min_m": float(min(sc.get("elevation", 0.0) for sc in coords.values())) if coords else None,
        "station_elev_max_m": float(max(sc.get("elevation", 0.0) for sc in coords.values())) if coords else None,
        "version": "v2",
    }

    bundle = {"distances": distances, "coords": coords, "meta": meta}
    try:
        with open(path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[DIST:WARN] Could not save distances cache: {e}")

    meta = meta | {"from_cache": False, "cache_path": path}
    return distances, coords, meta


def distances_signature(node_distances_km: Dict[str, np.ndarray]) -> tuple:
    """
    Stable mini-signature for a distances dict:
    (sorted station ids, n_nodes, coarse checksum).
    """
    sids = tuple(sorted(node_distances_km.keys()))
    if not sids:
        return ("EMPTY", 0, 0.0)
    n_nodes = int(next(iter(node_distances_km.values())).size)
    acc = 0.0
    for sid in sids:
        arr = np.asarray(node_distances_km[sid], float)
        if arr.size:
            acc += float(arr[0]) + float(arr[-1]) + float(np.nanmean(arr[: min(8, arr.size)]))
    return (sids, n_nodes, round(acc, 6))


def geo_distance_3d_km(lat1: float, lon1: float, elev1_m: float | None,
                       lat2: float, lon2: float, elev2_m: float | None) -> float:
    """
    Great-circle horizontal + vertical delta, in km.
    """
    d_m, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dz_m = float((elev2_m or 0.0) - (elev1_m or 0.0))
    return math.hypot(d_m / 1000.0, dz_m / 1000.0)