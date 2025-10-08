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
    if cache_dir is None:
        return

    key = _hash16(("Distances", grid_sig), tuple(sorted(station_ids)), bool(use_3d))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"Distances_{key}.pkl")

'''
def _meters_per_degree(lat0: float) -> tuple[float, float]:
    # Simple spherical/ellipsoidal approximation (close enough for local distances)
    # Latitudinal meters/deg is ~111.132 km; longitudinal scales with cos(lat)
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat0))
    return m_per_deg_lat, m_per_deg_lon

def _compute_horiz_distances_km(
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    st_lon: float,
    st_lat: float,
) -> np.ndarray:
    """
    Vectorized local-ENU approximation (km). Accurate to ~<0.5% for small areas.
    """
    lons = np.asarray(nodes_lon, dtype=float)
    lats = np.asarray(nodes_lat, dtype=float)
    m_per_deg_lat, m_per_deg_lon = _meters_per_degree(st_lat)
    dx = (lons - float(st_lon)) * m_per_deg_lon
    dy = (lats - float(st_lat)) * m_per_deg_lat
    return np.hypot(dx, dy) / 1000.0
'''

'''
Old functions that ignored masks - though they still work, just with an overhead
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")  # reuse across calls to avoid reinit cost

def _compute_horiz_distances_km(
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    st_lon: float,
    st_lat: float,
    geod: Geod = _GEOD,
) -> np.ndarray:
    """
    Vectorized great-circle distances (km) on WGS84 using pyproj.Geod.inv.
    nodes_lon/nodes_lat: 1D arrays of same length.
    """
    lons1 = np.asarray(nodes_lon, dtype=float)
    lats1 = np.asarray(nodes_lat, dtype=float)
    # Broadcast station coords to node shape
    lons2 = np.broadcast_to(float(st_lon), lons1.shape)
    lats2 = np.broadcast_to(float(st_lat), lats1.shape)
    _, _, dist_m = geod.inv(lons1, lats1, lons2, lats2)  # vectorized
    return dist_m / 1000.0

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
'''


########### NEW MASK AWARE FUNCTIONS #############
from typing import Dict, Tuple, Optional
import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")  # reuse

def _active_indexer(grid) -> Optional[np.ndarray]:
    """Return flat indices of active nodes, or None if no mask."""
    return grid._node_mask_idx if grid._node_mask_idx is not None else None

def _flatten_lonlat(grid, idx: Optional[np.ndarray]):
    """Return flattened lon/lat (optionally subset by idx)."""
    lon_all = grid.gridlon.ravel()
    lat_all = grid.gridlat.ravel()
    if idx is None:
        return lon_all, lat_all
    return lon_all[idx], lat_all[idx]

def _flatten_elev(grid, idx: Optional[np.ndarray], use_elevation_3d: bool):
    """Return flattened node elevations (meters) or zeros; optionally subset by idx."""
    if (grid.node_elev_m is not None) and use_elevation_3d:
        z = np.asarray(grid.node_elev_m, float).ravel()
    else:
        z = np.zeros(grid.gridlat.size, dtype=float)
    return z if idx is None else z[idx]


def _horiz_dist_km_vectorized(nodes_lon, nodes_lat, st_lon, st_lat, geod=_GEOD) -> np.ndarray:
    """
    Vectorized great-circle distances (km) using pyproj.Geod.inv.
    Robust to empty inputs and NaNs; explicitly broadcasts station coords
    to match node array length for pyproj builds that don't auto-broadcast.
    """
    lon = np.asarray(nodes_lon, float).reshape(-1)
    lat = np.asarray(nodes_lat, float).reshape(-1)

    n = lon.size
    if n != lat.size:
        raise ValueError(
            f"[dist] lon/lat length mismatch: lon={n}, lat={lat.size} "
            "(check gridlon/gridlat and any mask indices)"
        )
    if n == 0:
        return np.empty(0, dtype=float)

    valid = np.isfinite(lon) & np.isfinite(lat)
    out = np.full(n, np.nan, dtype=float)
    if not valid.any():
        return out

    # --- EXPLICIT BROADCAST HERE ---
    st_lon_arr = np.full(valid.sum(), float(st_lon), dtype=float)
    st_lat_arr = np.full(valid.sum(), float(st_lat), dtype=float)

    # Compute only on valid rows
    _, _, dist_m = geod.inv(lon[valid], lat[valid], st_lon_arr, st_lat_arr)
    out[valid] = dist_m / 1000.0
    return out

def distances_mask_aware(
    grid,
    inventory,
    *,
    use_elevation_3d: bool = False,
    dense_output: bool = True,
    mask_dense_threshold: float = 0.90,  # if active_ratio >= threshold, skip subsetting
) -> Dict[str, np.ndarray]:
    """
    Compute distances (km) from grid nodes to each station/channel, honoring Grid masks.
    - Fully vectorized across nodes.
    - If a mask exists and is sufficiently sparse, compute only on active nodes, then scatter.
    - If dense_output=True, return arrays shaped (grid.nlat * grid.nlon,)
      with masked nodes set to np.nan; else return compact arrays of length N_active.

    Returns
    -------
    dict: seed_id -> distances_km (flat vector; compact or dense as requested)
    """
    # Decide whether to subset
    idx = _active_indexer(grid)
    n_all = int(np.asarray(grid.gridlat).size)

    # --- EARLY RETURN FOR EMPTY MASK ---
    if idx is not None and idx.size == 0:
        # No active nodes: return all-NaN dense vectors or empty compact vectors
        dists: Dict[str, np.ndarray] = {}
        if dense_output:
            nanvec = np.full(n_all, np.nan, dtype=float)
        for net in inventory:
            for sta in net:
                for cha in sta:
                    sid = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                    dists[sid] = (nanvec.copy() if dense_output else np.empty(0, dtype=float))
        return dists

    # Optionally treat very dense masks as full
    if idx is not None:
        active_ratio = idx.size / n_all
        if active_ratio >= mask_dense_threshold:
            idx = None

    node_lon, node_lat = _flatten_lonlat(grid, idx)
    node_z   = _flatten_elev(grid, idx, use_elevation_3d)

    if dense_output and idx is not None:
        dense_template = np.full(n_all, np.nan, dtype=float)

    dists: Dict[str, np.ndarray] = {}

    for net in inventory:
        for sta in net:
            sta_elev = float(sta.elevation or 0.0)
            for cha in sta:
                sid   = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                slat  = float(cha.latitude  if cha.latitude  is not None else sta.latitude)
                slon  = float(cha.longitude if cha.longitude is not None else sta.longitude)
                selev = float(cha.elevation if cha.elevation is not None else sta_elev)

                d_horiz = _horiz_dist_km_vectorized(node_lon, node_lat, slon, slat)


                if use_elevation_3d and d_horiz.size:
                    d_km = np.hypot(d_horiz * 1000.0, (node_z - selev)) / 1000.0
                else:
                    d_km = d_horiz

                if dense_output and idx is not None:
                    out = dense_template.copy()
                    if d_km.size:
                        out[idx] = d_km
                    dists[sid] = out
                else:
                    dists[sid] = d_km

    return dists

def compute_or_load_distances(
    grid,
    cache_dir,
    *,
    inventory: Inventory,
    stream: Optional[Stream] = None,
    force_recompute: bool = False,
    use_elevation: bool = True,   # default 3-D if grid provides elevations
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict]:
    """
    Load node→station distances from cache if available, else compute and cache.

    - Uses vectorized distances_mask_aware() under the hood.
    - Cache key includes grid signature, station IDs, 3-D flag, and mask signature (if any).
    - Distances are stored/returned as dense flat vectors (length nlat*nlon), with masked
      nodes set to NaN when a grid mask is active.
    """

    # sanity: lon/lat arrays must be same size
    lonA = np.asarray(grid.gridlon)
    latA = np.asarray(grid.gridlat)
    if lonA.size != latA.size:
        raise ValueError(
            f"[grid] gridlon/gridlat size mismatch: {lonA.shape} vs {latA.shape}"
        )


    ids = _station_ids(inventory, stream)
    grid_sig = _grid_signature_tuple(grid)
    node_lon, node_lat, node_elev = _grid_nodes_lonlat_elev(grid)
    want_3d = bool(use_elevation and (node_elev is not None))
    mask_sig = grid.mask_signature() if hasattr(grid, "mask_signature") else None

    # Build an in-memory cache key identical in spirit to the disk cache
    mem_key = (grid_sig, tuple(sorted(ids)), want_3d, mask_sig)

    # ---- IN-MEMORY CACHE (per grid instance) ----
    # lazily create the cache dict on the grid object
    if not hasattr(grid, "_distance_cache"):
        grid._distance_cache = {}
    # return immediately if we already have it
    if not force_recompute and mem_key in grid._distance_cache:
        bundle = grid._distance_cache[mem_key]
        dists, coords, meta = bundle["distances"], bundle["coords"], bundle["meta"]
        # annotate and return
        meta = dict(meta) | {"from_cache": True, "cache_path": meta.get("cache_path"), "source": "memory"}
        return dists, coords, meta

    # ---- DISK CACHE (existing logic), but don’t spam logs when loading ----
    if cache_dir is None:
        print("[COMPUTE OR LOAD DISTANCES] No cache_dir provided; skipping disk cache.")
    else:
        path = _cache_path(cache_dir, grid_sig, ids, want_3d)
        if mask_sig:
            root, ext = os.path.splitext(path)
            path = f"{root}__{mask_sig}.pkl"

        if os.path.exists(path) and not force_recompute:
            try:
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                dists = bundle["distances"]
                coords = bundle["coords"]
                meta = bundle.get("meta", {})

                # sanity checks (unchanged) ...

                meta.update({
                    "from_cache": True,
                    "cache_path": path,
                    "version": "v3",
                    "mask_signature": mask_sig,
                    "dense_output": True,
                    "source": "disk",
                })

                # put into the in-memory cache so subsequent configs use RAM
                grid._distance_cache[mem_key] = {"distances": dists, "coords": coords, "meta": meta}
                print(f"[COMPUTE OR LOAD DISTANCES] Distances loaded from {path}.")
                return dists, coords, meta

            except Exception as e:
                print(f"[COMPUTE OR LOAD DISTANCES]:WARN] Failed to load distances cache ({e}); recomputing…")

    # ---- COMPUTE FRESH (existing logic) ----
    print("[COMPUTE OR LOAD DISTANCES] Computing fresh distances…")
    coords = _station_coords_from_inventory(inventory)
    distances = distances_mask_aware(
        grid,
        inventory,
        use_elevation_3d=want_3d,
        dense_output=True,
        mask_dense_threshold=0.90,
    )

    # meta (existing) ...
    meta = {
        "grid_signature": grid_sig,
        "n_nodes": int(node_lon.size),
        "n_stations": len(distances),
        "station_ids": tuple(sorted(distances.keys())),
        "used_3d": want_3d,
        "has_node_elevations": (getattr(grid, "node_elev_m", None) is not None),
        "version": "v3",
        "mask_signature": mask_sig,
        "dense_output": True,
    }

    # save to disk (existing) ...

    meta = meta | {"from_cache": False, "cache_path": path, "source": "compute"}

    # **also** save to in-memory cache for this grid
    grid._distance_cache[mem_key] = {"distances": distances, "coords": coords, "meta": meta}

    return distances, coords, meta


def distances_signature(node_distances_km: Dict[str, np.ndarray]) -> tuple:
    """
    Stable mini-signature for a distances dict:
    (sorted station ids, n_nodes, coarse checksum).

    - Never calls np.nanmean on an empty slice
    - Ignores all-NaN head slices
    """
    sids = tuple(sorted(node_distances_km.keys()))
    if not sids:
        return ("EMPTY", 0, 0.0)

    # First array may be empty if the grid has 0 nodes under a mask.
    first = np.asarray(next(iter(node_distances_km.values())), float)
    n_nodes = int(first.size)

    acc = 0.0
    for sid in sids:
        arr = np.asarray(node_distances_km[sid], float)

        # Empty array → contribute nothing; continue safely
        if arr.size == 0:
            raise ValueError(f"[DIST:ERROR] Station {sid} has empty distance array (grid has 0 nodes?)")

        acc += float(arr[0]) + float(arr[-1])

        head = arr[: min(8, arr.size)]
        # Only take mean if there's at least one finite value
        if head.size and np.isfinite(head).any():
            acc += float(np.nanmean(head))

    return (sids, n_nodes, round(acc, 6))


def geo_distance_3d_km(lat1: float, lon1: float, elev1_m: float | None,
                       lat2: float, lon2: float, elev2_m: float | None) -> float:
    """
    Great-circle horizontal + vertical delta, in km.
    """
    d_m, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dz_m = float((elev2_m or 0.0) - (elev1_m or 0.0))
    return math.hypot(d_m / 1000.0, dz_m / 1000.0)

def compute_azimuthal_gap(origin_lat: float, origin_lon: float,
                          station_coords: Iterable[Tuple[float, float]]) -> tuple[float, int]:
    """
    Compute the classical azimuthal gap and station count.

    station_coords: iterable of (lat, lon)
    Returns: (max_gap_deg, n_stations)
    """
    azimuths: list[float] = []
    for stalat, stalon in station_coords:
        _, az, _ = gps2dist_azimuth(origin_lat, origin_lon, stalat, stalon)
        azimuths.append(float(az))

    n = len(azimuths)
    if n < 2:
        return 360.0, n

    azimuths.sort()
    azimuths.append(azimuths[0] + 360.0)
    gaps = [azimuths[i + 1] - azimuths[i] for i in range(n)]
    return max(gaps), n

# Functions to support reducing Trace objects by travel time from dome to station
# flovopy/asl/envelope_locate.py (or a utils module)

def _station_key(seed_id: str, mode: str = "sta") -> str:
    """'MV.MBGB..HHZ' -> 'MBGB' when mode='sta'."""
    if mode == "full":
        return seed_id
    parts = str(seed_id).split(".")
    if len(parts) >= 2:
        net, sta = parts[0], parts[1]
    else:
        sta = parts[0]
    return sta if mode == "sta" else f"{parts[0]}.{sta}"

def _collapse_distances_by_station(
    node_distances_km: Dict[str, np.ndarray],
    prefer_z: bool = True,
    station_key_mode: str = "sta",
) -> Dict[str, np.ndarray]:
    """
    Collapse per-channel keys to one vector per station key.
    Preference: a key ending with 'Z' if available.
    """
    collapsed: Dict[str, np.ndarray] = {}
    chosen: Dict[str, str] = {}
    for full_id, vec in node_distances_km.items():
        key = _station_key(full_id, station_key_mode)
        take = False
        if key not in collapsed:
            take = True
        elif prefer_z and (full_id.endswith("Z") and not chosen[key].endswith("Z")):
            take = True
        if take:
            collapsed[key] = np.asarray(vec, float)
            chosen[key] = full_id
    return collapsed

def _resolve_dome_node_index(grid, dome_location) -> int:
    """
    Accept int index, {'lon','lat'}, or (lon,lat). Returns node index.
    """
    if isinstance(dome_location, int):
        return int(dome_location)
    glon = np.asarray(grid.gridlon).ravel()
    glat = np.asarray(grid.gridlat).ravel()
    if isinstance(dome_location, dict) and {'lon','lat'} <= set(dome_location):
        lon, lat = float(dome_location['lon']), float(dome_location['lat'])
    elif isinstance(dome_location, (tuple, list)) and len(dome_location) >= 2:
        lon, lat = float(dome_location[0]), float(dome_location[1])
    else:
        raise ValueError("dome_location must be node index or {'lon','lat'} or (lon,lat).")
    return int(np.argmin((glon - lon)**2 + (glat - lat)**2))

def shift_stream_by_travel_time(
    st: Stream,
    *,
    grid,
    inventory: Inventory,
    cache_dir: str,
    dome_location,
    speed_km_s: float,
    station_key_mode: str = "sta",
    prefer_z: bool = True,
    use_elevation: bool = True,
    inplace: bool = False,
) -> Tuple[Stream, Dict[str, float]]:
    """
    Shift each station’s trace so that t=0 aligns to the dome-origin time by
    subtracting the dome→station travel time:  starttime -= (distance / speed).

    Returns (shifted_stream, travel_times_s_by_station).
    """
    if speed_km_s is None or not np.isfinite(speed_km_s) or speed_km_s <= 0:
        raise ValueError(f"speed_km_s must be positive; got {speed_km_s}")

    # Load or compute distances for all channels
    node_distances_km, _, _ = compute_or_load_distances(
        grid, cache_dir=cache_dir, inventory=inventory, stream=None, use_elevation=use_elevation
    )
    dist_by_sta = _collapse_distances_by_station(
        node_distances_km, prefer_z=prefer_z, station_key_mode=station_key_mode
    )
    dome_idx = _resolve_dome_node_index(grid, dome_location)

    # Build travel-time lookup (seconds) for available stations
    tt_by_sta: Dict[str, float] = {}
    for sta, dvec in dist_by_sta.items():
        d = float(dvec[dome_idx])
        if np.isfinite(d):
            tt_by_sta[sta] = d / speed_km_s  # km / (km/s) = s

    # Shift copies (or in place) by -tt
    out = st if inplace else st.copy()
    for tr in out:
        sta = tr.stats.station
        tt = tt_by_sta.get(sta)
        if tt is None or not np.isfinite(tt):
            continue
        # Remove travel time so arrivals align to source time
        tr.stats.starttime -= tt

    return out, tt_by_sta