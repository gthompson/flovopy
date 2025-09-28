# flovopy/asl/utils.py
from __future__ import annotations

from typing import Iterable, Tuple, Optional
from types import SimpleNamespace

import numpy as np
from obspy.geodetics import gps2dist_azimuth

__all__ = [
    "compute_spatial_connectedness",
    "_as_regular_view",
    "_movavg_1d",
    "_grid_shape_or_none",
    "_median_filter_indices",
    "_grid_pairwise_km",
    "_viterbi_smooth_indices",
    "_grid_mask_indices",
]

# ---------- helpers ----------



def compute_spatial_connectedness(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    dr: np.ndarray | None = None,
    misfit: np.ndarray | None = None,
    select_by: str = "dr",         # {"dr", "misfit", "all"}
    top_frac: float = 0.15,        # used when select_by in {"dr","misfit"}
    min_points: int = 12,
    max_points: int = 200,
    fallback_if_empty: bool = True,
    eps_km: float = 1e-6,
) -> dict:
    """
    Quantify how tightly clustered the ASL locations are in space.
    score = 1 / (1 + mean_pairwise_distance_km)
    """
    # Defensive clamps
    try:
        top_frac = float(np.clip(top_frac, 0.0, 1.0))
    except Exception:
        top_frac = 0.15

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)

    base_mask = np.isfinite(lat) & np.isfinite(lon)

    if dr is not None:
        dr = np.asarray(dr, float)
    if misfit is not None:
        misfit = np.asarray(misfit, float)

    idx_all = np.flatnonzero(base_mask)
    if idx_all.size == 0:
        return {"score": 0.0, "n_used": 0, "mean_km": np.nan, "median_km": np.nan,
                "p90_km": np.nan, "indices": []}

    # ----- choose subset indices -----
    if select_by == "all":
        idx = idx_all

    elif select_by == "dr":
        if dr is None or not np.isfinite(dr[idx_all]).any():
            if fallback_if_empty:
                if dr is not None and np.isfinite(dr).any():
                    j = int(np.nanargmax(dr))
                    idx = np.array([j], dtype=int)
                else:
                    idx = idx_all
            else:
                return {"score": 0.0, "n_used": 0, "mean_km": np.nan, "median_km": np.nan,
                        "p90_km": np.nan, "indices": []}
        else:
            valid = idx_all[np.isfinite(dr[idx_all])]
            k = max(min_points, int(np.ceil(top_frac * valid.size)))
            k = min(k, max_points, valid.size)
            order = np.argsort(dr[valid])[::-1]  # descending DR
            idx = valid[order[:k]]

    elif select_by == "misfit":
        if misfit is None or not np.isfinite(misfit[idx_all]).any():
            if fallback_if_empty:
                if misfit is not None and np.isfinite(misfit).any():
                    j = int(np.nanargmin(misfit))
                    idx = np.array([j], dtype=int)
                else:
                    idx = idx_all
            else:
                return {"score": 0.0, "n_used": 0, "mean_km": np.nan, "median_km": np.nan,
                        "p90_km": np.nan, "indices": []}
        else:
            valid = idx_all[np.isfinite(misfit[idx_all])]
            k = max(min_points, int(np.ceil(top_frac * valid.size)))
            k = min(k, max_points, valid.size)
            order = np.argsort(misfit[valid])     # ascending misfit
            idx = valid[order[:k]]

    else:
        raise ValueError("select_by must be one of {'dr','misfit','all'}")

    # Degenerate 0–1 point cases
    if idx.size == 0:
        return {"score": 0.0, "n_used": 0, "mean_km": np.nan, "median_km": np.nan,
                "p90_km": np.nan, "indices": []}
    if idx.size == 1:
        return {"score": 1.0, "n_used": 1, "mean_km": 0.0, "median_km": 0.0, "p90_km": 0.0,
                "indices": idx.tolist()}

    # ----- pairwise great-circle distances (vectorized haversine) -----
    R = 6371.0  # km
    phi = np.radians(lat[idx])
    lam = np.radians(lon[idx])
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2.0) ** 2
    d = 2.0 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))  # (K,K), zeros on diag

    iu = np.triu_indices(idx.size, k=1)
    pair = d[iu]
    mean_km = float(np.nanmean(pair))
    median_km = float(np.nanmedian(pair))
    p90_km = float(np.nanpercentile(pair, 90))

    score = 1.0 / (1.0 + max(eps_km, mean_km))

    return {
        "score": float(score),
        "n_used": int(idx.size),
        "mean_km": mean_km,
        "median_km": median_km,
        "p90_km": p90_km,
        "indices": idx.tolist(),
    }


# --- small utilities used inside ASL (kept ASL-agnostic) ---
def _as_regular_view(obj):
    """Ensure we only use attributes ASL actually touches."""
    return SimpleNamespace(
        id=getattr(obj, "id", "gridview"),
        gridlat=np.asarray(obj.gridlat).reshape(-1),
        gridlon=np.asarray(obj.gridlon).reshape(-1),
        nlat=getattr(obj, "nlat", None),
        nlon=getattr(obj, "nlon", None),
        node_spacing_m=getattr(obj, "node_spacing_m", None),
    )


def _movavg_1d(x: np.ndarray, w: int) -> np.ndarray:
    """Simple centered moving average; returns x if w invalid."""
    x = np.asarray(x, float)
    if w is None or w < 3 or w % 2 == 0 or x.size < w:
        return x
    k = np.ones(w, dtype=float) / w
    # fill NaNs with median to avoid gaps
    x_fill = np.where(np.isfinite(x), x, np.nanmedian(x))
    y = np.convolve(x_fill, k, mode="same")
    return y


def _grid_shape_or_none(gridobj):
    if hasattr(gridobj, "shape"):
        return tuple(gridobj.shape) if gridobj.shape else None
    nlat = getattr(gridobj, "nlat", None)
    nlon = getattr(gridobj, "nlon", None)
    glat = np.asarray(gridobj.gridlat).reshape(-1)
    if nlat and nlon and nlat * nlon == glat.size:
        return int(nlat), int(nlon)
    return None

def _median_filter_indices(idx: np.ndarray, win: int) -> np.ndarray:
    """Centered odd-window median filter on integer indices with simple edge padding."""
    if win < 3 or win % 2 == 0:
        return idx
    n, r = idx.size, win // 2
    out = idx.copy()
    # reflect-pad
    pad_left  = idx[1:r+1][::-1] if n > 1 else idx[:1]
    pad_right = idx[-r-1:-1][::-1] if n > 1 else idx[-1:]
    padded = np.concatenate([pad_left, idx, pad_right])
    for i in range(n):
        out[i] = int(np.median(padded[i:i+win]))
    return out


def _grid_pairwise_km(flat_lat: np.ndarray, flat_lon: np.ndarray) -> np.ndarray:
    """Square matrix D[j,k] = great-circle distance (km) between grid nodes j,k."""
    J = flat_lat.size
    D = np.empty((J, J), dtype=float)
    for j in range(J):
        latj, lonj = float(flat_lat[j]), float(flat_lon[j])
        for k in range(J):
            latk, lonk = float(flat_lat[k]), float(flat_lon[k])
            D[j, k] = 0.001 * gps2dist_azimuth(latj, lonj, latk, lonk)[0]
    return D


def _viterbi_smooth_indices(
    misfits_TJ: np.ndarray,    # shape (T, J)
    flat_lat: np.ndarray,      # shape (J,)
    flat_lon: np.ndarray,      # shape (J,)
    lambda_km: float = 1.0,    # step penalty [cost per km]
    max_step_km: float | None = None,  # forbid jumps larger than this (None=disabled)
) -> np.ndarray:
    """
    Dynamic programming path: minimize sum_t ( misfit[t, j_t] + lambda_km * dist(j_{t-1}, j_t) ).
    Returns best indices j_t (shape (T,)).
    """
    T, J = misfits_TJ.shape
    D = _grid_pairwise_km(flat_lat, flat_lon)   # (J, J)
    if max_step_km is not None:
        mask = D > float(max_step_km)
        D = np.where(mask, np.inf, D)

    dp  = np.full((T, J), np.inf, dtype=float)
    bp  = np.full((T, J), -1,   dtype=int)
    dp[0, :] = misfits_TJ[0, :]

    for t in range(1, T):
        trans = dp[t-1, :, None] + lambda_km * D[None, :, :]  # (J, J)
        kbest = np.nanargmin(trans, axis=0)                   # (J,)
        dp[t, :] = misfits_TJ[t, np.arange(J)] + trans[kbest, np.arange(J)]
        bp[t, :] = kbest

    jT = int(np.nanargmin(dp[-1, :]))
    path = np.empty(T, dtype=int)
    path[-1] = jT
    for t in range(T - 2, -1, -1):
        path[t] = int(bp[t + 1, path[t + 1]])
    return path


def _grid_mask_indices(grid) -> np.ndarray | None:
    """
    Prefer a typed mask getter if the grid provides one; otherwise probe common attributes.
    Returns flat int indices or None.
    """
    # 1) Preferred: first-class API on Grid/NodeGrid
    if hasattr(grid, "get_mask_indices") and callable(getattr(grid, "get_mask_indices")):
        idx = grid.get_mask_indices()
        if idx is None:
            return None
        idx = np.asarray(idx, int).ravel()
        return idx if idx.size else np.array([], dtype=int)

    # 2) Legacy attribute probing (for foreign grid-like objects)
    cand = ("_node_mask_idx", "node_mask_idx", "node_mask", "mask", "valid_mask", "land_mask")
    nn = int(np.asarray(getattr(grid, "gridlat")).size)
    for name in cand:
        if not hasattr(grid, name):
            continue
        raw = getattr(grid, name)
        if raw is None:
            continue
        arr = np.asarray(raw)
        if arr.dtype == bool:
            b = arr.reshape(-1) if arr.size == nn else arr
            if b.size != nn:
                raise ValueError("Grid mask boolean length != number of nodes")
            return np.flatnonzero(b).astype(int)
        else:
            idx = arr.reshape(-1).astype(int)
            return idx if idx.size else np.array([], dtype=int)
    return None