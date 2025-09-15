# flovopy/asl/misfit.py
from __future__ import annotations
import numpy as np
from typing import Protocol, Any, Tuple, Iterable, Optional

# plotting / map
import xarray as xr
import pygmt

# local
from flovopy.asl.map import topo_map   # your existing topo map helper


# ---------------------------------------------------------------------
# Backend protocol: pluggable misfit engines
# ---------------------------------------------------------------------
class MisfitBackend(Protocol):
    def prepare(self, aslobj, seed_ids: list[str], *, dtype=np.float32) -> Any:
        """
        Build and return any matrices/context needed for fast evaluation
        (e.g., C, C2, distances). Called once per locate run.
        """
        ...

    def evaluate(
        self,
        y: np.ndarray,     # (nsta,) amplitudes at one time
        ctx: Any,          # object from prepare()
        *,
        min_stations: int = 3,
        eps: float = 1e-9,
    ) -> Tuple[np.ndarray, dict]:
        """
        Return (misfit_vec, extras) for all nodes.
        misfit_vec : (nnodes,) with np.inf for nodes that can’t be evaluated
        extras     : optional dict with per-node stats (e.g., mean/std or Nj)
        """
        ...


# ---------------------------------------------------------------------
# Default backend: std/|mean| per node, computed via dot-products
# (no nsta×nnodes temporary R)
# ---------------------------------------------------------------------
class StdOverMeanMisfit:
    """misfit_j = std(R[:,j]) / (|mean(R[:,j])|+eps), where R = y[:,None] * C."""

    def prepare(self, aslobj, seed_ids: list[str], *, dtype=np.float32):
        # Build corrections matrix C (nsta, nnodes) once
        first_corr = aslobj.amplitude_corrections[seed_ids[0]]
        nnodes = len(first_corr)
        C = np.empty((len(seed_ids), nnodes), dtype=dtype)
        for k, sid in enumerate(seed_ids):
            ck = np.asarray(aslobj.amplitude_corrections[sid], dtype=dtype)
            if ck.size != nnodes:
                raise ValueError(f"Corrections mismatch for {sid}: {ck.size} != {nnodes}")
            C[k, :] = ck
        return {"C": C, "C2": C * C}

    def evaluate(self, y, ctx, *, min_stations=3, eps=1e-9):
        C  = ctx["C"]
        C2 = ctx["C2"]
        dtype = C.dtype

        # finite stations in y
        y_fin = np.isfinite(y)
        N_total = int(y_fin.sum())
        nnodes = C.shape[1]
        if N_total < min_stations:
            return np.full(nnodes, np.inf, dtype=dtype), {"N": 0}

        y0  = np.where(y_fin, y, 0.0).astype(dtype, copy=False)
        y02 = y0 * y0

        C_fin = np.isfinite(C)
        if C_fin.all():
            # dot-products
            S1 = y0  @ C
            S2 = y02 @ C2
            N  = float(N_total)
            mean = S1 / N
            var  = S2 / N - mean * mean
            np.maximum(var, 0.0, out=var)
            std = np.sqrt(var, dtype=dtype)
            misfit = std / (np.abs(mean) + eps)
            return misfit, {"mean": mean, "std": std, "N": N}
        else:
            # mixed-finite corrections
            Nj = (y_fin.astype(np.int16) @ C_fin.astype(np.int16)).astype(np.float32)
            valid = Nj >= min_stations
            S1 = (y0[:, None]  * np.where(C_fin, C,  0.0)).sum(axis=0)
            S2 = (y02[:, None] * np.where(C_fin, C2, 0.0)).sum(axis=0)

            mean = np.divide(S1, Nj, out=np.full_like(S1, np.nan, dtype=dtype), where=Nj > 0)
            var  = np.divide(S2, Nj, out=np.full_like(S2, np.nan, dtype=dtype), where=Nj > 0) - mean * mean
            np.maximum(var, 0.0, out=var)
            std = np.sqrt(var, dtype=dtype)

            misfit = std / (np.abs(mean) + eps)
            misfit[~valid] = np.inf
            return misfit, {"mean": mean, "std": std, "N": Nj}


# ---------------------------------------------------------------------
# Alternative backend: linear fit vs distance → misfit = 1 - R^2
# Requires per-station, per-node distances in aslobj.node_distances_km
# ---------------------------------------------------------------------
class R2DistanceMisfit:
    """
    Misfit_j = 1 - R^2 of linear regression y = a + b * d_j across stations,
    where d_j are station→node distances (km) at node j.
    """
    def prepare(self, aslobj, seed_ids: list[str], *, dtype=np.float32):
        # Build distance matrix D (nsta, nnodes) from aslobj.node_distances_km
        first_sid = seed_ids[0]
        nnodes = len(aslobj.amplitude_corrections[first_sid])  # ensure same length as corrections
        D = np.empty((len(seed_ids), nnodes), dtype=dtype)
        for k, sid in enumerate(seed_ids):
            dk = np.asarray(aslobj.node_distances_km[sid], dtype=dtype)
            if dk.size != nnodes:
                raise ValueError(f"Distance vector mismatch for {sid}: {dk.size} != {nnodes}")
            D[k, :] = dk
        return {"D": D}

    def evaluate(self, y, ctx, *, min_stations=3, eps=1e-9):
        D = ctx["D"]                       # (nsta, nnodes)
        dtype = D.dtype

        y_fin = np.isfinite(y)
        N_total = int(y_fin.sum())
        if N_total < min_stations:
            return np.full(D.shape[1], np.inf, dtype=dtype), {"N": 0}

        y0 = np.where(y_fin, y, 0.0).astype(dtype, copy=False)
        D_fin = np.isfinite(D)             # should be all True; keep robust

        # Effective sample sizes per node (stations with finite y and finite D)
        Nj = (y_fin.astype(np.int16) @ D_fin.astype(np.int16)).astype(np.float32)
        valid = Nj >= min_stations

        X  = np.where(D_fin, D, 0.0)
        X2 = X * X
        Y  = y0[:, None]

        # Sufficient statistics
        Sx  = (X  * y_fin[:, None]).sum(axis=0)    # Σx
        Sx2 = (X2 * y_fin[:, None]).sum(axis=0)    # Σx^2
        Sy  = (Y  * D_fin).sum(axis=0)             # Σy
        Sy2 = ((y0**2)[:, None] * D_fin).sum(axis=0)  # Σy^2
        Sxy = (X * Y).sum(axis=0)                  # Σxy

        N   = Nj
        denom = (N * Sx2 - Sx*Sx)
        b = np.divide(N * Sxy - Sx * Sy, denom, out=np.zeros_like(denom), where=(denom != 0))
        a = (Sy - b * Sx) / np.maximum(N, 1.0)

        # R^2 = 1 - SSE/SST
        SSE = (Sy2
               - 2*a*Sy - 2*b*Sxy
               + (a*a)*N + 2*a*b*Sx + (b*b)*Sx2)
        SST = Sy2 - (Sy*Sy) / np.maximum(N, 1.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            R2 = 1.0 - np.divide(SSE, SST, out=np.zeros_like(SSE), where=(SST > 0))

        misfit = 1.0 - R2
        misfit[~valid] = np.inf
        return misfit.astype(dtype), {"a": a, "b": b, "N": N, "R2": R2}


# ---------------------------------------------------------------------
# Single-time, full per-node misfit (no nsta×nnodes temporaries)
# ---------------------------------------------------------------------
def compute_node_misfit_for_time(
    aslobj,
    time_index: int,
    *,
    misfit_backend: Optional[MisfitBackend] = None,
    min_stations: int = 3,
    eps: float = 1e-9,
    dtype=np.float32,
):
    """
    Recompute the per-node misfit for a single time index using the chosen backend.

    Returns
    -------
    misfit : (nnodes,) np.ndarray
    jstar  : int (index of minimum-misfit node)
    extras : dict with optional per-node stats from the backend
    """
    if misfit_backend is None:
        misfit_backend = StdOverMeanMisfit()

    # Stream → station vector y at that time
    st = aslobj.metric2stream()
    seed_ids = [tr.id for tr in st]
    Y = np.vstack([tr.data.astype(dtype, copy=False) for tr in st])
    y = Y[:, time_index]  # (nsta,)

    # Prepare backend context (e.g., C/C2 or D)
    ctx = misfit_backend.prepare(aslobj, seed_ids, dtype=dtype)

    # Evaluate
    misfit, extras = misfit_backend.evaluate(
        y, ctx, min_stations=min_stations, eps=eps
    )
    jstar = int(np.nanargmin(misfit))
    return misfit, jstar, extras


# ---------------------------------------------------------------------
# Plot: misfit heatmap (colored layer) at time of maximum DR
# ---------------------------------------------------------------------
def plot_misfit_heatmap_for_peak_DR(
    aslobj,
    *,
    backend: Optional[MisfitBackend] = None,
    min_stations: int = 3,
    eps: float = 1e-9,
    dtype=np.float32,
    cmap: str = "turbo",          # e.g., "turbo", "viridis", "hot"
    transparency: int = 35,       # 0 opaque … 100 fully transparent
    zoom_level: int = 0,
    add_labels: bool = True,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
):
    """
    Compute per-node misfit at the time of maximum DR and overlay it as a semi-transparent
    colored raster on top of a grayscale topo_map. Marks the best-misfit node.

    Returns
    -------
    fig : pygmt.Figure
    (misfit_grid : np.ndarray of shape (nlat, nlon))
    """
    if aslobj.source is None:
        print("[PLOT] No source on ASL object; run locate()/fast_locate() first.")
        return None

    # pick time of max DR
    t_idx = int(np.nanargmax(aslobj.source["DR"]))

    # compute misfit per node at that time
    misfit_vec, jstar, _ = compute_node_misfit_for_time(
        aslobj, t_idx, misfit_backend=backend, min_stations=min_stations, eps=eps, dtype=dtype
    )

    # prepare grid for plotting
    grid = aslobj.gridobj
    nlat = getattr(grid, "nlat", grid.gridlat.shape[0])
    nlon = getattr(grid, "nlon", grid.gridlat.shape[1])
    M = misfit_vec.reshape((nlat, nlon))

    # mask inf → NaN (won't draw)
    M = np.where(np.isfinite(M), M, np.nan)

    # build xarray DataArray with proper coords
    lat_axis = getattr(grid, "latrange", None)
    lon_axis = getattr(grid, "lonrange", None)
    if lat_axis is None or lon_axis is None:
        # fallback from 2D arrays
        lat_axis = np.unique(grid.gridlat.ravel())
        lon_axis = np.unique(grid.gridlon.ravel())

    misfit_da = xr.DataArray(
        M,
        coords={"lat": lat_axis, "lon": lon_axis},
        dims=["lat", "lon"],
        name="misfit",
    )

    # grayscale topo base, centered near chosen location
    center_lat = float(aslobj.source["lat"][t_idx])
    center_lon = float(aslobj.source["lon"][t_idx])
    fig = topo_map(
        show=False,
        zoom_level=zoom_level,
        inv=None,                 # overlay stations separately if desired
        add_labels=add_labels,
        centerlat=center_lat,
        centerlon=center_lon,
        topo_color=False,         # grayscale base
        resolution="03s",
        DEM_DIR=None,
        title=title or "Misfit heatmap (peak DR time)",
    )

    # color palette based on finite range
    vmin = float(np.nanmin(M)) if np.isfinite(M).any() else 0.0
    vmax = float(np.nanmax(M)) if np.isfinite(M).any() else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    pygmt.makecpt(cmap=cmap, series=[vmin, vmax], continuous=True)

    # overlay the misfit raster as semi-transparent color layer
    fig.grdimage(
        grid=misfit_da,
        cmap=True,
        transparency=transparency,   # 0=opaque … 100=fully transparent
    )

    # colorbar
    fig.colorbar(frame='+l"Misfit"')

    # mark best node
    lon_flat = grid.gridlon.ravel()
    lat_flat = grid.gridlat.ravel()
    fig.plot(
        x=[lon_flat[jstar]],
        y=[lat_flat[jstar]],
        style="c0.26c",
        fill="red",
        pen="1p,red",
    )

    # show / save
    if outfile:
        fig.savefig(outfile)
        print(f"[PLOT] misfit heatmap saved: {outfile}")
    else:
        fig.show()

    return fig, M