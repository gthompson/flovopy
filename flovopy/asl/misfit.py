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
# flovopy/asl/misfit.py

import numpy as np

class R2DistanceMisfit:
    """
    Correlation/R²-style backend between station amplitudes y and a
    per-node regressor built from station→node distances.

    cost_j = alpha * (1 - R2_j) + (1 - alpha) * (std/|mean|)_j

    Options
    -------
    use_log : regress log|y| against log(distance)   (power-law proxy)
    square  : if not use_log, regress y against 1/d  (body-wave proxy). If False, use d.
    alpha   : 1.0 => pure R²; 0.0 => pure StdOverMean (blend in between).
    """
    def __init__(self, use_log: bool = True, square: bool = True, alpha: float = 0.5):
        self.use_log = bool(use_log)
        self.square = bool(square)
        self.alpha = float(alpha)

    def prepare(self, aslobj, seed_ids, dtype=np.float32):
        # Corrections matrix C (nsta, nnodes)
        C = np.empty((len(seed_ids), aslobj.gridobj.gridlat.size), dtype=dtype)
        for k, sid in enumerate(seed_ids):
            C[k, :] = np.asarray(aslobj.amplitude_corrections[sid], dtype=dtype)

        # Pre-stack distances as a matrix D (nsta, nnodes) for speed
        # node_distances_km is {seed_id -> (nnodes,)}; order must match seed_ids
        D = np.vstack([np.asarray(aslobj.node_distances_km[sid], dtype=float) for sid in seed_ids]).astype(dtype, copy=False)

        # Standard backend for the blend term
        from .misfit import StdOverMeanMisfit
        std_backend = StdOverMeanMisfit()
        std_ctx = std_backend.prepare(aslobj, seed_ids, dtype=dtype)

        return {"C": C, "D": D, "seed_ids": tuple(seed_ids),
                "std_backend": std_backend, "std_ctx": std_ctx}

    @staticmethod
    def _r2_from_zscores(yz: np.ndarray, xz: np.ndarray) -> float:
        # R² via Pearson r^2 on z-scored variables
        # (finite-safe; returns [0,1])
        num = np.nansum(yz * xz)
        den = np.sqrt(np.nansum(yz**2) * np.nansum(xz**2))
        if den <= 0:
            return 0.0
        r = num / den
        if not np.isfinite(r):
            return 0.0
        r2 = r * r
        # guard tiny negative from rounding
        return 0.0 if r2 < 0 else (1.0 if r2 > 1.0 else float(r2))

    def evaluate(self, y, ctx, *, min_stations=3, eps=1e-9):
        """
        Parameters
        ----------
        y : (nsta,)
        Returns
        -------
        misfit : (nnodes,) cost (lower is better)
        extras : dict with keys:
           'mean' : per-node mean reduced amplitude (for DR extraction)
           'N'    : per-node usable station count
        """
        C = ctx["C"]                     # (nsta, nnodes)
        D = ctx["D"]                     # (nsta, nnodes)
        nsta, nnodes = C.shape

        fin_y = np.isfinite(y)
        if fin_y.sum() < min_stations:
            return np.full(nnodes, np.inf, dtype=float), {"N": 0}

        # Optionally log the amplitudes (power-law flavor) — use |y| and clip
        y_work = np.where(fin_y, y, 0.0).astype(float, copy=False)
        if self.use_log:
            y_work = np.log(np.clip(np.abs(y_work), 1e-12, None))

        # z-score y over the finite stations
        ym = np.nanmean(y_work[fin_y])
        ys = np.nanstd(y_work[fin_y])
        yz_all = (y_work - ym) / (ys if (np.isfinite(ys) and ys > 0) else 1.0)

        # Optional blend with std/|mean|
        std_misfit = None
        if self.alpha < 1.0:
            std_misfit, _extras_std = ctx["std_backend"].evaluate(
                y, ctx["std_ctx"], min_stations=min_stations, eps=eps
            )
            std_misfit = np.where(np.isfinite(std_misfit), std_misfit, np.inf)

        # R² for each node
        r2 = np.empty(nnodes, dtype=float)
        N_used = np.zeros(nnodes, dtype=int)

        # Precompute correction finite mask once
        Cfin = np.isfinite(C)

        # Build x feature per node from distances:
        # - if use_log: x = log(d)
        # - elif square: x = 1/d
        # - else: x = d
        # Then z-score per node using its valid-station subset.
        for j in range(nnodes):
            mask = fin_y & Cfin[:, j]
            nuse = int(mask.sum())
            N_used[j] = nuse
            if nuse < min_stations:
                r2[j] = 0.0
                continue

            dj = D[:, j].astype(float, copy=False)
            if self.use_log:
                x = np.log(np.clip(dj, 1e-6, None))
            else:
                x = 1.0 / np.clip(dj, 1e-6, None) if self.square else dj

            x = x[mask]
            yz = yz_all[mask]

            # z-score x on the masked subset
            xm = np.nanmean(x)
            xs = np.nanstd(x)
            if not np.isfinite(xs) or xs <= 0:
                r2[j] = 0.0
                continue
            xz = (x - xm) / xs

            r2[j] = self._r2_from_zscores(yz, xz)

        # Cost to minimize: 1 - R²
        r2_cost = 1.0 - r2

        # Blend with std/|mean| if requested
        if self.alpha >= 1.0 or std_misfit is None:
            misfit_out = r2_cost
        elif self.alpha <= 0.0:
            misfit_out = std_misfit
        else:
            misfit_out = self.alpha * r2_cost + (1.0 - self.alpha) * std_misfit

        # Enforce min-station rule and finite guard
        valid_nodes = N_used >= min_stations
        misfit_out = np.where(valid_nodes, misfit_out, np.inf)
        misfit_out = np.where(np.isfinite(misfit_out), misfit_out, np.inf)

        # Extras: per-node mean of reduced amplitudes R = y * C[:, j]
        Rmask = Cfin
        y0 = np.where(fin_y, y, 0.0)
        num = (y0[:, None] * np.where(Rmask, C, 0.0)).sum(axis=0)
        den = (fin_y[:, None] & Rmask).sum(axis=0)  # integer counts per node
        mean = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)

        return misfit_out, {"mean": mean, "N": N_used.astype(float)}

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