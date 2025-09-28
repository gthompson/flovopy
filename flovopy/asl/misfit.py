# flovopy/asl/misfit.py
from __future__ import annotations
from typing import Protocol, Any, Tuple, Optional, List
import numpy as np


# ---------------------------------------------------------------------
# Pluggable misfit interface
# ---------------------------------------------------------------------
class MisfitBackend(Protocol):
    def prepare(self, aslobj, seed_ids: List[str], *, dtype=np.float32) -> Any: ...
    def evaluate(
        self,
        y: np.ndarray,     # (nsta,)
        ctx: Any,          # object from prepare()
        *,
        min_stations: int = 3,
        eps: float = 1e-9,
    ) -> Tuple[np.ndarray, dict]: ...
    

# ---------------------------------------------------------------------
# Default backend: std(R)/(|mean(R)|+eps), R = y[:,None] * C
# Vectorized via dot-products; handles mixed-finite C
# ---------------------------------------------------------------------
class StdOverMeanMisfit:
    """misfit_j = std(R[:,j]) / (|mean(R[:,j])|+eps), where R = y[:,None] * C."""

    def prepare(self, aslobj, seed_ids: List[str], *, dtype=np.float32):
        # Build corrections matrix C (nsta, nnodes)
        first_corr = aslobj.amplitude_corrections[seed_ids[0]]
        nnodes = len(first_corr)
        C = np.empty((len(seed_ids), nnodes), dtype=dtype)
        for k, sid in enumerate(seed_ids):
            ck = np.asarray(aslobj.amplitude_corrections[sid], dtype=dtype)
            if ck.size != nnodes:
                raise ValueError(f"Corrections mismatch for {sid}: {ck.size} != {nnodes}")
            C[k, :] = ck
        C2 = C * C

        ctx = {"C": C, "C2": C2}

        # Optional sub-grid masking provided by ASL.refine_and_relocate()
        mask = getattr(aslobj, "_node_mask", None)
        if mask is not None:
            mask = np.asarray(mask, int)
            ctx["C"] = C[:, mask]
            ctx["C2"] = C2[:, mask]
            ctx["node_index"] = mask  # local→global mapping for callers

        return ctx

    def evaluate(self, y, ctx, *, min_stations=3, eps=1e-9):
        C  = ctx["C"]
        C2 = ctx["C2"]
        dtype = C.dtype

        y_fin = np.isfinite(y)
        N_total = int(y_fin.sum())
        nnodes = C.shape[1]
        if N_total < min_stations:
            return np.full(nnodes, np.inf, dtype=dtype), {"N": 0}

        y0  = np.where(y_fin, y, 0.0).astype(dtype, copy=False)
        y02 = y0 * y0

        C_fin = np.isfinite(C)
        if C_fin.all():
            # Fast path (all finite)
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
            # Mixed-finite corrections
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
# Alternative backend: correlation/R² against distance-based regressor
# cost_j = alpha*(1 - R²_j) + (1-alpha)*(std/|mean|)_j
# ---------------------------------------------------------------------
class R2DistanceMisfit:
    """
    Correlate station amplitudes y with a distance-derived feature per node.

    Options
    -------
    use_log : regress log|y|; distance feature uses log(d)
    square  : if not use_log, use 1/d (body-wave proxy); else use d
    alpha   : 1.0 => pure R² cost; 0.0 => pure StdOverMean
    """
    def __init__(self, use_log: bool = True, square: bool = True, alpha: float = 0.5):
        self.use_log = bool(use_log)
        self.square  = bool(square)
        self.alpha   = float(alpha)

    def prepare(self, aslobj, seed_ids: List[str], *, dtype=np.float32):
        # Corrections matrix C (nsta, nnodes)
        first_corr = aslobj.amplitude_corrections[seed_ids[0]]
        nnodes = len(first_corr)
        C = np.empty((len(seed_ids), nnodes), dtype=dtype)
        for k, sid in enumerate(seed_ids):
            ck = np.asarray(aslobj.amplitude_corrections[sid], dtype=dtype)
            if ck.size != nnodes:
                raise ValueError(f"Corrections mismatch for {sid}: {ck.size} != {nnodes}")
            C[k, :] = ck

        # Distances matrix D (nsta, nnodes) from dict seed_id -> (nnodes,)
        D = np.vstack([np.asarray(aslobj.node_distances_km[sid], dtype=float) for sid in seed_ids]).astype(dtype, copy=False)

        # Standard backend for blending
        std_backend = StdOverMeanMisfit()
        std_ctx = std_backend.prepare(aslobj, seed_ids, dtype=dtype)

        ctx = {"C": C, "D": D, "seed_ids": tuple(seed_ids),
               "std_backend": std_backend, "std_ctx": std_ctx}

        # Optional sub-grid masking
        mask = getattr(aslobj, "_node_mask", None)
        if mask is not None:
            mask = np.asarray(mask, int)
            ctx["C"] = C[:, mask]
            ctx["D"] = D[:, mask]
            ctx["node_index"] = mask  # local→global for the caller

        return ctx

    @staticmethod
    def _r2_from_zscores(yz: np.ndarray, xz: np.ndarray) -> float:
        num = np.nansum(yz * xz)
        den = np.sqrt(np.nansum(yz**2) * np.nansum(xz**2))
        if den <= 0:
            return 0.0
        r = num / den
        if not np.isfinite(r):
            return 0.0
        r2 = r * r
        # numeric guard
        if r2 < 0.0: return 0.0
        if r2 > 1.0: return 1.0
        return float(r2)

    def evaluate(self, y, ctx, *, min_stations=3, eps=1e-9):
        C = ctx["C"]               # (nsta, nnodes)
        D = ctx["D"]               # (nsta, nnodes)
        nsta, nnodes = C.shape

        fin_y = np.isfinite(y)
        if fin_y.sum() < min_stations:
            return np.full(nnodes, np.inf, dtype=float), {"N": 0}

        # Amplitude transform (optional log on |y|)
        y_work = np.where(fin_y, y, 0.0).astype(float, copy=False)
        if self.use_log:
            y_work = np.log(np.clip(np.abs(y_work), 1e-12, None))

        # z-score across usable stations
        ym = np.nanmean(y_work[fin_y])
        ys = np.nanstd(y_work[fin_y])
        y_z = (y_work - ym) / (ys if (np.isfinite(ys) and ys > 0) else 1.0)

        # Optional blend with std/|mean|
        std_misfit = None
        if self.alpha < 1.0:
            std_misfit, _ = ctx["std_backend"].evaluate(
                y, ctx["std_ctx"], min_stations=min_stations, eps=eps
            )
            std_misfit = np.where(np.isfinite(std_misfit), std_misfit, np.inf)

        # Node loop (O(nnodes * nsta))
        Cfin = np.isfinite(C)
        r2 = np.empty(nnodes, dtype=float)
        N_used = np.zeros(nnodes, dtype=int)

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
                x = (1.0 / np.clip(dj, 1e-6, None)) if self.square else dj

            x = x[mask]
            yz = y_z[mask]

            # z-score x on this subset
            xm = np.nanmean(x)
            xs = np.nanstd(x)
            if not np.isfinite(xs) or xs <= 0:
                r2[j] = 0.0
                continue
            xz = (x - xm) / xs

            r2[j] = self._r2_from_zscores(yz, xz)

        # Cost to minimize
        r2_cost = 1.0 - r2
        if self.alpha >= 1.0 or std_misfit is None:
            misfit_out = r2_cost
        elif self.alpha <= 0.0:
            misfit_out = std_misfit
        else:
            misfit_out = self.alpha * r2_cost + (1.0 - self.alpha) * std_misfit

        # Enforce min-station rule & finite guard
        valid_nodes = N_used >= min_stations
        misfit_out = np.where(valid_nodes, misfit_out, np.inf)
        misfit_out = np.where(np.isfinite(misfit_out), misfit_out, np.inf)

        # Extras: per-node mean reduced amplitude (consistent with default backend)
        Rmask = Cfin
        y0 = np.where(fin_y, y, 0.0)
        num = (y0[:, None] * np.where(Rmask, C, 0.0)).sum(axis=0)
        den = (fin_y[:, None] & Rmask).sum(axis=0)
        mean = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)

        return misfit_out, {"mean": mean, "N": N_used.astype(float)}


# ---------------------------------------------------------------------
# Plot: misfit heatmap (colored layer) at time of maximum DR
# ---------------------------------------------------------------------



def compute_node_misfit_for_time(
    aslobj,
    time_index: int,
    *,
    misfit_backend: Optional[MisfitBackend] = None,
    min_stations: int = 3,
    eps: float = 1e-9,
    dtype=np.float32,
) -> Tuple[np.ndarray, int, dict]:
    """
    Recompute per-node misfit at a single time index using the chosen backend.

    Returns
    -------
    misfit_vec : (nnodes_visible,) np.ndarray    (after any node-mask/subgrid)
    jstar      : int                              index of min-misfit in this view
    extras     : dict                             backend extras (e.g., mean, N)
    """

    if misfit_backend is None:
        misfit_backend = StdOverMeanMisfit()

    # Stream → station-by-time matrix, then pull one time column
    st = aslobj.metric2stream()
    seed_ids = [tr.id for tr in st]
    Y = np.vstack([tr.data.astype(dtype, copy=False) for tr in st])
    y = Y[:, time_index]  # (nsta,)

    # Prepare backend context (respects ASL._node_mask if backend implements it)
    ctx = misfit_backend.prepare(aslobj, seed_ids, dtype=dtype)

    # Evaluate misfit for all visible nodes
    misfit_vec, extras = misfit_backend.evaluate(
        y, ctx, min_stations=min_stations, eps=eps
    )
    jstar = int(np.nanargmin(misfit_vec))
    return misfit_vec, jstar, extras


def plot_misfit_heatmap_for_peak_DR(
    aslobj,
    *,
    backend: Optional[MisfitBackend] = None,
    min_stations: int = 3,
    eps: float = 1e-9,
    dtype=np.float32,
    cmap: str = "turbo",
    transparency: int = 35,
    zoom_level: int = 0,
    add_labels: bool = True,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    region: Optional[Tuple[float, float, float, float]] = None,
    dem_tif: Optional[str] = None,
    simple_basemap: bool = True,   # passed through to topo_map
):
    """
    Compute per-node misfit at the time of maximum DR and overlay it (semi-transparent)
    on top of a topo_map() basemap. Marks the best node.
    """
    import xarray as xr
    import pygmt
    from datetime import datetime
    from flovopy.asl.map import topo_map
    ts = lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    if aslobj.source is None:
        print(f"[{ts()}] [MISFIT] No source; run locate()/fast_locate() first.")
        return None

    try:
        # ---- pick the time of max DR
        t_idx = int(np.nanargmax(aslobj.source["DR"]))
        print(f"[{ts()}] [MISFIT] peak DR index={t_idx}")

        # ---- per-node misfit at that time
        misfit_vec, jstar_local, meta = compute_node_misfit_for_time(
            aslobj, t_idx, misfit_backend=backend, min_stations=min_stations, eps=eps, dtype=dtype
        )
        print(f"[{ts()}] [MISFIT] misfit_vec shape={misfit_vec.shape} finite={np.isfinite(misfit_vec).sum()}")

        # ---- reconstruct full (nlat, nlon) field (handle optional mask)
        mask = getattr(aslobj, "_node_mask", None)
        grid = aslobj.gridobj
        nlat = getattr(grid, "nlat", grid.gridlat.shape[0])
        nlon = getattr(grid, "nlon", grid.gridlat.shape[1])

        M_full = np.full((nlat * nlon,), np.nan, dtype=float)
        if mask is None:
            M_full[:] = misfit_vec
            jstar_global = jstar_local
        else:
            mask = np.asarray(mask, int)
            if misfit_vec.shape[0] != mask.shape[0]:
                raise ValueError(f"Mask length mismatch: misfit_vec={misfit_vec.shape[0]} mask={mask.shape[0]}")
            M_full[mask] = misfit_vec
            jstar_global = int(mask[jstar_local])

        M = M_full.reshape((nlat, nlon))
        print(f"[{ts()}] [MISFIT] grid M shape={M.shape} finite={np.isfinite(M).sum()}/{M.size}")

        # ---- coordinate axes (ensure monotonic latitude for plotting)
        lat_axis = getattr(grid, "latrange", None)
        lon_axis = getattr(grid, "lonrange", None)
        if lat_axis is None or lon_axis is None:
            lat_axis = np.unique(grid.gridlat.ravel())
            lon_axis = np.unique(grid.gridlon.ravel())

        flipped = False
        if lat_axis[0] > lat_axis[-1]:
            lat_axis = lat_axis[::-1]
            M = M[::-1, :]
            flipped = True
            print(f"[{ts()}] [MISFIT] flipped latitude axis to increasing")

        misfit_da = xr.DataArray(M, coords={"lat": lat_axis, "lon": lon_axis}, dims=["lat", "lon"], name="misfit")

        # ---- region: use caller’s region if given; else tight extents with a little pad
        if region is None:
            dlon = max(1e-4, 0.02)
            dlat = max(1e-4, 0.02)
            region = [float(lon_axis.min() - dlon), float(lon_axis.max() + dlon),
                      float(lat_axis.min() - dlat), float(lat_axis.max() + dlat)]

        center_lat = float(aslobj.source["lat"][t_idx])
        center_lon = float(aslobj.source["lon"][t_idx])

        # ---- basemap via topo_map (this is the user’s requirement)
        fig = topo_map(
            show=False,
            zoom_level=zoom_level,
            inv=None,
            add_labels=add_labels,
            centerlat=center_lat,
            centerlon=center_lon,
            topo_color=False,           # keep base subdued so heatmap stands out
            region=region,
            dem_tif=dem_tif,
            title=title or "Misfit heatmap (peak DR time)",
        )
        print(f"[{ts()}] [MISFIT] basemap ready (region={region})")

        # ---- color scale from finite values
        finite = np.isfinite(M)
        if finite.any():
            vmin = float(np.nanmin(M[finite])); vmax = float(np.nanmax(M[finite]))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
        pygmt.makecpt(cmap=cmap, series=[vmin, vmax], continuous=True)
        print(f"[{ts()}] [MISFIT] CPT: {cmap} [{vmin:.3g}, {vmax:.3g}]")

        # ---- overlay the misfit raster (semi-transparent)
        fig.grdimage(grid=misfit_da, cmap=True, transparency=int(transparency))
        fig.colorbar(frame='+l"Misfit"')

        # ---- mark best node
        lon_flat = grid.gridlon.ravel()
        lat_flat = grid.gridlat.ravel()
        best_lon = float(lon_flat[jstar_global])
        best_lat = float(lat_flat[jstar_global])
        fig.plot(x=[best_lon], y=[best_lat], style="c0.26c", fill="red", pen="1p,red")
        print(f"[{ts()}] [MISFIT] best node @ (lon,lat)=({best_lon:.5f},{best_lat:.5f}) idx={jstar_global}")

        if outfile:
            fig.savefig(outfile)
            print(f"[{ts()}] [MISFIT] saved: {outfile}")
        else:
            fig.show()

        return fig, M

    except Exception as e:
        import traceback
        print(f"[{ts()}] [MISFIT:ERR] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

# --- in flovopy/asl/misfit.py ---

import numpy as np

class LinearizedDecayMisfit:
    """
    Linearized amplitude-decay backend:

      log|A_i| ≈ A0_log  - N * log(R_ij)  - k * R_ij

    For each node j:
      - build regressors X = [log(R_ij), R_ij]
      - regress y = log|A_i| (finite stations)
      - score with R^2; misfit = 1 - R^2 (lower is better)

    Options
    -------
    f_hz, v_kms : used only to report the implied Q from k_hat (Q = π f / (k v))
    alpha       : blend with StdOverMean (0..1). alpha=1.0 => pure linearized decay
    clip_R_min  : avoid log(0), 1/R blowups; in km
    """

    def __init__(self, f_hz=8.0, v_kms=1.5, alpha=1.0, clip_R_min=1e-3):
        self.f_hz = float(f_hz)
        self.v_kms = float(v_kms)
        self.alpha = float(alpha)
        self.clip_R_min = float(clip_R_min)

    def prepare(self, aslobj, seed_ids, *, dtype=np.float32):
        # Corrections matrix (nsta, nnodes)
        nnodes = aslobj.gridobj.gridlat.size
        C = np.empty((len(seed_ids), nnodes), dtype=dtype)
        for k, sid in enumerate(seed_ids):
            C[k, :] = np.asarray(aslobj.amplitude_corrections[sid], dtype=dtype)

        # Distances stacked into D (nsta, nnodes)
        D = np.vstack([np.asarray(aslobj.node_distances_km[sid], dtype=float) for sid in seed_ids]).astype(dtype, copy=False)

        # Optional blend backend
        from .misfit import StdOverMeanMisfit
        std_backend = StdOverMeanMisfit()
        std_ctx = std_backend.prepare(aslobj, seed_ids, dtype=dtype)

        ctx = {
            "C": C, "D": D, "seed_ids": tuple(seed_ids),
            "std_backend": std_backend, "std_ctx": std_ctx
        }

        # Honor ASL sub-grid masks (refine_and_relocate)
        mask = getattr(aslobj, "_node_mask", None)
        if mask is not None:
            mask = np.asarray(mask, int)
            ctx["C"] = C[:, mask]
            ctx["D"] = D[:, mask]
            ctx["node_index"] = mask  # local→global mapping

        return ctx

    def evaluate(self, y, ctx, *, min_stations=3, eps=1e-9):
        C = ctx["C"]               # (nsta, nnodes)
        D = ctx["D"]               # (nsta, nnodes)
        nsta, nnodes = C.shape

        fin_y = np.isfinite(y)
        if fin_y.sum() < min_stations:
            return np.full(nnodes, np.inf, dtype=float), {"N": 0}

        # Reduced amplitudes are NOT needed for the regression itself (we use raw y),
        # but we still provide per-node 'mean' so ASL can expose DR consistently.
        # (This mirrors StdOverMean behavior.)
        y0 = np.where(fin_y, y, 0.0).astype(float, copy=False)

        # Build per-node stats
        r2_vec   = np.zeros(nnodes, dtype=float)
        N_used   = np.zeros(nnodes, dtype=int)
        N_hat    = np.full(nnodes, np.nan, dtype=float)
        k_hat    = np.full(nnodes, np.nan, dtype=float)
        A0_log   = np.full(nnodes, np.nan, dtype=float)

        # We’ll also return mean(R= y*C) like the other backends
        Cfin = np.isfinite(C)
        num = (y0[:, None] * np.where(Cfin, C, 0.0)).sum(axis=0)
        den = (fin_y[:, None] & Cfin).sum(axis=0).astype(float)
        mean = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)

        # Prepare y for regression in log-space (|y|, clipped)
        y_reg = np.where(fin_y, np.log(np.clip(np.abs(y), 1e-12, None)), np.nan)

        # Loop nodes
        for j in range(nnodes):
            mask = fin_y & Cfin[:, j]
            nuse = int(mask.sum())
            N_used[j] = nuse
            if nuse < min_stations:
                r2_vec[j] = 0.0
                continue

            Rj = np.clip(D[:, j].astype(float, copy=False), self.clip_R_min, None)
            logR = np.log(Rj[mask])
            Rlin = Rj[mask]
            yj = y_reg[mask]

            # X beta ≈ y  with X = [logR, R]
            # Add column of ones for intercept (A0_log)
            X = np.column_stack([np.ones_like(logR), logR, Rlin])  # [A0_log, -N, -k]
            # Least-squares
            try:
                beta, _, _, _ = np.linalg.lstsq(X, yj, rcond=None)
            except np.linalg.LinAlgError:
                r2_vec[j] = 0.0
                continue

            A0, b_logR, b_R = beta
            # Map to parameters (we fit y ≈ A0 + b1*logR + b2*R, expected b1≈-N, b2≈-k)
            A0_log[j] = float(A0)
            N_hat[j]  = float(-b_logR)
            k_hat[j]  = float(-b_R)

            # R^2 on this masked subset
            yhat = X @ beta
            ss_res = np.nansum((yj - yhat) ** 2)
            ss_tot = np.nansum((yj - np.nanmean(yj)) ** 2)
            r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
            # clip to [0,1] for robustness
            r2_vec[j] = float(0.0 if r2 < 0 else (1.0 if r2 > 1.0 else r2))

        # Misfit to minimize
        misfit_r2 = 1.0 - r2_vec

        # Optional blend with StdOverMean (alpha in [0,1])
        if self.alpha < 1.0:
            std_misfit, _ = ctx["std_backend"].evaluate(y, ctx["std_ctx"], min_stations=min_stations, eps=eps)
            std_misfit = np.where(np.isfinite(std_misfit), std_misfit, np.inf)
            misfit = self.alpha * misfit_r2 + (1.0 - self.alpha) * std_misfit
        else:
            misfit = misfit_r2

        # Enforce min-stations
        valid = N_used >= min_stations
        misfit = np.where(valid, misfit, np.inf)

        # --- DR proxy in *linear* domain (independent of log-regression) ---
        Cfin = np.isfinite(C)
        y_lin = np.where(np.isfinite(y), y, 0.0)
        num = (y_lin[:, None] * np.where(Cfin, C, 0.0)).sum(axis=0)
        den = (np.isfinite(y)[:, None] & Cfin).sum(axis=0)
        mean_linear = np.divide(
            num, den,
            out=np.full_like(num, np.nan, dtype=float),
            where=den > 0
        )


        # Bundle extras
        extras = {
            "mean": mean_linear,             # allows DR extraction at best node
            #"N": N_used.astype(float),
            "N": den.astype(float),
            "N_hat": N_hat,
            "k_hat": k_hat,
            "A0_log": A0_log,
            "r2": r2_vec,
            # convenience: implied Q per node (may be NaN for k≈0)
            "Q_hat": (np.pi * self.f_hz) / (np.where(k_hat > 0, k_hat, np.nan) * self.v_kms)
        }
        return misfit, extras