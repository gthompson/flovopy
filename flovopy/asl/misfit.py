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

    def prepare(self, aslobj, seed_ids, dtype=np.float32):
        # Gather corrections and distances in station order
        Crows, Drows = [], []
        for sid in seed_ids:
            Crows.append(np.asarray(aslobj.amplitude_corrections[sid], dtype=dtype))
            Drows.append(np.asarray(aslobj.node_distances_km[sid],       dtype=dtype))
        C = np.vstack(Crows)                      # (S, N_all)
        D = np.vstack(Drows)                      # (S, N_all)

        # Subset to active nodes (mask-aware)
        idx = aslobj.gridobj.get_mask_indices()  # None or 1D int
        if idx is not None:
            C = C[:, idx]
            D = D[:, idx]
            node_index = idx
        else:
            node_index = np.arange(C.shape[1], dtype=int)

        # Precompute distance features (NaN-safe; will re-mask at evaluate)
        eps_d = np.finfo(C.dtype).tiny
        Dclamp = np.maximum(D, eps_d)
        x_log  = np.log(Dclamp)                   # (S, N)
        x_inv  = 1.0 / Dclamp                     # (S, N)
        x_lin  = D                                # (S, N)

        # Keep a nested StdOverMean for blending
        std_backend = StdOverMeanMisfit()
        std_ctx = std_backend.prepare(aslobj, seed_ids, dtype=dtype)

        return {
            "C": C, "D": D, "node_index": node_index,
            "x_log": x_log, "x_inv": x_inv, "x_lin": x_lin,
            "std_backend": std_backend, "std_ctx": std_ctx,
            "dtype": C.dtype,
        }

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
        """
        Vectorized per-node linear regression: y' = a + b * x
        where y' = y or log|y|, and x ∈ {log(D), 1/D, D} depending on options.
        Returns:
        misfit : (N_active,) float
        extras : {"N": counts, "mean": DR_proxy}
        """
        C      = ctx["C"]     # (S, N)
        D      = ctx["D"]     # (S, N)
        x_log  = ctx["x_log"]
        x_inv  = ctx["x_inv"]
        x_lin  = ctx["x_lin"]
        dtype  = ctx["dtype"]

        y = np.asarray(y, dtype=dtype).reshape(-1)        # (S,)
        S, N = C.shape

        # Choose target & feature
        # self.use_log (bool), self.square (bool): same semantics as your class
        eps_y = np.finfo(dtype).tiny
        if getattr(self, "use_log", True):
            y_t = np.log(np.maximum(np.abs(y), eps_y))    # (S,)
            x   = x_log                                   # (S, N)
        else:
            y_t = y
            x   = (x_inv if getattr(self, "square", False) else x_lin)

        # Valid mask per (station,node): finite y, finite C (use C to gate station availability), finite x
        fin_y  = np.isfinite(y_t)[:, None]               # (S, 1)
        fin_C  = np.isfinite(C)                          # (S, N)
        fin_x  = np.isfinite(x)                          # (S, N)
        M      = fin_y & fin_C & fin_x                   # (S, N)
        w      = M.astype(dtype)

        # Weighted sums (per node)
        n  = w.sum(axis=0)                                # (N,)
        sx = (w * x).sum(axis=0)                          # Σ x
        sy = (w * y_t[:, None]).sum(axis=0)               # Σ y
        sxx = (w * (x * x)).sum(axis=0)                   # Σ x^2
        sxy = (w * (x * y_t[:, None])).sum(axis=0)        # Σ x y
        # Means
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_x = sx / np.maximum(n, 1.0)
            mean_y = sy / np.maximum(n, 1.0)

        # Covariance/variance
        cov_xy = sxy - n * mean_x * mean_y               # Σ(xy) − n μx μy
        var_x  = sxx - n * mean_x * mean_x               # Σ(x^2) − n μx^2

        # Slope & intercept (guard tiny var_x)
        tiny = np.asarray(eps, dtype=dtype)
        b = np.where(var_x > tiny, cov_xy / var_x, 0.0)
        a = mean_y - b * mean_x

        # SSE, SST per node
        # yhat = a + b*x, computed lazily via expanded formula:
        # SSE = Σ w * (y - a - b x)^2
        yhat  = a[None, :] + b[None, :] * x              # (S, N)
        resid = np.where(M, (y_t[:, None] - yhat), 0.0)
        SSE   = (resid * resid).sum(axis=0)              # (N,)

        # SST = Σ w * (y - μy)^2
        dy   = np.where(M, (y_t[:, None] - mean_y[None, :]), 0.0)
        SST  = (dy * dy).sum(axis=0)

        # R^2 = 1 - SSE/SST  (guard zero SST)
        with np.errstate(invalid="ignore", divide="ignore"):
            R2 = 1.0 - (SSE / np.maximum(SST, tiny))

        # Convert to cost = 1 - R^2 and enforce min_stations
        cost_r2 = 1.0 - R2
        ok = n >= int(min_stations)
        cost_r2 = np.where(ok, cost_r2, np.inf)

        # Optional blend with StdOverMean
        alpha = float(getattr(self, "alpha", 1.0))
        if alpha < 1.0:
            som_cost, _extras = ctx["std_backend"].evaluate(
                y, ctx["std_ctx"], min_stations=min_stations, eps=eps
            )
            som_cost = np.asarray(som_cost, dtype=float)
            misfit = alpha * cost_r2 + (1.0 - alpha) * som_cost
        else:
            misfit = cost_r2

        # DR proxy from linear domain reductions at each node (median for robustness)
        Rlin = y.reshape(S, 1) * C                       # (S, N)
        DR   = np.nanmedian(np.where(np.isfinite(Rlin), Rlin, np.nan), axis=0)

        return np.asarray(misfit, float), {"N": n.astype(int), "mean": DR}




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


    Model (per node):
        t = log|y| ≈ β0 + β1 * log R + β2 * R, where physically β0 = A0, β1 = −N, β2 = −k.

     We solve all nodes at once via the normal equations (no loops). We never build a big (S,N,3) matrix; instead we assemble each node’s G = XᵀX and b = Xᵀt from scalar sums.
    """

    def __init__(self, f_hz=8.0, v_kms=1.5, alpha=1.0, clip_R_min=1e-3):
        self.f_hz = float(f_hz)
        self.v_kms = float(v_kms)
        self.alpha = float(alpha)
        self.clip_R_min = float(clip_R_min)

    def prepare(self, aslobj, seed_ids, dtype=np.float32):
        # C and D in station order
        Crows, Drows = [], []
        for sid in seed_ids:
            Crows.append(np.asarray(aslobj.amplitude_corrections[sid], dtype=dtype))
            Drows.append(np.asarray(aslobj.node_distances_km[sid],       dtype=dtype))
        C = np.vstack(Crows)  # (S, N)
        D = np.vstack(Drows)  # (S, N)

        idx = aslobj.gridobj.get_mask_indices()
        if idx is not None:
            C = C[:, idx]; D = D[:, idx]
            node_index = idx
        else:
            node_index = np.arange(C.shape[1], dtype=int)

        eps_d = np.finfo(dtype).tiny
        R  = np.maximum(D, eps_d)              # distances (km)
        L  = np.log(R)                         # log(R)
        R1 = R                                 # R

        # Blend backend
        std_backend = StdOverMeanMisfit()
        std_ctx     = std_backend.prepare(aslobj, seed_ids, dtype=dtype)

        return {
            "C": C, "R": R, "L": L, "node_index": node_index,
            "std_backend": std_backend, "std_ctx": std_ctx,
            "dtype": C.dtype
        }

    def evaluate(self, y, ctx, *, min_stations=3, eps=1e-9):
        """
        Vectorized multi-linear regression per node:
            t = log|y| ≈ β0 + β1 * log R + β2 * R
        Cost = 1 - R² (lower is better), optionally blended with StdOverMean.
        """
        C   = ctx["C"]        # (S, N)
        R   = ctx["R"]        # (S, N)
        L   = ctx["L"]        # (S, N)
        dt  = ctx["dtype"]

        y = np.asarray(y, dtype=dt).reshape(-1)          # (S,)
        S, N = C.shape

        eps_y = np.finfo(dt).tiny
        t = np.log(np.maximum(np.abs(y), eps_y))         # (S,)

        # Valid mask: finite target, finite C and finite features
        fin_t = np.isfinite(t)[:, None]                  # (S, 1)
        fin_C = np.isfinite(C)
        fin_L = np.isfinite(L)
        fin_R = np.isfinite(R)
        M     = fin_t & fin_C & fin_L & fin_R           # (S, N)
        w     = M.astype(dt)

        # --- Build normal equations from scalar sums (per node) ---
        # Columns: X = [1, L, R]
        n   = w.sum(axis=0)                               # Σ 1
        sL  = (w * L).sum(axis=0)                         # Σ L
        sR  = (w * R).sum(axis=0)                         # Σ R

        sLL = (w * (L * L)).sum(axis=0)                   # Σ L^2
        sRR = (w * (R * R)).sum(axis=0)                   # Σ R^2
        sLR = (w * (L * R)).sum(axis=0)                   # Σ L R

        tcol = t[:, None]
        sT  = (w * tcol).sum(axis=0)                      # Σ t
        sLT = (w * (L * tcol)).sum(axis=0)                # Σ L t
        sRT = (w * (R * tcol)).sum(axis=0)                # Σ R t

        # Assemble Gram matrix G (N, 3, 3) and RHS b (N, 3)
        G = np.empty((N, 3, 3), dtype=dt)
        G[:, 0, 0] = n;   G[:, 0, 1] = sL;  G[:, 0, 2] = sR
        G[:, 1, 0] = sL;  G[:, 1, 1] = sLL; G[:, 1, 2] = sLR
        G[:, 2, 0] = sR;  G[:, 2, 1] = sLR; G[:, 2, 2] = sRR

        b = np.empty((N, 3), dtype=dt)
        b[:, 0] = sT
        b[:, 1] = sLT
        b[:, 2] = sRT

        # Solve G β = b for each node (batched)
        # Guard singular systems: mark bad nodes for +inf cost
        # Solve G β = b for each node (batched)
        tiny = np.asarray(eps, dtype=dt)
        detG = (
            G[:,0,0]*(G[:,1,1]*G[:,2,2] - G[:,1,2]*G[:,2,1]) -
            G[:,0,1]*(G[:,1,0]*G[:,2,2] - G[:,1,2]*G[:,2,0]) +
            G[:,0,2]*(G[:,1,0]*G[:,2,1] - G[:,1,1]*G[:,2,0])
        )
        nonsing = np.isfinite(detG) & (np.abs(detG) > tiny)

        beta = np.full((N, 3), np.nan, dtype=dt)
        if nonsing.any():
            A = G[nonsing]                 # (K, 3, 3)
            B = b[nonsing][..., None]      # (K, 3, 1)
            sol = np.linalg.solve(A, B)    # (K, 3, 1)
            beta[nonsing] = sol[..., 0]    # (K, 3)

        # Predictions and R²
        # yhat = β0 + β1*L + β2*R
        yhat  = beta[:, 0][None, :] + beta[:, 1][None, :] * L + beta[:, 2][None, :] * R   # (S, N)
        resid = np.where(M, (t[:, None] - yhat), 0.0)
        SSE   = (resid * resid).sum(axis=0)

        mean_t = np.where(n > 0, sT / np.maximum(n, 1.0), np.nan)
        dy     = np.where(M, (t[:, None] - mean_t[None, :]), 0.0)
        SST    = (dy * dy).sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            R2 = 1.0 - (SSE / np.maximum(SST, tiny))

        # Base cost: 1 - R²; enforce min_stations and nonsingular system
        cost = 1.0 - R2
        ok = (n >= int(min_stations)) & nonsing
        cost = np.where(ok, cost, np.inf)

        # Optional blend with StdOverMean
        alpha = float(getattr(self, "alpha", 1.0))
        if alpha < 1.0:
            som_cost, _extras = ctx["std_backend"].evaluate(
                y, ctx["std_ctx"], min_stations=min_stations, eps=eps
            )
            som_cost = np.asarray(som_cost, dtype=float)
            cost = alpha * cost + (1.0 - alpha) * som_cost

        # DR proxy from linear reductions (robust)
        Rlin = y.reshape(S, 1) * C
        DR   = np.nanmedian(np.where(np.isfinite(Rlin), Rlin, np.nan), axis=0)

        return np.asarray(cost, float), {"N": n.astype(int), "mean": DR}
    
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
    topo_kw: Optional[dict] = None,   # <-- single source for topo_map kwargs
    outfile: Optional[str] = None,
    show: bool = True,
):
    """
    Compute the per-node misfit at the time of maximum DR and plot it atop a topo_map().
    - Works with 2D regular grids via grdimage or 1D "stream grids" via scatter.
    - Honors ASL grid masks (masked nodes show as NaN / not plotted).
    - Basemap is created *only* from passed topo_kw (no duplicate kwargs here).

    Returns
    -------
    (fig, misfit_2d_or_1d)
        fig: PyGMT Figure
        misfit array: 2D (nlat,nlon) for regular grids; 1D for stream grids
    """

    import pygmt
    import xarray as xr
    from datetime import datetime
    from flovopy.asl.map import topo_map

    ts = lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    topo_kw = dict(topo_kw or {})  # copy so we don't mutate caller's dict

    if aslobj.source is None:
        print(f"[{ts()}] [MISFIT] No source; run locate()/fast_locate() first.")
        return None

    # 1) Select time-of-peak DR
    t_idx = int(np.nanargmax(aslobj.source["DR"]))
    print(f"[{ts()}] [MISFIT] peak DR index = {t_idx}")

    # 2) Compute per-node misfit (local space if backend subsets nodes)
    misfit_vec, jstar_local, meta = compute_node_misfit_for_time(
        aslobj,
        t_idx,
        misfit_backend=backend,
        min_stations=min_stations,
        eps=eps,
        dtype=dtype,
    )
    misfit_vec = np.asarray(misfit_vec, float)
    print(f"[{ts()}] [MISFIT] misfit_vec shape={misfit_vec.shape} finite={np.isfinite(misfit_vec).sum()}")



    # 3) Handle grid/mask; build 2D field when possible
    grid = aslobj.gridobj
    grid_mask_idx = getattr(aslobj.gridobj, "_node_mask_idx", None)  # persistent mask on Grid

    # Mapping from local (backend) index → global grid index if provided
    node_index = meta.get("node_index") if isinstance(meta, dict) else None
    if node_index is not None:
        node_index = np.asarray(node_index, int)

    def _local_to_global(ix_local: int) -> int:
        if node_index is not None:
            return int(node_index[ix_local])
        if grid_mask_idx is not None:
            return int(np.asarray(grid_mask_idx, int)[ix_local])
        return int(ix_local)

    jstar_global = _local_to_global(jstar_local)

    # Regular 2D grid?
    has_2d = getattr(grid, "gridlat", None) is not None and np.ndim(grid.gridlat) == 2
    gridlat_flat = grid.gridlat.reshape(-1)
    gridlon_flat = grid.gridlon.reshape(-1)

    if has_2d:
        nlat, nlon = grid.gridlat.shape
        M_full = np.full(nlat * nlon, np.nan, dtype=float)

        if (node_index is None) and (grid_mask_idx is None):
            # misfit_vec already corresponds to full grid in order
            if misfit_vec.size == M_full.size:
                M_full[:] = misfit_vec
            else:
                # defensive fallback: place what we can at the start
                M_full[:misfit_vec.size] = misfit_vec
        else:
            # scatter from local space into the full field
            if node_index is not None:
                M_full[node_index] = misfit_vec
            else:  # use grid persistent mask indices
                M_full[np.asarray(grid_mask_idx, int)] = misfit_vec

        M2 = M_full.reshape(nlat, nlon)

        # Ensure increasing lat axis
        lat_axis = np.unique(grid.gridlat.ravel())
        lon_axis = np.unique(grid.gridlon.ravel())
        if lat_axis[0] > lat_axis[-1]:
            lat_axis = lat_axis[::-1]
            M2 = M2[::-1, :]

        misfit_da = xr.DataArray(M2, coords={"lat": lat_axis, "lon": lon_axis}, dims=("lat", "lon"))
        M_plot = M2
    else:
        # 1D node set ("streams" style) → scatter plot
        if (node_index is None) and (grid_mask_idx is None):
            M_plot = misfit_vec
        else:
            M_full = np.full_like(gridlat_flat, np.nan, dtype=float)
            if node_index is not None:
                M_full[node_index] = misfit_vec
            else:
                M_full[np.asarray(grid_mask_idx, int)] = misfit_vec
            M_plot = M_full
        misfit_da = None  # use scatter

    # 4) Create basemap strictly via topo_kw (caller controls everything there)
    #    If caller didn't supply a title, provide a friendly default.
    topo_kw.setdefault("title", "Misfit heatmap (peak DR time)")
    topo_kw.setdefault("show", False)

    # Optionally center on the peak location if not overridden by topo_kw
    if "centerlat" not in topo_kw or "centerlon" not in topo_kw:
        topo_kw.setdefault("centerlat", float(aslobj.source["lat"][t_idx]))
        topo_kw.setdefault("centerlon", float(aslobj.source["lon"][t_idx]))

    fig = topo_map(**topo_kw)

    # 5) Color scale from data range
    finite = np.isfinite(M_plot)
    if finite.any():
        vmin = float(np.nanmin(M_plot[finite]))
        vmax = float(np.nanmax(M_plot[finite]))
        if vmin == vmax:
            vmin, vmax = (0.0, 1.0)
    else:
        vmin, vmax = (0.0, 1.0)

    pygmt.makecpt(cmap=cmap, series=[vmin, vmax], continuous=True)

    # 6) Draw heatmap or scatter
    if misfit_da is not None:
        fig.grdimage(grid=misfit_da, cmap=True, transparency=transparency)
    else:
        # 1D scatter
        lats = gridlat_flat
        lons = gridlon_flat
        tbl = np.column_stack([lons[finite], lats[finite], M_plot[finite]])
        if tbl.size:
            fig.plot(data=tbl, style="c0.15c", cmap=True, transparency=transparency)

    fig.colorbar(frame='+l"Misfit"')

    # 7) Mark best node
    best_lon = float(gridlon_flat[jstar_global])
    best_lat = float(gridlat_flat[jstar_global])

    try:
        region = topo_kw.get("region", None)
        if region is not None:
            xmin, xmax, ymin, ymax = region
            if not (xmin <= best_lon <= xmax and ymin <= best_lat <= ymax):
                print(f"[MISFIT] Best node ({best_lat:.5f},{best_lon:.5f}) is outside region {region}")
    except Exception:
        pass

    fig.plot(x=[best_lon], y=[best_lat], style="c0.26c", fill="red", pen="1p,red")

    # 8) Output
    if outfile:
        fig.savefig(outfile)
        print(f"[{ts()}] [MISFIT] saved: {outfile}")
    elif show:
        fig.show()

    return fig, M_plot