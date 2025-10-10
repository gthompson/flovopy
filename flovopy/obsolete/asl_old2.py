from __future__ import annotations

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pygmt
import pandas as pd
from scipy.ndimage import uniform_filter1d, gaussian_filter

from obspy import Stream, UTCDateTime
from obspy.core.event import (
    Event, Catalog, ResourceIdentifier, Origin, Amplitude, QuantityError,
    OriginQuality, Comment,
)
from obspy.geodetics import gps2dist_azimuth

from flovopy.processing.sam import VSAM
from flovopy.asl.ampcorr import AmpCorr, AmpCorrParams
from flovopy.asl.station_corrections import apply_interval_station_gains
from flovopy.utils.make_hash import make_hash
from flovopy.asl.map import topo_map
from flovopy.asl.distances import compute_distances, distances_signature, geo_distance_3d_km
from flovopy.asl.grid import Grid, NodeGrid
from flovopy.asl.misfit import plot_misfit_heatmap_for_peak_DR, StdOverMeanMisfit, R2DistanceMisfit
from flovopy.asl.utils import (
    compute_azimuthal_gap,
    compute_spatial_connectedness,
    _as_regular_view,
    _grid_mask_indices,
    _movavg_1d,
    _median_filter_indices,
    _grid_shape_or_none,
    _viterbi_smooth_indices,
)

# ---------- ASL ----------
class ASL:
    """
    Amplitude-based Source Location (ASL) for volcano-seismic data.

    Expects:
      - VSAM object (already computed outside)
      - metric name
      - grid object
      - precomputed node_distances_km, station_coordinates, amplitude_corrections (injected by caller)
    """

    def __init__(self, samobject: VSAM, metric: str, gridobj: Grid | NodeGrid, window_seconds: float, id: Optional[str] = None):    
        if not isinstance(samobject, VSAM):
            raise TypeError("samobject must be a VSAM instance")
        self.samobject = samobject
        self.metric = metric
        self.gridobj = gridobj
        self.window_seconds = int(window_seconds)

        # injected later by caller (precomputed)
        self.node_distances_km: dict[str, np.ndarray] = {}
        self.station_coordinates: dict[str, dict] = {}
        self.amplitude_corrections: dict[str, np.ndarray] = {}

        # params (for provenance in filenames)
        self.surfaceWaves = False
        self.wavespeed_kms = None
        self.Q = None
        self.peakf = None
        self.wavelength_km = None

        self.located = False
        self.source = None
        self.event = None

        # If the Grid carries a node mask, normalize to GLOBAL indices
        try:
            self._node_mask = _grid_mask_indices(self.gridobj)
        except Exception:
            self._node_mask = None

        self.id = self.set_id()

    def set_id(self):
        # don't hash large arrays; use their ids/keys
        keys = tuple(sorted(self.amplitude_corrections.keys()))
        return make_hash(self.metric, self.window_seconds, self.gridobj.id, keys)
    
    def compute_grid_distances(self, *, inventory=None, stream=None, verbose=True):
        """
        Convenience wrapper around flovopy.asl.distances.compute_distances.

        Fills:
          - self.node_distances_km : dict[seed_id -> np.ndarray]
          - self.station_coordinates : dict[seed_id -> {latitude, longitude, elevation}]
        """
        if verbose: print("[ASL] Computing node→station distances…")
        node_distances_km, station_coords = compute_distances(
            self.gridobj, inventory=inventory, stream=stream
        )
        self.node_distances_km = node_distances_km
        self.station_coordinates = station_coords
        if verbose:
            nchan = len(node_distances_km)
            nnodes = self.gridobj.gridlat.size
            print(f"[ASL] Distances ready for {nchan} channels over {nnodes} nodes.")
        return node_distances_km, station_coords

    def compute_amplitude_corrections(
        self,
        *,
        surface_waves=True,
        wavespeed_kms: float = 2.5,
        Q: float | None = 23,
        peakf: float = 8.0,
        dtype: str = "float32",
        require_all: bool = False,
        force_recompute: bool = True,   # kept for API parity; here it's always in-memory
        verbose: bool = True,
    ):
        """
        Convenience wrapper to build amplitude corrections in memory and attach to self.

        Uses the currently stored:
          - self.node_distances_km
          - self.gridobj

        Fills:
          - self.amplitude_corrections : dict[seed_id -> np.ndarray]
          - self._ampcorr              : AmpCorr (for metadata; optional)
          - self.Q, self.peakf, self.wavespeed_kms, self.surfaceWaves
        """
        if not self.node_distances_km:
            raise RuntimeError("node_distances_km is empty; call compute_grid_distances() first.")

        if verbose:
            print(f"[ASL] Computing amplitude corrections "
                  f"(surface_waves={surface_waves}, v={wavespeed_kms} km/s, Q={Q}, peakf={peakf} Hz)…")

        params = AmpCorrParams(
            surface_waves=surface_waves,
            wave_speed_kms=wavespeed_kms,
            Q=Q,
            peakf=float(peakf),
            grid_sig=self.gridobj.signature().as_tuple(),
            inv_sig=tuple(sorted(self.node_distances_km.keys())),
            dist_sig=distances_signature(self.node_distances_km),
            mask_sig=None,
            code_version="v1",
        )
        ampcorr = AmpCorr(params)  # in-memory; cache_dir irrelevant here
        ampcorr.compute(self.node_distances_km, inventory=None, dtype=dtype, require_all=require_all)
        ampcorr.validate_against_nodes(self.gridobj.gridlat.size)

        # attach
        self.amplitude_corrections = ampcorr.corrections
        self._ampcorr = ampcorr

        # provenance for filenames/plots
        self.Q = params.Q
        self.peakf = params.peakf
        self.wavespeed_kms = params.wave_speed_kms
        self.surfaceWaves = params.surface_waves

        if verbose:
            print(f"[ASL] Amplitude corrections ready for {len(self.amplitude_corrections)} channels.")
        return self.amplitude_corrections    

    # ---------- data prep ----------
    def metric2stream(self) -> Stream:
        st = self.samobject.to_stream(metric=self.metric)

        # Determine SAM dt
        dt = float(getattr(self.samobject, "sampling_interval", 1.0))
        if not np.isfinite(dt) or dt <= 0:
            fs = float(getattr(st[0].stats, "sampling_rate", 1.0) or 1.0)
            dt = 1.0 / fs

        win_sec = float(self.window_seconds or 0.0)
        if win_sec > dt:
            for tr in st:
                fs_tr = float(getattr(tr.stats, "sampling_rate", 1.0) or 1.0)
                dt_tr = 1.0 / fs_tr
                w_tr = max(2, int(round(win_sec / dt_tr)))

                x = tr.data.astype(float)
                m = np.isfinite(x).astype(float)
                xf = np.where(np.isfinite(x), x, 0.0)

                num = uniform_filter1d(xf, size=w_tr, mode="nearest")
                den = uniform_filter1d(m,  size=w_tr, mode="nearest")
                tr.data = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                tr.stats.sampling_rate = fs_tr
        return st

    # ---------- locator ----------
    def locate(
        self,
        *,
        min_stations: int = 3,
        eps: float = 1e-9,
        use_median_for_DR: bool = False,
        spatial_smooth_sigma: float = 0.0,   # Gaussian sigma in node units; 0 disables
        temporal_smooth_win: int = 0,        # centered odd window; 0 disables
        verbose: bool = True,
    ):
        """
        Slow (loop-based) locator that mirrors fast_locate() outputs/behavior.

        - Scans each time sample and every (possibly masked) grid node with Python loops.
        - Misfit per node: std(reduced) / (|mean(reduced)| + eps), reduced = y * C[:, j].
        - DR(t): median(reduced) if use_median_for_DR else mean(reduced) at the best node.
        - Optional spatial smoothing: Gaussian filter applied to the per-time misfit surface
        with NaN/invalid-aware normalization (same logic as fast_locate).
        - Optional temporal smoothing: moving average on (lat, lon, DR) tracks.
        - Computes azimuthal gap and a spatial connectedness score (same as fast_locate).
        - Records chosen GLOBAL node indices in source["node_index"] (to match fast_locate).
        """
        if verbose:
            print("[ASL] locate (slow): preparing data…")

        # Grid vectors (global space)
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)
        nnodes  = gridlat.size

        # Optional global→local masking (keep only allowed nodes)
        node_index = _grid_mask_indices(self.gridobj)  # GLOBAL indices or None
        if node_index is not None and node_index.size > 0:
            gridlat_local = gridlat[node_index]
            gridlon_local = gridlon[node_index]
            nnodes_local  = int(node_index.size)
        else:
            node_index    = None
            gridlat_local = gridlat
            gridlon_local = gridlon
            nnodes_local  = nnodes

        # Stream → Y (nsta, ntime)
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        Y = np.vstack([tr.data.astype(np.float32, copy=False) for tr in st])
        t = st[0].times("utcdatetime")
        nsta, ntime = Y.shape

        # Corrections → C (nsta, nnodes_local) from amplitude_corrections (global→local if masked)
        C = np.empty((nsta, nnodes_local), dtype=np.float32)
        for k, sid in enumerate(seed_ids):
            if sid not in self.amplitude_corrections:
                raise KeyError(f"Missing amplitude corrections for channel '{sid}'")
            vec = np.asarray(self.amplitude_corrections[sid], dtype=np.float32)
            if vec.size != nnodes:
                raise ValueError(f"Corrections length mismatch for {sid}: {vec.size} != {nnodes}")
            C[k, :] = vec[node_index] if node_index is not None else vec

        # Station coords (for azgap)
        station_coords = []
        for sid in seed_ids:
            coords = self.station_coordinates.get(sid)
            if coords:
                station_coords.append((coords["latitude"], coords["longitude"]))

        # Outputs
        source_DR     = np.empty(ntime, dtype=float)
        source_lat    = np.empty(ntime, dtype=float)
        source_lon    = np.empty(ntime, dtype=float)
        source_misfit = np.empty(ntime, dtype=float)
        source_azgap  = np.empty(ntime, dtype=float)
        source_nsta   = np.empty(ntime, dtype=int)
        source_node   = np.empty(ntime, dtype=int)  # GLOBAL indices (to match fast_locate)

        # Optional spatial blur setup (NaN/invalid-aware, like fast_locate)
        try:
            from scipy.ndimage import gaussian_filter as _gauss
        except Exception:
            _gauss = None

        # If grid has a regular shape, we can reshape for blurring
        def _grid_shape_or_none(grid):
            try:
                return int(grid.nlat), int(grid.nlon)
            except Exception:
                return None

        grid_shape = _grid_shape_or_none(self.gridobj)
        # If we applied a mask, the local sub-grid generally won't be reshapeable → skip blur gracefully
        can_blur = (
            spatial_smooth_sigma and spatial_smooth_sigma > 0.0 and
            (grid_shape is not None) and (_gauss is not None) and
            (nnodes_local == (grid_shape[0] * grid_shape[1]))
        )

        if verbose:
            print(f"[ASL] locate (slow): nsta={nsta}, ntime={ntime}, "
                f"spatial_blur={'on' if can_blur else 'off'}, "
                f"temporal_smooth_win={temporal_smooth_win or 0}")

        # Time loop (slow path)
        for i in range(ntime):
            y = Y[:, i]

            misfit_j = np.full(nnodes_local, np.inf, dtype=float)
            dr_j     = np.full(nnodes_local, np.nan, dtype=float)
            nused_j  = np.zeros(nnodes_local, dtype=int)

            # Node loop (local indices)
            for j in range(nnodes_local):
                reduced = y * C[:, j]              # (nsta,)
                finite  = np.isfinite(reduced)
                nfin    = int(finite.sum())
                nused_j[j] = nfin
                if nfin < min_stations:
                    continue

                r = reduced[finite]
                if r.size == 0:
                    continue

                DRj = float(np.median(r)) if use_median_for_DR else float(np.mean(r))
                dr_j[j] = DRj

                mu = float(np.mean(r))
                sg = float(np.std(r, ddof=0))
                if not np.isfinite(mu) or abs(mu) < eps:
                    continue
                misfit_j[j] = sg / (abs(mu) + eps)

            # Optional spatial smoothing on the misfit surface (NaN/invalid-aware)
            if can_blur:
                nlat, nlon = grid_shape
                try:
                    m2 = misfit_j.reshape(nlat, nlon)

                    # Validity mask: finite = True
                    valid = np.isfinite(m2)

                    if valid.any():
                        # Replace invalid with a large finite number so the minimum never migrates into invalid areas
                        LARGE = np.nanmax(m2[valid]) if np.isfinite(m2[valid]).any() else 1.0
                        LARGE = float(LARGE) if np.isfinite(LARGE) else 1.0
                        F = np.where(valid, m2, LARGE)

                        # Blur the finite costs and the mask separately, then renormalize
                        W = _gauss(valid.astype(float), sigma=spatial_smooth_sigma, mode="nearest")
                        G = _gauss(F,                sigma=spatial_smooth_sigma, mode="nearest")

                        with np.errstate(invalid="ignore", divide="ignore"):
                            Mblur = np.where(W > 1e-6, G / np.maximum(W, 1e-6), LARGE)

                        # Re-impose invalid cells as LARGE so they can’t be selected
                        Mblur = np.where(valid, Mblur, LARGE)
                        misfit_for_pick = Mblur.reshape(-1)
                    else:
                        misfit_for_pick = misfit_j
                except Exception:
                    # If reshape fails (e.g., masked sub-grid), fall back to unblurred costs
                    misfit_for_pick = misfit_j
            else:
                misfit_for_pick = misfit_j

            # Choose best node (local → global)
            jstar_local  = int(np.nanargmin(misfit_for_pick))
            jstar_global = int(node_index[jstar_local]) if node_index is not None else jstar_local

            # Record outputs at time i
            source_lat[i]    = gridlat[jstar_global]
            source_lon[i]    = gridlon[jstar_global]
            source_misfit[i] = float(misfit_j[jstar_local])
            source_DR[i]     = float(dr_j[jstar_local])
            source_nsta[i]   = int(nused_j[jstar_local])
            source_node[i]   = jstar_global

            azgap, _ = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
            source_azgap[i] = azgap

        # Optional temporal smoothing on tracks (centered moving average)
        def _movavg_1d(x, w):
            x = np.asarray(x, float)
            if w < 3 or w % 2 == 0:
                return x
            k = w // 2
            pad = np.pad(x, (k, k), mode="edge")
            ker = np.ones(w, dtype=float) / float(w)
            y = np.convolve(pad, ker, mode="valid")
            return y

        if temporal_smooth_win and temporal_smooth_win >= 3 and temporal_smooth_win % 2 == 1:
            source_lat = _movavg_1d(source_lat, temporal_smooth_win)
            source_lon = _movavg_1d(source_lon, temporal_smooth_win)
            source_DR  = _movavg_1d(source_DR,  temporal_smooth_win)

        # Package source (keep DR scale factor for continuity with fast_locate)
        self.source = {
            "t": t,
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
            "node_index": source_node,   # NEW: mirror fast_locate output
        }

        # Spatial connectedness (same as fast_locate)
        conn = compute_spatial_connectedness(
            self.source["lat"], self.source["lon"], dr=source_DR,
            top_frac=0.15, min_points=12, max_points=200,
        )
        self.connectedness = conn
        if verbose:
            print(f"[ASL] connectedness: score={conn['score']:.3f}  "
                f"n_used={conn['n_used']}  mean_km={conn['mean_km']:.2f}  p90_km={conn['p90_km']:.2f}")

        # Expose scalar in source for easy tabulation
        self.source["connectedness"] = conn["score"]

        # ObsPy event packaging
        self.source_to_obspyevent()
        self.located = True
        if hasattr(self, "set_id"):
            self.id = self.set_id()
        if verbose:
            print("[ASL] locate (slow): done.")


    def fast_locate(
        self,
        *,
        misfit_backend=None,
        min_stations: int = 3,
        eps: float = 1e-9,
        use_median_for_DR: bool = False,
        batch_size: int = 1024,
        spatial_smooth_sigma: float = 0.0,   # Gaussian sigma in node units (0 disables)
        temporal_smooth_mode: str = "none",  # "none" | "median" | "viterbi"
        temporal_smooth_win: int = 5,        # window (median) or horizon (viterbi)
        viterbi_lambda_km: float = 5.0,      # km cost per step in viterbi
        viterbi_max_step_km: float = 25.0,   # km hard cutoff for viterbi steps
        verbose: bool = True,
    ):
        """
        Locate seismic sources by minimizing a misfit function on a spatial grid.

        Cheat-sheet (common options)
        ----------------------------
        misfit_backend        Which misfit engine to use
                            - StdOverMeanMisfit (default, std/|mean|)
                            - R2DistanceMisfit (amplitude vs. distance correlation)
        min_stations          Require this many usable stations (default=3)
        use_median_for_DR     DR at best node = median (True) or mean (False)
        batch_size            Process this many time samples per loop (default=1024)

        spatial_smooth_sigma  >0 → Gaussian blur misfit maps (stabilizes per-frame noise)
                            Units = grid cells; e.g. 1.5 ≈ smooth over ~1.5 nodes
        temporal_smooth_mode  Temporal smoothing strategy
                            - "none"    : raw best nodes
                            - "median"  : median filter over window
                            - "viterbi" : dynamic-programming optimal path
        temporal_smooth_win   Window length (odd int) for "median" or cost horizon for "viterbi"
        viterbi_lambda_km     Step penalty (cost per km moved per frame) in "viterbi" mode
        viterbi_max_step_km   Hard cutoff (disallow jumps > this distance) in "viterbi" mode

        verbose               Print progress + diagnostics (default=True)

        What it does (high level)
        -------------------------
        1) Converts the selected VSAM metric to a station×time matrix Y.
        2) For each time step, evaluates a node-wise misfit across the grid using
        the chosen backend and the precomputed amplitude corrections C (and, for
        certain backends, station→node distances D).
        3) Picks the best node per time; computes DR, azgap, etc.
        4) Optionally smooths the per-time misfit spatially (Gaussian blur), and/or
        smooths the resulting node track temporally (median / Viterbi).
        5) Stores a source track in `self.source` and a scalar connectedness metric.

        Returns (via object state)
        --------------------------
        self.source : dict of arrays/lists
        - "t"          : UTCDateTime array (length T)
        - "lat","lon"  : per-time best node coordinates (length T)
        - "DR"         : reduced displacement (scaled by 1e7)
        - "misfit"     : misfit at chosen node
        - "azgap"      : azimuthal gap at chosen node
        - "nsta"       : usable station count
        - "node_index" : chosen node index (GLOBAL index on the full grid)
        - "connectedness": scalar [0,1], spatial compactness of track (top-DR subset)
        self.connectedness : full diagnostic dict for the connectedness computation
        self.located       : set True
        self.id            : optionally set via self.set_id() if present

        -------------------------------------------------------------------------
        Misfit backends (how the “goodness” per node is computed)
        -------------------------------------------------------------------------

        1) StdOverMeanMisfit  (default; robust, fast, no distances required)
        ---------------------------------------------------------------
        Definition:
            Let R[:, j] = y * C[:, j] be the reduced amplitudes at node j.
            misfit_j = std(R[:, j]) / (|mean(R[:, j])| + eps).

        Intuition:
            If a node is correct, station reductions should agree up to scatter
            → small std and non-zero mean ⇒ small ratio.

        Pros:
            - Very stable when amplitudes are well corrected by C (geometry + Q).
            - No explicit distance model; needs only C.
            - Vectorized via dot-products; very fast.

        Cons:
            - Does not directly leverage a hypothesized decay law vs distance.

        Options:
            - None specific; use overall options like spatial/temporal smoothing.

        When to use:
            - As a baseline for most data (VT/RSAM/DSAM-style metrics).
            - When distance corrections are already baked into C reasonably well.

        2) R2DistanceMisfit  (distance-aware, correlation-style)
        -----------------------------------------------------
        Prereqs:
            - Requires node distances per station: self.node_distances_km[sid] (nnodes,)

        Definition (per node j):
            Build a distance feature x_j from station→node distances d_ij, then
            correlate with station amplitudes y (or log|y|), compute R², and
            define cost as 1 − R² (lower is better). You can also blend this
            with the StdOverMean cost via α:

            cost_j = α * (1 − R²_j) + (1 − α) * (std/|mean|)_j

        Feature choices:
            - use_log=True  → x = log(d) and y' = log|y| (power-law proxy)
            - square=True   → if use_log=False, x = 1/d (body-wave-like proxy),
                            else x = d (surface-wave-like proxy)
            - alpha ∈ [0,1] → blend with StdOverMean (α=1 pure R², α=0 pure StdOverMean)

        Intuition:
            If amplitudes decay with distance in a consistent way, a correct node
            should maximize correlation between the distance feature and y (or log|y|).

        Pros:
            - Encodes distance physics directly; can be more discriminative for tremor.
        Cons:
            - Sensitive to amplitude sign, zero/near-zero values (we use |y| and clip for log).
            - Needs distances; may be less stable if the decay law is weak/inverted.

        When to use:
            - Continuous tremor, flows, or when you expect a clear distance decay.

        Recommended starting points:
            - Pure distance:         R2DistanceMisfit(use_log=True,  alpha=1.0)
            - Blended, more robust:  R2DistanceMisfit(use_log=True,  alpha=0.5)
            - Body-wave flavor:      R2DistanceMisfit(use_log=False, square=True,  alpha=0.5)
            - Surface-wave flavor:   R2DistanceMisfit(use_log=False, square=False, alpha=0.5)

        Notes on amplitude corrections (C):
        - Both backends use the same C matrix; R² backend also uses distances.
        - If some channels lack corrections at certain nodes (NaN in C), those
            stations are ignored for that node (respecting min_stations).

        -------------------------------------------------------------------------
        Smoothing options (why/when/how)
        -------------------------------------------------------------------------

        A) Spatial smoothing (per-frame Gaussian blur)
        -------------------------------------------
        What:
            Before selecting the best node at each time frame, the node-wise misfit
            map can be blurred with a Gaussian of sigma=spatial_smooth_sigma (in
            *grid cells*). This suppresses pixel-level noise and encourages the
            minimum to sit in a basin rather than a single spiky node.

        Units:
            Sigma is in node units. If grid spacing is ~400 m and sigma=1.5,
            you smooth over roughly ~600 m scale. Convert as:
                sigma_km ≈ sigma_nodes * node_spacing_km.

        Edge behavior & sub-grids:
            We use "nearest" mode at edges to avoid shrinking the domain. If the
            backend prepared a sub-grid (refine pass), we blur in that sub-grid
            shape. If shapes don’t match or blur is unavailable, we silently skip.

        When to use:
            - High-noise data, sparse stations, or coarse grids where you want
            stability without committing to temporal smoothing.

        Pitfalls:
            - Too large sigma can bias the minimum toward broad basins and smear
            distinct sources together.

        B) Temporal smoothing (post-selection track filtering)
        ---------------------------------------------------
        1) "median" (robust local smoothing)
            - Apply a centered, odd-length median filter to the chosen lat/lon/DR.
            - Great for knocking down isolated jumps/outliers; preserves overall path.
            - Choose temporal_smooth_win (3, 5, 7…). Larger → smoother but laggier.

        2) "viterbi" (optimal path under movement penalty)
            - Model a time series of node choices as a path through the grid,
                with per-frame “emission” cost = misfit at that node, plus a
                “transition” cost proportional to the distance moved between
                consecutive frames:
                    total_cost = Σ_t [ misfit(node_t) + λ * dist_km(node_{t-1}, node_t) ]
            - λ is viterbi_lambda_km (km cost per step). Bigger λ → smoother paths
                (less movement). Smaller λ → tracks hug the raw misfit minima.
            - viterbi_max_step_km imposes a hard cap; jumps larger than this are disallowed.
            - temporal_smooth_win acts as a horizon / band-limit in some implementations;
                we use it to keep neighbor search tractable if you customize further.

        When “viterbi” wins:
            - Tremor/flows where a physically continuous source should move gradually,
            and you want the best single, globally consistent track.

        Pitfalls:
            - If λ is too small and the misfit surface is noisy, you get twitchy tracks.
            - If λ is too big, the track may lag or over-smooth real motion.
            - Hard caps that are too tight can force suboptimal paths when the grid is sparse.

        -------------------------------------------------------------------------
        Follow-on refinement: refine_and_relocate()
        -------------------------------------------------------------------------
        `refine_and_relocate(top_frac=0.25, margin_km=1.0, …)` subsets the current grid
        to a tight bounding box around the *top* fraction of DR locations and re-runs
        `fast_locate` on that sub-grid. Because we keep the same amplitude corrections
        (just masked), no recomputation of C is needed. This often sharpens source
        tracks and reduces edge effects. You can combine refinement with the same
        spatial/temporal smoothing options used here.

        Example workflow:
        >>> asl.fast_locate(spatial_smooth_sigma=1.5, temporal_smooth_mode="median", temporal_smooth_win=5)
        >>> asl.refine_and_relocate(top_frac=0.25, margin_km=1.0,
        ...                         spatial_smooth_sigma=1.0,
        ...                         temporal_smooth_mode="viterbi",
        ...                         temporal_smooth_win=7, viterbi_lambda_km=8.0)

        -------------------------------------------------------------------------
        Practical tips
        -------------------------------------------------------------------------
        - Start simple: default backend, no smoothing. If jittery, add spatial blur
        (sigma ~1–2 nodes). If still jittery (tremor), switch temporal smoothing to
        "median" (win=5) or "viterbi" (λ~5–10 km, cap 20–30 km).
        - R² backend: begin with use_log=True, alpha=0.5. If distances dominate
        (clear decay), increase alpha → 0.8–1.0. If unstable, reduce alpha.
        - The connectedness metric in `self.connectedness` helps sanity-check how
        spatially coherent the track is (higher ≈ tighter cluster).
        - `batch_size` mainly impacts speed; accuracy is unchanged.

        Implementation notes
        --------------------
        - Backends must implement:
            prepare(aslobj, seed_ids, dtype) -> ctx
            evaluate(y, ctx, min_stations, eps) -> (misfit, extras)
        where `extras` may include per-node 'mean' (so we can extract DR) and 'N'.
        - Spatial blur uses `scipy.ndimage.gaussian_filter` if available; otherwise
        it is silently disabled.
        - Temporal "median" uses a centered odd window; ends are handled by reflection.
        - "Viterbi" requires a step-distance function (we compute great-circle km
        between nodes) and a DP pass to recover the minimum-cost path.

        Examples
        --------
        # 1) Default backend, no smoothing
        asl.fast_locate()

        # 2) R² backend (pure), log correlation
        from flovopy.asl.misfit import R2DistanceMisfit
        asl.fast_locate(misfit_backend=R2DistanceMisfit(use_log=True, alpha=1.0))

        # 3) Spatial + median temporal smoothing
        asl.fast_locate(spatial_smooth_sigma=1.5,
                        temporal_smooth_mode="median", temporal_smooth_win=5)

        # 4) Viterbi smoothing with moderate movement penalty
        asl.fast_locate(temporal_smooth_mode="viterbi", temporal_smooth_win=7,
                        viterbi_lambda_km=8.0, viterbi_max_step_km=30.0)
        """
        if misfit_backend is None:
            misfit_backend = StdOverMeanMisfit()

        if verbose:
            print("[ASL] fast_locate: preparing data…")

        # Grid vectors
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)

        # Stream → Y
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        Y = np.vstack([tr.data.astype(np.float32, copy=False) for tr in st])
        t = st[0].times("utcdatetime")
        nsta, ntime = Y.shape

        # Refresh any grid-provided node mask each call
        self._node_mask = _grid_mask_indices(self.gridobj)

        # Backend context (will honor self._node_mask if present)
        ctx = misfit_backend.prepare(self, seed_ids, dtype=np.float32)

        # Station coords (for azgap)
        station_coords = []
        for sid in seed_ids:
            c = self.station_coordinates.get(sid)
            if c:
                station_coords.append((c["latitude"], c["longitude"]))

        # Outputs
        source_DR     = np.empty(ntime, dtype=float)
        source_lat    = np.empty(ntime, dtype=float)
        source_lon    = np.empty(ntime, dtype=float)
        source_misfit = np.empty(ntime, dtype=float)
        source_azgap  = np.empty(ntime, dtype=float)
        source_nsta   = np.empty(ntime, dtype=int)
        source_node   = np.empty(ntime, dtype=int)

        # Spatial blur feasibility (full-grid reshape only)
        grid_shape = _grid_shape_or_none(self.gridobj)
        can_blur = (
            spatial_smooth_sigma and spatial_smooth_sigma > 0.0 and
            (grid_shape is not None) and (gaussian_filter is not None)
        )

        if verbose:
            print(f"[ASL] fast_locate: nsta={nsta}, ntime={ntime}, batch={batch_size}, "
                f"spatial_blur={'on' if can_blur else 'off'}, "
                f"temporal_smooth_win={temporal_smooth_win or 0}")

        # Viterbi collections (filled only if requested)
        want_viterbi = (temporal_smooth_mode or "none").lower() == "viterbi"
        misfit_hist = [] if want_viterbi else None
        mean_hist   = [] if want_viterbi else None

        for i0 in range(0, ntime, batch_size):
            i1 = min(i0 + batch_size, ntime)
            if verbose:
                print(f"[ASL] fast_locate: [{i0}:{i1})")
            Yb = Y[:, i0:i1]

            for k in range(Yb.shape[1]):
                y = Yb[:, k]
                misfit, extras = misfit_backend.evaluate(y, ctx, min_stations=min_stations, eps=eps)

                # Keep full misfit & mean vectors if we’ll need Viterbi later
                if want_viterbi:
                    misfit_hist.append(np.asarray(misfit, float).copy())
                    mvec = extras.get("mean", extras.get("DR", None))
                    if mvec is None:
                        mvec = np.full_like(misfit, np.nan, dtype=float)
                    mean_hist.append(np.asarray(mvec, float).copy())

                # Optional per-time spatial smoothing (NaN/invalid-aware)
                if can_blur:
                    nlat, nlon = grid_shape
                    try:
                        m2 = misfit.reshape(nlat, nlon)

                        # Valid mask and large-fill
                        valid = np.isfinite(m2)
                        if valid.any():
                            LARGE = np.nanmax(m2[valid]) if np.isfinite(m2[valid]).any() else 1.0
                            LARGE = float(LARGE) if np.isfinite(LARGE) else 1.0
                            F = np.where(valid, m2, LARGE)

                            # Blur mask and values, then renormalize
                            W = gaussian_filter(valid.astype(float), sigma=spatial_smooth_sigma, mode="nearest")
                            G = gaussian_filter(F,                sigma=spatial_smooth_sigma, mode="nearest")
                            with np.errstate(invalid="ignore", divide="ignore"):
                                Mblur = np.where(W > 1e-6, G / np.maximum(W, 1e-6), LARGE)
                            Mblur = np.where(valid, Mblur, LARGE)
                            misfit = Mblur.reshape(-1)
                        # else: all invalid, leave as-is
                    except Exception:
                        # Sub-grid / reshape mismatch → skip blur silently
                        pass

                # Pick best node (map local→global if backend subsetted nodes)
                jstar_local = int(np.nanargmin(misfit))
                jstar_global = jstar_local
                node_index = ctx.get("node_index", None)
                if node_index is not None:
                    jstar_global = int(node_index[jstar_local])

                # DR at best node
                DR_t = extras.get("mean", extras.get("DR", None))
                if DR_t is not None and np.ndim(DR_t) == 1:
                    DR_t = DR_t[jstar_local]
                else:
                    # Fallback: use reductions at best node if C was provided by backend
                    C = ctx.get("C", None)
                    if C is not None and C.shape[0] == y.shape[0]:
                        rbest = y * C[:, jstar_local]
                        rbest = rbest[np.isfinite(rbest)]
                        DR_t = (np.median(rbest) if use_median_for_DR else np.mean(rbest)) if rbest.size else np.nan
                    else:
                        DR_t = np.nan

                i = i0 + k
                source_DR[i]     = DR_t
                source_lat[i]    = gridlat[jstar_global]
                source_lon[i]    = gridlon[jstar_global]
                source_misfit[i] = float(misfit[jstar_local])
                source_node[i]   = jstar_global

                # Station count (vector N or scalar); fallback to finite(y)
                N = extras.get("N", None)
                source_nsta[i] = int(N if np.isscalar(N) else (N[jstar_local] if N is not None else np.sum(np.isfinite(y))))

                azgap, _ = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
                source_azgap[i] = azgap

            # ---- temporal smoothing after each batch (works on the full arrays so far) ----
            flat_lat = self.gridobj.gridlat.ravel()
            flat_lon = self.gridobj.gridlon.ravel()
            mode = (temporal_smooth_mode or "none").lower()

            if mode == "median" and (temporal_smooth_win and temporal_smooth_win >= 3 and temporal_smooth_win % 2 == 1):
                sm_node = _median_filter_indices(source_node, temporal_smooth_win)

                # Optional small-jump guard
                try:
                    def _km(j1, j2):
                        return 0.001 * gps2dist_azimuth(
                            float(flat_lat[j1]), float(flat_lon[j1]),
                            float(flat_lat[j2]), float(flat_lon[j2])
                        )[0]
                    JUMP_KM = 5.0
                    for ii in range(len(sm_node)):
                        if _km(source_node[ii], sm_node[ii]) > JUMP_KM:
                            sm_node[ii] = source_node[ii]
                except Exception:
                    pass

                source_lat = flat_lat[sm_node].astype(float, copy=False)
                source_lon = flat_lon[sm_node].astype(float, copy=False)
                # Gentle DR average to match locate()’s option
                if temporal_smooth_win and temporal_smooth_win >= 3 and temporal_smooth_win % 2 == 1:
                    source_DR = _movavg_1d(source_DR, temporal_smooth_win)

            elif mode == "viterbi" and want_viterbi and misfit_hist:
                # Build (T,J_local) misfit matrix in the same local space used by backend
                M = np.vstack(misfit_hist)
                # Local lat/lon (respect any node subsetting in ctx)
                node_index = ctx.get("node_index", None)
                flat_lat_local = flat_lat if node_index is None else flat_lat[node_index]
                flat_lon_local = flat_lon if node_index is None else flat_lon[node_index]

                sm_local = _viterbi_smooth_indices(
                    misfits_TJ=M,
                    flat_lat=flat_lat_local,
                    flat_lon=flat_lon_local,
                    lambda_km=float(viterbi_lambda_km),
                    max_step_km=(None if viterbi_max_step_km is None else float(viterbi_max_step_km)),
                )
                sm_global = sm_local if node_index is None else np.asarray(node_index, int)[sm_local]
                source_lat = flat_lat[sm_global]
                source_lon = flat_lon[sm_global]
                # DR along the Viterbi path, if mean vectors were collected
                try:
                    mean_M = np.vstack(mean_hist)
                    source_DR = np.array([mean_M[t, sm_local[t]] for t in range(mean_M.shape[0])], dtype=float)
                except Exception:
                    pass
            # else: "none" → do nothing

        # Save source
        self.source = {
            "t": t,
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
            "node_index": source_node,
        }

        # Connectedness metric
        conn = compute_spatial_connectedness(
            self.source["lat"], self.source["lon"], dr=source_DR,
            top_frac=0.15, min_points=12, max_points=200,
        )
        self.connectedness = conn
        if verbose:
            print(f"[ASL] connectedness: score={conn['score']:.3f}  "
                f"n_used={conn['n_used']}  mean_km={conn['mean_km']:.2f}  p90_km={conn['p90_km']:.2f}")

        self.source["connectedness"] = conn["score"]

        # Package to ObsPy Event and finish
        self.source_to_obspyevent()
        self.located = True
        if hasattr(self, "set_id"):
            self.id = self.set_id()
        if verbose:
            print("[ASL] fast_locate: done.")

    # ---------- plots ----------
    def plot(
        self,
        zoom_level=0,
        threshold_DR=0,
        scale=1,
        join=False,
        number=0,
        add_labels=False,
        outfile=None,
        stations=None,
        title=None,
        region=None,
        normalize=True,
        dem_tif: str | None = None,
        simple_basemap: bool = True,
    ):
        """
        Plot the time-ordered source track on a simple basemap.
        - Uses lon/lat/DR from self.source (no external symsize input).
        - If normalize=True: sizes ~ DR / max(DR) * scale
        else: sizes ~ sqrt(max(DR,0)) * scale   (historic behavior)
        - Thresholding zeroes DR and masks lon/lat below threshold.
        - Emits detailed console logs; raises on hard failures.
        """

        def _ts(msg: str) -> None:
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] [ASL:PLOT] {msg}")

        source = self.source
        if not source:
            _ts(f"no source; drawing simple basemap (dem_tif={dem_tif}, simple_basemap={simple_basemap})")
            fig = topo_map(
                zoom_level=zoom_level, inv=None, show=True, add_labels=add_labels,
                topo_color=True, cmap=("land" if simple_basemap else None),
                region=region, dem_tif=dem_tif
            )
            return

        _ts("plot() called with args: "
            f"zoom_level={zoom_level}, threshold_DR={float(threshold_DR)}, scale={float(scale)}, "
            f"join={join}, number={number}, add_labels={add_labels}, outfile={outfile}, title={title}, "
            f"region={region}, normalize={normalize}, dem_tif={dem_tif}, simple_basemap={simple_basemap}")
        _ts(f"source keys: {sorted(list(source.keys()))}")

        # --- DR time series quicklook (non-fatal)
        try:
            t_dt = [tt.datetime for tt in source["t"]]
            DR_ts = np.asarray(source["DR"], float)
            _ts(f"timeseries: len(t)={len(t_dt)} len(DR)={DR_ts.size} "
                f"DR[min,max]=({np.nanmin(DR_ts):.3g},{np.nanmax(DR_ts):.3g})")
            plt.figure()
            plt.plot(t_dt, DR_ts)
            plt.plot(t_dt, np.ones(DR_ts.size) * threshold_DR)
            plt.xlabel("Date/Time (UTC)"); plt.ylabel("Reduced Displacement (${cm}^2$)")
            plt.close()
        except Exception as e:
            _ts(f"WARNING: timeseries preview failed: {e!r}")

        # --- Core arrays
        try:
            x = np.asarray(source["lon"], float)
            y = np.asarray(source["lat"], float)
            DR = np.asarray(source["DR"], float)
        except KeyError as e:
            _ts(f"ERROR: required key missing in source: {e!r}")
            raise
        except Exception as e:
            _ts(f"ERROR: failed to coerce lon/lat/DR: {e!r}")
            raise

        if not (x.size == y.size == DR.size):
            _ts(f"ERROR: length mismatch: len(lon)={x.size}, len(lat)={y.size}, len(DR)={DR.size}")
            raise ValueError("ASL.plot(): lon, lat, DR must have identical lengths.")

        # --- Thresholding (mirror GitHub semantics, but without mutating self.source)
        if float(threshold_DR) > 0:
            mask = DR < float(threshold_DR)
            DR = DR.copy(); x = x.copy(); y = y.copy()
            DR[mask] = 0.0
            x[mask] = np.nan
            y[mask] = np.nan
            _ts(f"threshold applied at {threshold_DR}; masked {int(np.count_nonzero(mask))} points")

        # --- Symbol sizes (what GitHub used, with guards)
        try:
            if normalize:
                mx = float(np.nanmax(DR))
                if not np.isfinite(mx) or mx <= 0:
                    symsize = scale * np.ones_like(DR, dtype=float)
                    _ts("normalize=True but max(DR)<=0; using constant symsize")
                else:
                    symsize = (DR / mx) * float(scale)
            else:
                symsize = float(scale) * np.sqrt(np.maximum(DR, 0.0))
        except Exception as e:
            _ts(f"ERROR: building symsize failed: {e!r}")
            raise

        # --- Validity mask
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(DR) & np.isfinite(symsize) & (symsize > 0)
        n_valid = int(np.count_nonzero(m))
        _ts(f"valid mask: {n_valid}/{x.size} points valid")
        if n_valid == 0:
            _ts("ERROR: no valid points after filtering.")
            raise RuntimeError("ASL.plot(): no valid points to plot after filtering.")

        x, y, DR, symsize = x[m], y[m], DR[m], symsize[m]
        _ts(f"filtered stats: x[{x.min():.5f},{x.max():.5f}] "
            f"y[{y.min():.5f},{y.max():.5f}] "
            f"DR[min,med,max]=({np.nanmin(DR):.3g},{np.nanmedian(DR):.3g},{np.nanmax(DR):.3g}); "
            f"size[min,med,max]=({np.nanmin(symsize):.3g},{np.nanmedian(symsize):.3g},{np.nanmax(symsize):.3g})")

        # --- Map center on max DR
        maxi = int(np.nanargmax(DR))
        center_lat, center_lon = float(y[maxi]), float(x[maxi])
        _ts(f"center on max DR idx={maxi} at (lon,lat)=({center_lon:.5f},{center_lat:.5f})")

        # --- Plain basemap
        try:
            fig = topo_map(
                zoom_level=zoom_level,
                inv=None,
                centerlat=center_lat,
                centerlon=center_lon,
                add_labels=add_labels,
                topo_color=False,                 # <- plain background
                stations=stations,
                title=title,
                region=region,
                dem_tif=dem_tif if dem_tif else None,
                cmap='land',
            )
            _ts("basemap created.")
        except Exception as e:
            _ts(f"ERROR: topo_map() failed: {e!r}")
            raise

        # --- Top-N subselect (by DR) if requested
        if number and number < len(x):
            _ts(f"subselecting top-N by DR: N={number}")
            ind = np.argpartition(DR, -number)[-number:]
            x, y, DR, symsize = x[ind], y[ind], DR[ind], symsize[ind]

        # --- Color by chronological index
        try:
            pygmt.makecpt(cmap="viridis", series=[0, len(x) - 1])
            timecolor = np.arange(len(x), dtype=float)
            _ts("cpt created (viridis).")
        except Exception as e:
            _ts(f"ERROR: makecpt failed: {e!r}")
            raise

        # --- Scatter
        try:
            # NOTE: use style="c" with per-point sizes via `size=symsize`
            fig.plot(x=x, y=y, size=symsize, style="c", pen=None, fill=timecolor, cmap=True)
            _ts(f"scatter plotted: n={len(x)}")
        except Exception as e:
            _ts(f"ERROR: scatter plot failed: {e!r}")
            raise

        # --- Colorbar
        try:
            fig.colorbar(frame='+l"Sequence"')
            _ts("colorbar added.")
        except Exception as e:
            _ts(f"ERROR: colorbar failed: {e!r}")
            raise

        # --- Optional line join in chronological order
        if join and len(x) > 1:
            order = np.argsort(timecolor)
            try:
                fig.plot(x=x[order], y=y[order], pen="1p,red")
                _ts("joined points chronologically with red line.")
            except Exception as e:
                _ts(f"ERROR: join failed: {e!r}")
                raise

        if region:
            try:
                fig.basemap(region=region, frame=True)
                _ts("frame drawn for region.")
            except Exception as e:
                _ts(f"ERROR: frame draw failed: {e!r}")
                raise

        # --- Output
        if outfile:
            try:
                fig.savefig(outfile)
                _ts(f"saved figure: {outfile}")
            except Exception as e:
                _ts(f"ERROR: savefig failed: {e!r}")
                raise
        else:
            _ts("showing figure (interactive).")
            fig.show()


    def plot_reduced_displacement(self, threshold_DR=0, outfile=None):
        if not self.source:
            return
        t_dt = [this_t.datetime for this_t in self.source["t"]]
        plt.figure(); plt.plot(t_dt, self.source["DR"]); plt.plot(t_dt, np.ones(self.source["DR"].size) * threshold_DR)
        plt.xlabel("Date/Time (UTC)"); plt.ylabel("Reduced Displacement (${cm}^2$)")
        (plt.savefig(outfile) if outfile else plt.show())

    def plot_misfit(self, outfile=None):
        if not self.source:
            return
        t_dt = [this_t.datetime for this_t in self.source["t"]]
        plt.figure(); plt.plot(t_dt, self.source["misfit"])
        plt.xlabel("Date/Time (UTC)"); plt.ylabel("Misfit (std/median)")
        (plt.savefig(outfile) if outfile else plt.show())

    def plot_misfit_heatmap(self, outfile=None):
        plot_misfit_heatmap_for_peak_DR(self, backend=StdOverMeanMisfit(), cmap="turbo", transparency=40, outfile=outfile)
        #plot_misfit_heatmap_for_peak_DR(aslobj, backend=R2DistanceMisfit(), cmap="oleron", transparency=45)

    # ---------- ObsPy event ----------
    def source_to_obspyevent(self, event_id=None):
        src = self.source
        if not src:
            return
        if not event_id:
            event_id = src["t"][0].strftime("%Y%m%d%H%M%S")

        ev = Event()
        ev.resource_id = ResourceIdentifier(f"smi:example.org/event/{event_id}")
        ev.event_type = "landslide"
        ev.comments.append(Comment(text="Origin.quality distance fields are in kilometers, not degrees."))

        azgap = src.get("azgap", [None] * len(src["t"]))
        nsta = src.get("nsta", [None] * len(src["t"]))
        misfits = src.get("misfit", [None] * len(src["t"]))

        for i, (t, lat, lon, DR, misfit_val) in enumerate(zip(src["t"], src["lat"], src["lon"], src["DR"], misfits)):
            origin = Origin()
            origin.resource_id = ResourceIdentifier(f"smi:example.org/origin/{event_id}_{i:03d}")
            origin.time = UTCDateTime(t) if not isinstance(t, UTCDateTime) else t
            origin.latitude = lat; origin.longitude = lon; origin.depth = 0
            oq = OriginQuality()
            oq.standard_error = float(misfit_val) if misfit_val is not None else None
            oq.azimuthal_gap = float(azgap[i]) if azgap[i] is not None else None
            oq.used_station_count = int(nsta[i]) if nsta[i] is not None else None

            # pick a node elevation near the current origin (works for Grid or NodeGrid)
            node_z = getattr(self.gridobj, "node_elev_m", None)
            elev_node_m = 0.0
            if node_z is not None and np.isfinite(node_z).any():
                glat = np.asarray(self.gridobj.gridlat).reshape(-1)
                glon = np.asarray(self.gridobj.gridlon).reshape(-1)
                j0 = int(np.nanargmin((glat - lat)**2 + (glon - lon)**2))
                elev_node_m = float(np.asarray(node_z).reshape(-1)[j0])

            distances_km = []
            for coords in self.station_coordinates.values():
                stalat = float(coords["latitude"])
                stalon = float(coords["longitude"])
                staelev = float(coords.get("elevation", 0.0))  # meters
                d_km = geo_distance_3d_km(lat, lon, elev_node_m, stalat, stalon, staelev)
                distances_km.append(d_km)

            if distances_km:
                oq.minimum_distance = min(distances_km)
                oq.maximum_distance = max(distances_km)
                oq.median_distance = float(np.median(distances_km))

            origin.quality = oq
            origin.time_errors = QuantityError(uncertainty=misfit_val)
            ev.origins.append(origin)

            amp = Amplitude()
            amp.resource_id = ResourceIdentifier(f"smi:example.org/amplitude/{event_id}_{i:03d}")
            amp.generic_amplitude = DR
            amp.unit = "other"
            amp.pick_id = origin.resource_id
            ev.amplitudes.append(amp)

        self.event = ev

    def save_event(self, outfile=None):
        if self.located and outfile:
            cat = Catalog(events=[self.event])
            cat.write(outfile, format="QUAKEML")
            print(f"[✓] QuakeML: {outfile}")

    def print_event(self):
        if self.located:
            pprint(self.event)


    def source_to_dataframe(self) -> "pd.DataFrame | None":
        """
        Convert the ASL source dict to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of the source information, or None if no source exists.
        """
        if not hasattr(self, "source") or self.source is None:
            print("[ASL] No source data available. Did you run locate() or fast_locate()?")
            return None

        return pd.DataFrame(self.source)


    def print_source(self, max_rows: int = 100):
        """
        Pretty-print the ASL source dictionary as a DataFrame.

        Parameters
        ----------
        max_rows : int, default=20
            Maximum number of rows to display. Use None to show all.
        """
        df = self.source_to_dataframe()
        if df is None:
            return

        if max_rows is not None and len(df) > max_rows:
            print(f"[ASL] Showing first {max_rows} of {len(df)} rows:")
            print(df.head(max_rows).to_string(index=False))
        else:
            print(df.to_string(index=False))

        print(f'Unique node indices: {np.unique(self.source["node_index"]).size}')
        return df


    def source_to_csv(self, csvfile: str, index: bool = False):
        """
        Save the ASL source dictionary as a CSV file.

        Parameters
        ----------
        csvfile : str
            Path to output CSV file.
        index : bool, default=False
            Whether to include the DataFrame index in the CSV.
        """
        df = self.source_to_dataframe()
        if df is None:
            return

        df.to_csv(csvfile, index=index)
        print(f"[ASL] Source written to CSV: {csvfile}")
        return csvfile


    def estimate_decay_params(
        self,
        time_index: int | None = None,
        node_index: int | None = None,
        *,
        use_reduced: bool = False,          # fit on raw amplitudes by default
        bounded: bool = False,              # enforce N>=0, k>=0 if SciPy available
        neighborhood_k: int | None = None,  # e.g., 9 → search nearest 9 nodes and pick best R²
        min_stations: int = 5,
        eps_amp: float = 1e-12,
        eps_dist_km: float = 1e-6,
        freq_hz: float | None = None,       # if None, falls back to self.peakf
        wavespeed_kms: float | None = None, # if None, falls back to self.wavespeed_kms
        verbose: bool = True,
    ) -> dict:
        """
        Estimate geometric spreading exponent N and attenuation factor k from station amplitudes:
            A(R) ≈ A0 * R^{-N} * exp(-k R)
        using the log-linear model:
            log A = b0 + b1 * log R + b2 * R,   with  N = -b1,  k = -b2
        and convert k → Q via  Q = (π f) / (k v)  when f and v are available.

        - Prefers precomputed distances in self.node_distances_km; falls back to gps2dist_azimuth.
        - Can optionally enforce N,k ≥ 0 with bounded least squares (SciPy).
        - Can search a neighborhood of nodes and return the best fit by R².
        - Returns standard errors if available.
        """

        # ---- guards / defaults from located source ----
        if time_index is None or node_index is None:
            if not getattr(self, "source", None):
                raise ValueError("Provide time_index/node_index or run locate()/fast_locate() first.")
            if time_index is None:
                time_index = int(np.nanargmax(self.source["DR"]))
            if node_index is None:
                glat = self.gridobj.gridlat.reshape(-1)
                glon = self.gridobj.gridlon.reshape(-1)
                lat0 = float(self.source["lat"][time_index])
                lon0 = float(self.source["lon"][time_index])
                node_index = int(np.nanargmin((glat - lat0) ** 2 + (glon - lon0) ** 2))

        # convenience handles
        glat = self.gridobj.gridlat.reshape(-1)
        glon = self.gridobj.gridlon.reshape(-1)
        nnodes = glat.size

        if not (0 <= node_index < nnodes):
            raise IndexError("node_index out of bounds.")

        # choose frequency/velocity for Q
        f_hz = float(freq_hz if freq_hz is not None else getattr(self, "peakf", np.nan))
        v_kms = float(wavespeed_kms if wavespeed_kms is not None else getattr(self, "wavespeed_kms", np.nan))

        # --- helpers ---
        def _distances_for_node(nj: int, seed_ids: list[str]) -> np.ndarray:
            """
            Return distances (km) station→node nj.
            Prefer self.node_distances_km[sid][nj]; otherwise compute 2-D great-circle.
            """
            d_km = np.empty(len(seed_ids), dtype=float)
            have_all = bool(self.node_distances_km) and all(
                (sid in self.node_distances_km and
                np.asarray(self.node_distances_km[sid]).size == nnodes)
                for sid in seed_ids
            )
            if have_all:
                for k, sid in enumerate(seed_ids):
                    d_km[k] = float(np.asarray(self.node_distances_km[sid], float)[nj])
                return d_km
            # fallback: 2-D distances
            lat_j, lon_j = float(glat[nj]), float(glon[nj])
            for k, sid in enumerate(seed_ids):
                c = self.station_coordinates.get(sid)
                if c is None or not np.isfinite(c.get("latitude")) or not np.isfinite(c.get("longitude")):
                    d_km[k] = np.nan
                else:
                    dm, _, _ = gps2dist_azimuth(lat_j, lon_j, float(c["latitude"]), float(c["longitude"]))
                    d_km[k] = dm / 1000.0
            return d_km

        def _fit_once(ti: int, nj: int) -> dict:
            # Stream → station amplitudes at a single time
            st = self.metric2stream()
            seed_ids = [tr.id for tr in st]
            Y = np.vstack([tr.data.astype(np.float64, copy=False) for tr in st])
            a_raw = Y[:, ti]  # (nsta,)

            # choose amplitude vector
            if use_reduced:
                # multiply by amplitude corrections at node nj if available
                C = np.ones_like(a_raw, dtype=np.float64)
                for k, sid in enumerate(seed_ids):
                    ck = self.amplitude_corrections.get(sid)
                    if ck is not None:
                        val = np.asarray(ck, float)[nj]
                        C[k] = val if np.isfinite(val) else 1.0
                A = a_raw * C
            else:
                A = a_raw

            # distances
            R = _distances_for_node(nj, seed_ids)

            # validity mask
            m = np.isfinite(A) & np.isfinite(R) & (A > 0.0) & (R > eps_dist_km)
            nuse = int(m.sum())
            lat_j, lon_j = float(glat[nj]), float(glon[nj])

            if nuse < min_stations:
                return dict(N=np.nan, k=np.nan, Q=np.nan, A0_log=np.nan, r2=0.0,
                            nsta=nuse, time_index=ti, node_index=nj,
                            lat=lat_j, lon=lon_j, se_N=np.nan, se_k=np.nan, se_A0=np.nan,
                            method="insufficient")

            A = A[m]
            R = R[m]
            logA = np.log(np.maximum(A, eps_amp))
            logR = np.log(np.maximum(R, eps_dist_km))

            # quick well-posedness checks
            rspan_min_km = 2.0
            logr_span_min = 0.5
            if (np.nanmax(R) - np.nanmin(R)) < rspan_min_km or (np.ptp(logR) < logr_span_min):
                return dict(N=np.nan, k=np.nan, Q=np.nan, A0_log=np.nan, r2=-np.inf,
                            nsta=nuse, time_index=ti, node_index=nj,
                            lat=lat_j, lon=lon_j, se_N=np.nan, se_k=np.nan, se_A0=np.nan,
                            method="illposed")

            # Design matrix: [1, logR, R]
            X = np.column_stack([np.ones_like(logA), logR, R])

            b = None
            cov = None
            used_method = "OLS"

            if bounded:
                # optional bounded fit (N>=0, k>=0) i.e., b1<=0, b2<=0
                try:
                    from scipy.optimize import lsq_linear
                    # center predictors to reduce collinearity; solve without intercept then recover it
                    logR_c = logR - logR.mean()
                    R_c = R - R.mean()
                    Xc = np.column_stack([logR_c, R_c])
                    y_c = logA - logA.mean()
                    res = lsq_linear(Xc, y_c, bounds=([-np.inf, -np.inf], [0.0, 0.0]))
                    b1_c, b2_c = res.x
                    b0 = float(logA.mean())
                    b = np.array([b0, b1_c, b2_c], float)
                    used_method = "bounded"
                except Exception:
                    pass  # fall back to OLS

            if b is None:
                # OLS with covariance
                b, *_ = np.linalg.lstsq(X, logA, rcond=None)
                try:
                    resid = logA - X @ b
                    dof = max(1, X.shape[0] - X.shape[1])
                    sigma2 = float((resid @ resid) / dof)
                    XtX_inv = np.linalg.inv(X.T @ X)
                    cov = sigma2 * XtX_inv
                except Exception:
                    cov = None

            b0, b1, b2 = map(float, b)
            N_hat = -b1
            k_hat = -b2

            # R²
            yhat = X @ np.array([b0, b1, b2], float)
            ss_res = float(np.sum((logA - yhat) ** 2))
            ss_tot = float(np.sum((logA - logA.mean()) ** 2)) + 1e-20
            r2 = 1.0 - ss_res / ss_tot

            # Q from k
            if np.isfinite(f_hz) and np.isfinite(v_kms) and k_hat > 0:
                Q_hat = float(np.pi * f_hz / (k_hat * v_kms))
            else:
                Q_hat = np.nan

            # standard errors
            se_A0 = se_N = se_k = np.nan
            if cov is not None and cov.shape == (3, 3):
                se_A0 = float(np.sqrt(max(0.0, cov[0, 0])))
                se_N  = float(np.sqrt(max(0.0, cov[1, 1])))
                se_k  = float(np.sqrt(max(0.0, cov[2, 2])))

            out = dict(
                N=float(N_hat), k=float(k_hat), Q=float(Q_hat), A0_log=float(b0), r2=float(r2),
                nsta=int(nuse), time_index=int(ti), node_index=int(nj),
                lat=lat_j, lon=lon_j,
                se_N=se_N, se_k=se_k, se_A0=se_A0,
                method=used_method
            )

            if verbose:
                print(f"[ASL] decay-fit @ t={ti}, node={nj} (lat={lat_j:.5f}, lon={lon_j:.5f})  "
                    f"N={out['N']:.3f}, k={out['k']:.5g} 1/km, Q={out['Q']:.1f} "
                    f"(f={None if not np.isfinite(f_hz) else f_hz}, "
                    f"v={None if not np.isfinite(v_kms) else v_kms})  "
                    f"r2={out['r2']:.3f}  nsta={out['nsta']}  method={used_method}")
            return out

        # --- single node or neighborhood search ---
        if neighborhood_k is None or neighborhood_k <= 1:
            return _fit_once(int(time_index), int(node_index))

        # nearest-k nodes to the starting node (planar metric OK for local grids)
        d2 = (glat - float(glat[int(node_index)])) ** 2 + (glon - float(glon[int(node_index)])) ** 2
        cand = np.argsort(d2)[:int(neighborhood_k)]

        fits = [_fit_once(int(time_index), int(j)) for j in cand]
        best = max(
            fits,
            key=lambda p: (np.nan_to_num(p.get("r2", np.nan), nan=-np.inf),
                        -abs(p.get("N", np.nan)) if np.isfinite(p.get("N", np.nan)) else -np.inf)
        )
        best["candidates"] = [(int(f["node_index"]), float(f["r2"])) for f in fits]
        return best
    

    # --- NEW: apply a boolean mask over nodes (no recompute) ---
    def apply_node_mask(self, mask: np.ndarray):
        """
        Keep only nodes where mask==True by slicing:
          - grid (lat/lon)
          - amplitude_corrections per channel
          - node_distances_km per channel
        No recomputation is performed.

        Returns self (for chaining).
        """
        mask = np.asarray(mask, bool).reshape(-1)
        old = _as_regular_view(self.gridobj)
        if mask.size != old.gridlat.size:
            raise ValueError("mask length != number of grid nodes")

        # Slice grid → a lightweight view
        sub_lat = old.gridlat[mask]
        sub_lon = old.gridlon[mask]
        self.gridobj = SimpleNamespace(
            id=f"{old.id}-subset-{mask.sum()}",
            gridlat=sub_lat,
            gridlon=sub_lon,
            nlat=None, nlon=None,
            node_spacing_m=getattr(old, "node_spacing_m", None),
        )

        # Slice corrections & distances
        if getattr(self, "amplitude_corrections", None):
            for sid, vec in list(self.amplitude_corrections.items()):
                v = np.asarray(vec)
                if v.size != mask.size:
                    raise ValueError(f"{sid}: corrections length mismatch {v.size} vs {mask.size}")
                self.amplitude_corrections[sid] = v[mask]

        if getattr(self, "node_distances_km", None):
            for sid, vec in list(self.node_distances_km.items()):
                v = np.asarray(vec)
                if v.size != mask.size:
                    raise ValueError(f"{sid}: distances length mismatch {v.size} vs {mask.size}")
                self.node_distances_km[sid] = v[mask]

        return self

    # --- NEW: subset by bounding box (lat/lon) ---
    def subset_nodes_by_bbox(self, lat_min, lat_max, lon_min, lon_max, *, inplace: bool = True):
        """Convenience wrapper to call apply_node_mask() for a lat/lon bbox."""
        g = _as_regular_view(self.gridobj)
        mask = (
            (g.gridlat >= min(lat_min, lat_max)) & (g.gridlat <= max(lat_min, lat_max)) &
            (g.gridlon >= min(lon_min, lon_max)) & (g.gridlon <= max(lon_min, lon_max))
        )
        return self.apply_node_mask(mask)

    # --- NEW: coarse→fine without new grid: subset + relocate ---
    

    @contextmanager
    def _temporary_node_mask(aslobj, idx: np.ndarray | None):
        """Temporarily set aslobj._node_mask to idx, then restore."""
        old = getattr(aslobj, "_node_mask", None)
        try:
            aslobj._node_mask = None if idx is None or idx.size == 0 else np.asarray(idx, int)
            yield
        finally:
            aslobj._node_mask = old

    def _passthrough_locate_kwargs(
        *,
        misfit_backend=None,
        spatial_smooth_sigma=None,
        temporal_smooth_mode=None,
        temporal_smooth_win=None,
        viterbi_lambda_km=None,
        viterbi_max_step_km=None,
        min_stations=None,
        eps=None,
        use_median_for_DR=None,
        batch_size=None,
    ):
        kw = {}
        if misfit_backend is not None:       kw["misfit_backend"] = misfit_backend
        if spatial_smooth_sigma is not None: kw["spatial_smooth_sigma"] = spatial_smooth_sigma
        if temporal_smooth_mode is not None: kw["temporal_smooth_mode"] = temporal_smooth_mode
        if temporal_smooth_win is not None:  kw["temporal_smooth_win"] = temporal_smooth_win
        if viterbi_lambda_km is not None:    kw["viterbi_lambda_km"] = viterbi_lambda_km
        if viterbi_max_step_km is not None:  kw["viterbi_max_step_km"] = viterbi_max_step_km
        if min_stations is not None:         kw["min_stations"] = min_stations
        if eps is not None:                  kw["eps"] = eps
        if use_median_for_DR is not None:    kw["use_median_for_DR"] = use_median_for_DR
        if batch_size is not None:           kw["batch_size"] = batch_size
        return kw

    def refine_and_relocate(
        self,
        *,
        # How to build the mask
        mask_method: str = "bbox",            # "bbox" | "sector"
        # --- common selection controls ---
        top_frac: float = 0.20,               # fraction of samples (by DR) to define bbox or bearing
        min_top_points: int = 8,              # lower bound on #points used to make bbox/bearing
        # --- bbox-specific controls ---
        margin_km: float = 1.0,               # half-margin added around bbox (converted to deg)
        # --- sector-specific controls ---
        apex_lat: float | None = None,
        apex_lon: float | None = None,
        length_km: float = 8.0,
        inner_km: float = 0.0,
        half_angle_deg: float = 25.0,
        prefer_misfit: bool = True,           # True: use min-misfit bearing; False: use median of top-DR
        top_frac_for_bearing: float | None = None,  # fallback to `top_frac` if None
        # Application mode
        mode: str = "mask",                   # "mask" (non-destructive) or "slice" (destructive)
        # fast_locate() passthroughs / aliases
        misfit_backend=None,
        spatial_smooth_sigma: float | None = None,
        spatial_sigma_nodes: float | None = None,   # alias (legacy)
        temporal_smooth_mode: str | None = None,    # "none" | "median" | "viterbi"
        temporal_smooth_win: int | None = None,
        viterbi_lambda_km: float | None = None,
        viterbi_max_step_km: float | None = None,
        min_stations: int | None = None,
        eps: float | None = None,
        use_median_for_DR: bool | None = None,
        batch_size: int | None = None,
        verbose: bool = True,
    ):
        """
        Refine the spatial search domain by building a node mask, then re-run
        :meth:`fast_locate` on the restricted grid.

        Overview
        --------
        This method **does not recompute** distances or amplitude corrections. It only
        builds a boolean mask over the *existing* grid nodes and then calls
        :meth:`fast_locate` again so the locator evaluates fewer candidate nodes.

        Two masking strategies are supported via ``mask_method``:

        1) ``mask_method="bbox"`` (default)
        Build a **bounding box** around the top fraction of samples ranked by DR
        (reduced displacement) from the **current** source track, expand it by a
        margin (km → degrees), and keep nodes inside the box.

        2) ``mask_method="sector"``
        Build a **triangular wedge** (“apex-to-coast” sector) that starts at the
        dome apex and extends outward along an inferred bearing (either toward the
        minimum-misfit sample or toward the median location of the top-DR samples).
        Keep nodes whose (apex→node) distance is within [inner_km, length_km] and
        whose azimuth is within ±half_angle_deg of the inferred bearing.

        In both cases, the newly built mask is **intersected** with:
        • any persistent mask carried by the grid object (e.g., land-only mask), and  
        • any temporary mask currently active in ``self._node_mask``.

        After the intersection, you can either:
        • apply the mask temporarily and run on the full grid data (``mode="mask"``), or  
        • **destructively slice** the grid and its per-station vectors
            (``mode="slice"``), permanently reducing memory/footprint for this ASL object.

        Parameters
        ----------
        mask_method : {"bbox", "sector"}, default "bbox"
            How to build the refinement mask.
            - "bbox": Use a DR-based bounding box (see ``top_frac`` / ``margin_km``).
            - "sector": Use an apex-anchored wedge (see sector-specific parameters).

        top_frac : float, default 0.20
            Fraction (0–1] of valid samples (by DR) used to define the **bbox** or,
            for the sector method, used (when ``prefer_misfit=False``) to compute the
            bearing from the median of the top-DR points. A sensible minimum count
            is enforced via ``min_top_points``.

        min_top_points : int, default 8
            Lower bound on the number of samples used to define the bbox or the
            top-DR median (sector bearing fallback).

        margin_km : float, default 1.0
            **Bbox only.** Half-width expansion added to the min–max lat/lon of the
            selected top-DR samples, expressed in kilometers. Internally converted to
            degrees (lat: km/111, lon: km/(111·cos φ)) using the median latitude of
            the selected points.

        apex_lat, apex_lon : float or None, default None
            **Sector only.** Geographic coordinates of the wedge apex (e.g., dome
            summit). If ``None``, the method attempts to read ``gridobj.apex_lat`` and
            ``gridobj.apex_lon``. Required for the sector method.

        length_km : float, default 8.0
            **Sector only.** Maximum radial extent of the wedge from the apex.

        inner_km : float, default 0.0
            **Sector only.** Inner radial exclusion from the apex (useful to skip
            near-field nodes, e.g., if they sit inside the crater).

        half_angle_deg : float, default 25.0
            **Sector only.** Half-angle (degrees) of the wedge around the inferred
            bearing; total wedge aperture is 2×half_angle_deg.

        prefer_misfit : bool, default True
            **Sector only.** If True, infer the wedge bearing from the **global
            minimum of misfit** in the current source track. If False (or when the
            misfit isn’t usable), infer the bearing from the **median location of the
            top-DR subset** (see ``top_frac_for_bearing``).

        top_frac_for_bearing : float or None, default None
            **Sector only.** If provided (and ``prefer_misfit=False``), the fraction
            used specifically for selecting the top-DR subset to infer the bearing.
            If None, falls back to ``top_frac``.

        mode : {"mask", "slice"}, default "mask"
            How to apply the resulting mask:
            - "mask": apply as a temporary mask (non-destructive), run :meth:`fast_locate`,
            then restore the full grid.
            - "slice": destructively **subset** the grid and attached vectors
            (amplitude corrections and distances) to the masked nodes, then run
            :meth:`fast_locate` on the smaller, permanent grid.

        misfit_backend : object, optional
            Passed through to :meth:`fast_locate` (see that docstring). If None,
            :class:`StdOverMeanMisfit` is used.

        spatial_smooth_sigma : float, optional
            Passed through to :meth:`fast_locate`. If provided, enables Gaussian blur
            of the misfit per time step (units: grid nodes). ``spatial_sigma_nodes`` is
            accepted as an alias for backward compatibility.

        temporal_smooth_mode : {"none", "median", "viterbi"}, optional
            Passed through to :meth:`fast_locate`. If None, current default of
            :meth:`fast_locate` is used.

        temporal_smooth_win, viterbi_lambda_km, viterbi_max_step_km : optional
            Passed through to :meth:`fast_locate` (see that docstring).

        min_stations, eps, use_median_for_DR, batch_size : optional
            Passed through to :meth:`fast_locate` (see that docstring).

        verbose : bool, default True
            Print diagnostics about mask construction, intersection, and final size.

        Returns
        -------
        self : ASL
            The same object, updated in-place with a new ``self.source`` (and possibly
            a sliced grid if ``mode="slice"``).

        Side Effects
        ------------
        - Updates ``self.source`` with a newly computed track from :meth:`fast_locate`.
        - Updates ``self.connectedness`` based on the refined track.
        - If ``mode="slice"``, permanently modifies:
            * ``self.gridobj`` (lat/lon arrays, shape/metadata)
            * ``self.amplitude_corrections`` per station (sliced)
            * ``self.node_distances_km`` per station (sliced)

        Mask Intersection
        -----------------
        The constructed mask is AND-ed with:
        1) any persistent node mask that your grid carries (e.g., ``gridobj.node_mask``,
            ``gridobj.mask``, …), and
        2) any temporary mask already active in ``self._node_mask``.
        This guarantees land-only (or otherwise pre-filtered) grids remain respected.

        Algorithmic Notes
        -----------------
        **bbox**:
        1) Select indices with finite (lat, lon, DR). Rank by DR and take
            ``max(min_top_points, ceil(top_frac*N))`` samples.
        2) Compute min/max lat/lon of those samples; expand by ``margin_km`` converted
            to degrees with a latitude-dependent lon scaling.
        3) Keep nodes inside the expanded box.

        **sector**:
        1) Determine bearing from apex:
            - If ``prefer_misfit=True`` and any finite misfit exists, take the index
            of the minimum misfit sample and compute apex→that point azimuth.
            - Else, take the median (lat,lon) of the top-DR subset (size controlled by
            ``top_frac_for_bearing`` or ``top_frac``) and compute apex→median azimuth.
        2) For each grid node, compute apex→node great-circle distance and azimuth.
        3) Keep nodes where distance∈[inner_km, length_km] **and**
            circular_angle_difference(az, bearing) ≤ half_angle_deg.

        Examples
        --------
        # 1) Classic DR-bbox refinement (non-destructive), then re-run fast_locate
        asl.refine_and_relocate(mask_method="bbox", top_frac=0.25, margin_km=1.0)

        # 2) Sector refinement from dome apex toward inferred direction of propagation
        #    (Soufrière Hills example coordinates), with Viterbi temporal smoothing
        asl.refine_and_relocate(
            mask_method="sector",
            apex_lat=16.712, apex_lon=-62.181,  # dome/apex
            length_km=8.0, inner_km=0.0, half_angle_deg=25.0,
            prefer_misfit=True,                 # use min-misfit bearing
            temporal_smooth_mode="viterbi", temporal_smooth_win=7,
            viterbi_lambda_km=8.0, viterbi_max_step_km=30.0,
        )

        # 3) Same as (2) but *destructively* slice the grid to the sector for speed
        asl.refine_and_relocate(
            mask_method="sector",
            apex_lat=16.712, apex_lon=-62.181,
            length_km=8.0, half_angle_deg=25.0,
            mode="slice"
        )

        Caveats
        -------
        - If the intersection yields **no nodes**, the method falls back to running
        :meth:`fast_locate` on the **current** domain and returns without error.
        - For the sector method, you **must** provide an apex (via parameters or
        grid attributes). If none is available, a ValueError is raised.
        - Extremely small grids or very narrow wedges can cause over-constrained
        masks; consider relaxing ``half_angle_deg`` or increasing ``length_km``.
        """

        if self.source is None:
            raise RuntimeError("No source yet. Run fast_locate()/locate() first.")

        # alias support
        if spatial_smooth_sigma is None and spatial_sigma_nodes is not None:
            spatial_smooth_sigma = spatial_sigma_nodes
        if top_frac_for_bearing is None:
            top_frac_for_bearing = top_frac

        # -- convenience handles
        lat = np.asarray(self.source["lat"], float)
        lon = np.asarray(self.source["lon"], float)
        dr  = np.asarray(self.source.get("DR", np.full_like(lat, np.nan)), float)
        mis = np.asarray(self.source.get("misfit", np.full_like(lat, np.nan)), float)

        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            if verbose:
                print("[ASL] refine_and_relocate: no finite source coords; skipping")
            return self

        g_lat = np.asarray(self.gridobj.gridlat).reshape(-1)
        g_lon = np.asarray(self.gridobj.gridlon).reshape(-1)
        nn = g_lat.size

        def _intersect_with_existing_mask(mask_bool: np.ndarray) -> np.ndarray:
            # Intersect with persistent mask on the grid, if present
            base_idx = _grid_mask_indices(self.gridobj)  # None or GLOBAL indices
            if base_idx is not None and base_idx.size:
                base_bool = np.zeros(nn, dtype=bool)
                base_bool[base_idx] = True
                mask_bool &= base_bool
            # Intersect with any active temporary mask
            if getattr(self, "_node_mask", None) is not None and self._node_mask.size:
                temp_bool = np.zeros(nn, dtype=bool)
                temp_bool[self._node_mask] = True
                mask_bool &= temp_bool
            return mask_bool

        # ---------- mask builders ----------
        def _mask_bbox() -> np.ndarray:
            # choose top-fraction by DR with sensible minimum count
            idx_all = np.flatnonzero(ok & np.isfinite(dr))
            if idx_all.size == 0:
                # fallback: use all finite coords
                idx_all = np.flatnonzero(ok)

            k_frac = int(np.ceil(top_frac * idx_all.size))
            k = max(1, min_top_points, k_frac)
            k = min(k, idx_all.size)
            order = np.argsort(dr[idx_all])[::-1] if np.isfinite(dr[idx_all]).any() else np.arange(idx_all.size)
            idx_top = idx_all[order[:k]]

            lat_top = lat[idx_top]; lon_top = lon[idx_top]

            lat0 = float(np.nanmedian(lat_top))
            dlat = float(margin_km) / 111.0
            dlon = float(margin_km) / (111.0 * max(0.1, np.cos(np.deg2rad(lat0))))

            lat_min = float(np.nanmin(lat_top) - dlat)
            lat_max = float(np.nanmax(lat_top) + dlat)
            lon_min = float(np.nanmin(lon_top) - dlon)
            lon_max = float(np.nanmax(lon_top) + dlon)

            mask = (
                (g_lat >= lat_min) & (g_lat <= lat_max) &
                (g_lon >= lon_min) & (g_lon <= lon_max)
            )
            if verbose:
                print(f"[ASL] refine_and_relocate[bbox]: bbox lat[{lat_min:.5f},{lat_max:.5f}] "
                    f"lon[{lon_min:.5f},{lon_max:.5f}] using top {k}/{idx_all.size}")
            return mask

        def _mask_sector() -> np.ndarray:
            # apex: explicit or from grid
            apx_lat = apex_lat if apex_lat is not None else getattr(self.gridobj, "apex_lat", None)
            apx_lon = apex_lon if apex_lon is not None else getattr(self.gridobj, "apex_lon", None)
            if apx_lat is None or apx_lon is None:
                raise ValueError("sector mask requires apex_lat/apex_lon (or gridobj.apex_lat/apex_lon).")

            def _bearing(aplat, aplon, tlat, tlon) -> float:
                _, az, _ = gps2dist_azimuth(float(aplat), float(aplon), float(tlat), float(tlon))
                return float(az % 360.0)

            # bearing: min misfit or median of top-DR
            if prefer_misfit and np.isfinite(mis[ok]).any():
                j = int(np.nanargmin(mis))
                bearing_deg = _bearing(apx_lat, apx_lon, lat[j], lon[j])
                if verbose:
                    print(f"[ASL] refine_and_relocate[sector]: bearing from min misfit @ idx={j}: {bearing_deg:.1f}°")
            else:
                idx_all = np.flatnonzero(ok & np.isfinite(dr))
                if idx_all.size == 0:
                    tlat = float(np.nanmedian(lat[ok])); tlon = float(np.nanmedian(lon[ok]))
                else:
                    k = max(3, min(int(np.ceil((top_frac_for_bearing or top_frac) * idx_all.size)), idx_all.size))
                    order = np.argsort(dr[idx_all])[::-1]
                    idx_top = idx_all[order[:k]]
                    tlat = float(np.nanmedian(lat[idx_top])); tlon = float(np.nanmedian(lon[idx_top]))
                bearing_deg = _bearing(apx_lat, apx_lon, tlat, tlon)
                if verbose:
                    print(f"[ASL] refine_and_relocate[sector]: bearing from top-DR median: {bearing_deg:.1f}°")

            # distances/azimuths apex → nodes
            dist_km = np.empty(nn, float)
            az_deg  = np.empty(nn, float)
            for i in range(nn):
                d_m, az, _ = gps2dist_azimuth(float(apx_lat), float(apx_lon), float(g_lat[i]), float(g_lon[i]))
                dist_km[i] = 0.001 * float(d_m)
                az_deg[i]  = float(az % 360.0)

            # circular angular difference
            def _angdiff(a, b):
                d = (a - b + 180.0) % 360.0 - 180.0
                return np.abs(d)

            ang_err = _angdiff(az_deg, bearing_deg)
            mask = (
                (dist_km >= float(inner_km)) &
                (dist_km <= float(length_km)) &
                (ang_err  <= float(half_angle_deg))
            )
            if verbose:
                print(f"[ASL] refine_and_relocate[sector]: span=[{inner_km:.2f},{length_km:.2f}] km, "
                    f"half-angle={half_angle_deg:.1f}°, apex=({apx_lat:.5f},{apx_lon:.5f})")
            return mask

        # ---------- build mask ----------
        mm = (mask_method or "bbox").lower()
        if mm == "bbox":
            mask_bool = _mask_bbox()
        elif mm == "sector":
            mask_bool = _mask_sector()
        else:
            raise ValueError("mask_method must be one of {'bbox','sector'}")

        # intersect with any existing masks
        mask_bool = _intersect_with_existing_mask(mask_bool)
        sub_idx = np.flatnonzero(mask_bool)

        # ---------- run or bail gracefully ----------
        """
        kwargs = {}
        if misfit_backend is not None:        kwargs["misfit_backend"] = misfit_backend
        if spatial_smooth_sigma is not None:  kwargs["spatial_smooth_sigma"] = spatial_smooth_sigma
        if temporal_smooth_mode is not None:  kwargs["temporal_smooth_mode"] = temporal_smooth_mode
        if temporal_smooth_win is not None:   kwargs["temporal_smooth_win"]  = temporal_smooth_win
        if viterbi_lambda_km is not None:     kwargs["viterbi_lambda_km"]    = viterbi_lambda_km
        if viterbi_max_step_km is not None:   kwargs["viterbi_max_step_km"]  = viterbi_max_step_km
        if min_stations is not None:          kwargs["min_stations"] = min_stations
        if eps is not None:                   kwargs["eps"] = eps
        if use_median_for_DR is not None:     kwargs["use_median_for_DR"] = use_median_for_DR
        if batch_size is not None:            kwargs["batch_size"] = batch_size
        """

        kwargs = self._passthrough_locate_kwargs(
            misfit_backend=misfit_backend,
            spatial_smooth_sigma=spatial_smooth_sigma,
            temporal_smooth_mode=temporal_smooth_mode,
            temporal_smooth_win=temporal_smooth_win,
            viterbi_lambda_km=viterbi_lambda_km,
            viterbi_max_step_km=viterbi_max_step_km,
            min_stations=min_stations,
            eps=eps,
            use_median_for_DR=use_median_for_DR,
            batch_size=batch_size,
        )

        if sub_idx.size == 0:
            if verbose:
                print("[ASL] refine_and_relocate: sub-grid empty after intersection; keeping current domain.")
            self.fast_locate(verbose=verbose, **kwargs)
            return self

        if verbose:
            frac = 100.0 * sub_idx.size / nn
            print(f"[ASL] refine_and_relocate[{mm}]: {sub_idx.size} nodes (~{frac:.1f}% of grid)")

        if mode.lower() == "slice":
            # Destructive grid reduction
            self.apply_node_mask(mask_bool)
            self.fast_locate(verbose=verbose, **kwargs)
            return self

        # Temporary mask
        try:
            self._node_mask = sub_idx
            self.fast_locate(verbose=verbose, **kwargs)
        finally:
            self._node_mask = None

        return self
 