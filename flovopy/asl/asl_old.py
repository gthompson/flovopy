# flovopy/asl/asl.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Iterable, Tuple
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt
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
from flovopy.asl.diagnostics import extract_asl_diagnostics, compare_asl_sources
from flovopy.asl.grid import Grid, NodeGrid
from flovopy.asl.misfit import (
    plot_misfit_heatmap_for_peak_DR, StdOverMeanMisfit, R2DistanceMisfit,
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

    def __init__(self, samobject: VSAM, metric: str, gridobj: Grid | NodeGrid, window_seconds: int):    
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

        - Scans each time sample and every grid node with Python loops.
        - Misfit per node: std(reduced) / (|mean(reduced)| + eps), reduced = y * C[:, j].
        - DR(t): median(reduced) if use_median_for_DR else mean(reduced) at the best node.
        - Optional spatial smoothing: Gaussian filter applied to the per-time misfit
        surface before choosing the best node.
        - Optional temporal smoothing: moving average on (lat, lon, DR) tracks.
        - Computes azimuthal gap and a spatial connectedness score (same as fast_locate).

        Notes
        -----
        This is intended for clarity/debugging and will be much slower than fast_locate().
        """

        if verbose:
            print("[ASL] locate (slow): preparing data…")

        # Grid vectors
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)
        nnodes = gridlat.size

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

        # Corrections → C (nsta, nnodes_local)
        C = np.empty((nsta, nnodes_local), dtype=np.float32)
        for k, sid in enumerate(seed_ids):
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

        # Optional spatial blur setup
        try:
            from scipy.ndimage import gaussian_filter as _gauss
        except Exception:
            _gauss = None

        # Helper: grid shape if regular (so we can reshape/blur)
        def _grid_shape_or_none(grid):
            # Works for regular lat/lon mesh created by Grid
            try:
                return int(grid.nlat), int(grid.nlon)
            except Exception:
                return None

        grid_shape = _grid_shape_or_none(self.gridobj)
        can_blur = (
            spatial_smooth_sigma and spatial_smooth_sigma > 0.0 and
            (grid_shape is not None) and (_gauss is not None)
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
            # Scan all nodes
            for j in range(nnodes_local):
                reduced = y * C[:, j]          # (nsta,)
                finite  = np.isfinite(reduced)
                nfin    = int(finite.sum())
                nused_j[j] = nfin
                if nfin < min_stations:
                    continue

                r = reduced[finite]
                if r.size == 0:
                    continue

                # DR summary across stations at this node
                if use_median_for_DR:
                    DRj = float(np.median(r))
                else:
                    DRj = float(np.mean(r))
                dr_j[j] = DRj

                mu = float(np.mean(r))
                sg = float(np.std(r, ddof=0))
                if not np.isfinite(mu) or abs(mu) < eps:
                    continue

                misfit_j[j] = sg / (abs(mu) + eps)

            # Optional spatial smoothing on misfit surface before picking best node
            if can_blur:
                nlat, nlon = grid_shape
                m2 = misfit_j.reshape(nlat, nlon)
                m2 = _gauss(m2, sigma=spatial_smooth_sigma, mode="nearest")
                misfit_for_pick = m2.reshape(-1)
            else:
                misfit_for_pick = misfit_j

            # Choose best node
            #jstar = int(np.nanargmin(misfit_for_pick))
            jstar_local = int(np.nanargmin(misfit_for_pick))
            jstar_global = (int(node_index[jstar_local]) if node_index is not None else jstar_local)

            # Record outputs at time i
            source_lat[i]    = gridlat[jstar_global]
            source_lon[i]    = gridlon[jstar_global]
            source_misfit[i] = float(misfit_j[jstar_local])
            source_DR[i]     = float(dr_j[jstar_local])
            source_nsta[i]   = int(nused_j[jstar_local])

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

        # Package source (keep DR scale factor for continuity)
        self.source = {
            "t": t,
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
        }

        # Spatial connectedness (same as fast_locate)
        conn = compute_spatial_connectedness(
            source_lat, source_lon, dr=source_DR,
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

        if verbose: print("[ASL] fast_locate: preparing data…")
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)

        # Stream → Y
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        Y = np.vstack([tr.data.astype(np.float32, copy=False) for tr in st])
        t = st[0].times("utcdatetime")
        nsta, ntime = Y.shape

        # Refresh _node_mask from the (possibly replaced) grid each call
        self._node_mask = _grid_mask_indices(self.gridobj)

        # Backend context (honors self._node_mask if present)
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

        # Spatial blur feasibility
        grid_shape = _grid_shape_or_none(self.gridobj)
        can_blur = (
            spatial_smooth_sigma and spatial_smooth_sigma > 0.0 and
            (grid_shape is not None) and (gaussian_filter is not None)
        )

        if verbose:
            print(f"[ASL] fast_locate: nsta={nsta}, ntime={ntime}, batch={batch_size}, "
                f"spatial_blur={'on' if can_blur else 'off'}, "
                f"temporal_smooth_win={temporal_smooth_win or 0}")

        for i0 in range(0, ntime, batch_size):
            i1 = min(i0 + batch_size, ntime)
            if verbose: print(f"[ASL] fast_locate: [{i0}:{i1})")
            Yb = Y[:, i0:i1]

            for k in range(Yb.shape[1]):
                y = Yb[:, k]
                misfit, extras = misfit_backend.evaluate(y, ctx, min_stations=min_stations, eps=eps)

                # Keep full misfit & mean vectors if we’ll need Viterbi later
                collect_for_viterbi = (temporal_smooth_mode.lower() == "viterbi")
                if i0 == 0 and k == 0:
                    misfit_hist = [] if collect_for_viterbi else None
                    mean_hist   = [] if collect_for_viterbi else None

                if collect_for_viterbi:
                    misfit_hist.append(np.asarray(misfit, float).copy())
                    # we expect 'mean' (or 'DR') vector to exist; if missing, stash NaNs (we’ll fallback)
                    mvec = extras.get("mean", None)
                    if mvec is None:
                        mvec = np.full_like(misfit, np.nan, dtype=float)
                    mean_hist.append(np.asarray(mvec, float).copy())

                # Optional per-time spatial smoothing (requires full-grid shape)
                if can_blur:
                    nlat, nlon = grid_shape
                    try:
                        m2 = misfit.reshape(nlat, nlon)

                        # Build validity mask: finite = True
                        valid = np.isfinite(m2)

                        if valid.any():
                            # Replace invalid with a very large finite number so the minimum
                            # never migrates into previously invalid nodes.
                            LARGE = np.nanmax(m2[valid]) if np.isfinite(m2[valid]).any() else 1.0
                            LARGE = float(LARGE) if np.isfinite(LARGE) else 1.0
                            F = np.where(valid, m2, LARGE)

                            # Blur the *finite* costs and the mask separately, then renormalize
                            W = gaussian_filter(valid.astype(float), sigma=spatial_smooth_sigma, mode="nearest")
                            G = gaussian_filter(F,                sigma=spatial_smooth_sigma, mode="nearest")

                            # Avoid division by ~0: where W is tiny (all-invalid neighborhood), keep LARGE
                            with np.errstate(invalid="ignore", divide="ignore"):
                                Mblur = np.where(W > 1e-6, G / np.maximum(W, 1e-6), LARGE)

                            # Re-impose invalid cells as LARGE so they can’t be selected
                            Mblur = np.where(valid, Mblur, LARGE)
                            misfit = Mblur.reshape(-1)
                        else:
                            # everything invalid – leave as is
                            pass
                    except Exception:
                        # silently fall back if sub-grid shape doesn't match full grid
                        pass

                jstar_local = int(np.nanargmin(misfit))
                # Map local index to global grid index if backend subsetted the grid
                jstar_global = jstar_local
                node_index = ctx.get("node_index", None)
                if node_index is not None:
                    jstar_global = int(node_index[jstar_local])

                # DR from backend if available (vector)
                DR_t = extras.get("mean", None)
                if DR_t is None:
                    DR_t = extras.get("DR", None)
                if DR_t is not None and np.ndim(DR_t) == 1:
                    DR_t = DR_t[jstar_local]
                else:
                    # fallback: compute reduced amplitudes at jstar using C if provided
                    C = ctx.get("C", None)
                    if C is not None and C.shape[0] == y.shape[0]:
                        rbest = y * C[:, jstar_local]
                        rbest = rbest[np.isfinite(rbest)]
                        if rbest.size:
                            DR_t = np.median(rbest) if use_median_for_DR else np.mean(rbest)
                        else:
                            DR_t = np.nan
                    else:
                        DR_t = np.nan

                i = i0 + k
                source_DR[i]     = DR_t
                source_lat[i]    = gridlat[jstar_global]
                source_lon[i]    = gridlon[jstar_global]
                source_misfit[i] = float(misfit[jstar_local])
                source_node[i]   = jstar_global

                # Station count at best node (if backend reported N as vector)
                N = extras.get("N", None)
                source_nsta[i] = int(N if np.isscalar(N) else (N[jstar_local] if N is not None else np.sum(np.isfinite(y))))

                azgap, _ = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
                source_azgap[i] = azgap

            flat_lat = self.gridobj.gridlat.ravel()
            flat_lon = self.gridobj.gridlon.ravel()

            mode = (temporal_smooth_mode or "none").lower()
            if mode == "median" and (temporal_smooth_win and temporal_smooth_win >= 3 and temporal_smooth_win % 2 == 1):
                sm_node = _median_filter_indices(source_node, temporal_smooth_win)

                # Optional: small jump guard (keeps pathological flips out)
                try:
                    def _km(j1, j2):
                        return 0.001 * gps2dist_azimuth(
                            float(flat_lat[j1]), float(flat_lon[j1]),
                            float(flat_lat[j2]), float(flat_lon[j2])
                        )[0]
                    JUMP_KM = 5.0
                    for i in range(len(sm_node)):
                        if _km(source_node[i], sm_node[i]) > JUMP_KM:
                            sm_node[i] = source_node[i]
                except Exception:
                    pass

                # Re-derive lat/lon at smoothed nodes; keep DR as (optionally) movavg
                source_lat = flat_lat[sm_node].astype(float, copy=False)
                source_lon = flat_lon[sm_node].astype(float, copy=False)
                # If you still want a gentle DR temporal average:
                if temporal_smooth_win and temporal_smooth_win >= 3 and temporal_smooth_win % 2 == 1:
                    source_DR = _movavg_1d(source_DR, temporal_smooth_win)

            elif mode == "viterbi":
                if misfit_hist is None:
                    # If we somehow didn’t collect, silently skip to no smoothing
                    pass
                else:
                    M = np.vstack(misfit_hist)  # (T, J_subgrid_or_full)
                    # If we used a sub-grid during locate, misfit is already in local-node space,
                    # but the indices we stored in source_node are *global*. That’s OK; the
                    # Viterbi path is chosen in the same (local) space we evaluated M in.
                    sm_local = _viterbi_smooth_indices(
                        misfits_TJ=M,
                        flat_lat=flat_lat if "node_index" not in ctx else flat_lat[ctx["node_index"]],
                        flat_lon=flat_lon if "node_index" not in ctx else flat_lon[ctx["node_index"]],
                        lambda_km=float(viterbi_lambda_km),
                        max_step_km=(None if viterbi_max_step_km is None else float(viterbi_max_step_km)),
                    )
                    # Map local indices back to global if needed
                    node_index = ctx.get("node_index", None)
                    sm_global = sm_local if node_index is None else np.asarray(node_index, int)[sm_local]

                    source_lat = flat_lat[sm_global]
                    source_lon = flat_lon[sm_global]
                    # Re-pick DR at the smoothed node using stored per-time means (if present)
                    try:
                        mean_M = np.vstack(mean_hist)  # (T, J_local)
                        source_DR = np.array([mean_M[t, sm_local[t]] for t in range(mean_M.shape[0])], dtype=float)
                    except Exception:
                        # fallback: keep previously computed DR scalars
                        pass

            else:
                # no temporal smoothing; nothing to do
                pass

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
            source_lat, source_lon, dr=source_DR,
            top_frac=0.15, min_points=12, max_points=200,
        )
        self.connectedness = conn
        if verbose:
            print(f"[ASL] connectedness: score={conn['score']:.3f}  "
                f"n_used={conn['n_used']}  mean_km={conn['mean_km']:.2f}  p90_km={conn['p90_km']:.2f}")

        self.source["connectedness"] = conn["score"]

        self.source_to_obspyevent()
        self.located = True
        if hasattr(self, "set_id"):
            self.id = self.set_id()
        if verbose: print("[ASL] fast_locate: done.")

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
    def apply_node_mask(self, mask: np.ndarray, *, inplace: bool = True):
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
        return self.apply_node_mask(mask, inplace=inplace)

    # --- NEW: coarse→fine without new grid: subset + relocate ---
    def refine_and_relocate(
        self,
        *,
        top_frac: float = 0.20,
        margin_km: float = 1.0,
        mode: str = "mask",                   # "mask" (non-destructive) or "slice" (destructive)
        min_top_points: int = 8,              # lower bound on #points used to make bbox
        # fast_locate() passthroughs / aliases
        misfit_backend=None,
        spatial_smooth_sigma: float | None = None,
        spatial_sigma_nodes: float | None = None,   # alias (from older version)
        temporal_smooth_win: int | None = None,
        min_stations: int | None = None,
        eps: float | None = None,
        use_median_for_DR: bool | None = None,
        batch_size: int | None = None,
        verbose: bool = True,
    ):
        """
        Focus the search around the top-DR portion of the current track and re-run fast_locate().

        Two modes:
        • mode="mask"  (default): apply a temporary node mask to the *current* grid (non-destructive).
        • mode="slice": permanently subset the grid (and ampcorr/distances) to the bbox.

        Notes
        -----
        - No recomputation of amplitude corrections or distances is performed.
        - `spatial_sigma_nodes` is accepted as an alias for `spatial_smooth_sigma`.
        """

        import numpy as np

        if self.source is None:
            raise RuntimeError("No source yet. Run fast_locate()/locate() first.")

        # prefer explicit spatial_smooth_sigma; otherwise honor alias if provided
        if spatial_smooth_sigma is None and spatial_sigma_nodes is not None:
            spatial_smooth_sigma = spatial_sigma_nodes

        # --- gather current track arrays
        lat = np.asarray(self.source["lat"], float)
        lon = np.asarray(self.source["lon"], float)
        dr  = np.asarray(self.source["DR"],  float)

        mask_fin = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(dr)
        n_fin = int(mask_fin.sum())
        if n_fin == 0:
            if verbose:
                print("[ASL] refine_and_relocate: no finite samples; skipping")
            return self

        # choose top-fraction by DR with a sensible minimum count
        idx_all = np.flatnonzero(mask_fin)
        k_frac = int(np.ceil(top_frac * idx_all.size))
        k = max(1, min_top_points, k_frac)
        k = min(k, idx_all.size)

        order = np.argsort(dr[idx_all])[::-1]
        idx_top = idx_all[order[:k]]

        # --- build bounding box with margin (km → deg; lon scale depends on latitude)
        lat_top = lat[idx_top]
        lon_top = lon[idx_top]

        lat0 = float(np.nanmedian(lat_top))
        dlat = float(margin_km) / 111.0
        dlon = float(margin_km) / (111.0 * max(0.1, np.cos(np.deg2rad(lat0))))

        lat_min = float(np.nanmin(lat_top) - dlat)
        lat_max = float(np.nanmax(lat_top) + dlat)
        lon_min = float(np.nanmin(lon_top) - dlon)
        lon_max = float(np.nanmax(lon_top) + dlon)

        if verbose:
            print(f"[ASL] refine_and_relocate: top {k}/{idx_all.size} by DR → "
                f"bbox lat[{lat_min:.5f},{lat_max:.5f}] lon[{lon_min:.5f},{lon_max:.5f}] "
                f"(mode='{mode}')")

        # --- compute node selection on the current grid
        g_lat = self.gridobj.gridlat.reshape(-1)
        g_lon = self.gridobj.gridlon.reshape(-1)
        node_mask = (g_lat >= lat_min) & (g_lat <= lat_max) & (g_lon >= lon_min) & (g_lon <= lon_max)
        sub_idx = np.flatnonzero(node_mask)

        if sub_idx.size == 0:
            if verbose:
                print("[ASL] refine_and_relocate: sub-grid empty; keeping full grid")
            # still run fast_locate to allow smoothing tweaks, if any
            kwargs = {}
            if misfit_backend is not None:        kwargs["misfit_backend"] = misfit_backend
            if spatial_smooth_sigma is not None:  kwargs["spatial_smooth_sigma"] = spatial_smooth_sigma
            if temporal_smooth_win is not None:   kwargs["temporal_smooth_win"]  = temporal_smooth_win
            if min_stations is not None:          kwargs["min_stations"] = min_stations
            if eps is not None:                   kwargs["eps"] = eps
            if use_median_for_DR is not None:     kwargs["use_median_for_DR"] = use_median_for_DR
            if batch_size is not None:            kwargs["batch_size"] = batch_size
            self.fast_locate(verbose=verbose, **kwargs)
            return self

        # --- apply subset (two modes)
        if mode.lower() == "slice":
            # Destructive grid reduction (also slices corrections/distances)
            self.subset_nodes_by_bbox(lat_min, lat_max, lon_min, lon_max, inplace=True)

            kwargs = {}
            if misfit_backend is not None:        kwargs["misfit_backend"] = misfit_backend
            if spatial_smooth_sigma is not None:  kwargs["spatial_smooth_sigma"] = spatial_smooth_sigma
            if temporal_smooth_win is not None:   kwargs["temporal_smooth_win"]  = temporal_smooth_win
            if min_stations is not None:          kwargs["min_stations"] = min_stations
            if eps is not None:                   kwargs["eps"] = eps
            if use_median_for_DR is not None:     kwargs["use_median_for_DR"] = use_median_for_DR
            if batch_size is not None:            kwargs["batch_size"] = batch_size

            self.fast_locate(verbose=verbose, **kwargs)
            return self

        # Default: non-destructive temporary mask
        kwargs = {}
        if misfit_backend is not None:        kwargs["misfit_backend"] = misfit_backend
        if spatial_smooth_sigma is not None:  kwargs["spatial_smooth_sigma"] = spatial_smooth_sigma
        if temporal_smooth_win is not None:   kwargs["temporal_smooth_win"]  = temporal_smooth_win
        if min_stations is not None:          kwargs["min_stations"] = min_stations
        if eps is not None:                   kwargs["eps"] = eps
        if use_median_for_DR is not None:     kwargs["use_median_for_DR"] = use_median_for_DR
        if batch_size is not None:            kwargs["batch_size"] = batch_size

        try:
            self._node_mask = sub_idx  # consumed by misfit backends in fast_locate()
            if verbose:
                frac = 100.0 * sub_idx.size / g_lat.size
                print(f"[ASL] refine_and_relocate: restricting to {sub_idx.size} nodes "
                    f"(~{frac:.1f}% of grid) via temporary mask")
            self.fast_locate(verbose=verbose, **kwargs)
        finally:
            # always clear the mask so future calls run on full grid unless refined again
            self._node_mask = None

        return self
 
# ---------- helpers ----------
def compute_azimuthal_gap(origin_lat: float, origin_lon: float,
                          station_coords: Iterable[Tuple[float, float]]) -> tuple[float, int]:
    """
    Compute the classical azimuthal gap and station count.

    station_coords: iterable of (lat, lon)
    Returns: (max_gap_deg, n_stations)
    """
    azimuths = []
    for stalat, stalon in station_coords:
        _, az, _ = gps2dist_azimuth(origin_lat, origin_lon, stalat, stalon)
        azimuths.append(float(az))

    n = len(azimuths)
    if n < 2:
        return 360.0, n

    azimuths.sort()
    azimuths.append(azimuths[0] + 360.0)
    gaps = [azimuths[i+1] - azimuths[i] for i in range(n)]
    return max(gaps), n

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

    Idea: take the strongest subset of points (by DR), compute the mean
    pairwise great-circle distance (km) among them, and map it to a
    unitless score in (0, 1] where higher means more compact.

    score = 1 / (1 + mean_pairwise_distance_km)

    Parameters
    ----------
    lat, lon : arrays of shape (N,)
        Per-sample lat/lon from ASL (may contain NaN).
    dr : array or None
        Per-sample DR; if provided, selects the top fraction by DR.
        If None, use all finite lat/lon equally.
    top_frac : float
        Fraction (0-1] of samples (by DR) to use. Ignored if dr=None.
    min_points : int
        Minimum number of points to include (if available).
    max_points : int
        Hard cap for pairwise work (keeps O(K^2) reasonable).
    eps_km : float
        Small epsilon to avoid division issues.

    Selection policy
    ----------------
    - select_by="dr"     : take the top fraction by DR (largest values)
    - select_by="misfit" : take the top fraction by *low* misfit (smallest values)
    - select_by="all"    : use all finite (lat,lon) samples, ignore dr/misfit

    Returns
    -------
    dict:
      score       : float in (0, 1], higher = tighter clustering
      n_used      : number of points used
      mean_km     : mean pairwise great-circle distance among selected points
      median_km   : median pairwise distance
      p90_km      : 90th percentile distance
      indices     : indices of selected samples (into the original arrays)

      
    •	The score increases as the track tightens (mean pairwise distance shrinks).
	•	score = 1 / (1 + mean_km) → 1.0 for perfectly colocated, ~0 for very spread.
	•	Using top_frac focuses the metric on the most energetic portion of the track.
	•	self.connectedness is a dict with rich diagnostics; self.source["connectedness"]
        is a convenient scalar for tables/CSV (pandas will broadcast the scalar when building a DataFrame).
	•	If you’d prefer a different mapping (e.g., exp(-mean_km / s0)), it’s a one-liner change.
  
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)

    # base finite mask on coordinates
    base_mask = np.isfinite(lat) & np.isfinite(lon)

    # optional weights/criteria arrays
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
            # fallback: use all valid coords or a single best point if requested
            if fallback_if_empty:
                # single best by whatever we have: try DR, else give up to "all"
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


# run on each event


def asl_sausage(
    stream: Stream,
    event_dir: str,
    asl_config: dict,
    output_dir: str,
    dry_run: bool = False,
    peakf_override: Optional[float] = None,
    station_gains_df: Optional["pd.DataFrame"] = None,  # long/tidy: start_time,end_time,seed_id,gain,…
    allow_station_fallback: bool = True,
):
    """
    Run ASL on a single (already preprocessed) event stream.

    Required in `asl_config`:
      - gridobj : Grid
      - node_distances_km : dict[seed_id -> np.ndarray]
      - station_coords : dict[seed_id -> {latitude, longitude, ...}]
      - ampcorr : AmpCorr
      - vsam_metric, window_seconds, min_stations, interactive, Q, surfaceWaveSpeed_kms

    Optional:
      - station_gains_df : **long-form** DataFrame with columns:
            start_time | end_time | seed_id | gain | (optional: method, n_events, …)
        Behavior:
          * Picks the row whose [start_time, end_time) contains the event start time.
          * Divides each trace by the matching gain for its seed_id.
          * If `allow_station_fallback=True`, tries station-scoped fallbacks:
              NET.STA.LOC.CHA → NET.STA..CHA → NET.STA.*.CHA → NET.STA.*.* → NET.STA
          * Traces without a matching gain are left unchanged.
    """

    print(f"[ASL] Preparing VSAM for event folder: {event_dir}")
    os.makedirs(event_dir, exist_ok=True)

    # --------------------------
    # 1) Apply station gains (interval table, long/tidy)
    # --------------------------
    if station_gains_df is not None and len(station_gains_df):
        try:
            info = apply_interval_station_gains(
                stream,
                station_gains_df,
                allow_station_fallback=allow_station_fallback,
                verbose=True,
            )
            s = info.get("interval_start"); e = info.get("interval_end")
            used = info.get("used", []); miss = info.get("missing", [])
            print(f"[GAINS] Interval used: {s} → {e} | corrected {len(used)} traces; missing {len(miss)}")
        except Exception as e:
            print(f"[GAINS:WARN] Failed to apply interval gains: {e}")
    else:
        print("[GAINS] No station gains DataFrame provided; skipping.")

    # Ensure velocity units for downstream plots (don’t overwrite if already set)
    for tr in stream:
        if tr.stats.get("units") in (None, ""):
            tr.stats["units"] = "m/s"

    # --------------------------
    # 2) Build VSAM
    # --------------------------
    vsamObj = VSAM(stream=stream, sampling_interval=1.0)
    if len(vsamObj.dataframes) == 0:
        raise IOError("[ASL:ERR] No dataframes in VSAM object")

    if not dry_run:
        print("[ASL:PLOT] Writing VSAM preview (VSAM.png)")
        vsamObj.plot(equal_scale=False, outfile=os.path.join(event_dir, "VSAM.png"))
        # close any pyplot figs opened by VSAM.plot to avoid figure buildup
        try:
            plt.close('all')
        except Exception:
            pass

    # --------------------------
    # 3) Decide event peakf
    # --------------------------
    if peakf_override is None:
        freqs = [df.attrs.get("peakf") for df in vsamObj.dataframes.values() if df.attrs.get("peakf") is not None]
        if freqs:
            peakf_event = int(round(sum(freqs) / len(freqs)))
            print(f"[ASL] Event peak frequency inferred from VSAM: {peakf_event} Hz")
        else:
            peakf_event = int(round(asl_config["ampcorr"].params.peakf))
            print(f"[ASL] Using global/default peak frequency from ampcorr: {peakf_event} Hz")
    else:
        peakf_event = int(round(peakf_override))
        print(f"[ASL] Using peak frequency override: {peakf_event} Hz")

    # --------------------------
    # 4) Amplitude corrections cache (swap if peakf differs)
    # --------------------------
    ampcorr: AmpCorr = asl_config["ampcorr"]
    if abs(float(ampcorr.params.peakf) - float(peakf_event)) > 1e-6:
        print(f"[ASL] Switching amplitude corrections to peakf={peakf_event} Hz (from {ampcorr.params.peakf})")

        # Use the grid’s actual signature object and the distance signature you *already* computed
        grid_sig = asl_config["gridobj"].signature()                # works for Grid or NodeGrid
        dist_sig = distances_signature(asl_config["node_distances_km"])
        inv_sig  = tuple(sorted(asl_config["node_distances_km"].keys()))

        params = ampcorr.params
        new_params = AmpCorrParams(
            surface_waves=params.surface_waves,
            wave_speed_kms=params.wave_speed_kms,
            Q=params.Q,
            peakf=float(peakf_event),
            grid_sig=grid_sig,            # ← was wrong before
            inv_sig=inv_sig,
            dist_sig=dist_sig,
            mask_sig=None,
            code_version=params.code_version,
        )

        ampcorr = AmpCorr(new_params, cache_dir=ampcorr.cache_dir)
        ampcorr.compute_or_load(asl_config["node_distances_km"])    # reuse the same (now 3-D) distances
        asl_config[f"ampcorr_peakf_{peakf_event}"] = ampcorr
    else:
        print(f"[ASL] Using existing amplitude corrections (peakf={ampcorr.params.peakf} Hz)")

    # --------------------------
    # 5) Build ASL object and inject geometry/corrections
    # --------------------------
    print("[ASL] Building ASL object…")
    aslobj = ASL(
        vsamObj,
        asl_config["vsam_metric"],
        asl_config["gridobj"],
        asl_config["window_seconds"],
    )
    # If the grid was masked, propagate to the ASL object
    try:
        aslobj._node_mask = _grid_mask_indices(asl_config["gridobj"])
    except Exception:
        pass
    aslobj.node_distances_km   = asl_config["node_distances_km"]
    aslobj.station_coordinates = asl_config["station_coords"]
    aslobj.amplitude_corrections = ampcorr.corrections

    # Params for provenance / filenames
    aslobj.Q = ampcorr.params.Q
    aslobj.peakf = ampcorr.params.peakf
    aslobj.wavespeed_kms = ampcorr.params.wave_speed_kms
    aslobj.surfaceWaves = ampcorr.params.surface_waves

    # --------------------------
    # 6) Locate
    # --------------------------
    # Extra sanity check
    # From asl_config
    station_coords = asl_config["station_coords"]
    meta = asl_config.get("dist_meta", {})

    print(f"[DIST] used_3d={meta.get('used_3d')}  "
        f"has_node_elevations={meta.get('has_node_elevations')}  "
        f"n_nodes={meta.get('n_nodes')}  n_stations={meta.get('n_stations')}")
    z_grid = getattr(asl_config["gridobj"], "node_elev_m", None)
    if z_grid is not None:
        print(f"[GRID] Node elevations: min={np.nanmin(z_grid):.1f} m  max={np.nanmax(z_grid):.1f} m")
    ze = [c.get("elevation", 0.0) for c in station_coords.values()]
    print(f"[DIST] Station elevations: min={min(ze):.1f} m  max={max(ze):.1f} m")

    print("[ASL] Locating source with fast_locate()…")
    try:
        aslobj.fast_locate()
        print("[ASL] Location complete.")
    except Exception:
        print("[ASL:ERR] Location failed.")
        raise

    aslobj.print_event()

    # --------------------------
    # 7) Outputs
    # --------------------------
    if not dry_run:
        qml_out = os.path.join(event_dir, f"event_Q{int(aslobj.Q)}_F{int(peakf_event)}.qml")
        print(f"[ASL:OUT] Writing QuakeML: {qml_out}")
        aslobj.save_event(outfile=qml_out)

        dem_tif_for_bmap = asl_config.get("dem_tif") or asl_config.get("dem_tif_for_bmap")

        try:
            print("[ASL:PLOT] Writing map and diagnostic plots…")
            # Simple basemap, no shading, same DEM as channels if provided
            aslobj.plot(
                zoom_level=0,
                threshold_DR=0.0,
                scale=0.2,
                join=True,
                number=0,
                add_labels=True,
                stations=[tr.stats.station for tr in stream],
                outfile=os.path.join(event_dir, f"map_Q{int(aslobj.Q)}_F{int(peakf_event)}.png"),
                dem_tif=dem_tif_for_bmap,
                simple_basemap=True,
            )
            plt.close('all')

            print("[ASL:SOURCE_TO_CSV] Writing source to a CSV…")
            aslobj.source_to_csv(os.path.join(event_dir, f"source_Q{int(aslobj.Q)}_F{int(peakf_event)}.csv"))

            print("[ASL:PLOT_REDUCED_DISPLACEMENT]")
            aslobj.plot_reduced_displacement(
                outfile=os.path.join(event_dir, f"reduced_disp_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
            )
            plt.close()

            print("[ASL:PLOT_MISFIT]")
            aslobj.plot_misfit(
                outfile=os.path.join(event_dir, f"misfit_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
            )
            plt.close()

            print("[ASL:PLOT_MISFIT_HEATMAP]")
            aslobj.plot_misfit_heatmap(
                outfile=os.path.join(event_dir, f"misfit_heatmap_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
            )
            # PyGMT figures are managed inside; still good to close pyplot
            plt.close('all')

        except Exception:
            print("[ASL:ERR] Plotting failed.")
            raise

        if asl_config.get("interactive", False):
            input("[ASL] Press Enter to continue to next event…")


from types import SimpleNamespace
def _as_regular_view(obj):
    """Small helper to ensure we only use attributes ASL actually touches."""
    return SimpleNamespace(
        id=getattr(obj, "id", "gridview"),
        gridlat=np.asarray(obj.gridlat).reshape(-1),
        gridlon=np.asarray(obj.gridlon).reshape(-1),
        # nlat/nlon are optional; most ASL code uses reshape(-1)
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
    """Return (nlat, nlon) if present/consistent; else None."""
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
        # disallow large jumps by inflating cost to inf
        mask = D > float(max_step_km)
        D = np.where(mask, np.inf, D)

    # DP tables
    dp  = np.full((T, J), np.inf, dtype=float)
    bp  = np.full((T, J), -1,   dtype=int)
    dp[0, :] = misfits_TJ[0, :]

    for t in range(1, T):
        # cost to arrive at node j at time t from all k at t-1
        # dp[t-1, k] + lambda*D[k,j]  → take min over k
        trans = dp[t-1, :, None] + lambda_km * D[None, :, :]  # (J, J)
        kbest = np.nanargmin(trans, axis=0)                   # (J,)
        dp[t, :] = misfits_TJ[t, np.arange(J)] + trans[kbest, np.arange(J)]
        bp[t, :] = kbest

    # backtrack
    jT = int(np.nanargmin(dp[-1, :]))
    path = np.empty(T, dtype=int)
    path[-1] = jT
    for t in range(T-2, -1, -1):
        path[t] = int(bp[t+1, path[t+1]])
    return path


# --- NEW: discover a node mask on the Grid and return GLOBAL index array ---
def _grid_mask_indices(grid) -> "np.ndarray | None":
    """
    Look for a node mask on the Grid and return 1-D array of GLOBAL indices.
    Accepts either a boolean mask with True=keep, or an index array.
    Returns None if no mask is present.
    """
    # common attribute names people pick
    cand = ("node_mask", "mask", "valid_mask", "land_mask")
    raw = None
    for name in cand:
        raw = getattr(grid, name, None)
        if raw is not None:
            break
    if raw is None:
        return None

    raw = np.asarray(raw)
    # allow same shape as grid or flattened
    nn = np.asarray(grid.gridlat).size
    if raw.dtype == bool:
        b = raw.reshape(-1) if raw.size == nn else raw
        if b.size != nn:
            raise ValueError("Grid mask boolean length != number of nodes")
        idx = np.flatnonzero(b).astype(int)
    else:
        idx = raw.reshape(-1).astype(int)
    if idx.size == 0:
        return np.array([], dtype=int)
    return idx