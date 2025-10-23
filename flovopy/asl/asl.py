"""
Amplitude Source Location (ASL) module
======================================

This module implements the ASL class for locating volcanic sources (e.g.,
pyroclastic flows, lahars, rockfalls) using amplitude-based inversion of
seismic network data. It is part of the `flovopy` package.

Overview
--------
The ASL workflow combines:
- Precomputed station–node distances (km) on a regular or masked grid
- Amplitude correction tables (frequency, attenuation, path effects)
- VSAM (Velocity Seismic Amplitude Measurement) time series per station
- Misfit evaluation across candidate nodes
- Fast search refinements (bounding box or sector masking)
- Output in QuakeML, CSV, and diagnostic plots

Design
------
- Public methods (`locate`, `fast_locate`, `refine_and_relocate`, `plot`, …)
  are intended for users.
- Internal/private helpers begin with an underscore (`_`) and encapsulate
  low-level operations (masking, smoothing, pathfinding, grid utilities).
- All methods update the object in-place unless explicitly documented.
"""

from __future__ import annotations

# stdlib
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
#from contextlib import contextmanager
from obspy import Stream, UTCDateTime
import numpy as np
from pathlib import Path

from obspy.core.event import Catalog

# third-party

import matplotlib.pyplot as plt
import pandas as pd


from obspy.geodetics.base import gps2dist_azimuth

# SciPy (guarded: locate/fast_locate accept None and skip blur;
#          metric2stream needs uniform_filter1d; we’ll set None if unavailable)

try:
    from scipy.ndimage import gaussian_filter, uniform_filter1d
except Exception:
    gaussian_filter = None
    uniform_filter1d = None

# PyGMT (guarded for environments without GMT runtime)
try:
    import pygmt
except Exception:
    pygmt = None

# flovopy locals
#from flovopy.processing.sam import VSAM
#from flovopy.asl.ampcorr import AmpCorr, AmpCorrParams
from flovopy.asl.distances import compute_azimuthal_gap #, geo_distance_3d_km, compute_distances, distances_signature, 
from flovopy.utils.make_hash import make_hash
#from flovopy.asl.grid import NodeGrid, Grid
from flovopy.asl.utils import compute_spatial_connectedness, _grid_shape_or_none, _median_filter_indices, _viterbi_smooth_indices, _as_regular_view, _movavg_1d, _movmed_1d

# plotting helpers (topo base, diagnostic heatmap)
from flovopy.asl.map import topo_map
from flovopy.asl.misfit import plot_misfit_heatmap_for_peak_DR, StdOverMeanMisfit #,  R2DistanceMisfit  # (import when needed)

# for inverse locate
from scipy.optimize import minimize
from flovopy.core.mvo import dome_location
from flovopy.asl.grid import _meters_per_degree

# ----------------------------------------------------------------------
# Main ASL class
# ----------------------------------------------------------------------




# assume you already have these somewhere in your codebase
# from flovopy.asl.config import ASLConfig
# from flovopy.processing.sam import VSAM, DSAM
# from flovopy.utils.hashing import make_hash   # or any stable hasher you use

class ASL:
    """
    Amplitude Source Location (ASL) engine.

    This version keeps a *single* reference to an ASLConfig instance (`cfg`)
    and exposes all configuration/derived arrays via properties. Nothing is
    copied; large artifacts like distances and amplitude-corrections are
    accessed directly from `cfg`.

    Parameters
    ----------
    samobject : object
        A VSAM/DSAM instance (must match cfg.sam_class).
    cfg : ASLConfig
        A fully built configuration (cfg.build() must have been called),
        or a config that can be built on-demand.

    Notes
    -----
    - We validate that `cfg` is built (ampcorr & distances present). If not,
      we call `cfg.build()` (which is idempotent).
    - To minimize downstream changes, this class exposes attributes commonly
      used before (e.g., `metric`, `window_seconds`, `amplitude_corrections`)
      as *properties* that delegate to `cfg`.
    """
    def __init__(self, samobject, cfg):
        # Type/consistency checks
        if getattr(cfg, "sam_class", None) is not None and not isinstance(samobject, cfg.sam_class):
            raise TypeError(f"samobject must be an instance of {cfg.sam_class.__name__}")
        self.samobject = samobject
        self.cfg = cfg

        # Ensure config is "built" (distances, ampcorr, etc.)
        if getattr(self.cfg, "ampcorr", None) is None or getattr(self.cfg, "node_distances_km", None) is None:
            # This is safe/idempotent — build() will fill the derived fields.
            self.cfg.build()

        # Active node mask (indices); if the grid already carries a mask,
        # we use that as the default "active indexer".
        self._node_mask: Optional[np.ndarray] = None
        self._idx_active = getattr(self.cfg.gridobj, "_node_mask_idx", None)
        self.id = self._make_id()

        # Runtime outputs
        self.source: Optional[Dict[str, Any]] = None
        self.connectedness: Optional[Dict[str, Any]] = None

    # ------------------ Lightweight property pass-throughs ------------------

    @property
    def starttime(self) -> Optional[UTCDateTime]:
        return self.samobject.starttime
    
    @property
    def endtime(self) -> Optional[UTCDateTime]:
        return self.samobject.endtime    
    
    @property
    def duration(self) -> Optional[float]:
        return self.samobject.duration
    
    @property
    def metric(self) -> str:
        return self.cfg.sam_metric

    @property
    def gridobj(self):
        return self.cfg.gridobj

    @property
    def window_seconds(self) -> float:
        return float(self.cfg.window_seconds)

    @property
    def node_distances_km(self) -> Dict[str, np.ndarray]:
        return self.cfg.node_distances_km

    @property
    def amplitude_corrections(self) -> Dict[str, np.ndarray]:
        # Provided by AmpCorr instance inside cfg
        return self.cfg.ampcorr.corrections

    @property
    def station_coordinates(self) -> Dict[str, dict]:
        return self.cfg.station_coords

    @property
    def Q(self) -> int:
        return int(self.cfg.ampcorr.params.Q)

    @property
    def peakf(self) -> float:
        return float(self.cfg.ampcorr.params.peakf)

    @property
    def wave_speed_kms(self) -> float:
        return float(self.cfg.ampcorr.params.wave_speed_kms)

    @property
    def assume_surface_waves(self) -> bool:
        return bool(self.cfg.ampcorr.params.assume_surface_waves)

    # ------------------ Internal helpers ------------------

    def _make_id(self) -> str:
        """
        Build a stable ID for this ASL instance based on the *configuration*.
        We avoid hashing large arrays; we use their keys/ids instead.
        """
        corr_keys = tuple(sorted(self.amplitude_corrections.keys()))
        grid_id   = getattr(self.gridobj, "id", None) or self.cfg.tag_str
        # Use your existing stable hash function
        return make_hash(self.metric, self.window_seconds, grid_id, corr_keys)

    def _active_indexer(self):
        """
        Return the active subset of node indices to use.
        If a runtime mask is set (`self._node_mask`), it takes precedence;
        otherwise we fall back to any mask already on the grid object;
        otherwise `slice(None)` to use all nodes.
        """
        if self._node_mask is not None and np.size(self._node_mask):
            return self._node_mask
        if self._idx_active is not None and np.size(self._idx_active):
            return self._idx_active
        return slice(None)

    '''
    def recompute_ampcorr(self, peakf: float | None = None):
        """
        Recompute amplitude corrections for a new peak frequency.

        Parameters
        ----------
        peakf : float, optional
            Override frequency in Hz. If None, uses cfg.ampcorr.params.peakf.
        """
        params = self.cfg.ampcorr.params
        new_params = AmpCorrParams(
            assume_surface_waves=params.assume_surface_waves,
            wave_speed_kms=params.wave_speed_kms,
            Q=params.Q,
            peakf=float(peakf if peakf is not None else params.peakf),
            grid_sig=self.cfg.gridobj.signature(),
            inv_sig=tuple(sorted(self.cfg.node_distances_km.keys())),
            dist_sig=distances_signature(self.cfg.node_distances_km),
            mask_sig=None,
            code_version=params.code_version,
        )
        self.cfg.ampcorr = AmpCorr(new_params, cache_dir=self.cfg.ampcorr.cache_dir)
        self.cfg.ampcorr.compute_or_load(self.cfg.node_distances_km, inventory=self.cfg.inventory)
        return self.cfg.ampcorr
    '''


    def metric2stream(self) -> Stream:
        """
        Convert the selected VSAM metric into an ObsPy :class:`~obspy.core.stream.Stream`.

        The returned stream contains one trace per station/channel with the chosen
        metric as the data array. If ``self.window_seconds`` exceeds the native
        sampling interval, a **NaN-aware moving average** of length
        ``round(window_seconds / dt)`` samples is applied trace-by-trace using
        ``scipy.ndimage.uniform_filter1d`` over a finite mask (so missing values do
        not depress the average). Sampling rates are preserved.

        Returns
        -------
        obspy.Stream
            Stream with metric data (dtype float32) and corrected/smoothed samples.

        Notes
        -----
        - The sampling interval is taken from ``self.samobject.sampling_interval`` when
        available and valid; otherwise it is inferred from the first trace's
        ``stats.sampling_rate``.
        - When smoothing is active, each trace is processed independently with its own
        local sampling interval, so heterogeneous sampling rates are handled.
        - The smoothing window length per trace is:
            ``w_tr = max(2, int(round(window_seconds / dt_tr)))``.
        If ``window_seconds <= dt_tr`` for a trace, that trace is left untouched.
        - This method does not change timing metadata (starttime, etc.). If you need
        to reset start/end times, use :class:`~obspy.UTCDateTime` for consistency.
        """
        # Build stream from VSAM
        st = self.samobject.to_stream(metric=self.metric)
        if not isinstance(st, Stream) or len(st) == 0:
            raise RuntimeError("[ASL] metric2stream(): VSAM produced an empty or invalid Stream.")

        # Determine a representative dt from VSAM or fall back to the first trace
        dt = float(getattr(self.samobject, "sampling_interval", np.nan))
        if not np.isfinite(dt) or dt <= 0:
            fs0 = float(getattr(st[0].stats, "sampling_rate", 0.0) or 0.0)
            if fs0 <= 0:
                raise RuntimeError("[ASL] metric2stream(): cannot infer sampling interval (fs<=0).")
            dt = 1.0 / fs0

        win_sec = float(self.window_seconds or 0.0)
        if win_sec <= 0:
            # No smoothing requested; ensure data are float32 for downstream math
            for tr in st:
                if tr.data.dtype != np.float32:
                    tr.data = np.asarray(tr.data, dtype=np.float32)
            return st

        # Apply NaN-aware moving average if the window exceeds the nominal dt
        if win_sec > dt:
            for tr in st:
                # Per-trace sampling
                fs_tr = float(getattr(tr.stats, "sampling_rate", 0.0) or 0.0)
                if fs_tr <= 0:
                    tr.data = np.asarray(tr.data, dtype=np.float32)
                    continue

                dt_tr = 1.0 / fs_tr
                if win_sec <= dt_tr:
                    tr.data = np.asarray(tr.data, dtype=np.float32)
                    continue

                w_tr = max(2, int(round(win_sec / dt_tr)))

                # NaN-aware averaging via finite-mask normalization
                x = np.asarray(tr.data, dtype=np.float32)
                finite_mask = np.isfinite(x).astype(np.float32)
                x_finite = np.where(np.isfinite(x), x, 0.0).astype(np.float32, copy=False)

                num = uniform_filter1d(x_finite, size=w_tr, mode="nearest")
                den = uniform_filter1d(finite_mask, size=w_tr, mode="nearest")

                with np.errstate(invalid="ignore", divide="ignore"):
                    y = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

                tr.data = y
                tr.stats.sampling_rate = fs_tr
                # If you need to update time metadata explicitly, use UTCDateTime:
                # tr.stats.starttime = UTCDateTime(tr.stats.starttime)

        else:
            # Window not greater than nominal dt → just ensure float32 dtype
            for tr in st:
                tr.data = np.asarray(tr.data, dtype=np.float32)

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

        - Iterates over each time sample and every (possibly masked) grid node.
        - Misfit per node: std(reduced) / (|mean(reduced)| + eps),
          where reduced = y * C[:, j].
        - DR(t): median(reduced) if use_median_for_DR else mean(reduced)
          at the best node.
        - Optional spatial smoothing: Gaussian filter applied to the per-time
          misfit surface with NaN/invalid-aware normalization.
        - Optional temporal smoothing: moving average/median applied to tracks.
        - Computes azimuthal gap and spatial connectedness score.
        - Records chosen GLOBAL node indices in source["node_index"]
          (to match fast_locate).

        Parameters
        ----------
        min_stations : int, default 3
            Minimum number of stations required per node to evaluate misfit.
        eps : float, default 1e-9
            Numerical stabilizer to avoid division by zero.
        use_median_for_DR : bool, default False
            Whether to use median instead of mean when computing DR.
        spatial_smooth_sigma : float, default 0.0
            Sigma for Gaussian blur in node-units (0 disables).
        temporal_smooth_win : int, default 0
            Window length for temporal smoothing (0 disables).
        verbose : bool, default True
            Print progress messages.

        Side Effects
        ------------
        Sets self.source (dict) and self.connectedness (dict).
        """
        if verbose:
            print("[ASL] locate (slow): preparing data…")

        cfg = self.cfg  # shorthand

        # Grid vectors
        gridlat = cfg.gridobj.gridlat.reshape(-1)
        gridlon = cfg.gridobj.gridlon.reshape(-1)
        nnodes  = gridlat.size

        # Apply mask if present
        node_index = cfg.gridobj.get_mask_indices()
        if node_index is not None and node_index.size > 0:
            gridlat_local = gridlat[node_index]
            gridlon_local = gridlon[node_index]
            nnodes_local  = int(node_index.size)
        else:
            node_index    = None
            gridlat_local = gridlat
            gridlon_local = gridlon
            nnodes_local  = nnodes

        # Stream → Y
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        Y = np.vstack([tr.data.astype(np.float32, copy=False) for tr in st])
        t = st[0].times("utcdatetime")  # already UTCDateTime objects
        nsta, ntime = Y.shape

        # Corrections matrix C
        C = np.empty((nsta, nnodes_local), dtype=np.float32)
        for k, sid in enumerate(seed_ids):
            if sid not in cfg.ampcorr.corrections:
                raise KeyError(f"Missing amplitude corrections for channel '{sid}'")
            vec = np.asarray(cfg.ampcorr.corrections[sid], dtype=np.float32)
            if vec.size != nnodes:
                raise ValueError(f"Corrections length mismatch for {sid}: {vec.size} != {nnodes}")
            C[k, :] = vec[node_index] if node_index is not None else vec

        # Station coords for azimuthal gap
        station_coords = []
        for sid in seed_ids:
            coords = cfg.station_coords.get(sid)
            if coords:
                station_coords.append((coords["latitude"], coords["longitude"]))

        # Outputs
        source_DR     = np.empty(ntime, dtype=float)
        source_lat    = np.empty(ntime, dtype=float)
        source_lon    = np.empty(ntime, dtype=float)
        source_misfit = np.empty(ntime, dtype=float)
        source_azgap  = np.empty(ntime, dtype=float)
        source_nsta   = np.empty(ntime, dtype=int)
        source_node   = np.empty(ntime, dtype=int)

        # Grid reshape check
        def _grid_shape_or_none(grid):
            try:
                return int(grid.nlat), int(grid.nlon)
            except Exception:
                return None

        grid_shape = _grid_shape_or_none(cfg.gridobj)
        can_blur = (
            spatial_smooth_sigma > 0.0
            and (grid_shape is not None)
            and (nnodes_local == (grid_shape[0] * grid_shape[1]))
        )

        if verbose:
            print(f"[ASL] locate (slow): nsta={nsta}, ntime={ntime}, "
                  f"spatial_blur={'on' if can_blur else 'off'}, "
                  f"temporal_smooth_win={temporal_smooth_win or 0}")

        # Time loop
        for i in range(ntime):
            y = Y[:, i]

            misfit_j = np.full(nnodes_local, np.inf, dtype=float)
            dr_j     = np.full(nnodes_local, np.nan, dtype=float)
            nused_j  = np.zeros(nnodes_local, dtype=int)

            # Node loop
            for j in range(nnodes_local):
                reduced = y * C[:, j]
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

            # Spatial smoothing
            if can_blur:
                nlat, nlon = grid_shape
                try:
                    m2 = misfit_j.reshape(nlat, nlon)
                    valid = np.isfinite(m2)

                    if valid.any():
                        LARGE = float(np.nanmax(m2[valid]) if np.isfinite(m2[valid]).any() else 1.0)
                        F = np.where(valid, m2, LARGE)

                        W = gaussian_filter(valid.astype(float), sigma=spatial_smooth_sigma, mode="nearest")
                        G = gaussian_filter(F, sigma=spatial_smooth_sigma, mode="nearest")

                        with np.errstate(invalid="ignore", divide="ignore"):
                            Mblur = np.where(W > 1e-6, G / np.maximum(W, 1e-6), LARGE)

                        Mblur = np.where(valid, Mblur, LARGE)
                        misfit_for_pick = Mblur.reshape(-1)
                    else:
                        misfit_for_pick = misfit_j
                except Exception:
                    misfit_for_pick = misfit_j
            else:
                misfit_for_pick = misfit_j

            # Pick best node
            if not np.isfinite(misfit_for_pick).any():
                source_lat[i] = np.nan
                source_lon[i] = np.nan
                source_misfit[i] = np.nan
                source_DR[i] = np.nan
                source_nsta[i] = 0
                source_node[i] = -1
                source_azgap[i] = np.nan
                continue

            jstar_local  = int(np.nanargmin(misfit_for_pick))
            jstar_global = int(node_index[jstar_local]) if node_index is not None else jstar_local

            source_lat[i]    = gridlat[jstar_global]
            source_lon[i]    = gridlon[jstar_global]
            source_misfit[i] = float(misfit_j[jstar_local])
            source_DR[i]     = float(dr_j[jstar_local])
            source_nsta[i]   = int(nused_j[jstar_local])
            source_node[i]   = jstar_global

            azgap, _ = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
            source_azgap[i] = azgap

        # Temporal smoothing
        w = int(temporal_smooth_win or 0)
        if w >= 3:
            if w % 2 == 0:
                w += 1
            source_lat = _movmed_1d(source_lat, w)
            source_lon = _movmed_1d(source_lon, w)
            source_DR  = _movavg_1d(source_DR,  w)

        # Package source
        self.source = {
            "t": t,  # UTCDateTime array
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
            "node_index": source_node,
        }

        self.connectedness = compute_spatial_connectedness(
            self.source["lat"], self.source["lon"], dr=source_DR,
            top_frac=0.15, min_points=12, max_points=200,
        )
        self.source["connectedness"] = self.connectedness["score"]

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
        viterbi_max_step_km: float = 1.0,   # km hard cutoff for viterbi steps
        candidate_idx: np.ndarray | list | None = None,   # indices (into full grid) to evaluate
        verbose: bool = True,
    ):
        """
        Vectorized (fast) amplitude‐based locator on a fixed spatial grid.

        This method locates a time-varying seismic source by minimizing a node-wise
        misfit across the grid using a chosen backend. It uses geometry and
        amplitude-correction information precomputed in the bound :class:`ASLConfig`
        (``self.cfg``), including:

        - Grid geometry (``self.cfg.gridobj``),
        - Station→node distance tables (``self.cfg.node_distances_km``),
        - Per-station amplitude correction vectors (``self.cfg.ampcorr.corrections``),
        - Station coordinates (``self.cfg.station_coords``).

        Time is handled in absolute UTC using ObsPy’s ``UTCDateTime``; the returned
        track contains a per-sample ``UTCDateTime`` array.

        Parameters
        ----------
        misfit_backend : object or None, optional
            A misfit backend implementing the interface::

                ctx = backend.prepare(aslobj, seed_ids, dtype)
                misfit, extras = backend.evaluate(y, ctx, min_stations, eps)

            If ``None``, uses :class:`StdOverMeanMisfit`.
            Backends may leverage amplitude corrections and/or distances via
            the context ``ctx`` created in ``prepare``.
        min_stations : int, default 3
            Require at least this many usable stations at a node for it to be
            considered (applies per time frame).
        eps : float, default 1e-9
            Small positive constant to stabilize divisions in certain backends.
        use_median_for_DR : bool, default False
            When a DR (reduced amplitude) vector is available but not provided
            by the backend, compute DR at the winning node using the
            median (True) or mean (False) of station reductions.
        batch_size : int, default 1024
            Number of time samples to process per inner loop. Controls speed/memory;
            does not affect the result.
        spatial_smooth_sigma : float, default 0.0
            If > 0, apply a Gaussian blur (sigma given **in grid cells**) to each
            frame’s misfit map *before* picking the best node. Improves stability
            in noisy cases. Requires a rectangular grid and ``scipy`` available.
        temporal_smooth_mode : {"none", "median", "viterbi"}, default "none"
            Post-selection smoothing of the node track:
            - "none": raw per-frame minima.
            - "median": centered median filter on indices; robust to outliers.
            - "viterbi": dynamic-programming path optimization with movement penalty.
        temporal_smooth_win : int, default 5
            For "median", the odd window length. For "viterbi", the planning
            horizon used internally by the smoother (implementation-dependent).
        viterbi_lambda_km : float, default 5.0
            Movement penalty (km cost per step) used by the "viterbi" smoother.
            Larger values enforce smoother, slower tracks.
        viterbi_max_step_km : float, default 25.0
            Hard cap on per-frame movement in "viterbi" mode (disallow larger jumps).
        verbose : bool, default True
            Print progress and diagnostics.

        What it does (high level)
        -------------------------
        1) Converts the selected VSAM metric to a station×time matrix ``Y`` via
        :meth:`metric2stream`. Times come back as an array of ``UTCDateTime``.
        2) For each time step, evaluates the node-wise misfit using the selected
        backend and the precomputed amplitude corrections (and distances if needed).
        3) Picks the best node; extracts DR and azimuthal gap at that node.
        4) Optional spatial smoothing (Gaussian blur) on the misfit surface per frame.
        5) Optional temporal smoothing of the resulting node track (median/Viterbi).
        6) Stores the source track and a connectedness score on ``self`` for downstream
        plotting, export, and cataloging.

        Misfit backends
        ---------------
        Backends are pluggable. Two canonical choices:

        **StdOverMeanMisfit (default; stable, fast)**
        - For node ``j`` with station reductions ``R[:, j] = y * C[:, j]``:
            ``misfit_j = std(R[:, j]) / (|mean(R[:, j])| + eps)``.
        - Requires only amplitude corrections ``C``; robust when C captures
            geometry + Q reasonably well.

        **R2DistanceMisfit (distance-aware)**
        - Correlates station amplitudes (or ``log|y|``) with a distance feature
            derived from station→node distances to compute an R² and uses ``1 - R²``
            (optionally blended with StdOverMean).
        - Requires distances ``self.cfg.node_distances_km`` and uses corrections
            ``C`` as well.

        Spatial smoothing (per-frame)
        -----------------------------
        If ``spatial_smooth_sigma > 0`` and the grid is rectangular, each frame’s
        misfit map is blurred with a Gaussian (sigma in **grid cells**). Invalid
        cells are handled with a mask-aware normalization so the minimum doesn’t
        drift into NaNs.

        Temporal smoothing (track-level)
        --------------------------------
        - **"median"**: centered odd-length median filter on indices; DR is smoothed
        with a gentle moving average.
        - **"viterbi"**: optimal path under cost =
        ``per-frame misfit + λ * step_distance_km``, with an optional hard cap
        on step size.

        Returns (via object state)
        --------------------------
        Sets these attributes on ``self``:

        ``self.source`` : dict
            - ``"t"`` : array of ``UTCDateTime``
            - ``"lat"``, ``"lon"`` : best-node coordinates per time
            - ``"DR"`` : reduced amplitude at best node (scaled by 1e7 for plotting)
            - ``"misfit"`` : misfit value at best node
            - ``"azgap"`` : azimuthal gap at best node
            - ``"nsta"`` : usable station count per time
            - ``"node_index"`` : chosen **global** node indices (full-grid indexing)
            - ``"connectedness"`` : scalar in [0, 1] describing spatial compactness
        ``self.connectedness`` : dict
            Full diagnostics from the connectedness computation.
        ``self.located`` : bool
            Set ``True`` on success.

        Notes
        -----
        - All geometry/corrections come from ``self.cfg`` (an :class:`ASLConfig`).
        This method **does not** recompute distances or amplitude corrections.
        - If your grid provides a persistent node mask, it’s honored each call.
        - The DR definition used for plotting/exports matches the slow locator.

        Examples
        --------
        # 1) Default backend, no smoothing
        >>> asl.fast_locate()

        # 2) Distance-aware backend (pure R², log correlation)
        >>> from flovopy.asl.misfit import R2DistanceMisfit
        >>> asl.fast_locate(misfit_backend=R2DistanceMisfit(use_log=True, alpha=1.0))

        # 3) Spatial + median temporal smoothing
        >>> asl.fast_locate(spatial_smooth_sigma=1.5,
        ...                 temporal_smooth_mode="median",
        ...                 temporal_smooth_win=5)

        # 4) Viterbi smoothing with moderate movement penalty
        >>> asl.fast_locate(temporal_smooth_mode="viterbi",
        ...                 temporal_smooth_win=7,
        ...                 viterbi_lambda_km=8.0,
        ...                 viterbi_max_step_km=30.0)

        Changes:

        Now supports an optional `candidate_idx` sub-grid mapping and exports the
        misfit vector for the frame with peak DR for plotting heatmaps after
        refinement.

        Side-effects (unchanged + new):
        self.source                : dict with track arrays
        self.connectedness         : dict
        self.located               : bool
        self._last_eval_idx        : np.ndarray[int] (global node indices evaluated)
        self._last_misfit_vec      : np.ndarray[float] (misfit at _last_eval_idx for peak-DR frame)
        self._last_misfit_frame    : int (frame index where DR peaked)
        """
        
        cfg = self.cfg
        if misfit_backend is None:
            misfit_backend = StdOverMeanMisfit()

        if verbose:
            print("[ASL] fast_locate: preparing data…")

        # Grid vectors (global)
        gridlat = cfg.gridobj.gridlat.reshape(-1)
        gridlon = cfg.gridobj.gridlon.reshape(-1)
        n_nodes_full = gridlat.size

        # --- Candidate (sub-grid) mapping ---
        # Order of precedence:
        # 1) explicit candidate_idx, else
        # 2) grid mask (if any), else
        # 3) full grid
        if candidate_idx is not None:
            node_index = np.asarray(candidate_idx, dtype=int).ravel()
        else:
            mask_idx = cfg.gridobj.get_mask_indices()
            if mask_idx is not None and np.size(mask_idx) > 0:
                node_index = np.asarray(mask_idx, dtype=int).ravel()
            else:
                node_index = np.arange(n_nodes_full, dtype=int)

        # Stream → Y
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        Y = np.vstack([tr.data.astype(np.float32, copy=False) for tr in st])
        t = st[0].times("utcdatetime")
        nsta, ntime = Y.shape

        # Refresh grid-provided mask for legacy compatibility
        self._node_mask = cfg.gridobj.get_mask_indices()

        # Backend context (uses config + mapping)
        ctx = misfit_backend.prepare(self, seed_ids, dtype=np.float32)
        ctx["node_index"] = node_index  # <— expose mapping to backend/downstream

        # Station coords (for azgap)
        station_coords = []
        for sid in seed_ids:
            c = cfg.station_coords.get(sid)
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
        grid_shape = _grid_shape_or_none(cfg.gridobj)
        can_blur = (
            spatial_smooth_sigma > 0.0 and
            (grid_shape is not None) and
            (gaussian_filter is not None)
        )

        if verbose:
            print(f"[ASL] fast_locate: nsta={nsta}, ntime={ntime}, batch={batch_size}, "
                f"spatial_blur={'on' if can_blur else 'off'}, "
                f"temporal_smooth={temporal_smooth_mode}, "
                f"temporal_win={temporal_smooth_win}, "
                f"nodes_eval={node_index.size}/{n_nodes_full}")

        # Viterbi support
        want_viterbi = (temporal_smooth_mode or "none").lower() == "viterbi"
        misfit_hist = [] if want_viterbi else None
        mean_hist   = [] if want_viterbi else None

        # --- Track the frame with maximum DR and keep its misfit vector ---
        _best_DR_val = -np.inf
        _best_DR_idx = None
        _best_misfit_vec_local = None  # local (length == len(node_index))

        for i0 in range(0, ntime, batch_size):
            i1 = min(i0 + batch_size, ntime)
            if verbose:
                print(f"[ASL] fast_locate: batch {i0}:{i1}")
            Yb = Y[:, i0:i1]

            for k in range(Yb.shape[1]):
                y = Yb[:, k]

                # Evaluate misfit over the candidate set
                misfit, extras = misfit_backend.evaluate(y, ctx, min_stations=min_stations, eps=eps)
                misfit = np.asarray(misfit, float).ravel()  # shape == len(node_index)

                # Collect history for Viterbi (local-node domain)
                if want_viterbi:
                    misfit_hist.append(misfit.copy())
                    mvec = extras.get("mean", extras.get("DR", None))
                    if mvec is None:
                        mvec = np.full_like(misfit, np.nan, dtype=float)
                    mean_hist.append(np.asarray(mvec, float).ravel().copy())

                # Optional spatial smoothing (requires rectangular grid)
                if can_blur:
                    nlat, nlon = grid_shape
                    try:
                        # Build a dense misfit frame and blur only the candidate cells
                        M_full = np.full(n_nodes_full, np.nan, dtype=float)
                        M_full[node_index] = misfit
                        m2 = M_full.reshape(nlat, nlon)
                        valid = np.isfinite(m2)
                        if valid.any():
                            LARGE = float(np.nanmax(m2[valid]) if np.isfinite(m2[valid]).any() else 1.0)
                            F = np.where(valid, m2, LARGE)
                            W = gaussian_filter(valid.astype(float), sigma=spatial_smooth_sigma, mode="nearest")
                            G = gaussian_filter(F, sigma=spatial_smooth_sigma, mode="nearest")
                            with np.errstate(invalid="ignore", divide="ignore"):
                                Mblur = np.where(W > 1e-6, G / np.maximum(W, 1e-6), LARGE)
                            Mblur = np.where(valid, Mblur, LARGE)
                            # Pull back the smoothed values ONLY at the candidate nodes
                            misfit = Mblur.reshape(-1)[node_index]
                    except Exception:
                        pass

                i = i0 + k
                if not np.isfinite(misfit).any():
                    source_lat[i] = np.nan
                    source_lon[i] = np.nan
                    source_misfit[i] = np.nan
                    source_node[i] = -1
                    source_DR[i] = np.nan
                    source_nsta[i] = int(np.sum(np.isfinite(y)))
                    source_azgap[i] = np.nan
                    continue

                jstar_local = int(np.nanargmin(misfit))
                jstar_global = int(node_index[jstar_local])  # local → global

                # DR at the winning node
                DR_t = extras.get("mean", extras.get("DR", None))
                if DR_t is not None and np.ndim(DR_t) == 1:
                    DR_t = DR_t[jstar_local]
                else:
                    C = ctx.get("C", None)
                    if C is not None and C.shape[0] == y.shape[0]:
                        rbest = y * C[:, jstar_local]
                        rbest = rbest[np.isfinite(rbest)]
                        DR_t = (np.median(rbest) if use_median_for_DR else np.mean(rbest)) if rbest.size else np.nan
                    else:
                        DR_t = np.nan

                source_DR[i]     = DR_t
                source_lat[i]    = gridlat[jstar_global]
                source_lon[i]    = gridlon[jstar_global]
                source_misfit[i] = float(misfit[jstar_local])
                source_node[i]   = jstar_global

                N = extras.get("N", None)
                source_nsta[i] = int(N if np.isscalar(N) else (N[jstar_local] if N is not None else np.sum(np.isfinite(y))))
                azgap, _ = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
                source_azgap[i] = azgap

                # Keep the misfit vector for the frame with maximal DR
                if np.isfinite(DR_t) and DR_t > _best_DR_val:
                    _best_DR_val = float(DR_t)
                    _best_DR_idx = int(i)
                    _best_misfit_vec_local = misfit.copy()  # shape == len(node_index)

        # --- Temporal smoothing ---
        flat_lat = gridlat
        flat_lon = gridlon
        mode = (temporal_smooth_mode or "none").lower()

        if mode == "median" and temporal_smooth_win and temporal_smooth_win >= 3 and temporal_smooth_win % 2 == 1:
            sm_node = _median_filter_indices(source_node, temporal_smooth_win)
            source_lat = flat_lat[sm_node].astype(float, copy=False)
            source_lon = flat_lon[sm_node].astype(float, copy=False)
            source_DR  = _movavg_1d(source_DR, temporal_smooth_win)

        elif mode == "viterbi" and want_viterbi and misfit_hist:
            # Histories are in the local-node domain; supply lat/lon in local too
            M = np.vstack(misfit_hist)                        # (T, J_local)
            flat_lat_local = flat_lat[node_index]             # (J_local,)
            flat_lon_local = flat_lon[node_index]             # (J_local,)

            sm_local = _viterbi_smooth_indices(
                misfits_TJ=M,
                flat_lat=flat_lat_local,
                flat_lon=flat_lon_local,
                lambda_km=float(viterbi_lambda_km),
                max_step_km=(None if viterbi_max_step_km is None else float(viterbi_max_step_km)),
            )
            sm_global = node_index[np.asarray(sm_local, dtype=int)]
            source_lat = flat_lat[sm_global]
            source_lon = flat_lon[sm_global]
            try:
                mean_M = np.vstack(mean_hist)                # (T, J_local)
                source_DR = np.array([mean_M[t, sm_local[t]] for t in range(mean_M.shape[0])], dtype=float)
            except Exception:
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
            "node_index": source_node,  # global indices per frame
        }

        # Connectedness
        conn = compute_spatial_connectedness(
            self.source["lat"], self.source["lon"], dr=source_DR,
            top_frac=0.15, min_points=12, max_points=200,
        )
        self.connectedness = conn
        self.source["connectedness"] = conn["score"]

        # --- Export evaluated-node mapping + misfit for the peak-DR frame ---
        self._last_eval_idx = node_index                               # global indices evaluated this run
        self._last_misfit_vec = _best_misfit_vec_local                 # len == len(node_index)
        self._last_misfit_frame = _best_DR_idx                         # int or None

        self.located = True
        if hasattr(self, "set_id"):
            self.id = self.set_id()
        if verbose:
            print("[ASL] fast_locate: done.")


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
        # Application mode (kept for API; always temporary now)
        mode: str = "mask",
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
        • any persistent mask carried by the grid object (e.g., land-only, streams+dome mask), and  
        • any temporary mask currently active in ``self._node_mask`` (legacy support).

        The mask is then applied **temporarily** to the grid for a new
        :meth:`fast_locate` run. The original mask state of the grid is always
        restored afterwards, so refinement never permanently alters the grid.

        Parameters
        ----------
        mask_method : {"bbox", "sector"}, default "bbox"
            How to build the refinement mask.
            - "bbox": Use a DR-based bounding box (see ``top_frac`` / ``margin_km``).
            - "sector": Use an apex-anchored wedge (see sector-specific parameters).
        top_frac : float, default 0.20
            Fraction (0–1] of valid samples (by DR) used to define the **bbox** or,
            for the sector method, used (when ``prefer_misfit=False``) to compute the
            bearing from the median of the top-DR points. A sensible minimum count is
            enforced via ``min_top_points``.
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
            near-field nodes).
        half_angle_deg : float, default 25.0
            **Sector only.** Half-angle (degrees) of the wedge around the inferred
            bearing; total wedge aperture is 2×half_angle_deg.
        prefer_misfit : bool, default True
            **Sector only.** If True, infer the bearing from the **global minimum of
            misfit** in the current source track. If False (or misfit isn’t usable),
            infer the bearing from the **median location of the top-DR subset**.
        top_frac_for_bearing : float or None, default None
            **Sector only.** If provided (and ``prefer_misfit=False``), the fraction
            used specifically for selecting the top-DR subset to infer the bearing.
            If None, falls back to ``top_frac``.
        mode : str, default "mask"
            Kept for API compatibility. Refinement is always applied as a temporary
            mask and the grid is restored afterwards. Destructive “slice” mode is no
            longer supported.

        misfit_backend, spatial_smooth_sigma, temporal_smooth_mode, temporal_smooth_win,
        viterbi_lambda_km, viterbi_max_step_km, min_stations, eps, use_median_for_DR, batch_size
            Passed through to :meth:`fast_locate` (see that docstring).
            ``spatial_sigma_nodes`` is accepted as an alias of ``spatial_smooth_sigma``
            for backward compatibility.
        verbose : bool, default True
            Print diagnostics about mask construction, intersection, and final size.

        Returns
        -------
        self : ASL
            The same object, updated in-place with a new ``self.source``.

        Side Effects
        ------------
        - Updates ``self.source`` with a newly computed track from :meth:`fast_locate`.
        - Updates ``self.connectedness`` based on the refined track.
        - Grid masking is only temporary: the persistent baseline mask is restored
        after relocation.

        Mask Intersection
        -----------------
        The constructed mask is AND-ed with:
        1) any persistent node mask carried by the grid (e.g., land-only, dome+streams), and  
        2) any temporary mask already active in ``self._node_mask``.  
        This guarantees baseline masks remain respected.

        Algorithmic Notes
        -----------------
        **bbox**
        1) Select indices with finite (lat, lon, DR). Rank by DR and take
            ``max(min_top_points, ceil(top_frac*N))`` samples.
        2) Compute min/max lat/lon of those samples; expand by ``margin_km`` converted
            to degrees with a latitude-dependent lon scaling.
        3) Keep nodes inside the expanded box.

        **sector**
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
        # 1) Classic DR-bbox refinement (temporary mask), then re-run fast_locate
        >>> asl.refine_and_relocate(mask_method="bbox", top_frac=0.25, margin_km=1.0)

        # 2) Sector refinement from dome apex with Viterbi smoothing
        >>> asl.refine_and_relocate(
        ...     mask_method="sector",
        ...     apex_lat=16.712, apex_lon=-62.181,
        ...     length_km=8.0, inner_km=0.0, half_angle_deg=25.0,
        ...     prefer_misfit=True,
        ...     temporal_smooth_mode="viterbi", temporal_smooth_win=7,
        ...     viterbi_lambda_km=8.0, viterbi_max_step_km=30.0,
        ... )

        CHANGES:
        Temporarily restrict the spatial search domain, run fast_locate() on that
        sub-grid, and then restore the original Grid mask.

        In addition, this stashes:
        - self._last_eval_idx  (np.ndarray[int]): node indices actually evaluated
        - self._last_misfit_vec (np.ndarray[float], if available): per-node misfit
        so plotting code can paint misfit back onto the full grid safely.
        """
        import numpy as np
        from obspy.geodetics.base import gps2dist_azimuth

        if self.source is None:
            raise RuntimeError("No source yet. Run fast_locate()/locate() first.")

        # ---------- sanitize knobs ----------
        top_frac = float(np.clip(top_frac, 0.0, 1.0))
        if top_frac_for_bearing is not None:
            top_frac_for_bearing = float(np.clip(top_frac_for_bearing, 0.0, 1.0))
        margin_km   = max(0.0, float(margin_km))
        length_km   = max(0.0, float(length_km))
        inner_km    = max(0.0, float(inner_km))
        half_angle_deg = float(np.clip(half_angle_deg, 0.0, 180.0))
        if inner_km > length_km:
            inner_km, length_km = length_km, inner_km  # ensure inner<=length

        # alias support
        if spatial_smooth_sigma is None and spatial_sigma_nodes is not None:
            spatial_smooth_sigma = spatial_sigma_nodes
        if top_frac_for_bearing is None:
            top_frac_for_bearing = top_frac

        # ---------- current track handles ----------
        lat = np.asarray(self.source.get("lat"), float)
        lon = np.asarray(self.source.get("lon"), float)
        dr  = np.asarray(self.source.get("DR", np.full_like(lat, np.nan)), float)
        mis = np.asarray(self.source.get("misfit", np.full_like(lat, np.nan)), float)

        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            if verbose:
                print("[ASL] refine_and_relocate: no finite source coords; skipping")
            return self

        grid = self.gridobj
        g_lat = np.asarray(grid.gridlat).reshape(-1)
        g_lon = np.asarray(grid.gridlon).reshape(-1)
        nn = g_lat.size

        # ---------- mask builders ----------
        def _mask_bbox() -> np.ndarray:
            idx_all = np.flatnonzero(ok & np.isfinite(dr))
            if idx_all.size == 0:
                idx_all = np.flatnonzero(ok)

            k_frac = int(np.ceil(top_frac * idx_all.size))
            k = max(1, min_top_points, k_frac)
            k = min(k, idx_all.size)

            if np.isfinite(dr[idx_all]).any():
                order = np.argsort(dr[idx_all])[::-1]
            else:
                order = np.arange(idx_all.size)

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
            apx_lat = apex_lat if apex_lat is not None else getattr(grid, "apex_lat", None)
            apx_lon = apex_lon if apex_lon is not None else getattr(grid, "apex_lon", None)
            if apx_lat is None or apx_lon is None:
                raise ValueError("sector mask requires apex_lat/apex_lon (or grid.apex_lat/apex_lon).")

            def _bearing(aplat, aplon, tlat, tlon) -> float:
                _, az, _ = gps2dist_azimuth(float(aplat), float(aplon), float(tlat), float(tlon))
                return float(az % 360.0)

            valid_mis = np.isfinite(mis) & ok
            if prefer_misfit and valid_mis.any():
                j_local = np.nanargmin(mis[valid_mis])
                j = np.flatnonzero(valid_mis)[j_local]
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

            # great-circle distance & azimuth from apex to all nodes
            dist_km = np.empty(nn, float)
            az_deg  = np.empty(nn, float)
            for i in range(nn):
                d_m, az, _ = gps2dist_azimuth(float(apx_lat), float(apx_lon),
                                            float(g_lat[i]), float(g_lon[i]))
                dist_km[i] = 0.001 * float(d_m)
                az_deg[i]  = float(az % 360.0)

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

        # ---------- build refinement mask ----------
        mm = (mask_method or "bbox").lower()
        if mm == "bbox":
            mask_bool = _mask_bbox()
        elif mm == "sector":
            mask_bool = _mask_sector()
        else:
            raise ValueError("mask_method must be one of {'bbox','sector'}")

        # ---------- intersect with existing masks ----------
        combined_bool = np.asarray(mask_bool, bool).ravel()

        # Intersect with persistent Grid mask (if any)
        base_idx = grid.get_mask_indices()
        if base_idx is not None and np.size(base_idx) > 0:
            base_bool = np.zeros_like(combined_bool, dtype=bool)
            base_bool[np.asarray(base_idx, int)] = True
            combined_bool &= base_bool

        # Intersect with any ASL temp mask (legacy support)
        if getattr(self, "_node_mask", None) is not None and np.size(self._node_mask) > 0:
            temp_bool = np.zeros_like(combined_bool, dtype=bool)
            temp_bool[np.asarray(self._node_mask, int)] = True
            combined_bool &= temp_bool

        sub_idx = np.flatnonzero(combined_bool)

        # ---------- kwargs passthrough to fast_locate ----------
        kwargs = {
            "misfit_backend":        misfit_backend,
            "spatial_smooth_sigma":  spatial_smooth_sigma if spatial_smooth_sigma is not None else spatial_sigma_nodes,
            "temporal_smooth_mode":  temporal_smooth_mode,
            "temporal_smooth_win":   temporal_smooth_win,
            "viterbi_lambda_km":     viterbi_lambda_km,
            "viterbi_max_step_km":   viterbi_max_step_km,
            "min_stations":          min_stations,
            "eps":                   eps,
            "use_median_for_DR":     use_median_for_DR,
            "batch_size":            batch_size,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # ---------- run (or bail gracefully) ----------
        if sub_idx.size == 0:
            if verbose:
                print("[ASL] refine_and_relocate: sub-grid empty after intersection; keeping current domain.")
            # Run as-is (full/current mask), and let fast_locate set its own _last_eval_idx
            self.fast_locate(verbose=verbose, **kwargs)
            # If fast_locate didn’t set an eval index, default to current mask or full grid
            if not hasattr(self, "_last_eval_idx"):
                current_idx = grid.get_mask_indices()
                self._last_eval_idx = (np.asarray(current_idx, int).ravel()
                                    if current_idx is not None and np.size(current_idx) > 0
                                    else np.arange(nn, dtype=int))
            return self

        if verbose:
            frac = 100.0 * sub_idx.size / nn
            print(f"[ASL] refine_and_relocate[{mm}]: {sub_idx.size} nodes (~{frac:.1f}% of grid)")

        # ---------- TEMPORARY mask application on Grid ----------
        old_idx = grid.get_mask_indices()  # stash current mask (indices or None)

        try:
            # Apply combined mask temporarily (replace)
            grid.apply_mask_boolean(combined_bool, mode="replace")

            # If your fast_locate accepts a node list, pass it (best). If not, the
            # grid mask will already restrict the domain.
            try:
                self.fast_locate(candidate_idx=sub_idx, verbose=verbose, **kwargs)
            except TypeError:
                # Older API: no candidate_idx; rely on mask
                self.fast_locate(verbose=verbose, **kwargs)

            # Stash *exact* indices evaluated this run (for plotting)
            self._last_eval_idx = np.asarray(sub_idx, int).ravel()

            # Try to stash the per-node misfit vector if the locator exposes it
            # (these attribute names are just common patterns—adjust if your code differs)
            if hasattr(self, "last_misfit_vector") and self.last_misfit_vector is not None:
                self._last_misfit_vec = np.asarray(self.last_misfit_vector, float).ravel()
            elif hasattr(self, "_last_misfit_vec") and isinstance(self._last_misfit_vec, np.ndarray):
                # already set by fast_locate; ensure it matches size or drop it
                if self._last_misfit_vec.size != sub_idx.size:
                    # size mismatch—discard to avoid plotting errors later
                    delattr(self, "_last_misfit_vec")
            elif hasattr(self, "misfit_backend") and hasattr(self.misfit_backend, "last_per_node"):
                vec = np.asarray(self.misfit_backend.last_per_node, float).ravel()
                if vec.size == sub_idx.size:
                    self._last_misfit_vec = vec

        finally:
            # Restore prior Grid mask exactly
            if old_idx is None:
                grid.clear_mask()
            else:
                grid.apply_mask_indices(old_idx)

        return self
        

    def inverse_locate(
        self,
        *,
        time_index: int | None = None,
        reference_lonlat: tuple[float, float] | None = None,
        init_lonlat: tuple[float, float] | None = None,
        optimizer: str = "Nelder-Mead",
        bounds_km: tuple[tuple[float, float], tuple[float, float]] | None = None,
        clip_R_min_km: float = 0.05,
        verbose: bool = True,
        return_design: bool = True,   # NEW
    ):
        """
        Hybrid inversion for amplitude source location.

        log|A_i| = logA0 - N*log(R_i) - k*R_i + e_i
                = [1, log(R_i), R_i] · [logA0, -N, -k]^T + e_i

        Inner LSQ (β linear):   β = (XᵀX)⁻¹ Xᵀ y   at each (x,y)
        Outer minimize (x,y):   SSE(β, x, y) = ||y - Xβ||²

        Returns a dict with solution and, if return_design=True, the
        design matrix and residual diagnostics at the optimum.
        """

        # ---- 1) pick time index (default = peak DR) ----
        if time_index is None:
            if getattr(self, "source", None) and "DR" in self.source:
                time_index = int(np.nanargmax(self.source["DR"]))
            else:
                raise RuntimeError("time_index not given and no DR available to pick automatically.")

        # ---- 2) pull amplitudes at that time ----
        st = self.metric2stream()
        seed_all = [tr.id for tr in st]
        Y = np.vstack([np.asarray(tr.data, dtype=np.float64) for tr in st])
        amps = Y[:, time_index]
        ok_amp = np.isfinite(amps)

        seed_ids = [sid for sid, ok in zip(seed_all, ok_amp) if ok]
        amps = amps[ok_amp]
        if len(amps) < 3:
            raise RuntimeError("Need ≥3 finite station amplitudes.")

        # ---- 3) station geometry (lon/lat) in the same order ----
        if not getattr(self, "station_coordinates", None):
            raise RuntimeError("ASL.station_coordinates missing; compute_or_load_distances() first.")
        slon = np.array([self.station_coordinates[sid]["longitude"] for sid in seed_ids], float)
        slat = np.array([self.station_coordinates[sid]["latitude"]  for sid in seed_ids], float)
        ok_geo = np.isfinite(slon) & np.isfinite(slat)
        if ok_geo.sum() < 3:
            raise RuntimeError("Fewer than 3 stations with valid geometry.")
        seed_ids = [sid for sid, ok in zip(seed_ids, ok_geo) if ok]
        slon, slat, amps = slon[ok_geo], slat[ok_geo], amps[ok_geo]

        # ---- 4) local tangent plane (km) ----
        if reference_lonlat is None:
            reference_lonlat = (dome_location["lon"], dome_location["lat"])
        ref_lon, ref_lat = map(float, reference_lonlat)
        m_per_deg_lat, m_per_deg_lon = _meters_per_degree(ref_lat)
        X_sta = (slon - ref_lon) * (m_per_deg_lon / 1000.0)  # km
        Y_sta = (slat - ref_lat) * (m_per_deg_lat  / 1000.0) # km
        STA = np.column_stack([X_sta, Y_sta])

        # ---- 5) initial guess (x,y) in km ----
        if init_lonlat is None:
            init_lonlat = reference_lonlat
        x0 = np.array([
            (init_lonlat[0] - ref_lon) * (m_per_deg_lon / 1000.0),
            (init_lonlat[1] - ref_lat) * (m_per_deg_lat  / 1000.0),
        ], dtype=float)

        # ---- 6) inner LSQ at fixed (x,y) ----
        y = np.log(np.clip(np.abs(amps), 1e-12, None))  # log|A|

        def lsq_at_xy(xy_km: np.ndarray):
            dx = STA[:, 0] - xy_km[0]
            dy = STA[:, 1] - xy_km[1]
            R = np.clip(np.hypot(dx, dy), clip_R_min_km, None)  # km
            X = np.column_stack([np.ones_like(R), np.log(R), R])  # [1, logR, R]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)          # β = [logA0, -N, -k]
            yhat = X @ beta
            r = y - yhat
            sse = float(np.dot(r, r))
            return beta, X, yhat, r, R, sse

        def objective(xy):
            _, _, _, _, _, sse = lsq_at_xy(np.asarray(xy, float))
            return sse

        # ---- 7) outer optimization over (x,y) ----
        opt_kwargs = {}
        if bounds_km is not None and optimizer in ("L-BFGS-B", "TNC", "Powell"):
            opt_kwargs["bounds"] = bounds_km
        res = minimize(objective, x0, method=optimizer, **opt_kwargs)

        # ---- 8) final fit at optimum ----
        beta, X, yhat, r, R, sse = lsq_at_xy(res.x)
        logA0, negN, negk = map(float, beta)
        N, k = -negN, -negk

        lon = ref_lon + (res.x[0] * 1000.0) / m_per_deg_lon
        lat = ref_lat + (res.x[1] * 1000.0) / m_per_deg_lat

        if verbose:
            print(f"[ASL:INV] success={res.success}  SSE={sse:.3g}")
            print(f"          lon={lon:.6f}, lat={lat:.6f}, x={res.x[0]:.2f} km, y={res.x[1]:.2f} km")
            print(f"          logA0={logA0:.3f}, N={N:.3f}, k={k:.4f} (1/km)  nsta={len(amps)}")

        out = {
            "lon": float(lon), "lat": float(lat),
            "x_km": float(res.x[0]), "y_km": float(res.x[1]),
            "logA0": float(logA0), "N": float(N), "k": float(k),
            "sse": float(sse), "success": bool(res.success), "message": str(res.message),
            "nsta": int(len(amps)), "time_index": int(time_index),
        }

        if return_design:
            out["design"] = {
                "X": X,               # shape (nsta, 3)  columns: [1, logR, R]
                "y": y,               # log|A|
                "yhat": yhat,
                "residuals": r,
                "R_km": R,
                "seed_ids": tuple(seed_ids),
                "columns": ("1", "logR", "R"),
            }

        return out
    

    def inverse_locate_nonlinear(
        self,
        *,
        time_index: int | None = None,
        reference_lonlat: tuple[float, float] | None = None,
        init_lonlat: tuple[float, float] | None = None,   # seed for (x,y)
        init_params: tuple[float, float, float] | None = None,  # (logA0, N, k) seed
        use_3d: bool = False,
        source_elev_mode: str = "zero",   # "zero" | "fixed" | "dem"
        source_elev_m: float | None = None,  # used if mode=="fixed"
        clip_R_min_km: float = 0.05,
        bounds_xy_km: tuple[tuple[float,float], tuple[float,float]] | None = None,  # ((xmin,xmax),(ymin,ymax))
        bounds_params: tuple[tuple[float,float], tuple[float,float], tuple[float,float]] = ((-np.inf, np.inf),(0,10),(0,5)),
        robust_loss: str = "soft_l1",     # "linear"|"soft_l1"|"huber" ...
        f_scale: float = 1.0,
        verbose: bool = True,
    ):
        """
        Full nonlinear ASL inversion:
        unknowns = (x_km, y_km[, z_km], logA0, N, k)
        model    = log|A_i| = logA0 - N*log(R_i) - k*R_i

        - Distances R_i are km (2D or 3D).
        - 3D uses station elevation and a source elevation from DEM/fixed/zero.
        - Returns parameters, location (lon/lat), residuals, and diagnostics.

        Notes
        -----
        * For speed/robustness, seed (x,y) with the hybrid solver you already have,
        then pass that as init_lonlat.
        * Keep units consistent: distances in km; k is 1/km; N is dimensionless.
        """
        import numpy as np
        from scipy.optimize import least_squares
        from flovopy.core.mvo import dome_location
        from .grid import _meters_per_degree
        from .distances import geo_distance_3d_km  # optional guard for 3D check

        # ---- 0) choose time index (default: peak DR) ----
        if time_index is None:
            if getattr(self, "source", None) is not None and "DR" in self.source:
                time_index = int(np.nanargmax(self.source["DR"]))
            else:
                raise RuntimeError("time_index not provided and no DR to auto-select.")

        # ---- 1) amplitudes y_i = log|A_i| ----
        st = self.metric2stream()
        seed_all = [tr.id for tr in st]
        Y = np.vstack([np.asarray(tr.data, dtype=np.float64) for tr in st])
        amps = Y[:, time_index]
        ok = np.isfinite(amps)
        if ok.sum() < 3:
            raise RuntimeError("Need at least 3 finite station amplitudes.")
        amps = amps[ok]
        seed_ids = [sid for sid, t in zip(seed_all, ok) if t]

        y_obs = np.log(np.clip(np.abs(amps), 1e-12, None))

        # ---- 2) station geometry ----
        if not getattr(self, "station_coordinates", None):
            raise RuntimeError("ASL.station_coordinates missing; compute distances first.")
        slon = np.array([self.station_coordinates[sid]["longitude"] for sid in seed_ids], float)
        slat = np.array([self.station_coordinates[sid]["latitude"]  for sid in seed_ids], float)
        selev_m = np.array([self.station_coordinates[sid].get("elevation", 0.0) for sid in seed_ids], float)

        ok_geo = np.isfinite(slon) & np.isfinite(slat)
        if ok_geo.sum() < 3:
            raise RuntimeError("Fewer than 3 stations with valid lon/lat.")
        slon, slat, selev_m, y_obs, seed_ids = (
            slon[ok_geo], slat[ok_geo], selev_m[ok_geo], y_obs[ok_geo], [sid for sid, g in zip(seed_ids, ok_geo) if g]
        )

        # ---- 3) local tangent plane (km) ----
        if reference_lonlat is None:
            reference_lonlat = (dome_location["lon"], dome_location["lat"])
        ref_lon, ref_lat = map(float, reference_lonlat)
        m_per_deg_lat, m_per_deg_lon = _meters_per_degree(ref_lat)
        X_sta = (slon - ref_lon) * (m_per_deg_lon / 1000.0)  # km
        Y_sta = (slat - ref_lat) * (m_per_deg_lat  / 1000.0) # km
        STA_xy = np.column_stack([X_sta, Y_sta])

        # Station z (km, positive up)
        Z_sta_km = (selev_m / 1000.0) if use_3d else np.zeros_like(X_sta)

        # ---- 4) source z (km) policy ----
        def source_z_km_at(lon, lat):
            if not use_3d:
                return 0.0
            if source_elev_mode == "zero":
                return 0.0
            if source_elev_mode == "fixed":
                return float((source_elev_m or 0.0) / 1000.0)
            if source_elev_mode == "dem":
                # Try to pull from grid DEM if available; else 0
                z_m = 0.0
                try:
                    g = getattr(self, "gridobj", None)
                    if g is not None and getattr(g, "node_elev_m", None) is not None:
                        # nearest-neighbor from grid (simple & robust)
                        j = int(np.abs(g.lonrange - lon).argmin())
                        i = int(np.abs(g.latrange - lat).argmin())
                        z_m = float(g.node_elev_m[i, j])
                except Exception:
                    z_m = 0.0
                return z_m / 1000.0
            raise ValueError("source_elev_mode must be 'zero'|'fixed'|'dem'.")

        # ---- 5) initial guess θ0 ----
        if init_lonlat is None:
            init_lonlat = reference_lonlat  # seed at center
        x0_km = (init_lonlat[0] - ref_lon) * (m_per_deg_lon / 1000.0)
        y0_km = (init_lonlat[1] - ref_lat) * (m_per_deg_lat  / 1000.0)
        if use_3d:
            z0_km = source_z_km_at(init_lonlat[0], init_lonlat[1])
        else:
            z0_km = 0.0

        if init_params is None:
            # modest, neutral seeds
            logA0_0, N0, k0 = (np.nanmedian(y_obs), 1.0, 0.1)
        else:
            logA0_0, N0, k0 = map(float, init_params)

        if use_3d:
            theta0 = np.array([x0_km, y0_km, z0_km, logA0_0, N0, k0], float)
        else:
            theta0 = np.array([x0_km, y0_km, logA0_0, N0, k0], float)

        # ---- 6) bounds ----
        if bounds_xy_km is not None:
            (xmin, xmax), (ymin, ymax) = bounds_xy_km
        else:
            # default ±10 km box
            xmin, xmax, ymin, ymax = -10.0, 10.0, -10.0, 10.0

        (logA0_bounds, N_bounds, k_bounds) = bounds_params
        if use_3d:
            # z is usually small near surface; allow, say, ±2 km
            z_bounds = (-2.0, 2.0) if source_elev_mode != "fixed" else (z0_km, z0_km)
            lb = [xmin, ymin, z_bounds[0], logA0_bounds[0], N_bounds[0], k_bounds[0]]
            ub = [xmax, ymax, z_bounds[1], logA0_bounds[1], N_bounds[1], k_bounds[1]]
        else:
            lb = [xmin, ymin, logA0_bounds[0], N_bounds[0], k_bounds[0]]
            ub = [xmax, ymax, logA0_bounds[1], N_bounds[1], k_bounds[1]]
        bounds = (np.array(lb, float), np.array(ub, float))

        # ---- 7) residuals and Jacobian ----
        def residuals(theta):
            if use_3d:
                x, y, z_km, logA0, N, k = theta
            else:
                x, y, logA0, N, k = theta
                z_km = 0.0

            # source lon/lat for DEM z if needed
            lon = ref_lon + (x * 1000.0) / m_per_deg_lon
            lat = ref_lat + (y * 1000.0) / m_per_deg_lat
            if use_3d and source_elev_mode == "dem":
                z_km = source_z_km_at(lon, lat)

            dx = X_sta - x
            dy = Y_sta - y
            dz = Z_sta_km - z_km
            if use_3d:
                R = np.sqrt(dx*dx + dy*dy + dz*dz)
            else:
                R = np.sqrt(dx*dx + dy*dy)

            R = np.clip(R, clip_R_min_km, None)
            yhat = logA0 - N * np.log(R) - k * R
            return y_obs - yhat  # residuals

        def jacobian(theta):
            if use_3d:
                x, y, z_km, logA0, N, k = theta
            else:
                x, y, logA0, N, k = theta
                z_km = 0.0

            # source lon/lat for DEM z if needed
            lon = ref_lon + (x * 1000.0) / m_per_deg_lon
            lat = ref_lat + (y * 1000.0) / m_per_deg_lat
            if use_3d and source_elev_mode == "dem":
                z_km = source_z_km_at(lon, lat)

            dx = X_sta - x
            dy = Y_sta - y
            dz = Z_sta_km - z_km
            if use_3d:
                R = np.sqrt(dx*dx + dy*dy + dz*dz)
            else:
                R = np.sqrt(dx*dx + dy*dy)

            R = np.clip(R, clip_R_min_km, None)
            invR = 1.0 / R
            # dR/dx = (x - x_i)/R = -dx/R   (since dx = x_i - x)
            dRdx = -dx * invR
            dRdy = -dy * invR
            dRdz = (-dz * invR) if use_3d else None

            # r = y - (logA0 - N logR - k R)  =>  ∂r/∂θ = [-∂f/∂θ]
            # ∂r/∂logA0 = -1
            # ∂r/∂N     = +logR
            # ∂r/∂k     = +R
            # ∂r/∂x     = +[ N*(1/R)*dRdx + k*dRdx ] = (N*invR + k)*dRdx
            # similarly for y, (and z if 3D)
            common = (N * invR + k)
            Jx = common * dRdx
            Jy = common * dRdy
            cols = []
            if use_3d:
                Jz = common * dRdz
                cols.extend([Jx, Jy, Jz, -np.ones_like(R), np.log(R), R])
            else:
                cols.extend([Jx, Jy, -np.ones_like(R), np.log(R), R])
            return np.column_stack(cols)

        # ---- 8) solve ----
        res = least_squares(
            residuals, theta0, jac=jacobian, bounds=bounds,
            loss=robust_loss, f_scale=f_scale, xtol=1e-8, ftol=1e-8, gtol=1e-8,
            max_nfev=2000, verbose=2 if verbose else 0,
        )

        # ---- 9) unpack solution ----
        theta = res.x
        if use_3d:
            xk, yk, zk, logA0, N, k = theta
        else:
            xk, yk, logA0, N, k = theta
            zk = 0.0

        lon = ref_lon + (xk * 1000.0) / m_per_deg_lon
        lat = ref_lat + (yk * 1000.0) / m_per_deg_lat

        # final diagnostics
        r = residuals(theta)
        J = jacobian(theta)
        sse = float(r @ r)
        dof = max(1, r.size - J.shape[1])
        sigma2 = sse / dof
        try:
            # covariance ~ (JᵀJ)⁻¹ σ² (approximate, at optimum)
            JTJ = J.T @ J
            cov = np.linalg.inv(JTJ) * sigma2
        except np.linalg.LinAlgError:
            cov = np.full((J.shape[1], J.shape[1]), np.nan)

        out = {
            "success": bool(res.success),
            "message": str(res.message),
            "cost": float(res.cost),
            "sse": sse,
            "dof": int(dof),
            "sigma2": float(sigma2),
            "x_km": float(xk), "y_km": float(yk), "z_km": float(zk),
            "lon": float(lon), "lat": float(lat),
            "logA0": float(logA0), "N": float(N), "k": float(k),
            "nsta": int(y_obs.size),
            "time_index": int(time_index),
            "seed_ids": tuple(seed_ids),
            "residuals": r,
            "J": J,
            "bounds": bounds,
            "use_3d": bool(use_3d),
            "source_elev_mode": source_elev_mode,
        }
        return out

    # ---------- plots ----------
    def plot(
        self,
        threshold_DR: float = 0.0,
        scale: float = 1.0,
        join: bool = False,
        number: int = 0,
        outfile: str | None = None,
        stations=None,
        title: str | None = None,
        normalize: bool = True,
        *,
        color_by: str = "time",          # "time" | "dr"
        cmap: str = "viridis",           # PyGMT CPT name
        topo_kw: dict | None = None,
        show: bool = True,
        return_fig: bool = True,
    ):
        """
        Plot the time-ordered ASL source track on a PyGMT basemap.

        Parameters
        ----------
        threshold_DR : float, default 0.0
            Mask points with DR < threshold from the plot (lon/lat → NaN).
        scale : float, default 1.0
            Overall marker size scale.
        join : bool, default False
            If True, draw a polyline connecting points chronologically.
        number : int, default 0
            If > 0, subselect the top-N points by DR after filtering.
        outfile : str or None
            If provided, save figure to this path.
        stations : any, optional
            Passed through to `topo_map` for station plotting.
        title : str or None
            Title; if None, one is auto-built from time span and connectedness.
        normalize : bool, default True
            - True  → marker size ∝ DR / max(DR) (with sqrt scaling for visibility)
            - False → marker size ∝ sqrt(max(DR,0))
        color_by : {"time","dr"}, default "time"
            Color points by chronological index or by DR.
        cmap : str, default "viridis"
            CPT colormap name passed to PyGMT (`makecpt`).
        topo_kw : dict or None
            Forwarded to `topo_map(**topo_kw)`. Ignored when None.
        show : bool, default True
            If True and `outfile` is None, show interactively.
        return_fig : bool, default True
            If True, return the PyGMT Figure.

        Returns
        -------
        pygmt.Figure | None
        """
        import pygmt
        from obspy import UTCDateTime

        topo_kw = dict(topo_kw or {})

        def _ts(msg: str) -> None:
            now = UTCDateTime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] [ASL:PLOT] {msg}")

        src = getattr(self, "source", None)
        if not src:
            # still return a basemap so callers can overlay later
            fig = topo_map(stations=stations, **topo_kw)
            if outfile:
                fig.savefig(outfile)
            elif show:
                fig.show()
            return fig if return_fig else None

        # ---- extract arrays
        try:
            x = np.asarray(src["lon"], float)
            y = np.asarray(src["lat"], float)
            DR = np.asarray(src["DR"], float)
            t  = src.get("t", None)  # expect UTCDateTime array
        except KeyError as e:
            _ts(f"ERROR: required key missing in source: {e!r}")
            raise

        if not (x.size == y.size == DR.size):
            raise ValueError("ASL.plot(): lon, lat, DR must have identical lengths.")

        # ---- thresholding (copy; no mutation)
        if float(threshold_DR) > 0:
            mask = DR < float(threshold_DR)
            x = x.copy(); y = y.copy(); DR = DR.copy()
            x[mask] = np.nan; y[mask] = np.nan; DR[mask] = 0.0
            _ts(f"threshold applied at {threshold_DR}; masked {int(np.count_nonzero(mask))} points")

        # ---- marker sizes
        mx = float(np.nanmax(DR)) if np.isfinite(DR).any() else 0.0
        if normalize and mx > 0:
            symsize = np.sqrt(np.maximum(DR / mx, 0.0)) * float(scale)
        else:
            symsize = np.sqrt(np.maximum(DR, 0.0)) * float(scale)

        # ---- validity filter
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(DR) & np.isfinite(symsize) & (symsize > 0)
        if not np.any(m):
            raise RuntimeError("ASL.plot(): no valid points to plot after filtering.")
        x, y, DR, symsize = x[m], y[m], DR[m], symsize[m]

        # ---- optional top-N by DR
        if number and number > 0 and number < len(x):
            _ts(f"subselecting top-N by DR: N={number}")
            ind = np.argpartition(DR, -number)[-number:]
            x, y, DR, symsize = x[ind], y[ind], DR[ind], symsize[ind]

        # ---- default title
        if title is None:
            try:
                if t is not None and len(t) > 0:
                    t0 = t[0].datetime.strftime("%Y-%m-%d %H:%M:%S")
                    t1 = t[-1].datetime.strftime("%Y-%m-%d %H:%M:%S")
                    conn = getattr(self, "connectedness", None)
                    cstr = ""
                    if isinstance(conn, dict) and "score" in conn and np.isfinite(conn["score"]):
                        cstr = f"  (score={conn['score']:.2f})"
                    title = f"ASL: {t0}({t1-t0:.0f} s): {cstr}"
                    title += '\n' + self.cfg.tag
            except Exception:
                pass
        if title is not None:
            topo_kw.setdefault("title", title) #.replace("\n", "~"))

        # ---- basemap
        fig = topo_map(stations=stations, **topo_kw)

        # ---- coloring
        color_by_norm = (color_by or "time").strip().lower()
        if color_by_norm not in {"time", "dr"}:
            _ts(f"unknown color_by={color_by!r}; defaulting to 'time'")
            color_by_norm = "time"

        if color_by_norm == "time":
            cvals = np.arange(len(x), dtype=float)
            pygmt.makecpt(cmap=cmap, series=[0, len(x) - 1])
            cbar_label = "Time (s)"
        else:
            dmin = float(np.nanmin(DR)) if np.isfinite(DR).any() else 0.0
            dmax = float(np.nanmax(DR)) if np.isfinite(DR).any() else 1.0
            if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
                dmin, dmax = 0.0, 1.0
            cvals = DR.astype(float, copy=False)
            pygmt.makecpt(cmap=cmap, series=[dmin, dmax])
            cbar_label = "Reduced Displacement"

        # ---- scatter
        fig.plot(x=x, y=y, size=symsize, style="c", pen=None, fill=cvals, cmap=True)
        fig.colorbar(frame=f'+l"{cbar_label}"')

        # ---- optional chronological polyline
        if join and len(x) > 1:
            fig.plot(x=x, y=y, pen="1p,red")

        # ---- output
        if outfile:
            fig.savefig(outfile)
            _ts(f"saved figure: {outfile}")
        elif show:
            fig.show()

        return fig if return_fig else None


    def plot_reduced_displacement(self, threshold_DR: float = 0.0, outfile: str | None = None, *, displacement=True, show: bool = True):
        """
        Plot the time series of reduced displacement (DR) or reduced velocity.

        Parameters
        ----------
        threshold_DR : float, default 0.0
            Optional horizontal threshold line to draw.
        outfile : str or None
            If provided, write the PNG/SVG/etc. to this path.
        show : bool, default True
            If True and `outfile` is None, call `plt.show()`.

        Notes
        -----
        - Expects `self.source["t"]` (UTCDateTime array) and `self.source["DR"]`.
        - Units label is kept as cm² to match the convention used by the locator
        (`DR` stored *already scaled* in cm² by 1e7 factor).
        """
        if not getattr(self, "source", None):
            return
        t = self.source.get("t", None)
        if t is None:
            return

        t_dt = [this_t.datetime for this_t in t]
        plt.figure()
        plt.plot(t_dt, self.source["DR"])
        if threshold_DR:
            plt.axhline(float(threshold_DR), linestyle="--")
        plt.xlabel("Date/Time (UTC)")
        if displacement:
            plt.ylabel("Reduced Displacement (${cm}^2$)")
        else:
            plt.ylabel("Reduced Velocity (${cm}^2/s$)")
        if outfile:
            plt.savefig(outfile); plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def plot_misfit(self, outfile: str | None = None, *, show: bool = True):
        """
        Plot the misfit time series (value at the chosen node per frame).

        Parameters
        ----------
        outfile : str or None
            If provided, save to file.
        show : bool, default True
            If True and `outfile` is None, show interactively.
        """
        t = self.source.get("t", None)
        mis = self.source.get("misfit", None)
        if t is None or mis is None:
            return

        t_dt = [this_t.datetime for this_t in t]
        plt.figure()
        plt.plot(t_dt, mis)
        plt.xlabel("Date/Time (UTC)")
        plt.ylabel("Misfit (std/|mean|)")
        if outfile:
            plt.savefig(outfile); plt.close()
        elif show:
            plt.show()
        else:
            plt.close()


    def plot_misfit_heatmap(
        self,
        outfile: str | None = None,
        *,
        backend=None,
        cmap: str = "turbo",
        transparency: int = 40,
        topo_kw: dict | None = None,
        show: bool = True,
    ):
        """
        Plot a per-node misfit heatmap for the frame with peak DR (diagnostic).

        Parameters
        ----------
        backend : object or None
            Misfit backend instance. If None, uses `StdOverMeanMisfit()`.
        cmap : str, default "turbo"
            CPT colormap name for the heatmap.
        transparency : int, default 40
            Transparency percentage (0–100) for the overlay.
        topo_kw : dict or None
            Forwarded to the underlying topo/plot routine.
        """
        if backend is None:
            backend = StdOverMeanMisfit()
        plot_misfit_heatmap_for_peak_DR(
            self,
            backend=backend,
            cmap=cmap,
            transparency=transparency,
            outfile=outfile,
            topo_kw=(topo_kw or {}),
            show=show,
        )

    def source_to_dataframe(self) -> "pd.DataFrame | None":
        """
        Convert the ASL source dict to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of the source information, or None if no source exists.

        Notes
        -----
        - This is a shallow conversion (no copies of large arrays are forced).
        - The DataFrame will include whatever keys exist in ``self.source``.
        """
        if not hasattr(self, "source") or self.source is None:
            print("[ASL] No source data available. Did you run locate() or fast_locate()?")
            return None

        try:
            df = pd.DataFrame(self.source)
            if "t" in df.columns:
                try:
                    # Convert ObsPy UTCDateTime objects to Python datetime first
                    df["t"] = pd.to_datetime([tt.datetime for tt in self.source["t"]], utc=True, errors="coerce")
                except Exception as e:
                    print(f"[ASL] Failed to convert t column: {e}")
            return df
        except Exception as e:
            print(f"[ASL] Could not convert source to DataFrame: {e!r}")
            return None


    # Source output methods
    def print_source(self, max_rows: int = 100):
        """
        Pretty-print the current ASL source track as a DataFrame.

        - Converts ObsPy UTCDateTime to pandas datetimes (UTC).
        - Shows a stable, human-friendly column order.
        - Summarizes unique node indices if present.

        Parameters
        ----------
        max_rows : int, default 100
            Max rows to display. Use None to show all.

        Returns
        -------
        pandas.DataFrame or None
            The DataFrame that was printed, or None if no source available.
        """
        df = self.source_to_dataframe()
        if df is None or df.empty:
            print("[ASL] No source to print.")
            return None

        # Stable column order (include only those present)
        preferred = ["t", "lat", "lon", "DR", "misfit", "azgap", "nsta", "node_index", "connectedness"]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df.loc[:, cols]

        try:
            n = len(df)
            if max_rows is not None and n > max_rows:
                print(f"[ASL] Showing first {max_rows} of {n} rows:")
                print(df.head(max_rows).to_string(index=False))
            else:
                print(df.to_string(index=False))

            if "node_index" in df.columns:
                unique_nodes = int(pd.Series(df["node_index"]).nunique(dropna=True))
                print(f"Unique node indices: {unique_nodes}")
        except Exception as e:
            print(f"[ASL] Could not print source DataFrame: {e!r}")

        return df


    def source_to_csv(self, csvfile: str, index: bool = False):
        """
        Write the current ASL source track to CSV.

        Output
        ------
        - `t` column in ISO-8601 UTC (YYYY-MM-DDTHH:MM:SS.ssssssZ).
        - Stable, analysis-friendly column ordering.
        - Only writes if a source is available and non-empty.
        """
        df = self.source_to_dataframe()
        if df is None or df.empty:
            print("[ASL] No source to write.")
            return None

        # Stable column order
        preferred = ["t", "lat", "lon", "DR", "misfit", "azgap",
                    "nsta", "node_index", "connectedness"]
        cols = [c for c in preferred if c in df.columns] \
            + [c for c in df.columns if c not in preferred]
        df = df.loc[:, cols]

        # Format `t` as ISO-8601 with trailing Z
        if "t" in df.columns and pd.api.types.is_datetime64_any_dtype(df["t"]):
            df["t"] = df["t"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # Ensure destination folder exists
        Path(csvfile).parent.mkdir(parents=True, exist_ok=True)

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
        wave_speed_kms: float | None = None,# if None, falls back to self.wave_speed_kms
        verbose: bool = True,
    ) -> dict:
        """
        Estimate geometric spreading exponent N and attenuation factor k
        from station amplitudes using a log-linear decay model:

            A(R) ≈ A0 · R^{-N} · exp(-kR)
            log A = b0 + b1·log R + b2·R,   with   N = -b1,   k = -b2.

        Converts k → Q via  Q = (π f) / (k v)  when frequency f and wave speed v are available.

        Assumptions
        -----------
        - `self.node_distances_km` is available and contains, for each seed id,
        a vector of distances (km) to **every** grid node (flattened order).

        Parameters
        ----------
        time_index : int or None, default None
            Time sample to fit. If None, uses the time of maximum DR in `self.source["DR"]`.
        node_index : int or None, default None
            Grid node index to use. If None, picks the nearest grid node to the source
            (lat, lon) at the chosen time.
        use_reduced : bool, default False
            If True, multiply raw amplitudes by per-station amplitude corrections at the
            chosen node (if available). If False, use raw amplitudes from `metric2stream()`.
        bounded : bool, default False
            If True and SciPy is available, perform a constrained fit with N ≥ 0 and k ≥ 0
            using `scipy.optimize.lsq_linear`. Falls back to unconstrained OLS otherwise.
        neighborhood_k : int or None, default None
            If >1, also evaluate the nearest `k` nodes to `node_index` and return the fit
            with the highest R² (includes a `candidates` list).
        min_stations : int, default 5
            Minimum number of usable stations required; otherwise returns method="insufficient".
        eps_amp : float, default 1e-12
            Amplitude floor to avoid log(0).
        eps_dist_km : float, default 1e-6
            Distance floor (km) to avoid log(0).
        freq_hz : float or None, default None
            Frequency for Q conversion. Defaults to `self.peakf` if None.
        wave_speed_kms : float or None, default None
            Wave speed (km/s) for Q conversion. Defaults to `self.wave_speed_kms` if None.
        verbose : bool, default True
            If True, print a compact fit summary.

        Returns
        -------
        dict
            {
            "N","k","Q","A0_log","r2","nsta",
            "time_index","node_index","lat","lon",
            "se_N","se_k","se_A0","method",
            # present only if neighborhood_k > 1:
            "candidates": List[Tuple[node_index, r2]]
            }

        Notes
        -----
        - Requires a minimal spread in station distances for a well-posed fit.
        If the span is too small (ΔR < ~2 km or ΔlogR < ~0.5), returns method="illposed".
        - Methods reported: "OLS", "bounded", "illposed", or "insufficient".
        """
        # ---- choose (t, node) from current source if needed ----
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

        # ---- convenience handles ----
        glat = self.gridobj.gridlat.reshape(-1)
        glon = self.gridobj.gridlon.reshape(-1)
        nnodes = glat.size

        if not (0 <= node_index < nnodes):
            raise IndexError("node_index out of bounds.")

        # frequency/velocity for Q
        f_hz = float(freq_hz if freq_hz is not None else getattr(self, "peakf", np.nan))
        v_kms = float(wave_speed_kms if wave_speed_kms is not None else getattr(self, "wave_speed_kms", np.nan))

        # distances-at-node helper (assumes node_distances_km is complete)
        def _distances_for_node(nj: int, seed_ids: list[str]) -> np.ndarray:
            d_km = np.empty(len(seed_ids), dtype=float)
            for k, sid in enumerate(seed_ids):
                vec = np.asarray(self.node_distances_km[sid], float)
                if vec.size != nnodes:
                    raise ValueError(f"Distance vector length mismatch for {sid}: {vec.size} != {nnodes}")
                d_km[k] = float(vec[nj])
            return d_km

        def _fit_once(ti: int, nj: int) -> dict:
            # station amplitudes at a single time
            st = self.metric2stream()
            seed_ids = [tr.id for tr in st]
            Y = np.vstack([tr.data.astype(np.float64, copy=False) for tr in st])
            a_raw = Y[:, ti]  # (nsta,)

            # choose amplitude vector (reduced or raw)
            if use_reduced:
                C = np.ones_like(a_raw, dtype=np.float64)
                for k, sid in enumerate(seed_ids):
                    ck = self.amplitude_corrections.get(sid)
                    if ck is not None:
                        val = np.asarray(ck, float)[nj]
                        C[k] = val if np.isfinite(val) else 1.0
                A = a_raw * C
            else:
                A = a_raw

            # distances (precomputed, guaranteed)
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

            # simple well-posedness checks
            if (np.nanmax(R) - np.nanmin(R)) < 2.0 or (np.ptp(logR) < 0.5):
                return dict(N=np.nan, k=np.nan, Q=np.nan, A0_log=np.nan, r2=-np.inf,
                            nsta=nuse, time_index=ti, node_index=nj,
                            lat=lat_j, lon=lon_j, se_N=np.nan, se_k=np.nan, se_A0=np.nan,
                            method="illposed")

            # design matrix: [1, logR, R]
            X = np.column_stack([np.ones_like(logA), logR, R])

            b = None
            cov = None
            used_method = "OLS"

            if bounded:
                # bounded fit (N>=0, k>=0) ↔ b1<=0, b2<=0
                try:
                    from scipy.optimize import lsq_linear
                    # center predictors to stabilize; recover intercept from means
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

        # nearest-k nodes to the starting node (planar metric is OK on local grids)
        d2 = (glat - float(glat[int(node_index)])) ** 2 + (glon - float(glon[int(node_index)])) ** 2
        cand = np.argsort(d2)[:int(neighborhood_k)]

        fits = [_fit_once(int(time_index), int(j)) for j in cand]
        best = max(
            fits,
            key=lambda p: (np.nan_to_num(p.get("r2", np.nan), nan=-np.inf),
                        -abs(p.get("N", np.nan)) if np.isfinite(p.get("N", np.nan)) else -np.inf)
        )
        
        best["candidates"]= [(int(f["node_index"]), float(f["r2"])) for f in fits]
        B = best['candidates']
        print(f"A={B['A0']}*R^{B['N']}*exp(-{B['k']}R)")
        print(f"Q={B['Q']}, f={freq_hz}Hz, c={wave_speed_kms}km/s ")
        return best
