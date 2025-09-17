# flovopy/asl/asl.py
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from obspy import UTCDateTime, Stream
from obspy.core.event import Event, Catalog, ResourceIdentifier, Origin, Amplitude, QuantityError, OriginQuality, Comment
from obspy.geodetics import gps2dist_azimuth

from flovopy.processing.sam import VSAM
from flovopy.utils.make_hash import make_hash
from flovopy.core.mvo import dome_location
from flovopy.asl.map import topo_map #, plot_heatmap_montserrat_colored
from flovopy.asl.distances import compute_distances, distances_signature
from flovopy.asl.diagnostics import extract_asl_diagnostics, compare_asl_sources
from flovopy.asl.grid import make_grid, Grid
from flovopy.asl.misfit import plot_misfit_heatmap_for_peak_DR, StdOverMeanMisfit, R2DistanceMisfit

from scipy.ndimage import gaussian_filter

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

    def __init__(self, samobject: VSAM, metric: str, gridobj: Grid, window_seconds: int):
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
            grid_sig=self.gridobj.signature(),
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

        # Stream → Y (nsta, ntime)
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        Y = np.vstack([tr.data.astype(np.float32, copy=False) for tr in st])
        t = st[0].times("utcdatetime")
        nsta, ntime = Y.shape

        # Corrections → C (nsta, nnodes)
        C = np.empty((nsta, nnodes), dtype=np.float32)
        for k, sid in enumerate(seed_ids):
            vec = np.asarray(self.amplitude_corrections[sid], dtype=np.float32)
            if vec.size != nnodes:
                raise ValueError(f"Corrections length mismatch for {sid}: {vec.size} != {nnodes}")
            C[k, :] = vec

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

            misfit_j = np.full(nnodes, np.inf, dtype=float)
            dr_j     = np.full(nnodes, np.nan, dtype=float)
            nused_j  = np.zeros(nnodes, dtype=int)

            # Scan all nodes
            for j in range(nnodes):
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
            jstar = int(np.nanargmin(misfit_for_pick))

            # Record outputs at time i
            source_lat[i]    = gridlat[jstar]
            source_lon[i]    = gridlon[jstar]
            source_misfit[i] = float(misfit_j[jstar])  # store the *unblurred* misfit at jstar
            source_DR[i]     = float(dr_j[jstar])
            source_nsta[i]   = int(nused_j[jstar])

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
        spatial_smooth_sigma: float = 0.0,   # NEW: Gaussian sigma (in node units)
        temporal_smooth_win: int = 0,        # NEW: centered window (odd int); 0 disables
        verbose: bool = True,
    ):
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

        # Backend context
        ctx = misfit_backend.prepare(self, seed_ids, dtype=np.float32)

        # Station coords (azgap)
        station_coords = []
        for sid in seed_ids:
            c = self.station_coordinates.get(sid)
            if c: station_coords.append((c["latitude"], c["longitude"]))

        # Outputs
        source_DR     = np.empty(ntime, dtype=float)
        source_lat    = np.empty(ntime, dtype=float)
        source_lon    = np.empty(ntime, dtype=float)
        source_misfit = np.empty(ntime, dtype=float)
        source_azgap  = np.empty(ntime, dtype=float)
        source_nsta   = np.empty(ntime, dtype=int)

        # NEW: can we reshape misfit to (nlat,nlon) for spatial blur?
        grid_shape = _grid_shape_or_none(self.gridobj)
        can_blur = (
            spatial_smooth_sigma and spatial_smooth_sigma > 0.0 and
            (grid_shape is not None) and (gaussian_filter is not None)
        )
        if verbose:
            print(f"[ASL] fast_locate: nsta={nsta}, ntime={ntime}, batch={batch_size}, "
                f"spatial_blur={'on' if can_blur else 'off'}, "
                f"temporal_smooth_win={temporal_smooth_win or 0}")
        chosen = np.full(ntime, -1, dtype=int)
        for i0 in range(0, ntime, batch_size):
            i1 = min(i0 + batch_size, ntime)
            if verbose: print(f"[ASL] fast_locate: [{i0}:{i1})")
            Yb = Y[:, i0:i1]

            for k in range(Yb.shape[1]):
                y = Yb[:, k]
                misfit, extras = misfit_backend.evaluate(y, ctx, min_stations=min_stations, eps=eps)

                # NEW: optional spatial smoothing of per-time misfit
                if can_blur:
                    nlat, nlon = grid_shape
                    m2 = misfit.reshape(nlat, nlon)
                    m2 = gaussian_filter(m2, sigma=spatial_smooth_sigma, mode="nearest")
                    misfit = m2.reshape(-1)

                jstar = int(np.nanargmin(misfit))

                # DR: from backend if available
                DR_t = extras.get("mean", None)
                if DR_t is None:
                    DR_t = extras.get("DR", None)
                if DR_t is not None and np.ndim(DR_t) == 1:
                    DR_t = DR_t[jstar]
                else:
                    # fallback compute using corrections for jstar (median optional)
                    C = ctx.get("C", None)
                    if C is not None:
                        rbest = y * C[:, jstar]
                        rbest = rbest[np.isfinite(rbest)]
                        if rbest.size:
                            DR_t = np.median(rbest) if use_median_for_DR else np.mean(rbest)
                        else:
                            DR_t = np.nan
                    else:
                        DR_t = np.nan

                i = i0 + k
                chosen[i] = jstar
                source_DR[i]     = DR_t
                source_lat[i]    = gridlat[jstar]
                source_lon[i]    = gridlon[jstar]
                source_misfit[i] = float(misfit[jstar])

                # station count used at jstar if backend reports vector N
                N = extras.get("N", None)
                source_nsta[i] = int(N if np.isscalar(N) else (N[jstar] if N is not None else np.sum(np.isfinite(y))))

                azgap, _ = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
                source_azgap[i] = azgap

        # NEW: optional temporal smoothing of tracks
        if temporal_smooth_win and temporal_smooth_win >= 3 and temporal_smooth_win % 2 == 1:
            source_lat = _movavg_1d(source_lat, temporal_smooth_win)
            source_lon = _movavg_1d(source_lon, temporal_smooth_win)
            source_DR  = _movavg_1d(source_DR,  temporal_smooth_win)

        self.source = {
            "t": t,
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
            "node_index": chosen,
        }

        # Connectedness (unchanged)
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
    def plot(self, zoom_level=1, threshold_DR=0, scale=1, join=False, number=0,
             add_labels=False, outfile=None, stations=None, title=None, region=None, normalize=True):
        source = self.source
        if not source:
            fig = topo_map(zoom_level=zoom_level, inv=None, show=True, add_labels=add_labels)
            return

        t_dt = [this_t.datetime for this_t in source["t"]]
        plt.figure()
        plt.plot(t_dt, source["DR"])
        plt.plot(t_dt, np.ones(source["DR"].size) * threshold_DR)
        plt.xlabel("Date/Time (UTC)"); plt.ylabel("Reduced Displacement (${cm}^2$)")

        if threshold_DR > 0:
            mask = source["DR"] < threshold_DR
            source["DR"][mask] = 0.0
            source["lat"][mask] = None
            source["lon"][mask] = None

        x, y, DR = source["lon"], source["lat"], source["DR"]
        symsize = (scale * np.ones_like(DR) if normalize is False and number
                   else (scale * np.ones_like(DR) if normalize and np.nanmax(DR) == 0
                         else (np.divide(DR, np.nanmax(DR)) * scale if normalize else scale * np.sqrt(np.maximum(DR, 0)))))

        maxi = np.nanargmax(DR)
        fig = topo_map(zoom_level=zoom_level, inv=None, centerlat=y[maxi], centerlon=x[maxi],
                       add_labels=add_labels, topo_color=False, stations=stations, title=title, region=region)

        if number and number < len(x):
            ind = np.argpartition(DR, -number)[-number:]
            x, y, DR, symsize = x[ind], y[ind], DR[ind], symsize[ind]

        pygmt = __import__("pygmt")
        pygmt.makecpt(cmap="viridis", series=[0, len(x)])
        timecolor = list(range(len(x)))
        fig.plot(x=x, y=y, size=symsize, style="cc", pen=None, fill=timecolor, cmap=True)
        fig.colorbar(frame='+l"Sequence"')

        if region:
            fig.basemap(region=region, frame=True)
        if outfile:
            fig.savefig(outfile)
        else:
            fig.show()
        if join:
            fig.plot(x=x, y=y, style="r-", pen="1p,red")

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

            distances_km = []
            for coords in self.station_coordinates.values():
                dist_m, _, _ = gps2dist_azimuth(lat, lon, coords["latitude"], coords["longitude"])
                distances_km.append(dist_m / 1000.0)
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
            from pprint import pprint
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

        import pandas as pd
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
        min_stations: int = 5,
        eps_amp: float = 1e-12,
        eps_dist_km: float = 1e-6,
        freq_hz: float | None = None,      # if None, uses self.peakf (if available)
        wavespeed_kms: float | None = None, # if None, uses self.wavespeed_kms (if available)
        use_reduced: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Estimate geometric spreading exponent N and attenuation Q from
        a single time sample and node, by fitting:

            log A  =  log A0  - N * log R  - k * R,

        where R is distance (km). If `freq_hz` and `wavespeed_kms` are provided
        (or found on the object), convert k to Q via Q = π f / (k v).

        Parameters
        ----------
        time_index : int or None
            Which time sample to use. If None, chooses the time of max DR
            if self.source exists; else raises.
        node_index : int or None
            Which grid node to use for distances. If None, uses the best node
            at the chosen time (argmin misfit) if available; else raises.
        min_stations : int
            Minimum usable stations required to fit.
        eps_amp : float
            Small floor added inside log() for amplitudes (to avoid -inf).
        eps_dist_km : float
            Small floor for distances (to avoid log(0)).
        freq_hz, wavespeed_kms : float or None
            Needed to convert k→Q. If None, fall back to self.peakf / self.wavespeed_kms.
        use_reduced : bool
            If True (default), fit uses *reduced* station amplitudes at the chosen
            node, i.e., y * C[:, j]. If False, uses raw VSAM metric y.
            (Reduced is usually more stable.)
        verbose : bool
            Print a short summary.

        Returns
        -------
        dict with fields:
        - N : geometric spreading exponent (unitless)
        - k : attenuation coefficient (1/km)
        - Q : estimated quality factor (or np.nan if freq/velocity missing or k<=0)
        - A0_log : intercept (log A0)
        - r2 : coefficient of determination
        - nsta : number of stations used
        - time_index, node_index : indices used
        - lat, lon : node coordinates
        """
        # --- choose time index ---
        st = self.metric2stream()
        if time_index is None:
            if getattr(self, "source", None) and "DR" in self.source:
                time_index = int(np.nanargmax(self.source["DR"]))
            else:
                raise ValueError("time_index=None and no self.source['DR'] available to infer it.")
        if time_index < 0 or time_index >= len(st[0].data):
            raise IndexError("time_index out of bounds.")

        # data matrix Y (nsta, ntime)
        Y = np.vstack([tr.data.astype(np.float64, copy=False) for tr in st])
        y = Y[:, time_index]  # (nsta,)

        # seed ids order
        seed_ids = [tr.id for tr in st]
        nsta = len(seed_ids)

        # --- choose node index ---
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)
        nnodes = gridlat.size

        if node_index is None:
            # try to infer from misfit minimum if available
            if getattr(self, "source", None) and "lat" in self.source and "lon" in self.source:
                # pick the node nearest to the stored lat/lon at that time
                lat_t = float(self.source["lat"][time_index])
                lon_t = float(self.source["lon"][time_index])
                if np.isfinite(lat_t) and np.isfinite(lon_t):
                    d2 = (gridlat - lat_t)**2 + (gridlon - lon_t)**2
                    node_index = int(np.nanargmin(d2))
                else:
                    raise ValueError("node_index=None and stored lat/lon are NaN at this time.")
            else:
                raise ValueError("node_index=None and no prior location available to infer it.")

        if node_index < 0 or node_index >= nnodes:
            raise IndexError("node_index out of bounds.")

        # --- distances for that node (km), one per station ---
        R = np.empty(nsta, dtype=np.float64)
        for k, sid in enumerate(seed_ids):
            dk = np.asarray(self.node_distances_km.get(sid), dtype=np.float64)
            if dk.size != nnodes:
                raise ValueError(f"Distances for {sid} length {dk.size} != grid nodes {nnodes}")
            R[k] = dk[node_index]

        # station subset with finite amplitudes and positive distances
        mask = np.isfinite(y) & np.isfinite(R) & (R > eps_dist_km)
        if mask.sum() < min_stations:
            return {
                "N": np.nan, "k": np.nan, "Q": np.nan, "A0_log": np.nan,
                "r2": 0.0, "nsta": int(mask.sum()),
                "time_index": int(time_index), "node_index": int(node_index),
                "lat": float(gridlat[node_index]), "lon": float(gridlon[node_index]),
            }

        y_use = y[mask]

        # optionally reduce y using amplitude corrections at this node
        if use_reduced:
            C = np.empty(mask.sum(), dtype=np.float64)
            kk = 0
            for m, sid in zip(mask, seed_ids):
                if not m:
                    continue
                ck = self.amplitude_corrections.get(sid)
                if ck is None:
                    C[kk] = 1.0
                else:
                    ck = np.asarray(ck, dtype=np.float64)
                    C[kk] = ck[node_index] if np.isfinite(ck[node_index]) else 1.0
                kk += 1
            A = y_use * C
        else:
            A = y_use

        R_use = R[mask].astype(np.float64)

        # --- linear least squares on log(A) = b0 + b1*log(R) + b2*R  ---
        logA = np.log(np.maximum(A, eps_amp))
        logR = np.log(np.maximum(R_use, eps_dist_km))
        X = np.column_stack([np.ones_like(logR), logR, R_use])  # [1, logR, R]
        # Solve
        beta, *_ = np.linalg.lstsq(X, logA, rcond=None)
        b0, b1, b2 = beta
        # coefficients map to model: logA = logA0 - N*logR - k*R
        N_hat = -float(b1)
        k_hat = -float(b2)
        A0_log = float(b0)

        # R^2
        yhat = X @ beta
        ss_res = float(np.sum((logA - yhat)**2))
        ss_tot = float(np.sum((logA - np.mean(logA))**2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # convert k -> Q if possible
        f = float(freq_hz if freq_hz is not None else (getattr(self, "peakf", np.nan) or np.nan))
        v = float(wavespeed_kms if wavespeed_kms is not None else (getattr(self, "wavespeed_kms", np.nan) or np.nan))
        if np.isfinite(f) and np.isfinite(v) and k_hat > 0:
            Q_hat = float(np.pi * f / (k_hat * v))
        else:
            Q_hat = np.nan

        out = {
            "N": N_hat,
            "k": k_hat,             # 1/km
            "Q": Q_hat,
            "A0_log": A0_log,
            "r2": float(r2),
            "nsta": int(mask.sum()),
            "time_index": int(time_index),
            "node_index": int(node_index),
            "lat": float(gridlat[node_index]),
            "lon": float(gridlon[node_index]),
        }

        if verbose:
            fstr = f if np.isfinite(f) else None
            vstr = v if np.isfinite(v) else None
            print(
                f"[ASL] decay-fit @ t={time_index}, node={node_index} "
                f"(lat={out['lat']:.5f}, lon={out['lon']:.5f})  "
                f"N={out['N']:.3f}, k={out['k']:.5g} 1/km, "
                f"Q={out['Q']:.1f} (f={fstr}, v={vstr})  r2={out['r2']:.3f}  nsta={out['nsta']}"
            )
        return out    
    
    def estimate_decay_params(
        self,
        time_index: int | None = None,
        node_index: int | None = None,
        *,
        use_reduced: bool = False,      # ← default: fit on raw amplitudes
        bounded: bool = False,          # enforce N>=0, k>=0 if SciPy available
        neighborhood_k: int | None = None,  # e.g. 9 → search nearest 9 nodes, pick best R²
        min_stations: int = 3,
        eps: float = 1e-12,
        verbose: bool = False,
    ) -> dict:
        """
        Estimate geometric decay exponent N and attenuation factor k in:
            A(R) ≈ A0 * R^{-N} * exp(-k R)
        where Q ≈ (π f) / (k v). Uses a log-linear model:

            log A = b0 + b1*log R + b2*R,   with  N = -b1,  k = -b2  (so we expect b1<=0, b2<=0)

        Parameters
        ----------
        time_index : int or None
            Time sample to use. If None, pick DR maximum from self.source (requires locate()).
        node_index : int or None
            Grid node index to use. If None, pick the node at that time from self.source (requires locate()).
        use_reduced : bool
            If True, fit on amplitudes already multiplied by amplitude corrections (geometric/attenuation removed).
            Usually set False to keep physical distance-dependence in the data.
        bounded : bool
            If True and SciPy is available, solve with bounds b1<=0, b2<=0 (i.e., N,k >= 0).
        neighborhood_k : int or None
            If set, evaluate this many nearest nodes to the starting node and return the fit with the highest R².
        min_stations : int
            Require at least this many usable stations.
        eps : float
            Numerical guard for logs/divisions.
        verbose : bool
            Print a one-line summary.

        Returns
        -------
        dict with keys:
        N, k, Q, A0_log, r2, nsta, time_index, node_index, lat, lon,
        se_N, se_k, se_A0 (standard errors; NaN if not available),
        method: "OLS" or "bounded" (if SciPy used),
        candidates: optional list of (node_index, r2) if neighborhood search was used.
        """
        rspan_min_km = 2.0         # reject if distance span too small
        logr_span_min = 0.5        # ~e^0.5 ≈ 1.65× spread in distance
        r2_min = 0.25              # minimum acceptable R^2


        # --- pick time/node defaults from located source if needed ---
        if time_index is None or node_index is None:
            if not getattr(self, "source", None):
                raise ValueError("Provide time_index/node_index or run locate()/fast_locate() first.")
            if time_index is None:
                time_index = int(np.nanargmax(self.source["DR"]))
            if node_index is None:
                # nearest grid node to (lat,lon) at this time
                glat = self.gridobj.gridlat.reshape(-1)
                glon = self.gridobj.gridlon.reshape(-1)
                lat0 = float(self.source["lat"][time_index])
                lon0 = float(self.source["lon"][time_index])
                node_index = int(np.nanargmin((glat - lat0)**2 + (glon - lon0)**2))

        def _fit_once(ti: int, nj: int) -> dict:
            # Stream → per-station scalar at this time
            st = self.metric2stream()
            seed_ids = [tr.id for tr in st]
            y = np.vstack([tr.data.astype(np.float64, copy=False) for tr in st])[:, ti]  # (nsta,)

            # distances to this node
            glat = self.gridobj.gridlat.reshape(-1)
            glon = self.gridobj.gridlon.reshape(-1)
            lat_j = float(glat[nj]); lon_j = float(glon[nj])

            # build distances (km) from stored station coordinates
            d_km = []
            for sid in seed_ids:
                c = self.station_coordinates.get(sid)
                if c is None:
                    d_km.append(np.nan); continue
                from obspy.geodetics import gps2dist_azimuth
                dm, _, _ = gps2dist_azimuth(lat_j, lon_j, c["latitude"], c["longitude"])
                d_km.append(dm / 1000.0)
            d_km = np.asarray(d_km, float)

            # choose amplitude vector
            if use_reduced:
                C = np.vstack([self.amplitude_corrections[sid] for sid in seed_ids])  # (nsta, nnodes)
                a = y * C[:, nj]
            else:
                a = y

            # validity mask
            m = np.isfinite(a) & np.isfinite(d_km) & (a > 0.0) & (d_km > 0.0)
            if int(m.sum()) < min_stations:
                return dict(N=np.nan, k=np.nan, Q=np.nan, A0_log=np.nan, r2=0.0,
                            nsta=int(m.sum()), time_index=ti, node_index=nj,
                            lat=lat_j, lon=lon_j, se_N=np.nan, se_k=np.nan, se_A0=np.nan,
                            method="insufficient")

            A = a[m]
            R = d_km[m]
            logA = np.log(A + eps)
            logR = np.log(R + eps)

            # quick well-posedness checks
            if (np.nanmax(R) - np.nanmin(R)) < rspan_min_km or (np.ptp(logR) < logr_span_min):
                return dict(N=np.nan, k=np.nan, Q=np.nan, A0_log=np.nan, r2=-np.inf,
                            nsta=int(m.sum()), time_index=ti, node_index=nj,
                            lat=lat_j, lon=lon_j, se_N=np.nan, se_k=np.nan, se_A0=np.nan,
                            method="illposed")

            # Design matrix: [1, logR, R]
            X = np.column_stack([np.ones_like(logA), logR, R])

            used_method = "OLS"
            b = None
            cov = None

            if bounded:
                # optional bounded fit (requires SciPy)
                try:
                    from scipy.optimize import lsq_linear
                    # center predictors to reduce collinearity
                    logR_c = logR - logR.mean()
                    R_c = R - R.mean()
                    Xc = np.column_stack([logR_c, R_c])  # we fit without intercept, then recover it
                    y_c = logA - logA.mean()
                    res = lsq_linear(Xc, y_c, bounds=([-np.inf, -np.inf], [0.0, 0.0]))
                    b1_c, b2_c = res.x
                    b0 = logA.mean()  # since predictors are centered
                    b = np.array([b0, b1_c, b2_c], float)
                    used_method = "bounded"
                except Exception:
                    # fall back to OLS
                    pass

            if b is None:
                # ordinary least squares with pseudo-inverse
                b, *_ = np.linalg.lstsq(X, logA, rcond=None)
                # covariance (X'X)^{-1} * sigma^2
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

            # Q from k (prefer instance params if present)
            f_hz = float(getattr(self, "peakf", np.nan))
            v_kms = float(getattr(self, "wavespeed_kms", np.nan))
            if np.isfinite(f_hz) and np.isfinite(v_kms) and k_hat >= 0:
                Q_hat = (np.pi * f_hz) / (k_hat * v_kms) if k_hat > 0 else np.inf
            else:
                Q_hat = np.nan

            # R²
            yhat = X @ b
            ss_res = float(np.sum((logA - yhat) ** 2))
            ss_tot = float(np.sum((logA - logA.mean()) ** 2)) + eps
            r2 = 1.0 - ss_res / ss_tot
            if not np.isfinite(r2) or r2 < r2_min:
                return dict(N=np.nan, k=np.nan, Q=np.nan, A0_log=b0, r2=r2,
                            nsta=int(m.sum()), time_index=ti, node_index=nj,
                            lat=lat_j, lon=lon_j, se_N=np.nan, se_k=np.nan, se_A0=np.nan,
                            method=("bounded" if used_method=="bounded" else "OLS_reject"))

            # standard errors
            se_A0 = se_N = se_k = np.nan
            if cov is not None and cov.shape == (3, 3):
                se_A0 = float(np.sqrt(max(0.0, cov[0, 0])))
                se_N  = float(np.sqrt(max(0.0, cov[1, 1])))
                se_k  = float(np.sqrt(max(0.0, cov[2, 2])))

            out = dict(N=N_hat, k=k_hat, Q=Q_hat, A0_log=b0, r2=r2,
                    nsta=int(m.sum()), time_index=ti, node_index=nj,
                    lat=lat_j, lon=lon_j,
                    se_N=se_N, se_k=se_k, se_A0=se_A0,
                    method=used_method)
            if verbose:
                print(f"[ASL] decay-fit @ t={ti}, node={nj} (lat={lat_j:.5f}, lon={lon_j:.5f})  "
                    f"N={N_hat:.3f}, k={k_hat:.5g} 1/km, Q={Q_hat:.1f} (f={getattr(self,'peakf',np.nan)}, "
                    f"v={getattr(self,'wavespeed_kms',np.nan)})  r2={r2:.3f}  nsta={int(m.sum())}")
            return out

        # --- single node or neighborhood search ---
        if neighborhood_k is None or neighborhood_k <= 1:
            return _fit_once(int(time_index), int(node_index))

        # nearest-k nodes to starting node (planar metric is OK for local grid)
        glat = self.gridobj.gridlat.reshape(-1); glon = self.gridobj.gridlon.reshape(-1)
        lat0 = float(glat[int(node_index)]); lon0 = float(glon[int(node_index)])
        d2 = (glat - lat0) ** 2 + (glon - lon0) ** 2
        cand = np.argsort(d2)[:int(neighborhood_k)]

        fits = [_fit_once(int(time_index), int(j)) for j in cand]
        best = max(fits, key=lambda p: (np.nan_to_num(p["r2"], nan=-np.inf), -abs(p["N"]) if np.isfinite(p["N"]) else -np.inf))
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
    def refine_and_relocate(self, *, top_frac: float = 0.20, margin_km: float = 1.5,
                            spatial_sigma_nodes: float = 0.0,
                            temporal_smooth_win: int = 0,
                            verbose: bool = True):
        """
        Take top-fraction DR samples, build a bbox + margin on the current grid,
        subset nodes (and slice ampcorr/distances), then rerun fast_locate().

        No recomputation of amplitude corrections/distances.
        """
        if self.source is None:
            raise RuntimeError("Run fast_locate()/locate() before refine_and_relocate().")

        lat = np.asarray(self.source["lat"], float)
        lon = np.asarray(self.source["lon"], float)
        dr  = np.asarray(self.source["DR"], float)
        ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(dr)
        if ok.sum() < 4:
            if verbose: print("[REFINE] Not enough valid points; skipping.")
            return self

        k = max(8, int(np.ceil(top_frac * ok.sum())))
        sel = np.argsort(dr[ok])[::-1][:k]
        la = lat[ok][sel]; lo = lon[ok][sel]

        # bbox + margin
        km2deg = 1.0 / 111.0
        ddeg = margin_km * km2deg
        la0, la1 = np.nanmin(la) - ddeg, np.nanmax(la) + ddeg
        lo0, lo1 = np.nanmin(lo) - ddeg, np.nanmax(lo) + ddeg

        if verbose:
            print(f"[REFINE] bbox lat[{la0:.5f},{la1:.5f}] lon[{lo0:.5f},{lo1:.5f}] "
                  f"from top {k}/{ok.sum()} samples")

        # subset nodes (slices corrections/distances)
        self.subset_nodes_by_bbox(la0, la1, lo0, lo1, inplace=True)

        # rerun fast_locate with optional smoothing
        self.fast_locate(
            verbose=verbose,
            spatial_smooth_sigma=spatial_sigma_nodes,
            temporal_smooth_win=temporal_smooth_win,
        )
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
import os
from typing import Optional
from obspy import Stream
from flovopy.processing.sam import VSAM
from flovopy.asl.ampcorr import AmpCorr, AmpCorrParams
from flovopy.asl.station_corrections import apply_interval_station_gains

def asl_sausage(
    stream: Stream,
    event_dir: str,
    asl_config: dict,
    output_dir: str,
    dry_run: bool = False,
    peakf_override: Optional[float] = None,
    station_gains_df: Optional["pd.DataFrame"] = None,  # interval table; columns=seed_ids; rows define start_time/end_time
    allow_station_fallback: bool = True,
):
    """
    Run ASL on a single (already preprocessed) event stream.

    Required in asl_config:
      - gridobj : Grid
      - node_distances_km : dict[seed_id -> np.ndarray]
      - station_coords : dict[seed_id -> {latitude, longitude, ...}]
      - ampcorr : AmpCorr
      - vsam_metric, window_seconds, min_stations, interactive, Q, surfaceWaveSpeed_kms

    Optional:
      - station_gains_df : DataFrame with columns:
            start_time, end_time, <seed_id_1>, <seed_id_2>, ...
        We divide each trace by the gain matching the event time interval.
    """
    import numpy as np
    import pandas as pd

    print(f"[ASL] Preparing VSAM for event folder: {event_dir}")
    os.makedirs(event_dir, exist_ok=True)

    # --------------------------
    # 1) Apply station gains (interval table)
    # --------------------------
    if station_gains_df is not None and len(station_gains_df):
        try:
            info = apply_interval_station_gains(
                stream,
                station_gains_df,
                allow_station_fallback=allow_station_fallback,
                verbose=True,
            )
            _ = info  # not used further; printed for transparency
        except Exception as e:
            print(f"[GAINS:WARN] Failed to apply interval gains: {e}")
    else:
        print("[GAINS] No station gains DataFrame provided; skipping.")

    # Ensure velocity units for downstream plots
    for tr in stream:
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
        params = ampcorr.params
        new_params = AmpCorrParams(
            surface_waves=params.surface_waves,
            wave_speed_kms=params.wave_speed_kms,
            Q=params.Q,
            peakf=float(peakf_event),
            grid_sig=params.grid_sig,
            inv_sig=params.inv_sig,
            dist_sig=params.dist_sig,
            mask_sig=params.mask_sig,
            code_version=params.code_version,
        )
        ampcorr = AmpCorr(new_params, cache_dir=ampcorr.cache_dir)
        ampcorr.compute_or_load(asl_config["node_distances_km"])  # inventory not needed here
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

        try:
            print("[ASL:PLOT] Writing map and diagnostic plots…")
            aslobj.plot(
                zoom_level=0,
                threshold_DR=0.0,
                scale=0.2,
                join=True,
                number=0,
                add_labels=True,
                stations=[tr.stats.station for tr in stream],
                outfile=os.path.join(event_dir, f"map_Q{int(aslobj.Q)}_F{int(peakf_event)}.png"),
            )
            aslobj.source_to_csv(os.path.join(event_dir, f"source_Q{int(aslobj.Q)}_F{int(peakf_event)}.csv"))
            aslobj.plot_reduced_displacement(
                outfile=os.path.join(event_dir, f"reduced_disp_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
            )
            aslobj.plot_misfit(
                outfile=os.path.join(event_dir, f"misfit_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
            )
            aslobj.plot_misfit_heatmap(
                outfile=os.path.join(event_dir, f"misfit_heatmap_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
            )
        except Exception:
            print("[ASL:ERR] Plotting failed.")
            raise

        if asl_config.get("interactive", False):
            input("[ASL] Press Enter to continue to next event…")

    return aslobj  # handy if the caller wants the object

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