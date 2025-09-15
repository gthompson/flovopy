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

from .map import topo_map
from .grid import Grid  # for typing only

# ---------- helpers ----------
def compute_azimuthal_gap(origin_lat, origin_lon, station_coords):
    azimuths = []
    for stalat, stalon in station_coords:
        _, az, _ = gps2dist_azimuth(origin_lat, origin_lon, stalat, stalon)
        azimuths.append(az)
    if len(azimuths) < 2:
        return 360.0, len(azimuths)
    azimuths = sorted(azimuths)
    azimuths.append(azimuths[0] + 360.0)
    gaps = [azimuths[i + 1] - azimuths[i] for i in range(len(azimuths) - 1)]
    return max(gaps), len(azimuths) - 1


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
    def locate(self, *, min_stations: int = 3, eps: float = 1e-9):
        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)

        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        nsta = len(seed_ids)
        lendata = len(st[0].data)
        t = st[0].times("utcdatetime")

        # Y: (nsta, lendata)
        Y = np.vstack([tr.data.astype(float) for tr in st])

        # C: (nsta, nnodes)
        first_corr = self.amplitude_corrections[seed_ids[0]]
        nnodes = len(first_corr)
        if not (nnodes == gridlat.size == gridlon.size):
            raise AssertionError(f"Node count mismatch: corr={nnodes}, grid={gridlat.size}")

        C = np.empty((nsta, nnodes), dtype=float)
        for k, sid in enumerate(seed_ids):
            ck = np.asarray(self.amplitude_corrections[sid], dtype=float)
            if ck.size != nnodes:
                raise ValueError(f"Corrections length mismatch for {sid}: {ck.size} != {nnodes}")
            C[k, :] = ck

        station_coords = []
        for sid in seed_ids:
            coords = self.station_coordinates.get(sid)
            if coords:
                station_coords.append((coords["latitude"], coords["longitude"]))

        source_DR     = np.empty(lendata, dtype=float)
        source_lat    = np.empty(lendata, dtype=float)
        source_lon    = np.empty(lendata, dtype=float)
        source_misfit = np.empty(lendata, dtype=float)
        source_azgap  = np.empty(lendata, dtype=float)
        source_nsta   = np.empty(lendata, dtype=int)

        for i in range(lendata):
            y = Y[:, i]
            best_j = -1
            best_m = np.inf

            # scan nodes
            for j in range(nnodes):
                reduced = y * C[:, j]
                finite = np.isfinite(reduced)
                if int(finite.sum()) < min_stations:
                    continue
                r = reduced[finite]
                med = np.nanmedian(r)
                if not np.isfinite(med) or abs(med) < eps:
                    continue
                m = float(np.nanstd(r) / (abs(med) + eps))
                if m < best_m:
                    best_m, best_j = m, j

            if best_j < 0:
                # fallback if all nodes filtered out
                for j in range(nnodes):
                    r = (y * C[:, j])[np.isfinite(y * C[:, j])]
                    if r.size == 0:
                        continue
                    med = np.nanmedian(r)
                    if not np.isfinite(med) or abs(med) < eps:
                        continue
                    m = float(np.nanstd(r) / (abs(med) + eps))
                    if m < best_m:
                        best_m, best_j = m, j

            reduced_best = y * C[:, best_j]
            rbest = reduced_best[np.isfinite(reduced_best)]
            source_DR[i] = np.nanmedian(rbest) if rbest.size else np.nan
            source_lat[i], source_lon[i] = gridlat[best_j], gridlon[best_j]
            source_misfit[i] = best_m

            azgap, nsta_eff = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
            source_azgap[i], source_nsta[i] = azgap, nsta_eff

        self.source = {
            "t": t,
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
        }
        self.source_to_obspyevent()
        self.located = True
        self.id = self.set_id()
        return
    

    def fast_locate(
        self,
        *,
        min_stations: int = 3,
        eps: float = 1e-9,
        use_median_for_DR: bool = False,
        batch_size: int = 1024,
        verbose: bool = True,
    ):
        """
        Fast amplitude-based location using vectorized reductions.

        For each time sample t:
        - Reduce station amplitudes to each node j: R[:, j] = Y[:, t] * C[:, j]
        - Compute misfit per node: std(R[:, j]) / (mean(R[:, j]) + eps)  [finite stations only]
        - Pick node with minimum misfit.
        - DR(t) = median(R[:, j*]) if use_median_for_DR else mean(R[:, j*])

        Parameters
        ----------
        min_stations : int
            Minimum finite stations required to evaluate a node at a given time.
        eps : float
            Small number to stabilize divisions.
        use_median_for_DR : bool
            If True, use median across stations for DR; else mean (slightly faster).
        batch_size : int
            Number of time samples to process per batch (controls memory usage).
        verbose : bool
            Print progress info.
        """
        if verbose:
            print("[ASL] fast_locate: preparing data…")

        gridlat = self.gridobj.gridlat.reshape(-1)
        gridlon = self.gridobj.gridlon.reshape(-1)

        # 1) Stream → matrix Y (nsta, ntime)
        st = self.metric2stream()
        seed_ids = [tr.id for tr in st]
        nsta = len(seed_ids)
        if nsta == 0:
            raise RuntimeError("No traces in stream for fast_locate().")
        Y = np.vstack([tr.data.astype(float) for tr in st])  # (nsta, ntime)
        ntime = Y.shape[1]
        t = st[0].times("utcdatetime")

        # 2) Corrections → matrix C (nsta, nnodes)
        first_corr = self.amplitude_corrections[seed_ids[0]]
        nnodes = len(first_corr)
        if nnodes != gridlat.size or nnodes != gridlon.size:
            raise AssertionError(f"Node count mismatch: corr={nnodes}, grid={gridlat.size}")
        C = np.empty((nsta, nnodes), dtype=float)
        for k, sid in enumerate(seed_ids):
            ck = np.asarray(self.amplitude_corrections[sid], dtype=float)
            if ck.size != nnodes:
                raise ValueError(f"Corrections length mismatch for {sid}: {ck.size} != {nnodes}")
            C[k, :] = ck

        # 3) Station coords (for azgap)
        station_coords = []
        for sid in seed_ids:
            coords = self.station_coordinates.get(sid)
            if coords:
                station_coords.append((coords["latitude"], coords["longitude"]))

        # 4) Outputs
        source_DR     = np.empty(ntime, dtype=float)
        source_lat    = np.empty(ntime, dtype=float)
        source_lon    = np.empty(ntime, dtype=float)
        source_misfit = np.empty(ntime, dtype=float)
        source_azgap  = np.empty(ntime, dtype=float)
        source_nsta   = np.empty(ntime, dtype=int)

        if verbose:
            print(f"[ASL] fast_locate: nsta={nsta}, ntime={ntime}, nnodes={nnodes}, batch_size={batch_size}")

        # 5) Batched time loop
        for i0 in range(0, ntime, batch_size):
            i1 = min(i0 + batch_size, ntime)
            if verbose:
                print(f"[ASL] fast_locate: processing samples [{i0}:{i1})")

            # Yb: (nsta, B)
            Yb = Y[:, i0:i1]  # B = i1-i0

            # For each time column, find best node
            # We'll process one time column at a time to limit memory: (nsta, nnodes)
            for k in range(Yb.shape[1]):
                y = Yb[:, k]  # (nsta,)

                # Broadcast to nodes: reduced amplitudes R = y[:,None] * C  (nsta, nnodes)
                R = y[:, None] * C

                # Finite mask per station/node
                finite = np.isfinite(R)

                # Count usable stations per node
                nfin = finite.sum(axis=0)  # (nnodes,)

                # Guard: nodes with too few stations → set misfit to +inf
                valid_nodes = nfin >= min_stations

                # Compute per-node stats on finite values only
                # mean and std along station axis
                # Use masked arrays for nan-safe stats
                R_masked = np.ma.array(R, mask=~finite)
                mean_j = R_masked.mean(axis=0).filled(np.nan)   # (nnodes,)
                std_j  = R_masked.std(axis=0, ddof=0).filled(np.nan)

                # Misfit = std/mean (stabilized)
                misfit = std_j / (np.abs(mean_j) + eps)
                misfit[~valid_nodes] = np.inf
                jstar = int(np.nanargmin(misfit))  # best node index

                # DR(t) across stations at best node
                rbest = R[:, jstar]
                rbest = rbest[np.isfinite(rbest)]
                if rbest.size == 0:
                    DR_t = np.nan
                    nsta_eff = 0
                else:
                    DR_t = (np.median(rbest) if use_median_for_DR else np.mean(rbest))
                    nsta_eff = rbest.size

                i = i0 + k
                source_DR[i]     = DR_t
                source_lat[i]    = gridlat[jstar]
                source_lon[i]    = gridlon[jstar]
                source_misfit[i] = float(misfit[jstar])
                # azgap for this location (station set fixed)
                azgap, nsta_all = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
                source_azgap[i] = azgap
                source_nsta[i]  = nsta_eff

        # 6) Package results (note: your DR scale factor of 1e7 kept for continuity)
        self.source = {
            "t": t,
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
        }
        self.source_to_obspyevent()
        self.located = True
        if hasattr(self, "set_id") and callable(self.set_id):
            self.id = self.set_id()
        if verbose:
            print("[ASL] fast_locate: done.")
        return

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


    def print_source(self, max_rows: int = 20):
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