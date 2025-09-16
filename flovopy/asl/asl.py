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
from flovopy.asl.diagnostics import extract_asl_diagnostics, compare_asl_sources
from flovopy.asl.grid import Grid
from flovopy.asl.misfit import plot_misfit_heatmap_for_peak_DR, StdOverMeanMisfit, R2DistanceMisfit

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
        misfit_backend=None,
        min_stations: int = 3,
        eps: float = 1e-9,
        use_median_for_DR: bool = False,
        batch_size: int = 1024,
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

        if verbose: print(f"[ASL] fast_locate: nsta={nsta}, ntime={ntime}, batch={batch_size}")

        for i0 in range(0, ntime, batch_size):
            i1 = min(i0 + batch_size, ntime)
            if verbose: print(f"[ASL] fast_locate: [{i0}:{i1})")
            Yb = Y[:, i0:i1]

            for k in range(Yb.shape[1]):
                y = Yb[:, k]
                misfit, extras = misfit_backend.evaluate(y, ctx, min_stations=min_stations, eps=eps)
                jstar = int(np.nanargmin(misfit))

                # DR: default to mean at best node if backend provided it
                DR_t = extras.get("mean", None)
                if DR_t is not None and np.ndim(DR_t) == 1:
                    DR_t = DR_t[jstar]
                else:
                    # fallback compute using corrections for jstar (median optional)
                    # NB: you can also let the backend return DR directly if desired.
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
                source_DR[i]     = DR_t
                source_lat[i]    = gridlat[jstar]
                source_lon[i]    = gridlon[jstar]
                source_misfit[i] = float(misfit[jstar])

                # station count used at jstar if backend reports vector N
                N = extras.get("N", None)
                source_nsta[i] = int(N if np.isscalar(N) else (N[jstar] if N is not None else np.sum(np.isfinite(y))))

                azgap, _ = compute_azimuthal_gap(source_lat[i], source_lon[i], station_coords)
                source_azgap[i] = azgap

        self.source = {
            "t": t,
            "lat": source_lat,
            "lon": source_lon,
            "DR": source_DR * 1e7,
            "misfit": source_misfit,
            "azgap": source_azgap,
            "nsta": source_nsta,
        }

        # ---- NEW: spatial connectedness (stores diagnostics on the object) ----
        conn = compute_spatial_connectedness(
            source_lat,
            source_lon,
            dr=source_DR,          # uses top DR samples
            top_frac=0.15,         # tweak if you like (10–25% works well)
            min_points=12,
            max_points=200,
        )
        self.connectedness = conn  # dict with score, n_used, stats, indices
        if verbose:
            print(f"[ASL] connectedness: score={conn['score']:.3f}  "
                  f"n_used={conn['n_used']}  mean_km={conn['mean_km']:.2f}  p90_km={conn['p90_km']:.2f}")

        # Optional: expose the scalar score in the source dict (broadcast is OK in pandas)
        self.source["connectedness"] = conn["score"]

        self.source_to_obspyevent()
        self.located = True
        if hasattr(self, "set_id"): self.id = self.set_id()
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
    

def compute_spatial_connectedness(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    dr: np.ndarray | None = None,
    top_frac: float = 0.15,
    min_points: int = 12,
    max_points: int = 200,
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

    Returns
    -------
    dict with keys:
      - 'score' : float in (0, 1]
      - 'n_used' : number of points used
      - 'mean_km', 'median_km', 'p90_km' : distance diagnostics
      - 'indices' : indices (into original arrays) of points used

      
    •	The score increases as the track tightens (mean pairwise distance shrinks).
	•	score = 1 / (1 + mean_km) → 1.0 for perfectly colocated, ~0 for very spread.
	•	Using top_frac focuses the metric on the most energetic portion of the track.
	•	self.connectedness is a dict with rich diagnostics; self.source["connectedness"]
        is a convenient scalar for tables/CSV (pandas will broadcast the scalar when building a DataFrame).
	•	If you’d prefer a different mapping (e.g., exp(-mean_km / s0)), it’s a one-liner change.
  
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    mask = np.isfinite(lat) & np.isfinite(lon)
    if dr is not None:
        dr = np.asarray(dr, float)
        mask &= np.isfinite(dr)

    idx_all = np.flatnonzero(mask)
    if idx_all.size == 0:
        return {'score': 0.0, 'n_used': 0, 'mean_km': np.nan, 'median_km': np.nan, 'p90_km': np.nan, 'indices': []}

    if dr is None:
        idx = idx_all
    else:
        k = max(min_points, int(np.ceil(top_frac * idx_all.size)))
        k = min(k, max_points, idx_all.size)
        # choose top-k by DR
        order = np.argsort(dr[idx_all])[::-1]  # descending
        idx = idx_all[order[:k]]

    if idx.size < 2:
        return {'score': 1.0, 'n_used': idx.size, 'mean_km': 0.0, 'median_km': 0.0, 'p90_km': 0.0, 'indices': idx.tolist()}

    # Vectorized haversine (km)
    R = 6371.0
    phi = np.radians(lat[idx])
    lam = np.radians(lon[idx])
    # pairwise via broadcasting
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2.0) ** 2
    d = 2.0 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))  # (K,K) symmetric, zeros on diag

    iu = np.triu_indices(idx.size, k=1)
    pair = d[iu]
    mean_km = float(np.nanmean(pair))
    median_km = float(np.nanmedian(pair))
    p90_km = float(np.nanpercentile(pair, 90))

    score = 1.0 / (1.0 + max(eps_km, mean_km))

    return {
        'score': float(score),
        'n_used': int(idx.size),
        'mean_km': mean_km,
        'median_km': median_km,
        'p90_km': p90_km,
        'indices': idx.tolist(),
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