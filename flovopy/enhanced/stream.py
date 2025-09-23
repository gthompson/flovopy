# --- EnhancedStream ---------------------------------------------------
from obspy import Stream, Trace, read
import numpy as np
import pandas as pd
from collections import defaultdict

# import your helpers from the split modules
from flovopy.core.spectral import get_bandwidth, _ssam, _band_ratio, compute_amplitude_spectra

def _copy_with_data(tr: Trace, data: np.ndarray) -> Trace:
    tc = tr.copy()
    tc.data = np.asarray(data, dtype=np.float64)
    return tc

# --- imports you likely already have near the top ---
from flovopy.core.physics import (
    estimate_distance,
    estimate_source_energy,
    Eseismic_Boatwright,
    Eacoustic_Boatwright,
    Eseismic2magnitude,
    estimate_local_magnitude,
)
# if your SNR util is elsewhere, adjust import:
from flovopy.core.trace_qc import estimate_snr  # or wherever you put it
import numpy as np
import pandas as pd
# if you kept the time-domain band-ratio helpers here:
# from .stream import _td_rsam, _td_band_ratio   # (shown in your current file)

class EnhancedStream(Stream):
    """ObsPy Stream with convenience metrics + station aggregation."""
    def __init__(self, stream=None, traces=None):
        if traces is not None:
            # coerce all to EnhancedTrace
            coerced = [t if isinstance(t, EnhancedTrace) else EnhancedTrace(data=t.data, header=t.stats) for t in traces]
            super().__init__(traces=coerced)
        elif stream is not None:
            coerced = [t if isinstance(t, EnhancedTrace) else EnhancedTrace(data=t.data, header=t.stats) for t in stream.traces]
            super().__init__(traces=coerced)
        else:
            super().__init__()

        # a place to store station-level results
        self.station_metrics: pd.DataFrame | None = None

    # ---------------- core per-trace metrics ----------------
    def ampengfft(
        self,
        *,
        differentiate: bool = False,          # applies to seismic only
        compute_spectral: bool = False,
        compute_ssam: bool = False,
        compute_bandratios: bool = False,
        threshold: float = 0.707,
        window_length: int = 9,
        polyorder: int = 2,
        td_band_pairs=((0.5,2.0, 2.0,8.0),),  # cheap TD band-features if spectral=False
    ) -> None:
        """Compute per-trace metrics; then aggregate per-station."""

        if len(self) == 0:
            self.station_metrics = pd.DataFrame()
            return

        for tr in self:
            m = tr.ensure_metrics()

            # detrend/taper neutrally (feel free to disable if you do this elsewhere)
            try:
                tr.detrend("linear").taper(0.01)
            except Exception:
                pass

            dt = float(tr.stats.delta)
            y_raw = np.asarray(tr.data, dtype=np.float64)

            # --- branch by data type ---
            if tr.is_infrasound():
                # infrasound: treat series as pressure [Pa] (or whatever units you're using)
                # time-domain stats on raw series
                self._td_stats(tr, y_raw)

                # PAP + bandpassed PAP (1–20 Hz default inside helper)
                pap, pap_bp = self._pressure_metrics(tr, band=(1.0, 20.0))
                m["pap"] = pap
                m["pap_bp_1_20"] = pap_bp

                # optional spectral
                if compute_spectral:
                    self._spectral_block(tr, y_raw, dt, threshold, window_length, polyorder,
                                         compute_ssam, compute_bandratios)
                else:
                    # optional cheap TD band-features
                    if td_band_pairs:
                        for (l1,l2,h1,h2) in td_band_pairs[:1]:
                            br = self._td_band_ratio(tr, low=(l1,l2), high=(h1,h2), stat="mean_abs", log2=True)
                            m[f"rsam_{l1}_{l2}"] = br["RSAM_low"]
                            m[f"rsam_{h1}_{h2}"] = br["RSAM_high"]
                            m[br["ratio_key"]]   = br["ratio"]

            else:
                # seismic: choose working series
                if differentiate:
                    # input is displacement → derive vel/acc
                    disp = tr.copy()               # treat as displacement
                    vel  = tr.copy().differentiate()
                    acc  = vel.copy().differentiate()
                else:
                    # input is velocity → derive disp/acc
                    vel  = tr
                    disp = tr.copy().integrate()
                    acc  = tr.copy().differentiate()

                y = np.asarray(vel.data, dtype=np.float64)

                # time-domain stats on working series (vel)
                self._td_stats(tr, y)

                # PGM suite
                m["pgd"] = float(np.nanmax(np.abs(disp.data))) if disp.stats.npts else np.nan
                m["pgv"] = float(np.nanmax(np.abs(vel.data)))  if vel.stats.npts  else np.nan
                m["pga"] = float(np.nanmax(np.abs(acc.data)))  if acc.stats.npts  else np.nan

                # “energy” (velocity-based by default)
                m["energy"] = float(np.nansum(y * y) * dt) if y.size else np.nan

                # dominant freq via time-domain ratio
                try:
                    num = np.abs(vel.data).astype(np.float64)
                    den = 2.0 * np.pi * np.abs(disp.data).astype(np.float64) + 1e-20
                    fdom_series = num / den
                    m["fdom"] = float(np.nanmedian(fdom_series)) if fdom_series.size else np.nan
                except Exception:
                    m["fdom"] = np.nan

                # spectral (optional)
                if compute_spectral:
                    self._spectral_block(tr, y_raw, dt, threshold, window_length, polyorder,
                                         compute_ssam, compute_bandratios)
                else:
                    if td_band_pairs:
                        for (l1,l2,h1,h2) in td_band_pairs[:1]:
                            br = self._td_band_ratio(tr, low=(l1,l2), high=(h1,h2), stat="mean_abs", log2=True)
                            m[f"rsam_{l1}_{l2}"] = br["RSAM_low"]
                            m[f"rsam_{h1}_{h2}"] = br["RSAM_high"]
                            m[br["ratio_key"]]   = br["ratio"]

        # --- station-level aggregation after per-trace loop ---
        self.station_metrics = self._station_level_metrics()
        # mirror station values back to each trace (handy downstream)
        if not self.station_metrics.empty:
            station_map = self.station_metrics.set_index("station_key").to_dict(orient="index")
            for tr in self:
                key = tr.station_key()
                if key in station_map:
                    for k,v in station_map[key].items():
                        if k == "station_key":
                            continue
                        tr.stats.metrics[f"station_{k}"] = v

    # ---------------- internals (kept short & testable) ----------------
    def _td_stats(self, tr, y: np.ndarray) -> None:
        m = tr.ensure_metrics()
        if y.size:
            m["sample_min"]     = float(np.nanmin(y))
            m["sample_max"]     = float(np.nanmax(y))
            m["sample_mean"]    = float(np.nanmean(y))
            m["sample_median"]  = float(np.nanmedian(y))
            m["sample_rms"]     = float(np.sqrt(np.nanmean(y*y)))
            m["sample_stdev"]   = float(np.nanstd(y))
            try:
                from scipy.stats import skew, kurtosis
                m["skewness"]   = float(skew(y, nan_policy="omit"))
                m["kurtosis"]   = float(kurtosis(y, nan_policy="omit"))
            except Exception:
                m["skewness"] = np.nan; m["kurtosis"] = np.nan
        else:
            for k in ("sample_min","sample_max","sample_mean","sample_median",
                      "sample_rms","sample_stdev","skewness","kurtosis"):
                m[k] = np.nan

        # peakamp/peaktime on |y|
        absy = np.abs(y)
        if absy.size:
            m["peakamp"] = float(np.nanmax(absy))
            m["peaktime"] = tr.stats.starttime + (int(np.nanargmax(absy)) * float(tr.stats.delta))
        else:
            m["peakamp"] = np.nan
            m["peaktime"] = None

        # velocity-based energy if not already set elsewhere
        if "energy" not in m:
            m["energy"] = float(np.nansum(y*y) * float(tr.stats.delta)) if y.size else np.nan

    def _spectral_block(
        self,
        tr,
        y,
        dt,
        threshold,
        window_length,
        polyorder,
        compute_ssam,
        compute_bandratios,
        *,
        use_helper: bool = True,
        helper_kwargs: dict | None = None,
    ):
        """
        Populate tr.stats.spectral (freqs, amplitudes) and derive spectral metrics.

        Parameters
        ----------
        y : np.ndarray
            The time series you want to analyze (caller decides: raw/vel/disp).
        use_helper : bool
            If True, use flovopy.core.spectral.compute_amplitude_spectra() for
            detrend/taper/window/padding. If False, fast inline rFFT.
        helper_kwargs : dict
            Extra knobs passed to compute_amplitude_spectra(), e.g.:
            {
            "one_sided": True, "detrend": True, "taper": 0.01,
            "window": "hann", "pad_to_pow2": True
            }
        """
        # ---- prepare target containers ----
        m = tr.ensure_metrics() if hasattr(tr, "ensure_metrics") else (
            tr.stats.metrics if hasattr(tr.stats, "metrics") else setattr(tr.stats, "metrics", {}) or tr.stats.metrics
        )
        if not hasattr(tr.stats, "spectral") or tr.stats.spectral is None:
            tr.stats.spectral = {}

        # ---- sanitize input ----
        y = np.asarray(y, dtype=np.float64)
        if y.size < 2:
            return
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- compute spectrum ----
        if use_helper:
            # delegate to shared helper for consistent preprocessing
            try:
                from obspy import Stream
                from flovopy.core.spectral import compute_amplitude_spectra

                tmp = tr.copy()
                tmp.data = y  # analyze exactly what caller passed
                st = Stream([tmp])

                kw = {
                    "one_sided": True,
                    "detrend": True,
                    "taper": 0.01,
                    "window": "hann",
                    "pad_to_pow2": True,
                }
                if helper_kwargs:
                    kw.update(helper_kwargs)

                compute_amplitude_spectra(st, **kw)

                spec = getattr(st[0].stats, "spectral", None)
                if not spec or "freqs" not in spec or "amplitudes" not in spec:
                    return
                F = np.asarray(spec["freqs"], dtype=np.float64)
                A = np.asarray(spec["amplitudes"], dtype=np.float64)

            except Exception as e:
                # fall back to inline rFFT if helper path fails
                # (keeps things robust in odd environments)
                # print(f"[{tr.id}] helper spectra failed ({e}); falling back to rFFT.")
                from numpy.fft import rfft, rfftfreq
                N = tr.stats.npts
                if N < 2:
                    return
                A = np.abs(rfft(y))
                F = rfftfreq(N, d=dt)
        else:
            # fast inline rFFT
            from numpy.fft import rfft, rfftfreq
            N = tr.stats.npts
            if N < 2:
                return
            A = np.abs(rfft(y))
            F = rfftfreq(N, d=dt)

        # store on trace
        tr.stats.spectral["freqs"] = F
        tr.stats.spectral["amplitudes"] = A

        # ---- bandwidth/cutoffs ----
        try:
            get_bandwidth(
                F, A,
                threshold=threshold,
                window_length=window_length,
                polyorder=polyorder,
                trace=tr
            )
        except Exception as e:
            print(f"[{tr.id}] Skipping bandwidth metrics: {e}")

        # ---- SSAM (optional) ----
        if compute_ssam:
            try:
                _ssam(tr)
            except Exception as e:
                print(f"[{tr.id}] SSAM computation failed: {e}")

        # ---- spectral band ratios (optional) ----
        if compute_bandratios:
            try:
                _band_ratio(tr, freqlims=[1.0, 6.0, 11.0])
                _band_ratio(tr, freqlims=[0.5, 3.0, 18.0])
            except Exception as e:
                print(f"[{tr.id}] Band ratio computation failed: {e}")

        # ---- spectral summaries (+ back-compat to stats.spectrum) ----
        try:
            if A.size and np.any(A > 0):
                peak_idx = int(np.nanargmax(A))
                peakf = float(F[peak_idx])
                meanf = float(np.nansum(F * A) / np.nansum(A))
                medianf = float(np.nanmedian(F[A > 0]))
                peakA = float(np.nanmax(A))
            else:
                peakf = meanf = medianf = peakA = np.nan

            m["peakf"]   = peakf
            m["meanf"]   = meanf
            m["medianf"] = medianf

            s = getattr(tr.stats, "spectrum", {}) or {}
            s["peakF"]   = peakf
            s["medianF"] = medianf
            s["peakA"]   = peakA
            tr.stats.spectrum = s
        except Exception as e:
            print(f"[{tr.id}] Spectral summary failed: {e}")

    def _pressure_metrics(self, tr, band=(1.0, 20.0)):
        y = np.asarray(tr.data, dtype=np.float64)
        pap = float(np.nanmax(np.abs(y))) if y.size else np.nan
        try:
            trf = tr.copy().filter("bandpass", freqmin=band[0], freqmax=band[1],
                                   corners=2, zerophase=True)
            yf = np.asarray(trf.data, dtype=np.float64)
            pap_band = float(np.nanmax(np.abs(yf))) if yf.size else np.nan
        except Exception:
            pap_band = np.nan
        return pap, pap_band

    def _td_rsam(self, tr, f1, f2, *, stat="mean_abs", corners=2, zerophase=True):
        try:
            trf = tr.copy().filter("bandpass", freqmin=float(f1), freqmax=float(f2),
                                   corners=int(corners), zerophase=bool(zerophase))
            y = trf.data.astype(np.float64)
            if not y.size:
                return np.nan
            if stat == "median_abs":
                return float(np.nanmedian(np.abs(y)))
            if stat == "rms":
                return float(np.sqrt(np.nanmean(y*y)))
            return float(np.nanmean(np.abs(y)))
        except Exception:
            return np.nan

    def _td_band_ratio(self, tr, low=(0.5,2.0), high=(2.0,8.0), *, stat="mean_abs", log2=True):
        a1,b1 = low; a2,b2 = high
        r_low  = self._td_rsam(tr, a1, b1, stat=stat)
        r_high = self._td_rsam(tr, a2, b2, stat=stat)
        if not np.isfinite(r_low) or not np.isfinite(r_high) or r_low <= 0:
            ratio = np.nan
        else:
            ratio = r_high / r_low
            if log2 and ratio > 0:
                ratio = float(np.log2(ratio))
        return {
            "RSAM_low":  r_low,
            "RSAM_high": r_high,
            "ratio":     ratio,
            "ratio_key": f"bandratio_{a1}_{b1}__{a2}_{b2}" + ("_log2" if log2 else ""),
        }

    # ---------------- station-level aggregation ----------------
    def _station_level_metrics(self) -> pd.DataFrame:
        """
        Return a DF with one row per NET.STA.LOC:
          station_key, pgd_vec, pgv_vec, pga_vec, pap, pap_bp_1_20
        """
        groups: dict[str, dict[str, list]] = defaultdict(lambda: {"seis": [], "infra": []})
        for tr in self:
            key = tr.station_key()
            if tr.is_seismic():
                groups[key]["seis"].append(tr)
            elif tr.is_infrasound():
                groups[key]["infra"].append(tr)

        rows = []
        for key, d in groups.items():
            # seismic vector PGMs using components present (Z/N/E or Z/1/2 or Z/R/T)
            pgd_vec = pgv_vec = pga_vec = np.nan
            if d["seis"]:
                # group by component; we’ll attempt plausible triplets
                comp_groups = defaultdict(list)
                for tr in d["seis"]:
                    comp_groups[tr.component].append(tr)
                def pick_one(c):  # pick the first if multiple
                    return comp_groups[c][0] if comp_groups.get(c) else None

                # try in order of preference
                triplets = [("Z","N","E"), ("Z","1","2"), ("Z","R","T")]
                got = False
                for (cz, cx, cy) in triplets:
                    tz, tx, ty = (pick_one(cz), pick_one(cx), pick_one(cy))
                    if tz and tx and ty:
                        pgd_vec = self._vector_max_3c(tz, tx, ty, kind="disp")
                        pgv_vec = self._vector_max_3c(tz, tx, ty, kind="vel")
                        pga_vec = self._vector_max_3c(tz, tx, ty, kind="acc")
                        got = True
                        break
                if not got:
                    # fallback: L2 of per-trace PGMs that ampengfft already computed
                    pgv_list = []; pgd_list=[]; pga_list=[]
                    for tr in d["seis"]:
                        m = tr.ensure_metrics()
                        pgv_list.append(m.get("pgv", np.nan))
                        pgd_list.append(m.get("pgd", np.nan))
                        pga_list.append(m.get("pga", np.nan))
                    def l2(vals):
                        arr = np.asarray([x for x in vals if np.isfinite(x)])
                        return float(np.sqrt(np.sum(arr**2))) if arr.size else np.nan
                    pgv_vec, pgd_vec, pga_vec = l2(pgv_list), l2(pgd_list), l2(pga_list)

            # infrasound median PAP across sensors
            pap = pap_bp = np.nan
            if d["infra"]:
                paps = []; paps_bp=[]
                for tr in d["infra"]:
                    m = tr.ensure_metrics()
                    paps.append(m.get("pap", np.nan))
                    paps_bp.append(m.get("pap_bp_1_20", np.nan))
                def med(vals):
                    arr = np.asarray([x for x in vals if np.isfinite(x)])
                    return float(np.nanmedian(arr)) if arr.size else np.nan
                pap, pap_bp = med(paps), med(paps_bp)


            # Mirror station-level metrics to member traces for convenience
            for tr in d["seis"] + d["infra"]:
                m = tr.ensure_metrics()  # or: m = getattr(tr.stats, "metrics", {}) ; tr.stats.metrics = m
                m["station_pgv_vec"]   = pgv_vec
                m["station_pgd_vec"]   = pgd_vec
                m["station_pga_vec"]   = pga_vec
                m["station_pap_med"]   = pap
                m["station_pap_bp_1_20_med"] = pap_bp
                m["station_num_seis_traces"]  = len(d["seis"])
                m["station_num_infra_traces"] = len(d["infra"])

            rows.append({
                "station_key": key,
                "pgd_vec": pgd_vec, "pgv_vec": pgv_vec, "pga_vec": pga_vec,
                "pap": pap, "pap_bp_1_20": pap_bp
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["station_key","pgd_vec","pgv_vec","pga_vec","pap","pap_bp_1_20"]
        )

    # --- small 3C helper (reused) ---
    def _vector_max_3c(self, trZ, trN, trE, *, kind="vel") -> float:
        def as_kind(tr):
            if kind == "vel":  return tr.copy()
            if kind == "disp": return tr.copy().integrate()
            if kind == "acc":  return tr.copy().differentiate()
            raise ValueError("kind must be vel|disp|acc")
        # trim to common overlap
        t0 = max(tr.stats.starttime for tr in (trZ, trN, trE))
        t1 = min(tr.stats.endtime   for tr in (trZ, trN, trE))
        if t1 <= t0:
            return np.nan
        trs = [as_kind(tr).trim(t0, t1, pad=False) for tr in (trZ, trN, trE)]
        if any(t.stats.npts <= 1 for t in trs):
            return np.nan
        z, n, e = (np.asarray(t.data, dtype=np.float64) for t in trs)
        vec = np.sqrt(z*z + n*n + e*e)
        return float(np.nanmax(np.abs(vec))) if vec.size else np.nan
    

    def compute_station_magnitudes(
        self,
        inventory,
        source_coords,
        *,
        model: str = "body",         # 'body' or 'surface' for geometric spreading in legacy model
        Q: float = 50.0,
        c_earth: float = 2500.0,
        correction: float = 3.7,     # Hanks & Kanamori correction (Joules)
        a: float = 1.6, b: float = -0.15, g: float = 0.0,  # ML coefficients
        use_boatwright: bool = True,
        rho_earth: float = 2000.0, S: float = 1.0, A: float = 1.0,
        rho_atmos: float = 1.2, c_atmos: float = 340.0, z: float = 100000.0,
        attach_coords: bool = True,
        compute_distances: bool = True,
    ) -> None:
        """
        Per-trace magnitude/energy calculations with seismic/infra branching.

        Side-effects
        ------------
        - tr.stats['distance'] (meters) if compute_distances=True
        - tr.stats.metrics['source_energy']
        - tr.stats.metrics['energy_magnitude'] (Me from energy)
        - tr.stats.metrics['local_magnitude'] (ML for seismic channels only)
        """
        if attach_coords:
            self.attach_station_coordinates_from_inventory(inventory)

        for tr in self:
            try:
                # distance
                R = estimate_distance(tr, source_coords)
                if compute_distances:
                    tr.stats['distance'] = R

                # ensure metrics
                if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
                    tr.stats.metrics = {}

                # choose energy model
                if use_boatwright:
                    if len(tr.stats.channel) >= 2 and tr.stats.channel[1].upper() == "D":
                        # infrasound Boatwright
                        E0 = Eacoustic_Boatwright(tr, R, rho_atmos=rho_atmos, c_atmos=c_atmos, z=z)
                    else:
                        # seismic Boatwright
                        E0 = Eseismic_Boatwright(tr, R, rho_earth=rho_earth, c_earth=c_earth, S=S, A=A)

                        # optional Q correction using spectral peak if present
                        spec = getattr(tr.stats, "spectral", {})
                        freqs = spec.get("freqs"); amps = spec.get("amplitudes")
                        if freqs is not None and amps is not None and np.any(np.asarray(amps) > 0):
                            f_peak = float(freqs[int(np.nanargmax(amps))])
                            A_att = np.exp(-np.pi * f_peak * R / (Q * c_earth))
                            if A_att > 0:
                                E0 /= A_att
                else:
                    # legacy geometric+attenuation model (uses tr.stats.metrics['energy'])
                    E0 = estimate_source_energy(tr, R, model=model, Q=Q, c_earth=c_earth)

                # convert to magnitude from energy
                ME = Eseismic2magnitude(E0, correction=correction) if E0 is not None else None
                tr.stats.metrics["source_energy"] = E0
                tr.stats.metrics["energy_magnitude"] = ME

                # ML only for seismic
                if len(tr.stats.channel) >= 2 and tr.stats.channel[1].upper() in ("H", "B", "E", "S", "L"):
                    R_km = R / 1000.0
                    ML = estimate_local_magnitude(tr, R_km, a=a, b=b, g=g)
                    tr.stats.metrics["local_magnitude"] = ML

            except Exception as e:
                print(f"[{tr.id}] Magnitude estimation failed: {e}")

    def magnitudes2dataframe(self) -> pd.DataFrame:
        """
        Summarize per-trace and per-station magnitudes/energies.

        Returns
        -------
        DataFrame with at least:
        id, starttime, distance_m, ML, ME, source_energy, peakamp, energy
        + station-level columns if available (station_pgv_vec, station_pap, ...)
        """
        rows = []
        # station mirrors (if ampengfft was run) live in tr.stats.metrics['station_*']
        for tr in self:
            m = getattr(tr.stats, "metrics", {}) or {}
            row = {
                "id": tr.id,
                "starttime": tr.stats.starttime,
                "distance_m": tr.stats.get("distance", np.nan),
                "ML": m.get("local_magnitude", np.nan),
                "ME": m.get("energy_magnitude", np.nan),
                "source_energy": m.get("source_energy", np.nan),
                "peakamp": m.get("peakamp", np.nan),
                "energy": m.get("energy", np.nan),
            }
            # include station-level fields if present
            for k, v in m.items():
                if k.startswith("station_"):
                    row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def estimate_network_magnitude(self, key: str = "local_magnitude"):
        """
        Mean ± std of a given magnitude metric across traces.

        key : 'local_magnitude' or 'energy_magnitude'
        Returns (mean, std, n)
        """
        vals = []
        for tr in self:
            m = getattr(tr.stats, "metrics", {}) or {}
            val = m.get(key)
            if val is not None and np.isfinite(val):
                vals.append(float(val))
        if not vals:
            return None, None, 0
        arr = np.asarray(vals, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr)), int(arr.size)

    def ampengfftmag(
        self,
        inventory,
        source_coords: dict,
        *,
        # ampengfft params
        differentiate: bool = True,
        compute_spectral: bool = True,
        compute_ssam: bool = False,
        compute_bandratios: bool = False,
        threshold: float = 0.707,
        window_length: int = 9,
        polyorder: int = 2,
        # SNR/filtering
        snr_method: str = "std",
        snr_window_length: float = 1.0,
        snr_split_time=None,
        snr_min=None,
        # magnitude params
        model: str = "body",
        Q: float = 50.0,
        c_earth: float = 2500.0,
        correction: float = 3.7,
        a: float = 1.6, b: float = -0.15, g: float = 0.0,
        use_boatwright: bool = True,
        rho_earth: float = 2000.0, S: float = 1.0, A: float = 1.0,
        rho_atmos: float = 1.2, c_atmos: float = 340.0, z: float = 100000.0,
        verbose: bool = True,
    ):
        """
        Orchestrate per-trace metrics, SNR filtering, and magnitude calculations.

        Returns
        -------
        (EnhancedStream, pd.DataFrame)
            The stream (mutated in place) and a dataframe from magnitudes2dataframe()
        """
        # --- 1) SNR pass (before metrics) ---
        if verbose:
            print("[1] Estimating SNR…")
        for tr in list(self):  # list() in case we filter later
            if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
                tr.stats.metrics = {}
            if "snr" not in tr.stats.metrics:
                try:
                    snr, signal_val, noise_val = estimate_snr(
                        tr,
                        method=snr_method,
                        window_length=snr_window_length,
                        split_time=snr_split_time,
                        verbose=False,
                    )
                    tr.stats.metrics["snr"] = snr
                    tr.stats.metrics["signal"] = signal_val
                    tr.stats.metrics["noise"] = noise_val
                except Exception as e:
                    if verbose:
                        print(f"[{tr.id}] SNR estimation failed: {e}")
                    tr.stats.metrics.setdefault("snr", np.nan)

        # --- 2) Filter by SNR if requested ---
        if snr_min is not None:
            if verbose:
                print(f"[2] Filtering traces with SNR < {snr_min:.1f}…")
            keep = []
            for tr in self:
                snr = tr.stats.metrics.get("snr", -np.inf)
                if snr is not None and np.isfinite(snr) and snr >= snr_min:
                    keep.append(tr)
            self._traces = keep  # mutate in place (ObsPy style)

        # --- 3) Per-trace metrics (time-domain + optional spectral) ---
        if verbose:
            print("[3] Computing amplitude/spectral metrics…")
        self.ampengfft(
            differentiate=differentiate,
            compute_spectral=compute_spectral,
            compute_ssam=compute_ssam,
            compute_bandratios=compute_bandratios,
            threshold=threshold,
            window_length=window_length,
            polyorder=polyorder,
        )

        # --- 4) Magnitudes from physics models ---
        if verbose:
            print("[4] Estimating magnitudes/energies…")
        self.compute_station_magnitudes(
            inventory,
            source_coords,
            model=model,
            Q=Q,
            c_earth=c_earth,
            correction=correction,
            a=a, b=b, g=g,
            use_boatwright=use_boatwright,
            rho_earth=rho_earth, S=S, A=A,
            rho_atmos=rho_atmos, c_atmos=c_atmos, z=z,
            attach_coords=True,
            compute_distances=True,
        )

        # --- 5) Summarize ---
        if verbose:
            mean_ml, std_ml, n_ml = self.estimate_network_magnitude("local_magnitude")
            mean_me, std_me, n_me = self.estimate_network_magnitude("energy_magnitude")
            if mean_ml is not None:
                print(f"    → Network ML: {mean_ml:.2f} ± {std_ml:.2f} (n={n_ml})")
            if mean_me is not None:
                print(f"    → Network ME: {mean_me:.2f} ± {std_me:.2f} (n={n_me})")

        df = self.magnitudes2dataframe()
        return self, df
    
    def compute_station_pgms(self, *, pressure_band=(1.0, 20.0)) -> pd.DataFrame:
        """
        Compute station-level PGMs:
        - seismic vector maxima: station_pgv_vec, station_pgd_vec, station_pga_vec
        - infrasound (per-station): station_pap_max, station_pap_band_max, station_pap_median

        Mirrors station values back onto each member trace as tr.stats.metrics['station_*'].
        Returns a station-level DataFrame.
        """
        groups = _group_by_station_band(self)  # {(net,sta,loc,band2): {'Z':[...],'N':[...],'E':[...],'P':[...]}}

        rows = []
        for (net, sta, loc, band2), compmap in groups.items():
            # --- seismic 3C vector PGMs ---
            pgv_vec = pgd_vec = pga_vec = np.nan
            for set_ in _COMPONENT_SETS:
                if all(c in compmap and len(compmap[c]) > 0 for c in set_):
                    # pick one trace per component (first); trim to common overlap
                    trZ = compmap[set_[0]][0]
                    trN = compmap[set_[1]][0]
                    trE = compmap[set_[2]][0]
                    pgv_vec = _vector_max_3c(trZ, trN, trE, kind="vel")
                    pgd_vec = _vector_max_3c(trZ, trN, trE, kind="disp")
                    pga_vec = _vector_max_3c(trZ, trN, trE, kind="acc")
                    break

            # --- infrasound per station (any “*D?” channel) ---
            pap_vals = []
            pap_band_vals = []
            for comp, traces in compmap.items():
                for tr in traces:
                    if len(tr.stats.channel) >= 2 and tr.stats.channel[1].upper() == "D":
                        pap, pap_band = _pressure_metrics(tr, band=pressure_band)
                        pap_vals.append(pap)
                        pap_band_vals.append(pap_band)

            station_pap_max = np.nanmax(pap_vals) if pap_vals else np.nan
            station_pap_band_max = np.nanmax(pap_band_vals) if pap_band_vals else np.nan
            station_pap_median = float(np.nanmedian(pap_vals)) if pap_vals else np.nan

            # mirror station vals back onto member traces for convenience
            for comp, traces in compmap.items():
                for tr in traces:
                    m = getattr(tr.stats, "metrics", None) or {}
                    m["station_pgv_vec"] = pgv_vec
                    m["station_pgd_vec"] = pgd_vec
                    m["station_pga_vec"] = pga_vec
                    m["station_pap_max"] = station_pap_max
                    m["station_pap_band_max"] = station_pap_band_max
                    m["station_pap_median"] = station_pap_median
                    tr.stats.metrics = m

            rows.append({
                "network": net, "station": sta, "location": loc, "band": band2,
                "station_pgv_vec": pgv_vec, "station_pgd_vec": pgd_vec, "station_pga_vec": pga_vec,
                "station_pap_max": station_pap_max, "station_pap_band_max": station_pap_band_max,
                "station_pap_median": station_pap_median,
                "n_infra_traces": int(np.sum([len(trs) for c, trs in compmap.items()
                                            if len(trs) and len(trs[0].stats.channel) >= 2
                                            and trs[0].stats.channel[1].upper() == "D"])),
                "n_traces": int(np.sum([len(trs) for trs in compmap.values()])),
            })

        return pd.DataFrame(rows)
    
    @classmethod
    def read(cls, basepath: str, match_on: str = "id"):
        """
        Load an EnhancedStream from:
        - <base>.mseed  (waveforms)
        - <base>.csv    (per-trace metrics)
        - <base>_station.csv (station-level summary; optional)

        Parameters
        ----------
        basepath : str
            Path without extension (or with '.mseed' which will be stripped).
        match_on : {'id','starttime'}
            Column to match rows in the CSV to traces.

        Returns
        -------
        EnhancedStream
        """
        import os
        import pandas as pd
        from obspy import read
        from obspy.core.util.attribdict import AttribDict
        from obspy.core.utcdatetime import UTCDateTime

        # normalize base
        base = basepath[:-6] if basepath.endswith(".mseed") else basepath
        mseed = base + ".mseed"
        csv_tr = base + ".csv"
        csv_sta = base + "_station.csv"

        # --- 1) load waveforms ---
        st = read(mseed, format="MSEED")

        # --- 2) load per-trace metrics CSV ---
        if not os.path.exists(csv_tr):
            # still return a valid EnhancedStream if only mseed is present
            es = cls(stream=st)
            # try to populate station_metrics if available
            if os.path.exists(csv_sta):
                try:
                    es.station_metrics = pd.read_csv(csv_sta)
                except Exception as e:
                    print(f"[WARN] failed reading station CSV: {e}")
            return es

        df = pd.read_csv(csv_tr)

        # help: make starttime comparable when matching on starttime
        def _to_utcdt(x):
            try:
                return UTCDateTime(str(x))
            except Exception:
                return None

        if match_on not in df.columns:
            # fall back to id
            match_on = "id"

        # precompute a lookup dict for faster row matching
        if match_on == "starttime":
            # build dictionary keyed by UTCDateTime
            df["_starttime_utc"] = df["starttime"].apply(_to_utcdt)
            row_map = {r["_starttime_utc"]: i for _, r in df.iterrows()}
        else:
            row_map = {r["id"]: i for _, r in df.iterrows()}

        # columns that belong to the 'spectrum' convenience dict
        SPECTRUM_KEYS = {"medianF", "peakF", "peakA", "bw_min", "bw_max"}
        COORD_KEYS = {"latitude", "longitude", "elevation"}

        for tr in st:
            # pick row by id or starttime
            if match_on == "starttime":
                key = tr.stats.starttime
            else:
                key = tr.id

            idx = row_map.get(key, None)
            if idx is None:
                # try relaxed starttime matching by string if needed
                if match_on == "starttime":
                    try:
                        idx = df.index[df["starttime"].astype(str) == str(tr.stats.starttime)].tolist()
                        idx = idx[0] if idx else None
                    except Exception:
                        idx = None

            if idx is None:
                continue

            row = df.iloc[idx]

            # 2a) rebuild metrics dict
            tr.stats.metrics = {}

            for col in df.columns:
                if col in ("id", "starttime", "Fs", "calib", "units", "quality"):
                    continue
                if col in SPECTRUM_KEYS or col in COORD_KEYS:
                    # handled below
                    continue

                val = row[col]

                # Optional: split compound keys 'foo_bar' -> metrics['foo']['bar']
                # Keeps your previous convention (first underscore split).
                if isinstance(col, str) and "_" in col:
                    main_key, sub_key = col.split("_", 1)
                    if main_key not in tr.stats.metrics:
                        tr.stats.metrics[main_key] = {}
                    tr.stats.metrics[main_key][sub_key] = val
                else:
                    tr.stats.metrics[col] = val

            # 2b) restore convenience .spectrum dict if present
            if any(k in df.columns for k in SPECTRUM_KEYS):
                spec = {}
                for k in SPECTRUM_KEYS:
                    if k in df.columns and pd.notna(row.get(k)):
                        spec[k] = row.get(k)
                if spec:
                    tr.stats.spectrum = spec

            # 2c) restore coordinates if present
            if COORD_KEYS.issubset(df.columns):
                lat = row.get("latitude"); lon = row.get("longitude"); elev = row.get("elevation")
                if pd.notna(lat) and pd.notna(lon):
                    tr.stats.coordinates = AttribDict({
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "elevation": float(elev) if pd.notna(elev) else None
                    })

            # 2d) restore basic stat fields when available
            if "calib" in df.columns and pd.notna(row.get("calib")):
                tr.stats.calib = row.get("calib")
            if "units" in df.columns and pd.notna(row.get("units")):
                tr.stats.units = row.get("units")
            if "quality" in df.columns and pd.notna(row.get("quality")):
                tr.stats.quality_factor = row.get("quality")

        # --- 3) construct EnhancedStream and attach station metrics if they exist ---
        es = cls(stream=st)

        if os.path.exists(csv_sta):
            try:
                es.station_metrics = pd.read_csv(csv_sta)
            except Exception as e:
                print(f"[WARN] failed reading station CSV: {e}")

        return es

    def save(self, basepath: str, save_pickle: bool = False) -> None:
        """
        Save EnhancedStream to disk: waveform, per-trace metrics, and station-level metrics.

        Parameters
        ----------
        basepath : str
            Base path (no extension) for outputs.
            Will write basepath.mseed, basepath.csv, basepath_station.csv, [basepath.pkl].
        save_pickle : bool, default False
            If True, also save a Python pickle of the stream.
        """
        import os
        import pandas as pd

        # strip .mseed if user passed it
        if basepath.endswith(".mseed"):
            basepath = basepath[:-6]

        outdir = os.path.dirname(basepath)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        # --- 1. waveform ---
        self.write(basepath + ".mseed", format="MSEED")

        # --- 2. per-trace metrics ---
        trace_rows = []
        for tr in self:
            s = tr.stats
            row = {
                "id": tr.id,
                "starttime": s.starttime,
                "Fs": s.sampling_rate,
                "calib": getattr(s, "calib", None),
                "units": getattr(s, "units", None),
                "quality": getattr(s, "quality_factor", None),
            }

            if hasattr(s, "spectrum"):
                for item in ["medianF", "peakF", "peakA", "bw_min", "bw_max"]:
                    row[item] = s.spectrum.get(item, None)

            if hasattr(s, "metrics"):
                for k, v in s.metrics.items():
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            row[f"{k}_{subk}"] = subv
                    else:
                        row[k] = v

            if hasattr(s, "coordinates"):
                row["latitude"] = s.coordinates.latitude
                row["longitude"] = s.coordinates.longitude
                row["elevation"] = s.coordinates.elevation

            trace_rows.append(row)

        df_traces = pd.DataFrame(trace_rows)
        df_traces.to_csv(basepath + ".csv", index=False)

        # --- 3. station-level metrics ---
        try:
            sdf = getattr(self, "station_metrics", None)
            if sdf is None or sdf.empty:
                sdf = self._station_level_metrics()
            if not sdf.empty:
                sdf.to_csv(basepath + "_station.csv", index=False)
        except Exception as e:
            print(f"[WARN] station-level CSV not written: {e}")

        # --- 4. pickle (optional) ---
        if save_pickle:
            import pickle
            with open(basepath + ".pkl", "wb") as f:
                pickle.dump(self, f)

        print(f"[✓] Saved {len(self)} traces to {basepath}.mseed/.csv and station summary.")