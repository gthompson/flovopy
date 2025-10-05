# --- EnhancedStream ---------------------------------------------------
from obspy import Stream, Trace, read
import numpy as np
import pandas as pd
from collections import defaultdict
from obspy.core.util.attribdict import AttribDict

# new consolidated spectral API (writes scalars into m["spectral"])
from flovopy.core.spectral import spectral_block

# physics / magnitude helpers
from flovopy.core.physics import (
    estimate_distance,
    estimate_source_energy,
    Eseismic_Boatwright,
    Eacoustic_Boatwright,
    Eseismic2magnitude,
    estimate_local_magnitude,
)

# SNR
from flovopy.core.trace_qc import estimate_snr

def _copy_with_data(tr: Trace, data: np.ndarray) -> Trace:
    tc = tr.copy()
    tc.data = np.asarray(data, dtype=np.float64)
    return tc

class EnhancedStream(Stream):
    """ObsPy Stream with convenience metrics + station aggregation."""
    def __init__(self, stream=None, traces=None):
        # no EnhancedTrace coercion; keep it simple/robust
        if traces is not None:
            super().__init__(traces=traces)
        elif stream is not None:
            super().__init__(traces=stream.traces)
        else:
            super().__init__()
        self.station_metrics: pd.DataFrame | None = None

    # ---------------- convenience: metric container + type helpers ----------------
    @staticmethod
    def _ensure_metrics(tr: Trace) -> dict:
        if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
            tr.stats.metrics = {}
        return tr.stats.metrics

    @staticmethod
    def _is_infrasound(tr: Trace) -> bool:
        cha = getattr(tr.stats, "channel", "") or ""
        return len(cha) >= 2 and cha[1].upper() == "D"

    @staticmethod
    def _is_seismic(tr: Trace) -> bool:
        cha = getattr(tr.stats, "channel", "") or ""
        # broaden as needed ('H','B','E','S','L' etc.)
        return len(cha) >= 2 and cha[1].upper() in ("H", "B", "E", "S", "L")

    @staticmethod
    def _component(tr: Trace) -> str:
        cha = getattr(tr.stats, "channel", "") or ""
        return cha[-1].upper() if cha else ""

    @staticmethod
    def _station_key(tr: Trace) -> str:
        # NET.STA.LOC
        return ".".join([tr.stats.network, tr.stats.station, (tr.stats.location or "")])

    # ---------------- time-domain stats on series y ----------------
    def _td_stats(self, tr: Trace, y: np.ndarray) -> None:
        m = self._ensure_metrics(tr)
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

        absy = np.abs(y)
        if absy.size:
            m["abs_max"]      = float(np.nanmax(absy))
            m["abs_max_time"] = tr.stats.starttime + (int(np.nanargmax(absy)) * float(tr.stats.delta))
            m["abs_mean"]     = float(np.nanmean(absy))
            m["abs_median"]   = float(np.nanmedian(absy))
            m["abs_rms"]      = float(np.sqrt(np.nanmean(absy*absy)))
            m["abs_stdev"]    = float(np.nanstd(absy))
        else:
            m["abs_max"] = m["abs_mean"] = m["abs_median"] = m["abs_rms"] = m["abs_stdev"] = np.nan
            m["abs_max_time"] = None

        if "energy" not in m:
            m["energy"] = float(np.nansum(y*y) * float(tr.stats.delta)) if y.size else np.nan

    # ---------------- pressure metrics (PAP, bandpassed PAP) ----------------
    @staticmethod
    def _pressure_metrics(tr: Trace, band=(1.0, 20.0)) -> tuple[float, float]:
        y = np.asarray(tr.data, dtype=np.float64)
        pap = float(np.nanmax(np.abs(y))) if y.size else np.nan
        try:
            trf = tr.copy().filter("bandpass", freqmin=band[0], freqmax=band[1],
                                   corners=2, zerophase=True)
            yf = np.asarray(trf.data, dtype=np.float64)
            pap_bp = float(np.nanmax(np.abs(yf))) if yf.size else np.nan
        except Exception:
            pap_bp = np.nan
        return pap, pap_bp

    # ---------------- unified SAM/SEM (single filter pass per band pair) ----------------
    def _abs_stat(self, y: np.ndarray, stat: str) -> float:
        if y.size == 0:
            return np.nan
        if stat == "median_abs":
            return float(np.nanmedian(np.abs(y)))
        if stat == "rms":
            return float(np.sqrt(np.nanmean(y*y)))
        return float(np.nanmean(np.abs(y)))  # mean_abs default

    @staticmethod
    def _ratio(high_val: float, low_val: float, *, log2: bool) -> float:
        if not (np.isfinite(high_val) and np.isfinite(low_val)) or low_val <= 0:
            return np.nan
        r = high_val / low_val
        return float(np.log2(r)) if (log2 and r > 0) else float(r)

    def _bandpass_series(self, tr: Trace, f1: float, f2: float, *, corners=2, zerophase=True) -> np.ndarray:
        try:
            trf = tr.copy().filter(
                "bandpass",
                freqmin=float(f1), freqmax=float(f2),
                corners=int(corners), zerophase=bool(zerophase),
            )
            return np.asarray(trf.data, dtype=np.float64)
        except Exception:
            return np.asarray([], dtype=np.float64)
        
    def _compute_sam_sem(
        self,
        tr: Trace,
        *,
        low: tuple[float, float],
        high: tuple[float, float],
        sam_stat: str = "mean_abs",
        log2_ratio: bool = True,
        corners: int = 2,
        zerophase: bool = True,
    ) -> None:
        """
        Compute SAM and SEM reusing the same bandpassed series.
        Persists a single pair of bands into m["sam"] and m["sem"].
        """
        l1, l2 = low; h1, h2 = high
        dt = float(tr.stats.delta)

        y_low  = self._bandpass_series(tr, l1, l2, corners=corners, zerophase=zerophase)
        y_high = self._bandpass_series(tr, h1, h2, corners=corners, zerophase=zerophase)
        y_full = self._bandpass_series(tr, l1, h2, corners=corners, zerophase=zerophase)

        # SAM (amplitude)
        sam_low   = self._abs_stat(y_low,  sam_stat)
        sam_high  = self._abs_stat(y_high, sam_stat)
        sam_full  = self._abs_stat(y_full, sam_stat)
        sam_ratio = self._ratio(sam_high, sam_low, log2=log2_ratio)

        # SEM (energy)
        sem_low   = float(np.nansum(y_low  * y_low )) * dt if y_low.size  else np.nan
        sem_high  = float(np.nansum(y_high * y_high)) * dt if y_high.size else np.nan
        sem_full  = float(np.nansum(y_full * y_full)) * dt if y_full.size else np.nan
        sem_ratio = self._ratio(sem_high, sem_low, log2=log2_ratio)

        m = self._ensure_metrics(tr)
        band_meta = {
            "low_band":  [float(l1), float(l2)],
            "high_band": [float(h1), float(h2)],
            "full_band": [float(l1), float(h2)],
        }
        m["sam"] = {
            **band_meta,
            "stat": sam_stat,
            "log2": bool(log2_ratio),
            "values": {"low": sam_low, "high": sam_high, "full": sam_full, "ratio": sam_ratio},
        }
        m["sem"] = {
            **band_meta,
            "log2": bool(log2_ratio),
            "values": {"low": sem_low, "high": sem_high, "full": sem_full, "ratio": sem_ratio},
        }        

    # ---------------- core per-trace metrics ----------------
    def ampengfft(
        self,
        *,
        differentiate: bool = False,          # applies to seismic only
        compute_spectral: bool = False,
        compute_ssam: bool = False,
        compute_bandratios: bool = False,
        # SAM / SEM controls
        compute_sam: bool = True,
        compute_sem: bool = False,
        bands: tuple[float, float, float, float] = (0.5, 2.0, 2.0, 8.0),
        sam_stat: str = "mean_abs",
        sam_log2: bool = True,
        sem_log2: bool = True,
        # spectral params
        threshold: float = 0.707,
        window_length: int = 9,
        polyorder: int = 2,
    ) -> None:
        """Compute per-trace metrics (TD + optional spectral) and station rollup."""
        if len(self) == 0:
            self.station_metrics = pd.DataFrame()
            return

        for tr in self:
            m = self._ensure_metrics(tr)

            # detrend/taper (light)
            try:
                tr.detrend("linear").taper(0.01)
            except Exception:
                pass

            dt = float(tr.stats.delta)
            y_raw = np.asarray(tr.data, dtype=np.float64)

            # ----- branch by data type -----
            if self._is_infrasound(tr):
                # pressure series
                self._td_stats(tr, y_raw)
                pap, pap_bp = self._pressure_metrics(tr, band=(1.0, 20.0))
                m["pap"] = pap
                m["pap_bp_1_20"] = pap_bp
            else:
                # seismic: treat input as velocity unless differentiate=True flips semantics
                if differentiate:
                    disp = tr.copy()
                    vel  = tr.copy().differentiate()
                    acc  = vel.copy().differentiate()
                else:
                    vel  = tr
                    disp = tr.copy().integrate()
                    acc  = tr.copy().differentiate()

                y = np.asarray(vel.data, dtype=np.float64)
                self._td_stats(tr, y)

                m["pgd"] = float(np.nanmax(np.abs(disp.data))) if disp.stats.npts else np.nan
                m["pgv"] = float(np.nanmax(np.abs(vel.data)))  if vel.stats.npts  else np.nan
                m["pga"] = float(np.nanmax(np.abs(acc.data)))  if acc.stats.npts  else np.nan

                # dominant frequency (TD ratio)
                try:
                    num = np.abs(vel.data).astype(np.float64)
                    den = 2.0 * np.pi * np.abs(disp.data).astype(np.float64) + 1e-20
                    fdom_series = num / den
                    m["fdom"] = float(np.nanmedian(fdom_series)) if fdom_series.size else np.nan
                except Exception:
                    m["fdom"] = np.nan

            # ----- unified band features -----
            if compute_sam or compute_sem:
                l1, l2, h1, h2 = bands
                self._compute_sam_sem(
                    tr,
                    low=(l1, l2),
                    high=(h1, h2),
                    sam_stat=sam_stat,
                    log2_ratio=(sam_log2 if compute_sam else sem_log2),
                    corners=2,
                    zerophase=True,
                )
                # drop unused structure if only one requested
                if not compute_sam:
                    self._ensure_metrics(tr).pop("sam", None)
                if not compute_sem:
                    self._ensure_metrics(tr).pop("sem", None)

            # ----- spectral (optional) -----
            if compute_spectral:
                spectral_block(
                    tr, y_raw, dt,
                    threshold=threshold,
                    window_length=window_length,
                    polyorder=polyorder,
                    compute_ssam=compute_ssam,
                    compute_bandratios=compute_bandratios,
                    helper=None,  # or provide a callable adapter to compute_amplitude_spectra
                )

        # --- station-level aggregation after per-trace loop ---
        self.station_metrics = self._station_level_metrics()
        if self.station_metrics is not None and not self.station_metrics.empty:
            station_map = self.station_metrics.set_index("station_key").to_dict(orient="index")
            for tr in self:
                key = self._station_key(tr)
                if key in station_map:
                    for k, v in station_map[key].items():
                        if k != "station_key":
                            self._ensure_metrics(tr)[f"station_{k}"] = v
                            

    # ---------------- station-level aggregation ----------------

    def _vector_max_3c(self, trZ: Trace, trN: Trace, trE: Trace, *, kind="vel") -> float:
        def as_kind(tr):
            if kind == "vel":  return tr.copy()
            if kind == "disp": return tr.copy().integrate()
            if kind == "acc":  return tr.copy().differentiate()
            raise ValueError("kind must be vel|disp|acc")
        # trim to overlap
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


    def _station_level_metrics(self) -> pd.DataFrame:
        """
        Return a DF with one row per NET.STA.LOC:
          station_key, pgd_vec, pgv_vec, pga_vec, pap, pap_bp_1_20
        """
        groups: dict[str, dict[str, list]] = defaultdict(lambda: {"seis": [], "infra": []})
        for tr in self:
            key = self._station_key(tr)
            if self._is_seismic(tr):
                groups[key]["seis"].append(tr)
            elif self._is_infrasound(tr):
                groups[key]["infra"].append(tr)

        rows = []
        for key, d in groups.items():
            # seismic vector PGMs using components present (Z/N/E or Z/1/2 or Z/R/T)
            pgd_vec = pgv_vec = pga_vec = np.nan
            if d["seis"]:
                comp_groups = defaultdict(list)
                for tr in d["seis"]:
                    comp_groups[self._component(tr)].append(tr)

                def pick_one(c):  # first if multiple
                    return comp_groups[c][0] if comp_groups.get(c) else None

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
                    # fallback: L2 of per-trace PGMs
                    pgv_list = []; pgd_list=[]; pga_list=[]
                    for tr in d["seis"]:
                        m = self._ensure_metrics(tr)
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
                    m = self._ensure_metrics(tr)
                    paps.append(m.get("pap", np.nan))
                    paps_bp.append(m.get("pap_bp_1_20", np.nan))

                def med(vals):
                    arr = np.asarray([x for x in vals if np.isfinite(x)])
                    return float(np.nanmedian(arr)) if arr.size else np.nan

                pap, pap_bp = med(paps), med(paps_bp)

            # mirror station-level metrics to member traces
            for tr in d["seis"] + d["infra"]:
                m = self._ensure_metrics(tr)
                m["station_pgv_vec"]          = pgv_vec
                m["station_pgd_vec"]          = pgd_vec
                m["station_pga_vec"]          = pga_vec
                m["station_pap_med"]          = pap
                m["station_pap_bp_1_20_med"]  = pap_bp
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

    # ---------------- magnitude/energy (seismic + infra) ----------------
    def compute_station_magnitudes(
        self,
        inventory,
        source_coords,
        *,
        model: str = "body",
        Q: float = 50.0,
        c_earth: float = 2500.0,
        correction: float = 3.7,
        a: float = 1.6, 
        b: float = -0.15, 
        g: float = 0.0,
        use_boatwright: bool = True,
        rho_earth: float = 2000.0, 
        S: float = 1.0, 
        A: float = 1.0,
        rho_atmos: float = 1.2, 
        c_atmos: float = 340.0, 
        z: float = 100000.0,
        attach_coords: bool = True,
        compute_distances: bool = True,
    ) -> None:
        if attach_coords:
            self.attach_station_coordinates_from_inventory(inventory)

        for tr in self:
            try:
                R = estimate_distance(tr, source_coords)
                if compute_distances:
                    tr.stats["distance"] = R

                if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
                    tr.stats.metrics = {}

                E0 = None
                if use_boatwright:
                    if len(tr.stats.channel) >= 2 and tr.stats.channel[1].upper() == "D":
                        E0 = Eacoustic_Boatwright(tr, R, rho_atmos=rho_atmos, c_atmos=c_atmos, z=z)
                    else:
                        E0 = Eseismic_Boatwright(tr, R, rho_earth=rho_earth, c_earth=c_earth, S=S, A=A)

                        # optional Q correction using spectral peak if present
                        '''
                        spec = getattr(tr.stats, "spectral", {})
                        freqs = spec.get("freqs"); amps = spec.get("amplitudes")
                        if E0 is not None and freqs is not None and amps is not None:
                            F = np.asarray(freqs, float); A = np.asarray(amps, float)
                            if A.size and np.any(np.isfinite(A)) and np.any(A > 0):
                                f_peak = float(F[int(np.nanargmax(A))])
                                A_att = np.exp(-np.pi * f_peak * R / (Q * c_earth))
                                if np.isfinite(A_att) and A_att > 0:
                                    E0 = float(E0) / float(A_att)
                        '''
                else:
                    E0 = estimate_source_energy(tr, R, model=model, Q=Q, c_earth=c_earth)

                # Convert only if E0 is usable
                if E0 is not None and np.isfinite(E0) and E0 > 0:
                    ME = Eseismic2magnitude(E0, correction=correction)
                    tr.stats.metrics["source_energy"] = E0
                    tr.stats.metrics["energy_magnitude"] = ME
                else:
                    tr.stats.metrics["source_energy"] = np.nan
                    tr.stats.metrics["energy_magnitude"] = np.nan

                if self._is_seismic(tr):
                    R_km = R / 1000.0
                    ML = estimate_local_magnitude(tr, R_km, a=a, b=b, g=g)
                    tr.stats.metrics["local_magnitude"] = ML

            except Exception as e:
                print(f"[{tr.id}] Magnitude estimation failed: {e}")

    # ---------------- summaries & orchestration ----------------
    def magnitudes2dataframe(self) -> pd.DataFrame:
        rows = []
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
            for k, v in m.items():
                if k.startswith("station_"):
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def estimate_network_magnitude(self, key: str = "local_magnitude"):
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
        Orchestrate SNR → metrics → magnitudes. Returns (self, magnitudes DF).
        """
        # --- 1) SNR pass ---
        if verbose: print("[1] Estimating SNR…")
        for tr in list(self):
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

        # --- 2) SNR filter ---
        if snr_min is not None:
            if verbose: print(f"[2] Filtering traces with SNR < {snr_min:.1f}…")
            keep = []
            for tr in self:
                snr = tr.stats.metrics.get("snr", -np.inf)
                if snr is not None and np.isfinite(snr) and snr >= snr_min:
                    keep.append(tr)
            self._traces = keep  # mutate in place

        # --- 3) Per-trace metrics ---
        if verbose: print("[3] Computing amplitude/spectral metrics…")
        self.ampengfft(
            differentiate=differentiate,
            compute_spectral=compute_spectral,
            compute_ssam=compute_ssam,
            compute_bandratios=compute_bandratios,
            # SAM/SEM use defaults already set in ampengfft
        )

        # --- 4) Magnitudes ---
        if verbose: print("[4] Estimating magnitudes/energies…")
        self.compute_station_magnitudes(
            inventory,
            source_coords,
            model=model, Q=Q, c_earth=c_earth, correction=correction,
            a=a, b=b, g=g,
            use_boatwright=use_boatwright,
            rho_earth=rho_earth, S=S, A=A,
            rho_atmos=rho_atmos, c_atmos=c_atmos, z=z,
            attach_coords=True, compute_distances=True,
        )

        # --- 5) Summary ---
        if verbose:
            mean_ml, std_ml, n_ml = self.estimate_network_magnitude("local_magnitude")
            mean_me, std_me, n_me = self.estimate_network_magnitude("energy_magnitude")
            if mean_ml is not None:
                print(f"    → Network ML: {mean_ml:.2f} ± {std_ml:.2f} (n={n_ml})")
            if mean_me is not None:
                print(f"    → Network ME: {mean_me:.2f} ± {std_me:.2f} (n={n_me})")

        df = self.magnitudes2dataframe()
        return self, df


    def attach_station_coordinates_from_inventory(self, inventory):
        """
        Fill tr.stats.coordinates using ObsPy's Inventory.get_coordinates().

        Notes
        -----
        - Uses each trace's id (NET.STA.LOC.CHA) and starttime to query coordinates.
        - Falls back to without time if time-dependent metadata isn’t available.
        - Silently skips traces that cannot be found in the inventory.
        """
        if inventory is None:
            return

        for tr in self:
            try:
                # Prefer time-resolved coordinates if possible
                coords = inventory.get_coordinates(tr.id, tr.stats.starttime)
            except Exception:
                try:
                    # Fallback: no time argument
                    coords = inventory.get_coordinates(tr.id)
                except Exception:
                    coords = None

            if coords:
                # Only keep the common keys we actually use elsewhere
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                elev = coords.get("elevation")
                if lat is not None and lon is not None:
                    tr.stats.coordinates = AttribDict({
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "elevation": float(elev) if elev is not None else None,
                    })
    
    # ---------------- persistence ----------------

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

# --------------------------------------------------------------------------------------
# 0) Optional helpers: station-metrics–aware reduce_fn for EnhancedStream
#    These return vector PGV or median peak pressure for a station
# --------------------------------------------------------------------------------------
def reduce_station_pgv_vec(seed_id: str, st: Stream, meta: dict) -> float:
    """
    Return the station-level vector PGV for the station that contains this seed_id.

    Works best with EnhancedStream where `station_metrics` exists and includes:
      - station_key, pgv_vec (and pgd_vec, pga_vec)

    Fallback for plain Stream: take the L-infinity across trace-level PGV
    at the same station (max of |pgv| across that station's traces).

    Parameters
    ----------
    seed_id : str
        "NET.STA.LOC.CHA"
    st : Stream (ObsPy or EnhancedStream)
    meta : dict
        Unused here, kept for signature compatibility.

    Returns
    -------
    float (np.nan if unavailable)
    """
    try:
        # EnhancedStream fast-path
        es = st if hasattr(st, "station_metrics") else None
        net, sta, *_ = seed_id.split(".")
        station_prefix = f"{net}.{sta}"

        if es is not None and getattr(es, "station_metrics", None) is not None:
            sm = es.station_metrics
            if not sm.empty and "station_key" in sm.columns and "pgv_vec" in sm.columns:
                row = sm.loc[sm["station_key"].str.startswith(station_prefix)]
                if not row.empty:
                    return float(row["pgv_vec"].median(skipna=True))

        # Fallback: per-trace PGV at that station
        vals = []
        for tr in st:
            if tr.stats.network == net and tr.stats.station == sta:
                m = getattr(tr.stats, "metrics", {})
                v = m.get("pgv")
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
        return float(np.nanmax(vals)) if vals else np.nan
    except Exception:
        return np.nan


def reduce_station_pap(seed_id: str, st: Stream, meta: dict) -> float:
    """
    Return the median PAP (peak air pressure) across pressure sensors for the station.

    Works best with EnhancedStream where `station_metrics` includes:
      - station_key, pap

    Fallback for plain Stream: median of trace-level `metrics['pap']` at station.

    Returns
    -------
    float (np.nan if unavailable)
    """
    try:
        es = st if hasattr(st, "station_metrics") else None
        net, sta, *_ = seed_id.split(".")
        station_prefix = f"{net}.{sta}"

        if es is not None and getattr(es, "station_metrics", None) is not None:
            sm = es.station_metrics
            if not sm.empty and "station_key" in sm.columns and "pap" in sm.columns:
                row = sm.loc[sm["station_key"].str.startswith(station_prefix)]
                if not row.empty:
                    return float(row["pap"].median(skipna=True))

        vals = []
        for tr in st:
            if tr.stats.network == net and tr.stats.station == sta and tr.stats.channel[1].upper() == "D":
                m = getattr(tr.stats, "metrics", {})
                v = m.get("pap")
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
        return float(np.nanmedian(vals)) if vals else np.nan
    except Exception:
        return np.nan        