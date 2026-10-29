# flovopy/core/spectral.py
from __future__ import annotations

import os
import numpy as np

from obspy.core.stream import Stream
from obspy.core.trace import Trace
from scipy.signal import savgol_filter, get_window
from numpy.fft import rfft, rfftfreq


# ---------------------------------------------------------------------------
# Core spectrum computation
# ---------------------------------------------------------------------------

def compute_amplitude_spectra(
    stream: Stream,
    *,
    one_sided: bool = True,
    detrend: bool = True,
    taper: float = 0.01,
    window: str | None = "hann",
    pad_to_pow2: bool = True,
) -> Stream:
    """
    Compute amplitude spectra for all traces and store in tr.stats.spectral.

    Parameters
    ----------
    stream : obspy.Stream
    one_sided : bool
        If True, use rFFT (non-negative freqs). If False, use full FFT.
    detrend : bool
        Apply linear detrend (ObsPy) before FFT.
    taper : float
        Fractional cosine taper (0..1). 0 disables.
    window : str | None
        scipy.signal window name (e.g. 'hann', 'hamming'). None disables.
    pad_to_pow2 : bool
        Zero-pad to next power-of-two for speed/freq resolution.

    Returns
    -------
    Stream (same object), with:
        tr.stats.spectral['freqs'], ['amplitudes'], ['nfft'], ['window']
    """
    for tr in stream:
        # defensive copy of data to float64
        y = np.asarray(tr.data, dtype=np.float64)
        if y.size < 2:
            continue

        work = tr.copy()
        if detrend:
            work.detrend("linear")
        if taper and taper > 0:
            work.taper(max(0.0, min(1.0, float(taper))))
        y = np.asarray(work.data, dtype=np.float64)

        if window:
            try:
                w = get_window(window, y.size, fftbins=True).astype(np.float64)
                y = y * w
                wname = str(window)
            except Exception:
                wname = "none"
        else:
            wname = "none"

        n = y.size
        if pad_to_pow2:
            nfft = int(2 ** np.ceil(np.log2(n)))
        else:
            nfft = n

        dt = float(tr.stats.delta)
        if one_sided:
            Y = np.abs(np.fft.rfft(y, n=nfft))
            F = np.fft.rfftfreq(nfft, d=dt)
        else:
            Y = np.abs(np.fft.fft(y, n=nfft))
            F = np.fft.fftfreq(nfft, d=dt)

        # store
        spec = getattr(tr.stats, "spectral", {}) or {}
        spec["freqs"] = F
        spec["amplitudes"] = Y
        spec["nfft"] = int(nfft)
        spec["window"] = wname
        tr.stats.spectral = spec

    return stream


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_amplitude_spectra(
    stream: Stream,
    *,
    max_freq: float = 50.0,
    log_y: bool = False,
    normalize: bool = False,
    outfile: str | None = None,
):
    """
    Plot amplitude spectra for all traces that already have .stats.spectral.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    for tr in stream:
        try:
            f = tr.stats.spectral["freqs"]
            a = tr.stats.spectral["amplitudes"]
            if normalize and np.nanmax(a) > 0:
                a = a / np.nanmax(a)
            y = np.log10(a + 1e-20) if log_y else a
            plt.plot(f, y, label=tr.id)
        except Exception:
            continue

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Amplitude)" if log_y else "Amplitude")
    plt.title("Amplitude Spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.xlim(0, max_freq)

    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Amplitude ratios (signal / noise)
# ---------------------------------------------------------------------------

def _trim_to_overlap(sig: Trace, noi: Trace) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return overlapping segments for signal/noise traces as float64 arrays and dt.
    Pads the shorter one to match the longer within the overlap window.
    """
    t0 = max(sig.stats.starttime, noi.stats.starttime)
    t1 = min(sig.stats.endtime, noi.stats.endtime)
    if t1 <= t0:
        return np.array([]), np.array([]), float(sig.stats.delta)

    sigc = sig.copy().trim(t0, t1, pad=False)
    noic = noi.copy().trim(t0, t1, pad=False)
    y1 = np.asarray(sigc.data, dtype=np.float64)
    y2 = np.asarray(noic.data, dtype=np.float64)
    n = max(y1.size, y2.size)
    if y1.size < n:
        y1 = np.pad(y1, (0, n - y1.size))
    if y2.size < n:
        y2 = np.pad(y2, (0, n - y2.size))
    return y1, y2, float(sigc.stats.delta)


def _check_spectral_qc(freqs: np.ndarray, ratio: np.ndarray, *,
                       threshold: float, min_fraction_pass: float) -> tuple[bool, float]:
    """
    Simple QC: fraction of bins with ratio >= threshold over finite bins.
    Returns (passed, failed_fraction).
    """
    finite = np.isfinite(ratio)
    if not finite.any():
        return False, 1.0
    frac_pass = np.mean((ratio[finite] >= threshold).astype(float))
    return (frac_pass >= min_fraction_pass), (1.0 - frac_pass)


def compute_amplitude_ratios(
    signal_stream: Stream | Trace,
    noise_stream: Stream | Trace,
    *,
    smooth_window: int | None = None,
    verbose: bool = False,
    average: str = "geometric",  # 'geometric' | 'median' | 'mean'
    qc_threshold: float | None = None,
    qc_fraction: float = 0.5,
):
    """
    Compute spectral amplitude ratios between signal and noise traces.
    Returns:
      avg_freqs, avg_spectral_ratio, individual_ratios, freqs_list, trace_ids, qc_results
    """
    if isinstance(signal_stream, Trace):
        signal_stream = Stream([signal_stream])
    if isinstance(noise_stream, Trace):
        noise_stream = Stream([noise_stream])

    # map noise by id for 1:1 pairing
    noise_dict = {tr.id: tr for tr in noise_stream}

    individual_ratios: list[np.ndarray] = []
    freqs_list: list[np.ndarray] = []
    trace_ids: list[str] = []
    qc_results: dict[str, dict] = {}

    for sig_tr in signal_stream:
        tid = sig_tr.id
        noi_tr = noise_dict.get(tid)
        if noi_tr is None:
            if verbose:
                print(f"[ratios] skip {tid}: no matching noise trace")
            continue

        y_sig, y_noi, dt = _trim_to_overlap(sig_tr, noi_tr)
        if y_sig.size < 2:
            if verbose:
                print(f"[ratios] skip {tid}: insufficient overlap")
            continue

        n = max(y_sig.size, y_noi.size)
        nfft = int(2 ** np.ceil(np.log2(n)))  # power-of-two pad

        # FFTs (one-sided)
        S = np.abs(np.fft.rfft(y_sig, n=nfft))
        N = np.abs(np.fft.rfft(y_noi, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=dt)

        # avoid /0
        N[N == 0] = 1e-20
        ratio = S / N

        if smooth_window and smooth_window > 1:
            kernel = np.ones(int(smooth_window), dtype=float)
            kernel /= kernel.sum()
            ratio = np.convolve(ratio, kernel, mode="same")

        individual_ratios.append(ratio)
        freqs_list.append(freqs)
        trace_ids.append(tid)

        if qc_threshold is not None:
            passed, frac_failed = _check_spectral_qc(freqs, ratio,
                                                     threshold=qc_threshold,
                                                     min_fraction_pass=qc_fraction)
            qc_results[tid] = {"passed": passed, "failed_fraction": frac_failed}

    if not individual_ratios:
        return None, None, [], [], [], {}

    # Average across traces
    R = np.vstack(individual_ratios)
    if average == "geometric":
        avg = np.exp(np.nanmean(np.log(R + 1e-20), axis=0))
    elif average == "median":
        avg = np.nanmedian(R, axis=0)
    else:
        avg = np.nanmean(R, axis=0)

    avg_freqs = freqs_list[0]
    if verbose:
        print(f"[ratios] computed for {len(individual_ratios)} traces")

    return avg_freqs, avg, individual_ratios, freqs_list, trace_ids, qc_results


def plot_amplitude_ratios(
    avg_freqs: np.ndarray,
    avg_spectral_ratio: np.ndarray,
    *,
    individual_ratios: list[np.ndarray] | None = None,
    freqs_list: list[np.ndarray] | None = None,
    trace_ids: list[str] | None = None,
    log_scale: bool = False,
    outfile: str | None = None,
    max_freq: float = 50.0,
    threshold: float | None = None,
):
    """
    Plot spectral amplitude ratios (signal / noise).
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    if individual_ratios and freqs_list:
        for i, ratio in enumerate(individual_ratios):
            f = freqs_list[i]
            y = np.log10(ratio + 1e-20) if log_scale else ratio
            label = trace_ids[i] if trace_ids else f"Trace {i}"
            plt.plot(f, y, alpha=0.5, linewidth=1, label=label)

    if avg_spectral_ratio is not None:
        y_avg = np.log10(avg_spectral_ratio + 1e-20) if log_scale else avg_spectral_ratio
        plt.plot(avg_freqs, y_avg, color="black", linewidth=2.5, label="Average")

    if threshold is not None:
        y_thr = np.log10(threshold + 1e-20) if log_scale else threshold
        plt.axhline(y_thr, color="red", linestyle="--", label="SNR threshold")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Ratio)" if log_scale else "Amplitude Ratio")
    plt.title("Amplitude Spectrum Ratio (Signal / Noise)")
    plt.xlim(0, max_freq)
    plt.ylim(bottom=None if log_scale else 0.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    if outfile:
        print(f"Saving amplitude_ratio plot to {outfile} from {os.getcwd()}")
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()




def _ensure_metrics(tr: Trace):
    if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
        tr.stats.metrics = {}
    return tr.stats.metrics

def spectral_block(
    tr: Trace,
    y: np.ndarray,
    dt: float,
    *,
    threshold: float = 0.707,
    window_length: int = 9,
    polyorder: int = 2,
    compute_ssam: bool = False,
    compute_bandratios: bool = False,
    helper: callable | None = None,     # optional: fn(Stream, **opts) that fills .stats.spectral
    helper_opts: dict | None = None,
    bands: tuple[float, float, float, float] = (0.8, 4.0, 4.0, 20.0),
) -> None:
    """
    Compute one-sided amplitude spectrum for `y` and store:
      - arrays:  tr.stats.spectral["freqs"], ["amplitudes"]
      - summaries: tr.stats.metrics["spectral"] = {...}
        keys include: peakf, meanf, peakA, bandwidth, f_low, f_high
        and optional: ssam {f, A}, bandratio {freqlims, low_sum, high_sum, ratio}
    """
    m = _ensure_metrics(tr)
    if "spectral" not in m or not isinstance(m["spectral"], dict):
        m["spectral"] = {}
    spm = m["spectral"]  # shorthand to the spectral sub-dict

    if not hasattr(tr.stats, "spectral") or tr.stats.spectral is None:
        tr.stats.spectral = {}

    # ---- sanitize input ----
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- compute spectrum (helper first, fallback to rFFT) ----
    F = A = None
    if helper is not None:
        try:
            from obspy import Stream
            tmp = tr.copy(); tmp.data = y
            st = Stream([tmp])
            st = helper(st, **(helper_opts or {})) or st
            spec = getattr(st[0].stats, "spectral", None)
            if spec and "freqs" in spec and "amplitudes" in spec:
                F = np.asarray(spec["freqs"], dtype=float)
                A = np.asarray(spec["amplitudes"], dtype=float)
        except Exception:
            F = A = None

    if F is None or A is None:
        N = y.size
        A = np.abs(rfft(y))
        F = rfftfreq(N, d=dt)

    # persist arrays on trace
    tr.stats.spectral["freqs"] = F
    tr.stats.spectral["amplitudes"] = A

    # ---- bandwidth/peak metrics (populate into m["spectral"]) ----
    bw = get_bandwidth(
        F, A,
        threshold=threshold,
        window_length=window_length,
        polyorder=polyorder,
        trace=None  # we’ll merge into m["spectral"] ourselves
    )
    spm.update(bw)

    # ---- power-weighted mean frequency (spectral centroid) ----
    # P_i = |FFT(y)_i|**2. The common FFT scaling cancels in the ratio.
    # This is NOT the median frequency and NOT an amplitude-weighted mean.
    valid = np.isfinite(F) & np.isfinite(A) & (A >= 0)
    fv = np.asarray(F[valid], dtype=float)
    av = np.asarray(A[valid], dtype=float)
    if av.size and np.any(av > 0):
        peak_idx = int(np.argmax(av))
        peakf = float(fv[peak_idx])
        peakA = float(av[peak_idx])
        # Scale before squaring to avoid overflow on large spectra.
        scaled = av / peakA
        power = scaled * scaled
        total_power = float(np.sum(power))
        meanf = float(np.sum(fv * power) / total_power) if total_power > 0 else np.nan
    else:
        peakf = meanf = peakA = np.nan

    spm["peakf"] = peakf
    spm["meanf"] = meanf
    spm["peakA"] = peakA
    # Remove stale legacy median-frequency metrics on repeated processing.
    spm.pop("medianf", None)

    s = getattr(tr.stats, "spectrum", {}) or {}
    s["peakF"] = peakf
    s["meanF"] = meanf
    s["peakA"] = peakA
    s.pop("medianF", None)
    tr.stats.spectrum = s

    # ---- optional extras under m["spectral"] ----
    if compute_ssam:
        _ssam(tr)  # writes into m["spectral"]["ssam"]

    if compute_bandratios:
        _band_ratio(tr, bands=bands)


def _ssam(tr: Trace, freq_bins: np.ndarray | None = None) -> None:
    """
    Bin the already-computed amplitude spectrum into SSAM bands.
    Stores under: tr.stats.metrics["spectral"]["ssam"] = {'f': centers, 'A': values}
    """
    if freq_bins is None:
        freq_bins = np.arange(0.0, 16.0, 1.0)

    spec = getattr(tr.stats, "spectral", None)
    if not spec or "freqs" not in spec or "amplitudes" not in spec:
        # silent no-op to keep pipelines flowing
        return

    f = np.asarray(spec["freqs"], dtype=float)
    A = np.asarray(spec["amplitudes"], dtype=float)

    centers, values = [], []
    for i in range(len(freq_bins) - 1):
        fmin, fmax = freq_bins[i], freq_bins[i + 1]
        idx = np.where((f >= fmin) & (f < fmax))[0]
        centers.append(0.5 * (fmin + fmax))
        values.append(np.nanmean(A[idx]) if idx.size else np.nan)

    m = _ensure_metrics(tr)
    spm = m.setdefault("spectral", {})
    spm["ssam"] = {"f": np.asarray(centers), "A": np.asarray(values)}


def _band_ratio(
    tr: Trace,
    freqlims: list[float] | None = None,
    *,
    bands: tuple[float, float, float, float] | None = None,
) -> None:
    """Log2 ratio of mean spectral amplitude per Hz in two bands.

    `freqlims=(low, split, high)` is accepted for backward compatibility.
    The output retains `low_sum` and `high_sum` for historical consumers,
    but ratio uses bandwidth-normalized means so unequal linear bandwidths
    do not bias the ratio. `ratio_sum_legacy` records the old definition.
    """
    if bands is None:
        bands = (freqlims[0], freqlims[1], freqlims[1], freqlims[2]) if freqlims is not None else (0.8, 4.0, 4.0, 20.0)
    l0, l1, h0, h1 = map(float, bands)
    if not (0 <= l0 < l1 <= h0 < h1):
        raise ValueError(f"Invalid bands: {bands}")
    spec = getattr(tr.stats, "spectral", None)
    if not spec or "freqs" not in spec or "amplitudes" not in spec:
        return
    f = np.asarray(spec["freqs"], dtype=float)
    A = np.asarray(spec["amplitudes"], dtype=float)
    valid = np.isfinite(f) & np.isfinite(A) & (A >= 0)
    low = valid & (f >= l0) & (f < l1)
    high = valid & (f >= h0) & (f <= h1)
    low_sum = float(np.sum(A[low])) if low.any() else np.nan
    high_sum = float(np.sum(A[high])) if high.any() else np.nan
    low_mean = float(np.mean(A[low])) if low.any() else np.nan
    high_mean = float(np.mean(A[high])) if high.any() else np.nan
    ratio = float(np.log2(high_mean / low_mean)) if low_mean > 0 and high_mean > 0 else np.nan
    legacy = float(np.log2(high_sum / low_sum)) if low_sum > 0 and high_sum > 0 else np.nan
    m = _ensure_metrics(tr)
    m.setdefault("spectral", {})["bandratio"] = {
        "bands": [l0, l1, h0, h1],
        "freqlims": [l0, l1, h1] if l1 == h0 else None,
        "low_sum": low_sum, "high_sum": high_sum,
        "low_mean": low_mean, "high_mean": high_mean,
        "ratio": ratio, "ratio_sum_legacy": legacy,
        "definition": "log2(mean_fft_amplitude_high / mean_fft_amplitude_low)",
    }


def get_bandwidth(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    *,
    threshold: float = 0.707,
    window_length: int = 9,
    polyorder: int = 2,
    trace: Trace | None = None,  # kept for API compat; we return the dict
) -> dict:
    """
    Estimate peak frequency and -3 dB style bandwidth on a smoothed spectrum.
    Returns a dict; caller can merge into metrics.
    """
    f = np.asarray(frequencies, dtype=float)
    A = np.asarray(amplitudes, dtype=float)

    if f.size == 0 or A.size == 0:
        return {"f_peak": np.nan, "A_peak": np.nan, "f_low": np.nan, "f_high": np.nan, "bandwidth": np.nan}

    wl = int(window_length)
    if wl < 3: wl = 3
    if wl % 2 == 0: wl += 1
    wl = min(wl, max(3, (A.size // 2) * 2 - 1))

    try:
        smoothed = savgol_filter(A, window_length=wl, polyorder=int(polyorder))
    except Exception:
        smoothed = A

    if np.any(np.isfinite(smoothed)):
        peak_index = int(np.nanargmax(smoothed))
        f_peak = float(f[peak_index]) if f.size else np.nan
        A_peak = float(smoothed[peak_index]) if smoothed.size else np.nan
        cutoff = A_peak * float(threshold) if np.isfinite(A_peak) else np.nan
    else:
        f_peak = A_peak = cutoff = np.nan
        peak_index = 0

    if np.isfinite(cutoff):
        lower = np.where(smoothed[:peak_index] < cutoff)[0]
        f_low = float(f[lower[-1]]) if lower.size else float(f[0])
        upper = np.where(smoothed[peak_index:] < cutoff)[0]
        f_high = float(f[peak_index + upper[0]]) if upper.size else float(f[-1])
        bw = f_high - f_low
    else:
        f_low = f_high = bw = np.nan

    return {
        "f_peak": f_peak,
        "A_peak": A_peak,
        "f_low": f_low,
        "f_high": f_high,
        "bandwidth": float(bw) if np.isfinite(bw) else np.nan,
    }