from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Dict

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.filter import envelope as obspy_envelope

def envelope_trace(
    tr: Trace,
    *,
    smooth_s: float = 0.0,             # 0 disables extra smoothing
    smooth_mode: str = "ma",           # "ma" | "lowpass" (zero-phase) | "causal_lowpass"
    decimate_to: Optional[float] = None,  # post-envelope resample target (Hz)
) -> Trace:
    """
    Compute the Hilbert envelope (ObsPy) and return it as an ObsPy Trace.

    Preprocessing is assumed done upstream (interpolate/detrend/taper/filter).

    Parameters
    ----------
    tr : Trace
        Preprocessed waveform trace.
    smooth_s : float
        Optional envelope smoothing window (seconds).
    smooth_mode : {"ma","lowpass","causal_lowpass"}
        Smoothing method. "ma" = moving average (centered).
    decimate_to : float or None
        If set, resample envelope trace to this sampling rate (Hz).

    Returns
    -------
    env_tr : Trace
        Trace whose .data is the envelope. starttime, id, etc. preserved.
    """
    fs = float(tr.stats.sampling_rate)
    x = tr.data.astype(np.float64, copy=False)
    env = obspy_envelope(x)  # ObsPy Hilbert-envelope

    # Optional smoothing
    if smooth_s and smooth_s > 0:
        if smooth_mode == "ma":
            n = max(1, int(round(smooth_s * fs)))
            if n > 1:
                k = np.ones(n, dtype=np.float64) / float(n)
                env = np.convolve(env, k, mode="same")
        else:
            # use ObsPy filtering by wrapping into a Trace
            tmp = tr.copy()
            tmp.data = env
            fc = max(0.5 / smooth_s, 0.1)
            fc = min(fc, 0.45 * fs)
            if smooth_mode == "lowpass":
                tmp.filter("lowpass", freq=fc, corners=2, zerophase=True)
            elif smooth_mode == "causal_lowpass":
                tmp.filter("lowpass", freq=fc, corners=2, zerophase=False)
            else:
                raise ValueError(f"Unknown smooth_mode={smooth_mode}")
            env = tmp.data

    env_tr = tr.copy()
    env_tr.data = env

    # Optional final resample (envelope only)
    if decimate_to and decimate_to < fs:
        env_tr.resample(decimate_to)

    return env_tr


def vector_envelope(
    st: Stream,
    station: Optional[str] = None,
    select_components: Sequence[str] = ("Z", "N", "E"),
    **env_kwargs,
) -> Optional[Trace]:
    """
    Compute a 3-component RSS (vector) envelope as an ObsPy Trace.

    Returns None if required components are missing.
    """
    st_use = st.copy()
    if station:
        st_use = st_use.select(station=station)

    comps = []
    for c in select_components:
        trc = st_use.select(component=c)
        if len(trc) == 0:
            return None
        comps.append(trc[0])

    # Compute envelope per component with identical kwargs
    env_traces = [envelope_trace(trc, **env_kwargs) for trc in comps]

    # Align to shortest
    L = min(tr.stats.npts for tr in env_traces)
    for trc in env_traces:
        if trc.stats.npts > L:
            trc.data = trc.data[:L]
            trc.stats.npts = L

    rss = np.sqrt(np.sum([trc.data**2 for trc in env_traces], axis=0))

    out = env_traces[0].copy()
    out.data = rss.astype(np.float64)
    # Tag component as 'V' (vector) if you like
    if hasattr(out.stats, "channel") and len(out.stats.channel) == 3:
        out.stats.channel = out.stats.channel[:2] + "V"
    return out


def envelopes_stream(
    st: Stream,
    mode: str = "per-trace",   # "per-trace" or "vector-per-station"
    select_components: Sequence[str] = ("Z", "N", "E"),
    stations: Optional[Iterable[str]] = None,
    **env_kwargs,
) -> Stream:
    """
    Convert a waveform Stream to an envelope Stream (ObsPy Traces),
    ready for enveloc-style location (which expects envelopes as input).

    mode="per-trace": envelope for each trace independently (1:1).
    mode="vector-per-station": one RSS envelope per station (3C required).
    """
    if mode not in {"per-trace", "vector-per-station"}:
        raise ValueError("mode must be 'per-trace' or 'vector-per-station'")

    if mode == "per-trace":
        out = Stream()
        for tr in st:
            out += envelope_trace(tr, **env_kwargs)
        return out

    # vector-per-station
    out = Stream()
    sta_list = sorted(stations) if stations else sorted({tr.stats.station for tr in st})
    for sta in sta_list:
        vtr = vector_envelope(st, station=sta, select_components=select_components, **env_kwargs)
        if vtr is not None:
            out += vtr
    return out


# ---------- Envelope cross-correlation (pairwise delays) ---------- #

def _norm_xcorr(a: np.ndarray, b: np.ndarray, max_lag_s: float, fs: float) -> Tuple[float, float]:
    """
    Normalized cross-correlation (via FFT). Returns (lag_s, peak_corr)

    Positive lag means: b(t) best matches a(t - lag)
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    L = min(a.size, b.size)
    a = a[:L] - a[:L].mean()
    b = b[:L] - b[:L].mean()
    if L < 2 or np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0, 0.0

    nfft = 1 << int(np.ceil(np.log2(2 * L - 1)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    r = np.fft.irfft(A * np.conj(B), nfft)
    # shift to [-L+1, L-1]
    r = np.concatenate((r[-(L - 1):], r[:L]))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0, 0.0
    r /= denom

    lags = np.arange(-L + 1, L) / fs
    if max_lag_s is not None:
        m = np.abs(lags) <= max_lag_s
        r = r[m]
        lags = lags[m]
        if r.size == 0:
            return 0.0, 0.0

    i = int(np.argmax(r))
    return float(lags[i]), float(r[i])


def envelope_delays(
    env_st: Stream,
    max_lag_s: float = 40.0,
    min_corr: float = 0.2,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Build pairwise envelope delays from an envelope Stream.

    Parameters
    ----------
    env_st : Stream
        Stream of envelope Traces (e.g., from envelopes_stream()).
    max_lag_s : float
        Restrict search window for stability/speed.
    min_corr : float
        Drop very weak pairs.

    Returns
    -------
    delays : dict
        {(sta_i, sta_j): (lag_seconds, peak_correlation)}, for i<j.
    """
    # Use station code as key; if multiple traces per station, take the first
    by_sta: Dict[str, Trace] = {}
    for tr in env_st:
        by_sta.setdefault(tr.stats.station, tr)

    stations = sorted(by_sta.keys())
    delays: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            si, sj = stations[i], stations[j]
            tri = by_sta[si]; trj = by_sta[sj]
            fs = float(tri.stats.sampling_rate)
            # align to shortest, same fs assumed in pre-processing path
            L = min(tri.stats.npts, trj.stats.npts)
            lag, cc = _norm_xcorr(tri.data[:L], trj.data[:L], max_lag_s, fs)
            if cc >= min_corr:
                delays[(si, sj)] = (lag, cc)
    return delays



'''
def envelope_trace(tr: Trace) -> Trace:
    """Return Hilbert envelope of a single Trace as a Trace (Z-component)."""
    env = obspy_envelope(tr.data.astype(np.float64))
    out = tr.copy()
    out.data = env
    # Optionally tag channel with 'E' for envelope
    if hasattr(out.stats, "channel") and len(out.stats.channel) == 3:
        out.stats.channel = out.stats.channel[:2] + "E"
    return out


def envelopes_stream(st: Stream) -> Stream:
    """Return per-trace envelopes for a Stream of Z components."""
    out = Stream()
    for tr in st:
        out += envelope_trace(tr)
    return out


def _norm_xcorr(a: np.ndarray, b: np.ndarray, max_lag_s: float, fs: float) -> Tuple[float, float]:
    """Normalized cross-correlation; returns (lag_s, peak_corr)."""
    L = min(len(a), len(b))
    a = a[:L] - a[:L].mean()
    b = b[:L] - b[:L].mean()
    if L < 2 or np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0, 0.0

    nfft = 1 << int(np.ceil(np.log2(2 * L - 1)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    r = np.fft.irfft(A * np.conj(B), nfft)
    r = np.concatenate((r[-(L - 1):], r[:L]))  # shift to [-L+1, L-1]

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0, 0.0
    r /= denom

    lags = np.arange(-L + 1, L) / fs
    m = np.abs(lags) <= max_lag_s
    r = r[m]; lags = lags[m]
    if r.size == 0:
        return 0.0, 0.0

    i = int(np.argmax(r))
    return float(lags[i]), float(r[i])


def envelope_delays(env_st: Stream, max_lag_s: float = 30.0, min_corr: float = 0.25) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Compute pairwise delays between station envelopes.

    Returns dict {(sta_i, sta_j): (lag_s, peak_corr)}, for i<j.
    """
    stations = sorted({tr.stats.station for tr in env_st})
    by_sta = {sta: env_st.select(station=sta)[0] for sta in stations}
    fs = float(next(iter(by_sta.values())).stats.sampling_rate)

    delays: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            si, sj = stations[i], stations[j]
            tri, trj = by_sta[si], by_sta[sj]
            lag, cc = _norm_xcorr(tri.data, trj.data, max_lag_s, fs)
            if cc >= min_corr:
                delays[(si, sj)] = (lag, cc)
    return delays

'''
def pick_reference_station(env_st: Stream) -> str:
    """Pick reference station (longest record length, as fallback)."""
    return max(env_st, key=lambda tr: tr.stats.npts).stats.station



def compute_station_lags(delays: Dict[Tuple[str, str], Tuple[float, float]], reference: str) -> Dict[str, float]:
    """
    Turn pairwise delays into per-station lag relative to reference.

    Simple approach: average all available pairwise paths to reference.
    """
    lags = {reference: 0.0}
    for (si, sj), (lag, _) in delays.items():
        if si == reference:
            lags[sj] = -lag
        elif sj == reference:
            lags[si] = lag
    return lags


def shift_trace(tr: Trace, lag_s: float) -> Trace:
    """
    Shift a trace in time by lag_s.
    Positive lag means: move trace later (starttime += lag).
    """
    tr2 = tr.copy()
    tr2.stats.starttime += lag_s
    return tr2


def align_waveforms(st: Stream, max_lag_s: float = 30.0, min_corr: float = 0.25, reference: Optional[str] = None) -> Stream:
    """
    Align waveforms by cross-correlating their envelopes.

    Parameters
    ----------
    st : Stream of Z-component Traces (preprocessed).
    max_lag_s : float
        Max lag to search in seconds.
    min_corr : float
        Minimum correlation to keep a delay.
    reference : str or None
        Station to use as reference (None => auto-pick).

    Returns
    -------
    aligned_st : Stream
        Copy of input Stream with .starttime shifted to align traces.
    lags : dict
        {station: lag_seconds relative to reference}.
    """
    env_st = envelopes_stream(st)
    delays = envelope_delays(env_st, max_lag_s=max_lag_s, min_corr=min_corr)

    if reference is None:
        reference = pick_reference_station(env_st)

    lags = compute_station_lags(delays, reference)

    out = Stream()
    for tr in st:
        sta = tr.stats.station
        lag = lags.get(sta, 0.0)
        out += shift_trace(tr, lag)
    return out, lags