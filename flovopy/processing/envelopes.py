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


def align_waveforms(st: Stream, max_lag_s: float = 30.0, min_corr: float = 0.01, reference: Optional[str] = None, smooth_s: Optional[float]=1.0, smooth_mode: Optional[str]="ma", decimate_to: Optional[float]=5.0, plot_envelopes: bool = True) -> Stream:
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
    env_st = envelopes_stream(st, smooth_s=smooth_s, smooth_mode=smooth_mode, decimate_to=decimate_to, mode="per-trace")
    if plot_envelopes:
        env_st.plot(equal_scale=False);
    delays = envelope_delays(env_st, max_lag_s=max_lag_s, min_corr=min_corr)

    if reference is None:
        reference = pick_reference_station(env_st)

    lags = compute_station_lags(delays, reference)

    out = Stream()
    for tr in st:
        sta = tr.stats.station
        lag = lags.get(sta, 0.0)
        out += shift_trace(tr, lag)

    if plot_envelopes:
        es = Stream()
        for tr in env_st:
            sta = tr.stats.station
            lag = lags.get(sta, 0.0)
            es += shift_trace(tr, lag)
        es.plot(equal_scale=False);     
    return out, lags


# --------- Global lag solve (no grid required) --------- #
# Solve s_i - s_j ≈ tau_ij (weighted). Pin one station to zero.

def solve_global_station_lags(
    delays: Dict[Tuple[str, str], Tuple[float, float]],
    stations: Iterable[str],
    ref_station: Optional[str] = None,
    weight_power: float = 1.0,   # use corr^p as weights
) -> Dict[str, float]:
    """
    Least-squares lags s (seconds) for each station so that (s_i - s_j) fits τ_ij.
    Returns lags dict with one station fixed to 0.
    """
    stations = list(sorted(stations))
    if ref_station is None:
        ref_station = stations[0]
    idx = {sta: k for k, sta in enumerate(stations)}
    n = len(stations)

    rows = []
    b = []
    w = []

    for (si, sj), (tau, corr) in delays.items():
        if si not in idx or sj not in idx:
            continue
        row = np.zeros(n)
        row[idx[si]] = 1.0
        row[idx[sj]] = -1.0
        rows.append(row)
        b.append(tau)
        w.append(max(corr, 1e-6) ** weight_power)

    if not rows:
        return {sta: 0.0 for sta in stations}

    A = np.vstack(rows)
    b = np.asarray(b, float)
    W = np.diag(np.sqrt(np.asarray(w, float)))  # weight matrix

    # Pin reference by removing its column (equivalent to s_ref=0)
    keep = [k for k, s in enumerate(stations) if s != ref_station]
    A_red = A[:, keep]
    Aw = W @ A_red
    bw = W @ b

    s_red, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    s = np.zeros(n); s[keep] = s_red

    return {sta: float(s[idx[sta]]) for sta in stations}


def shift_trace_starttime(tr: Trace, lag_s: float) -> Trace:
    tr2 = tr.copy()
    tr2.stats.starttime += lag_s
    return tr2


def align_waveforms_global(
    st: Stream,
    max_lag_s: float = 30.0,
    min_corr: float = 0.25,
    ref_station: Optional[str] = None,
    weight_power: float = 1.0,
) -> Tuple[Stream, Dict[str, float], Dict[Tuple[str, str], Tuple[float, float]]]:
    """
    Envelopes -> pairwise delays -> global lags -> shift original waveforms.

    Returns
    -------
    aligned_st : Stream
    lags       : {station: lag_s} (ref station = 0)
    delays     : {(si,sj): (tau_ij_s, corr)}
    """
    env = envelopes_stream(st)
    delays = envelope_delays(env, max_lag_s=max_lag_s, min_corr=min_corr)
    stations = sorted({tr.stats.station for tr in st})
    if ref_station is None:
        # choose the station with the most/longest samples as a stable ref
        ref_station = max(st, key=lambda tr: tr.stats.npts).stats.station
    lags = solve_global_station_lags(delays, stations, ref_station=ref_station, weight_power=weight_power)

    out = Stream()
    for tr in st:
        out += shift_trace_starttime(tr, lags.get(tr.stats.station, 0.0))
    return out, lags, delays


# --------- Optional: grid + speed scan using node_distances_km --------- #
# Minimizes residuals of: tau_ij ≈ (d_i - d_j)/c + s_i - s_j

def locate_with_grid_from_delays(
    node_distances_km: Dict[str, np.ndarray],  # {station: array[n_nodes]}
    delays: Dict[Tuple[str, str], Tuple[float, float]],
    c_range: Tuple[float, float] = (0.4, 3.0),
    n_c: int = 41,
    corr_power: float = 1.0,
    robust: str = "mad"  # "mad" or "rms"
) -> Dict[str, object]:
    stations = sorted(node_distances_km.keys())
    S = len(stations)
    n_nodes = next(iter(node_distances_km.values())).size
    D = np.vstack([node_distances_km[s] for s in stations])  # (S, n_nodes)

    # Pack observed delays and weights
    pairs = list(delays.keys())
    tau_obs = np.array([delays[p][0] for p in pairs], float)
    w = np.array([max(delays[p][1], 1e-6) for p in pairs], float) ** corr_power

    speeds = np.linspace(c_range[0], c_range[1], n_c)
    best = dict(score=np.inf)

    # helper to solve station statics for a set of pair residuals b_ij
    def solve_statics(b_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Build A s = b where each row encodes (s_i - s_j) = b_ij
        rows = []
        for k, (si, sj) in enumerate(pairs):
            ri = np.zeros(S); rj = np.zeros(S)
            ri[stations.index(si)] = 1.0
            rj[stations.index(sj)] = -1.0
            rows.append(ri + rj)
        A = np.vstack(rows)
        # Pin last station to 0
        A_red = A[:, :S-1]
        s_red, *_ = np.linalg.lstsq(A_red * np.sqrt(w[:, None]), b_vec * np.sqrt(w), rcond=None)
        s = np.zeros(S); s[:S-1] = s_red
        # residuals r_ij = (s_i - s_j) - b_ij
        res = []
        for k, (si, sj) in enumerate(pairs):
            i = stations.index(si); j = stations.index(sj)
            res.append((s[i] - s[j]) - b_vec[k])
        return s, np.asarray(res, float)

    for g in range(n_nodes):
        d = D[:, g]  # distances at node g (km)
        for c in speeds:
            # predicted differential times for each pair at (g,c)
            pred = np.array([(d[stations.index(si)] - d[stations.index(sj)]) / c for (si, sj) in pairs], float)
            # b_ij = tau_obs - pred
            b = tau_obs - pred
            s, res = solve_statics(b)
            rw = res * np.sqrt(w)
            if robust == "mad":
                score = 1.4826 * np.median(np.abs(rw))
            else:
                score = float(np.sqrt(np.mean(rw**2)))

            if score < best.get("score", np.inf):
                best = dict(node=g, speed=c, score=score, residuals=res, statics={stations[i]: float(s[i]) for i in range(S)}, n_pairs=len(pairs))

    return best

from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable
import numpy as np
from obspy import Stream, Trace
from obspy.signal.filter import envelope as obspy_envelope


# ---------------------------- Envelopes (per Trace) ----------------------------

def envelope_trace(
    tr: Trace,
    *,
    smooth_s: float = 0.0,                 # 0 => no extra smoothing
    smooth_mode: str = "ma",               # "ma" | "lowpass" | "causal_lowpass"
    decimate_to: Optional[float] = None,   # resample **envelope** to this rate
) -> Trace:
    """
    Compute ObsPy Hilbert envelope for a single (Z) Trace and return as a Trace.
    Preprocessing (taper, filter, etc.) is assumed done upstream.
    """
    fs = float(tr.stats.sampling_rate)
    env = obspy_envelope(tr.data.astype(np.float64))

    # Optional smoothing of the envelope (not the waveform)
    if smooth_s and smooth_s > 0:
        if smooth_mode == "ma":
            n = max(1, int(round(smooth_s * fs)))
            if n > 1:
                k = np.ones(n, dtype=np.float64) / float(n)
                env = np.convolve(env, k, mode="same")
        else:
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

    # Optional envelope-only resample (cheaper xcorr later)
    if decimate_to and decimate_to < fs:
        env_tr.resample(decimate_to)

    # Tag channel last char as 'E' (envelope) if 3-char channel
    if hasattr(env_tr.stats, "channel") and len(env_tr.stats.channel) == 3:
        env_tr.stats.channel = env_tr.stats.channel[:2] + "E"
    return env_tr


def envelopes_stream(
    st: Stream,
    *,
    smooth_s: float = 0.0,
    smooth_mode: str = "ma",
    decimate_to: Optional[float] = None,
) -> Stream:
    """Per-trace envelopes for a Stream of Z-components."""
    out = Stream()
    for tr in st:
        out += envelope_trace(tr, smooth_s=smooth_s, smooth_mode=smooth_mode, decimate_to=decimate_to)
    return out


# --------------------- Envelope xcorr → Pairwise Delays ------------------------

def _norm_xcorr(a: np.ndarray, b: np.ndarray, max_lag_s: float, fs: float) -> Tuple[float, float]:
    """
    Normalized cross-correlation via FFT. Returns (lag_s, peak_corr).
    Positive lag means: b(t) best matches a(t - lag).
    """
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
    if not np.any(m):
        return 0.0, 0.0
    r = r[m]; lags = lags[m]
    i = int(np.argmax(r))
    return float(lags[i]), float(r[i])


def envelope_delays(
    env_st: Stream,
    *,
    max_lag_s: float = 30.0,
    min_corr: float = 0.25,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Pairwise envelope delays between stations in env_st.
    Returns {(sta_i, sta_j): (lag_s, corr)} for i<j.
    """
    stations = sorted({tr.stats.station for tr in env_st})
    by_sta = {sta: env_st.select(station=sta)[0] for sta in stations}
    fs = float(next(iter(by_sta.values())).stats.sampling_rate)

    delays: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            si, sj = stations[i], stations[j]
            tri, trj = by_sta[si], by_sta[sj]
            L = min(tri.stats.npts, trj.stats.npts)
            lag, cc = _norm_xcorr(tri.data[:L], trj.data[:L], max_lag_s, fs)
            if cc >= min_corr:
                delays[(si, sj)] = (lag, cc)
    return delays


# --------------------- Global lag solve & waveform alignment -------------------

def solve_global_station_lags(
    delays: Dict[Tuple[str, str], Tuple[float, float]],
    stations: Iterable[str],
    ref_station: Optional[str] = None,
    weight_power: float = 1.0,
) -> Dict[str, float]:
    """
    Solve station lags s so (s_i - s_j) ≈ τ_ij for all pairs (weighted by corr^p).
    One station is pinned to 0 (reference).
    """
    stations = list(sorted(stations))
    if ref_station is None:
        ref_station = stations[0]
    idx = {sta: k for k, sta in enumerate(stations)}
    n = len(stations)

    rows, b, w = [], [], []
    for (si, sj), (tau, corr) in delays.items():
        if si not in idx or sj not in idx:
            continue
        row = np.zeros(n); row[idx[si]] = 1.0; row[idx[sj]] = -1.0
        rows.append(row); b.append(tau); w.append(max(corr, 1e-6) ** weight_power)

    if not rows:
        return {sta: 0.0 for sta in stations}

    A = np.vstack(rows)                 # M x S
    b = np.asarray(b, float)            # M
    W = np.diag(np.sqrt(np.asarray(w, float)))  # M x M

    keep = [k for k, s in enumerate(stations) if s != ref_station]
    A_red = A[:, keep]
    Aw = W @ A_red
    bw = W @ b

    s_red, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    s = np.zeros(n); s[keep] = s_red

    return {sta: float(s[idx[sta]]) for sta in stations}


def _shift_starttime(tr: Trace, lag_s: float) -> Trace:
    tr2 = tr.copy()
    tr2.stats.starttime += lag_s
    return tr2


def align_waveforms(
    st: Stream,
    *,
    max_lag_s: float = 30.0,
    min_corr: float = 0.25,
    smooth_s: float = 0.0,
    smooth_mode: str = "ma",
    decimate_to: Optional[float] = None,
    reference: Optional[str] = None,
) -> Tuple[Stream, Dict[str, float]]:
    """
    Reference-based alignment (simple): uses only pairs with the reference.
    Kept for convenience; global method below is recommended.
    """
    env_st = envelopes_stream(st, smooth_s=smooth_s, smooth_mode=smooth_mode, decimate_to=decimate_to)
    delays = envelope_delays(env_st, max_lag_s=max_lag_s, min_corr=min_corr)
    stations = sorted({tr.stats.station for tr in st})
    if reference is None:
        reference = max(st, key=lambda tr: tr.stats.npts).stats.station

    # Average all available delays to/from reference
    lags = {reference: 0.0}
    for (si, sj), (tau, _) in delays.items():
        if si == reference:
            lags[sj] = -tau
        elif sj == reference:
            lags[si] = tau

    out = Stream()
    for tr in st:
        out += _shift_starttime(tr, lags.get(tr.stats.station, 0.0))
    return out, lags


def align_waveforms_global(
    st: Stream,
    *,
    max_lag_s: float = 30.0,
    min_corr: float = 0.25,
    smooth_s: float = 0.0,
    smooth_mode: str = "ma",
    decimate_to: Optional[float] = None,
    ref_station: Optional[str] = None,
    weight_power: float = 1.0,
) -> Tuple[Stream, Dict[str, float], Dict[Tuple[str, str], Tuple[float, float]]]:
    """
    Global alignment: envelopes → pairwise delays → least-squares lags → shift waveforms.
    Returns (aligned Stream, {station: lag_s}, pairwise delays).
    """
    env = envelopes_stream(st, smooth_s=smooth_s, smooth_mode=smooth_mode, decimate_to=decimate_to)
    delays = envelope_delays(env, max_lag_s=max_lag_s, min_corr=min_corr)
    stations = sorted({tr.stats.station for tr in st})
    if ref_station is None and len(stations) > 0:
        ref_station = max(st, key=lambda tr: tr.stats.npts).stats.station
    lags = solve_global_station_lags(delays, stations, ref_station=ref_station, weight_power=weight_power)

    out = Stream()
    for tr in st:
        out += _shift_starttime(tr, lags.get(tr.stats.station, 0.0))
    return out, lags, delays


from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from obspy import Inventory, Stream

# Reuse your distances loader
from flovopy.asl.distances import compute_or_load_distances


def locate_with_grid_from_delays(
    grid,
    *,
    inventory: Inventory,
    delays: Dict[Tuple[str, str], Tuple[float, float]],
    stream: Optional[Stream] = None,   # lets compute_or_load_distances pick station IDs
    c_range: Tuple[float, float] = (0.4, 3.0),
    n_c: int = 61,
    corr_power: float = 1.0,
    robust: str = "mad"  # "mad" | "rms"
) -> Dict[str, object]:
    """
    Use precomputed envelope delays + grid node→station distances to estimate
    (node, apparent speed) and per-station statics, à la enveloc.
    """
    # 1) Distances from your cache/compute path
    node_distances_km, coords_by_sta, meta = compute_or_load_distances(
        grid,
        inventory=inventory,
        stream=stream,
        use_elevation=True,
    )
    stations = sorted(node_distances_km.keys())
    S = len(stations)
    n_nodes = next(iter(node_distances_km.values())).size
    D = np.vstack([node_distances_km[s] for s in stations])  # (S, n_nodes)

    # 2) Pack observed delays (τ_ij) and weights (corr^p)
    pairs = [(si, sj) for (si, sj) in delays.keys() if si in stations and sj in stations]
    if not pairs:
        return {"ok": False, "reason": "no_valid_pairs"}
    tau_obs = np.array([delays[p][0] for p in pairs], float)
    w = np.array([max(delays[p][1], 1e-6) for p in pairs], float) ** corr_power

    speeds = np.linspace(c_range[0], c_range[1], n_c)
    best = dict(score=np.inf)

    # Build pairwise design (s_i - s_j rows) once
    row_idx = {s: i for i, s in enumerate(stations)}
    A_rows = []
    for (si, sj) in pairs:
        row = np.zeros(S); row[row_idx[si]] = 1.0; row[row_idx[sj]] = -1.0
        A_rows.append(row)
    A = np.vstack(A_rows)  # M x S

    # Helper: solve station statics s for a given residual vector b_ij
    def solve_statics_weighted(b_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A_red = A[:, :S-1]  # pin the last station to 0
        W = np.diag(np.sqrt(w))
        s_red, *_ = np.linalg.lstsq(W @ A_red, W @ b_vec, rcond=None)
        s = np.zeros(S); s[:S-1] = s_red
        # residuals r_ij = (s_i - s_j) - b_ij
        res = []
        for k, (si, sj) in enumerate(pairs):
            i = row_idx[si]; j = row_idx[sj]
            res.append((s[i] - s[j]) - b_vec[k])
        return s, np.asarray(res, float)

    # 3) Scan nodes and speeds
    for g in range(n_nodes):
        d = D[:, g]
        for c in speeds:
            pred = np.array([(d[row_idx[si]] - d[row_idx[sj]]) / c for (si, sj) in pairs], float)
            b = tau_obs - pred
            s, res = solve_statics_weighted(b)
            rw = res * np.sqrt(w)
            if robust == "mad":
                score = 1.4826 * np.median(np.abs(rw))
            else:
                score = float(np.sqrt(np.mean(rw**2)))
            if score < best.get("score", np.inf):
                best = dict(
                    ok=True,
                    node=g,
                    speed=float(c),
                    score=float(score),
                    statics={stations[i]: float(s[i]) for i in range(S)},
                    n_pairs=len(pairs),
                )

    # 4) Attach a few grid helpers if available
    try:
        lon, lat, elev = grid.node_lonlat(g=best["node"])
        best["lon"], best["lat"], best["elev_m"] = float(lon), float(lat), float(elev if elev is not None else np.nan)
    except Exception:
        pass

    return best