"""
flovopy.processing.envelopes
----------------------------

ObsPy-native envelope utilities + alignment + optional grid/speed locator,
with VERY VERBOSE debugging when requested.

Author: Glenn-ready, 2025-10-06
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Dict, Union, List
from pathlib import Path
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from obspy import Inventory, Stream, Trace
from obspy.signal.filter import envelope as obspy_envelope

from flovopy.asl.distances import compute_or_load_distances


# --------------------------------------------------------------------------- #
#                               Small utilities                               #
# --------------------------------------------------------------------------- #

def _log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)


def _fmt_stats(st: Stream) -> str:
    if not st:
        return "Stream(empty)"
    stations = sorted({tr.stats.station for tr in st})
    fs = sorted({float(tr.stats.sampling_rate) for tr in st})
    npts = [tr.stats.npts for tr in st]
    return (f"{len(st)} traces | stations={stations} | fs={fs} | "
            f"npts[min/med/max]={min(npts)}/{int(np.median(npts))}/{max(npts)}")


# --------------------------------------------------------------------------- #
#                            Envelope computation                             #
# --------------------------------------------------------------------------- #

def envelope_trace(
    tr: Trace,
    *,
    smooth_s: float = 0.0,               # 0 => no extra smoothing
    smooth_mode: str = "ma",             # "ma" | "lowpass" | "causal_lowpass"
    decimate_to: Optional[float] = None, # resample **envelope** to this rate (Hz)
    verbose: bool = False,
) -> Trace:
    """
    Compute ObsPy Hilbert envelope for a single (Z) Trace and return as a Trace.
    Upstream preprocessing (interpolate/detrend/taper/filter) is assumed.
    """
    fs = float(tr.stats.sampling_rate)
    _log(f"[envelope_trace] {tr.id} fs={fs:.3f}Hz npts={tr.stats.npts}", verbose)
    env = obspy_envelope(tr.data.astype(np.float64))

    # Optional smoothing of the envelope
    if smooth_s and smooth_s > 0:
        _log(f"[envelope_trace] smoothing: mode={smooth_mode} window={smooth_s}s", verbose)
        if smooth_mode == "ma":
            n = max(1, int(round(smooth_s * fs)))
            if n > 1:
                k = np.ones(n, dtype=np.float64) / float(n)
                env = np.convolve(env, k, mode="same")
        else:
            tmp = tr.copy()
            tmp.data = env
            fc = max(0.5 / smooth_s, 0.1)   # crude mapping from window to LP cutoff
            fc = min(fc, 0.45 * fs)
            _log(f"[envelope_trace] lowpass fc≈{fc:.3f}Hz zerophase={smooth_mode=='lowpass'}", verbose)
            if smooth_mode == "lowpass":
                tmp.filter("lowpass", freq=fc, corners=2, zerophase=True)
            elif smooth_mode == "causal_lowpass":
                tmp.filter("lowpass", freq=fc, corners=2, zerophase=False)
            else:
                raise ValueError(f"Unknown smooth_mode={smooth_mode!r}")
            env = tmp.data

    env_tr = tr.copy()
    env_tr.data = env

    # Optional envelope-only resample (for cheaper xcorr later)
    if decimate_to and decimate_to < fs:
        _log(f"[envelope_trace] resample envelope -> {decimate_to:.3f}Hz", verbose)
        env_tr.resample(decimate_to)

    # Tag channel last char as 'E' if 3-char code (cosmetic)
    if hasattr(env_tr.stats, "channel") and len(env_tr.stats.channel) == 3:
        env_tr.stats.channel = env_tr.stats.channel[:2] + "E"

    return env_tr


def vector_envelope(
    st: Stream,
    station: Optional[str] = None,
    select_components: Sequence[str] = ("Z", "N", "E"),
    verbose: bool = False,
    **env_kwargs,
) -> Optional[Trace]:
    """
    Compute a 3-C RSS (vector) envelope as an ObsPy Trace. Returns None if any
    requested component is missing.
    """
    st_use = st.copy()
    if station:
        st_use = st_use.select(station=station)

    comps = []
    for c in select_components:
        trc = st_use.select(component=c)
        if len(trc) == 0:
            _log(f"[vector_envelope] {station}: missing component {c}", verbose)
            return None
        comps.append(trc[0])

    env_traces = [envelope_trace(trc, verbose=verbose, **env_kwargs) for trc in comps]

    # Align to shortest
    L = min(tr.stats.npts for tr in env_traces)
    for trc in env_traces:
        if trc.stats.npts > L:
            trc.data = trc.data[:L]
            trc.stats.npts = L

    rss = np.sqrt(np.sum([trc.data**2 for trc in env_traces], axis=0))

    out = env_traces[0].copy()
    out.data = rss.astype(np.float64)
    if hasattr(out.stats, "channel") and len(out.stats.channel) == 3:
        out.stats.channel = out.stats.channel[:2] + "V"
    return out


def envelopes_stream(
    st: Stream,
    *,
    mode: str = "per-trace",                   # "per-trace" | "vector-per-station"
    select_components: Sequence[str] = ("Z", "N", "E"),
    stations: Optional[Iterable[str]] = None,
    smooth_s: float = 0.0,
    smooth_mode: str = "ma",
    decimate_to: Optional[float] = None,
    verbose: bool = False,
) -> Stream:
    """
    Convert a waveform Stream to an envelope Stream (ObsPy Traces).

    mode="per-trace": envelope for each trace (1:1).
    mode="vector-per-station": one RSS envelope per station (needs 3-C).
    """
    _log(f"[envelopes_stream] mode={mode} :: { _fmt_stats(st) }", verbose)

    if mode not in {"per-trace", "vector-per-station"}:
        raise ValueError("mode must be 'per-trace' or 'vector-per-station'")

    out = Stream()
    if mode == "per-trace":
        for tr in st:
            out += envelope_trace(tr, smooth_s=smooth_s, smooth_mode=smooth_mode,
                                  decimate_to=decimate_to, verbose=verbose)
        return out

    sta_list = sorted(stations) if stations else sorted({tr.stats.station for tr in st})
    for sta in sta_list:
        vtr = vector_envelope(st, station=sta, select_components=select_components,
                              smooth_s=smooth_s, smooth_mode=smooth_mode,
                              decimate_to=decimate_to, verbose=verbose)
        if vtr is not None:
            out += vtr
    return out


# --------------------------------------------------------------------------- #
#                 Envelope cross-correlation → pairwise delays                #
# --------------------------------------------------------------------------- #

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
    if max_lag_s is not None:
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
    return_ccfs: bool = False,
    verbose: bool = False,
) -> Union[
    Dict[Tuple[str, str], Tuple[float, float]],
    Tuple[Dict[Tuple[str, str], Tuple[float, float]], Dict[Tuple[str, str], Dict[str, np.ndarray]]]
]:
    """
    Pairwise envelope delays between stations in env_st.
    If return_ccfs=True, also returns per-pair CCFs:
      ccfs[(si,sj)] = {"lags": array, "r": array, "peak_lag": float, "peak_corr": float}
    """
    stations = sorted({tr.stats.station for tr in env_st})
    _log(f"[envelope_delays] stations={stations}", verbose)
    if not stations:
        _log("[envelope_delays] WARNING: no stations", verbose)
        return ({}, {}) if return_ccfs else {}

    by_sta = {sta: env_st.select(station=sta)[0] for sta in stations}
    fs_set = {float(by_sta[s].stats.sampling_rate) for s in stations}
    _log(f"[envelope_delays] sampling rates present: {sorted(fs_set)}", verbose)
    if len(fs_set) > 1:
        _log("[envelope_delays] WARNING: mixed sampling rates across envelopes", verbose)

    fs = float(next(iter(by_sta.values())).stats.sampling_rate)

    delays: Dict[Tuple[str, str], Tuple[float, float]] = {}
    ccfs: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    tested = kept = 0
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            si, sj = stations[i], stations[j]
            tri, trj = by_sta[si], by_sta[sj]
            L = min(tri.stats.npts, trj.stats.npts)

            a = tri.data[:L] - tri.data[:L].mean()
            b = trj.data[:L] - trj.data[:L].mean()
            if L < 2 or np.allclose(a, 0) or np.allclose(b, 0):
                _log(f"[envelope_delays] skip {si}-{sj}: short/zero", verbose)
                continue
            tested += 1

            nfft = 1 << int(np.ceil(np.log2(2 * L - 1)))
            A = np.fft.rfft(a, nfft)
            B = np.fft.rfft(b, nfft)
            r_full = np.fft.irfft(A * np.conj(B), nfft)
            r_full = np.concatenate((r_full[-(L - 1):], r_full[:L]))
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom <= 0:
                _log(f"[envelope_delays] skip {si}-{sj}: denom<=0", verbose)
                continue
            r_full /= denom

            lags_full = np.arange(-L + 1, L) / fs
            m = np.abs(lags_full) <= max_lag_s
            if not np.any(m):
                _log(f"[envelope_delays] skip {si}-{sj}: no lags within ±{max_lag_s}s", verbose)
                continue
            r = r_full[m]; lags = lags_full[m]

            ipeak = int(np.argmax(r))
            lag = float(lags[ipeak]); corr = float(r[ipeak])
            _log(f"[envelope_delays] {si}-{sj}: lag={lag:+.3f}s corr={corr:.3f}", verbose)

            if corr >= min_corr:
                delays[(si, sj)] = (lag, corr)
                kept += 1
                if return_ccfs:
                    ccfs[(si, sj)] = {"lags": lags, "r": r, "peak_lag": lag, "peak_corr": corr}

    _log(f"[envelope_delays] pairs tested={tested}, kept={kept} (min_corr={min_corr})", verbose)

    if return_ccfs:
        return delays, ccfs
    return delays


# --------------------------------------------------------------------------- #
#                         Alignment (reference / global)                      #
# --------------------------------------------------------------------------- #

def pick_reference_station(env_st: Stream) -> str:
    """Pick reference station (longest record length, as a simple heuristic)."""
    return max(env_st, key=lambda tr: tr.stats.npts).stats.station


def compute_station_lags(
    delays: Dict[Tuple[str, str], Tuple[float, float]],
    reference: str,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Simple lags relative to a reference: use only pairs involving the reference.
    """
    lags = {reference: 0.0}
    for (si, sj), (lag, _) in delays.items():
        if si == reference:
            lags[sj] = -lag
        elif sj == reference:
            lags[si] = lag
    _log(f"[compute_station_lags] reference={reference} -> {lags}", verbose)
    return lags


def _shift_starttime(tr: Trace, lag_s: float) -> Trace:
    tr2 = tr.copy()
    tr2.stats.starttime += lag_s
    return tr2


def solve_global_station_lags(
    delays: Dict[Tuple[str, str], Tuple[float, float]],
    stations: Iterable[str],
    ref_station: Optional[str] = None,
    weight_power: float = 1.0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Least-squares lags s (seconds) so that (s_i - s_j) ≈ τ_ij for all pairs,
    weighted by corr^weight_power. One station pinned to 0.
    """
    stations = list(sorted(stations))
    if ref_station is None:
        ref_station = stations[0]
    idx = {sta: k for k, sta in enumerate(stations)}
    n = len(stations)

    rows, b, w = [], [], []
    for (si, sj), (tau, corr) in delays.items():
        if si not in idx or sj not in idx:
            _log(f"[solve_global_station_lags] skip pair {si}-{sj}: not in station set", verbose)
            continue
        row = np.zeros(n); row[idx[si]] = 1.0; row[idx[sj]] = -1.0
        rows.append(row); b.append(tau); w.append(max(corr, 1e-6) ** weight_power)

    if not rows:
        _log("[solve_global_station_lags] no usable rows; returning zeros", verbose)
        return {sta: 0.0 for sta in stations}

    A = np.vstack(rows)
    b = np.asarray(b, float)
    W = np.diag(np.sqrt(np.asarray(w, float)))

    keep = [k for k, s in enumerate(stations) if s != ref_station]
    A_red = A[:, keep]
    s_red, *_ = np.linalg.lstsq(W @ A_red, W @ b, rcond=None)
    s = np.zeros(n); s[keep] = s_red

    lags = {sta: float(s[idx[sta]]) for sta in stations}
    _log(f"[solve_global_station_lags] ref={ref_station} -> {lags}", verbose)
    return lags


def align_waveforms(
    st: Stream,
    *,
    max_lag_s: float = 30.0,
    min_corr: float = 0.25,
    smooth_s: float = 0.0,
    smooth_mode: str = "ma",
    decimate_to: Optional[float] = None,
    reference: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Stream, Dict[str, float]]:
    """
    Reference-based alignment using envelope delays to/from a chosen reference.
    """
    _log(f"[align_waveforms] BEGIN :: { _fmt_stats(st) }", verbose)
    env_st = envelopes_stream(st, smooth_s=smooth_s, smooth_mode=smooth_mode,
                              decimate_to=decimate_to, verbose=verbose)

    #env_st.plot(equal_scale=False);
    delays = envelope_delays(env_st, max_lag_s=max_lag_s, min_corr=min_corr, verbose=verbose)
    stations = sorted({tr.stats.station for tr in st})

    if reference is None and env_st:
        reference = pick_reference_station(env_st)
    _log(f"[align_waveforms] reference={reference}", verbose)

    lags = compute_station_lags(delays, reference, verbose=verbose)
    out = Stream()
    for tr in st:
        out += _shift_starttime(tr, lags.get(tr.stats.station, 0.0))

    _log("[align_waveforms] DONE", verbose)
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
    verbose: bool = False,
) -> Tuple[Stream, Dict[str, float], Dict[Tuple[str, str], Tuple[float, float]]]:
    """
    Global alignment: envelopes → pairwise delays → least-squares lags → shift waveforms.
    Returns (aligned Stream, {station: lag_s}, pairwise delays).
    """
    _log(f"[align_waveforms_global] BEGIN :: { _fmt_stats(st) }", verbose)
    env = envelopes_stream(st, smooth_s=smooth_s, smooth_mode=smooth_mode,
                           decimate_to=decimate_to, verbose=verbose)
    delays = envelope_delays(env, max_lag_s=max_lag_s, min_corr=min_corr, verbose=verbose)
    stations = sorted({tr.stats.station for tr in st})

    if ref_station is None and len(stations) > 0:
        ref_station = max(st, key=lambda tr: tr.stats.npts).stats.station
    _log(f"[align_waveforms_global] ref_station={ref_station}", verbose)

    lags = solve_global_station_lags(delays, stations, ref_station=ref_station,
                                     weight_power=weight_power, verbose=verbose)
    out = Stream()
    for tr in st:
        out += _shift_starttime(tr, lags.get(tr.stats.station, 0.0))

    _log("[align_waveforms_global] DONE", verbose)
    return out, lags, delays


# --------------------------------------------------------------------------- #
#                       Grid + speed locator (with debug)                     #
# --------------------------------------------------------------------------- #

def _resolve_dome_node(grid, dome_location: Union[int, Tuple[float, float], None]) -> Optional[int]:
    if dome_location is None:
        return None
    if isinstance(dome_location, int):
        return dome_location
    if isinstance(dome_location, (tuple, list)) and len(dome_location) == 3:
        lon, lat, elev = dome_location
        if hasattr(grid, "nearest_node"):
            try:
                return int(grid.nearest_node(lon, lat))
            except Exception:
                pass
        try:
            glon = np.asarray(grid.gridlon).ravel()
            glat = np.asarray(grid.gridlat).ravel()
            k = np.argmin((glon - lon) ** 2 + (glat - lat) ** 2)
            return int(k)
        except Exception:
            return None
    return None



def _station_key(id_or_sta: str, mode: str = "sta") -> str:
    """
    Map a station identifier to a matching key:
      - "sta":       MBFL
      - "net.sta":   MV.MBFL
      - "full":      MV.MBFL..EHZ (no change)
    """
    if mode == "full":
        return id_or_sta
    parts = id_or_sta.split(".")
    if len(parts) >= 2:
        net, sta = parts[0], parts[1]
    else:
        net, sta = "", id_or_sta
    if mode == "sta":
        return sta
    if mode == "net.sta":
        return f"{net}.{sta}" if net else sta
    raise ValueError(f"Unknown station_key_mode={mode!r}")


def _collapse_distances_by_station(
    node_distances_km: Dict[str, np.ndarray],
    *,
    station_key_mode: str = "sta",
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Collapse per-channel keys in distances to a single entry per station key.
    Preference order: any key whose channel endswith('Z'); else first seen.
    """
    collapsed: Dict[str, np.ndarray] = {}
    chosen_id: Dict[str, str] = {}

    for full_id, vec in node_distances_km.items():
        key = _station_key(full_id, station_key_mode)
        is_z = full_id.endswith("Z")
        if key not in collapsed:
            collapsed[key] = vec
            chosen_id[key] = full_id
        else:
            if is_z and not chosen_id[key].endswith("Z"):
                collapsed[key] = vec
                chosen_id[key] = full_id

    if verbose:
        print(f"[locate] collapsed distances: {len(node_distances_km)} -> {len(collapsed)} (key='{station_key_mode}')")
        for k, fid in list(chosen_id.items())[:8]:
            print(f"    {k:10s} <= {fid}")

    return collapsed


def locate_with_grid_from_delays(
    grid,
    cache_dir: str,
    *,
    inventory: Inventory,
    delays: Dict[Tuple[str, str], Tuple[float, float]],
    stream: Optional[Stream] = None,
    # scan config
    c_range: Tuple[float, float] = (0.3, 3.5),
    n_c: int = 61,
    c_phys: Optional[Tuple[float, float]] = (0.3, 3.5),  # clamp to plausible band
    # weighting / modes
    corr_power: float = 1.0,
    robust: str = "mad",                 # "mad" | "rms"
    eps_inlier_s: float = 0.40,          # Stage-A inlier gate |tau - (a + Δd/c)| <= eps
    # reference/keys
    station_key_mode: str = "sta",       # collapse id → station
    ref_station: Optional[str] = None,   # None → auto (highest degree in delay graph)
    dome_location: Optional[Union[int, Tuple[float, float], Dict[str, float]]] = None,
    auto_ref: bool = True,               # try to choose a good ref automatically
    # degeneracy guards
    min_delta_d_km: float = 0.7,         # drop tiny baselines in Stage-A
    min_abs_lag_s: float = 0.20,         # drop near-zero τ in Stage-A
    delta_d_weight: bool = True,         # weight Stage-A pairs by Δd
    # diagnostics / output
    debug: bool = True,
    verbose: bool = False,
    ccfs: Optional[Dict[Tuple[str, str], Dict[str, np.ndarray]]] = None,
    plot_ccf_dir: Optional[str] = None,
    plot_score_vs_c: Optional[str] = None,   # PNG: Stage-A score curve & Δd–τ panel
    topo_map_out: Optional[str] = None,      # PNG: topo map with best node
    topo_dem_path: Optional[str] = None,
    progress_every: int = 5,
    dump_debug_json: Optional[str] = None,
) -> Dict[str, object]:
    """
    Two-stage locator using envelope delays.

    Stage A (reference mode):
      • Choose a reference station (auto or provided).
      • At the dome node (or first usable node), fit τ ≈ a + Δd/c by weighted LS,
        then scan c on a grid using an inlier-first selection:
          - maximize inlier count (|τ−(a+Δd/c)| ≤ eps_inlier_s),
          - break ties by minimizing robust misfit (MAD or RMS).
      • Produce a diagnostic figure: score vs c (+ optional #inliers), and Δd–τ
        with the fitted line τ = a_best + Δd / c_best.

    Stage B:
      • Fix c_best from Stage A.
      • Scan all usable nodes; for each node, re-estimate intercept 'a' (weighted
        median residual) and compute robust misfit. Pick best node.

    Returns a dict with best node, speed, score, lon/lat, and rich debug info.
    """
    import json, os
    import matplotlib.pyplot as plt
    from flovopy.asl.distances import compute_or_load_distances
    # Optional: fallbacks if helpers are missing in your module
    # def _station_key(seedid, mode="sta"):
    #     try:
    #         parts = str(seedid).split(".")
    #         return parts[1] if mode == "sta" and len(parts) >= 2 else str(seedid)
    #     except Exception:
    #         return str(seedid)
    # def _collapse_distances_by_station(dists_dict, station_key_mode="sta", verbose=False):
    #     out = {}
    #     for seedid, vec in dists_dict.items():
        #     key = _station_key(seedid, station_key_mode)
        #     if key not in out: out[key] = vec
    #     if verbose:
    #         print(f"[locate] collapsed distances: {len(dists_dict)} -> {len(out)} (key='{station_key_mode}')")
    #     return out

    _log("[locate] BEGIN locate_with_grid_from_delays()", verbose)
    _log(f"Using distances cache dir: {cache_dir}", verbose)

    # --- Load distances (dense) and collapse IDs to station keys ---
    node_distances_km, coords_by_seed, meta = compute_or_load_distances(
        grid, cache_dir=cache_dir, inventory=inventory, stream=stream, use_elevation=True
    )
    dist_collapsed = _collapse_distances_by_station(node_distances_km, station_key_mode=station_key_mode, verbose=verbose)
    stations = sorted(dist_collapsed.keys())
    if not stations:
        return {"ok": False, "reason": "no_stations"}

    S = len(stations)
    any_vec = next(iter(dist_collapsed.values()))
    n_nodes = int(np.asarray(any_vec).size)
    D = np.vstack([dist_collapsed[s] for s in stations])  # (S, n_nodes)
    row_idx = {s: i for i, s in enumerate(stations)}
    _log(f"[locate] stations={stations}", verbose)
    _log(f"[locate] grid nodes={n_nodes} | D shape={D.shape}", verbose)

    # --- Re-key delays to station keys (take max-corr if duplicates) ---
    def rekey_delays(delays_in):
        out: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for (si, sj), (lag, corr) in delays_in.items():
            ki = _station_key(si, station_key_mode)
            kj = _station_key(sj, station_key_mode)
            if ki == kj:
                continue
            pair = (ki, kj) if ki < kj else (kj, ki)
            if pair not in out or corr > out[pair][1]:
                out[pair] = (float(lag), float(corr))
        return out

    delays_rekeyed = rekey_delays(delays)

    # --- Choose reference (highest degree in delay graph or provided) ---
    if ref_station is not None:
        ref = _station_key(ref_station, station_key_mode)
        if ref not in row_idx:
            return {"ok": False, "reason": f"ref_station '{ref}' not in distances"}
    else:
        # degree = number of pairs touching each station
        deg = {s: 0 for s in stations}
        for (si, sj) in delays_rekeyed:
            if si in deg: deg[si] += 1
            if sj in deg: deg[sj] += 1
        ref = max(deg, key=deg.get) if auto_ref and deg else stations[0]
    _log(f"[locate] reference={ref}", verbose)

    # --- Build reference-only τ with correct polarity ---
    def build_ref_delays(ref_key):
        tau_ref, w_ref = {}, {}
        for (si, sj), (lag, corr) in delays_rekeyed.items():
            # Our envelope lag convention from envelope_delays(): lag is (si,sj) such that sj arrives later by +lag
            if si == ref_key and sj in row_idx:
                tau_ref[sj] = +lag
                w_ref[sj]   = max(corr, 1e-6) ** corr_power
            elif sj == ref_key and si in row_idx:
                tau_ref[si] = -lag
                w_ref[si]   = max(corr, 1e-6) ** corr_power
        return tau_ref, w_ref

    tau_ref, w_ref = build_ref_delays(ref)
    ref_stations_all = sorted(tau_ref.keys())
    if len(ref_stations_all) < 2:
        return {"ok": False, "reason": "not_enough_ref_pairs", "ref": ref}

    # --- Node usability for ref subset ---
    ref_idx = sorted({row_idx[ref]} | {row_idx[s] for s in ref_stations_all})
    node_ok_ref = np.all(np.isfinite(D[ref_idx, :]), axis=0)
    usable_nodes = np.nonzero(node_ok_ref)[0]
    _log(f"[locate] ref subset stations={ref_stations_all} | usable nodes for ref subset: {usable_nodes.size}/{n_nodes}", verbose)
    if usable_nodes.size == 0:
        return {"ok": False, "reason": "no_usable_nodes_for_ref_subset"}

    # --- Resolve dome node (index) if given; else pick first usable ---
    dome_node = _resolve_dome_node(grid, dome_location)
    if dome_node is not None and (dome_node < 0 or dome_node >= n_nodes or not node_ok_ref[dome_node]):
        _log(f"[locate] WARNING: dome node unusable; falling back to trial node", verbose)
        dome_node = None
    stageA_node = dome_node if dome_node is not None else int(usable_nodes[0])
    _log(f"[locate] Stage A node={stageA_node} ({'dome' if dome_node is not None else 'trial'})", verbose)

    # --- Stage A helpers ---
    def _stageA_arrays(node_idx: int):
        """Build Δd, τ, w, label for Stage A with filters."""
        dd, tt, ww, labels = [], [], [], []
        d = D[:, node_idx]
        d_ref = d[row_idx[ref]]
        for s in ref_stations_all:
            tau = tau_ref.get(s)
            if tau is None or not np.isfinite(tau):
                continue
            Δd = float(d[row_idx[s]] - d_ref)
            if abs(Δd) < min_delta_d_km:   # drop tiny baselines
                continue
            if abs(tau) < min_abs_lag_s:   # drop near-zero lags
                continue
            w = float(w_ref.get(s, 1.0))
            if delta_d_weight:
                w *= max(abs(Δd), 1e-6)    # informative baselines carry more weight
            dd.append(Δd); tt.append(float(tau)); ww.append(w); labels.append(s)
        return np.asarray(dd, float), np.asarray(tt, float), np.asarray(ww, float), labels

    def _fit_ls_with_intercept(dd, tt, ww):
        """Weighted LS: τ ≈ a + m*Δd. Return (c_hat, a_hat, m_hat)."""
        if dd.size < 2 or np.var(dd) <= 0:
            return (np.nan, 0.0, np.nan)
        X = np.column_stack([dd, np.ones_like(dd)])
        W = np.sqrt(ww)[:, None]
        beta, *_ = np.linalg.lstsq(W * X, W[:, 0] * tt, rcond=None)
        m, a = float(beta[0]), float(beta[1])
        c = float(1.0 / m) if m > 0 else np.nan
        if c_phys and np.isfinite(c):
            c = float(np.clip(c, c_phys[0], c_phys[1]))
        return c, a, m

    def _robust_score(resid_w):
        if resid_w.size == 0:
            return np.inf
        if robust == "mad":
            mad = 1.4826 * np.median(np.abs(resid_w))
            return float(mad if np.isfinite(mad) else np.inf)
        return float(np.sqrt(np.mean(resid_w ** 2)))

    def _stageA_score(dd, tt, ww, c, a_hat):
        pred = a_hat + dd / c
        resid = tt - pred
        inliers = np.isfinite(resid) & (np.abs(resid) <= eps_inlier_s)
        nin = int(np.count_nonzero(inliers))
        if nin == 0:
            return np.inf, nin, inliers
        resid_w = np.sqrt(ww[inliers]) * resid[inliers]
        return _robust_score(resid_w), nin, inliers

    # --- Stage A data at chosen node ---
    dd0, tt0, ww0, lab0 = _stageA_arrays(stageA_node)
    if dd0.size < 2:
        return {"ok": False, "reason": "stageA_too_few_pairs_after_filters",
                "n_pairs_kept": int(dd0.size)}

    # LS seed with intercept
    c_ls, a_ls, m_ls = _fit_ls_with_intercept(dd0, tt0, ww0)

    # Build c grid (refined around c_ls if available)
    c_lo, c_hi = float(c_range[0]), float(c_range[1])
    speeds = np.linspace(c_lo, c_hi, int(n_c), dtype=float)
    if np.isfinite(c_ls):
        half = max(0.40 * c_ls, 0.30)  # ±40% window, at least ±0.3 km/s
        lo2, hi2 = max(c_lo, c_ls - half), min(c_hi, c_ls + half)
        speeds = np.linspace(lo2, hi2, int(n_c), dtype=float)
    _log(f"[locate] Stage A c-grid: [{speeds[0]:.3f} … {speeds[-1]:.3f}], n={speeds.size}", verbose)

    # Scan c with inlier-first selection; re-estimate a_hat per c as weighted median of (τ - Δd/c)
    scores_c = []
    best_idx = None
    best_nin = -1
    best_score = np.inf
    best_inliers = None
    best_a = 0.0

    # weighted median helper
    def _wmedian(x, w):
        if x.size == 0:
            return 0.0
        idx = np.argsort(x)
        xs, ws = x[idx], w[idx]
        p = np.cumsum(ws) / np.sum(ws)
        j = int(np.searchsorted(p, 0.5))
        return float(xs[min(max(j, 0), xs.size - 1)])

    for ic, c in enumerate(speeds):
        if ic % max(1, int(progress_every)) == 0:
            _log(f"[locate]  c={c:.3f} ({ic+1}/{len(speeds)})", verbose)
        if not np.isfinite(c) or c <= 0:
            scores_c.append({"c": float(c), "score": float("inf"), "nin": 0, "a": 0.0})
            continue
        r0 = tt0 - (dd0 / c)
        a_hat = _wmedian(r0, ww0)  # robust intercept
        sc, nin, inliers = _stageA_score(dd0, tt0, ww0, c, a_hat)
        scores_c.append({"c": float(c), "score": float(sc), "nin": int(nin), "a": float(a_hat)})
        if (nin > best_nin) or (nin == best_nin and sc < best_score):
            best_idx, best_nin, best_score = ic, nin, sc
            best_inliers, best_a = inliers, a_hat

    c_best = float(scores_c[best_idx]["c"])
    a_best = float(scores_c[best_idx]["a"])
    _log(f"[locate] Stage A: c_best≈{c_best:.3f} km/s (nin={best_nin}, score={best_score:.3f})", verbose)

    # --- Diagnostic PNG: left (score vs c, +inliers), right (Δd vs τ, used/ignored, model line) ---
    if plot_score_vs_c:
        try:
            fig = plt.figure(figsize=(11, 4))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot([e["c"] for e in scores_c], [e["score"] for e in scores_c], label="Stage-A score vs c (stageA node)")
            ax1.axvline(c_best, ls="--", label=f"best c≈{c_best:.3f}")
            if np.isfinite(c_ls):
                ax1.axvline(c_ls, ls=":", label=f"LS c≈{c_ls:.3f}")
            ax1.set_xlabel("Apparent speed c (km/s)")
            ax1.set_ylabel("Robust misfit (normalized)")
            ax1.grid(True, alpha=0.3)
            ax1b = ax1.twinx()
            ax1b.plot([e["c"] for e in scores_c], [e["nin"] for e in scores_c], linestyle=":", alpha=0.35, label="#inliers")
            # tidy legend
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1b.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")

            ax2 = fig.add_subplot(1, 2, 2)
            used = best_inliers
            ign  = ~used
            ax2.scatter(dd0[ign],  tt0[ign],  marker="x", label="ignored")
            ax2.scatter(dd0[used], tt0[used], label="used")
            xline = np.linspace(dd0.min()*1.1, dd0.max()*1.1, 200)
            ax2.plot(xline, a_best + xline / c_best, label="model @ c_best")
            ax2.set_xlabel("Δd (km)")
            ax2.set_ylabel("τ (s)")
            ax2.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_score_vs_c, dpi=160)
            plt.close(fig)
            _log(f"[locate] score-vs-c written -> {plot_score_vs_c}", verbose)
        except Exception as e:
            _log(f"[locate] WARN: score-vs-c plot failed: {e}", verbose)

    # --- Stage B: grid scan with c_best; re-estimate intercept 'a' per node, score robustly ---
    best = dict(score=np.inf)
    for g in np.nonzero(node_ok_ref)[0]:
        d = D[:, g]
        dd = np.asarray([d[row_idx[s]] - d[row_idx[ref]] for s in ref_stations_all], float)
        tt = np.asarray([tau_ref[s] for s in ref_stations_all], float)
        # filters consistent with Stage-A
        m = (np.abs(dd) >= min_delta_d_km)
        dd, tt = dd[m], tt[m]
        if dd.size == 0:
            continue
        # robust intercept at this node:
        r0 = tt - dd / c_best
        # simple weights (reuse Stage-A’s corr weights; Δd-weight optional)
        ww = np.asarray([w_ref[s] for s in np.array(ref_stations_all)[m]], float)
        if delta_d_weight:
            ww *= np.maximum(np.abs(dd), 1e-6)
        a_hat = _wmedian(r0, ww)
        resid = (tt - (a_hat + dd / c_best))
        resid_w = np.sqrt(ww) * resid
        sc = _robust_score(resid_w)
        if np.isfinite(sc) and sc < best.get("score", np.inf):
            best = dict(
                ok=True,
                node=int(g),
                speed=float(c_best),
                intercept=float(a_hat),
                score=float(sc),
                n_pairs=int(dd.size),
            )

    if not best.get("ok"):
        return {"ok": False, "reason": "no_finite_scores_in_stageB", "c_best": c_best}

    # --- Attach lon/lat/elev for best node ---
    try:
        lon, lat, elev = grid.node_lonlat(g=best["node"])
        best["lon"], best["lat"], best["elev_m"] = float(lon), float(lat), float(elev if elev is not None else np.nan)
    except Exception:
        try:
            # fallback: nearest from flat arrays
            glon = np.asarray(grid.gridlon).ravel()
            glat = np.asarray(grid.gridlat).ravel()
            best["lon"], best["lat"], best["elev_m"] = float(glon[best["node"]]), float(glat[best["node"]]), float("nan")
        except Exception:
            pass

    # --- Optional topo map with best node ---
    if topo_map_out:
        try:
            from flovopy.asl.map import topo_map
            title = "Best node (red dot) & stations"
            fig = topo_map(
                show=False, inv=inventory, title=title,
                dem_tif=topo_dem_path, outfile=None, return_region=False,
            )
            # add best node as a red circle
            if hasattr(fig, "plot"):
                fig.plot(x=[best["lon"]], y=[best["lat"]], style="c0.22c", fill="red", pen="1p,black")
            fig.savefig(topo_map_out, dpi=220)
            _log(f"[locate] topo map written -> {topo_map_out}", verbose)
        except Exception as e:
            _log(f"[locate] WARN: topo_map failed ({e})", verbose)

    # --- Per-pair CCF plots (if provided) ---
    if plot_ccf_dir and ccfs:
        import os
        os.makedirs(plot_ccf_dir, exist_ok=True)
        for (si, sj), dct in ccfs.items():
            try:
                fig = plt.figure(figsize=(6, 3)); ax = fig.add_subplot(111)
                ax.plot(dct["lags"], dct["r"], lw=1)
                ax.axvline(dct["peak_lag"], ls="--")
                ax.set_title(f"CCF: {si}–{sj}  lag={dct['peak_lag']:.2f}s  corr={dct['peak_corr']:.2f}")
                ax.set_xlabel("Lag (s)"); ax.set_ylabel("Norm. corr"); fig.tight_layout()
                fig.savefig(os.path.join(plot_ccf_dir, f"ccf_{si}_{sj}.png"), dpi=140); plt.close(fig)
            except Exception:
                pass

    # --- Debug payload ---
    if debug:
        dbg = dict(
            mode="reference",
            stations=stations,
            ref=ref,
            ref_stations_all=ref_stations_all,
            stageA_node=int(stageA_node),
            c_ls=float(c_ls) if np.isfinite(c_ls) else None,
            c_best=float(c_best),
            a_best=float(a_best),
            scores_c=scores_c,               # list of {c, score, nin, a}
            best=best.copy(),
        )
        best["debug_info"] = dbg

    # Optional JSON dump
    if dump_debug_json:
        try:
            with open(dump_debug_json, "w") as f:
                json.dump(best, f, indent=2, default=float)
        except Exception:
            pass

    _log("[locate] DONE (reference mode)", verbose)
    return best


import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Union
from obspy import Inventory
from flovopy.asl.distances import compute_or_load_distances

def _station_key(seed_id: str, mode: str = "sta") -> str:
    """
    Map 'NET.STA.LOC.CHA' or 'STA' → station key used by your delays/stats.
    mode='sta' keeps only STA.
    """
    if mode != "sta":
        raise ValueError("Only station_key_mode='sta' is supported here.")
    parts = str(seed_id).split(".")
    return parts[1] if len(parts) >= 2 else parts[0]

def _resolve_dome_node(grid, dome_location: Union[int, Tuple[float,float], Dict[str,float]]):
    """
    Accepts: node index, (lon,lat), or dict {'lon':..., 'lat':...}.
    Returns node index (int) or None.
    """
    if dome_location is None:
        return None
    if isinstance(dome_location, int):
        return dome_location
    if isinstance(dome_location, dict):
        lon = float(dome_location.get("lon"))
        lat = float(dome_location.get("lat"))
    elif isinstance(dome_location, (tuple, list)) and len(dome_location) == 2:
        lon, lat = float(dome_location[0]), float(dome_location[1])
    else:
        return None
    # prefer grid helper if available
    if hasattr(grid, "nearest_node"):
        try:
            return int(grid.nearest_node(lon, lat))
        except Exception:
            pass
    # fall back to brute force
    try:
        glon = np.asarray(grid.gridlon).ravel()
        glat = np.asarray(grid.gridlat).ravel()
        k = int(np.argmin((glon - lon)**2 + (glat - lat)**2))
        return k
    except Exception:
        return None

def estimate_speed_source_at_dome_from_stable_pairs(
    *,
    grid,
    cache_dir: str,
    inventory: Inventory,
    stable_pairs_csv: str,              # e.g. pairwise_lagdiff_stats_global_stable.csv
    dome_location: Union[int, Tuple[float,float], Dict[str,float]],
    station_key_mode: str = "sta",
    use_stat: str = "mean_kept",        # Δτ statistic: "mean_kept" or "median"
    uncert_col: str = "std_kept",       # per-pair scatter for weights: "std_kept" or "mad_scaled"
    min_delta_d_km: float = 0.2,        # drop nearly-coincident station pairs
    min_abs_tau_s: float = 0.05,        # drop near-zero lag diffs to avoid blow-ups
    use_elevation: bool = True,         # 3-D path lengths if available
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Source fixed at dome. For each stable pair (a,b), compute c_ab = (d_a - d_b)/Δτ_ab,
    then robustly combine to a single speed estimate.

    Returns dict with:
      - 'speed_median_km_s', 'speed_mad_km_s', 'speed_weighted_mean_km_s'
      - 'pairs_used' (count), 'pairs_dropped' (reasons)
      - 'per_pair' (DataFrame with Δd, Δτ, c_ab, weights, etc.)
    """
    df = pd.read_csv(stable_pairs_csv)
    if df.empty:
        raise ValueError("Stable-pairs CSV is empty.")

    # sanity
    for col in (use_stat,):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {stable_pairs_csv}")

    if uncert_col and uncert_col not in df.columns:
        # allow missing col: treat as equal weights
        uncert_col = None

    # Load/compute node→station distances (dense arrays)
    node_distances_km, coords_by_sta, meta = compute_or_load_distances(
        grid,
        cache_dir=cache_dir,
        inventory=inventory,
        stream=None,
        use_elevation=use_elevation,
    )

    # Collapse multi-channel IDs to station keys and pick one representative per station
    # Build a dict: {STA: distances_vector}
    dist_by_station: Dict[str, np.ndarray] = {}
    for seed_id, vec in node_distances_km.items():
        key = _station_key(seed_id, station_key_mode)
        # pick first occurrence (or you could prefer 'HHZ' etc. if you want)
        if key not in dist_by_station:
            dist_by_station[key] = np.asarray(vec, float)

    stations = sorted(dist_by_station.keys())
    n_nodes = next(iter(dist_by_station.values())).size

    # Resolve dome node index
    dome_node = _resolve_dome_node(grid, dome_location)
    if dome_node is None or dome_node < 0 or dome_node >= n_nodes:
        raise ValueError("Could not resolve a valid dome node index from dome_location.")

    # Extract distances from dome to each station
    d_sta: Dict[str, float] = {}
    for sta in stations:
        dvec = dist_by_station[sta]
        val = float(dvec[dome_node]) if np.isfinite(dvec[dome_node]) else np.nan
        d_sta[sta] = val

    # Parse "pair" -> sta1, sta2
    def _split_pair(p: str) -> Tuple[str, str]:
        a, b = str(p).split("-")
        return a.strip(), b.strip()

    rows = []
    dropped = {"missing_distance": 0, "small_delta_d": 0, "small_abs_tau": 0, "nan": 0}
    for _, r in df.iterrows():
        pair = str(r["pair"])
        a, b = _split_pair(pair)
        if a not in d_sta or b not in d_sta:
            dropped["missing_distance"] += 1
            continue
        da, db = d_sta[a], d_sta[b]
        if not (np.isfinite(da) and np.isfinite(db)):
            dropped["missing_distance"] += 1
            continue
        delta_d = da - db  # km

        tau = float(r[use_stat])  # s
        if not np.isfinite(tau):
            dropped["nan"] += 1
            continue

        if abs(delta_d) < float(min_delta_d_km):
            dropped["small_delta_d"] += 1
            continue
        if abs(tau) < float(min_abs_tau_s):
            dropped["small_abs_tau"] += 1
            continue

        c_ab = delta_d / tau  # km/s (sign handled by the consistent Δτ definition)
        # weight: (|Δd| / σ_τ)^2 to reward geometric leverage and penalize noisy pairs
        if uncert_col:
            sig = float(r[uncert_col]) if np.isfinite(r[uncert_col]) else np.nan
            if not np.isfinite(sig) or sig <= 0:
                w = (abs(delta_d) / 0.5)**2  # fallback σ≈0.5 s
            else:
                w = (abs(delta_d) / sig)**2
        else:
            w = abs(delta_d)  # simple leverage weight if no uncertainty

        rows.append({
            "pair": pair, "sta_a": a, "sta_b": b,
            "delta_d_km": delta_d, "delta_tau_s": tau,
            "c_pair_km_s": c_ab, "weight": float(w),
        })

    if not rows:
        raise ValueError("No usable stable pairs after guards (Δd and Δτ too small or distances missing).")

    per_pair = pd.DataFrame(rows)

    # Robust center & spread (median and MAD)
    c_vals = per_pair["c_pair_km_s"].values.astype(float)
    c_med = float(np.median(c_vals))
    mad = float(1.4826 * np.median(np.abs(c_vals - c_med)))  # ~1σ equiv if Gaussian

    # Weighted mean (cap extreme weights)
    w = per_pair["weight"].values.astype(float)
    w = np.clip(w, np.nanpercentile(w, 5), np.nanpercentile(w, 95))
    c_wmean = float(np.sum(w * c_vals) / np.sum(w))

    out = {
        "speed_median_km_s": c_med,
        "speed_mad_km_s": mad,
        "speed_weighted_mean_km_s": c_wmean,
        "pairs_used": int(per_pair.shape[0]),
        "pairs_dropped": dropped,
        "per_pair": per_pair.sort_values("c_pair_km_s").reset_index(drop=True),
        "dome_node": int(dome_node),
    }
    if verbose:
        print(f"Stable pairs used: {out['pairs_used']}  | dropped: {dropped}")
        print(f"Speed (median ± MAD): {c_med:.3f} ± {mad:.3f} km/s")
        print(f"Speed (weighted mean): {c_wmean:.3f} km/s")
    return out