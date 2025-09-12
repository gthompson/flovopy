"""
corrkit_eq: correlation & clustering toolkit built on EQcorrscan + ObsPy.

Hard dependency: eqcorrscan

Install:
    conda install -c conda-forge obspy scipy matplotlib eqcorrscan

Features
--------
- Pairwise normalized cross-correlation matrix (max over +/-lag window).
- Uses EQcorrscan's correlation kernel(s).
- Optional station-level multichannel combine (mean/sum) before CC.
- Hierarchical clustering (SciPy) + dendrogram.
- Cross-correlogram heatmap (optionally reordered by dendrogram leaves).
- Families vs CC threshold curve.

Examples
--------
    # Basic correlogram + clustering from MiniSEED files:
    python corrkit_eq.py "data/*.mseed" --bp 1 8 --resample-hz 100 --plot --save-prefix out/cc

    # You’ll get:
	•	Cross-correlogram (ordered) heatmap (reordered by dendrogram leaves)
	•	Hierarchical clustering dendrogram (average linkage on D=1-CC)
	•	Families vs CC threshold curve

    python corrkit_eq.py "LP_MBB_SHZ/*.mseed" \
        --bp 0.8 3.5 --resample-hz 100 --max-shift 50 --plot --save-prefix out/lp_auto

Notes & quick tweaks
	•	Where EQcorrscan plugs in: _resolve_eq_xcorr() tries core.match_filter.normxcorr2 first, then utils.correlate.xcorr. If your build exposes a different name/location, update that small function once and everything else just works.
	•	Max-lag vs zero-lag: pass --zero-lag to match a strict same-phase comparison; otherwise we seek the max CC within ±max_shift samples.
	•	Multichannel stations: --combine-by station collapses channels per station via mean (or sum) before comparisons; that often stabilizes families.
	•	Speed: For large N, we can trivially parallelize the i<j loop (joblib/concurrent.futures). Tell me when you want this and I’ll wire it in.

⸻


Author: Glenn Thompson
License: MIT


New in this version
-------------------
- --auto-win <sec>: per-trace auto-windowing using envelope energy (no picks needed)
- --smooth-ms <ms>, --edge-guard <sec>: control auto-window behavior
- --align-ref <idx>: align all traces to a reference (by max CC), then you can use --zero-lag
- --align-max-shift <samples>: max shift (±) allowed during alignment
- --save-npz <path>: save CC matrix, labels, and linkage to an .npz file

# One SEED ID, LP events only, auto-window 12 s, bandpass 0.8–3.5 Hz
python corrkit_eq.py "LP_MBB_SHZ/*.mseed" \
  --auto-win 12 --bp 0.8 3.5 --resample-hz 100 \
  --max-shift 50 --plot --save-prefix out/shv_lp_mbb_shz --save-npz out/shv_lp_mbb_shz.npz

#  Auto-window, then align to the strongest trace, then zero-lag CC
#  Use this to squeeze out slightly tighter families once alignment is good
python corrkit_eq.py "LP_MBB_SHZ/*.mseed" \
  --auto-win 12 --bp 0.8 3.5 --resample-hz 100 \
  --align-ref 0 --align-max-shift 80 \
  --zero-lag --plot --save-prefix out/shv_lp_aligned

# Multiple stations, but combine per station (averaging channels)
python corrkit_eq.py "LP_multi/*.mseed" \
  --auto-win 12 --bp 0.8 3.5 --resample-hz 100 \
  --combine-by station --max-shift 50 --plot 

"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream, Trace
from obspy.signal.filter import bandpass
from scipy.signal import hilbert
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import squareform

import eqcorrscan  # required


# ---- resolve an EQcorrscan kernel for cross-correlation
def _resolve_eq_xcorr():
    try:
        from eqcorrscan.core.match_filter import normxcorr2  # type: ignore
        def _f(a, b, max_shift, use_max):
            a = np.asarray(a, float); b = np.asarray(b, float)
            sa = np.linalg.norm(a); sb = np.linalg.norm(b)
            if sa == 0 or sb == 0: return np.nan
            a /= sa; b /= sb
            cc = normxcorr2(a, b)  # full lags, zero-lag at idx=len(a)-1
            center = len(a) - 1
            if max_shift is None:
                return float(np.max(cc) if use_max else cc[center])
            lo = max(0, center - max_shift); hi = min(cc.size, center + max_shift + 1)
            return float(np.max(cc[lo:hi]) if use_max else cc[center])
        return _f
    except Exception:
        pass
    try:
        from eqcorrscan.utils.correlate import xcorr as _xcorr  # type: ignore
        def _f(a, b, max_shift, use_max):
            a = np.asarray(a, float); b = np.asarray(b, float)
            sa = np.linalg.norm(a); sb = np.linalg.norm(b)
            if sa == 0 or sb == 0: return np.nan
            a /= sa; b /= sb
            cc = _xcorr(a, b)
            center = len(a) - 1
            if max_shift is None:
                return float(np.max(cc) if use_max else cc[center])
            lo = max(0, center - max_shift); hi = min(cc.size, center + max_shift + 1)
            return float(np.max(cc[lo:hi]) if use_max else cc[center])
        return _f
    except Exception:
        pass
    raise ImportError(
        "Could not locate an EQcorrscan xcorr kernel. Edit _resolve_eq_xcorr() "
        "to import the correct function for your eqcorrscan version."
    )

_EQ_XCORR = _resolve_eq_xcorr()


# -----------------------
# Preprocessing & helpers
# -----------------------
def _prep_trace(
    tr: Trace,
    *,
    resample_hz: Optional[float] = None,
    demean: bool = True,
    taper: float = 0.05,
    bp: Tuple[float, float] | None = None,
) -> np.ndarray:
    x = tr.copy()
    if resample_hz and resample_hz > 0 and abs(x.stats.sampling_rate - resample_hz) > 1e-9:
        x.resample(resample_hz)
    if demean: x.detrend("demean")
    x.detrend("linear")
    if taper and taper > 0: x.taper(taper, type="hann")
    if bp:
        fmin, fmax = bp
        x.data = bandpass(x.data.astype(np.float64), fmin, fmax, x.stats.sampling_rate,
                          corners=4, zerophase=True)
    return np.asarray(x.data, float)

def _envelope(x: np.ndarray, smooth_samp: int = 25) -> np.ndarray:
    env = np.abs(hilbert(x))
    if smooth_samp > 1:
        k = np.ones(smooth_samp) / smooth_samp
        env = np.convolve(env, k, mode="same")
    return env

def auto_window_trace(
    tr: Trace,
    *,
    win_len: float,
    smooth_ms: float = 100.0,
    bp: Tuple[float, float] | None = None,
    resample_hz: float | None = None,
    demean: bool = True,
    taper: float = 0.05,
    edge_guard: float = 0.5,
) -> Trace:
    """Return a trimmed copy containing the loudest window (length win_len seconds)."""
    x = tr.copy()
    x.data = _prep_trace(x, resample_hz=resample_hz, demean=demean, taper=taper, bp=bp)
    fs = x.stats.sampling_rate
    W = max(1, int(round(win_len * fs)))
    guard = max(0, int(round(edge_guard * fs)))
    env = _envelope(x.data, smooth_samp=max(1, int(round(smooth_ms * 1e-3 * fs))))
    power = np.convolve(env, np.ones(W), mode="valid")
    lo = guard
    hi = max(lo, power.size - guard - 1)
    idx0 = lo + int(np.argmax(power[lo:hi+1]))
    i0, i1 = idx0, idx0 + W
    start = x.stats.starttime + i0 / fs
    end   = x.stats.starttime + i1 / fs
    x.trim(starttime=start, endtime=end, pad=True, fill_value=0)
    return x

def auto_window_stream(st: Stream, *, win_len: float, **kw) -> Stream:
    out = Stream()
    for tr in st:
        out += auto_window_trace(tr, win_len=win_len, **kw)
    return out

def align_by_reference(st: Stream, *, ref_index: int = 0, max_shift: int = 200) -> Stream:
    """
    Shift traces (sample-wise) to align to a reference by maximizing CC in ±max_shift.
    Uses an FFT cross-correlation internally for speed.
    """
    assert len(st) > 0
    ref = st[ref_index]
    fs = ref.stats.sampling_rate
    r = ref.data.astype(float)
    r /= (np.linalg.norm(r) or 1.0)

    out = Stream()
    for k, tr in enumerate(st):
        if k == ref_index:
            out += tr.copy()
            continue
        v = tr.data.astype(float)
        v /= (np.linalg.norm(v) or 1.0)
        L = int(2 ** np.ceil(np.log2(r.size + v.size - 1)))
        fr = np.fft.rfft(r, L)
        fv = np.fft.rfft(v, L)
        cc = np.fft.irfft(fr * np.conj(fv), L)
        cc = np.concatenate((cc[-(v.size - 1):], cc[:r.size]))  # center zero lag
        center = v.size - 1
        lo = max(0, center - max_shift); hi = min(cc.size, center + max_shift + 1)
        best = int(np.argmax(cc[lo:hi])) + lo - center  # samples (positive => v lags ref)
        shifted = tr.copy()
        if best > 0:
            shifted.data = np.r_[shifted.data[best:], np.zeros(best)]
        elif best < 0:
            s = -best
            shifted.data = np.r_[np.zeros(s), shifted.data[:-s]]
        out += shifted
    return out


# -----------------------
# Core API
# -----------------------
def compute_cc_matrix(
    stream: Stream | Sequence[Trace],
    *,
    max_shift: int = 200,
    use_max: bool = True,
    resample_hz: float | None = None,
    bp: Tuple[float, float] | None = None,
    demean: bool = True,
    taper: float = 0.05,
    combine_by: Optional[str] = None,   # None | "station" | "id"
    combine_func: str = "mean",         # "mean" | "sum"
    progress: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    if isinstance(stream, Stream):
        traces = list(stream.traces)
    else:
        traces = list(stream)

    def key_for(tr: Trace) -> str:
        if combine_by is None or combine_by == "id": return tr.id
        if combine_by == "station": return f"{tr.stats.network}.{tr.stats.station}"
        raise ValueError("combine_by must be None, 'station', or 'id'")

    buckets: dict[str, list[np.ndarray]] = {}
    for tr in traces:
        vec = _prep_trace(tr, resample_hz=resample_hz, demean=demean, taper=taper, bp=bp)
        buckets.setdefault(key_for(tr), []).append(vec)

    min_len = min(min(len(v) for v in vs) for vs in buckets.values())
    vectors: list[np.ndarray] = []; labels: list[str] = []
    for k, vs in buckets.items():
        vs = [v[:min_len] for v in vs]
        if combine_by is not None and len(vs) > 1:
            stack = np.vstack(vs)
            vec = np.nanmean(stack, axis=0) if combine_func == "mean" else np.nansum(stack, axis=0)
        else:
            vec = vs[0]
        labels.append(k); vectors.append(vec.astype(float))

    n = len(vectors)
    if progress:
        print(f"[corrkit_eq] N={n} vectors, L={min_len}, use_max={use_max}, max_shift={max_shift}")

    C = np.eye(n, dtype=float)
    for i in range(n):
        vi = vectors[i]
        for j in range(i + 1, n):
            vj = vectors[j]
            cij = _EQ_XCORR(vi, vj, max_shift=max_shift, use_max=use_max)
            C[i, j] = C[j, i] = cij
    return np.clip(C, -1.0, 1.0), labels


def cluster_cc(C: np.ndarray, *, method: str = "average") -> Tuple[np.ndarray, np.ndarray]:
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")
    Dfull = 1.0 - C
    np.fill_diagonal(Dfull, 0.0)
    D = squareform(Dfull, checks=False)
    Z = linkage(D, method=method)
    return Z, D

def families_vs_threshold(Z: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    out = []
    for tau in thresholds:
        labels = fcluster(Z, t=1.0 - float(tau), criterion="distance")
        out.append(len(np.unique(labels)))
    return np.asarray(out, int)


# -----------------------
# Plotting
# -----------------------
def plot_correlogram(
    C: np.ndarray,
    labels: Sequence[str] | None = None,
    Z: np.ndarray | None = None,
    *,
    reorder_by_dendrogram: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: str = "Cross-correlogram",
    figsize: Tuple[int, int] = (6, 5),
    show_colorbar: bool = True,
):
    Cplot = np.array(C, copy=True); order = None
    if reorder_by_dendrogram and Z is not None:
        order = leaves_list(Z)
        Cplot = Cplot[np.ix_(order, order)]
        if labels is not None:
            labels = [labels[i] for i in order]
    fig = plt.figure(figsize=figsize); ax = plt.gca()
    im = ax.imshow(Cplot, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    if labels is not None and len(labels) <= 60:
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    if show_colorbar:
        cb = fig.colorbar(im, ax=ax); cb.set_label("max CC" if vmax <= 1 else "similarity")
    fig.tight_layout()
    return Cplot, order

def plot_dendrogram(Z: np.ndarray, *, title: str = "Hierarchical clustering",
                    ylabel: str = "Distance = 1 - CC", figsize: Tuple[int, int] = (7, 4)):
    plt.figure(figsize=figsize)
    dendrogram(Z, no_labels=True, count_sort="ascending")
    plt.title(title); plt.ylabel(ylabel); plt.tight_layout()

def plot_families_curve(thresholds: Sequence[float], counts: Sequence[int],
                        *, title: str = "Families vs CC threshold"):
    plt.figure()
    plt.plot(thresholds, counts, marker="o")
    plt.xlabel("CC threshold"); plt.ylabel("# families"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()


# -----------------------
# CLI
# -----------------------
def _cli():
    import argparse
    from obspy import read

    p = argparse.ArgumentParser(
        prog="corrkit_eq",
        description="Cross-correlogram & clustering (EQcorrscan backend) with auto-window & alignment."
    )
    p.add_argument("inputs", nargs="+", help="Waveform files or globs (MiniSEED/SAC/etc.)")
    # Preprocessing
    p.add_argument("--resample-hz", type=float, default=None)
    p.add_argument("--bp", type=float, nargs=2, default=None, metavar=("FMIN","FMAX"))
    p.add_argument("--max-shift", type=int, default=200, help="+/- lag (samples) for CC")
    p.add_argument("--zero-lag", action="store_true", help="Use zero-lag CC (after alignment)")
    p.add_argument("--combine-by", choices=["station","id"], default=None,
                   help="Combine multichannel per key before CC")
    p.add_argument("--combine-func", choices=["mean","sum"], default="mean")
    # Auto-window (no picks needed)
    p.add_argument("--auto-win", type=float, default=None,
                   help="Auto-window each trace to this duration (seconds) using envelope energy")
    p.add_argument("--smooth-ms", type=float, default=100.0,
                   help="Envelope smoothing (ms) for auto-window")
    p.add_argument("--edge-guard", type=float, default=0.5,
                   help="Avoid picking within this many seconds of file edges")
    # Alignment to a reference
    p.add_argument("--align-ref", type=int, default=None,
                   help="Index of reference trace to align others to (0-based).")
    p.add_argument("--align-max-shift", type=int, default=80,
                   help="Max samples for alignment search (±).")
    # Clustering & I/O
    p.add_argument("--cluster-method", default="average",
                   choices=["single","complete","average","ward"])
    p.add_argument("--plot", action="store_true", help="Show plots")
    p.add_argument("--save-prefix", default=None, help="If set, save figures with this prefix")
    p.add_argument("--save-npz", default=None, help="Save C, labels, Z to this .npz file")
    p.add_argument("--thresholds", type=float, nargs=3, default=[0.6, 0.98, 20],
                   metavar=("START","END","N"), help="CC thresholds sweep")
    args = p.parse_args()

    # Read all inputs
    st = Stream()
    for pat in args.inputs:
        st += read(pat)

    # Auto-window (before everything else) if requested
    if args.auto_win:
        st = auto_window_stream(
            st,
            win_len=float(args.auto_win),
            smooth_ms=float(args.smooth_ms),
            bp=tuple(args.bp) if args.bp else None,
            resample_hz=args.resample_hz,
            edge_guard=float(args.edge_guard),
        )
        # After auto-window we’ve already filtered/resampled; turn off here to avoid double-doing.
        bp_for_cc = None
        resample_for_cc = None
    else:
        bp_for_cc = tuple(args.bp) if args.bp else None
        resample_for_cc = args.resample_hz

    # Optional alignment to a reference (on the auto-windowed or raw preprocessed data)
    if args.align_ref is not None:
        st = align_by_reference(st, ref_index=int(args.align_ref),
                                max_shift=int(args.align_max_shift))

    # Compute CC matrix
    C, labels = compute_cc_matrix(
        st,
        max_shift=args.max_shift,
        use_max=(not args.zero_lag),
        resample_hz=resample_for_cc,
        bp=bp_for_cc,
        combine_by=args.combine_by,
        combine_func=args.combine_func,
    )

    # Cluster
    Z, _ = cluster_cc(C, method=args.cluster_method)

    # Families curve
    t0, t1, npts = args.thresholds
    taus = np.linspace(float(t0), float(t1), int(npts))
    counts = families_vs_threshold(Z, taus)

    # Plot/save
    if args.plot or args.save_prefix:
        plot_correlogram(C, labels=labels, Z=Z, title="Cross-correlogram (ordered)")
        plot_dendrogram(Z)
        plot_families_curve(taus, counts)

    if args.save_prefix:
        for i, fig in enumerate(plt.get_fignums(), start=1):
            plt.figure(fig); plt.savefig(f"{args.save_prefix}_{i:02d}.png", dpi=200)

    if args.save_npz:
        np.savez(args.save_npz, C=C, labels=np.array(labels, dtype=object), Z=Z)

    if args.plot:
        plt.show()


if __name__ == "__main__":
    _cli()