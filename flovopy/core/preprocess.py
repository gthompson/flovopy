# ===============================
# flovopy/core/preprocessing.py
# Gap-aware normalization + safe cleaning wrappers (domain-agnostic)
# Notes:
# - Use tr.stats.processing for breadcrumbs (ObsPy convention).
# - No rocket-specific logic here.
# - Filtering/response helpers live in flovopy.core.remove_response.
# ===============================
from __future__ import annotations

from typing import Optional, Tuple, Union, Literal
import numpy as np
from obspy import Stream, Trace

from flovopy.core.gaputils import normalize_stream_gaps
from flovopy.core.trace_utils import add_processing_step
from flovopy.core.remove_response import safe_pad_taper_filter, stationxml_match_report

# -------------------------------
# QC metrics & utilities
# -------------------------------

def is_empty_trace(tr: Trace) -> bool:
    """Return True if this trace has no samples."""
    return tr.stats.npts == 0 or (np.asarray(tr.data).size == 0)


#from collections import defaultdict
def compute_stream_metrics(
    st: Stream,
    *,
    mode: str = "fast",          # "fast" or "heavy"
    heavy_max_traces: int = 200, # only use heavy mode if <= this many traces
    heavy_max_segments: int = 4000,  # rough cap; beyond this force fast
    attach_per_trace: bool = True,   # attach per-id metrics to each trace of that id
) -> dict:
    """
    Compute per-trace gap/overlap counts and percent availability.

    Modes
    -----
    - mode="fast" (default): no st.get_gaps(); compute per-ID metrics by sorting
      segments (O(n log n) per ID) and walking them once.
    - mode="heavy": uses st.get_gaps() BUT only if the stream is small enough,
      otherwise falls back to fast mode.

    Populates for each trace (if attach_per_trace=True):
      tr.stats.metrics = {
        "num_gaps": int,
        "num_overlaps": int,
        "lost_seconds": float,
        "percent_availability": float in [0, 100],
        "id_union_span_s": float,       # extra: union span per NSLC
        "id_union_covered_s": float,    # extra: covered (deduped) seconds per NSLC
        "id_segment_count": int,        # extra: segments for this NSLC in Stream
        "id_sampling_rates": list[float]
      }

    Returns
    -------
    dict stream-level summary
    """
    # Ensure metrics dict on each trace
    for tr in st:
        if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
            tr.stats.metrics = {}
        tr.stats.metrics.update({
            "num_gaps": 0,
            "num_overlaps": 0,
            "lost_seconds": 0.0,
            "percent_availability": 100.0,
        })

    # Group by NSLC
    by_id: Dict[Tuple[str, str, str, str], List[Trace]] = {}
    for tr in st:
        key = (tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel)
        by_id.setdefault(key, []).append(tr)

    # Decide mode
    total_traces = len(st)
    total_segments = sum(len(v) for v in by_id.values())
    use_heavy = (mode == "heavy" and
                 total_traces <= heavy_max_traces and
                 total_segments <= heavy_max_segments)

    stream_num_gaps = 0
    stream_num_overlaps = 0
    stream_total_lost_seconds = 0.0

    if use_heavy:
        # ---- HEAVY (ObsPy gap scanner), but only when small enough
        gaps = st.get_gaps() or []
        # Attribute gap/overlap counts to IDs (and then to each trace of that ID)
        id_counters: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
        for rec in gaps:
            if len(rec) < 7:
                continue
            net, sta, loc, cha, t0, t1, dt = rec[:7]
            key = (net, sta, loc, cha)
            c = id_counters.setdefault(key, {"gaps": 0, "overlaps": 0, "lost": 0.0})
            if dt > 0:
                c["gaps"] += 1
                c["lost"] += float(dt)
            elif dt < 0:
                c["overlaps"] += 1

        for key, traces in by_id.items():
            # union span and covered (approx) from segments
            spans = _sorted_spans(traces)
            union_span_s, union_cov_s = _union_span_and_covered(spans)
            sr_set = sorted({float(getattr(tr.stats, "sampling_rate", 0.0)) for tr in traces})
            c = id_counters.get(key, {"gaps": 0, "overlaps": 0, "lost": 0.0})
            stream_num_gaps += int(c["gaps"])
            stream_num_overlaps += int(c["overlaps"])
            stream_total_lost_seconds += float(c["lost"])

            if attach_per_trace:
                for tr in traces:
                    _attach_metrics(tr, c["gaps"], c["overlaps"], c["lost"],
                                    union_span_s, union_cov_s, len(traces), sr_set)

    else:
        # ---- FAST per-ID pass (no st.get_gaps())
        for key, traces in by_id.items():
            spans = _sorted_spans(traces)  # list of (start, end)
            # Count gaps/overlaps by walking sorted spans
            gaps, overlaps, lost_s = _count_gaps_overlaps(spans)
            union_span_s, union_cov_s = _union_span_and_covered(spans)
            sr_set = sorted({float(getattr(tr.stats, "sampling_rate", 0.0)) for tr in traces})

            stream_num_gaps += gaps
            stream_num_overlaps += overlaps
            stream_total_lost_seconds += lost_s

            if attach_per_trace:
                for tr in traces:
                    _attach_metrics(tr, gaps, overlaps, lost_s,
                                    union_span_s, union_cov_s, len(traces), sr_set)

    return {
        "stream_num_gaps": int(stream_num_gaps),
        "stream_num_overlaps": int(stream_num_overlaps),
        "stream_total_lost_seconds": float(stream_total_lost_seconds),
        "num_traces": int(total_traces),
        "num_ids": int(len(by_id)),
        "mode_used": "heavy" if use_heavy else "fast",
    }


# ---------------- helpers ----------------

def _sorted_spans(traces: List[Trace]) -> List[Tuple[float, float]]:
    """Return list of (start_s, end_s) as float seconds since epoch, sorted by start."""
    spans = []
    for tr in traces:
        t0 = float(tr.stats.starttime)
        t1 = float(tr.stats.endtime)
        if t1 > t0:
            spans.append((t0, t1))
    spans.sort(key=lambda x: x[0])
    return spans

def _count_gaps_overlaps(spans: List[Tuple[float, float]]) -> Tuple[int, int, float]:
    """Single pass over sorted spans to count gaps/overlaps and sum lost seconds (gaps)."""
    if not spans:
        return 0, 0, 0.0
    gaps = overlaps = 0
    lost = 0.0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:            # gap
            gaps += 1
            lost += (s - cur_e)
            cur_s, cur_e = s, e
        else:                     # overlap or touch
            if e <= cur_e:
                overlaps += 1     # fully contained
            else:
                if s < cur_e:
                    overlaps += 1 # partial overlap
                cur_e = e
    return gaps, overlaps, lost

def _union_span_and_covered(spans: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute total union span (max_e - min_s) and covered seconds (deduped)."""
    if not spans:
        return 0.0, 0.0
    min_s = spans[0][0]
    covered = 0.0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            covered += (cur_e - cur_s)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    covered += (cur_e - cur_s)
    union_span = max(sp[1] for sp in spans) - min_s
    return float(union_span), float(covered)

def _attach_metrics(
    tr: Trace,
    num_gaps: int,
    num_overlaps: int,
    lost_seconds: float,
    id_union_span_s: float,
    id_union_cov_s: float,
    id_segment_count: int,
    id_sampling_rates: List[float],
) -> None:
    """Attach metrics to a trace’s stats.metrics (per-ID metrics mirrored to each trace)."""
    tr.stats.metrics["num_gaps"] = int(num_gaps)
    tr.stats.metrics["num_overlaps"] = int(num_overlaps)
    tr.stats.metrics["lost_seconds"] = float(lost_seconds)
    # availability: use covered/union; if union=0, fallback to per-trace duration
    if id_union_span_s > 0.0:
        pct = max(0.0, min(100.0, (id_union_cov_s / id_union_span_s) * 100.0))
    else:
        dur = float(tr.stats.npts) * float(getattr(tr.stats, "delta", 0.0) or 0.0)
        pct = 100.0 if dur > 0.0 else 0.0
    tr.stats.metrics["percent_availability"] = float(pct)
    # extras (handy for debugging/decisions)
    tr.stats.metrics["id_union_span_s"] = float(id_union_span_s)
    tr.stats.metrics["id_union_covered_s"] = float(id_union_cov_s)
    tr.stats.metrics["id_segment_count"] = int(id_segment_count)
    tr.stats.metrics["id_sampling_rates"] = list(id_sampling_rates)

# -------------------------------
# Artifact detection/correction (mask-aware, robust)
# -------------------------------

def detect_and_correct_artifacts(
    tr: Trace,
    *,
    amp_limit: float = 1e10,
    count_thresh: int = 10,
    spike_thresh: float = 3.5,
    fill_method: Literal["median", "interpolate", "zero", "ignore"] = "median",
) -> None:
    """
    Detect clipping/spikes/steps and optionally correct them.

    Mask-aware: if ``tr.data`` is a masked array, masked samples are ignored. No
    split/merge is performed. A lightweight linear detrend is computed only on
    finite samples for spike detection.
    """
    data = tr.data
    if isinstance(data, np.ma.MaskedArray):
        y_raw = np.asarray(data.filled(np.nan), dtype=float)
        orig_mask = np.asarray(data.mask, dtype=bool)
    else:
        y_raw = np.asarray(data, dtype=float)
        orig_mask = ~np.isfinite(y_raw)

    finite = np.isfinite(y_raw)
    if finite.sum() < 8:
        # Too few usable samples; nothing to do
        tr.stats.artifacts = {  # minimal record
            "upper_clipped": False,
            "lower_clipped": False,
            "count_upper": 0,
            "count_lower": 0,
            "spike_count": 0,
            "step_count": 0,
        }
        return

    # --- Clipping on finite data
    yf = y_raw[finite]
    hi = min(np.nanmax(yf), amp_limit)
    lo = max(np.nanmin(yf), -amp_limit)
    count_upper = int(np.sum(yf >= hi))
    count_lower = int(np.sum(yf <= lo))
    upper_clipped = count_upper >= count_thresh
    lower_clipped = count_lower >= count_thresh

    # --- Linear detrend (finite only) for robust spike detection
    x = np.flatnonzero(finite).astype(float)
    if x.size >= 2:
        # least squares line fit on finite points
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, yf, rcond=None)[0]
        resid = yf - (a * x + b)
    else:
        resid = yf.copy()

    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    if mad == 0:
        mod_z = np.zeros_like(resid)
    else:
        mod_z = 0.6745 * np.abs(resid - med) / mad
    spike_local = np.where(mod_z > spike_thresh)[0]  # indices in the finite array
    spike_idx = np.flatnonzero(finite)[spike_local]  # map back to absolute indices

    # --- Step-function heuristic on finite differences (residual domain)
    # Use a conservative threshold relative to MAD
    step_threshold = 3.0 * (mad if mad > 0 else np.std(resid))
    d1 = np.abs(np.diff(resid))
    step_local = np.where(d1 > step_threshold)[0]
    # Convert edge-based indices to sample indices in absolute coordinates
    step_idx = np.flatnonzero(finite)[np.clip(step_local + 1, 0, np.count_nonzero(finite) - 1)]

    # --- Record
    tr.stats.artifacts = {
        "upper_clipped": bool(upper_clipped),
        "lower_clipped": bool(lower_clipped),
        "upper_limit": float(hi),
        "lower_limit": float(lo),
        "count_upper": int(count_upper),
        "count_lower": int(count_lower),
        "spike_count": int(spike_local.size),
        "spike_indices": spike_idx.tolist(),
        "step_count": int(step_local.size),
        "step_indices": step_idx.tolist(),
    }

    if fill_method == "ignore":
        return

    y_out = y_raw.copy()

    # Clip correction: pull extreme values back toward robust center
    if upper_clipped or lower_clipped:
        finite_inner = finite & (y_out < hi) & (y_out > lo)
        if np.any(finite_inner):
            fv = float(np.nanmedian(y_out[finite_inner]))
            y_out[finite & (y_out >= hi)] = fv
            y_out[finite & (y_out <= lo)] = fv

    # Spike correction
    if spike_idx.size > 0:
        if fill_method == "median":
            med_all = float(np.nanmedian(y_out[finite]))
            y_out[spike_idx] = med_all
        elif fill_method == "zero":
            y_out[spike_idx] = 0.0
        elif fill_method == "interpolate":
            # local linear interpolation using immediate finite neighbors
            for i in spike_idx:
                # search left/right finite, skipping other spikes
                l = i - 1
                while l >= 0 and (not finite[l] or l in spike_idx):
                    l -= 1
                r = i + 1
                n = y_out.size
                while r < n and (not finite[r] or r in spike_idx):
                    r += 1
                if l >= 0 and r < n:
                    y_out[i] = y_out[l] + (y_out[r] - y_out[l]) * ((i - l) / (r - l))
                elif l >= 0:
                    y_out[i] = y_out[l]
                elif r < n:
                    y_out[i] = y_out[r]
                else:
                    y_out[i] = 0.0

    # Step correction: smooth the jump sample by averaging neighbors
    if step_idx.size > 0:
        for i in step_idx:
            if i > 0 and i < (y_out.size - 1) and finite[i - 1] and finite[i + 1]:
                y_out[i] = 0.5 * (y_out[i - 1] + y_out[i + 1])

    # Write back, preserving original mask (if any)
    if isinstance(data, np.ma.MaskedArray):
        tr.data = np.ma.masked_array(y_out, mask=orig_mask)
    else:
        tr.data = y_out

    add_processing_step(tr, f"artifacts:corrected(method={fill_method})")


# -------------------------------
# Core: gap normalization + optional artifact pass + clean (filter/response)
# -------------------------------

def preprocess_trace(
    tr: Trace,
    *,
    # artifact stage & options
    run_artifact_fix: bool = False,
    artifact_kwargs: Optional[dict] = None,
    artifact_stage: Literal["pre", "post"] = "pre",
    # gaps
    normalize_gaps: bool = True,
    small_gap_sec: float = 2.0,
    long_gap_fill: Literal["leave", "zero", "previous", "noise", "linear"] = "zero",
    piecewise_detrend: bool = True,
    force_unmasked: bool = True,
    # cleaning / response removal
    do_clean: bool = True,
    taper_fraction: float = 0.05,
    filter_type: Literal["bandpass", "highpass", "lowpass"] = "bandpass",
    freq: Union[float, Tuple[float, float]] = (0.5, 30.0),
    corners: int = 6,
    zerophase: bool = False,
    inv=None,
    output_type: Literal["VEL", "DISP", "ACC", "DEF"] = "VEL",
    verbose: bool = False,
    post_filter_after_response: Optional[Tuple[
        Literal["bandpass", "highpass", "lowpass"], Union[float, Tuple[float, float]]
    ]] = None,
) -> bool:
    """
    Gap-aware, domain-agnostic preprocessing for a single Trace.

    Order of operations (configurable):
      • If ``artifact_stage == 'pre'`` and ``run_artifact_fix``,
        run :func:`detect_and_correct_artifacts` **before** any gap filling to
        avoid acting on synthetic samples.
      • Gap normalization (:func:`normalize_stream_gaps`): interpolate short gaps,
        optional piecewise detrend (mask-aware), then fill long gaps per policy.
      • If ``artifact_stage == 'post'`` and ``run_artifact_fix``, run artifact fix now
        (useful if you want spike stats on detrended data).
      • Safe pad→taper→filter→(response)→unpad via :func:`safe_pad_taper_filter`.

    Returns True if processed successfully; False if rejected.
    """
    if is_empty_trace(tr):
        add_processing_step(tr, "skip:empty")
        return False

    # (1) artifacts (pre)
    if run_artifact_fix and artifact_stage == "pre":
        try:
            detect_and_correct_artifacts(tr, **(artifact_kwargs or {}))
            add_processing_step(tr, "artifacts:fixed(pre)")
        except Exception as e:
            if verbose:
                print(f"[artifacts-pre] {tr.id}: {e}")

    # (2) gap normalization
    if normalize_gaps:
        tmp = Stream([tr.copy()])
        tmp2 = normalize_stream_gaps(
            tmp,
            small_gap_sec=small_gap_sec,
            long_gap_fill=long_gap_fill,
            piecewise=piecewise_detrend,
            force_unmasked=force_unmasked,
        )
        if len(tmp2):
            tr.data = tmp2[0].data
            add_processing_step(
                tr,
                f"gaps:normalized(s={small_gap_sec}, fill={long_gap_fill}, piecewise={piecewise_detrend})",
            )

    # (3) artifacts (post)
    if run_artifact_fix and artifact_stage == "post":
        try:
            detect_and_correct_artifacts(tr, **(artifact_kwargs or {}))
            add_processing_step(tr, "artifacts:fixed(post)")
        except Exception as e:
            if verbose:
                print(f"[artifacts-post] {tr.id}: {e}")

    # (4) cleaning: pad→taper→filter→(response)→unpad
    if do_clean:
        ok = safe_pad_taper_filter(
            tr,
            taper_fraction=taper_fraction,
            filter_type=filter_type,
            freq=freq,
            corners=corners,
            zerophase=zerophase,
            inv=inv,
            output_type=output_type,
            verbose=verbose,
        )
        if not ok:
            add_processing_step(tr, "clean:failed")
            return False

    return True


def preprocess_stream(
    st: Stream,
    *,
    run_artifact_fix: bool = False,
    artifact_kwargs: Optional[dict] = None,
    artifact_stage: Literal["pre", "post"] = "pre",
    normalize_gaps: bool = True,
    small_gap_sec: float = 2.0,
    long_gap_fill: Literal["leave", "zero", "previous", "noise", "linear"] = "zero",
    piecewise_detrend: bool = True,
    force_unmasked: bool = True,
    do_clean: bool = True,
    taper_fraction: float = 0.05,
    filter_type: Literal["bandpass", "highpass", "lowpass"] = "bandpass",
    freq: Union[float, Tuple[float, float]] = (0.5, 30.0),
    corners: int = 6,
    zerophase: bool = False,
    inv=None,
    output_type: Literal["VEL", "DISP", "ACC", "DEF"] = "VEL",
    verbose: bool = False,
    precheck_response: bool = False,
) -> Stream:
    """Process each trace with :func:`preprocess_trace`. Traces that fail are dropped."""
    out = Stream()

    # compute metrics before
    presummary = compute_stream_metrics(st)
    if inv and precheck_response:
        stationxml_match_report(st, inv)

    for tr in st:
        tr2 = tr.copy()
        ok = preprocess_trace(
            tr2,
            run_artifact_fix=run_artifact_fix,
            artifact_kwargs=artifact_kwargs,
            artifact_stage=artifact_stage,
            normalize_gaps=normalize_gaps,
            small_gap_sec=small_gap_sec,
            long_gap_fill=long_gap_fill,
            piecewise_detrend=piecewise_detrend,
            force_unmasked=force_unmasked,
            do_clean=do_clean,
            taper_fraction=taper_fraction,
            filter_type=filter_type,
            freq=freq,
            corners=corners,
            zerophase=zerophase,
            inv=inv,
            output_type=output_type,
            verbose=verbose,
        )
        if ok:
            out.append(tr2)

    postsummary = compute_stream_metrics(out)
    return out


# -------------------------------
# Sorting utility
# -------------------------------

def order_traces_by_id(st: Stream) -> Stream:
    sorted_ids = sorted(tr.id for tr in st)
    return Stream([tr.copy() for id_ in sorted_ids for tr in st if tr.id == id_])
