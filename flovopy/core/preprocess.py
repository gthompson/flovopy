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

# If you move/remove this module, keep these imports stable for callers.
try:
    from flovopy.core.remove_response import (
        safe_pad_taper_filter,
    )
except Exception as _e:  # pragma: no cover
    def safe_pad_taper_filter(*args, **kwargs) -> bool:  # type: ignore[override]
        raise RuntimeError(
            "safe_pad_taper_filter not available – ensure flovopy.core.remove_response is importable"
        )


# -------------------------------
# QC metrics & utilities
# -------------------------------

def is_empty_trace(tr: Trace) -> bool:
    """Return True if this trace has no samples."""
    return tr.stats.npts == 0 or (np.asarray(tr.data).size == 0)


#from collections import defaultdict
def compute_stream_metrics(st: Stream) -> dict:
    """
    Compute per-trace gap/overlap counts and percent availability using
    Stream.get_gaps(), and store results in tr.stats.metrics.

    Adds (per Trace):
      - tr.stats.metrics["num_gaps"]         : int
      - tr.stats.metrics["num_overlaps"]     : int
      - tr.stats.metrics["lost_seconds"]     : float (sum of gap durations > 0)
      - tr.stats.metrics["percent_availability"] : float in [0, 100]
    Returns a small stream-level summary dict.
    """
    # Ensure metrics dicts exist
    for tr in st:
        if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
            tr.stats.metrics = {}
        # initialize defaults
        tr.stats.metrics.update({
            "num_gaps": 0,
            "num_overlaps": 0,
            "lost_seconds": 0.0,
            "percent_availability": 100.0,
        })

    # Build a quick NSLC -> traces list map (usually 1 trace per id)
    by_id = {}
    for tr in st:
        key = (tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel)
        by_id.setdefault(key, []).append(tr)

    # Stream-level gaps: tuples are typically
    # (net, sta, loc, cha, t0, t1, dt_seconds, nsamples)  <-- 8 fields
    # but be defensive and only use the first 7 if needed.
    gaps = st.get_gaps() or []

    # Attribute each gap/overlap to its NSLC
    for rec in gaps:
        if len(rec) < 7:
            # Unexpected shape; skip
            continue
        net, sta, loc, cha, t0, t1, dt = rec[:7]
        key = (net, sta, loc, cha)
        # Only care about traces we have in this stream
        for tr in by_id.get(key, []):
            if dt > 0:
                tr.stats.metrics["num_gaps"] += 1
                tr.stats.metrics["lost_seconds"] += float(dt)
            elif dt < 0:
                tr.stats.metrics["num_overlaps"] += 1
            # dt == 0 is rare; ignore

    # Finalize percent availability per trace
    for tr in st:
        dur = float(tr.stats.npts) * float(tr.stats.delta or 0.0)
        if dur > 0.0:
            lost = float(tr.stats.metrics.get("lost_seconds", 0.0))
            pct = max(0.0, min(100.0, (dur - lost) / dur * 100.0))
            tr.stats.metrics["percent_availability"] = pct
        else:
            tr.stats.metrics["percent_availability"] = 0.0

    # Stream-level summary (optional, useful for logging)
    summary = {
        "stream_num_gaps": int(sum(tr.stats.metrics["num_gaps"] for tr in st)),
        "stream_num_overlaps": int(sum(tr.stats.metrics["num_overlaps"] for tr in st)),
        "stream_total_lost_seconds": float(sum(tr.stats.metrics["lost_seconds"] for tr in st)),
        "num_traces": len(st),
    }
    return summary

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
            post_filter_after_response=post_filter_after_response,
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
    post_filter_after_response: Optional[Tuple[
        Literal["bandpass", "highpass", "lowpass"], Union[float, Tuple[float, float]]
    ]] = None,
) -> Stream:
    """Process each trace with :func:`preprocess_trace`. Traces that fail are dropped."""
    out = Stream()

    # compute metrics before
    presummary = compute_stream_metrics(st)

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
            post_filter_after_response=post_filter_after_response,
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
