from __future__ import annotations
import numpy as np
from obspy import Trace, Stream, UTCDateTime

from typing import Literal
import re

def smart_fill(tr: Trace, short_thresh=5.0, long_fill_value=0.0):
    """
    Interpolate short masked gaps (<= short_thresh s) and fill long gaps with a constant.
    Returns a new Trace with plain ndarray (no mask).
    """
    data = tr.data
    if not isinstance(data, np.ma.MaskedArray) or not np.any(data.mask):
        return tr.copy()

    sr = float(tr.stats.sampling_rate or 0.0)
    short, long = classify_gaps(tr, threshold_seconds=short_thresh)
    out = tr.copy()
    arr = data.filled(long_fill_value).astype(np.float32, copy=True)
    mask = np.asarray(data.mask, dtype=bool)
    x = np.arange(arr.size)

    if short:
        valid = ~mask
        for s, e in short:
            # Find nearest valid on each side; handle edge gaps gracefully
            left = s - 1
            while left >= 0 and not valid[left]:
                left -= 1
            right = e
            while right < arr.size and not valid[right]:
                right += 1

            if left >= 0 and right < arr.size:
                # strict linear interp using the two neighbors
                arr[s:e] = np.interp(x[s:e], [left, right], [arr[left], arr[right]])
            elif left >= 0:
                arr[s:e] = arr[left]
            elif right < arr.size:
                arr[s:e] = arr[right]
            else:
                # all masked (degenerate)
                arr[s:e] = long_fill_value

    # Long gaps already filled by .filled(long_fill_value) above
    out.data = arr
    return out



def classify_gaps(tr: Trace, sampling_rate=None, threshold_seconds=5.0):
    """
    Return (short, long) masked spans as index pairs [start, end),
    split by duration threshold (seconds).
    """
    data = tr.data
    if not isinstance(data, np.ma.MaskedArray):
        return [], []
    mask = np.asarray(data.mask, dtype=bool)
    if not mask.any():
        return [], []

    sr = float(sampling_rate or tr.stats.sampling_rate or 0.0)
    if sr <= 0:
        # fall back: treat all gaps as "long"
        p = np.concatenate(([False], mask, [False]))
        idx = np.flatnonzero(p[1:] != p[:-1])
        starts, ends = idx[0::2], idx[1::2]
        return [], [(int(s), int(e)) for s, e in zip(starts, ends)]

    # runs of True (masked)
    p = np.concatenate(([False], mask, [False]))
    idx = np.flatnonzero(p[1:] != p[:-1])
    starts, ends = idx[0::2], idx[1::2]
    durs = (ends - starts) / sr

    short = [(int(s), int(e)) for (s, e), d in zip(zip(starts, ends), durs) if d <= threshold_seconds]
    long  = [(int(s), int(e)) for (s, e), d in zip(zip(starts, ends), durs) if d >  threshold_seconds]
    return short, long


def fill_stream_gaps(stream, method="constant", **kwargs):
    """
    Apply a gap-filling method to all Traces in a Stream.

    Parameters
    ----------
    stream : Stream
        ObsPy Stream with (preferably) masked gaps.
    method : {"constant", "linear", "previous", "noise"}
        Gap filling method.
    **kwargs :
        - constant: fill_value (float)
        - noise: taper_percentage (float)
        - others: reserved for future options
    """

    methods = {
        "constant": fill_gaps_with_constant,
        "linear": fill_gaps_with_linear_interpolation,
        "previous": fill_gaps_with_previous_value,
        "noise": fill_gaps_with_filtered_noise
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Valid options: {list(methods.keys())}")

    filled_stream = Stream()
    for tr in stream:
        filled_stream.append(methods[method](tr, **kwargs))
    return filled_stream


def fill_gaps_with_constant(tr: Trace, fill_value: float = 0.0, dtype_out=None, **kwargs) -> Trace:
    """
    Fill masked gaps in a Trace with a constant value (default is 0.0).

    Parameters:
    -----------
    tr : obspy.Trace
        Trace with masked gaps in tr.data.
    fill_value : float
        Value to fill the gaps with.

    Returns:
    --------
    obspy.Trace
        Trace with gaps filled and no masked array.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    arr = tr.data.filled(fill_value)
    if dtype_out is not None:
        arr = arr.astype(dtype_out, copy=False)
    new_tr.data = arr
    return new_tr

def fill_gaps_with_linear_interpolation(tr: Trace, **kwargs) -> Trace:
    """
    Fill masked gaps in a Trace using linear interpolation.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    data = new_tr.data.astype(np.float32)
    mask = data.mask

    if not np.any(mask):
        return new_tr

    x = np.arange(len(data))
    data[mask] = np.interp(x[mask], x[~mask], data[~mask])
    new_tr.data = data.data  # strip the mask
    return new_tr


def fill_gaps_with_previous_value(tr: Trace, **kwargs) -> Trace:
    """
    Fill masked gaps by repeating the previous valid value (forward-fill).
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    data = new_tr.data.astype(np.float32)
    mask = np.asarray(data.mask, dtype=bool)
    if not mask.any():
        return new_tr

    y = data.filled(np.nan)
    # Build forward-fill indices
    idx = np.arange(y.size)
    valid = ~np.isnan(y)
    if not valid.any():
        # nothing valid; return zeros
        new_tr.data = np.zeros_like(data, dtype=np.float32)
        return new_tr

    # last valid index up to i (forward fill)
    last = np.maximum.accumulate(np.where(valid, idx, -1))
    # For initial NaNs with no prior valid, back-fill from first valid
    first_valid = np.flatnonzero(valid)[0]
    last[last < 0] = first_valid
    y = y[last].astype(np.float32, copy=False)

    new_tr.data = y  # plain ndarray
    return new_tr


def fill_gaps_with_filtered_noise(tr: Trace, taper_percentage=0.2, **kwargs) -> Trace:
    """
    Fill masked gaps using spectrally matched noise.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    data = new_tr.data.astype(np.float32)
    mask = data.mask

    if not np.any(mask):
        return new_tr

    noise = _generate_spectrally_matched_noise(new_tr, taper_percentage=taper_percentage)
    data[mask] = noise[mask]
    new_tr.data = data.data
    return new_tr


def _next_pow2(n: int) -> int:
    """Small helper used by _generate_spectrally_matched_noise."""
    n = int(n)
    if n < 1: 
        return 1
    return 1 << (n - 1).bit_length()

def _generate_spectrally_matched_noise(tr: Trace, taper_percentage=0.2, **kwargs):
    """
    Generate noise matched to the magnitude spectrum of the (zero-mean) valid samples.
    If valid samples are scarce, fall back to white noise.
    """
    data = tr.data
    if not isinstance(data, np.ma.MaskedArray):
        data = np.ma.masked_array(data)

    valid = ~data.mask
    y = np.asarray(data[valid], dtype=np.float64)
    n_total = data.size
    if y.size < 8:
        # too little to estimate spectrum
        noise = np.random.randn(n_total).astype(np.float64)
    else:
        y = y - y.mean()
        nfft = _next_pow2(max(n_total, y.size))
        amp = np.abs(np.fft.rfft(y, n=nfft))
        # avoid zeros to prevent total silence
        amp = np.maximum(amp, np.percentile(amp, 5) * 0.1)
        phases = np.exp(2j * np.pi * np.random.rand(amp.size))
        spec = amp * phases
        synth = np.fft.irfft(spec, n=nfft).real[:n_total]
        noise = synth

    # Taper edges to avoid sharp starts/ends
    npts = n_total
    if 0.0 < taper_percentage < 1.0:
        tl = max(1, int(npts * taper_percentage / 2))
        if tl * 2 < npts:
            taper = np.ones(npts, dtype=np.float64)
            ramp = np.linspace(0, 1, tl, dtype=np.float64)
            taper[:tl] = ramp
            taper[-tl:] = ramp[::-1]
            noise *= taper

    return noise

def piecewise_detrend(tr: Trace,
                      mode: str = "linear",         # "linear" or "simple"
                      null_values=(0.0, np.nan),
                      use_null_values: bool = True,
                      keep_mask: bool = True) -> Trace:
    """
    Piecewise detrend by contiguous valid spans:
      - mode="linear": remove a*x + b per span (same as piecewise_detrend)
      - mode="simple": remove mean per span
    """
    if mode == "linear":
        from flovopy.core.gaputils import piecewise_detrend
        return piecewise_detrend(tr, null_values, use_null_values, keep_mask)

    # simple (mean removal)
    out = tr.copy()
    data = out.data
    if isinstance(data, np.ma.MaskedArray):
        mask = np.asarray(data.mask, dtype=bool)
        y = np.asarray(data.filled(0.0), dtype=np.float64)
        had_mask = True
    else:
        had_mask = False
        y = np.asarray(data, dtype=np.float64)
        if use_null_values:
            mask = ~np.isfinite(y)
            for nv in null_values:
                if isinstance(nv, float) and np.isnan(nv): continue
                mask |= (y == nv)
            mask = np.asarray(mask, dtype=bool)
        else:
            mask = np.zeros_like(y, dtype=bool)

    n = y.size
    if n < 1:
        return out

    # find valid segments vectorized
    valid = ~mask
    p = np.concatenate(([False], valid, [False]))
    changes = np.flatnonzero(p[1:] != p[:-1])
    starts, ends = changes[0::2], changes[1::2]

    for s, e in zip(starts, ends):
        seg = y[s:e]
        if seg.size:
            seg -= seg.mean()

    if had_mask and keep_mask:
        out.data = np.ma.masked_array(y.astype(data.dtype, copy=False), mask=mask)
    else:
        out.data = y.astype(data.dtype, copy=False)

    try:
        proc = getattr(out.stats, "processing", None)
        if proc is None: out.stats.processing = []
        out.stats.processing.append(f"piecewise_detrend(mode={mode})")
    except Exception:
        pass
    return out





import re
_GAP_RE = re.compile(r"GAP\\s+(?P<dur>[^\\s]+)s\\s+from\\s+(?P<t1>[^\\s]+)\\s+to\\s+(?P<t2>[^\\s]+)")

def mask_gaps(trace, fill_value=0.0, inplace=True, validate_fill_value=False):
    """
    Re-applies a mask to gap regions recorded in trace.stats.processing.

    Parameters
    ----------
    trace : obspy.Trace
        The trace with previously unmasked gaps.
    fill_value : float, optional
        Expected value used to fill gaps. Only used if `validate_fill_value=True`.
    inplace : bool, optional
        If True, modify in-place. If False, return a copy.
    validate_fill_value : bool, optional
        If True, only mask gaps where values match `fill_value`. Slower but safer.
    """

    if not inplace:
        trace = trace.copy()

    if not hasattr(trace.stats, "processing"):
        return trace if not inplace else None

    processing_lines = [p for p in trace.stats.processing if p.startswith("GAP")]
    if not processing_lines:
        return trace if not inplace else None

    sr = trace.stats.sampling_rate
    start = trace.stats.starttime
    npts = trace.stats.npts

    mask = np.zeros(npts, dtype=bool)

    for line in processing_lines:
        try:
            m = _GAP_RE.search(line)
            if not m:
                print(f"✘ Could not parse gap line '{line}'")
                continue
            t1 = UTCDateTime(m.group("t1"))
            t2 = UTCDateTime(m.group("t2"))
            idx1 = max(0, int((t1 - start) * sr))
            idx2 = min(npts, int((t2 - start) * sr))
            if idx2 > idx1:
                if validate_fill_value:
                    segment = trace.data[idx1:idx2]
                    if np.allclose(segment, fill_value, equal_nan=True):
                        mask[idx1:idx2] = True
                else:
                    mask[idx1:idx2] = True
        except Exception as e:
            print(f"✘ Could not parse gap line '{line}': {e}")

    trace.data = np.ma.masked_array(trace.data, mask=mask)
    return trace if not inplace else None

def unmask_gaps(trace, fill_value=0.0, inplace=True, verbose=False, log_gaps=False):
    if not inplace:
        trace = trace.copy()

    if not isinstance(trace.data, np.ma.MaskedArray) or trace.data.mask is np.ma.nomask:
        return trace if not inplace else None

    sr = trace.stats.sampling_rate
    start = trace.stats.starttime
    mask = trace.data.mask

    if isinstance(mask, np.ndarray):
        gap_idxs = np.where(mask)[0]
    else:
        gap_idxs = np.arange(trace.stats.npts) if mask else []

    n_gaps = 0
    if gap_idxs.size:
        if log_gaps:
            split_idx = np.where(np.diff(gap_idxs) != 1)[0] + 1
            gap_groups = np.split(gap_idxs, split_idx)
            n_gaps = len(gap_groups)

            if not hasattr(trace.stats, "processing"):
                trace.stats.processing = []
            trace.stats.processing.append(f"Filled {n_gaps} gaps with {fill_value}")

            for group in gap_groups:
                t1 = start + group[0] / sr
                t2 = start + (group[-1] + 1) / sr
                trace.stats.processing.append(f"GAP {t2 - t1:.2f}s from {t1} to {t2}")

    trace.data = trace.data.filled(fill_value)

    if verbose and n_gaps > 500:
        print(f"⚠️ Large number of GAP lines added to trace.stats.processing: {n_gaps}")

    return trace if not inplace else None



def _interp_short_gaps_only(tr: Trace, short_gap_sec: float) -> Trace:
    """
    Interpolate short masked gaps (<= short_gap_sec) and keep *long* gaps masked.
    Implemented via smart_fill(long_fill_value=np.nan) + re-masking NaNs.
    """
    data = tr.data
    if not isinstance(data, np.ma.MaskedArray) or not np.any(data.mask):
        return tr.copy()
    tmp = smart_fill(tr, short_thresh=short_gap_sec, long_fill_value=np.nan)  # ndarray
    arr = np.asarray(tmp.data)
    mask = ~np.isfinite(arr)  # re-mask NaNs (long gaps)
    out = tr.copy()
    out.data = np.ma.masked_array(arr, mask=mask)
    return out


def normalize_stream_gaps(
    st: Stream,
    *,
    small_gap_sec: float = 2.0,
    long_gap_fill: Literal["leave", "zero", "previous", "noise", "linear"] = "zero",
    piecewise: bool = True,
    force_unmasked: bool = True,
) -> Stream:
    """
    Normalize a Stream for downstream processing (no filtering, no global taper).

    Steps per Trace:
      1) Interpolate short masked gaps (<= small_gap_sec); keep long gaps masked.
      2) (Optional) piecewise linear detrend over valid spans only (mask-aware).
      3) Fill long gaps using `long_gap_fill` policy (or leave masked).
      4) If `force_unmasked`, convert remaining masks to a plain ndarray.

    Parameters
    ----------
    st : obspy.Stream
        Input stream (may contain masked arrays).
    small_gap_sec : float
        Threshold (seconds) for short gap interpolation.
    long_gap_fill : {"leave","zero","previous","noise","linear"}
        Long-gap policy after detrend:
          - "leave"   : keep masked (only valid if force_unmasked=False)
          - "zero"    : fill with 0.0
          - "previous": forward-fill from last valid
          - "noise"   : spectrally matched noise
          - "linear"  : linear interpolation across all masked samples
    piecewise : bool
        If True, apply piecewise linear detrend on valid spans.
    force_unmasked : bool
        If True, ensure plain ndarray (no masked arrays) in output.

    Returns
    -------
    obspy.Stream
        Stream with gaps normalized and (optionally) detrended, ready for
        algorithms that don’t accept masked arrays.
    """
    wip = st.copy()

    # 0) Optional front-door sanitation/merge (usually unnecessary if MiniSEED is clean)

    out = Stream()
    for tr in wip:
        work = tr.copy()

        # 1) Interpolate only short gaps; leave long masked for detrend
        if isinstance(work.data, np.ma.MaskedArray) and np.any(work.data.mask) and small_gap_sec > 0:
            work = _interp_short_gaps_only(work, short_gap_sec=small_gap_sec)

        # 2) Piecewise detrend over valid spans (no split/merge, no taper)
        if piecewise:
            # use_null_values=False because we already carry masks for long gaps
            work = piecewise_detrend(
                work,
                null_values=None,
                use_null_values=False,
                keep_mask=False,
            )

        # 3) Fill long gaps according to policy
        if isinstance(work.data, np.ma.MaskedArray) and np.any(work.data.mask):
            if long_gap_fill == "leave":
                pass  # keep masks (only meaningful if force_unmasked=False)
            elif long_gap_fill == "zero":
                work = fill_gaps_with_constant(work, fill_value=0.0)
            elif long_gap_fill == "previous":
                work = fill_gaps_with_previous_value(work)
            elif long_gap_fill == "noise":
                work = fill_gaps_with_filtered_noise(work, taper_percentage=0.2)
            elif long_gap_fill == "linear":
                work = fill_gaps_with_linear_interpolation(work)
            else:
                work = fill_gaps_with_constant(work, fill_value=0.0)

        # 4) Ensure plain ndarray if requested
        if force_unmasked and isinstance(work.data, np.ma.MaskedArray):
            work = fill_gaps_with_constant(work, fill_value=0.0)

        out.append(work)

    return out