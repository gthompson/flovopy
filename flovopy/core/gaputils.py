from __future__ import annotations
import numpy as np
from obspy import Trace, Stream, UTCDateTime
from typing import Literal
import re



def fill_gaps_with_linear_interpolation(trace: Trace, *, inplace: bool = False) -> Trace:
    """
    Fill masked gaps in a trace by linear interpolation.

    Parameters
    ----------
    trace
        Input ObsPy Trace. Data may be a masked array.
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Trace with masked gaps filled by interpolation.

    Notes
    -----
    - If there are no masked samples, the trace is returned unchanged.
    - If all samples are masked, the data are filled with zeros.
    - If only one sample is unmasked, all masked samples are filled with that value.
    """
    tr = trace if inplace else trace.copy()

    data = np.asanyarray(tr.data)
    if not np.ma.isMaskedArray(data):
        return tr

    data = np.ma.array(data, copy=True)
    mask = np.ma.getmaskarray(data)

    if not mask.any():
        tr.data = np.asarray(data, dtype=np.float32)
        return tr

    x = np.arange(data.size)
    valid = ~mask

    out = data.filled(np.nan).astype(np.float32, copy=False)

    if not valid.any():
        out[:] = 0.0
    elif valid.sum() == 1:
        out[mask] = out[valid][0]
    else:
        out[mask] = np.interp(x[mask], x[valid], out[valid])

    tr.data = out
    tr.stats.npts = len(tr.data)
    return tr


def mask_gaps(
    trace: Trace,
    *,
    null_values=(0.0,),
    use_null_values: bool = True,
    inplace: bool = False,
) -> Trace:
    """
    Convert trace data to a masked array, masking NaNs and optionally null values.

    Parameters
    ----------
    trace
        Input ObsPy Trace.
    null_values
        Values to treat as missing when `use_null_values=True`.
    use_null_values
        If True, mask samples equal to any value in `null_values`.
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Trace whose data are stored as a masked array.
    """
    tr = trace if inplace else trace.copy()

    data = np.asarray(tr.data, dtype=np.float32)
    mask = ~np.isfinite(data)

    if use_null_values and null_values is not None:
        for v in null_values:
            mask |= (data == v)

    tr.data = np.ma.masked_array(data, mask=mask, copy=False)
    tr.stats.npts = len(tr.data)
    return tr


def unmask_gaps(
    trace: Trace,
    *,
    fill_value: float = 0.0,
    inplace: bool = False,
) -> Trace:
    """
    Replace masked samples in a trace with a constant fill value.

    Parameters
    ----------
    trace
        Input ObsPy Trace. If data are not masked, the trace is returned unchanged.
    fill_value
        Value used to replace masked samples.
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Trace with plain ndarray data and no mask.
    """
    tr = trace if inplace else trace.copy()

    data = tr.data
    if np.ma.isMaskedArray(data):
        tr.data = np.asarray(data.filled(fill_value), dtype=np.float32)
    else:
        tr.data = np.asarray(data, dtype=np.float32)

    tr.stats.npts = len(tr.data)
    return tr


def piecewise_detrend(
    trace: Trace,
    *,
    null_values=(0.0,),
    use_null_values: bool = True,
    keep_mask: bool = True,
    mode: str = "simple",
    inplace: bool = False,
) -> Trace:
    """
    Detrend contiguous valid segments of a trace independently.

    Parameters
    ----------
    trace
        Input ObsPy Trace.
    null_values
        Values to treat as missing when `use_null_values=True`.
    use_null_values
        If True, mask samples equal to values in `null_values`.
    keep_mask
        If True, return masked data where gaps remain masked.
        If False, return plain ndarray data with masked samples filled by zero.
    mode
        Detrending mode:
            - "simple": subtract segment mean
            - "linear": remove best-fit linear trend from each segment
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Piecewise-detrended trace.

    Notes
    -----
    - Gaps are identified using masked samples.
    - Each contiguous unmasked segment is detrended independently.
    - If a segment has fewer than 2 samples, it is only mean-centered in
      "linear" mode.
    """
    tr = trace if inplace else trace.copy()

    # First ensure gaps are masked consistently
    tr = mask_gaps(
        tr,
        null_values=null_values,
        use_null_values=use_null_values,
        inplace=True,
    )

    data = np.ma.array(tr.data, copy=True)
    mask = np.ma.getmaskarray(data)

    if data.size == 0:
        return tr

    valid = ~mask
    if not valid.any():
        if keep_mask:
            tr.data = data.astype(np.float32, copy=False)
        else:
            tr.data = np.asarray(data.filled(0.0), dtype=np.float32)
        tr.stats.npts = len(tr.data)
        return tr

    # Find contiguous valid runs
    valid_idx = np.where(valid)[0]
    breaks = np.where(np.diff(valid_idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(valid_idx) - 1]

    out = data.astype(np.float32, copy=True)

    for s, e in zip(starts, ends):
        seg_idx = valid_idx[s : e + 1]
        y = np.asarray(out[seg_idx], dtype=np.float32)

        if y.size == 0:
            continue

        if mode == "simple":
            y = y - np.nanmean(y)

        elif mode == "linear":
            if y.size < 2:
                y = y - np.nanmean(y)
            else:
                x = np.arange(y.size, dtype=np.float32)
                coeffs = np.polyfit(x, y, 1)
                trend = coeffs[0] * x + coeffs[1]
                y = y - trend

        else:
            raise ValueError(f"Unknown piecewise_detrend mode: {mode!r}")

        out[seg_idx] = y

    if keep_mask:
        tr.data = np.ma.masked_array(
            np.asarray(out.filled(0.0), dtype=np.float32),
            mask=mask,
            copy=False,
        )
    else:
        tr.data = np.asarray(out.filled(0.0), dtype=np.float32)

    tr.stats.npts = len(tr.data)
    return tr





def smart_fill(
    trace: Trace,
    *,
    short_gap_threshold_samples: int = 5,
    long_gap_method: str = "noise",
    fill_value: float = 0.0,
    null_values=(0.0,),
    use_null_values: bool = True,
    inplace: bool = False,
) -> Trace:
    """
    Fill masked/null gaps in a trace using different strategies for short and long gaps.

    Parameters
    ----------
    trace
        Input ObsPy Trace.
    short_gap_threshold_samples
        Gaps of length <= this threshold are treated as short gaps and filled
        by linear interpolation.
    long_gap_method
        Method used for longer gaps. One of:
            - "noise": fill with spectrally matched / filtered noise
            - "previous": fill with previous valid value
            - "constant": fill with `fill_value`
            - "linear": fill with linear interpolation
    fill_value
        Constant fill value used when `long_gap_method="constant"`.
    null_values
        Values to treat as missing when `use_null_values=True`.
    use_null_values
        If True, values in `null_values` are first masked as gaps.
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Trace with plain ndarray data and gaps filled.

    Notes
    -----
    This function expects the supporting helpers:
    - `mask_gaps()`
    - `fill_gaps_with_linear_interpolation()`
    - `fill_gaps_with_previous_value()`
    - `fill_gaps_with_filtered_noise()`
    """
    tr = trace if inplace else trace.copy()

    # Start by masking invalid/null samples consistently
    tr = mask_gaps(
        tr,
        null_values=null_values,
        use_null_values=use_null_values,
        inplace=True,
    )

    data = tr.data
    if not np.ma.isMaskedArray(data):
        tr.data = np.asarray(data, dtype=np.float32)
        tr.stats.npts = len(tr.data)
        return tr

    mask = np.ma.getmaskarray(data)
    if not mask.any():
        tr = unmask_gaps(tr, fill_value=fill_value, inplace=True)
        tr.stats.npts = len(tr.data)
        return tr

    # If everything is masked, fall back to constant fill
    if mask.all():
        tr.data = np.full(data.shape, fill_value, dtype=np.float32)
        tr.stats.npts = len(tr.data)
        return tr

    # Identify contiguous masked runs
    masked_idx = np.where(mask)[0]
    d = np.diff(masked_idx)
    gap_starts = np.where(np.insert(d, 0, 2) > 1)[0]
    gap_ends = np.where(np.append(d, 2) > 1)[0] - 1

    # Work on a copy so each gap fill can update later context
    work = tr.copy()

    for i in range(len(gap_starts)):
        s = masked_idx[gap_starts[i]]
        e = masked_idx[gap_ends[i]] + 1
        gap_len = e - s

        gap_tr = work.copy()
        submask = np.zeros(gap_tr.stats.npts, dtype=bool)
        submask[s:e] = True

        # Keep only the current gap masked in this temporary trace
        if np.ma.isMaskedArray(gap_tr.data):
            base = gap_tr.data.filled(np.nan)
        else:
            base = np.asarray(gap_tr.data, dtype=np.float32)

        gap_tr.data = np.ma.masked_array(
            np.asarray(base, dtype=np.float32),
            mask=submask,
            copy=False,
        )

        if gap_len <= short_gap_threshold_samples:
            filled = fill_gaps_with_linear_interpolation(gap_tr, inplace=False)
        else:
            method = long_gap_method.lower()
            if method == "noise":
                filled = fill_gaps_with_filtered_noise(gap_tr, inplace=False)
            elif method == "previous":
                filled = fill_gaps_with_previous_value(gap_tr, inplace=False)
            elif method == "constant":
                filled = unmask_gaps(gap_tr, fill_value=fill_value, inplace=False)
            elif method == "linear":
                filled = fill_gaps_with_linear_interpolation(gap_tr, inplace=False)
            else:
                raise ValueError(f"Unknown long_gap_method: {long_gap_method!r}")

        # Update only the current gap in the working trace
        work_data = np.asarray(
            work.data.filled(np.nan) if np.ma.isMaskedArray(work.data) else work.data,
            dtype=np.float32,
        )
        filled_data = np.asarray(filled.data, dtype=np.float32)
        work_data[s:e] = filled_data[s:e]
        work.data = np.ma.masked_array(work_data, mask=np.ma.getmaskarray(work.data), copy=False)
        work.data.mask[s:e] = False

    # Return plain ndarray output
    work = unmask_gaps(work, fill_value=fill_value, inplace=True)
    work.stats.npts = len(work.data)

    if inplace:
        trace.data = work.data
        trace.stats = work.stats
        return trace

    return work


def normalize_stream_gaps(
    stream: Stream,
    *,
    detrend_first: bool = True,
    detrend_mode: str = "simple",
    mask_null_values: bool = True,
    null_values=(0.0,),
    fill_gaps: bool = False,
    short_gap_threshold_samples: int = 5,
    long_gap_method: str = "noise",
    fill_value: float = 0.0,
    keep_mask: bool = False,
    inplace: bool = False,
) -> Stream:
    """
    Normalize gaps across all traces in a stream.

    Parameters
    ----------
    stream
        Input ObsPy Stream.
    detrend_first
        If True, apply piecewise detrending before optional gap filling.
    detrend_mode
        Passed to `piecewise_detrend()`. One of:
            - "simple"
            - "linear"
    mask_null_values
        If True, values in `null_values` are treated as missing.
    null_values
        Values to mask as gaps when `mask_null_values=True`.
    fill_gaps
        If True, fill gaps using `smart_fill()`.
        If False, leave gaps masked unless `keep_mask=False`, in which case
        they are converted to plain arrays using `fill_value`.
    short_gap_threshold_samples
        Passed to `smart_fill()`.
    long_gap_method
        Passed to `smart_fill()`.
    fill_value
        Constant used when unmasking or constant-filling gaps.
    keep_mask
        If True and `fill_gaps=False`, leave masked arrays in output.
        If False, convert masked arrays to plain arrays using `fill_value`.
    inplace
        If True, modify the input stream in place.

    Returns
    -------
    Stream
        Stream with normalized gap handling.

    Notes
    -----
    Typical usage patterns:
    - `fill_gaps=False, keep_mask=True`:
        preserve gaps as masks for later processing
    - `fill_gaps=False, keep_mask=False`:
        preserve gaps structurally, but output plain arrays
    - `fill_gaps=True`:
        produce gap-filled plain arrays
    """
    st = stream if inplace else stream.copy()

    out = Stream()

    for tr in st:
        work = tr.copy()

        # Step 1: detrend piecewise across valid segments
        if detrend_first:
            work = piecewise_detrend(
                work,
                null_values=null_values,
                use_null_values=mask_null_values,
                keep_mask=True,
                mode=detrend_mode,
                inplace=False,
            )
        else:
            work = mask_gaps(
                work,
                null_values=null_values,
                use_null_values=mask_null_values,
                inplace=False,
            )

        # Step 2: optional gap fill
        if fill_gaps:
            work = smart_fill(
                work,
                short_gap_threshold_samples=short_gap_threshold_samples,
                long_gap_method=long_gap_method,
                fill_value=fill_value,
                null_values=null_values,
                use_null_values=mask_null_values,
                inplace=False,
            )
        else:
            if not keep_mask:
                work = unmask_gaps(work, fill_value=fill_value, inplace=False)

        work.stats.npts = len(work.data)
        out.append(work)

    if inplace:
        stream.clear()
        stream.extend(out)
        return stream

    return out




def classify_gaps(trace: Trace) -> list[tuple[int, int]]:
    """
    Return contiguous masked-gap intervals in a trace.

    Parameters
    ----------
    trace
        Input ObsPy Trace. Data should preferably be a masked array.

    Returns
    -------
    list[tuple[int, int]]
        List of `(start_index, end_index)` intervals, where `end_index` is exclusive.

    Notes
    -----
    If the trace has no masked samples, an empty list is returned.
    """
    data = trace.data
    if not np.ma.isMaskedArray(data):
        return []

    mask = np.ma.getmaskarray(data)
    if not mask.any():
        return []

    masked_idx = np.where(mask)[0]
    d = np.diff(masked_idx)

    gap_starts = np.where(np.insert(d, 0, 2) > 1)[0]
    gap_ends = np.where(np.append(d, 2) > 1)[0] - 1

    return [(masked_idx[s], masked_idx[e] + 1) for s, e in zip(gap_starts, gap_ends)]


def fill_gaps_with_constant(
    trace: Trace,
    *,
    fill_value: float = 0.0,
    inplace: bool = False,
) -> Trace:
    """
    Fill masked gaps in a trace with a constant value.

    Parameters
    ----------
    trace
        Input ObsPy Trace.
    fill_value
        Constant value used to fill masked samples.
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Trace with plain ndarray data and gaps filled by `fill_value`.
    """
    tr = trace if inplace else trace.copy()

    data = tr.data
    if np.ma.isMaskedArray(data):
        tr.data = np.asarray(data.filled(fill_value), dtype=np.float32)
    else:
        tr.data = np.asarray(data, dtype=np.float32)

    tr.stats.npts = len(tr.data)
    return tr


def fill_gaps_with_previous_value(
    trace: Trace,
    *,
    fallback_value: float = 0.0,
    inplace: bool = False,
) -> Trace:
    """
    Fill masked gaps using the previous valid sample value.

    Parameters
    ----------
    trace
        Input ObsPy Trace. Data should preferably be a masked array.
    fallback_value
        Value used when a gap occurs before any valid sample exists.
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Trace with plain ndarray data and gaps filled.

    Notes
    -----
    - If the trace is not masked, it is returned unchanged (except for dtype normalization).
    - Leading masked samples are filled with `fallback_value`.
    """
    tr = trace if inplace else trace.copy()

    data = tr.data
    if not np.ma.isMaskedArray(data):
        tr.data = np.asarray(data, dtype=np.float32)
        tr.stats.npts = len(tr.data)
        return tr

    filled = np.asarray(data.filled(np.nan), dtype=np.float32)
    mask = np.ma.getmaskarray(data)

    if not mask.any():
        tr.data = filled
        tr.stats.npts = len(tr.data)
        return tr

    last_value = fallback_value
    for i in range(len(filled)):
        if mask[i]:
            filled[i] = last_value
        else:
            if np.isfinite(filled[i]):
                last_value = filled[i]
            else:
                filled[i] = last_value

    tr.data = filled
    tr.stats.npts = len(tr.data)
    return tr


def _generate_spectrally_matched_noise(
    valid_data: np.ndarray,
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate approximate spectrally matched noise from a valid data segment.

    Parameters
    ----------
    valid_data
        1D array of valid data samples.
    n_samples
        Number of noise samples to generate.
    rng
        Optional NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Generated noise as float32.

    Notes
    -----
    This is a lightweight heuristic:
    - if there are too few valid samples, falls back to Gaussian noise
    - otherwise attempts to match the amplitude spectrum of the valid data
    """
    rng = np.random.default_rng() if rng is None else rng
    valid_data = np.asarray(valid_data, dtype=np.float32)

    if n_samples <= 0:
        return np.array([], dtype=np.float32)

    finite = np.isfinite(valid_data)
    valid_data = valid_data[finite]

    if valid_data.size < 8:
        scale = float(np.nanstd(valid_data)) if valid_data.size else 1.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        return rng.normal(0.0, scale, n_samples).astype(np.float32)

    # Remove mean
    valid_data = valid_data - np.nanmean(valid_data)

    # FFT amplitude from valid data
    spec = np.fft.rfft(valid_data)
    amp = np.abs(spec)

    # Random phases
    phase = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape)
    synth_spec = amp * np.exp(1j * phase)

    noise = np.fft.irfft(synth_spec, n=n_samples)

    # Match standard deviation roughly to valid data
    target_std = float(np.nanstd(valid_data))
    current_std = float(np.nanstd(noise))

    if np.isfinite(target_std) and target_std > 0 and np.isfinite(current_std) and current_std > 0:
        noise = noise * (target_std / current_std)

    return np.asarray(noise, dtype=np.float32)


def fill_gaps_with_filtered_noise(
    trace: Trace,
    *,
    taper_fraction: float = 0.1,
    fallback_value: float = 0.0,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
) -> Trace:
    """
    Fill masked gaps with approximate spectrally matched noise.

    Parameters
    ----------
    trace
        Input ObsPy Trace. Data should preferably be a masked array.
    taper_fraction
        Fraction of each gap length used for cosine tapering at gap edges.
        Should be between 0 and 0.5.
    fallback_value
        Value used if the trace has no valid samples at all.
    rng
        Optional NumPy random generator.
    inplace
        If True, modify the input trace in place.

    Returns
    -------
    Trace
        Trace with plain ndarray data and gaps filled.

    Notes
    -----
    - Uses the valid (unmasked) samples in the trace to estimate a rough spectrum.
    - Applies a light cosine taper to the generated noise inside each gap to reduce edge artifacts.
    - If no valid data exist, gaps are filled with `fallback_value`.
    """
    tr = trace if inplace else trace.copy()

    data = tr.data
    if not np.ma.isMaskedArray(data):
        tr.data = np.asarray(data, dtype=np.float32)
        tr.stats.npts = len(tr.data)
        return tr

    arr = np.asarray(data.filled(np.nan), dtype=np.float32)
    mask = np.ma.getmaskarray(data)

    if not mask.any():
        tr.data = arr
        tr.stats.npts = len(tr.data)
        return tr

    valid = arr[~mask]
    if valid.size == 0:
        tr.data = np.where(mask, fallback_value, arr).astype(np.float32)
        tr.stats.npts = len(tr.data)
        return tr

    out = arr.copy()
    gaps = classify_gaps(tr)

    taper_fraction = max(0.0, min(float(taper_fraction), 0.5))
    rng = np.random.default_rng() if rng is None else rng

    for start, end in gaps:
        n = end - start
        if n <= 0:
            continue

        noise = _generate_spectrally_matched_noise(valid, n, rng=rng)

        # Optional taper to reduce edge discontinuities
        n_taper = int(round(n * taper_fraction))
        if n_taper > 0 and n >= 2:
            taper = np.ones(n, dtype=np.float32)

            left = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_taper, dtype=np.float32)))
            right = left[::-1]

            taper[:n_taper] *= left
            taper[-n_taper:] *= right
            noise *= taper

        out[start:end] = noise

    # Keep any remaining non-finite values under control
    out = np.nan_to_num(out, nan=fallback_value, posinf=fallback_value, neginf=fallback_value).astype(np.float32)

    tr.data = out
    tr.stats.npts = len(tr.data)
    return tr

def _interp_short_gaps_only(tr: Trace, short_gap_sec: float) -> Trace:
    """
    Interpolate short masked gaps and keep long gaps masked.

    Parameters
    ----------
    tr
        Input trace with masked gaps.
    short_gap_sec
        Maximum gap duration (seconds) treated as short.

    Returns
    -------
    Trace
        Trace in which short gaps are interpolated and long gaps remain masked.
    """
    data = tr.data
    if not np.ma.isMaskedArray(data) or not np.any(np.ma.getmaskarray(data)):
        return tr.copy()

    sr = float(tr.stats.sampling_rate)
    short_gap_samples = max(1, int(round(short_gap_sec * sr)))

    tmp = smart_fill(
        tr,
        short_gap_threshold_samples=short_gap_samples,
        long_gap_method="constant",
        fill_value=np.nan,
        inplace=False,
    )

    arr = np.asarray(tmp.data, dtype=np.float32)
    mask = ~np.isfinite(arr)

    out = tr.copy()
    out.data = np.ma.masked_array(arr, mask=mask, copy=False)
    out.stats.npts = len(out.data)
    return out

def _next_pow2(n: int) -> int:
    """
    Return the next power of two greater than or equal to n.
    """
    n = int(n)
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()

def fill_stream_gaps(
    stream: Stream,
    *,
    method: str = "constant",
    inplace: bool = False,
    **kwargs,
) -> Stream:
    """
    Apply a gap-filling method to all traces in a Stream.

    Parameters
    ----------
    stream
        Input ObsPy Stream, preferably with masked gaps.
    method
        Gap-filling method. One of:
            - "constant"
            - "linear"
            - "previous"
            - "noise"
    inplace
        If True, modify the input stream in place.

    Returns
    -------
    Stream
        Stream with gaps filled trace by trace.
    """
    methods = {
        "constant": fill_gaps_with_constant,
        "linear": fill_gaps_with_linear_interpolation,
        "previous": fill_gaps_with_previous_value,
        "noise": fill_gaps_with_filtered_noise,
    }

    if method not in methods:
        raise ValueError(f"Unknown method {method!r}. Valid options: {list(methods)}")

    st = stream if inplace else stream.copy()

    out = Stream()
    for tr in st:
        out.append(methods[method](tr, inplace=False, **kwargs))

    if inplace:
        stream.clear()
        stream.extend(out)
        return stream

    return out