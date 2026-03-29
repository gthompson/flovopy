"""
trace_utils.py

Utility functions for working with ObsPy Trace objects in FLOVOpy.

Includes tools for:
- Ensuring data is in float32 format
- Enforcing masked arrays for data handling
- Comparing traces with tolerance, ignoring zeros/gaps

These helpers support robust preprocessing and testing workflows
when working with volcano-seismic time series data.

Author: [Your Name or FLOVOpy Development Team]
"""

import numpy as np
from obspy import Trace, Stream, UTCDateTime

def ensure_float32(tr: Trace) -> None:
    """
    Convert trace data to float32 if not already.

    Parameters
    ----------
    tr : Trace
        ObsPy Trace object to convert in-place.
    """
    if not np.issubdtype(tr.data.dtype, np.floating) or tr.data.dtype != np.float32:
        tr.data = tr.data.astype(np.float32)


def ensure_masked(trace: Trace) -> None:
    """
    Ensure the Trace data is a masked array (np.ma.MaskedArray).

    This allows for handling of missing/invalid values (e.g., gaps or artifacts)
    using mask-aware operations like merging, gap-filling, or comparison.

    Parameters
    ----------
    trace : Trace
        ObsPy Trace object to modify in-place.
    """
    if not np.ma.isMaskedArray(trace.data):
        trace.data = np.ma.masked_array(trace.data, mask=False)



def _is_empty_trace(trace: Trace) -> bool:
    """
    Return True if a Trace appears to contain no useful signal.

    Current criteria
    ----------------
    A trace is considered empty if any of the following hold:

    1. ``trace.stats.npts == 0``
    2. all samples are NaN
    3. all samples are equal to their mean value

    Notes
    -----
    Criterion (3) means a constant-valued trace is treated as empty,
    even if the constant is nonzero. This is useful for dropping flat,
    uninformative traces, but may be more aggressive than some workflows want.
    """
    if trace.stats.npts == 0:
        return True

    data = np.asanyarray(trace.data)

    if data.size == 0:
        return True

    if np.all(np.isnan(data)):
        return True

    try:
        return np.all(data == np.nanmean(data))
    except Exception:
        return False


def remove_empty_traces(stream: Stream, *, inplace: bool = True) -> Stream:
    """
    Remove traces that appear to contain no useful signal.

    Parameters
    ----------
    stream
        Input ObsPy Stream.
    inplace
        If True, modify the input stream in place.
        If False, operate on and return a copy.

    Returns
    -------
    Stream
        Stream with empty traces removed.
    """
    st = stream if inplace else stream.copy()

    for tr in list(st):
        if _is_empty_trace(tr):
            st.remove(tr)

    return st


def sanitize_trace(
    tr: Trace,
    *,
    unmask_short_zeros: bool = True,
    min_gap_duration_s: float = 1.0,
    inplace: bool = True,
) -> Trace:
    """
    Clean a trace by trimming leading/trailing zeros, masking zeros/NaNs,
    and optionally unmasking short internal zero gaps.

    Parameters
    ----------
    tr
        Input trace.
    unmask_short_zeros
        If True, unmask internal zero gaps shorter than `min_gap_duration_s`.
    min_gap_duration_s
        Maximum gap duration (seconds) to unmask when
        `unmask_short_zeros=True`.
    inplace
        If True, modify the trace in place. If False, return a cleaned copy.

    Returns
    -------
    Trace
        Sanitized trace.

    Notes
    -----
    Processing steps:
    1. Cast data to float32
    2. Trim leading/trailing zeros
    3. Shift starttime to match trimmed data
    4. Mask NaNs and zeros
    5. Optionally unmask short internal zero runs

    If the trace is entirely zero-valued, an empty masked trace is returned.
    """
    target = tr if inplace else tr.copy()

    data = np.asarray(target.data, dtype=np.float32)

    # Find nonzero extent
    nonzero = np.flatnonzero(data != 0.0)
    if nonzero.size == 0:
        target.data = np.ma.masked_array(np.array([], dtype=np.float32), mask=np.array([], dtype=bool))
        target.stats.npts = 0
        return target

    start, end = nonzero[0], nonzero[-1] + 1
    data = data[start:end]
    target.stats.starttime += start / target.stats.sampling_rate

    # Mask NaNs and zeros
    data = np.ma.masked_invalid(data)
    data = np.ma.masked_where(data == 0.0, data, copy=False)

    # Optionally unmask short internal zero gaps
    if unmask_short_zeros and data.size > 0:
        sr = float(target.stats.sampling_rate)
        min_gap_samples = max(1, int(round(sr * min_gap_duration_s)))

        mask = np.ma.getmaskarray(data)
        masked = np.where(mask)[0]

        if masked.size:
            d = np.diff(masked)
            gap_starts = np.where(np.insert(d, 0, 2) > 1)[0]
            gap_ends = np.where(np.append(d, 2) > 1)[0] - 1

            for i in range(len(gap_starts)):
                s = masked[gap_starts[i]]
                e = masked[gap_ends[i]] + 1
                if (e - s) < min_gap_samples:
                    data.mask[s:e] = False

    target.data = data
    target.stats.npts = len(target.data)
    return target


def sanitize_stream(
    stream: Stream,
    *,
    drop_empty: bool = True,
    drop_duplicates: bool = True,
    inplace: bool = True,
    **kwargs,
) -> Stream:
    """
    Sanitize all traces in a stream, with optional removal of empty and
    duplicate traces.

    Parameters
    ----------
    stream
        Input stream.
    drop_empty
        If True, remove traces with no usable data after sanitization.
    drop_duplicates
        If True, remove duplicate traces using `trace_equals()`.
    inplace
        If True, modify the stream in place. If False, return a cleaned copy.
    **kwargs
        Passed through to `sanitize_trace()`.

    Returns
    -------
    Stream
        Sanitized stream.
    """
    st = stream if inplace else stream.copy()

    # Sanitize each trace
    for tr in st:
        sanitize_trace(tr, inplace=True, **kwargs)

    # Remove empty traces
    if drop_empty:
        to_remove = [tr for tr in st if _is_empty_trace(tr)]
        for tr in to_remove:
            st.remove(tr)

    # Remove duplicates
    if drop_duplicates and len(st) > 1:
        deduped = Stream()
        for tr in st:
            if not any(trace_equals(tr, other) for other in deduped):
                deduped.append(tr)
        st.clear()
        st.extend(deduped)

    return st


def trace_diff_report(
    tr1: Trace,
    tr2: Trace,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    sanitize: bool = True,
    max_indices: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Compare two traces and report sample-level differences.

    This is a debugging/diagnostic helper intended for use when
    `trace_equals()` returns False and you want to inspect why.

    Parameters
    ----------
    tr1, tr2
        Traces to compare.
    rtol, atol
        Relative and absolute tolerances passed to ``numpy.isclose``.
    sanitize
        If True, apply ``sanitize_trace()`` to copies of both traces before
        comparison.
    max_indices
        Maximum number of differing indices to print in verbose mode.
    verbose
        If True, print a human-readable difference report.

    Returns
    -------
    dict
        Dictionary containing:
            - equal_length : bool
            - len1, len2 : int
            - trimmed_head_tail : str or None
            - n_differences : int
            - diff_indices : ndarray
            - a_values : ndarray
            - b_values : ndarray

    Notes
    -----
    If the traces differ in length by exactly one sample, this function tries
    head/tail alignment in the same spirit as ``trace_equals()``.
    """
    t1 = tr1.copy()
    t2 = tr2.copy()

    if sanitize:
        sanitize_trace(t1)
        sanitize_trace(t2)

    a = t1.data.filled(np.nan) if hasattr(t1.data, "filled") else np.asanyarray(t1.data)
    b = t2.data.filled(np.nan) if hasattr(t2.data, "filled") else np.asanyarray(t2.data)

    trim_mode = None

    if len(a) != len(b) and abs(len(a) - len(b)) == 1:
        if len(a) > len(b):
            if len(a) > 0 and len(b) > 0 and np.isclose(a[0], b[0], equal_nan=True):
                a = a[:len(b)]
                trim_mode = "trace1_tail"
            else:
                a = a[1:]
                trim_mode = "trace1_head"
        else:
            if len(a) > 0 and len(b) > 0 and np.isclose(a[0], b[0], equal_nan=True):
                b = b[:len(a)]
                trim_mode = "trace2_tail"
            else:
                b = b[1:]
                trim_mode = "trace2_head"

    equal_length = len(a) == len(b)

    if equal_length:
        diff_mask = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        diff_indices = np.where(diff_mask)[0]
        a_vals = a[diff_mask]
        b_vals = b[diff_mask]
    else:
        diff_indices = np.array([], dtype=int)
        a_vals = np.array([])
        b_vals = np.array([])

    report = {
        "equal_length": equal_length,
        "len1": len(tr1.data),
        "len2": len(tr2.data),
        "trimmed_head_tail": trim_mode,
        "n_differences": int(len(diff_indices)) if equal_length else None,
        "diff_indices": diff_indices,
        "a_values": a_vals,
        "b_values": b_vals,
    }

    if verbose:
        if tr1.id != tr2.id:
            print(f"trace_diff_report: id mismatch: {tr1.id} != {tr2.id}")

        if not equal_length:
            print(
                f"trace_diff_report: unequal lengths after optional 1-sample alignment: "
                f"{len(a)} vs {len(b)}"
            )
        else:
            if trim_mode is not None:
                print(f"trace_diff_report: applied 1-sample alignment mode: {trim_mode}")

            L = len(diff_indices)
            if L == 0:
                print("trace_diff_report: no differing samples")
            elif L < max_indices:
                print(f"trace_diff_report: {L} differing indices: {diff_indices}")
                print("a:", a_vals)
                print("b:", b_vals)
            else:
                print(f"trace_diff_report: {L} differing indices (too many to display)")

    return report


def trace_equals(
    trace1: Trace,
    trace2: Trace,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    sanitize: bool = True,
    starttime_tol_fraction: float = 0.25,
    verbose: bool = False,
) -> bool:
    """
    Compare two ObsPy Trace objects for effective equality.

    Parameters
    ----------
    trace1, trace2
        Traces to compare.
    rtol, atol
        Relative and absolute tolerances passed to ``numpy.allclose``.
    sanitize
        If True, apply ``sanitize_trace()`` to copies of both traces before
        comparison.
    starttime_tol_fraction
        Allowed fraction of one sample interval for starttime mismatch.
        Default is 0.25, meaning traces are considered aligned if their
        starttimes differ by no more than one quarter of ``delta``.
    verbose
        If True, print mismatch diagnostics.

    Returns
    -------
    bool
        True if traces are considered equal.

    Comparison criteria
    -------------------
    - trace id must match
    - starttimes must agree within tolerance
    - sampling rates must match exactly
    - data must match within tolerances
    - if lengths differ by exactly one sample, head/tail trimming is tried

    Notes
    -----
    Masked arrays are compared after converting masked values to NaN.
    """
    if trace1.id != trace2.id:
        if verbose:
            print(f"trace_equals: id mismatch: {trace1.id} != {trace2.id}")
            print(trace1, trace2)
        return False

    t1 = trace1.copy()
    t2 = trace2.copy()

    if sanitize:
        sanitize_trace(t1)
        sanitize_trace(t2)

    try:
        tol = t1.stats.delta * starttime_tol_fraction
    except Exception:
        tol = 0.0

    if abs(t1.stats.starttime - t2.stats.starttime) > tol:
        if verbose:
            print(
                f"trace_equals: starttime mismatch: "
                f"{t1.stats.starttime} != {t2.stats.starttime} "
                f"(tol={tol})"
            )
            print(t1, t2)
        return False

    if t1.stats.sampling_rate != t2.stats.sampling_rate:
        if verbose:
            print(
                f"trace_equals: sampling rate mismatch: "
                f"{t1.stats.sampling_rate} != {t2.stats.sampling_rate}"
            )
            print(t1, t2)
        return False

    a = t1.data.filled(np.nan) if hasattr(t1.data, "filled") else np.asanyarray(t1.data)
    b = t2.data.filled(np.nan) if hasattr(t2.data, "filled") else np.asanyarray(t2.data)

    len_diff = len(a) - len(b)

    if len_diff == 0:
        ok = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if not ok and verbose:
            print(f"trace_equals: equal lengths but data differ ({len(a)} samples)")
            trace_diff_report(
                t1, t2,
                rtol=rtol,
                atol=atol,
                sanitize=False,
                verbose=True,
            )
        return ok

    if abs(len_diff) == 1:
        if len_diff == 1:
            a_trim = a[:len(b)]
            if np.allclose(a_trim, b, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("trace_equals: trace1 has 1 extra tail sample")
                return True

            a_trim = a[1:]
            if np.allclose(a_trim, b, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("trace_equals: trace1 has 1 extra head sample")
                return True

        else:
            b_trim = b[:len(a)]
            if np.allclose(a, b_trim, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("trace_equals: trace2 has 1 extra tail sample")
                return True

            b_trim = b[1:]
            if np.allclose(a, b_trim, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("trace_equals: trace2 has 1 extra head sample")
                return True

    if verbose:
        print(f"trace_equals: length mismatch or content differs: {len(a)} vs {len(b)}")
        trace_diff_report(
            t1, t2,
            rtol=rtol,
            atol=atol,
            sanitize=False,
            verbose=True,
        )

    return False


def streams_equal(
    stream1: Stream,
    stream2: Stream,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    sanitize: bool = True,
    sort_by_id: bool = True,
    verbose: bool = False,
) -> bool:
    """
    Compare two ObsPy Stream objects for effective equality.

    Parameters
    ----------
    stream1, stream2
        Streams to compare.
    rtol, atol
        Relative and absolute tolerances passed to ``trace_equals``.
    sanitize
        If True, sanitize traces before comparison.
    sort_by_id
        If True, compare streams after sorting by trace id and starttime.
        If False, compare traces in current order.
    verbose
        If True, print diagnostics for mismatches.

    Returns
    -------
    bool
        True if all traces match, False otherwise.

    Notes
    -----
    This first checks stream length, then compares traces pairwise using
    ``trace_equals()``.
    """
    if len(stream1) != len(stream2):
        if verbose:
            print(f"streams_equal: stream length mismatch: {len(stream1)} != {len(stream2)}")
        return False

    s1 = stream1.copy()
    s2 = stream2.copy()

    if sort_by_id:
        s1.sort(keys=["id", "starttime"])
        s2.sort(keys=["id", "starttime"])

    all_equal = True

    for i, (tr1, tr2) in enumerate(zip(s1, s2)):
        ok = trace_equals(
            tr1,
            tr2,
            rtol=rtol,
            atol=atol,
            sanitize=sanitize,
            verbose=verbose,
        )
        if not ok:
            all_equal = False
            if verbose:
                print(f"streams_equal: mismatch at trace index {i}")
                print(tr1)
                print(tr2)
            else:
                break

    return all_equal



def summarize_stream(st):
    """
    Summarizes key metadata for an ObsPy Stream.

    Parameters
    ----------
    st : obspy.Stream

    Returns
    -------
    dict
        Summary dictionary with counts, stats, and unique trace IDs.
    """
    return {
        'ntraces': len(st),
        'trace_ids': list({tr.id for tr in st}),
        'starttime': str(min(tr.stats.starttime for tr in st)) if len(st) else None,
        'endtime': str(max(tr.stats.endtime for tr in st)) if len(st) else None,
        'stations': list({tr.stats.station for tr in st}),
        'networks': list({tr.stats.network for tr in st}),
        'sampling_rates': list({tr.stats.sampling_rate for tr in st})
    }

from math import isclose
import numpy as np
from obspy import Stream, Trace


def get_min_sampling_rate(
    st: Stream,
    *,
    skip_low_rate_channels: bool = True,
) -> float:
    """
    Return the minimum sampling rate in a Stream.

    Parameters
    ----------
    st
        Input stream.
    skip_low_rate_channels
        If True, exclude traces whose channel code starts with 'L'
        (commonly long-period channels).

    Returns
    -------
    float
        Minimum sampling rate among included traces.

    Raises
    ------
    ValueError
        If the stream is empty or no traces remain after filtering.
    """
    rates = []

    for tr in st:
        try:
            chan = tr.stats.channel
            sr = float(tr.stats.sampling_rate)
        except Exception:
            continue

        if skip_low_rate_channels and chan.startswith("L"):
            continue

        rates.append(sr)

    if not rates:
        raise ValueError("No valid traces found for sampling-rate calculation")

    return min(rates)


def downsample_trace(
    tr: Trace,
    target_rate: float,
    *,
    inplace: bool = True,
    verbose: bool = False,
) -> Trace:
    """
    Downsample a trace to a target sampling rate.

    Parameters
    ----------
    tr
        Input trace.
    target_rate
        Desired output sampling rate.
    inplace
        If True, modify trace in place.
    verbose
        If True, print diagnostics.

    Returns
    -------
    Trace
        Downsampled trace.

    Raises
    ------
    ValueError
        If upsampling would be required.
    """
    sr = float(tr.stats.sampling_rate)

    if isclose(sr, target_rate, rel_tol=0.0, abs_tol=1e-6):
        return tr if inplace else tr.copy()

    if sr < target_rate:
        raise ValueError(f"Upsampling not supported: {sr} Hz → {target_rate} Hz")

    target = tr if inplace else tr.copy()

    # Prefer integer decimation when exact
    ratio = sr / target_rate
    rounded = int(round(ratio))

    try:
        if rounded > 1 and isclose(sr / rounded, target_rate, rel_tol=0.0, abs_tol=1e-6):
            if verbose:
                print(f"Downsampling {tr.id}: decimate by factor {rounded}")
            target.decimate(factor=rounded, no_filter=False)
            proc = list(getattr(target.stats, "processing", []) or [])
            proc.append(f"flovopy:decimate factor={rounded} to {target.stats.sampling_rate} Hz")
            target.stats.processing = proc
        else:
            if verbose:
                print(f"Downsampling {tr.id}: resample {sr} Hz → {target_rate} Hz")
            target.resample(sampling_rate=target_rate)
            proc = list(getattr(target.stats, "processing", []) or [])
            proc.append(f"flovopy:resample to {target.stats.sampling_rate} Hz")
            target.stats.processing = proc

    except Exception as e:
        raise RuntimeError(f"Failed to downsample {tr.id} from {sr} Hz to {target_rate} Hz: {e}")

    return target


def downsample_stream_to_common_rate(
    st: Stream,
    *,
    inplace: bool = True,
    max_sampling_rate: float | None = None,
    skip_low_rate_channels: bool = True,
    verbose: bool = False,
) -> Stream:
    """
    Downsample all traces in a Stream to a common sampling rate.

    The target rate is chosen as:
    - the minimum sampling rate among included traces
    - optionally capped by `max_sampling_rate`

    Traces with sampling rates below the target are excluded rather than
    upsampled.

    Parameters
    ----------
    st
        Input stream.
    inplace
        If True, modify the input stream in place.
    max_sampling_rate
        Optional upper bound on the target sampling rate.
    skip_low_rate_channels
        If True, exclude traces whose channel code starts with 'L' when
        determining the target rate.
    verbose
        If True, print diagnostics.

    Returns
    -------
    Stream
        Stream whose traces all share a common sampling rate.
    """
    if len(st) == 0:
        return st if inplace else st.copy()

    target_stream = st if inplace else st.copy()

    try:
        min_rate = get_min_sampling_rate(
            target_stream,
            skip_low_rate_channels=skip_low_rate_channels,
        )
    except ValueError:
        return Stream() if not inplace else target_stream.__class__()

    target_rate = min_rate
    if max_sampling_rate is not None:
        target_rate = min(min_rate, float(max_sampling_rate))

    traces_out = []

    for tr in target_stream:
        try:
            sr = float(tr.stats.sampling_rate)

            if sr < target_rate:
                if verbose:
                    print(f"⚠️ Skipping {tr.id}: sampling rate too low ({sr} Hz < {target_rate} Hz)")
                continue

            tr_ds = downsample_trace(
                tr,
                target_rate,
                inplace=inplace,
                verbose=verbose,
            )
            traces_out.append(tr_ds)

        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to downsample {tr.id}: {e}")

    if inplace:
        target_stream._traces = traces_out
        return target_stream

    return Stream(traces_out)

#######################################################################
##               Fixing IDs                                          ##
#######################################################################

BAND_CODE_TABLE = {
    (0.0001, 0.001): "R",   # Extremely Long Period
    (0.001, 0.01): "U",     # Ultra Low Frequency
    (0.01, 0.1): "V",       # Very Low Frequency
    (0.1, 2): "L",          # Long Period
    (2, 10): "M",           # Mid Period
    (10, 80): "B",          # Broadband
    (80, 250): "H",         # High Frequency
    (250, 1000): "D",       # Very High Frequency
    (1000, 5000): "G",      # Extremely High Frequency
}


def fix_id_wrapper(tr):
    """
    Apply `fix_trace_id()` and return (old_id, new_id).
    """
    source_id = tr.id
    fix_trace_id(tr)
    return source_id, tr.id


def _get_band_code(sampling_rate):
    """
    Return the SEED band code implied by a sampling rate.

    Parameters
    ----------
    sampling_rate : float
        Trace sampling rate in Hz.

    Returns
    -------
    str or None
        SEED band code, or None if no range matches.
    """
    for (low, high), code in BAND_CODE_TABLE.items():
        if low <= sampling_rate < high:
            return code
    return None


def _adjust_band_code_for_sensor_type(current_band_code, expected_band_code, short_period=False):
    """
    Adjust a computed band code for short-period sensors.

    Parameters
    ----------
    current_band_code : str
        Existing band code from the channel name.
    expected_band_code : str
        Band code inferred from sampling rate.
    short_period : bool, optional
        If True, force short-period mapping.

    Returns
    -------
    str
        Adjusted band code.
    """
    short_period_codes = {"S", "E", "C", "F"}

    if current_band_code in short_period_codes or short_period:
        band_code_mapping = {"B": "S", "H": "E", "D": "C", "G": "F"}
        return band_code_mapping.get(expected_band_code, expected_band_code)

    return expected_band_code


def fix_trace_id(trace, legacy=False, netcode=None, verbose=False):
    """
    Apply generic SEED/NSLC normalization to a Trace.

    This function is intentionally generic. Observatory- or project-specific
    ID fixers (e.g. MVO, KSC) should be applied explicitly by caller code.

    Parameters
    ----------
    trace : obspy.Trace
        Trace to modify in place.
    legacy : bool, optional
        If True, call the VDAP/legacy ID decoder before generic cleanup.
    netcode : str, optional
        Network code to assign if the trace has no network code.
    verbose : bool, optional
        If True, print diagnostics.

    Returns
    -------
    bool
        True if the trace ID changed, False otherwise.
    """
    changed = False
    old_id = trace.id

    if verbose:
        print(f"Initial ID: {old_id}")

    if legacy:
        from flovopy.core.vdap import fix_legacy_vdap_id
        fix_legacy_vdap_id(trace, network=netcode)

    if not trace.stats.network and netcode:
        trace.stats.network = netcode

    sampling_rate = trace.stats.sampling_rate
    chan = trace.stats.channel or ""

    # Generic cleanup only
    if chan and not chan.startswith("A"):
        trace.stats.channel = fix_channel_code(chan, sampling_rate)

    trace.stats.location = fix_location_code(trace.stats.location)

    if trace.id != old_id:
        changed = True
        if verbose:
            print(f"Updated ID: {trace.id} (was {old_id}) based on fs={sampling_rate}")

    return changed


def fix_location_code(loc):
    """
    Normalize a SEED location code.

    Rules
    -----
    - None or '' -> ''
    - '-' -> '--'
    - numeric strings are zero-padded to length 2
    - longer strings are truncated to 2 characters

    Parameters
    ----------
    loc : str or None

    Returns
    -------
    str
        Normalized location code.
    """
    loc = (loc or "").strip().upper()
    if not loc:
        return ""
    if loc == "-":
        return "--"
    if loc.isdigit():
        return loc.zfill(2)
    return loc[:2]


def fix_channel_code(chan, sampling_rate, short_period=False):
    """
    Normalize the band code of a SEED channel using sampling rate.

    Parameters
    ----------
    chan : str
        Input channel code.
    sampling_rate : float
        Sampling rate in Hz.
    short_period : bool, optional
        If True, prefer short-period band-code mappings.

    Returns
    -------
    str
        Channel code with updated band code.
    """
    chan = (chan or "").strip().upper()
    if not chan:
        return chan

    current_band_code = chan[0]
    expected_band_code = _get_band_code(sampling_rate)

    if expected_band_code is None:
        return chan

    expected_band_code = _adjust_band_code_for_sensor_type(
        current_band_code,
        expected_band_code,
        short_period=short_period,
    )
    return expected_band_code + chan[1:]


def decompose_channel_code(chan):
    """
    Decompose a SEED channel code into band, instrument, and orientation.

    Parameters
    ----------
    chan : str
        SEED channel code (e.g. 'BHZ', 'BZ', 'BH', 'B').

    Returns
    -------
    tuple
        (bandcode, instrumentcode, orientationcode)
    """
    chan = (chan or "").strip().upper()

    if len(chan) == 3:
        return chan[0], chan[1], chan[2]
    if len(chan) == 2:
        if chan[1] in "ZNE":
            return chan[0], "H", chan[1]
        if chan[1] in "HDL":
            return chan[0], chan[1], "Z"
        return chan[0], "", ""
    if len(chan) == 1:
        return chan[0], "", ""
    return "", "", ""


def print_nslc_tree(nslc_or_seed_list):
    """
    Print a directory-like structure of NSLC codes in a hierarchical tree format.

    Accepts either:
    - A list of 4-tuples: (network, station, location, channel), OR
    - A list of SEED IDs like "NET.STA.LOC.CHA" or "NET.STA.CHA"

    Parameters:
    -----------
    nslc_or_seed_list : list of tuple or str
        The NSLC or SEED-style identifiers to display.
    """
    tree = {}

    for item in nslc_or_seed_list:
        # Convert SEED ID strings to 4-tuples
        if isinstance(item, str):
            parts = item.strip().split(".")
            if len(parts) == 3:
                net, sta, cha = parts
                loc = ""
            elif len(parts) == 4:
                net, sta, loc, cha = parts
            else:
                print(f"[WARN] Skipping invalid SEED ID: {item}")
                continue
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            net, sta, loc, cha = item
        else:
            print(f"[WARN] Skipping invalid item: {item}")
            continue

        # Build the nested tree
        tree.setdefault(net, {}).setdefault(sta, {}).setdefault(loc, []).append(cha)

    def _print_branch(branch, indent=""):
        for key, value in sorted(branch.items()):
            if isinstance(value, dict):
                print(f"{indent}{key}/")
                _print_branch(value, indent + "    ")
            else:
                for cha in sorted(value):
                    print(f"{indent}{key}/{cha}")

    _print_branch(tree)

# -------------------------------------------------------------------
# Stream summary / processing helpers
# -------------------------------------------------------------------

def stream_time_bounds(st: Stream) -> tuple[UTCDateTime, UTCDateTime, UTCDateTime, UTCDateTime]:
    """
    Return time bounds for a Stream.

    Parameters
    ----------
    st
        Input ObsPy Stream.

    Returns
    -------
    tuple
        (min_starttime, max_starttime, min_endtime, max_endtime)

    Raises
    ------
    ValueError
        If the stream is empty.
    """
    if len(st) == 0:
        raise ValueError("Empty stream")

    starttimes = [tr.stats.starttime for tr in st]
    endtimes = [tr.stats.endtime for tr in st]
    return min(starttimes), max(starttimes), min(endtimes), max(endtimes)


def Stream_min_starttime(all_traces: Stream):
    """
    Backward-compatible wrapper for `stream_time_bounds()`.

    Returns
    -------
    tuple
        (min_starttime, max_starttime, min_endtime, max_endtime)
    """
    return stream_time_bounds(all_traces)


# -------------------------------
# Processing markers (use stats.processing)
# -------------------------------

def add_processing_step(tr: Trace, msg: str) -> None:
    """
    Append a processing-history message to ``trace.stats.processing``.

    Parameters
    ----------
    tr
        Trace to annotate.
    msg
        Message to append.
    """
    if not hasattr(tr.stats, "processing") or tr.stats.processing is None:
        tr.stats.processing = []
    tr.stats.processing.append(str(msg))


def trace_ids_from_stream(st: Stream) -> tuple[str, ...]:
    """
    Return sorted, unique trace IDs from a Stream.

    Parameters
    ----------
    st
        Input stream.

    Returns
    -------
    tuple[str, ...]
        Sorted unique ``trace.id`` values.
    """
    return tuple(sorted({tr.id for tr in st}))


def station_ids_from_stream(st: Stream) -> tuple[str, ...]:
    """
    Return sorted, unique station identifiers from a Stream.

    Parameters
    ----------
    st
        Input stream.

    Returns
    -------
    tuple[str, ...]
        Sorted unique ``NET.STA`` identifiers.
    """
    return tuple(
        sorted(
            {
                f"{getattr(tr.stats, 'network', '')}.{getattr(tr.stats, 'station', '')}"
                for tr in st
            }
        )
    )


# -------------------------------------------------------------------
# Helper functions for machine-learning workflows
# -------------------------------------------------------------------

def choose_best_traces(
    st: Stream,
    max_traces: int = 8,
    include_seismic: bool = True,
    include_infrasound: bool = False,
    include_uncorrected: bool = False,
) -> np.ndarray:
    """
    Return indices of the highest-priority traces in a Stream.

    Priority is based primarily on ``trace.stats.quality_factor`` and is
    modified according to channel type and unit metadata.

    Rules
    -----
    - Seismic channels (second character 'H') are kept only if
      ``include_seismic=True``.
    - Infrasound channels (second character 'D') are kept only if
      ``include_infrasound=True``.
    - Vertical seismic channels ('?HZ') receive an extra priority boost.
    - If ``include_uncorrected=False``, traces whose units are missing or
      equal to ``"Counts"`` are excluded.

    Parameters
    ----------
    st
        Input stream.
    max_traces
        Maximum number of traces to select.
    include_seismic
        Whether to consider seismic channels.
    include_infrasound
        Whether to consider infrasound channels.
    include_uncorrected
        Whether to allow traces lacking physical-unit corrections.

    Returns
    -------
    numpy.ndarray
        Array of selected trace indices.
    """
    if len(st) == 0:
        return np.array([], dtype=int)

    priority = np.array(
        [float(getattr(tr.stats, "quality_factor", 0.0)) for tr in st],
        dtype=float,
    )

    for i, tr in enumerate(st):
        chan = (getattr(tr.stats, "channel", "") or "").upper()
        units = getattr(tr.stats, "units", None)

        if len(chan) >= 2 and chan[1] == "H":
            if include_seismic:
                if len(chan) >= 3 and chan[2] == "Z":
                    priority[i] *= 2.0
            else:
                priority[i] = 0.0

        elif len(chan) >= 2 and chan[1] == "D":
            if include_infrasound:
                priority[i] *= 2.0
            else:
                priority[i] = 0.0

        if not include_uncorrected:
            if units is None or units == "Counts":
                priority[i] = 0.0

    n = min(np.count_nonzero(priority > 0.0), max_traces)
    if n == 0:
        return np.array([], dtype=int)

    return np.argsort(priority)[-n:]


def select_by_index_list(st: Stream, chosen) -> Stream:
    """
    Return a new Stream containing traces at the selected indices.

    Parameters
    ----------
    st
        Input stream.
    chosen
        Iterable of integer indices to keep.

    Returns
    -------
    Stream
        Selected traces.
    """
    chosen = set(chosen)
    return Stream([tr for i, tr in enumerate(st) if i in chosen])


def stream_add_units(st: Stream, default_units: str = "m/s") -> None:
    """
    Heuristically assign ``trace.stats.units`` where missing.

    Rules
    -----
    - If units already exist, leave them unchanged.
    - If data are empty, assign ``"Counts"``.
    - If median absolute amplitude is < 1, assume the data are already in
      physical units and assign ``default_units``.
    - Otherwise assign ``"Counts"``.

    Parameters
    ----------
    st
        Input stream, modified in place.
    default_units
        Physical units to assign when amplitudes appear already corrected.
    """
    for tr in st:
        current_units = getattr(tr.stats, "units", None)
        if current_units:
            continue

        data = tr.data
        if data is None or len(data) == 0:
            tr.stats["units"] = "Counts"
            continue

        median_abs = float(np.nanmedian(np.abs(data)))
        if np.isfinite(median_abs) and median_abs < 1.0:
            tr.stats["units"] = default_units
        else:
            tr.stats["units"] = "Counts"