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
from obspy import Trace, Stream


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


def trace_equals(trace1: Trace, trace2: Trace, rtol=1e-5, atol=1e-8, sanitize=True) -> bool:
    """
    Compare two ObsPy Trace objects for equality, ignoring gaps and zeros.

    Parameters
    ----------
    trace1 : Trace
        First trace to compare.
    trace2 : Trace
        Second trace to compare.
    rtol : float
        Relative tolerance for floating-point comparison.
    atol : float
        Absolute tolerance for floating-point comparison.

    Returns
    -------
    bool
        True if traces are equal within tolerance (ignoring gaps), False otherwise.
    """
    if trace1.id != trace2.id:
        return False
    if abs(trace1.stats.starttime - trace2.stats.starttime) > trace1.stats.delta / 4:
        return False
    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        return False

    t1 = trace1.copy()
    t2 = trace2.copy()
    if sanitize:
        sanitize_trace(t1)
        sanitize_trace(t2)

    if len(t1.data) != len(t2.data):
        return False

    return np.allclose(
        t1.data.filled(np.nan),
        t2.data.filled(np.nan),
        rtol=rtol,
        atol=atol,
        equal_nan=True
    )

#from obspy import Stream, Trace
#from typing import Union

def streams_equal(stream1: Stream, stream2: Stream, rtol=1e-5, atol=1e-8, sanitize=True, sort=True) -> bool:
    """
    Compare two ObsPy Stream objects for equality.

    Parameters
    ----------
    stream1 : Stream
        First stream to compare.
    stream2 : Stream
        Second stream to compare.
    rtol : float
        Relative tolerance for floating-point comparison.
    atol : float
        Absolute tolerance for floating-point comparison.
    sanitize : bool
        Whether to sanitize each trace before comparison.
    sort : bool
        Whether to sort streams by trace.id before comparison.

    Returns
    -------
    bool
        True if all traces in both streams are equal, False otherwise.
    """
    if len(stream1) != len(stream2):
        return False

    s1 = stream1.copy()
    s2 = stream2.copy()

    if sort:
        s1.sort(keys=['id', 'starttime'])
        s2.sort(keys=['id', 'starttime'])

    for tr1, tr2 in zip(s1, s2):
        if not trace_equals(tr1, tr2, rtol=rtol, atol=atol, sanitize=sanitize):
            return False

    return True


def remove_empty_traces(stream, inplace=False):
    """
    Removes empty traces, traces full of zeros, and traces full of NaNs from an ObsPy Stream.

    Parameters
    ----------
    stream : obspy.Stream
        The input Stream object containing multiple seismic traces.
    inplace : bool, optional
        If True, modifies the original Stream in-place. If False (default), returns a cleaned copy.

    Returns
    -------
    obspy.Stream or None
        If inplace=False, returns a new Stream with only valid traces.
        If inplace=True, modifies the stream in-place and returns None.
    """
    if inplace:
        to_remove = [tr for tr in stream if _is_empty_trace(tr)]
        for tr in to_remove:
            stream.remove(tr)
        return None
    else:
        return Stream(tr for tr in stream if not _is_empty_trace(tr)) 
    

def _is_empty_trace(trace):
    """
    Determines whether a seismic trace is effectively empty.

    A trace is considered empty if:
    - It has zero data points (`npts == 0`).
    - All samples are identical (e.g., all zeros, all -1s).
    - All values are NaN.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to check.

    Returns:
    -------
    bool
        `True` if the trace is empty or contains only redundant values, otherwise `False`.

    Notes:
    ------
    - The function first checks if the trace has no data (`npts == 0`).
    - Then it checks if all values are identical (suggesting a completely flat signal).
    - Finally, it verifies if all values are NaN.

    Example:
    --------
    ```python
    from obspy import Trace
    import numpy as np

    # Create an empty trace
    empty_trace = Trace(data=np.array([]))
    print(_is_empty_trace(empty_trace))  # True

    # Create a flat-line trace
    flat_trace = Trace(data=np.zeros(1000))
    print(_is_empty_trace(flat_trace))  # True

    # Create a normal trace with random data
    normal_trace = Trace(data=np.random.randn(1000))
    print(_is_empty_trace(normal_trace))  # False
    ```
    """
    if trace.stats.npts == 0:
        return True
    
    # Check for flat trace (e.g. all zero, or all -1)
    if np.all(trace.data == np.nanmean(trace.data)):
        return True

    # Check if all values are NaN
    if np.all(np.isnan(trace.data)):
        return True 

    return False


def sanitize_trace(tr, unmask_short_zeros=True, min_gap_duration_s=1.0):
    """
    Cleans up a trace inplace by trimming, masking zeros/NaNs, and optionally unmasking short internal gaps.

    Parameters
    ----------
    tr : obspy.Trace
        Input trace.
    unmask_short_zeros : bool, optional
        If True, unmasks internal zero gaps shorter than `min_gap_duration_s`.
    min_gap_duration_s : float, optional
        Gaps shorter than this (in seconds) will be unmasked if `unmask_short_zeros=True`. Default is 1.0 s.


    """


    data = np.asarray(tr.data, dtype=np.float32)

    # Trim leading/trailing zeros
    nonzero = np.flatnonzero(data != 0.0)
    if nonzero.size == 0:
        tr.data = np.ma.masked_array([], mask=[])
        return

    start, end = nonzero[0], nonzero[-1] + 1
    data = data[start:end]
    tr.stats.starttime += start / tr.stats.sampling_rate

    # Mask all NaNs and zeros
    data = np.ma.masked_invalid(data)
    data = np.ma.masked_where(data == 0.0, data, copy=False)

    # Optionally unmask short internal zero gaps
    if unmask_short_zeros:
        sr = tr.stats.sampling_rate
        min_gap_samples = int(sr * min_gap_duration_s)
        masked = np.where(data.mask)[0]

        if masked.size:
            d = np.diff(masked)
            gap_starts = np.where(np.insert(d, 0, 2) > 1)[0]
            gap_ends = np.where(np.append(d, 2) > 1)[0] - 1

            for i in range(len(gap_starts)):
                s, e = masked[gap_starts[i]], masked[gap_ends[i]] + 1
                if (e - s) < min_gap_samples:
                    data.mask[s:e] = False  # Unmask short zero span

    tr.data = data




def sanitize_stream(stream, drop_empty=True, drop_duplicates=True, **kwargs):
    """
    Cleans and deduplicates an ObsPy Stream in-place.

    This function modifies the input stream directly by:
      - Sanitizing each trace in-place (via `sanitize_trace`)
      - Removing empty traces (if `drop_empty=True`)
      - Removing duplicate traces (if `drop_duplicates=True`)

    If you need to preserve the original stream, make a copy before calling.

    Parameters
    ----------
    stream : obspy.Stream
        Stream to sanitize. Will be modified in-place.
    drop_empty : bool, optional
        If True (default), remove traces with no usable data.
    drop_duplicates : bool, optional
        If True (default), remove exact duplicate traces based on ID and content.
    kwargs : dict
        Passed to `sanitize_trace` (e.g., unmask_short_zeros, min_gap_duration_s)

    Returns
    -------
    obspy.Stream
        The same modified stream object (for chaining or inspection).
    """
    # Sanitize all traces in-place
    for tr in stream:
        sanitize_trace(tr, **kwargs)

    # Remove empty traces
    if drop_empty:
        to_remove = [tr for tr in stream if _is_empty_trace(tr)]
        for tr in to_remove:
            stream.remove(tr)

    # Remove duplicates
    if drop_duplicates:
        deduped = Stream()
        for tr in stream:
            if not any(trace_equals(tr, other) for other in deduped):
                deduped.append(tr)
        stream.clear()
        stream.extend(deduped)