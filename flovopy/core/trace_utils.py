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


def trace_equals(trace1: Trace, trace2: Trace, rtol=1e-5, atol=1e-8, sanitize=True, verbose=False) -> bool:
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
        if verbose:
            print(f'different lengths {len(tr.data)}, {len(tr.data)}')
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

def streams_equal(stream1: Stream, stream2: Stream, rtol=1e-5, atol=1e-8, sanitize=True, sort=True, verbose=False) -> bool:
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
        if verbose:
            print(f'Different number of traces {len(stream1)}, {len(stream2)}')
        return False

    s1 = stream1.copy()
    s2 = stream2.copy()

    if sort:
        s1.sort()
        s2.sort()

    for tr1, tr2 in zip(s1, s2):
        if not trace_equals(tr1, tr2, rtol=rtol, atol=atol, sanitize=sanitize, verbose=verbose):
            if verbose:
                trace_diff(tr1, tr2)
            else:
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


def trace_diff(tr1, tr2):

    a = tr1.data
    b = tr2.data

    if len(a) != len(b):
        if abs(len(a)-len(b))==1:
            if len(a)>len(b):
                if a[0]==b[0]:
                    a = a[:len(b)]
                else:
                    a = a[1:]
                print('trace1 has 1 extra sample') 
            else:
                if a[0]==b[0]:
                    b = b[:len(a)]
                else:
                    b = b[1:]  
                print('trace2 has 1 extra sample')              

    # Create a mask of differences
    #diff_mask = a != b
    diff_mask = ~(np.isclose(a, b, equal_nan=True))


    # Show indices where they differ
    diff_indices = np.where(diff_mask)[0]
    L = len(diff_indices)
    if L>0:
        if L<100:
            print(f"{L} different indices: {diff_indices}")


            # Show values that differ
            print("a:", a[diff_mask])
            print("b:", b[diff_mask])
        else:
            print(f"{L} different indices")


def trace_equals(trace1: Trace, trace2: Trace, rtol=1e-5, atol=1e-8,
                 sanitize=True, verbose=False) -> bool:
    """
    Compare two ObsPy Trace objects for equality, with extra handling for small mismatches.

    Parameters
    ----------
    trace1 : Trace
    trace2 : Trace
    rtol : float : Relative tolerance for np.isclose
    atol : float : Absolute tolerance for np.isclose
    sanitize : bool : Whether to apply sanitize_trace() to both traces
    verbose : bool : Print detailed differences if found

    Returns
    -------
    bool : True if equal (including 1-sample tolerance), False otherwise
    """
    if trace1.id != trace2.id:
        if verbose:
            print(f"ID mismatch: {trace1.id} != {trace2.id}")
            print(trace1, trace2)
        return False



    t1 = trace1.copy()
    t2 = trace2.copy()
    if sanitize:
        sanitize_trace(t1)
        sanitize_trace(t2)

    if abs(t1.stats.starttime - t2.stats.starttime) > t1.stats.delta / 4:
        if verbose:
            print("Start time mismatch.")
            print(t1, t2)
        return False

    if t1.stats.sampling_rate != t2.stats.sampling_rate:
        if verbose:
            print("Sampling rate mismatch.")
            print(t1, t2)
        return False

    a = t1.data.filled(np.nan) if hasattr(t1.data, 'filled') else t1.data
    b = t2.data.filled(np.nan) if hasattr(t2.data, 'filled') else t2.data

    len_diff = len(a) - len(b)
    if len_diff == 0:
        if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            return True
    elif abs(len_diff) == 1:
        # Attempt to align and compare if only one sample is different
        if len_diff == 1:
            a_trim = a[:len(b)]
            if np.allclose(a_trim, b, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("Trace1 has 1 extra sample (tail trimmed).")
                return True
            a_trim = a[1:]
            if np.allclose(a_trim, b, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("Trace1 has 1 extra sample (head trimmed).")
                return True
        else:
            b_trim = b[:len(a)]
            if np.allclose(a, b_trim, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("Trace2 has 1 extra sample (tail trimmed).")
                return True
            b_trim = b[1:]
            if np.allclose(a, b_trim, rtol=rtol, atol=atol, equal_nan=True):
                if verbose:
                    print("Trace2 has 1 extra sample (head trimmed).")
                return True

    if verbose:
        print(f"Length mismatch or content differs: {len(a)} vs {len(b)}")
        # Compute differences
        min_len = min(len(a), len(b))
        diff_mask = ~np.isclose(a[:min_len], b[:min_len], rtol=rtol, atol=atol, equal_nan=True)
        diff_indices = np.where(diff_mask)[0]
        L = len(diff_indices)
        print(f"{L} differing sample(s).")
        if L > 0:
            if L < 100:
                print(f"Different indices: {diff_indices}")
                print("a:", a[diff_mask])
                print("b:", b[diff_mask])
            else:
                print(f"Too many differing samples to display ({L})")

    return False



def streams_equal(stream1: Stream, stream2: Stream, rtol=1e-5, atol=1e-8,
                  sanitize=True, sort=True, verbose=False) -> bool:
    """
    Compare two ObsPy Stream objects for equality.

    Parameters
    ----------
    stream1 : Stream
    stream2 : Stream
    rtol : float : Relative tolerance for floating-point comparison
    atol : float : Absolute tolerance for floating-point comparison
    sanitize : bool : Sanitize traces before comparison
    sort : bool : Sort streams by trace ID before comparing
    verbose : bool : Print differences if found

    Returns
    -------
    bool : True if all traces match, False otherwise
    """
    if len(stream1) != len(stream2):
        if verbose:
            print(f"Stream length mismatch: {len(stream1)} != {len(stream2)}")
        return False

    s1 = stream1.copy()
    s2 = stream2.copy()

    if sort:
        s1.sort()
        s2.sort()

    all_equal = True

    for tr1, tr2 in zip(s1, s2):
        if not trace_equals(tr1, tr2, rtol=rtol, atol=atol, sanitize=sanitize, verbose=verbose):
            all_equal = False
            if not verbose:
                break  # stop early only if not in verbose mode

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

def split_trace_at_midnight(tr):
    """
    Split a Trace at UTC midnight boundaries. Return list of Trace objects.
    """
    out = []
    t1 = tr.stats.starttime
    t2 = tr.stats.endtime

    while t1 < t2:
        next_midnight = UTCDateTime(t1.date) + 86400
        trim_end = min(t2, next_midnight)
        tr_piece = tr.copy().trim(starttime=t1, endtime=trim_end, nearest_sample=True)
        out.append(tr_piece)
        t1 = trim_end
    return out


def decimate(tr, max_sampling_rate=250.0):
    if tr.stats.sampling_rate > max_sampling_rate:
        try:
            factor = int(ceil(tr.stats.sampling_rate / max_sampling_rate))
            if factor > 1:
                tr.decimate(factor)  # Applies low-pass filter internally
                tr.stats.processing.append(f"Decimated by factor {factor} to {tr.stats.sampling_rate} Hz")
            else:
                tr.stats.processing.append("Sampling rate OK, no decimation needed")
        except Exception as e:
            tr.stats.processing.append(f"Decimation failed: {e}")

def get_min_sampling_rate(st):
    """
    Return the minimum sampling rate in a Stream, excluding traces
    whose channel code starts with 'L' (typically long-period).

    Parameters
    ----------
    st : obspy.Stream
        The input stream.

    Returns
    -------
    float
        The minimum sampling rate among the remaining traces.

    Raises
    ------
    ValueError
        If the stream is empty or all traces are excluded.
    """
    filtered = [tr.stats.sampling_rate for tr in st if not tr.stats.channel.startswith('L')]

    if not filtered:
        raise ValueError("No valid traces found (all filtered out)")

    return min(filtered)



def downsample_trace(tr, target_rate, inplace=True):
    sr = tr.stats.sampling_rate
    if sr == target_rate:
        return tr

    target = tr if inplace else tr.copy()

    if sr > target_rate:
        factor = int(sr // target_rate)
        print(f'trying to downsample by factor {factor} for {tr} where target_rate={target_rate}')
        if sr / factor == target_rate:
            
            target.decimate(factor=factor, no_filter=False)
        else:
            target.resample(sampling_rate=target_rate)
        return target
    else:
        raise ValueError(f"Upsampling not supported: {sr} Hz → {target_rate} Hz")
    
def downsample_stream_to_common_rate(st, inplace=True, max_sampling_rate=None):
    """
    Downsamples all traces in a Stream to a uniform target sampling rate.

    If max_sampling_rate is provided, the target rate is the minimum of the 
    lowest sampling rate in the stream and max_sampling_rate.

    Traces with lower-than-target sampling rate are excluded.

    Parameters
    ----------
    st : obspy.Stream
        Stream with potentially mixed sampling rates.
    inplace : bool
        If True, modifies the stream in place and returns it.
    max_sampling_rate : float or None
        Optional ceiling on target sampling rate.

    Returns
    -------
    obspy.Stream
        Stream with all traces downsampled to a uniform sampling rate.
    """
    if len(st) == 0:
        return st

    # Compute target sampling rate
    min_rate = get_min_sampling_rate(st)
    target_rate = min(min_rate, max_sampling_rate) if max_sampling_rate else min_rate

    traces_out = []
    for tr in st:
        try:
            if tr.stats.sampling_rate >= target_rate:
                tr_ds = downsample_trace(tr, target_rate, inplace=inplace)
                traces_out.append(tr_ds)
            else:
                print(f"⚠️ Skipping {tr.id} — sampling rate too low ({tr.stats.sampling_rate} Hz)")
        except Exception as e:
            print(f"⚠️ Failed to downsample {tr.id}: {e}")

    if inplace:
        st._traces = traces_out
        return st
    else:
        return Stream(traces_out)

#######################################################################
##               Fixing IDs                                          ##
#######################################################################

def fix_id_wrapper(tr):
    source_id = tr.id
    fix_trace_id(tr)
    return source_id, tr.id


def _get_band_code(sampling_rate):
    """
    Determines the appropriate band code based on the sampling rate.

    The band code is the first letter of the **SEED channel naming convention**, which 
    categorizes seismic channels based on frequency range.

    Parameters:
    ----------
    sampling_rate : float
        The sampling rate of the seismic trace in Hz.

    Returns:
    -------
    str or None
        The appropriate band code (e.g., 'B' for broadband, 'H' for high-frequency broadband).
        Returns `None` if no matching band code is found (should not happen if lookup table is correct).

    Notes:
    ------
    - This function relies on `BAND_CODE_TABLE`, a dictionary defining the mapping 
      between frequency ranges and SEED band codes.

    Example:
    --------
    ```python
    band_code = _get_band_code(100.0)
    print(band_code)  # Output: 'H' (High-frequency broadband)
    ```
    """
    # Band code lookup table based on IRIS SEED convention
    BAND_CODE_TABLE = {
        (0.0001, 0.001): "R",  # Extremely Long Period (0.0001 - 0.001 Hz)   
        (0.001, 0.01): "U",  # Ultra Low Frequency (~0.01 Hz)
        (0.01, 0.1): "V",  # Very Low Frequency (~0.1 Hz)
        (0.1, 2): "L",   # Long Period (~1 Hz)
        (2, 10): "M",  # Mid Period (1 - 10 Hz)
        (10, 80): "B", # Broadband (S if Short Period instrument, corner > 0.1 Hz)
        (80, 250): "H",  # High Frequency (80 - 250 Hz) (E if Short Period instrument, corner > 0.1 Hz)
        (250, 1000): "D",  # Very High Frequency (250 - 1000 Hz) (C if Short Period instrument, corner > 0.1 Hz)
        (1000, 5000): "G",  # Extremely High Frequency (1 - 5 kHz) (F if Short period)
    }

    for (low, high), code in BAND_CODE_TABLE.items():
        if low <= sampling_rate < high:
            return code
    return None  # Should not happen if lookup table is correct

def _adjust_band_code_for_sensor_type(current_band_code, expected_band_code, short_period=False):
    """
    Adjusts the band code if the current trace belongs to a short-period seismometer.

    SEED convention distinguishes between **broadband** and **short-period** seismometers.
    This function adjusts the expected band code based on the current sensor type.

    Mapping:
    - 'B' (Broadband) → 'S' (Short-period)
    - 'H' (High-frequency broadband) → 'E' (Short-period high-frequency)
    - 'D' (Very long period broadband) → 'C' (Short-period very long period)
    - 'G' (Extremely high-frequency broadband) → 'F' (Short-period extremely high-frequency)

    Parameters:
    ----------
    current_band_code : str
        The first character of the current `trace.stats.channel` (e.g., 'S', 'E', 'C', 'F').
    expected_band_code : str
        The computed band code based on the sampling rate.
    short_period : bool, optional
        If `True`, forces short-period band codes even if the current band code is not in the expected mapping.

    Returns:
    -------
    str
        The adjusted band code if applicable.

    Example:
    --------
    ```python
    adjusted_band_code = _adjust_band_code_for_sensor_type('S', 'B')
    print(adjusted_band_code)  # Output: 'S' (Short-period equivalent of 'B')
    ```
    """
    short_period_codes = {'S', 'E', 'C', 'F'}
    
    if current_band_code in short_period_codes or short_period:
        band_code_mapping = {'B': 'S', 'H': 'E', 'D': 'C', 'G': 'F'}
        return band_code_mapping.get(expected_band_code, expected_band_code)
    
    return expected_band_code



def fix_trace_id(trace, legacy=False, netcode=None, verbose=False):
    """
    Standardizes a seismic trace's ID to follow SEED naming conventions.

    This function:
    - Fixes legacy **VDAP/analog telemetry IDs** if `legacy=True`.
    - Ensures a valid **network code** if `netcode` is provided.
    - Adjusts the **band code** based on sampling rate.
    - Ensures the location code is **either empty or two characters**.
    - Fixes known **station name substitutions** (e.g., `CARL1` → `TANK` for KSC data).

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to modify.
    legacy : bool, optional
        If `True`, applies `_fix_legacy_id()` to correct old-style station codes (default: False).
    netcode : str, optional
        Network code to assign if missing (default: None).
    verbose : bool, optional
        If `True`, prints trace ID changes (default: False).

    Returns:
    -------
    bool
        `True` if the trace ID was changed, `False` otherwise.

    Notes:
    ------
    - Calls `_get_band_code()` to determine the correct band code based on sampling rate.
    - Calls `_adjust_band_code_for_sensor_type()` to refine the band code for short-period sensors.
    - Ensures **station names are corrected** for specific networks (e.g., `FL.CARL1 → FL.TANK`).

    Example:
    --------
    ```python
    from obspy import read

    trace = read("example.mseed")[0]
    changed = fix_trace_id(trace, legacy=True, netcode="XX", verbose=True)

    if changed:
        print(f"Updated Trace ID: {trace.id}")
    ```
    """
    changed = False


    if verbose:
        print(f"Initial ID: {trace.id}")

    if legacy:
        _fix_legacy_id(trace, network=netcode)
        changed = True

    if not trace.stats.network and netcode:
        trace.stats.network = netcode
        changed = True

    current_id = trace.id
    net = (trace.stats.network or "").upper()
    sampling_rate = trace.stats.sampling_rate

    if net in ["FL", "1R"]:
        fix_KSC_id(trace)

    elif net == "MV":
        # For legacy traces we already decoded the embedded station/channel info.
        # Only apply MVO-specific polishing here if needed.
        if not legacy:
            fix_trace_mvo(trace, verbose=verbose)
        else:
            trace.stats.location = fix_location_code(trace.stats.location)
            trace.stats.channel = fix_channel_code(trace.stats.channel, sampling_rate, short_period=True)

    else:
        chan = trace.stats.channel or ""
        if chan and not chan.startswith("A"):
            trace.stats.channel = fix_channel_code(chan, sampling_rate)
        trace.stats.location = fix_location_code(trace.stats.location)

    if trace.id != current_id:
        changed = True
        if verbose:
            print(f"Updated ID: {trace.id} (was {current_id}) based on fs={sampling_rate}")

    return changed

def fix_location_code(loc):
    """
    Normalizes location code to two characters:
    - None or '' becomes ''
    - '-' becomes '--'
    - Digits like '0' become '00'
    - Truncates anything longer than 2 characters
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
    Decomposes a SEED channel code into band, instrument, and orientation components.

    Parameters:
        chan (str): SEED channel code (e.g., 'BHZ', 'BZ', 'BH', 'B').

    Returns:
        tuple: (bandcode, instrumentcode, orientationcode), using 'x' for unknowns.
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


def _fix_legacy_id(trace, network=None, default_channel="EHZ", default_lowgain_channel="ELZ"):
    """
    First-pass fixer for legacy traces that may only have a station code and
    possibly a 1-character channel code.

    Rules
    -----
    - If station is IRIG, assign ACE and return.
    - If network is provided, set it.
    - If the station code ends with orientation/gain information, strip that
      suffix from station and use it to build a SEED channel code.
    - If channel is only one character (e.g. 'v', 'n', 'e'), treat it as an
      orientation hint.
    - Default to EHZ for most legacy seismic traces.
    - If low-gain is inferred, default to ELZ / ELN / ELE.

    Legacy suffix conventions handled
    ---------------------------------
    - STAZ / STAN / STAE  -> station=STA, channel=EHZ/EHN/EHE
    - STAV                -> station=STA, channel=EHZ
    - STAL                -> station=STA, channel=ELZ
    - STALZ / STALN / STALE -> station=STA, channel=ELZ/ELN/ELE
      (if 5-character station codes ever appear)
    """

    if network:
        trace.stats.network = network

    sta = (trace.stats.station or "").strip().upper()
    loc = (trace.stats.location or "").strip().upper()
    chan = (trace.stats.channel or "").strip().upper()

    if sta == "IRIG":
        trace.stats.channel = "ACE"
        return

    # ------------------------------------------------------------------
    # Start with defaults
    # ------------------------------------------------------------------
    bandcode = "E"
    instrumentcode = "H"
    orientationcode = "Z"

    # ------------------------------------------------------------------
    # If an existing channel is usable, decode what we can from it
    # ------------------------------------------------------------------
    if len(chan) == 3:
        # Already SEED-like: keep it unless we later find stronger evidence in station
        bandcode = chan[0]
        instrumentcode = chan[1]
        orientationcode = chan[2]

    elif len(chan) == 2:
        # Treat second char as orientation if plausible, otherwise instrument
        bandcode = chan[0]
        if chan[1] in "ZNE":
            instrumentcode = "H"
            orientationcode = chan[1]
        elif chan[1] in "HLD":
            instrumentcode = chan[1]
            orientationcode = "Z"

    elif len(chan) == 1:
        # Legacy one-letter channel: orientation only
        c = chan[0]
        if c in "VZ":
            orientationcode = "Z"
        elif c in "NE":
            orientationcode = c
        elif c == "L":
            instrumentcode = "L"
            orientationcode = "Z"

    # ------------------------------------------------------------------
    # Decode trailing info embedded in station code
    # Priority: handle low-gain+orientation if present, then single suffix
    # ------------------------------------------------------------------
    if len(sta) >= 2:
        last2 = sta[-2:]

        # e.g. MSSL, MHRL etc are low gain vertical by default only if final L
        # but if two-char suffix is LZ/LN/LE/LV, use both.
        if last2[0] == "L" and last2[1] in "ZNEV":
            instrumentcode = "L"
            orientationcode = "Z" if last2[1] == "V" else last2[1]
            sta = sta[:-2]

    if len(sta) >= 4:
        last1 = sta[-1]

        if last1 in "ZNEV":
            orientationcode = "Z" if last1 == "V" else last1
            sta = sta[:-1]
        elif last1 == "L":
            instrumentcode = "L"
            orientationcode = "Z"
            sta = sta[:-1]

    # ------------------------------------------------------------------
    # Apply defaults if nothing meaningful came through
    # ------------------------------------------------------------------
    if not bandcode or bandcode == "X":
        bandcode = "E"

    if not instrumentcode or instrumentcode == "X":
        instrumentcode = "H"

    if not orientationcode or orientationcode == "X":
        orientationcode = "Z"

    # Normalize weird vertical code if it somehow survived
    if orientationcode == "V":
        orientationcode = "Z"

    # ------------------------------------------------------------------
    # If original channel was empty/garbage and low-gain was inferred from station,
    # ensure EL? rather than EH?
    # ------------------------------------------------------------------
    if instrumentcode not in ("H", "L", "D"):
        instrumentcode = "H"

    trace.stats.network = (trace.stats.network or "").strip().upper()
    trace.stats.station = sta
    trace.stats.location = loc
    trace.stats.channel = f"{bandcode}{instrumentcode}{orientationcode}"

def fix_trace_mvo(trace, verbose=False):
    sta = (trace.stats.station or "").upper()
    fs = float(trace.stats.sampling_rate) if trace.stats.sampling_rate else np.nan

    legacy = True
    short_period = True

    if (
        (sta.startswith("MB") and sta != "MBET")
        or sta.startswith("MTB")
        or sta.startswith("BLV")
        or sta == "GHWS"
    ):
        legacy = False
        short_period = False

    if np.isfinite(fs) and 70.0 < fs < 80.0:
        fix_y2k_times_mvo(trace)
        fix_sample_rate(trace)

    if verbose:
        print(f"Fixing MVO trace ID: {trace.id} legacy={legacy} short_period={short_period}")

    if legacy:
        #_fix_legacy_id(trace, network="MV")
        pass # assume already done.
    else:
        trace.id = correct_nslc_mvo(
            trace.id,
            trace.stats.sampling_rate,
            short_period=short_period,
            net="MV",
            verbose=verbose,
        )

def fix_sample_rate(st, Fs=75.0, tol=0.01):
    """
    Snap near-matching sample rates to an exact target Fs.
    tol is fractional, so 0.01 means ±1%.
    """
    if isinstance(st, Stream):
        for tr in st:
            fix_sample_rate(tr, Fs=Fs, tol=tol)
    elif isinstance(st, Trace):
        tr = st
        sr = float(tr.stats.sampling_rate)
        if Fs * (1 - tol) < sr < Fs * (1 + tol):
            tr.stats.sampling_rate = Fs
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace object.")
    
def fix_y2k_times_mvo(st):
    if isinstance(st, Stream):
        for tr in st:
            fix_y2k_times_mvo(tr)
    elif isinstance(st, Trace):
        tr = st
        yyyy = tr.stats.starttime.year
        if yyyy in (1991, 1992, 1993):
            tr.stats.starttime._set_year(yyyy + 8)
        elif yyyy < 1908:
            tr.stats.starttime._set_year(yyyy + 100)
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace object.")
    
def correct_nslc_mvo(traceID, Fs, short_period=None, net="MV", verbose=False):
    """
    Standardize Montserrat trace IDs, handling legacy and special cases.

    Parameters
    ----------
    traceID : str
        Input trace ID in NET.STA.LOC.CHA form.
    Fs : float
        Sampling rate in Hz.
    short_period : bool or None
        If True, force short-period band code logic where appropriate.
        If None, leave to existing channel/station heuristics.
    net : str
        Network code to use in output ID.
    verbose : bool
        If True, print debug information.

    Returns
    -------
    str
        Corrected trace ID in NET.STA.LOC.CHA form.
    """

    def vprint(*args):
        if verbose:
            print(*args)

    def _safe_decompose(chan):
        band, inst, orient = decompose_channel_code(chan)
        # normalize "unknown" placeholders to empty strings
        band = "" if band in (None, "x", "X") else band.upper()
        inst = "" if inst in (None, "x", "X") else inst.upper()
        orient = "" if orient in (None, "x", "X") else orient.upper()
        return band, inst, orient

    def _normalize_loc(loc):
        loc = (loc or "").strip().upper()
        return fix_location_code(loc)

    def handle_microbarometer(chan, loc, sta, Fs):
        """
        Normalize infrasound / pressure channel names to SEED D?O / D?F style.
        """
        vprint(f"Handling microbarometer channel: {chan} at {sta} with Fs={Fs}")

        chan = (chan or "").strip().upper()
        loc = (loc or "").strip().upper()

        # Common legacy pressure names
        if chan in ("PR", "PRS"):
            # Let band code be set from sample rate below
            chan = "BDO"

        # Legacy broadband/short-period prefixes sometimes used with acoustic channels
        if chan.startswith("E"):
            chan = "H" + chan[1:]
        elif chan.startswith("S"):
            chan = "B" + chan[1:]

        bandcode, instrumentcode, orientationcode = _safe_decompose(chan)
        expected_band_code = fix_channel_code(
            chan if chan else "BDO", Fs, short_period=False
        )[0].upper()

        # Acoustic / pressure sensors should use instrument code D
        if orientationcode.isdigit():
            chan = expected_band_code + "D" + orientationcode
        elif orientationcode == "F":
            chan = expected_band_code + "DF"
        else:
            chan = expected_band_code + "DO"

        vprint(f"Converted microbarometer channel: {loc}.{chan}")
        return chan, _normalize_loc(loc)

    def handle_seismic(chan, loc, sta, Fs, short_period):
        """
        Normalize seismic channel names.
        """
        vprint(f"Handling seismic channel: {chan} at {sta} with Fs={Fs}")

        chan = (chan or "").strip().upper()
        loc = (loc or "").strip().upper()
        sta = (sta or "").strip().upper()

        # Legacy short-period form like SBZ/SBN/SBE -> BHZ/BHN/BHE and mark not short-period
        if chan.startswith("SB") and len(chan) == 3 and chan[2] in "ZNE":
            chan = "BH" + chan[2]
            short_period = False

        bandcode, instrumentcode, orientationcode = _safe_decompose(chan)

        # Sometimes orientation is stored in 1-char location code
        if not orientationcode and len(loc) == 1 and loc in "ZNE":
            orientationcode = loc
            loc = ""

        # Sensible defaults if missing
        if not instrumentcode:
            instrumentcode = "H"
        if not orientationcode:
            orientationcode = "Z"
        if not bandcode:
            bandcode = _get_band_code(Fs) or "S"

        chan = bandcode + instrumentcode + orientationcode

        # Legacy low-gain warning
        if sta and sta[:2] != "MB" and ((short_period and "L" in chan) or sta.endswith("L")):
            vprint(
                f"Warning: {traceID} might be a legacy ID for a low-gain sensor "
                f"from an old analog network"
            )

        corrected = fix_channel_code(chan, Fs, short_period=bool(short_period))
        return corrected, _normalize_loc(loc)

    # ------------------------------------------------------------------
    # Main logic
    # ------------------------------------------------------------------
    traceID = (traceID or "").replace("?", "X").strip()
    parts = traceID.split(".")
    if len(parts) != 4:
        raise ValueError(f"Invalid trace ID format: {traceID!r}. Expected NET.STA.LOC.CHA")

    oldnet, oldsta, oldloc, oldcha = parts

    if oldsta.strip().upper() == "GHWS":
        return traceID  # Weather station special case

    sta = oldsta.strip().upper() if oldsta else ""
    loc = oldloc.strip().upper() if oldloc else ""
    chan = oldcha.strip().upper() if oldcha else ""
    net = (net or oldnet or "").strip().upper()

    # Handle legacy loc codes containing timing / telemetry hints
    if "J" in loc or "I" in loc:
        Fs = 75.0
        loc = loc.replace("J", "").replace("I", "")

    loc = _normalize_loc(loc)

    if net == "MV":
        # Additional MVO-specific 75 Hz hints
        if "J" in chan:
            Fs = 75.0

        if chan != "DUM":
            # AEF file channel quirks
            if chan.startswith("S J"):
                chan = "SH" + chan[3:]
            elif chan.startswith("SBJ"):
                chan = "BH" + chan[3:]

            # Infrasound / pressure
            if len(chan) >= 2 and chan[:2] in ("AP", "PR"):
                chan, loc = handle_microbarometer(chan, loc, sta, Fs)

            # Already-acoustic-ish codes; mostly preserve, just normalize band if needed
            elif len(chan) >= 2 and (
                chan[:2] in ("AH", "PH") or (chan.startswith("S") and chan.endswith("A"))
            ):
                # If channel is malformed/short, still try to regularize band code
                if chan:
                    fixed = fix_channel_code(chan, Fs, short_period=False)
                    # Preserve acoustic instrument/orientation where possible
                    b, i, o = _safe_decompose(fixed)
                    if not i:
                        i = "D"
                    if not o:
                        o = "O"
                    chan = (b or "B") + i + o
                loc = _normalize_loc(loc)

            # Seismic
            else:
                chan, loc = handle_seismic(chan, loc, sta, Fs, short_period)

    newID = f"{net}.{sta}.{loc}.{chan}"
    return newID

def fix_KSC_id(trace):
    net = (trace.stats.network or "").upper()
    sta = (trace.stats.station or "").upper()
    loc = (trace.stats.location or "").upper()
    chan = (trace.stats.channel or "").upper()

    if sta.startswith("BHP") or sta in ("TANKP", "FIREP"):
        if chan.startswith("2"):
            if chan == "2000":
                chan = "EHZ"
            elif chan == "2001":
                chan = "EH1"
            elif chan == "2002":
                chan = "EH2"
            trace.stats.channel = chan
        trace.stats.network = "1R"

    if trace.stats.station == "CARL1":
        trace.stats.station = "TANK"
    elif trace.stats.station == "CARL0":
        trace.stats.station = "BCHH"
    elif trace.stats.station == "378":
        trace.stats.station = "DVEL1"
    elif trace.stats.station == "FIRE" and trace.stats.starttime.year == 2018:
        trace.stats.station = "DVEL2"

    if trace.stats.network == "FL":
        trace.stats.network = "1R"

    trace.stats.location = fix_location_code(trace.stats.location or "")
    trace.stats.channel = fix_channel_code(trace.stats.channel, trace.stats.sampling_rate)

def fake_trace(id, sampling_rate=100.0, npts=1000, starttime=UTCDateTime(), data=None):
    """
    Create a fake ObsPy Trace with specified parameters.

    Parameters
    ----------
    id : str
        Trace ID in the format 'NET.STA.LOC.CHAN'.
    sampling_rate : float
        Sampling rate in Hz (default: 100.0).
    npts : int
        Number of data points (default: 1000).
    starttime : obspy.UTCDateTime or None
        Start time of the trace (default: None, uses current time).
    data : array-like or None
        Data values for the trace (default: None, generates random data).

    Returns
    -------
    obspy.Trace
        The created trace object.
    """

    if data is None:
        data = np.random.randn(npts).astype(np.float32)

    tr = Trace()
    tr.id = id
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = id.split('.')
    tr.stats.sampling_rate = sampling_rate
    tr.stats.starttime = starttime
    tr.data = np.array(data, dtype=np.float32)

    return tr


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

def Stream_min_starttime(all_traces):
    """
    Computes the minimum and maximum start and end times for a given Stream.

    This function takes an **ObsPy Stream** containing multiple traces and 
    determines the following time statistics:
    - **Earliest start time** (`min_stime`)
    - **Latest start time** (`max_stime`)
    - **Earliest end time** (`min_etime`)
    - **Latest end time** (`max_etime`)

    Parameters:
    ----------
    all_traces : obspy.Stream
        A Stream object containing multiple seismic traces.

    Returns:
    -------
    tuple:
        - **min_stime (UTCDateTime)**: The earliest start time among all traces.
        - **max_stime (UTCDateTime)**: The latest start time among all traces.
        - **min_etime (UTCDateTime)**: The earliest end time among all traces.
        - **max_etime (UTCDateTime)**: The latest end time among all traces.

    Notes:
    ------
    - Useful for determining the **temporal coverage** of a Stream.
    - Created for the **CALIPSO data archive** (Alan Linde).

    Example:
    --------
    ```python
    from obspy import read

    # Load a Stream of seismic data
    st = read("example.mseed")

    # Compute time bounds
    min_stime, max_stime, min_etime, max_etime = Stream_min_starttime(st)

    print(f"Start Time Range: {min_stime} to {max_stime}")
    print(f"End Time Range: {min_etime} to {max_etime}")
    ```
    """ 
    min_stime = min([tr.stats.starttime for tr in all_traces])
    max_stime = max([tr.stats.starttime for tr in all_traces])
    min_etime = min([tr.stats.endtime for tr in all_traces])
    max_etime = max([tr.stats.endtime for tr in all_traces])    
    return min_stime, max_stime, min_etime, max_etime


# -------------------------------
# Processing markers (use stats.processing)
# -------------------------------

def add_processing_step(tr: Trace, msg: str) -> None:
    if not hasattr(tr.stats, "processing") or tr.stats.processing is None:
        tr.stats.processing = []
    tr.stats.processing.append(str(msg))

def station_ids_from_stream(st: Stream) -> tuple[str, ...]:
    """Return sorted, unique seed IDs from a Stream."""
    return tuple(sorted({tr.id for tr in st}))

#####  Helper functions for machine learning workflow #####
def choose_best_traces(st, MAX_TRACES=8, include_seismic=True, include_infrasound=False, include_uncorrected=False):

    priority = np.array([float(tr.stats.quality_factor) for tr in st])      
    for i, tr in enumerate(st):           
        if tr.stats.channel[1]=='H':
            if include_seismic:
                if tr.stats.channel[2] == 'Z':
                    priority[i] *= 2
            else:
                priority[i] = 0
        if tr.stats.channel[1]=='D':
            if include_infrasound:
                priority[i] *= 2 
            else:
                priority[i] = 0
        if not include_uncorrected:
            if 'units' in tr.stats:
                if tr.stats.units == 'Counts':
                    priority[i] = 0
            else:
                priority[i] = 0

    n = np.count_nonzero(priority > 0.0)
    n = min([n, MAX_TRACES])
    j = np.argsort(priority)
    chosen = j[-n:]  
    return chosen        
        
def select_by_index_list(st, chosen):
    st2 = Stream()
    for i, tr in enumerate(st):
        if i in chosen:
            st2.append(tr)
    return st2 


def stream_add_units(st, default_units="m/s"):
    """
    If a trace has no .stats.units, infer:
      - 'Counts' when amplitudes look large
      - default_units (e.g., 'm/s' or 'm') when amplitudes look small
    """
    for tr in st:
        # keep any existing unit annotation
        current_units = getattr(tr.stats, "units", None)
        if current_units:
            continue

        data = tr.data
        if data is None or len(data) == 0:
            tr.stats["units"] = "Counts"
            continue

        median_abs = float(np.nanmedian(np.abs(data)))
        if np.isfinite(median_abs) and median_abs < 1.0:
            # likely already in physical units
            tr.stats["units"] = default_units
        else:
            tr.stats["units"] = "Counts"