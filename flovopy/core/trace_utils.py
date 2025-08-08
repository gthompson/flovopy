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

    # Apply legacy VDAP-style fix (e.g., 4-char station name with orientation)
    if legacy:
        _fix_legacy_id(trace)

    if verbose:
        print(f"Initial ID: {trace.id}")

    # Ensure network code exists
    if not trace.stats.network and netcode:
        trace.stats.network = netcode
        changed = True

    current_id = trace.id
    net = trace.stats.network
    sampling_rate = trace.stats.sampling_rate

    # Route to specialized network fixers

        
    if not net or net in ['FL', '1R']:
        fix_KSC_id(trace)
    elif net == 'MV':
        print(f'Fixing MV trace ID: {trace.id}')
        fix_trace_mvo(trace)
        print(f'Fixed MV trace ID: {trace.id}')
        
    else:
        # Apply band code fix unless this is an analog QC channel (e.g., starts with 'A')
        chan = trace.stats.channel or ''
        if chan and not chan.startswith('A'):
            trace.stats.channel = fix_channel_code(chan, sampling_rate)

        # Ensure location code is 0 or 2 characters
        trace.stats.location = fix_location_code(trace.stats.location)

    # Final ID check
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
    print(f'Fixing location code: {loc}')
    if not loc:
        return ''
    if loc == '-':
        return '--'
    if loc.isdigit():
        return loc.zfill(2)
    return loc[:2].zfill(2) if len(loc) < 2 else loc[:2]

def fix_channel_code(chan, sampling_rate, short_period=False):
    print(f'Fixing channel code: {chan} with fs={sampling_rate}, short_period={short_period}')
    current_band_code = chan[0].upper() if chan else ''

    # Determine the correct band code
    expected_band_code = _get_band_code(sampling_rate) # this assumes broadband sensor

    # adjust if short-period sensor
    expected_band_code = _adjust_band_code_for_sensor_type(current_band_code, expected_band_code, short_period=short_period)
    chan = expected_band_code + chan[1:]
    return chan 


def decompose_channel_code(chan):
    """
    Decomposes a SEED channel code into band, instrument, and orientation components.

    Parameters:
        chan (str): SEED channel code (e.g., 'BHZ', 'BZ', 'BH', 'B').

    Returns:
        tuple: (bandcode, instrumentcode, orientationcode), using 'x' for unknowns.
    """
    print(f'Decomposing channel code: {chan}')
    chan = (chan or "").upper()
    band = chan[0] if len(chan) > 0 else 'x'

    if len(chan) == 3:
        return band, chan[1], chan[2]
    elif len(chan) == 2:
        if chan[1] in 'ZNE':
            return band, 'H', chan[1]
        elif chan[1] in 'HDL':
            return band, chan[1], 'Z'
        else:
            return band, 'x', 'x'
    elif len(chan) == 1:
        return band, 'x', 'x'
    else:
        return 'x', 'x', 'x'

def _fix_legacy_id(trace, network=None, add_T=True):
    """
    Fixes legacy trace IDs for old VDAP/analog telemetry networks.

    This function corrects **four-character station codes** where the orientation
    is embedded within the station name.

    Corrections:
    - If `trace.stats.station == 'IRIG'`, sets `trace.stats.channel = 'ACE'`.
    - Converts single-letter channels ('v', 'n', 'e') to SEED channel names ('EHZ', 'EHN', 'EHE').
    - Extracts orientation (4th character of station name) to determine the correct SEED channel.
    - Removes the orientation character from the station name.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to modify.

    Returns:
    -------
    None
        Modifies `trace.stats.station` and `trace.stats.channel` in place.

    Example:
    --------
    ```python
    from obspy import read

    trace = read("example.mseed")[0]
    _fix_legacy_id(trace)

    print(trace.id)  # Corrected SEED-compliant ID
    ```
    """
    #print('LEGACY')
    #print(trace)
    if network:
        trace.stats.network = network
    if trace.stats.station == 'IRIG':
        trace.stats.channel = 'ACE'
        return


    
    """
    else:
        orientation = ""

        if trace.stats.channel:
            chan = trace.stats.channel.upper() 
        elif len(trace.stats.station)==4:
            orientation  = trace.stats.station[3].upper()
            trace.stats.station = trace.stats.station[0:3].strip()

        orientation = trace.stats.station[3].strip()
        if orientation:
            if chan in 'VZT':
                trace.stats.channel='EHZ'
            elif orientation=='N':
                trace.stats.channel='EHN'
            elif orientation=='E':
                trace.stats.channel='EHE'   
            elif orientation=='L':
                trace.stats.channel='ELZ' 
            elif orientation=='P':
                trace.stats.channel='EDF'
            else:
                trace.stats.channel='EH' + chan                         
        
        # Position 4 
        channel = trace.stats.channel        
        if orientation in "ZNE":  # Standard orientations
            channel = f"EH{orientation}"
        elif orientation == "L":  # Special case for "L"
            channel = "ELZ"
        elif orientation == 'P': # assume pressure sensor?
            channel = 'EDF'
        else:
            pass
            #channel = f'??{orientation}'
            #raise ValueError(f"Unknown orientation '{orientation}' in '{trace.stats.station}'")
        
        trace.stats.channel = channel
        
        #trace.stats.station = trace.stats.station[0:3].strip()  # Positions 1-3
    """
    

    net, sta, loc, chan = trace.id.split('.')
    bandcode, instrumentcode, orientationcode = decompose_channel_code(chan)

    # Process the channel code
    if len(chan) > 0:
        if len(chan) == 1 and chan[0].upper() in 'ZNE':
            orientationcode = chan[0].upper()  

        first_chan_code = chan[0].upper()
        if first_chan_code in 'SE':
            bandcode = first_chan_code
        
        instrumentcode = chan[1]  # Default to high-frequency broadband

    # Process the station code
    last_sta_char = sta[-1].upper()
    if sta[-1] in 'ZNEVLXYTH':
        # this might be a legacy station code that includes an orientation, or sensor type
        if sta != 'MNEV':
            sta = sta[:-1]
        if last_sta_char in 'ZVEXNY':
            orientationcode = last_sta_char      
        elif last_sta_char in 'LH':
            instrumentcode = last_sta_char
    
    if add_T and len(sta) < 4: # end station in T?
        sta = sta + 'T'

    if not bandcode:
        bandcode = 'S'
    if not instrumentcode:
        instrumentcode = 'H'
    if not orientationcode:
        orientationcode = 'Z'

    chan = bandcode + instrumentcode + orientationcode
    trace.stats.network = net.strip().upper()
    trace.stats.station = sta.strip().upper()
    trace.stats.location = loc.strip().upper() if loc else ''
    trace.stats.channel = chan.strip().upper()        


def fix_trace_mvo(trace):
    legacy = True
    sta = trace.stats.station
    shortperiod = True
    if (sta[0:2] == 'MB' and sta!='MBET') or sta[0:3]=='MTB' or sta[0:3]=='BLV' or sta=='GHWS':
        legacy = False    
        shortperiod = False #shortperiod may be true or false now

    if trace.stats.sampling_rate > 70.0 and trace.stats.sampling_rate < 80.0:
        fix_y2k_times_mvo(trace)
        fix_sample_rate(trace)

    #if len(trace.stats.channel)==3 and trace.stats.channel[1:] == 'HA':
    #    trace.stats.channel = trace.stats.channel[0] + 'DO'  # Convert 'HA' to 'DO' for outdoor microbarometer
    print('Fixing MVO trace ID:', trace.id, 'legacy:', legacy, 'shortperiod:', shortperiod)
    if legacy:
        _fix_legacy_id(trace, network='MV')        
    else:
        trace.id = correct_nslc_mvo(trace.id, trace.stats.sampling_rate, shortperiod=shortperiod)

def fix_sample_rate(st, Fs=75.0):
    if isinstance(st, Stream):
        for tr in st:
            fix_sample_rate(tr, Fs=Fs)  # Recursive call for each Trace
    elif isinstance(st, Trace):
        tr = st
        if tr.stats.sampling_rate > Fs * 0.99 and tr.stats.sampling_rate < Fs * 1.01:
            tr.stats.sampling_rate = Fs 
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace object.")    

def fix_y2k_times_mvo(st):
    if isinstance(st, Stream):
        for tr in st:
            fix_y2k_times_mvo(tr, Fs=75.0)  # Recursive call for each Trace
    elif isinstance(st, Trace):
        tr = st
        yyyy = tr.stats.starttime.year
        if yyyy == 1991 or yyyy == 1992 or yyyy == 1993: # digitize
            # OS9/Seislog for a while subtracted 8 years to avoid Y2K problem with OS9
            # should be already fixed but you never know
            tr.stats.starttime._set_year(yyyy+8)
        if yyyy < 1908:
            # getting some files exactly 100 years off
            tr.stats.starttime._set_year(yyyy+100) 
    else:
        raise TypeError("Input must be an ObsPy Stream or Trace object.")
    
def correct_nslc_mvo(traceID, Fs, shortperiod=None, net='MV'):
    """
    Standardizes Montserrat trace IDs, handling legacy and special cases.
    """
    def handle_microbarometer(chan, loc, sta, Fs):
        print(f"Handling microbarometer channel: {chan} at {sta} with Fs={Fs}")

        if chan=='PR' or chan=='PRS':
            chan='BDO'

        if chan[0]=='E':
            chan = 'H' + chan[1:]  # Convert 'E' to 'H' for high-frequency broadband
        elif chan[0]=='S':
            chan = 'B' + chan[1:]
       
        bandcode, instrumentcode, orientationcode = decompose_channel_code(chan)
        print(f"Decomposed channel: {bandcode}, {instrumentcode}, {orientationcode}")
        expected_band_code = fix_channel_code(chan, Fs)[0]
        if orientationcode.isdigit():
            chan = expected_band_code + 'D' + orientationcode
        elif orientationcode == 'F':
            chan = expected_band_code + 'DF'
        else:
            chan = expected_band_code + 'DO'

        print(f"Converted channel: {loc}.{chan}")
        return chan, loc


        '''
        # --- MVO legacy Seisan (1990s-2000s) ---
        if chan == 'PRS':
            return 'BDO', loc.zfill(2) if loc.isdigit() else '00'

        if chan.startswith('PR') and len(chan) == 3 and chan[2].isdigit():
            # PR1, PR2 (component in channel)
            return 'BDO', f'{chan[2]}0'

        # --- Old DSN format ---
        if chan == 'A N' and loc == 'J':
            return 'BDO', ''
        
        # --- New DSN format ---
        if chan == 'AP' and loc == 'S':
            return 'HDO', ''
        if chan == 'PR' and loc.isdigit():
            return 'HDO', loc.zfill(2)

        # --- Generic acoustic sensor catch-all ---
        if any(x in chan for x in ['AP', 'PR', 'PH']) or chan == 'S A':
            if Fs<80.0:
                return 'BDO', loc.zfill(2) if loc.isdigit() else ''
            else:
                return 'HDO', loc.zfill(2) if loc.isdigit() else ''        

        return chan, loc # if no match, return original
        '''

    def handle_seismic(chan, loc, sta, Fs, shortperiod):
        # Extract instrument and orientation codes from channel if present
        print(f"Handling seismic channel: {chan} at {sta} with Fs={Fs}")

        # Process channel code
        # Shortperiod logic
        if chan.startswith("SB") and len(chan) == 3 and chan[2] in "ZNE":
            chan = "BH" + chan[2]
            shortperiod = False

        bandcode, instrumentcode, orientationcode = decompose_channel_code(chan)
        if orientationcode == 'x':
            if len(loc) == 1 and loc in 'ZNE':
                orientationcode = loc
                loc = ''
        chan = bandcode + instrumentcode + orientationcode

        # Low gain warning
        if sta[:2] != 'MB' and (shortperiod and 'L' in chan or sta[-1] == 'L'):
            print(f'Warning: {traceID} might be a legacy ID for a low gain sensor from an old analog network')

        # Determine the correct band code
        expected_band_code = fix_channel_code(chan, Fs, short_period=shortperiod)[0]


        # Compose new channel code
        chan = expected_band_code + instrumentcode + orientationcode
        return chan, loc


    
    # --- Main logic ---

    # where is this from? commenting out, and adding logic for channel DUM later
    '''
    if traceID == '.MBLG.M.DUM':
        traceID = f'{net}.MBLG.10.SHZ'
    '''
    traceID = traceID.replace("?", "x")
    oldnet, oldsta, oldloc, oldcha = traceID.split('.')
    if oldsta == 'GHWS':
        return traceID  # GHWS is a special case, a weather station, no changes needed
    sta = oldsta.strip().upper() if oldsta else ''
    loc = oldloc.strip().upper() if oldloc else ''
    chan = oldcha.strip().upper() if oldcha else '' # or 'SHZ' # commented out SHZ default - do not force anything

    # Normalize '--' and similar
    #if not loc or loc.strip('-') == '':
    #    loc = ''
    if 'J' in loc or 'I' in loc:
        Fs = 75.0
        loc = loc.replace('J', '')  # Remove 'J' from location code

    # Ensure location code is 0 or 2 characters
    loc = fix_location_code(loc)


    if net=='MV': # we can call this function for non-MVO data if we add this
        # Fix sample rate for 'J'
        if 'J' in loc or 'J' in chan:
            Fs = 75.0

        if chan!='DUM':
        
            # AEF file channel fixes
            if chan.startswith('S J'):
                chan = 'SH' + chan[3:]
            elif chan.startswith('SBJ'):
                chan = 'BH' + chan[3:]
            
            
            # Microbarometer/infrasound handling
            if len(chan)>1 and chan[:2] in ['AP', 'PR']:
                chan, loc = handle_microbarometer(chan, loc, sta, Fs)            
            elif len(chan)>1 and (chan[:2] in ['AH', 'PH'] or (chan[0]=='S' and chan[-1] == 'A')):
                pass
            else:
                # Seismic channel handling
                chan, loc = handle_seismic(chan, loc, sta, Fs, shortperiod)

    newID = f"{net}.{sta}.{loc}.{chan}"
    return newID

def fix_KSC_id(trace):
    net, sta, loc, chan = trace.id.split('.')
    if sta[0:3]=='BHP' or sta=='TANKP' or sta=='FIREP':
        if chan[0]=='2':
            if chan == '2000':
                chan = 'EHZ'
            elif chan == '2001':
                chan = 'EH1'       
            elif chan == '2002':
                chan = 'EH2'   
            trace.stats.channel = chan
        trace.stats.network = net = '1R' 


    if trace.stats.station=='CARL1':
        trace.stats.station = 'TANK'
    elif trace.stats.station == 'CARL0':
        trace.stats.station = 'BCHH'
    elif trace.stats.station == '378':
        trace.stats.station = 'DVEL1'
    elif trace.stats.station == 'FIRE' and trace.stats.starttime.year == 2018:
        trace.stats.station = 'DVEL2'

    if trace.stats.network == 'FL':
        trace.stats.network = '1R'

    # Ensure location code is 0 or 2 characters
    trace.stats.location = fix_location_code(loc)
    #if trace.stats.location in ['00', '0', '--', '', '10']: # special case to work with Excel metadata
    #    trace.stats.location = '00'

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
    from obspy import Trace, UTCDateTime

    if data is None:
        data = np.random.randn(npts).astype(np.float32)

    tr = Trace()
    tr.id = id
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = id.split('.')
    tr.stats.sampling_rate = sampling_rate
    tr.stats.starttime = starttime
    tr.data = np.array(data, dtype=np.float32)

    return tr