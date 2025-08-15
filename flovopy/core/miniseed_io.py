"""
miniseed_io.py

Waveform I/O utilities for reading and writing MiniSEED, and merging ObsPy Trace and Stream objects.
Designed to support gap-aware reading from and writing to MiniSEED files

Author: Glenn Thompson (2025)
"""

import os
import numpy as np
from obspy import read, Stream, Trace
from collections import defaultdict
from flovopy.core.trace_utils import sanitize_stream, streams_equal, downsample_stream_to_common_rate
from flovopy.core.gaputils import unmask_gaps
from obspy.signal.quality_control import MSEEDMetadata 

def smart_merge(stream_in, debug=False, strategy='obspy', allow_timeshift=False, max_shift_seconds=2):
    """
    Merge an ObsPy Stream in-place using a specified strategy.

    Parameters
    ----------
    stream_in : obspy.Stream
        The input stream to merge. Will be modified in-place.
    debug : bool, optional
        If True, print diagnostic info during merging.
    strategy : {'obspy', 'max'}, optional
        Strategy to resolve overlaps:
        - 'obspy': standard ObsPy merge + forward/reverse consistency check
        - 'max': retain maximum absolute value at overlapping points
        - 'both': try 'obspy' first, and if merge errors result, try 'max'
    allow_timeshift : bool, optional
        If True, attempts 0 or ±1 second shifts when conflicts occur.
    max_shift_seconds : int, optional
        Maximum number of integer seconds to try shifting if conflict.

    Returns
    -------
    dict
        Report containing merge status and any detected instabilities.
        report['status_by_id'][trace_id] ∈ {'ok', 'conflict', 'timeshifted', 'max'}
    """
    if len(stream_in)>0 and stream_in[0].stats.network == 'MV':
        allow_timeshift=True

    def merge_max(substream, sr, t0, npts):
        merged_data = np.ma.masked_all(npts, dtype=np.float32)
        for tr in substream:
            offset = int((tr.stats.starttime - t0) * sr)
            end = offset + tr.stats.npts
            incoming = np.ma.masked_invalid(tr.data)
            existing = merged_data[offset:end]

            use_new = existing.mask
            both = ~existing.mask & ~incoming.mask

            merged_data[offset:end][use_new] = incoming[use_new]
            merged_data[offset:end][both] = np.where(
                np.abs(incoming[both]) > np.abs(existing[both]),
                incoming[both], existing[both]
            )

        stats = substream[0].stats.copy()
        stats.starttime = t0
        stats.npts = npts
        return Trace(data=merged_data, header=stats)

    report = {
        'instabilities': [],
        'collision_samples': {},
        'status_by_id': {},
        'status': 'ok',
        'message': '',
        'summary': {
            'total_ids': 0,
            'merged': 0,
            'ok': 0,
            'conflict': 0,
            'duplicate': 0,
            'empty': 0,
            'timeshifted': 0,
            'max': 0
        },
        'time_shifts': {},
        'fallback_to_max': []
    }

    stream_by_id = defaultdict(Stream)
    for tr in stream_in:
        stream_by_id[tr.id].append(tr)
    report['summary']['total_ids'] = len(stream_by_id)

    merged_stream = []

    for trace_id, substream in stream_by_id.items():
        if debug:
            print(f"- Merging {trace_id} ({len(substream)} traces)")

        sanitize_stream(substream)

        if len(substream) == 0:
            report['status_by_id'][trace_id] = 'empty'
            report['summary']['empty'] += 1
            continue

        if len(substream) == 1:
            merged_stream.append(substream[0].copy())
            report['status_by_id'][trace_id] = 'ok'
            report['summary']['ok'] += 1
            continue

        sr = substream[0].stats.sampling_rate
        t0 = min(tr.stats.starttime for tr in substream)
        t1 = max(tr.stats.endtime for tr in substream)
        npts = int((t1 - t0) * sr) + 1
        status = 'ok'
        merged_trace = None

        if strategy == 'obspy' or strategy=='both':
            def try_merge(trs):
                return trs.merge(method=1, fill_value=np.nan)[0]

            fwd = try_merge(substream)
            rev = try_merge(Stream(substream[::-1]))
            data_fwd = np.ma.masked_invalid(fwd.data)
            data_rev = np.ma.masked_invalid(rev.data)

            unequal = (~np.isclose(data_fwd, data_rev, rtol=1e-5, atol=1e-8)) & \
                      ~data_fwd.mask & ~data_rev.mask
            n_diff = np.count_nonzero(unequal)

            if n_diff > 0 and allow_timeshift:
                for shift in range(1, max_shift_seconds + 1):
                    for sign in [-1, 1]:
                        shifted = substream.copy()
                        for tr in shifted:
                            tr.stats.starttime += sign * shift
                        fwd = try_merge(shifted)
                        rev = try_merge(Stream(shifted[::-1]))
                        df, dr = np.ma.masked_invalid(fwd.data), np.ma.masked_invalid(rev.data)
                        unequal = (~np.isclose(df, dr, rtol=1e-5, atol=1e-8)) & ~df.mask & ~dr.mask
                        n_diff = np.count_nonzero(unequal)
                        if n_diff == 0:
                            merged_trace = fwd
                            merged_trace.data = df
                            report['status_by_id'][trace_id] = 'timeshifted'
                            report['summary']['timeshifted'] += 1
                            report['time_shifts'][trace_id] = sign * shift
                            break
                    if trace_id in report['status_by_id']:
                        break

            if merged_trace is None:
                if n_diff > 0 and strategy=='both':
                    merged_trace = merge_max(substream, sr, t0, npts)
                    report['status_by_id'][trace_id] = 'max'
                    report['summary']['max'] += 1
                    report['fallback_to_max'].append(trace_id)
                else:
                    merged_trace = fwd
                    merged_trace.data = data_fwd
                    report['status_by_id'][trace_id] = status
                    report['summary'][status] += 1
            else:
                status = 'timeshifted'

        elif strategy == 'max':
            merged_trace = merge_max(substream, sr, t0, npts)
            report['status_by_id'][trace_id] = 'ok'
            report['summary']['ok'] += 1

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        merged_stream.append(merged_trace)

    stream_in.clear()
    stream_in.extend(merged_stream)
    report['summary']['merged'] = len(merged_stream)
    return report





def write_mseed(tr, mseedfile, fill_value=0.0, overwrite_ok=True, pickle_fallback=False):
    """
    Writes a Trace or Stream to MiniSEED, filling masked gaps with a constant value.

    Parameters
    ----------
    tr : obspy.Trace or obspy.Stream
        The trace(s) to write.
    mseedfile : str
        Destination MiniSEED filename.
    fill_value : float, optional
        Value used to fill masked (gap) regions. Default is 0.0.
    overwrite_ok : bool, optional
        If True (default), overwrites the specified file. If False, writes to an indexed filename. Ignored if file does not already exist.
    pickle_fallback : bool, optional
        If True, writes a .pkl backup if MiniSEED fails.

    Returns
    -------
    bool
        True if written to `mseedfile`, False if written to an indexed file, None if fallback was used.
    """
   

    # 1. Unmask gaps
    if isinstance(tr, Stream):
        marked = Stream([unmask_gaps(t, fill_value=fill_value, inplace=False) for t in tr])
    elif isinstance(tr, Trace):
        marked = unmask_gaps(tr, fill_value=fill_value, inplace=False)
    else:
        raise TypeError("Input must be an ObsPy Trace or Stream")

    # 2. Try to write
    if overwrite_ok:
        try:
            marked.write(mseedfile, format="MSEED")
            return True
        except Exception as e:
            print(f"✘ Failed to write MSEED: {e}")
    else:
        base, ext = os.path.splitext(mseedfile)
        for index in range(1, 100):  # Limit to 99 indexed retries
            indexed = f"{base}.{index:02d}{ext}"
            if not os.path.exists(indexed):
                try:
                    marked.write(indexed, format="MSEED")
                    print(f"✔ Indexed file written due to conflict: {indexed}")
                    return False
                except Exception as e:
                    print(f"✘ Failed to write indexed MSEED: {e}")
                    break

    # 3. Optional fallback to pickle
    if pickle_fallback:
        pklfile = mseedfile + ".pkl"
        if not os.path.exists(pklfile):
            try:
                marked.write(pklfile, format="PICKLE")
                print(f"✔ Fallback written to pickle: {pklfile}")
                return None
            except Exception as e2:
                print(f"✘ Failed to write fallback pickle: {e2}")

    return False


def read_mseed(
    mseedfile,
    fill_value=0.0,
    min_gap_duration_s=1.0,
    split_on_mask=False,
    merge=True,
    merge_strategy='obspy',
    starttime=None,
    endtime=None,
    min_sampling_rate=50.0,
    max_sampling_rate=250.0,
    unmask_short_zeros=True
):
    """
    Reads a MiniSEED file and applies trimming, masking, and optional splitting or merging.

    Parameters
    ----------
    mseedfile : str
        Path to the MiniSEED file to read.
    fill_value : float, optional
        Value used to fill gaps in unmasked data. Default is 0.0.
    unmask_short_zeros : bool, optional
        If True, unmasks internal zero gaps shorter than `min_gap_duration_s`.
    min_gap_duration_s : float, optional
        Gaps shorter than this (in seconds) will be unmasked if `unmask_short_zeros=True`. Default is 1.0 s.
    split_on_mask : bool, optional
        If True, splits the stream at masked gaps using `Stream.split()`. Overrides merge.
    merge : bool, optional
        If True, merges using `smart_merge()`. Ignored if `split_on_mask` is True.
    merge_strategy : either 'obspy', 'max', or 'both'
    starttime : UTCDateTime, optional
        Trim start time.
    endtime : UTCDateTime, optional
        Trim end time.
    max_sampling_rate : float, optional
        Decimate traces with higher sampling rates. Default is 250.0 Hz.


    Returns
    -------
    Stream
        Cleaned, optionally merged/split ObsPy Stream.
    """
    try:
        stream = read(mseedfile, format='MSEED')
    except:
        try:
            stream = read(mseedfile) # format unknown
        except:
            return Stream()
    if len(stream)==0:
        return Stream()

    # Optional trim
    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    
    for tr in stream:
        if tr.stats.sampling_rate < min_sampling_rate:
            stream.remove(tr)


    # Downsample to consistent rate, capped by max_sampling_rate
    downsample_stream_to_common_rate(stream, inplace=True, max_sampling_rate=max_sampling_rate)

    # Final optional stream-level sanitization
    try:
        sanitize_stream(stream, unmask_short_zeros=unmask_short_zeros, min_gap_duration_s=min_gap_duration_s)
    except ImportError:
        pass

    # Split or merge if requested
    if split_on_mask:
        stream = stream.split()
    elif merge:
        try:
            report = smart_merge(stream, strategy=merge_strategy)
        except Exception as e:
            print(f"⚠️ smart_merge failed: {e}, falling back to stream.merge()")
            stream.merge(method=1, fill_value=fill_value)

    return stream



def compare_mseed_files(src_file, dest_file):
    try:
        src_stream = read_mseed(src_file)
        dest_stream = read_mseed(dest_file)
        return streams_equal(src_stream, dest_stream), None
    except Exception as e:
        return False, str(e)
    

def _can_write_to_miniseed_and_read_back(tr, return_metrics=True):
    """
    Tests whether an ObsPy Trace can be written to and successfully read back from MiniSEED format.

    This function attempts to:
    1. **Convert trace data to float** (if necessary) to avoid MiniSEED writing issues.
    2. **Write the trace to a temporary MiniSEED file**.
    3. **Read the file back to confirm integrity**.
    4. If `return_metrics=True`, computes MiniSEED metadata using `MSEEDMetadata()`.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to test for MiniSEED compatibility.
    return_metrics : bool, optional
        If `True`, computes MiniSEED quality control metrics and stores them in `tr.stats['metrics']` (default: True).

    Returns:
    -------
    bool
        `True` if the trace can be written to and read back from MiniSEED successfully, `False` otherwise.

    Notes:
    ------
    - **Converts `tr.data` to float** (`trace.data.astype(float)`) if necessary.
    - **Removes temporary MiniSEED files** after testing.
    - Uses `MSEEDMetadata()` to compute quality metrics similar to **ISPAQ/MUSTANG**.
    - Sets `tr.stats['quality_factor'] = -100` if the trace has **no data**.

    Example:
    --------
    ```python
    from obspy import read

    # Load a trace
    tr = read("example.mseed")[0]

    # Check if it can be written & read back
    success = _can_write_to_miniseed_and_read_back(tr, return_metrics=True)

    print(f"MiniSEED compatibility: {success}")
    if success and "metrics" in tr.stats:
        print(tr.stats["metrics"])  # Print MiniSEED quality metrics
    ```
    """
    if len(tr.data) == 0:
        tr.stats['quality_factor'] = 0.0
        return False

    # Convert data type to float if necessary (prevents MiniSEED write errors)
    if not np.issubdtype(tr.data.dtype, np.floating):
        tr.data = tr.data.astype(float)

    tmpfilename = f"{tr.id}_{tr.stats.starttime.isoformat()}.mseed"

    try:
        # Attempt to write to MiniSEED
        if hasattr(tr.stats, "mseed") and "encoding" in tr.stats.mseed:
            del tr.stats.mseed["encoding"]
        tr.write(tmpfilename)

        # Try reading it back
        _ = read(tmpfilename)

        if return_metrics:
            # Compute MiniSEED metadata
            mseedqc = MSEEDMetadata([tmpfilename])
            tr.stats['metrics'] = mseedqc.meta
            add_to_trace_history(tr, "MSEED metrics computed (similar to ISPAQ/MUSTANG).")

        return True  # Successfully wrote and read back

    except Exception as e:
        if return_metrics:
            tr.stats['quality_factor'] = 0.0
        print(f"Failed MiniSEED write/read test for {tr.id}: {e}")
        return False

    finally:
        # Clean up the temporary file
        if os.path.exists(tmpfilename):
            os.remove(tmpfilename)