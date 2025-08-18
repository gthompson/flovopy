from __future__ import annotations
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

from typing import Optional
import time

def smart_merge(stream_in,
                debug=False,
                strategy='obspy',
                allow_timeshift=False,
                max_shift_seconds=2,
                verbose=False):
    """
    Merge an ObsPy Stream in-place using a specified strategy.

    Parameters
    ----------
    stream_in : obspy.Stream
        The input stream to merge. Will be modified in-place.
    debug : bool, optional
        (Deprecated) If True, print diagnostic info during merging.
    strategy : {'obspy', 'max', 'both'}, optional
        Strategy to resolve overlaps:
        - 'obspy': standard ObsPy merge + forward/reverse consistency check
        - 'max': retain maximum absolute value at overlapping points
        - 'both': try 'obspy' first, and if merge errors/conflicts remain, use 'max'
    allow_timeshift : bool, optional
        If True, attempts 0 or ±1..max_shift_seconds second shifts when conflicts occur.
    max_shift_seconds : int, optional
        Max integer seconds to try shifting if conflict.
    verbose : bool, optional
        Print detailed timing & diagnostics.

    Returns
    -------
    dict
        Report with per-id status and counters.
    """
    def log(msg):
        if verbose or debug:
            print(msg)

    if len(stream_in) > 0 and stream_in[0].stats.network == 'MV':
        allow_timeshift = True

    def merge_max(substream, sr, t0, npts):
        t0_merge = time.perf_counter()
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
        log(f"    [merge_max] took {time.perf_counter()-t0_merge:.3f}s")
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
        n_tr = len(substream)
        log(f"- Merging {trace_id} ({n_tr} traces)")
        if n_tr:
            srs = sorted({float(tr.stats.sampling_rate) for tr in substream})
            t0s = min(tr.stats.starttime for tr in substream)
            t1s = max(tr.stats.endtime for tr in substream)
            log(f"    span {t0s} → {t1s}  | sampling_rate(s)={srs}")

        t_sanitize = time.perf_counter()
        try:
            log("    calling sanitize_stream(...)")
            sanitize_stream(substream)  # existing helper
        except Exception as e:
            log(f"    ! sanitize_stream failed: {e}")
        log(f"    sanitize_stream took {time.perf_counter()-t_sanitize:.3f}s")

        if len(substream) == 0:
            report['status_by_id'][trace_id] = 'empty'
            report['summary']['empty'] += 1
            continue

        if len(substream) == 1:
            merged_stream.append(substream[0].copy())
            report['status_by_id'][trace_id] = 'ok'
            report['summary']['ok'] += 1
            continue

        # Compute union window and npts estimate
        sr = float(substream[0].stats.sampling_rate)
        t0 = min(tr.stats.starttime for tr in substream)
        t1 = max(tr.stats.endtime for tr in substream)
        npts = int(round((t1 - t0) * sr)) + 1
        if npts <= 0 or npts > 1e9:
            log(f"    ! suspicious npts={npts} (sr={sr}, span={float(t1-t0):.3f}s)")

        status = 'ok'
        merged_trace = None

        if strategy in ('obspy', 'both'):
            def try_merge(trs: Stream):
                return trs.merge(method=1, fill_value=np.nan)[0]

            # Forward merge
            t_merge_fwd = time.perf_counter()
            try:
                fwd = try_merge(substream)
            except Exception as e:
                log(f"    ! forward merge failed: {e}")
                fwd = Stream(substream).merge(method=1, fill_value=np.nan)[0]  # best effort
            log(f"    forward merge took {time.perf_counter()-t_merge_fwd:.3f}s")

            # Reverse merge
            t_merge_rev = time.perf_counter()
            try:
                rev = try_merge(Stream(substream[::-1]))
            except Exception as e:
                log(f"    ! reverse merge failed: {e}")
                rev = try_merge(Stream(substream[::-1]))  # retry once
            log(f"    reverse merge took {time.perf_counter()-t_merge_rev:.3f}s")

            data_fwd = np.ma.masked_invalid(fwd.data)
            data_rev = np.ma.masked_invalid(rev.data)
            unequal = (~np.isclose(data_fwd, data_rev, rtol=1e-5, atol=1e-8)) & \
                      ~data_fwd.mask & ~data_rev.mask
            n_diff = int(np.count_nonzero(unequal))
            if n_diff > 0:
                log(f"    merge diff samples: {n_diff}")

            if n_diff > 0 and allow_timeshift:
                t_shift_all = time.perf_counter()
                found = False
                for shift in range(1, int(max_shift_seconds) + 1):
                    for sign in (-1, 1):
                        t_shift = time.perf_counter()
                        shifted = substream.copy()
                        for tr in shifted:
                            tr.stats.starttime += sign * shift
                        try:
                            fwd_s = try_merge(shifted)
                            rev_s = try_merge(Stream(shifted[::-1]))
                            df, dr = np.ma.masked_invalid(fwd_s.data), np.ma.masked_invalid(rev_s.data)
                            unequal = (~np.isclose(df, dr, rtol=1e-5, atol=1e-8)) & ~df.mask & ~dr.mask
                            n_diff_s = int(np.count_nonzero(unequal))
                            log(f"      try timeshift {sign:+d}{shift}s → diff={n_diff_s} (took {time.perf_counter()-t_shift:.3f}s)")
                            if n_diff_s == 0:
                                merged_trace = fwd_s
                                merged_trace.data = df
                                report['status_by_id'][trace_id] = 'timeshifted'
                                report['summary']['timeshifted'] += 1
                                report['time_shifts'][trace_id] = sign * shift
                                found = True
                                break
                        except Exception as e:
                            log(f"      ! timeshift {sign:+d}{shift}s failed: {e}")
                    if found:
                        break
                log(f"    timeshift search took {time.perf_counter()-t_shift_all:.3f}s")

            if merged_trace is None:
                if n_diff > 0 and strategy == 'both':
                    log("    falling back to merge_max()")
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
            log("    using merge_max() strategy")
            merged_trace = merge_max(substream, sr, t0, npts)
            report['status_by_id'][trace_id] = 'ok'
            report['summary']['ok'] += 1

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        merged_stream.append(merged_trace)

    stream_in.clear()
    stream_in.extend(merged_stream)
    report['summary']['merged'] = len(merged_stream)

    # Final per-call summary
    if verbose or debug:
        s = report['summary']
        print(f"[smart_merge] ids={s['total_ids']} merged={s['merged']} "
              f"ok={s['ok']} timeshifted={s['timeshifted']} max={s['max']} empty={s['empty']}")
    return report






def _downcast_float64_to_float32(obj: Stream | Trace) -> Stream | Trace:
    """
    In-place downcast of float64 arrays to float32 for all traces in obj.
    Returns the same object (modified).
    """
    if isinstance(obj, Stream):
        for tr in obj:
            if np.issubdtype(tr.data.dtype, np.floating) and tr.data.dtype == np.float64:
                tr.data = tr.data.astype(np.float32, copy=False)
        return obj
    elif isinstance(obj, Trace):
        if np.issubdtype(obj.data.dtype, np.floating) and obj.data.dtype == np.float64:
            obj.data = obj.data.astype(np.float32, copy=False)
        return obj
    return obj

def write_mseed(
    tr: Stream | Trace,
    mseedfile: str,
    *,
    fill_value: float = 0.0,
    overwrite_ok: bool = True,
    pickle_fallback: bool = False,
    encoding: Optional[str] = None,   # None => let ObsPy choose based on dtype
    reclen: Optional[int] = None,     # None => ObsPy default (usually 4096)
):
    """
    Writes a Trace or Stream to MiniSEED, filling masked gaps with a constant value.

    Behavior:
      - If `encoding` is None, ObsPy chooses based on dtype.
      - To avoid bloated files, any float64 data are **downcast to float32**
        unless you explicitly set `encoding="FLOAT64"`.

    Returns
    -------
    bool | None
        True if written to `mseedfile`, False if an indexed filename was used,
        None if a PICKLE fallback was written, or False on failure.
    """
    # 1) Unmask gaps (produce new arrays as needed)
    if isinstance(tr, Stream):
        marked = Stream([unmask_gaps(t, fill_value=fill_value, inplace=False) for t in tr])
    elif isinstance(tr, Trace):
        marked = unmask_gaps(tr, fill_value=fill_value, inplace=False)
    else:
        raise TypeError("Input must be an ObsPy Trace or Stream")

    # 2) Downcast float64 → float32 unless FLOAT64 explicitly requested
    enc_upper = encoding.upper() if isinstance(encoding, str) else None
    if enc_upper != "FLOAT64":
        marked = _downcast_float64_to_float32(marked)

    # 3) Build write kwargs (only include encoding/reclen if provided)
    write_kwargs = {"format": "MSEED"}
    if encoding is not None:
        write_kwargs["encoding"] = encoding
    if reclen is not None:
        write_kwargs["reclen"] = reclen

    # 4) Try to write
    if overwrite_ok:
        try:
            marked.write(mseedfile, **write_kwargs)
            return True
        except Exception as e:
            print(f"✘ Failed to write MSEED: {e}")
    else:
        base, ext = os.path.splitext(mseedfile)
        for index in range(1, 100):  # up to .99
            indexed = f"{base}.{index:02d}{ext}"
            if not os.path.exists(indexed):
                try:
                    marked.write(indexed, **write_kwargs)
                    print(f"✔ Indexed file written due to conflict: {indexed}")
                    return False
                except Exception as e:
                    print(f"✘ Failed to write indexed MSEED: {e}")
                    break

    # 5) Optional pickle fallback
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