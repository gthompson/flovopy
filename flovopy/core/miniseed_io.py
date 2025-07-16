"""
miniseed_io.py

Waveform I/O utilities for reading, writing, and handling gaps in ObsPy Trace and Stream objects.
Designed to support gap-aware reading from and writing to MiniSEED files, while preserving or
encoding information about masked regions (e.g., zero-padding, data gaps).

Key features:
- Detect and mask long spans of zero padding.
- Preserve gap metadata in `trace.stats.processing`.
- Fill or unfill gaps safely when writing MiniSEED.
- Avoid overwriting files by writing indexed alternatives.

Typical usage:
    st = read_mseed_with_gap_masking("file.mseed")
    write_safely(st, "output.mseed")

Author: Glenn Thompson (2025)
"""

import os
import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
from collections import defaultdict

def sanitize_trace(tr, unmask_short_zeros=False, zero_gap_threshold=100):
    data = np.asarray(tr.data, dtype=np.float32)

    # --- Trim leading/trailing zeros ---
    nonzero = np.flatnonzero(data != 0.0)
    if nonzero.size == 0:
        tr.data = np.ma.masked_array([], mask=[])
        return tr

    start, end = nonzero[0], nonzero[-1] + 1
    data = data[start:end]
    tr.stats.starttime += start / tr.stats.sampling_rate

    # --- Mask all NaNs and zeros ---
    data = np.ma.masked_invalid(data)
    data = np.ma.masked_where(data == 0.0, data, copy=False)

    # --- Optionally unmask short zero spans ---
    if unmask_short_zeros:
        masked = np.where(data.mask)[0]
        if masked.size:
            d = np.diff(masked)
            gap_starts = np.where(np.insert(d, 0, 2) > 1)[0]
            gap_ends = np.where(np.append(d, 2) > 1)[0] - 1
            for i in range(len(gap_starts)):
                s, e = masked[gap_starts[i]], masked[gap_ends[i]] + 1
                if (e - s) < zero_gap_threshold:
                    data.mask[s:e] = False  # Unmask short gap

    tr.data = data
    return tr

def sanitize_stream(stream):
    return Stream([sanitize_trace(tr.copy()) for tr in stream])


def smart_merge(traces, debug=False, strategy='obspy'):
    """
    Merge a list of ObsPy Trace objects using a specified strategy, while detecting merge instabilities.

    Parameters
    ----------
    traces : list of Trace
        List of ObsPy Trace objects to merge.
    debug : bool, optional
        If True, print detailed debug information.
    strategy : {'obspy', 'max'}
        Merge strategy:
        - 'obspy': Use ObsPy's merge method and mask unstable regions.
        - 'max': Use maximum absolute value per sample, ignoring conflicts.

    Returns
    -------
    merged : Stream
        Merged stream of traces (one Trace per unique trace.id).
    report : dict
        Dictionary containing:
            - 'merged': The merged Stream.
            - 'instabilities': List of trace IDs with merge conflicts.
            - 'collision_samples': Dict of {trace_id: number of unstable samples}.
            - 'status_by_id': Dict of {trace_id: 'ok' or 'conflict'}.
            - 'status': 'ok' or 'conflict' (if any conflicts).
            - 'message': Optional status message.
    """
    report = {
        'instabilities': [],
        'collision_samples': {},
        'status_by_id': {},
        'status': 'ok',
        'message': ''
    }
    merged = Stream()

    traces_by_id = defaultdict(list)
    for tr in traces:
        traces_by_id[tr.id].append(tr.copy())

    for trace_id, trace_list in traces_by_id.items():
        if debug:
            print(f"- Merging {trace_id} with {len(trace_list)} traces")

        sanitized = sanitize_stream(Stream(trace_list))
        sr = sanitized[0].stats.sampling_rate
        t0 = min(tr.stats.starttime for tr in sanitized)
        t1 = max(tr.stats.endtime for tr in sanitized)
        npts = int((t1 - t0) * sr) + 1

        status = 'ok'
        n_diff = 0

        if strategy == 'obspy':
            merged_forward = sanitize_stream(Stream(trace_list)).merge(method=1, fill_value=np.nan)[0]
            merged_reverse = sanitize_stream(Stream(trace_list[::-1])).merge(method=1, fill_value=np.nan)[0]

            data_fwd = np.ma.masked_invalid(merged_forward.data)
            data_rev = np.ma.masked_invalid(merged_reverse.data)
            unequal = (~np.isclose(data_fwd, data_rev, rtol=1e-5, atol=1e-8)) & ~data_fwd.mask & ~data_rev.mask

            n_diff = np.count_nonzero(unequal)
            if n_diff > 0:
                status = 'conflict'
                report['instabilities'].append(trace_id)
                report['collision_samples'][trace_id] = n_diff
                if debug:
                    print(f"- {n_diff} merge-instability samples in {trace_id}")
                # Mask out these positions
                merged_forward.data.mask[unequal] = True

            merged_trace = merged_forward

        elif strategy == 'max':
            merged_data = np.ma.masked_all(npts, dtype=np.float32)
            for tr in sanitized:
                offset = int((tr.stats.starttime - t0) * sr)
                end = offset + tr.stats.npts
                incoming = tr.data
                existing = merged_data[offset:end]

                use_new = existing.mask
                both = ~existing.mask & ~incoming.mask

                merged_data[offset:end][use_new] = incoming[use_new]
                merged_data[offset:end][both] = np.where(
                    np.abs(incoming[both]) > np.abs(existing[both]),
                    incoming[both], existing[both]
                )

            stats = sanitized[0].stats.copy()
            stats.starttime = t0
            stats.npts = npts
            merged_trace = Trace(data=merged_data, header=stats)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        report['status_by_id'][trace_id] = status
        if status == 'conflict':
            report['status'] = 'conflict'  # Set global status

        merged += merged_trace

    return merged, report


def mask_gaps(trace, fill_value=0.0):
    """
    Masks regions of the trace previously filled with `fill_value`, based on
    metadata in trace.stats.processing.
    """
    trace = trace.copy()
    if not hasattr(trace.stats, "processing"):
        return trace

    processing_lines = [p for p in trace.stats.processing if p.startswith("GAP")]
    if not processing_lines:
        return trace

    data = np.ma.masked_array(trace.data, mask=False)

    for line in processing_lines:
        try:
            parts = line.split()
            t1 = UTCDateTime(parts[3])
            t2 = UTCDateTime(parts[5])
            idx1 = int((t1 - trace.stats.starttime) * trace.stats.sampling_rate)
            idx2 = int((t2 - trace.stats.starttime) * trace.stats.sampling_rate)
            data.mask[idx1:idx2] = True
        except Exception as e:
            print(f"✘ Could not parse gap line '{line}': {e}")

    trace.data = data
    return trace


def unmask_gaps(trace, fill_value=0.0):
    """
    Fills masked gaps in the trace with `fill_value`, and appends GAP metadata.
    We use this before writing MiniSEED files.
    """
    trace = trace.copy()

    if not isinstance(trace.data, np.ma.MaskedArray):
        return trace  # nothing to do

    gaps = []
    mask = trace.data.mask
    if mask is not np.ma.nomask:
        idx = np.where(mask)[0]
        if len(idx):
            gap_bounds = np.split(idx, np.where(np.diff(idx) != 1)[0]+1)
            for group in gap_bounds:
                t1 = trace.stats.starttime + group[0] / trace.stats.sampling_rate
                t2 = trace.stats.starttime + (group[-1]+1) / trace.stats.sampling_rate
                gaps.append((t1, t2))

    trace.data = trace.data.filled(fill_value)

    if gaps:
        if not hasattr(trace.stats, "processing"):
            trace.stats.processing = []
        trace.stats.processing.append(f"Filled {len(gaps)} gaps with {fill_value}")
        for t1, t2 in gaps:
            trace.stats.processing.append(f"GAP {t2 - t1:.2f}s from {t1} to {t2}")

    return trace


def write_mseed(tr, mseedfile, fill_value=0.0, overwrite_ok=False):
    """
    Writes a Trace or Stream to MiniSEED, filling masked gaps and recording metadata.
    Avoids overwriting by default, writing indexed files instead.
    """

    try:
        if isinstance(tr, Stream):
            marked = Stream()
            for real_tr in tr:
                marked.append(unmask_gaps(real_tr, fill_value=fill_value))
        elif isinstance(tr, Trace):
            marked = unmask_gaps(tr, fill_value=fill_value)

        if overwrite_ok:
            marked.write(mseedfile, format='MSEED')
            return True
        else:
            index = 1
            while True:
                indexed = f"{mseedfile}.{index:02d}"
                if not os.path.isfile(indexed):
                    marked.write(indexed, format='MSEED')
                    print(f"✔ Indexed file written due to merge conflict: {indexed}")
                    break
                index += 1
            return False
    except Exception as e:
        print(e)
        pklfile = mseedfile + '.pkl'
        if not os.path.isfile(pklfile):
            try:
                tr.write(pklfile, format='PICKLE')
            except:
                print(f'✘ Could not write {pklfile}')
        return False


def read_mseed(
    mseedfile,
    fill_value=0.0,
    zero_gap_threshold=100,
    split_on_mask=False,
    merge=True,
    starttime=None,
    endtime=None,
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
    zero_gap_threshold : int, optional
        Minimum number of consecutive 0.0 samples to treat as a gap. Default is 100.
    split_on_mask : bool, optional
        If True, splits the stream at masked gaps using `Stream.split()`. Overrides merge.
    merge : bool, optional
        If True, merges using `smart_merge()`. Ignored if `split_on_mask` is True.
    starttime : UTCDateTime, optional
        Trim start time.
    endtime : UTCDateTime, optional
        Trim end time.
    max_sampling_rate : float, optional
        decimate() higher sampling rates than this by an integer. Default is 250.0 Hz.
    unmask_short_zeros : bool, optional
        If True, unmask short internal zero spans (< `zero_gap_threshold` samples).

    Returns
    -------
    Stream
        Cleaned, optionally merged/split ObsPy Stream.
    """
    stream_in = read(mseedfile)
    if starttime or endtime:
        stream_in.trim(starttime=starttime, endtime=endtime)

    stream_out = Stream()
    for tr in stream_in:

        if tr.stats.sampling_rate > max_sampling_rate:
            try:
                factor = int(round(tr.stats.sampling_rate / max_sampling_rate))
                if factor > 1:
                    tr.decimate(factor)  # Applies low-pass filter internally
                    tr.stats.processing.append(f"Decimated by factor {factor} to {tr.stats.sampling_rate} Hz")
                else:
                    tr.stats.processing.append("Sampling rate OK, no decimation needed")
            except Exception as e:
                tr.stats.processing.append(f"Decimation failed: {e}")
                continue

        # Apply sanitization: trim zeros, mask zeros/NaNs, unmask short gaps if requested
        tr = sanitize_trace(tr, unmask_short_zeros=unmask_short_zeros, zero_gap_threshold=zero_gap_threshold)
        tr = mask_gaps(tr)
        if len(tr.data) == 0:
            continue
        stream_out.append(tr)

    # Final optional cleanup
    try:
        stream_out = sanitize_stream(stream_out)
    except ImportError:
        pass

    # Split or merge
    if split_on_mask:
        stream_out = stream_out.split()
    elif merge:
        stream_out, _ = smart_merge(stream_out)

    return stream_out