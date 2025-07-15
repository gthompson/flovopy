import numpy as np
from obspy import Stream, Trace, UTCDateTime, read
from obspy.core.trace import Stats
from collections import defaultdict
from obspy.core.util import AttribDict
import os
from itertools import groupby
from operator import itemgetter
from obspy.signal.util import _npts2nfft

"""
Suggested workflow: 
st = read_mseed_with_gap_masking(mseedfile, zero_gap_threshold=500) # get data with actual gaps
for tr in st:
    tr = smart_fill(tr, short_thresh=5.0)
    tr.detrend("linear")
    tr.filter("bandpass", freqmin=1, freqmax=10)
write_safely(st, "processed.mseed", fill_value=0.0, overwrite_ok=True)
"""

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


def write_safely(tr, mseedfile, fill_value=0.0, overwrite_ok=False):
    """
    Writes a Trace or Stream to MiniSEED, filling masked gaps and recording metadata.
    Avoids overwriting by default, writing indexed files instead.
    """
    from obspy import Stream

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






def read_mseed_with_gap_masking(mseedfile, fill_value=0.0, zero_gap_threshold=100, split_on_mask=False):
    """
    Reads a MiniSEED file and applies gap masking to all contained Traces.
    Also trims leading/trailing zero padding and identifies long spans of zeros to mask as gaps.

    Parameters
    ----------
    mseedfile : str
        Path to the MiniSEED file to read.
    fill_value : float, optional
        Value used to fill gaps in unmasked data. Default is 0.0.
    zero_gap_threshold : int, optional
        Minimum number of consecutive 0.0 samples to treat as a gap. Default is 100.
    split_on_mask : bool, optional
        If True, splits the stream at masked gaps using `Stream.split()`. Default is True.

    Returns
    -------
    Stream
        ObsPy Stream with trimmed, masked, and optionally split Traces.
    """
    stream_in = read(mseedfile)
    stream_out = Stream()

    for tr in stream_in:
        data = np.asarray(tr.data, dtype=np.float32)

        # --- Trim leading/trailing zeros ---
        nonzero_indices = np.where(data != 0.0)[0]
        if len(nonzero_indices) == 0:
            continue  # All zeros, skip
        start, end = nonzero_indices[0], nonzero_indices[-1] + 1
        data = data[start:end]
        tr.stats.starttime += start / tr.stats.sampling_rate
        tr.data = data

        # --- Detect long internal spans of zeros and mask them ---
        zero_mask = data == 0.0
        if not np.any(zero_mask):
            tr.data = np.ma.masked_array(data, mask=np.zeros(len(data), dtype=bool))
            stream_out.append(tr)
            continue

        # Find spans of consecutive zeros
        zero_indices = np.where(zero_mask)[0]
        groups = []
        for k, g in groupby(enumerate(zero_indices), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) >= zero_gap_threshold:
                groups.append((group[0], group[-1]))

        # No spans long enough to mask
        if not groups:
            tr.data = np.ma.masked_array(data, mask=np.zeros(len(data), dtype=bool))
            stream_out.append(tr)
            continue

        # Apply mask to long zero spans
        gap_mask = np.zeros(len(data), dtype=bool)
        for start_idx, end_idx in groups:
            gap_mask[start_idx:end_idx + 1] = True
            tr.stats.processing.append(
                f"Heuristic gap: {tr.stats.starttime + start_idx / tr.stats.sampling_rate} "
                f"to {tr.stats.starttime + end_idx / tr.stats.sampling_rate} (zero padding)"
            )

        tr.data = np.ma.masked_array(data, mask=gap_mask)
        stream_out.append(tr)

    # --- Optionally split at masked gaps ---
    if split_on_mask:
        stream_out = stream_out.split()

    return stream_out



'''
def fill_masked_gaps(stream, fill_value=0.0):
    """
    Converts masked arrays in Stream traces to unmasked arrays by filling gaps.

    Parameters:
    -----------
    stream : obspy.Stream
        Stream with masked gaps (i.e., from read_mseed_with_gap_masking()).
    fill_value : float
        Value to use for filling masked gaps (e.g., 0.0 or np.nan).

    Returns:
    --------
    obspy.Stream
        A new Stream with gaps filled and no masked arrays.
    """
    filled_stream = Stream()
    for tr in stream:
        new_tr = tr.copy()
        if np.ma.isMaskedArray(new_tr.data):
            new_tr.data = new_tr.data.filled(fill_value).astype(np.float32)
        filled_stream.append(new_tr)
    return filled_stream
'''

def ensure_float32(tr):
    """Convert trace data to float32 if not already."""
    if not np.issubdtype(tr.data.dtype, np.floating) or tr.data.dtype != np.float32:
        tr.data = tr.data.astype(np.float32)

def sanitize_trace(trace):
    """
    Convert Trace data to float32 and replace zeros with masked values.
    """
    data = trace.data.astype(np.float32)
    masked_data = np.ma.masked_equal(data, 0.0)
    trace.data = masked_data
    return trace


def trace_equals(trace1, trace2, rtol=1e-5, atol=1e-8):
    """
    Compare two ObsPy Trace objects for equality, ignoring zeros and gaps.
    """
    if trace1.id != trace2.id:
        return False
    if abs(trace1.stats.starttime - trace2.stats.starttime) > trace1.stats.delta / 4:
        return False
    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        return False

    t1 = sanitize_trace(trace1.copy())
    t2 = sanitize_trace(trace2.copy())

    if len(t1.data) != len(t2.data):
        return False

    return np.allclose(t1.data.filled(np.nan), t2.data.filled(np.nan), rtol=rtol, atol=atol, equal_nan=True)


def merge_two_traces(trace1, trace2):
    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        raise ValueError("Sampling rates do not match")

    delta = trace1.stats.delta
    t_start = min(trace1.stats.starttime, trace2.stats.starttime)
    t_end = max(trace1.stats.endtime, trace2.stats.endtime)

    npts = int(((t_end - t_start) * trace1.stats.sampling_rate)) + 1
    new_data = np.ma.masked_all(npts, dtype=np.float32)

    def insert_data(trace):
        offset = int((trace.stats.starttime - t_start) * trace.stats.sampling_rate)
        length = trace.stats.npts
        data = sanitize_trace(trace.copy()).data
        existing = new_data[offset:offset + length]
        combined = np.where(existing.mask, data, np.where(data.mask, existing, data))
        new_data[offset:offset + length] = combined

    insert_data(trace1)
    insert_data(trace2)

    stats = trace1.stats.copy()
    stats.starttime = t_start
    stats.npts = npts
    return Trace(data=new_data, header=stats)

def smart_merge(traces, max_gap=None, debug=False):
    merged = Stream()
    traces_by_id = defaultdict(list)
    status_report = {
        'merged': Stream(),
        'conflicts': [],
        'duplicates': [],
        'status': 'ok',
        'message': ''
    }

    for tr in traces:
        sanitized = sanitize_trace(tr)
        if sanitized:
            traces_by_id[sanitized.id].append(sanitized)

    for trace_id, trace_list in traces_by_id.items():
        trace_list.sort(key=lambda t: t.stats.starttime)
        current = trace_list[0]
        for next_tr in trace_list[1:]:
            if trace_equals(current, next_tr):
                if debug:
                    print(f"Duplicate trace found for {trace_id} — skipping")
                status_report['duplicates'].append((current, next_tr))
                continue

            overlap = current.stats.endtime >= next_tr.stats.starttime
            gap = next_tr.stats.starttime - current.stats.endtime

            if overlap or (max_gap is not None and gap <= max_gap):
                try:
                    current = merge_two_traces(current, next_tr)
                except Exception as e:
                    if debug:
                        print(f"Conflict merging {trace_id}: {e}")
                    status_report['conflicts'].append((current, next_tr))
                    status_report['status'] = 'conflict'
                    status_report['message'] = f"Conflict while merging {trace_id}"
                    continue
            else:
                merged.append(current)
                current = next_tr
        merged.append(current)

    status_report['merged'] = merged
    return merged, status_report





'''
def traces_overlap_and_conflict(tr1, tr2) -> tuple[bool, bool]:
    """
    Checks whether two traces overlap, and if so, whether their overlapping data differs.

    Returns:
        has_overlap (bool): True if traces share any time range.
        has_conflict (bool): True if overlapping samples differ.
    """
    latest_start = max(tr1.stats.starttime, tr2.stats.starttime)
    earliest_end = min(tr1.stats.endtime, tr2.stats.endtime)

    if latest_start >= earliest_end:
        return False, False  # No overlap, no conflict

    tr1_overlap = tr1.slice(starttime=latest_start, endtime=earliest_end, nearest_sample=True)
    tr2_overlap = tr2.slice(starttime=latest_start, endtime=earliest_end, nearest_sample=True)

    conflict = len(tr1_overlap.data) != len(tr2_overlap.data) or not np.allclose(tr1_overlap.data, tr2_overlap.data)
    return True, conflict
'''

def smart_fill(tr: Trace, short_thresh=5.0):
    short, long = classify_gaps(tr, threshold_seconds=short_thresh)
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()
    data = tr.data.copy()

    # Fill short gaps with interpolation
    for s, e in short:
        x = np.arange(len(data))
        valid = ~data.mask
        data[s:e] = np.interp(x[s:e], x[valid], data[valid])

    # Fill long gaps with 0.0 or NaN
    for s, e in long:
        data[s:e] = 0.0

    filled = tr.copy()
    filled.data = data.filled(0.0).astype(np.float32)
    return filled

def classify_gaps(tr: Trace, sampling_rate=None, threshold_seconds=5.0):
    """
    Returns indices of short vs. long masked spans based on duration threshold.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return [], []

    sampling_rate = sampling_rate or tr.stats.sampling_rate
    mask = tr.data.mask
    gaps = []
    in_gap = False
    for i, m in enumerate(mask):
        if m and not in_gap:
            start = i
            in_gap = True
        elif not m and in_gap:
            end = i
            in_gap = False
            duration = (end - start) / sampling_rate
            gaps.append((start, end, duration))

    short = [(s, e) for s, e, d in gaps if d <= threshold_seconds]
    long = [(s, e) for s, e, d in gaps if d > threshold_seconds]
    return short, long


def fill_stream_gaps(stream, method="constant", **kwargs):
    """
    Apply a gap-filling method to all Traces in a Stream.

    Parameters:
    -----------
    stream : Stream
        ObsPy Stream object with Traces (preferably with masked gaps).
    method : str
        Gap filling method. Options: "linear", "previous", "noise".
    kwargs : dict
        Additional keyword arguments passed to the selected method.

    Returns:
    --------
    Stream
        Gap-filled stream.
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


def fill_gaps_with_constant(tr: Trace, fill_value: float = 0.0, **kwargs) -> Trace:
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
    new_tr.data = tr.data.filled(fill_value).astype(np.float32)
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
    Fill masked gaps by repeating the previous valid data value.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    data = new_tr.data.astype(np.float32)
    mask = data.mask

    if not np.any(mask):
        return new_tr

    for i in range(1, len(data)):
        if mask[i]:
            data[i] = data[i - 1]
    new_tr.data = data.data
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


def _generate_spectrally_matched_noise(tr: Trace, taper_percentage=0.2, **kwargs):
    """
    Generate noise matched to the spectral characteristics of the unmasked signal.
    """
    data = tr.data
    if not isinstance(data, np.ma.MaskedArray):
        data = np.ma.masked_array(data)

    unmasked = data[~data.mask]
    nfft = _npts2nfft(len(data))
    fft_vals = np.fft.rfft(unmasked, n=nfft)
    amplitude_spectrum = np.abs(fft_vals)

    random_phases = np.exp(2j * np.pi * np.random.rand(len(amplitude_spectrum)))
    synthetic_fft = amplitude_spectrum * random_phases

    synthetic_noise = np.fft.irfft(synthetic_fft, n=nfft).real[:len(data)]

    # Apply taper to reduce edge effects
    npts = len(data)
    taper_len = int(npts * taper_percentage / 2)
    taper = np.ones(npts)
    taper[:taper_len] = np.linspace(0, 1, taper_len)
    taper[-taper_len:] = np.linspace(1, 0, taper_len)

    return synthetic_noise * taper

def prepare_stream_for_analysis(st: Stream,
                                zero_gap_threshold=500,
                                artifact_kwargs=None,
                                fill_method="smart") -> Stream:
    """
    Detect and correct artifacts, fill short gaps, and return a merged, clean Stream.

    Parameters:
    -----------
    st : Stream
        Input Stream object.
    zero_gap_threshold : int
        Number of consecutive zeros to consider a gap.
    artifact_kwargs : dict
        Keyword args for `_detect_and_correct_artifacts()`.
    fill_method : str
        How to fill gaps: "smart", "interpolate", "zero", etc.

    Returns:
    --------
    Stream
        Cleaned, filled, and merged stream.
    """
    # 1. Mask zeros
    st = mask_zeros_as_gaps(st, zero_gap_threshold=zero_gap_threshold)

    # 2. Correct artifacts per trace
    for tr in st:
        detect_and_correct_artifacts(tr, **(artifact_kwargs or {}))

    # 3. Merge traces and fill gaps
    st.merge(method=1, fill_value=None)  # Preserve masks
    st = smart_fill(st, method=fill_method)

    return st