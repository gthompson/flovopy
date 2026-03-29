from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from obspy import read, Stream, Trace

from flovopy.core.gaputils import unmask_gaps
from flovopy.core.trace_utils import (
    sanitize_stream,
    remove_empty_traces,
    downsample_stream_to_common_rate,
    ensure_float32,
    streams_equal,
    ensure_masked,
)

from typing import Callable, Optional

def _as_stream(obj: Stream | Trace) -> Stream:
    """
    Normalize an ObsPy Trace or Stream to a Stream.

    Parameters
    ----------
    obj
        Input ObsPy Trace or Stream.

    Returns
    -------
    obspy.Stream
        A Stream view of the input.

    Raises
    ------
    TypeError
        If the input is neither a Trace nor a Stream.
    """
    if isinstance(obj, Stream):
        return obj
    if isinstance(obj, Trace):
        return Stream([obj])
    raise TypeError("Input must be an ObsPy Trace or Stream.")


def _return_same_type(original: Stream | Trace, st: Stream) -> Stream | Trace:
    """
    Return output in the same top-level type as the input.

    Parameters
    ----------
    original
        Original object passed by caller.
    st
        Processed Stream.

    Returns
    -------
    obspy.Stream or obspy.Trace

    Raises
    ------
    ValueError
        If the input was a Trace but processing removed it or split it into
        multiple traces.
    """
    if isinstance(original, Trace):
        if len(st) == 0:
            raise ValueError("Processing removed the only trace.")
        if len(st) > 1:
            raise ValueError("Processing split one Trace into multiple traces.")
        return st[0]
    return st


def _downcast_float64_to_float32(obj: Stream | Trace) -> Stream | Trace:
    """
    Downcast float64 trace data to float32.

    This is mainly useful before MiniSEED writing, because float64 MiniSEED
    can become unnecessarily large.

    Parameters
    ----------
    obj
        Input Trace or Stream.

    Returns
    -------
    obspy.Stream or obspy.Trace
        Same top-level type as input, with float64 arrays converted to float32.
    """
    st = _as_stream(obj.copy())
    for tr in st:
        if np.issubdtype(tr.data.dtype, np.floating) and tr.data.dtype == np.float64:
            tr.data = tr.data.astype(np.float32)
    return _return_same_type(obj, st)


def merge_max(st: Stream) -> Stream:
    """
    Merge traces with identical IDs by taking the sample-wise maximum
    where overlaps occur.

    This is a conservative fallback merge mode sometimes useful when
    overlapping traces differ slightly and a direct ObsPy merge is not
    satisfactory.

    Parameters
    ----------
    st
        Input stream. Traces are grouped by full SEED id.

    Returns
    -------
    obspy.Stream
        Merged stream with one trace per id where possible.
    """
    grouped = defaultdict(list)
    for tr in st:
        grouped[tr.id].append(tr)

    out = Stream()

    for seed_id, traces in grouped.items():
        traces = sorted(traces, key=lambda tr: tr.stats.starttime)
        if len(traces) == 1:
            out.append(traces[0].copy())
            continue

        # Start from an ObsPy merge onto a single masked trace
        tmp = Stream([tr.copy() for tr in traces])
        tmp.merge(method=1, fill_value=np.nan)

        if len(tmp) == 1:
            out.append(tmp[0])
            continue

        # Fallback: build a unified time axis at the first trace's dt
        ref = traces[0]
        dt = ref.stats.delta
        start = min(tr.stats.starttime for tr in traces)
        end = max(tr.stats.endtime for tr in traces)
        npts = int(round((end - start) / dt)) + 1

        merged = np.full(npts, np.nan, dtype=np.float32)

        for tr in traces:
            i0 = int(round((tr.stats.starttime - start) / dt))
            i1 = i0 + tr.stats.npts
            data = tr.data.astype(np.float32, copy=False)

            existing = merged[i0:i1]
            if existing.shape != data.shape:
                n = min(len(existing), len(data))
                existing = existing[:n]
                data = data[:n]
                merged[i0:i0 + n] = np.nanmax(np.vstack([existing, data]), axis=0)
            else:
                merged[i0:i1] = np.nanmax(np.vstack([existing, data]), axis=0)

        new_tr = ref.copy()
        new_tr.stats.starttime = start
        new_tr.data = merged
        out.append(new_tr)

    return out


def smart_merge(
    stream_in: Stream,
    *,
    strategy: str = "obspy",
    sanitize_before_merge: bool = True,
    harmonize_rates: bool = False,
    max_sampling_rate: float | None = None,
    skip_low_rate_channels: bool = False,
    allow_timeshift: bool = False,
    max_shift_seconds: int = 2,
    force_close_sampling_rates_on_merge_failure: bool = True,
    sampling_rate_tolerance_hz: float = 0.1,
    return_stream_only: bool = True,
    verbose: bool = False,
) -> dict | Stream:
    """
    Merge traces in a stream using a selected strategy.

    Parameters
    ----------
    stream_in
        Input stream.
    strategy
        Merge strategy:
            - "obspy": standard ObsPy merge
            - "max": merge overlaps using sample-wise maximum
            - "both": try ObsPy first, fall back to max merge if needed
    sanitize_before_merge
        If True, run ``sanitize_stream()`` on each grouped substream before merge.
        This may remove empty traces and duplicates, depending on the current
        behavior of ``sanitize_stream()``.
    harmonize_rates
        If True, downsample each grouped substream to a common sampling rate
        before merge.
    max_sampling_rate
        Optional cap passed to ``downsample_stream_to_common_rate()`` when
        ``harmonize_rates=True``.
    skip_low_rate_channels
        Passed to ``downsample_stream_to_common_rate()`` when
        ``harmonize_rates=True``.
    allow_timeshift
        Reserved for future logic allowing small timing shifts before merge.
        Currently accepted for API compatibility but not actively used.
    max_shift_seconds
        Reserved for future timing-shift logic.
    force_close_sampling_rates_on_merge_failure
        If True, and an ObsPy merge fails, check whether all sampling rates in
        the grouped substream lie within ``sampling_rate_tolerance_hz`` of their
        weighted mean. If so, reset all trace sampling rates to that weighted
        mean and retry the ObsPy merge.
    sampling_rate_tolerance_hz
        Maximum allowed absolute deviation from the weighted mean sampling rate
        for the fallback rate-forcing logic to apply.
    return_stream_only
        If True, return only the merged stream. If False, return a report dict.
    verbose
        If True, print diagnostics.

    Returns
    -------
    dict or Stream
        If ``return_stream_only=False`` (default), returns a summary report
        containing:
            - merged_stream
            - n_input_traces
            - n_output_traces
            - strategy
            - ids
            - harmonize_rates
            - allow_timeshift
            - force_close_sampling_rates_on_merge_failure
            - sampling_rate_tolerance_hz
        If ``return_stream_only=True``, returns only the merged Stream.

    Notes
    -----
    - Traces are merged independently by SEED id.
    - ``allow_timeshift`` / ``max_shift_seconds`` are currently placeholders;
      they are retained for API compatibility but are not yet implemented.
    """

    from collections import defaultdict

    def log(msg: str):
        if verbose:
            print(msg)

    def _finalize(result_stream: Stream):
        if return_stream_only:
            return result_stream
        return {
            "merged_stream": result_stream,
            "n_input_traces": len(stream_in),
            "n_output_traces": len(result_stream),
            "strategy": strategy,
            "ids": sorted(set(tr.id for tr in result_stream)),
            "harmonize_rates": harmonize_rates,
            "allow_timeshift": allow_timeshift,
            "force_close_sampling_rates_on_merge_failure": force_close_sampling_rates_on_merge_failure,
            "sampling_rate_tolerance_hz": sampling_rate_tolerance_hz,
        }

    def _weighted_mean_sampling_rate(st: Stream) -> float | None:
        rates = []
        weights = []
        for tr in st:
            try:
                sr = float(tr.stats.sampling_rate)
                npts = int(getattr(tr.stats, "npts", len(tr.data)))
            except Exception:
                continue
            if np.isfinite(sr) and npts > 0:
                rates.append(sr)
                weights.append(npts)

        if not rates:
            return None

        return float(np.average(rates, weights=weights))

    def _can_force_sampling_rates_to_weighted_mean(
        st: Stream,
        tol_hz: float,
    ) -> tuple[bool, float | None]:
        mean_sr = _weighted_mean_sampling_rate(st)
        if mean_sr is None:
            return False, None

        for tr in st:
            try:
                sr = float(tr.stats.sampling_rate)
            except Exception:
                return False, None
            if not np.isfinite(sr):
                return False, None
            if abs(sr - mean_sr) > tol_hz:
                return False, mean_sr

        return True, mean_sr

    def _try_obspy_merge_with_optional_rate_fix(st: Stream, seed_id: str) -> Stream:
        tmp = st.copy()
        try:
            tmp.merge(method=1, fill_value=np.nan)
            return tmp
        except Exception as e:
            log(f"ObsPy merge failed for {seed_id}: {e}")

        if not force_close_sampling_rates_on_merge_failure:
            raise

        ok, mean_sr = _can_force_sampling_rates_to_weighted_mean(
            tmp,
            sampling_rate_tolerance_hz,
        )
        if not ok or mean_sr is None:
            log(
                f"[smart_merge] Sampling-rate fallback not applied for {seed_id}: "
                f"rates differ by more than ±{sampling_rate_tolerance_hz} Hz."
            )
            raise

        log(
            f"[smart_merge] Retrying merge for {seed_id} after forcing all sampling "
            f"rates to weighted mean {mean_sr:.6f} Hz "
            f"(tolerance ±{sampling_rate_tolerance_hz} Hz)."
        )

        for tr in tmp:
            tr.stats.sampling_rate = mean_sr

        tmp.merge(method=1, fill_value=np.nan)
        return tmp

    if len(stream_in) == 0:
        return _finalize(Stream())

    if allow_timeshift:
        log(
            f"[smart_merge] allow_timeshift=True requested, but time-shift "
            f"alignment is not yet implemented (max_shift_seconds={max_shift_seconds})."
        )

    grouped = defaultdict(Stream)
    for tr in stream_in:
        grouped[tr.id].append(tr)

    merged_out = Stream()

    for seed_id, substream in grouped.items():
        substream = substream.copy()

        if sanitize_before_merge:
            sanitize_stream(substream)

        if len(substream) == 0:
            continue

        if harmonize_rates and len(substream) > 1:
            try:
                substream = downsample_stream_to_common_rate(
                    substream,
                    inplace=False,
                    max_sampling_rate=max_sampling_rate,
                    skip_low_rate_channels=skip_low_rate_channels,
                    verbose=verbose,
                )
            except Exception as e:
                log(f"[smart_merge] Rate harmonization failed for {seed_id}: {e}")

        if len(substream) == 0:
            continue

        if len(substream) == 1:
            merged_out.append(substream[0])
            continue

        log(f"Merging {seed_id}: {len(substream)} traces")

        if strategy == "obspy":
            tmp = _try_obspy_merge_with_optional_rate_fix(substream, seed_id)
            merged_out.extend(tmp)

        elif strategy == "max":
            merged_out.extend(merge_max(substream))

        elif strategy == "both":
            try:
                tmp = _try_obspy_merge_with_optional_rate_fix(substream, seed_id)
                merged_out.extend(tmp)
            except Exception as e:
                log(f"ObsPy merge failed for {seed_id}: {e}; falling back to merge_max().")
                merged_out.extend(merge_max(substream))

        else:
            raise ValueError(f"Unknown merge strategy: {strategy!r}")

    return _finalize(merged_out)


def write_mseed(
    tr: Stream | Trace,
    mseedfile: str,
    *,
    fill_value: float = 0.0,
    overwrite_ok: bool = True,
    pickle_fallback: bool = False,
    encoding: Optional[str] = None,
    reclen: Optional[int] = None,
    bypass_processing: bool = False,
    use_preprocess_pipeline: bool = False,
    verbose: bool = False,
    **pipeline_kwargs,
):
    """
    Write a Trace or Stream to MiniSEED.

    This function can operate in two modes:

    1. Low-level mode (default):
       - unmask masked gaps with a constant fill value
       - optionally downcast float64 to float32
       - write directly with ObsPy

    2. Pipeline mode:
       - call `preprocess_stream_before_write()` from the shared FLOVOpy
         waveform pipeline before writing

    Parameters
    ----------
    tr
        Trace or Stream to write.
    mseedfile
        Output MiniSEED filename.
    fill_value
        Fill value used when replacing masked gaps before writing.
    overwrite_ok
        If True, overwrite the requested output path.
        If False, attempt indexed filenames like `.01`, `.02`, ...
    pickle_fallback
        If True, write a `.pkl` fallback if MiniSEED writing fails.
    encoding
        MiniSEED encoding passed to ObsPy, e.g. "STEIM2", "FLOAT32".
        If None, ObsPy chooses based on dtype.
    reclen
        MiniSEED record length passed to ObsPy.
    bypass_processing
        If True, bypass FLOVOpy pre-write processing and write raw ObsPy data.
    use_preprocess_pipeline
        If True, call `preprocess_stream_before_write()` before writing.
    verbose
        Print diagnostics.
    **pipeline_kwargs
        Additional keyword arguments passed to
        `preprocess_stream_before_write()`.

    Returns
    -------
    str or None
        Path written, or None on failure.
    """
    outpath = Path(mseedfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    obj = tr.copy()

    if use_preprocess_pipeline and not bypass_processing:

        obj = preprocess_stream_before_write(
            obj,
            bypass_processing=bypass_processing,
            fill_value=fill_value,
            **pipeline_kwargs,
        )
    else:
        if isinstance(obj, Stream):
            obj = Stream([unmask_gaps(t, fill_value=fill_value, inplace=False) for t in obj])
        elif isinstance(obj, Trace):
            obj = unmask_gaps(obj, fill_value=fill_value, inplace=False)
        else:
            raise TypeError("Input must be an ObsPy Trace or Stream")

        enc_upper = encoding.upper() if isinstance(encoding, str) else None
        if enc_upper != "FLOAT64":
            obj = _downcast_float64_to_float32(obj)

    write_kwargs = {"format": "MSEED"}
    if encoding is not None:
        write_kwargs["encoding"] = encoding
    if reclen is not None:
        write_kwargs["reclen"] = reclen

    def _do_write(target):
        obj.write(str(target), **write_kwargs)

    if overwrite_ok:
        try:
            _do_write(outpath)
            return str(outpath)
        except Exception as e:
            if verbose:
                print(f"[write_mseed] Failed to write {outpath}: {e}")

    else:
        base, ext = os.path.splitext(str(outpath))
        for index in range(1, 100):
            indexed = f"{base}.{index:02d}{ext}"
            if not os.path.exists(indexed):
                try:
                    _do_write(indexed)
                    if verbose:
                        print(f"[write_mseed] Indexed file written: {indexed}")
                    return indexed
                except Exception as e:
                    if verbose:
                        print(f"[write_mseed] Failed indexed write {indexed}: {e}")
                    break

    if pickle_fallback:
        pklfile = str(outpath) + ".pkl"
        if not os.path.exists(pklfile):
            try:
                obj.write(pklfile, format="PICKLE")
                if verbose:
                    print(f"[write_mseed] Fallback pickle written: {pklfile}")
                return pklfile
            except Exception as e:
                if verbose:
                    print(f"[write_mseed] Failed fallback pickle write: {e}")

    return None


def read_mseed(
    mseedfile,
    fill_value=0.0,
    min_gap_duration_s=1.0,
    split_on_mask=False,
    merge=True,
    merge_strategy="obspy",
    starttime=None,
    endtime=None,
    min_sampling_rate=50.0,
    max_sampling_rate=250.0,
    unmask_short_zeros=True,
    bypass_processing=False,
    use_postprocess_pipeline=False,
    verbose=False,
    **pipeline_kwargs,
):
    """
    Read a MiniSEED file and apply optional trimming, masking, splitting, and merging.

    Parameters
    ----------
    mseedfile
        Path to MiniSEED file.
    fill_value
        Fill value used if falling back to ObsPy merge.
    min_gap_duration_s
        Passed to `sanitize_stream()`.
    split_on_mask
        If True, split at masked gaps.
    merge
        If True, merge traces after read.
    merge_strategy
        Merge strategy for `smart_merge()`: "obspy", "max", or "both".
    starttime, endtime
        Optional time window for trimming.
    min_sampling_rate
        Remove traces with lower sample rates.
    max_sampling_rate
        If not None, optionally downsample stream to a common rate capped here.
    unmask_short_zeros
        Passed to `sanitize_stream()`.
    bypass_processing
        If True, bypass FLOVOpy post-read processing and return raw ObsPy data.
    use_postprocess_pipeline
        If True, call `postprocess_stream_after_read()` after reading.
    verbose
        Print diagnostics.
    **pipeline_kwargs
        Additional keyword arguments passed to
        `postprocess_stream_after_read()`.

    Returns
    -------
    obspy.Stream
        Processed stream.
    """
    try:
        stream = read(mseedfile, format="MSEED")
    except Exception:
        try:
            stream = read(mseedfile)
        except Exception as e:
            if verbose:
                print(f"[read_mseed] Failed to read {mseedfile}: {e}")
            return Stream()

    if len(stream) == 0:
        return Stream()

    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    if min_sampling_rate is not None:
        stream = Stream([
            tr for tr in stream
            if float(getattr(tr.stats, "sampling_rate", 0.0)) >= min_sampling_rate
        ])

    if len(stream) == 0:
        return Stream()

    if use_postprocess_pipeline and not bypass_processing:

        stream = postprocess_stream_after_read(
            stream,
            bypass_processing=bypass_processing,
            sanitize=True,
            drop_empty=True,
            harmonize_rates=False,
            **pipeline_kwargs,
        )
    else:
        if max_sampling_rate is not None:
            sampling_rates = [
                float(tr.stats.sampling_rate)
                for tr in stream
                if hasattr(tr.stats, "sampling_rate")
            ]
            if sampling_rates and max(sampling_rates) > max_sampling_rate:
                downsample_stream_to_common_rate(
                    stream,
                    inplace=True,
                    max_sampling_rate=max_sampling_rate,
                )

        sanitize_stream(
            stream,
            unmask_short_zeros=unmask_short_zeros,
            min_gap_duration_s=min_gap_duration_s,
        )

    if split_on_mask:
        stream = stream.split()
    elif merge:
        try:
            report = smart_merge(stream, strategy=merge_strategy, verbose=verbose)
            stream = report["merged_stream"]
        except Exception as e:
            if verbose:
                print(f"[read_mseed] smart_merge failed: {e}; falling back to stream.merge()")
            stream.merge(method=1, fill_value=fill_value)

    return stream


def compare_mseed_files(src_file, dest_file, **read_kwargs):
    """
    Compare two MiniSEED files after reading them through `read_mseed()`.

    Parameters
    ----------
    src_file, dest_file
        Paths to files to compare.
    **read_kwargs
        Keyword arguments passed through to `read_mseed()` for both files.

    Returns
    -------
    tuple[bool, str | None]
        (equal, error_message)
    """
    try:
        src_stream = read_mseed(src_file, **read_kwargs)
        dest_stream = read_mseed(dest_file, **read_kwargs)
        return streams_equal(src_stream, dest_stream), None
    except Exception as e:
        return False, str(e)


def can_write_to_miniseed_and_read_back(tr, return_metrics=True, verbose=False):
    """
    Test whether a trace can be written to and read back from MiniSEED.

    This is useful as a lightweight compatibility check before export.

    Parameters
    ----------
    tr
        Input ObsPy Trace.
    return_metrics
        If True, store any derived diagnostics back onto `tr.stats`
        when available.
    verbose
        Print diagnostics.

    Returns
    -------
    bool
        True if round-trip succeeded, False otherwise.
    """
    try:
        from obspy.signal.quality_control import MSEEDMetadata
    except Exception:
        MSEEDMetadata = None

    tr_test = tr.copy()

    if len(tr_test.data) == 0:
        if return_metrics:
            tr.stats["quality_factor"] = 0.0
        return False

    ensure_float32(tr_test)

    if hasattr(tr_test.stats, "mseed") and "encoding" in tr_test.stats.mseed:
        del tr_test.stats.mseed["encoding"]

    fd, tmpfilename = tempfile.mkstemp(suffix=".mseed")
    os.close(fd)

    try:
        tr_test.write(tmpfilename, format="MSEED")
        _ = read(tmpfilename)

        if return_metrics and MSEEDMetadata is not None:
            try:
                mseedqc = MSEEDMetadata([tmpfilename])
                tr.stats["metrics"] = mseedqc.meta
            except Exception:
                pass

        return True

    except Exception as e:
        if return_metrics:
            tr.stats["quality_factor"] = 0.0
        if verbose:
            print(f"[can_write_to_miniseed_and_read_back] Failed for {tr.id}: {e}")
        return False

    finally:
        if os.path.exists(tmpfilename):
            os.remove(tmpfilename)





def _as_stream(obj: Stream | Trace) -> Stream:
    if isinstance(obj, Stream):
        return obj
    if isinstance(obj, Trace):
        return Stream([obj])
    raise TypeError("Expected an ObsPy Stream or Trace.")


def _return_same_type(original: Stream | Trace, st: Stream) -> Stream | Trace:
    if isinstance(original, Trace):
        if len(st) == 0:
            raise ValueError("Processing removed the only trace.")
        if len(st) > 1:
            raise ValueError("Processing split one Trace into multiple traces.")
        return st[0]
    return st

def postprocess_stream_after_read(
    obj: Stream | Trace,
    *,
    bypass_processing: bool = False,
    copy: bool = True,
    ensure_float32_data: bool = True,
    ensure_masked_data: bool = False,
    sanitize: bool = True,
    drop_empty: bool = True,
    fix_ids: bool = False,
    legacy: bool = False,
    netcode: Optional[str] = None,
    trace_fixer: Optional[Callable[[Trace], None]] = None,
    merge: bool = False,
    merge_strategy: str = "obspy",
    allow_timeshift: bool = False,
    max_shift_seconds: int = 2,
    harmonize_rates: bool = False,
    min_sampling_rate: Optional[float] = None,
    max_sampling_rate: Optional[float] = None,
    verbose: bool = False,
) -> Stream | Trace:
    """
    Lightweight FLOVOpy post-read normalization for Streams/Traces.

    Parameters
    ----------
    obj
        Input Stream or Trace.
    bypass_processing
        If True, return early after optional copy.
    copy
        If True, operate on a copy.
    ensure_float32_data
        Convert data arrays to float32 where appropriate.
    ensure_masked_data
        Convert arrays to masked arrays where appropriate.
    sanitize
        Run general stream sanitization.
    drop_empty
        Remove empty traces before returning.
    fix_ids
        Apply ``fix_trace_id()`` if ``trace_fixer`` is not supplied.
    legacy
        Passed to ``fix_trace_id()``.
    netcode
        Passed to ``fix_trace_id()``.
    trace_fixer
        Optional custom trace-fixing function applied to each trace.
    merge
        If True, apply ``smart_merge()``.
    merge_strategy
        Merge strategy passed to ``smart_merge()``.
    allow_timeshift
        Passed to ``smart_merge()``.
    max_shift_seconds
        Passed to ``smart_merge()``.
    harmonize_rates
        If True, downsample traces to a common sampling rate.
    min_sampling_rate
        If not None, drop traces whose sampling rate is below this value.
    max_sampling_rate
        Upper bound passed to ``downsample_stream_to_common_rate()``.
    verbose
        If True, print warnings/debug information.

    Returns
    -------
    Stream | Trace
        Object of the same type as the input.
    """
    if bypass_processing:
        return obj.copy() if copy else obj

    original = obj
    st = _as_stream(obj.copy() if copy else obj)

    # ------------------------------------------------------------------
    # Per-trace dtype / masking normalization
    # ------------------------------------------------------------------
    for tr in st:
        if ensure_float32_data:
            ensure_float32(tr)
        if ensure_masked_data:
            ensure_masked(tr)

    # ------------------------------------------------------------------
    # General cleanup
    # ------------------------------------------------------------------
    if sanitize:
        sanitize_stream(st)

    if drop_empty:
        remove_empty_traces(st, inplace=True)

    # ------------------------------------------------------------------
    # Optional trace-ID / metadata fixing
    # ------------------------------------------------------------------
    if trace_fixer is not None:
        for tr in st:
            trace_fixer(tr)
    elif fix_ids:
        from flovopy.core.trace_utils import fix_trace_id
        for tr in st:
            fix_trace_id(tr, legacy=legacy, netcode=netcode, verbose=verbose)

    # ------------------------------------------------------------------
    # Cull low-rate traces before harmonization/merge
    # ------------------------------------------------------------------
    if min_sampling_rate is not None:
        for tr in list(st):
            try:
                if tr.stats.sampling_rate < min_sampling_rate:
                    st.remove(tr)
            except Exception as e:
                if verbose:
                    print(
                        f"[WARN] Could not evaluate sampling rate for "
                        f"{getattr(tr, 'id', 'unknown')}: {e}"
                    )

    if drop_empty:
        remove_empty_traces(st, inplace=True)

    # ------------------------------------------------------------------
    # Optional harmonization of sample rates
    # ------------------------------------------------------------------
    if harmonize_rates and len(st) > 1:
        downsample_stream_to_common_rate(
            st,
            inplace=True,
            max_sampling_rate=max_sampling_rate,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Optional merge
    # ------------------------------------------------------------------
    if merge and len(st) > 1:
        st = smart_merge(
            st,
            strategy=merge_strategy,
            sanitize_before_merge=False,   # already sanitized above
            harmonize_rates=False,         # already handled above if requested
            allow_timeshift=allow_timeshift,
            max_shift_seconds=max_shift_seconds,
            return_stream_only=True,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------
    if drop_empty:
        remove_empty_traces(st, inplace=True)

    return _return_same_type(original, st)


def preprocess_stream_before_write(
    obj: Stream | Trace,
    *,
    bypass_processing: bool = False,
    copy: bool = True,
    ensure_float32_data: bool = True,
    ensure_masked_data: bool = False,
    sanitize: bool = True,
    drop_empty: bool = True,
    fix_ids: bool = False,
    legacy: bool = False,
    netcode: Optional[str] = None,
    trace_fixer: Optional[Callable[[Trace], None]] = None,
    merge: bool = False,
    merge_strategy: str = "obspy",
    allow_timeshift: bool = False,
    max_shift_seconds: int = 2,
    harmonize_rates: bool = False,
    max_sampling_rate: Optional[float] = None,
    unmask_before_write: bool = True,
    fill_value: float = 0.0,
    verbose: bool = False,
) -> Stream | Trace:
    """
    Lightweight FLOVOpy pre-write normalization for Streams/Traces.

    Parameters
    ----------
    obj
        Input Stream or Trace.
    bypass_processing
        If True, return early after optional copy.
    copy
        If True, operate on a copy.
    ensure_float32_data
        Convert data arrays to float32 where appropriate.
    ensure_masked_data
        Convert arrays to masked arrays where appropriate.
    sanitize
        Run general stream sanitization.
    drop_empty
        Remove empty traces before returning.
    fix_ids
        Apply ``fix_trace_id()`` if ``trace_fixer`` is not supplied.
    legacy
        Passed to ``fix_trace_id()``.
    netcode
        Passed to ``fix_trace_id()``.
    trace_fixer
        Optional custom trace-fixing function applied to each trace.
    merge
        If True, apply ``smart_merge()``.
    merge_strategy
        Merge strategy passed to ``smart_merge()``.
    allow_timeshift
        Passed to ``smart_merge()``.
    max_shift_seconds
        Passed to ``smart_merge()``.
    harmonize_rates
        If True, downsample traces to a common sampling rate.
    max_sampling_rate
        Upper bound passed to ``downsample_stream_to_common_rate()``.
    unmask_before_write
        If True, replace masked gaps with ``fill_value`` before returning.
    fill_value
        Fill value used when unmasking masked gaps.
    verbose
        If True, print warnings/debug information.

    Returns
    -------
    Stream | Trace
        Object of the same type as the input.
    """
    if bypass_processing:
        return obj.copy() if copy else obj

    original = obj
    st = _as_stream(obj.copy() if copy else obj)

    # ------------------------------------------------------------------
    # Per-trace dtype / masking normalization
    # ------------------------------------------------------------------
    for tr in st:
        if ensure_float32_data:
            ensure_float32(tr)
        if ensure_masked_data:
            ensure_masked(tr)

    # ------------------------------------------------------------------
    # General cleanup
    # ------------------------------------------------------------------
    if sanitize:
        sanitize_stream(st)

    if drop_empty:
        remove_empty_traces(st, inplace=True)

    # ------------------------------------------------------------------
    # Optional trace-ID / metadata fixing
    # ------------------------------------------------------------------
    if trace_fixer is not None:
        for tr in st:
            trace_fixer(tr)
    elif fix_ids:
        from flovopy.core.trace_utils import fix_trace_id
        for tr in st:
            fix_trace_id(tr, legacy=legacy, netcode=netcode, verbose=verbose)

    if drop_empty:
        remove_empty_traces(st, inplace=True)

    # ------------------------------------------------------------------
    # Optional harmonization of sample rates
    # ------------------------------------------------------------------
    if harmonize_rates and len(st) > 1:
        downsample_stream_to_common_rate(
            st,
            inplace=True,
            max_sampling_rate=max_sampling_rate,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Optional merge
    # ------------------------------------------------------------------
    if merge and len(st) > 1:
        st = smart_merge(
            st,
            strategy=merge_strategy,
            sanitize_before_merge=False,   # already sanitized above
            harmonize_rates=False,         # already handled above if requested
            allow_timeshift=allow_timeshift,
            max_shift_seconds=max_shift_seconds,
            return_stream_only=True,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Convert masked gaps to plain arrays before write
    # ------------------------------------------------------------------
    if unmask_before_write:
        st2 = Stream()
        for tr in st:
            st2.append(unmask_gaps(tr, fill_value=fill_value, inplace=False))
        st = st2

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------
    if drop_empty:
        remove_empty_traces(st, inplace=True)

    return _return_same_type(original, st)