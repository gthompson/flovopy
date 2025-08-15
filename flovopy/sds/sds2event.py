# flovopy/sds/pipeline.py
from __future__ import annotations

"""
SDS orchestration helpers for event windows.

This module is intentionally *thin* and SDS-specific. It provides convenience
functions that:
  1) Read time windows from an SDS archive via `SDSobj`.
  2) Normalize gaps and optionally filter / remove instrument response using the
     source-agnostic utilities in `flovopy.core.preprocess` and `flovopy.core.remove_response`.
  3) (Optionally) write normalized waveforms to MiniSEED files.

Design goals
------------
- Keep `flovopy.core` *free* of SDS dependencies. The dependency direction is:
    flovopy.sds  ──>  flovopy.core
- Provide two pragmatic “presets” (see below) so pipelines are reproducible:
    • archive_preset  : gap-normalize only (no filtering/response removal).
    • analysis_preset : gap-normalize + safe pad→taper→filter (+ optional response).
- Allow optional StationXML `inv` to be passed through for response removal
  in the analysis preset.

When to use which preset?
-------------------------
- **archive_preset**
    Use when you want to *preserve* the raw spectral content and simply make
    traces numerically “safe” for downstream tools that cannot handle masked
    arrays. It:
      - Interpolates only *short* masked gaps (≤ `small_gap_sec`).
      - Leaves long gaps masked during detrend, then fills them (default=0.0)
        to produce unmasked arrays.
      - Does NOT taper, filter, or remove instrument response.
    Recommended for:
      - Building event archives / catalogs.
      - Feature extraction that prefers original bandwidth.
      - Reproducible re-processing later with different filter settings.

- **analysis_preset**
    Use when you want data *ready* for detection/estimation visualizations or
    algorithms that expect a light bandpass (and possibly response removal).
    It:
      - Applies the same gap normalization as archive mode.
      - Then runs the safe pad → taper → filter → (optional response removal) → unpad
        sequence from `flovopy.core.remove_response.safe_pad_taper_filter`.
    Recommended for:
      - STA/LTA or matched-filter pipelines in a controlled band.
      - Visual review with reduced low-frequency drift / high-frequency noise.
    Notes:
      - If `inv` is provided, the instrument response is removed (with sensible
        `pre_filt` and a conservative water-level). For infrasound channels
        (e.g. channel code 'D*'), output is set to 'DEF' automatically.

Edge behavior / failure modes
-----------------------------
- If SDS returns no traces or all are empty, functions return an empty Stream.
- If filtering fails for a trace (e.g., invalid cutoff vs. Nyquist), that trace
  is dropped in `preprocess_stream` (core). The caller still receives other
  successful traces.
- Response removal requires consistent metadata in `inv`. If removal fails,
  the trace falls back to the filtered-but-not-deconvolved result only if
  failure happens *after* filtering—otherwise, it’s dropped (core behavior).

Performance notes
-----------------
- Reading performance is controlled by `speed` in `SDSobj.read`.
- Gap normalization is vectorized and typically fast even for multi-minute windows.
- Filtering uses ObsPy’s IIR tools. Corner count and zero-phase can increase CPU time.

Examples
--------
>>> from obspy import UTCDateTime
>>> from flovopy.sds.pipeline import load_event_stream_from_sds, sds_to_event_miniseed
>>> t1 = UTCDateTime("2016-08-19T05:00:00")
>>> t2 = UTCDateTime("2016-08-19T05:10:00")

# Load with “archive” preset (no filtering/response)
>>> st = load_event_stream_from_sds(
...     sds_root="/data/SDS",
...     t1=t1, t2=t2,
...     net="1R", sta="BCHH", loc="10", cha="D*",
...     preset="archive_preset", verbose=True,
... )

# One-shot: read, normalize (analysis preset), and write MiniSEED
>>> out = sds_to_event_miniseed(
...     sds_root="/data/SDS",
...     t1=t1, t2=t2,
...     out_dir="/tmp/mseed",
...     preset="analysis_preset",
...     net="1R", sta="BCHH", loc="10", cha="D*",
...     filename_template="{net}.{sta}.{loc}.{cha}.{t0}_{t1}.mseed",
...     verbose=True,
... )
"""

from typing import Tuple, Optional, Literal
from obspy import UTCDateTime, Stream
from flovopy.sds.sds import SDSobj

# Core, source-agnostic utilities
from flovopy.core.preprocess import preprocess_stream
from flovopy.core.miniseed_io import write_mseed  # your existing writer

Preset = Literal["archive_preset", "analysis_preset"]


def load_event_stream_from_sds(
    sds_root: str,
    t1: UTCDateTime,
    t2: UTCDateTime,
    *,
    net: str = "*",
    sta: str = "*",
    loc: str = "*",
    cha: str = "*",
    speed: int = 2,
    preset: Preset = "archive_preset",
    inv=None,
    verbose: bool = False,
) -> Stream:
    """
    Read a time window from an SDS archive and normalize it according to a preset.

    Parameters
    ----------
    sds_root : str
        Root directory of the SDS archive (e.g., '/data/SDS').
    t1, t2 : UTCDateTime
        Start and end of the desired time window.
    net, sta, loc, cha : str
        SDS selectors. Wildcards are allowed (e.g., '1R', '*', '10', 'D*').
    speed : int
        Passed to `SDSobj.read`; higher values can increase read performance
        (implementation-specific).
    preset : {"archive_preset", "analysis_preset"}
        - "archive_preset": gap-normalize only (no filtering/response).
        - "analysis_preset": gap-normalize + pad→taper→filter (+ optional response).
    inv : obspy.Inventory or None
        StationXML used for instrument response removal in the analysis preset.
        Ignored in archive preset.
    verbose : bool
        If True, prints per-trace summary lines.

    Returns
    -------
    obspy.Stream
        A normalized stream. May be empty if nothing was found or all traces failed.

    Notes
    -----
    - This does *not* merge/sanitize SDS segments—assumes `SDSobj.read` returns
      already merged traces (as is typical for MiniSEED daily archives).
    - For *analysis_preset*, the default bandpass is (0.5, 30.0) Hz, corners=4,
      zerophase=True. You can tune these in code below if you fork the preset.
    """
    if isinstance(sds_root, SDSobj):
        sdsin = sds_root
    else:
        sdsin = SDSobj(sds_root)
    #sdsin.read(t1, t2, net=net, sta=sta, loc=loc, cha=cha, speed=speed)
    # BEFORE (broken)
    # sdsin.read(t1, t2, net=net, sta=sta, loc=loc, cha=cha, speed=speed)

    # AFTER (works with your SDSobj.read signature)
    sdsin.read(
        t1, t2,
        speed=speed,
        verbose=verbose,
        # keep other kwargs only if SDSobj.read supports them
    )

    # post-filter by SEED selectors (wildcards OK)
    st = sdsin.stream.select(network=net, station=sta, location=loc, channel=cha).copy()
    st = sdsin.stream or Stream()
    st = Stream(tr for tr in st if tr.stats.npts > 0)

    if not len(st):
        if verbose:
            print(f"[SDS] 0 traces for {t1}–{t2}")
        return st

    if verbose:
        print(f"[SDS] {len(st)} traces for {t1}–{t2}")

    if preset == "archive_preset":
        # Gap normalize only; preserve spectral content for downstream choices.
        st = preprocess_stream(
            st,
            run_artifact_fix=False,
            normalize_gaps=True,
            small_gap_sec=2.0,
            long_gap_fill="zero",     # convert masks to zeros post-detrend
            piecewise_detrend=True,
            force_unmasked=True,      # ensure ndarray (many ops can’t handle masks)
            do_clean=False,           # <-- no pad/taper/filter/response in archive flavor
            verbose=verbose,
        )
    elif preset == "analysis_preset":
        # Normalize + light filtering (and response removal if inv provided).
        st = preprocess_stream(
            st,
            run_artifact_fix=False,   # set True if spikes/clipping are common
            normalize_gaps=True,
            small_gap_sec=2.0,
            long_gap_fill="zero",
            piecewise_detrend=True,
            force_unmasked=True,
            do_clean=True,            # <-- pad→taper→filter (+response if inv provided)
            taper_fraction=0.05,
            filter_type="bandpass",
            freq=(0.5, 30.0),
            corners=4,
            zerophase=True,
            inv=inv,
            output_type="VEL",
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown preset '{preset}'")

    if verbose:
        for tr in st:
            sr = getattr(tr.stats, "sampling_rate", 0.0) or 0.0
            print(f"[OK] {tr.id} | {tr.stats.starttime} – {tr.stats.endtime} | {sr:.2f} Hz")
    return st


def write_event_miniseed(
    st: Stream,
    out_dir: str,
    *,
    event_id: Optional[str] = None,
    folder_by_station: bool = True,
    filename_template: str = "{net}.{sta}.{loc}.{cha}.{t0}_{t1}.mseed",
    encoding: str = "STEIM2",
    reclen: int = 4096,
    flush_empty: bool = False,
    verbose: bool = False,
) -> None:
    """
    Write a normalized Stream to per-trace MiniSEED files.

    Parameters
    ----------
    st : obspy.Stream
        Stream to write (ideally already normalized via `preprocess_stream()`).
    out_dir : str
        Root output directory.
    event_id : str, optional
        If provided, can be used by your `write_mseed()` to decorate paths or
        filenames (e.g., include event ID in filenames).
    folder_by_station : bool
        If True, files are placed under `{out_dir}/{NET.STA}/...` for a clean
        layout by station.
    filename_template : str
        Template for filenames. Available keys:
          {net, sta, loc, cha, t0, t1} (t0/t1 are UTC ISO strings without spaces).
        Example: "{net}.{sta}.{loc}.{cha}.{t0}_{t1}.mseed"
    encoding : str
        MiniSEED encoding, e.g., "STEIM2", "FLOAT32", "INT32".
    reclen : int
        MiniSEED record length (bytes).
    flush_empty : bool
        If True, write even traces that are all zeros. Usually False.
    verbose : bool
        If True, prints what is being written.

    Returns
    -------
    None

    Notes
    -----
    - This function delegates to your existing `flovopy.core.miniseed_io.write_mseed`.
    - If you need per-trace gating (e.g., skip extremely short or low-SNR traces),
      do that **before** calling this function.
    """
    if not len(st):
        if verbose:
            print("[write_event_miniseed] Empty Stream, nothing to write.")
        return

    write_mseed(
        st,
        out_dir=out_dir,
        folder_by_station=folder_by_station,
        filename_template=filename_template,
        encoding=encoding,
        reclen=reclen,
        event_id=event_id,
        flush_empty=flush_empty,
        verbose=verbose,
    )


def sds_to_event_miniseed(
    sds_root: str,
    t1: UTCDateTime,
    t2: UTCDateTime,
    out_dir: str,
    *,
    preset: Preset = "archive_preset",
    net: str = "*",
    sta: str = "*",
    loc: str = "*",
    cha: str = "*",
    speed: int = 2,
    inv=None,
    event_id: Optional[str] = None,
    filename_template: str = "{net}.{sta}.{loc}.{cha}.{t0}_{t1}.mseed",
    encoding: str = "STEIM2",
    reclen: int = 4096,
    folder_by_station: bool = True,
    flush_empty: bool = False,
    verbose: bool = False,
) -> Stream:
    """
    One-shot convenience: read from SDS, normalize according to a preset,
    and write MiniSEED files. Returns the normalized Stream.

    Parameters
    ----------
    sds_root : str
        Root directory of the SDS archive.
    t1, t2 : UTCDateTime
        Start/end of the target time window.
    out_dir : str
        Where to write MiniSEED output.
    preset : {"archive_preset", "analysis_preset"}
        Processing flavor (see module docstring for intent and tradeoffs).
    net, sta, loc, cha : str
        SDS selectors (wildcards allowed).
    speed : int
        Passed to SDS reader; higher often means faster (implementation-specific).
    inv : obspy.Inventory or None
        StationXML for response removal (used only in analysis preset).
    event_id : str, optional
        Optional label to associate with the written products.
    filename_template : str
        MiniSEED filename pattern (see `write_event_miniseed`).
    encoding : str
        MiniSEED encoding ("STEIM2", "FLOAT32", "INT32", ...).
    reclen : int
        MiniSEED record length in bytes.
    folder_by_station : bool
        Group output by station under `{out_dir}/{NET.STA}`.
    flush_empty : bool
        If True, write even all-zero traces. Default False.
    verbose : bool
        If True, prints info at each stage.

    Returns
    -------
    obspy.Stream
        The normalized Stream (possibly empty if no traces found or all failed).

    Examples
    --------
    >>> from obspy import UTCDateTime
    >>> t1 = UTCDateTime("2020-01-01T12:00:00")
    >>> t2 = UTCDateTime("2020-01-01T12:03:00")
    >>> st = sds_to_event_miniseed(
    ...     sds_root="/data/SDS",
    ...     t1=t1, t2=t2,
    ...     out_dir="/tmp/launch_evt",
    ...     preset="analysis_preset",
    ...     net="1R", sta="BCHH", loc="10", cha="D*",
    ...     verbose=True,
    ... )
    """
    st = load_event_stream_from_sds(
        sds_root,
        t1,
        t2,
        net=net,
        sta=sta,
        loc=loc,
        cha=cha,
        speed=speed,
        preset=preset,
        inv=inv,
        verbose=verbose,
    )
    if len(st):
        write_event_miniseed(
            st,
            out_dir=out_dir,
            event_id=event_id,
            folder_by_station=folder_by_station,
            filename_template=filename_template,
            encoding=encoding,
            reclen=reclen,
            flush_empty=flush_empty,
            verbose=verbose,
        )
    return st