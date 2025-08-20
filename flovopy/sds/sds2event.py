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
import os
import numpy as np
import pandas as pd
from typing import Tuple, Literal, Optional, List, Dict, Any
from obspy import UTCDateTime, Stream, Trace
from flovopy.sds.sds import SDSobj

# Core, source-agnostic utilities
from flovopy.core.preprocess import preprocess_stream
from flovopy.core.miniseed_io import write_mseed  # your existing writer

Preset = Literal["raw_preset", "archive_preset", "analysis_preset"]


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
    preset : {"raw_present", "archive_preset", "analysis_preset"}
        - "raw_preset": no preprocessing at all, best for reading from SDS and writing to event miniseed
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

    if preset == "raw_preset": # No preprocessing applied
        pass
    elif preset == "archive_preset":
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


def _filesafe_iso(ts: UTCDateTime) -> str:
    return str(UTCDateTime(ts)).replace(":", "-")

def write_event_miniseed(
    st: Union[Stream, Trace],
    out_dir: str,
    *,
    event_id: Optional[str] = None,
    filename_template: str = "{event_id}.{t0}_{t1}.mseed",
    # NEW: allow caller to pass the fully resolved path; if given, we won't rebuild it
    filename_path: Optional[str] = None,
    encoding: Optional[str] = None,   # None → let ObsPy choose
    reclen: Optional[int] = None,     # None → ObsPy default
    overwrite_ok: bool = True,
    flush_empty: bool = False,
    pickle_fallback: bool = False,
    verbose: bool = False,
) -> None:
    """Write ONE MiniSEED file per event window (multi-trace OK)."""
    if isinstance(st, Trace):
        st = Stream([st])
    elif not isinstance(st, Stream) or len(st) == 0:
        if verbose:
            print("[write_event_miniseed] Empty or invalid input; nothing to write.")
        return

    # Build destination path
    if filename_path is None:
        # If no explicit path, derive from the stream union (previous behavior).
        t0 = min(tr.stats.starttime for tr in st)
        t1 = max(tr.stats.endtime   for tr in st)
        t0s, t1s = _filesafe_iso(t0), _filesafe_iso(t1)
        eid = event_id or f"event_{t0s}"
        os.makedirs(out_dir, exist_ok=True)
        fname = filename_template.format(event_id=eid, t0=t0s, t1=t1s)
        mseedfile = os.path.join(out_dir, fname)
    else:
        mseedfile = filename_path
        os.makedirs(os.path.dirname(mseedfile) or out_dir, exist_ok=True)

    if verbose:
        print(f"[write_event_miniseed] Writing event → {mseedfile}")

    _ = write_mseed(
        st,
        mseedfile=mseedfile,
        fill_value=0.0,
        overwrite_ok=overwrite_ok,
        pickle_fallback=pickle_fallback,
        encoding=encoding,   # None → dtype-driven by ObsPy
        reclen=reclen,       # None → ObsPy default
    )


def _build_event_filepath(
    out_dir: str,
    event_id: Optional[str],
    t_start: UTCDateTime,
    t_end: UTCDateTime,
    filename_template: str,
) -> str:
    """Create the deterministic event file path from the *requested* window."""
    t0s = _filesafe_iso(t_start)
    t1s = _filesafe_iso(t_end)
    eid = event_id or f"event_{t0s}"
    fname = filename_template.format(event_id=eid, t0=t0s, t1=t1s)
    return os.path.join(out_dir, fname)

def sds_to_event_miniseed(
    sds_root: str,
    t1: UTCDateTime,
    t2: UTCDateTime,
    out_dir: str,
    *,
    preset: Preset = "raw_preset",
    net: str = "*",
    sta: str = "*",
    loc: str = "*",
    cha: str = "*",
    speed: int = 2,
    inv=None,
    event_id: Optional[str] = None,
    filename_template: str = "{event_id}.{t0}_{t1}.mseed",
    encoding: Optional[str] = None,   # <- None: dtype-driven
    reclen: Optional[int] = None,     # <- None: ObsPy default
    flush_empty: bool = False,
    verbose: bool = False,
    # NEW: if True, skip SDS read+write when the target file already exists
    skip_if_exists: bool = True,
) -> Stream:
    """Read from SDS, normalize, and write one MiniSEED file for the event."""
    #_preset = "archive_preset" if preset == "raw_preset" else preset

    # Build deterministic target path from the *requested* window (not stream union)
    target_path = _build_event_filepath(out_dir, event_id, t1, t2, filename_template)

    if skip_if_exists and os.path.exists(target_path):
        if verbose:
            print(f"[sds_to_event_miniseed] Skip: file exists → {target_path}")
        return Stream()  # empty Stream signals “nothing read/written”

    # Otherwise, read and write
    st = load_event_stream_from_sds(
        sds_root,
        t1,
        t2,
        net=net, sta=sta, loc=loc, cha=cha,
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
            filename_template=filename_template,
            filename_path=target_path,   # ensure name matches the existence check
            encoding=encoding,
            reclen=reclen,
            overwrite_ok=True,
            flush_empty=flush_empty,
            pickle_fallback=False,
            verbose=verbose,
        )
    return st

def _to_utc_any(val) -> UTCDateTime:
    """Epoch, ISO (with/without TZ), or '...Z' -> UTCDateTime."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        raise ValueError("Missing time value")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return UTCDateTime(float(val))
    s = str(val).strip()
    if "T" not in s and " " in s:
        s = s.replace(" ", "T", 1)
    ts = pd.to_datetime(s.replace("Z", "+00:00"), utc=True, errors="coerce")
    if not pd.isna(ts):
        return UTCDateTime(ts.to_pydatetime())
    return UTCDateTime(s)

def _safe_event_id(row: pd.Series, idx: int, event_id: str) -> str:
    """Return a sanitized event_id containing only alphanumeric characters, hyphens, or underscores."""
    return "".join(c for c in event_id if c.isalnum() or c in ("-", "_"))

def csv_to_event_miniseed(
    *,
    sds_root: str,
    csv_path: str,
    start_col: str,
    end_col: Optional[str] = None,
    out_dir: str,
    pad_before: float = 60.0,
    pad_after: float = 600.0,
    net: str = "*",
    sta: str = "*",
    loc: str = "*",
    cha: str = "*",
    preset: "Preset" = "raw_preset",
    inv=None,
    speed: int = 2,
    year_month_dirs: bool = True,
    event_id_col: Optional[str] = None,
    filename_template: str = "{event_id}.{t0}_{t1}.mseed",
    encoding: Optional[str] = None,   # <- None: dtype-driven
    reclen: Optional[int] = None,     # <- None: ObsPy default
    flush_empty: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Loop CSV and write ONE MiniSEED per window using sds_to_event_miniseed()."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if start_col not in df.columns:
        raise ValueError(f"CSV missing required start_col '{start_col}'.")

    summary: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        try:
            t_start = _to_utc_any(row[start_col])
        except Exception as e:
            if verbose:
                print(f"[{i:06d}] ✖ bad {start_col}: {e}")
            summary.append({
                "row_index": i, "event_id": _safe_event_id(row, i, event_id_col),
                "t_start": None, "t_end": None, "t1": None, "t2": None,
                "year": None, "month": None, "n_traces": 0, "wrote": False, "out_subdir": None
            })
            continue

        t_end_val = None
        if end_col and end_col in df.columns and pd.notna(row[end_col]):
            try:
                t_end_val = _to_utc_any(row[end_col])
            except Exception as e:
                if verbose:
                    print(f"[{i:06d}] ! bad {end_col}: {e} -> using start==end")
        t_end = t_end_val or t_start

        t1 = t_start - float(pad_before)
        t2 = t_end   + float(pad_after)

        yyyy = f"{t_start.year:04d}"
        mm   = f"{t_start.month:02d}"
        out_subdir = os.path.join(out_dir, yyyy, mm) if year_month_dirs else out_dir
        os.makedirs(out_subdir, exist_ok=True)

        event_id = _safe_event_id(row, i, event_id_col)

        if verbose:
            print(f"[{i:06d}] {event_id}  {t1} → {t2}  out={out_subdir}  "
                  f"(net={net} sta={sta} loc={loc} cha={cha})")

        st = sds_to_event_miniseed(
            sds_root=sds_root,
            t1=t1, t2=t2,
            out_dir=out_subdir,
            preset=preset,
            net=net, sta=sta, loc=loc, cha=cha,
            speed=speed,
            inv=inv,
            event_id=event_id,
            filename_template=filename_template,
            encoding=encoding,   # None → dtype-driven
            reclen=reclen,       # None → ObsPy default
            flush_empty=flush_empty,
            verbose=verbose,
        )

        wrote = len(st) > 0
        if verbose and not wrote:
            print(f"[{i:06d}] – no traces found in SDS for window")

        summary.append({
            "row_index": i,
            "event_id": event_id,
            "t_start": str(t_start),
            "t_end": str(t_end),
            "t1": str(t1),
            "t2": str(t2),
            "year": int(yyyy),
            "month": int(mm),
            "n_traces": int(len(st)),
            "wrote": bool(wrote),
            "out_subdir": out_subdir,
        })

    return pd.DataFrame(summary)