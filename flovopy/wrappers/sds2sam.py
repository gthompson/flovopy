#!/usr/bin/env python
"""
sds2sam.py
----------
Read an SDS archive and compute SAM time series.

- If --remove_response      → VSAM (velocity-/pressure-based SAM; units m/s, Pa)
- If NOT --remove_response  → RSAM  (counts)

Writes one SAM/RSAM file per {net.sta.loc.chan, year} under SAM_ROOT.
"""

from __future__ import annotations
import os
import sys
import json
from typing import Iterable, Optional, List, Dict
import math

from obspy import UTCDateTime, Stream, read_inventory
from tqdm import tqdm
import glob
import shutil
import pandas as pd
# flovopy imports
from flovopy.sds.sds import SDSobj
from flovopy.core.miniseed_io import smart_merge, unmask_gaps

# SAM classes (try processing/, then core/ for backwards compat)
try:
    from flovopy.processing.sam import RSAM, VSAM
except Exception:
    from flovopy.core.sam import RSAM, VSAM  # type: ignore

# -------- Storm/Hurricane presets (see notes above) --------
BAND_PRESETS: Dict[str, Dict[str, List[float]]] = {
    "storm_seismo": {"PRI": [0.05, 0.10], "SEC": [0.10, 0.35], "HI": [1.0, 5.0]},
    "storm_infra":  {"TC":  [0.01, 0.10], "MB":  [0.15, 0.35], "TH": [1.0, 10.0]},
    "storm":   {"PRI": [0.02, 0.10], "SEC": [0.10, 0.35], "HI": [1.0, 10.0]},
    "volcano":    {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 'VT':[4.0, 18.0]},
}

# -------------------------- CLI --------------------------

def parse_args(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser(
        description="Compute SAM (RSAM/VSAM) from an SDS archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sds_root", required=True,
                   help="Path to SDS archive root (SeisComP Data Structure).")
    p.add_argument("--sam_root", required=True,
                   help="Output root for SAM archive (folder that will contain RSAM/ or VSAM/).")

    # Time window (UTC)
    p.add_argument("--start", required=True,
                   help="Start time (UTC), e.g. 2011-03-10 or 2011-03-10T00:00:00")
    p.add_argument("--end", required=True,
                   help="End time (UTC, NON-inclusive of last midnight unless time given)")

    # Selection (applied after read). These are ObsPy-style wildcards.
    p.add_argument("--network", default="*",
                   help="Network code filter (e.g. IU, MV, or '*').")
    p.add_argument("--station", default="*",
                   help="Station code filter (e.g. DWPF, MB*).")
    p.add_argument("--location", default="*",
                   help="Location code filter (e.g. 00, 10, --, or '*').")
    p.add_argument("--channel", default="*",
                   help="Channel code filter (e.g. BHZ, BH?, *Z).")

    # SAM options
    p.add_argument("--sampling_interval", type=float, default=60.0,
                   help="SAM sample interval (seconds).")

    # Legacy single-band options (kept for backward compatibility)
    p.add_argument("--minfreq", type=float, default=0.5,
                   help="Bandpass low-cut for the primary RSAM/VSAM filter (Hz).")
    p.add_argument("--maxfreq", type=float, default=18.0,
                   help="Bandpass high-cut for the primary RSAM/VSAM filter (Hz).")
    p.add_argument("--corners", type=int, default=4,
                   help="Filter corners for bandpass.")
    p.add_argument("--fill_value", type=float, default=0.0,
                   help="Value to fill masked gaps before processing.")

    # New: bands dictionary
    p.add_argument("--bands", default=None,
                   help=("Band dictionary as inline JSON or a path to .json/.yml/.yaml. "
                         "Example: '{\"PRI\":[0.05,0.10],\"SEC\":[0.10,0.35],\"HI\":[1,5]}'"))
    p.add_argument("--bands-preset", choices=list(BAND_PRESETS.keys()), default=None,
                   help="Use a predefined storm/hurricane band set. Ignored if --bands is provided.")

    # Response removal
    p.add_argument("--remove_response", action="store_true",
                   help="If set, remove instrument response and compute VSAM instead of RSAM.")
    p.add_argument("--stationxml", default=None,
                   help="Path to a StationXML file OR a directory containing StationXML files.")
    p.add_argument("--output", choices=["VEL", "DISP", "ACC", "PA"], default="VEL",
                   help="When removing response, target physical output ('VEL','DISP','ACC','PA').")

    # SDS reading / perf
    p.add_argument("--speed", type=int, default=2, choices=[1, 2],
                   help="SDS read speed (2 uses SDS client merging/downsample; 1 reads files individually).")
    p.add_argument("--max_rate", type=float, default=250.0,
                   help="Optional upper bound for sample rate during SDS read (downsampling).")
    p.add_argument("--merge_strategy", default="obspy",
                   help="Strategy passed to smart_merge (e.g., 'obspy').")
    p.add_argument("--min_rate", type=float, default=None,
                help=("Minimum sampling rate to keep (Hz). "
                        "If omitted, computed as 2.2×highest requested band edge."))
    p.add_argument("--also_rsam", action="store_true",
               help="When --remove_response is set, also compute RSAM (from counts) "
                    "in addition to VSAM.")
    p.add_argument(
        "--nprocs",
        type=int,
        default=pick_nprocs(),  # auto half cores (rounded up)
        help="Number of parallel worker processes. Default = half of available CPUs."
    )
    p.add_argument("--staging_root", type=str, default=None,
                help="Directory for per-day shard files. Defaults to <sam_root>/.staging")
    p.add_argument("--shard_format", choices=["parquet", "pickle", "csv"], default="parquet",
                help="File format for per-day shards before consolidation.")
    p.add_argument("--keep_staging", action="store_true",
                help="Keep shard files after consolidation (default: removed).")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging.")
    return p.parse_args(argv)

# --------------------- helpers & core ---------------------

def _required_min_sr(bands: Optional[Dict[str, List[float]]],
                     bp: Optional[tuple[float, float]]) -> float:
    """
    Return the minimum sampling rate (Hz) required to support the requested bands,
    using a 2.2× guard over Nyquist.
    """
    fmax = None
    if bands:
        try:
            fmax = max(float(v[1]) for v in bands.values())
        except Exception:
            pass
    if (fmax is None) and bp:
        fmax = float(bp[1])
    if fmax is None or fmax <= 0.0:
        # conservative fallback
        return 1.0
    return 2.2 * fmax

def _expand(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))

def _iter_days(start: UTCDateTime, end: UTCDateTime) -> Iterable[UTCDateTime]:
    """Yield day-aligned boundaries from start to end (step = 1 day)."""
    day0 = UTCDateTime(start.year, start.month, start.day)
    t = day0
    while t < end:
        yield t
        t += 86400

def _load_inventory(stationxml: str, verbose: bool = False):
    """Load StationXML (single file or directory)."""
    if stationxml is None:
        return None
    stationxml = _expand(stationxml)
    if not os.path.exists(stationxml):
        print(f"WARNING: StationXML path does not exist: {stationxml}")
        return None
    try:
        if os.path.isdir(stationxml):
            from obspy.core.inventory.inventory import Inventory
            inv = None
            import glob
            for path in glob.glob(os.path.join(stationxml, "*.xml")):
                this = read_inventory(path)
                inv = this if inv is None else inv + this
            if verbose:
                print(f"Loaded StationXML from directory: {stationxml}")
            return inv
        else:
            inv = read_inventory(stationxml)
            if verbose:
                print(f"Loaded StationXML file: {stationxml}")
            return inv
    except Exception as e:
        print(f"WARNING: Could not read StationXML: {e}")
        return None

def _remove_response_inplace(st: Stream, inv, output: str = "VEL", verbose: bool = False):
    """Remove response on each trace in-place."""
    if inv is None:
        raise ValueError("remove_response requested but no StationXML inventory is available.")
    pre_filt = (0.005, 0.01, 45.0, 50.0)
    for tr in st:
        try:
            tr.remove_response(inventory=inv, output=output, pre_filt=pre_filt, taper=True, zero_mean=True)
            if output == "VEL":
                tr.stats.units = "m/s"
            elif output == "DISP":
                tr.stats.units = "m"
            elif output == "ACC":
                tr.stats.units = "m/s^2"
            elif output == "PA":
                tr.stats.units = "Pa"
        except Exception as e:
            if verbose:
                print(f"⚠️ remove_response failed for {tr.id}: {e}")
            st.remove(tr)

def _parse_bands_arg(bands_arg: Optional[str],
                     bands_preset: Optional[str],
                     verbose: bool = False) -> Optional[Dict[str, List[float]]]:
    """Parse --bands (JSON string or file path) or --bands-preset."""
    if bands_arg:
        txt = bands_arg.strip()
        # Inline JSON?
        if txt.startswith("{"):
            try:
                bands = json.loads(txt)
                if verbose:
                    print(f"Using bands (inline JSON): {bands}")
                return bands
            except Exception as e:
                raise SystemExit(f"--bands JSON parse error: {e}")
        # File path?
        path = _expand(txt)
        if not os.path.exists(path):
            raise SystemExit(f"--bands path not found: {path}")
        try:
            if path.endswith((".yml", ".yaml")):
                import yaml  # optional dependency
                with open(path, "r") as f:
                    bands = yaml.safe_load(f)
            else:
                with open(path, "r") as f:
                    bands = json.load(f)
            if verbose:
                print(f"Using bands (file): {bands}")
            return bands
        except Exception as e:
            raise SystemExit(f"--bands file parse error: {e}")
    # Preset?
    if bands_preset:
        bands = BAND_PRESETS[bands_preset]
        if verbose:
            print(f"Using bands preset '{bands_preset}': {bands}")
        return bands
    return None

def _staging_dir(base: str, subkind: str, trace_id: str, day_str: str) -> str:
    """
    Returns directory path for a shard:
      <base>/<subkind>_shards/<NET>/<ID>/<YYYY-MM-DD>/
    where subkind ∈ {"RSAM","VSAM"}
    """
    net = trace_id.split('.')[0] if '.' in trace_id else 'UNK'
    return os.path.join(base, f"{subkind}_shards", net, trace_id, day_str)

def _shard_path(base: str, subkind: str, trace_id: str, day_str: str,
                sampling_interval: int, shard_format: str) -> str:
    """
    Returns full shard filename, e.g.:
      .../VSAM_shards/IU/IU.DWPF.10.BHZ/2011-03-10/2011-03-10_60s.parquet
    """
    ddir = _staging_dir(base, subkind, trace_id, day_str)
    os.makedirs(ddir, exist_ok=True)
    ext = {"parquet": "parquet", "pickle": "pkl", "csv": "csv"}[shard_format]
    return os.path.join(ddir, f"{day_str}_{int(sampling_interval)}s.{ext}")

def _write_df(df: pd.DataFrame, path: str, shard_format: str):
    if shard_format == "parquet":
        df.to_parquet(path, index=False)
    elif shard_format == "pickle":
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)

def _read_df(path: str, shard_format: str) -> pd.DataFrame:
    if shard_format == "parquet":
        return pd.read_parquet(path)
    elif shard_format == "pickle":
        return pd.read_pickle(path)
    else:
        return pd.read_csv(path)

def _collect_ids_from_staging(staging_root: str, subkind: str) -> List[str]:
    """
    Find all trace IDs that have shards under <staging_root>/<subkind>_shards/*/<ID>/
    """
    pattern = os.path.join(staging_root, f"{subkind}_shards", "*", "*")
    # The leaf component is the trace_id directory name
    ids = sorted({ os.path.basename(p) for p in glob.glob(pattern) if os.path.isdir(p) })
    return ids

def _consolidate_one_kind(subkind: str,
                          classref,
                          staging_root: str,
                          sam_root: str,
                          start: UTCDateTime,
                          end: UTCDateTime,
                          sampling_interval: int,
                          shard_format: str,
                          verbose: bool):
    """
    Merge shards → yearly per-id files via classref(dataframes=...).write(sam_root,...).
    subkind: 'RSAM' or 'VSAM'
    classref: RSAM or VSAM
    """
    trace_ids = _collect_ids_from_staging(staging_root, subkind)
    if verbose:
        print(f"[consolidate] {subkind}: found {len(trace_ids)} trace IDs with shards")

    for trace_id in trace_ids:
        # gather shards between start and end
        # folder structure: .../<subkind>_shards/<NET>/<ID>/<YYYY-MM-DD>/*.ext
        net = trace_id.split('.')[0] if '.' in trace_id else '*'
        day_dirs_glob = os.path.join(staging_root, f"{subkind}_shards", net, trace_id, "*")
        day_dirs = sorted([p for p in glob.glob(day_dirs_glob) if os.path.isdir(p)])

        per_year_frames: Dict[int, List[pd.DataFrame]] = {}
        for ddir in day_dirs:
            day_str = os.path.basename(ddir)  # YYYY-MM-DD
            # filter by overall start/end (string compare on date is fine here)
            if not (str(start)[:10] <= day_str <= str(end)[:10]):
                continue
            for shard in sorted(glob.glob(os.path.join(ddir, f"*_{int(sampling_interval)}s.*"))):
                try:
                    df = _read_df(shard, shard_format)
                    if df.empty:
                        continue
                    # clip to start/end robustly
                    df["pddatetime"] = pd.to_datetime(df["time"], unit="s", utc=False)
                    start_ts = pd.to_datetime(start.datetime)
                    end_ts   = pd.to_datetime(end.datetime)
                    m = df["pddatetime"].between(start_ts, end_ts)
                    df = df.loc[m].drop(columns=["pddatetime"])
                    if df.empty:
                        continue
                    yr = pd.to_datetime(df["time"], unit="s").dt.year.mode()
                    if yr.empty:
                        # fallback to day string
                        yr = pd.Series([int(day_str[:4])])
                    year = int(yr.iloc[0])
                    per_year_frames.setdefault(year, []).append(df)
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ failed reading shard {shard}: {e}")

        # write consolidated per-year files using classref(dataframes=...).write(...)
        for year, frames in sorted(per_year_frames.items()):
            merged = pd.concat(frames, axis=0, ignore_index=True) if len(frames) > 1 else frames[0]
            # sort & de-duplicate by time
            merged = merged.sort_values("time")
            merged = merged.drop_duplicates(subset="time", keep="last").reset_index(drop=True)
            # build SAM object and write final yearly artifact
            sam_obj = classref(dataframes={trace_id: merged})
            sam_obj.write(os.path.abspath(sam_root), ext="pickle", overwrite=False, verbose=verbose)
            if verbose:
                print(f"[consolidate] wrote {subkind} {trace_id} {year} ({len(merged)} rows)")



def pick_nprocs(default=None):
    cores = os.cpu_count() or 1
    if default is not None:
        return default
    return math.ceil(cores / 2)

def process_one_day(
    day_start,
    day_end,
    sds_root: str,
    select: dict,
    remove_resp: bool,
    inv_path: Optional[str],
    output_mode: str,
    sampling_interval: float,
    bp: tuple,
    corners: int,
    fill_value: float,
    speed: int,
    max_rate: float,
    merge_strategy: str,
    sam_root: str,
    verbose: bool,
    bands: Optional[Dict[str, List[float]]] = None,
    min_sampling_rate: Optional[float] = None,
    also_rsam: bool = False,
    staging_root: Optional[str] = None,        # <-- NEW (accepted)
    shard_format: str = "pickle",              # <-- NEW (accepted)
):
    """
    Read one day's data from an SDS archive, optionally remove instrument response,
    compute SAM metrics, and write results to the SAM archive.

    Parameters
    ----------
    day_start : UTCDateTime
        Start time of the day to process.
    day_end : UTCDateTime
        End time of the day to process.
    sds_root : str
        Path to the SDS archive root directory.
    select : dict
        ObsPy-style selection dictionary with keys: 'network', 'station',
        'location', 'channel'. Wildcards allowed.
    remove_resp : bool
        If True, remove instrument response using StationXML metadata.
    inv_path : str
        Path to a StationXML file or directory containing StationXML files.
    output_mode : str
        Output units after response removal: 'VEL', 'DISP', 'ACC', or 'DEF'.
    sampling_interval : float
        Output sampling interval in seconds for the SAM time series.
    bp : tuple of float
        Bandpass frequency limits as (minfreq, maxfreq) in Hz.
    corners : int
        Number of filter corners for bandpass filtering.
    fill_value : float
        Value used to fill missing data segments before processing.
    speed : int
        SDS reading mode. 1 = file-by-file; 2 = ObsPy SDS client get_waveforms.
    max_rate : float
        Maximum sampling rate for downsampling before processing.
    merge_strategy : str
        Strategy for merging traces in `smart_merge()`.
    sam_root : str
        Output root directory for writing SAM results.
    verbose : bool
        If True, print detailed processing information.
    bands : dict, optional
        Dictionary of named frequency bands for multi-band SAM computation.
    min_sampling_rate : float, optional
        Minimum acceptable sampling rate for a trace. Lower rates are skipped.

    Returns
    -------
    None
        Writes output to `sam_root` and logs progress if `verbose=True`.
    """
    """Read one day from SDS, optionally remove response, compute SAM, write to archive."""
    sds = SDSobj(_expand(sds_root))
    rc = sds.read(
        startt=day_start,
        endt=day_end,
        skip_low_rate_channels=False,
        speed=speed,
        verbose=verbose,
        min_sampling_rate=min_sampling_rate,
        max_sampling_rate=max_rate,
        merge_strategy=merge_strategy,
    )
    st = sds.stream or Stream()
    if verbose:
        print(f"Read SDS: {st.__str__()}")
    if not len(st):
        return

    # selection
    st = st.select(network=select["network"], station=select["station"],
                   location=select["location"], channel=select["channel"])
    if not len(st):
        if verbose:
            print("No traces match selection for this day.")
        return

    # Keep a copy of COUNTS stream before response removal (for RSAM)
    st_counts = st.copy()

    # Response removal (for VSAM)
    if remove_resp:
        inv = _load_inventory(inv_path, verbose=verbose)
        _remove_response_inplace(st, inv, output=output_mode, verbose=verbose)

    if not len(st):
        if verbose:
            print("No usable traces after response removal/sanitization.")
        return


    fmin, fmax = bp
    day_str = str(day_start)[:10]

    # Default staging folder if not provided
    if not staging_root:
        staging_root = os.path.join(os.path.abspath(sam_root), "_staging")

    if remove_resp:
        vsam = VSAM(
            stream=st,
            sampling_interval=sampling_interval,
            bands=bands if bands else None,
            filter=None if bands else [fmin, fmax],
            corners=corners,
            verbose=verbose,
        )
        # write per-id shard for VSAM
        for trace_id, df in vsam.dataframes.items():
            shard = _shard_path(staging_root, "VSAM", trace_id, day_str, sampling_interval, shard_format)
            _write_df(df, shard, shard_format)

        if also_rsam:
            rsam = RSAM(
                stream=st_counts,
                sampling_interval=sampling_interval,
                bands=bands if bands else None,
                filter=None if bands else [fmin, fmax],
                corners=corners,
                verbose=verbose,
            )
            for trace_id, df in rsam.dataframes.items():
                shard = _shard_path(staging_root, "RSAM", trace_id, day_str, sampling_interval, shard_format)
                _write_df(df, shard, shard_format)

    else:
        rsam = RSAM(
            stream=st,
            sampling_interval=sampling_interval,
            bands=bands if bands else None,
            filter=None if bands else [fmin, fmax],
            corners=corners,
            verbose=verbose,
        )
        for trace_id, df in rsam.dataframes.items():
            shard = _shard_path(staging_root, "RSAM", trace_id, day_str, sampling_interval, shard_format)
            _write_df(df, shard, shard_format)
# ---------------------------- main ----------------------------

def main(argv: Optional[List[str]] = None):
    """
    Command-line entry point for SDS → SAM conversion.

    This function parses command-line arguments, determines processing
    parameters, and iterates over the requested date range one day at a time,
    calling `process_one_day()` for each day. It supports both RSAM and VSAM
    generation, optional instrument response removal, and frequency-band-
    based processing.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, uses `sys.argv[1:]`.

    Workflow
    --------
    1. Parse and normalize command-line arguments (paths, times, selections).
    2. Expand bands from arguments or presets and determine the minimum
       required sampling rate.
    3. Loop over all days in the start–end range:
       - Clip each day's processing window to the user-specified range.
       - Call `process_one_day()` with SDS location, selection filters,
         instrument response settings, and SAM configuration.
    4. Print progress and diagnostics if `--verbose` is enabled.

    Notes
    -----
    - `--bands` or `--bands-preset` will override `--minfreq`/`--maxfreq`
      for RSAM/VSAM computation.
    - The minimum sampling rate is auto-computed from band limits unless
      overridden with `--min-rate`.
    - `process_one_day()` handles all SDS reading, sanitization, merging,
      filtering, SAM computation, and writing.
    - Output is organized under `--sam_root` by subclass (RSAM/ or VSAM/).

    Examples
    --------
    Basic usage:
    >>> main([
    ...     "--sds_root", "/data/SDS",
    ...     "--sam_root", "/data/SAM_OUT",
    ...     "--start", "2020-01-01",
    ...     "--end", "2020-01-03",
    ...     "--network", "IU",
    ...     "--station", "ANMO",
    ...     "--location", "*",
    ...     "--channel", "*Z",
    ...     "--minfreq", "0.01",
    ...     "--maxfreq", "10.0",
    ...     "--sampling_interval", "60"
    ... ])
    """
    args = parse_args(argv)

    args.sds_root = _expand(args.sds_root)
    args.sam_root = _expand(args.sam_root)
    if args.stationxml:
        args.stationxml = _expand(args.stationxml)

    start = UTCDateTime(args.start)
    end = UTCDateTime(args.end)
    if end <= start:
        raise SystemExit("End must be after start.")

    bands = _parse_bands_arg(args.bands, args.bands_preset, verbose=args.verbose)

    # Compute min_rate unless user overrode it
    auto_min_rate = _required_min_sr(bands, (args.minfreq, args.maxfreq))
    min_rate_to_use = args.min_rate if getattr(args, "min_rate", None) is not None else auto_min_rate

    if args.verbose:
        print(f"SDS → SAM\n  SDS: {args.sds_root}\n  SAM: {args.sam_root}")
        print(f"  Time: {start} → {end}")
        print(f"  Select: {args.network}.{args.station}.{args.location}.{args.channel}")
        print(f"  Remove response: {args.remove_response} (output={args.output})")
        if bands:
            print(f"  Bands: {bands}  (overrides --minfreq/--maxfreq)")
        else:
            print(f"  RSAM/VSAM band: {args.minfreq}-{args.maxfreq} Hz; Δ={args.sampling_interval}s")
        print(f"  Min sampling rate: {min_rate_to_use:.3f} Hz ("
              f"{'override' if getattr(args, 'min_rate', None) is not None else 'auto from bands'})\n")

    # Resolve staging root
    staging_root = os.path.abspath(args.staging_root or os.path.join(args.sam_root, ".staging"))
    shard_format = args.shard_format.lower() or "parquet"

    # Build per-day list
    day_starts = list(_iter_days(start, end))
    # Prepare common kwargs for worker
    common_kwargs = dict(
        sds_root=args.sds_root,
        select=dict(network=args.network, station=args.station,
                    location=args.location, channel=args.channel),
        remove_resp=args.remove_response,
        inv_path=args.stationxml,
        output_mode=args.output,
        sampling_interval=args.sampling_interval,
        bp=(args.minfreq, args.maxfreq),
        corners=args.corners,
        fill_value=args.fill_value,
        speed=args.speed,
        max_rate=args.max_rate,
        merge_strategy=args.merge_strategy,
        sam_root=args.sam_root,          # not used in worker write; kept for signature compatibility
        verbose=args.verbose,
        bands=bands,
        min_sampling_rate=min_rate_to_use,
        also_rsam=getattr(args, "also_rsam", False),
        staging_root=staging_root,
        shard_format=shard_format,
    )

    # Parallelize per-day workers
    if args.nprocs and args.nprocs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.nprocs) as ex:
            futures = []
            for d0 in day_starts:
                d1 = min(d0 + 86400, end)
                futures.append(ex.submit(process_one_day, day_start=d0, day_end=d1, **common_kwargs))
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Days"):
                pass
    else:
        for d0 in tqdm(day_starts, desc="Days"):
            d1 = min(d0 + 86400, end)
            process_one_day(day_start=d0, day_end=d1, **common_kwargs)

    # ------------- Consolidation phase: shards → yearly files -------------
    try:
        if args.remove_response:
            _consolidate_one_kind("VSAM", VSAM, staging_root, args.sam_root, start, end,
                                  int(args.sampling_interval), args.shard_format, args.verbose)
            if getattr(args, "also_rsam", False):
                _consolidate_one_kind("RSAM", RSAM, staging_root, args.sam_root, start, end,
                                      int(args.sampling_interval), args.shard_format, args.verbose)
        else:
            _consolidate_one_kind("RSAM", RSAM, staging_root, args.sam_root, start, end,
                                  int(args.sampling_interval), args.shard_format, args.verbose)

        # Optionally clean up staging
        if not args.keep_staging:
            shutil.rmtree(staging_root, ignore_errors=True)
    except Exception as e:
        print(f"⚠️ Consolidation failed: {e}")

    # -------------------- Quick-look plots --------------------
    # Read back what we just wrote and save PNGs under <SAM_ROOT>/quicklooks.
    # -------------------- Quick-look plots --------------------
    try:
        os.makedirs(os.path.join(args.sam_root, "quicklooks"), exist_ok=True)
        ql_dir = os.path.join(args.sam_root, "quicklooks")

        date_a = str(start)[:10]
        date_b = str(end)[:10]
        sel_lbl = f"{args.network}.{args.station}.{args.location}.{args.channel}".replace("*", "ALL")

        # VSAM quicklook if we removed response
        if args.remove_response:
            try:
                vsam = VSAM.read(
                    startt=start, endt=end, SAM_DIR=args.sam_root,
                    network=args.network, sampling_interval=int(args.sampling_interval),
                    ext="pickle", verbose=args.verbose
                )
                out_vsam = os.path.join(
                    ql_dir, f"VSAM_{sel_lbl}_{date_a}_to_{date_b}_{int(args.sampling_interval)}s.png"
                )
                vsam.plot(metrics=['bands'], kind='line', logy=True, outfile=out_vsam)
                if os.path.exists(out_vsam) and args.verbose:
                    print(f"Quicklook written: {out_vsam}")
            except Exception as e:
                print(f"⚠️ VSAM quicklook failed: {e}")

        # RSAM quicklook if we produced RSAM (either RSAM-only or alongside VSAM)
        if (getattr(args, "also_rsam", False)) or (not args.remove_response):
            try:
                rsam = RSAM.read(
                    startt=start, endt=end, SAM_DIR=args.sam_root,
                    network=args.network, sampling_interval=int(args.sampling_interval),
                    ext="pickle", verbose=args.verbose
                )
                out_rsam = os.path.join(
                    ql_dir, f"RSAM_{sel_lbl}_{date_a}_to_{date_b}_{int(args.sampling_interval)}s.png"
                )
                rsam.plot(metrics=['mean'], kind='line', logy=True, outfile=out_rsam)
                if os.path.exists(out_rsam) and args.verbose:
                    print(f"Quicklook written: {out_rsam}")
            except Exception as e:
                print(f"⚠️ RSAM quicklook failed: {e}")

    except Exception as e:
        print(f"⚠️ Quicklook plotting skipped due to error: {e}")

    if args.verbose:
        print("Done.")

if __name__ == "__main__":
    main(sys.argv[1:])