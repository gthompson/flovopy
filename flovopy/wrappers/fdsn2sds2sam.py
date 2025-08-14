#!/usr/bin/env python3
"""
fdsn2sam_wrapper.py
-------------------
Orchestrates a two-phase pipeline:

  1) Download missing SDS days via ObsPy MassDownloader (fdsn2sds)
  2) Compute missing SAM days (sds2sam), in RSAM and/or VSAM as requested

"Missing" is defined against the *final* per-year SAM files in SAM_ROOT
(using SAM.missing_days()). We only operate on those UTC days.

This wrapper assumes:
- flovopy.processing.sam (or flovopy.core.sam) exposes RSAM, VSAM classes
- fdsn2sds.py and sds2sam.py each expose a `main(argv)` function

Typical usage:
    python fdsn2sam_wrapper.py \
        --service IRIS --network IU --station DWPF --location "*" --channels "BH?" \
        --start 2011-03-10 --end 2011-03-15 \
        --sds_root ~/work/SDS --sam_root ~/work/SAM_OUT \
        --remove_response --output VEL --bands-preset storm \
        --threads 4 --nprocs auto --verbose
"""

from __future__ import annotations
import os
import sys
import math
import argparse
from datetime import timedelta
from typing import List, Optional, Dict, Set

import pandas as pd
from obspy import UTCDateTime

# --- import the two tools as modules ---
from flovopy.wrappers import fdsn2sds as fdsn_tool
from flovopy.wrappers import sds2sam as sds2sam_tool

# SAM classes (processing first; fallback to core)
try:
    from flovopy.processing.sam import RSAM, VSAM
except Exception:
    from flovopy.core.sam import RSAM, VSAM  # type: ignore


# ---------------------- helpers ----------------------

def _expand(path: Optional[str]) -> Optional[str]:
    return None if path is None else os.path.abspath(os.path.expanduser(path))

def _pick_nprocs_auto() -> int:
    try:
        import os
        n = os.cpu_count() or 2
    except Exception:
        n = 2
    # “Half (rounded up)” of available logical CPUs
    return max(1, math.ceil(n / 2))

def _daterange_days(start: UTCDateTime, end: UTCDateTime) -> List[pd.Timestamp]:
    """UTC day midnights in [start, end) as pandas Timestamps (tz-naive UTC)."""
    s = pd.to_datetime(start.datetime)
    e = pd.to_datetime(end.datetime)
    return list(pd.date_range(start=s.floor("D"), end=e.floor("D"), freq="D", inclusive="left"))

def _contiguous_day_ranges(day_list: List[pd.Timestamp]):
    """Yield (start_UTCDateTime, end_UTCDateTime) for contiguous runs of days."""
    if not day_list:
        return
    day_list = sorted(day_list)
    run_start = day_list[0]
    prev = run_start
    one_day = pd.Timedelta(days=1)
    for d in day_list[1:]:
        if d - prev > one_day:
            # close previous [run_start, prev + 1 day)
            yield (UTCDateTime(run_start.to_pydatetime()), UTCDateTime((prev + one_day).to_pydatetime()))
            run_start = d
        prev = d
    # final run
    yield (UTCDateTime(run_start.to_pydatetime()), UTCDateTime((prev + one_day).to_pydatetime()))

def _union_missing_days(
    vsam_missing: Dict[str, List[UTCDateTime]] | None,
    rsam_missing: Dict[str, List[UTCDateTime]] | None,
) -> List[pd.Timestamp]:
    """Union per-ID missing-day lists → unique list of pd.Timestamp midnights."""
    all_days: Set[pd.Timestamp] = set()
    for dct in (vsam_missing or {}, rsam_missing or {}):
        for days in dct.values():
            for utcdt in days:
                all_days.add(pd.to_datetime(utcdt.datetime).floor("D"))
    return sorted(all_days)


# ---------------------- CLI ----------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Wrapper: download SDS (missing days only) and then compute SAM (missing days only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Time window
    p.add_argument("--start", required=True, help="Start UTC (YYYY-MM-DD[THH:MM:SS]).")
    p.add_argument("--end",   required=True, help="End UTC (YYYY-MM-DD[THH:MM:SS], exclusive).")

    # SDS + metadata (shared)
    p.add_argument("--sds_root", required=True, help="SDS archive root.")
    p.add_argument("--sam_root", required=True, help="SAM output root (contains RSAM/ and/or VSAM/).")
    p.add_argument("--save_inventory", action="store_true", help="Save StationXML alongside SDS.")
    p.add_argument("--metadata_dir", default=None, help="StationXML subdir (default: <sds_root>/stationxml)")

    # Selection (still applied with domains)
    p.add_argument("--service", default="IRIS", help="FDSN provider or URL for MassDownloader.")
    p.add_argument("--network", default="IU",   help="Network code (e.g., IU).")
    p.add_argument("--station", default="DWPF", help="Station code.")
    p.add_argument("--location", default="*",   help="Location code or '*'.")
    p.add_argument("--channels", default="HH?,BH?,EH?,LH?", help="Channel families (comma-separated).")
    p.add_argument("--id", action="append",
                   help="Optional SEED pattern NET.STA.LOC.CHA (wildcards ok). May be given multiple times.")
    p.add_argument("--ids_file", help="Path to file with one SEED pattern per line.")

    # -------- Domain controls (NEW) --------
    p.add_argument("--domain", choices=["global", "circle", "rect"], default="global",
                   help="Spatial domain filter to combine with network/station selection.")
    # circle
    p.add_argument("--lat", type=float, help="Center latitude for CircularDomain.")
    p.add_argument("--lon", type=float, help="Center longitude for CircularDomain.")
    p.add_argument("--radius_km", type=float, default=None, help="Circle radius (km).")
    p.add_argument("--minradius_km", type=float, default=None, help="Min radius (km).")
    p.add_argument("--maxradius_km", type=float, default=None, help="Max radius (km).")
    p.add_argument("--minradius_deg", type=float, default=None, help="Min radius (deg).")
    p.add_argument("--maxradius_deg", type=float, default=None, help="Max radius (deg).")
    # rect
    p.add_argument("--minlat", type=float, default=None, help="RectangularDomain: min latitude.")
    p.add_argument("--maxlat", type=float, default=None, help="RectangularDomain: max latitude.")
    p.add_argument("--minlon", type=float, default=None, help="RectangularDomain: min longitude.")
    p.add_argument("--maxlon", type=float, default=None, help="RectangularDomain: max longitude.")

    # fdsn2sds knobs
    p.add_argument("--threads", type=int, default=4, help="MassDownloader threads per client.")
    p.add_argument("--chunk",   type=int, default=86400, help="Chunk length seconds (keep daily files).")
    p.add_argument("--minlen",  type=float, default=0.0, help="Minimum fraction of requested chunk to keep.")
    p.add_argument("--reject_gaps", action="store_true", help="Reject channels with gaps.")
    p.add_argument("--sanitize", action="store_true", help="Drop waveforms lacking StationXML.")

    # sds2sam knobs
    p.add_argument("--sampling_interval", type=float, default=60.0, help="SAM Δ (seconds).")
    p.add_argument("--minfreq", type=float, default=0.5, help="Fallback single-band low cut (Hz).")
    p.add_argument("--maxfreq", type=float, default=18.0, help="Fallback single-band high cut (Hz).")
    p.add_argument("--corners", type=int, default=4, help="Filter corners.")
    p.add_argument("--fill_value", type=float, default=0.0, help="Fill masked gaps prior to processing.")
    p.add_argument("--bands", default=None,
                   help="Bands dict as inline JSON or path (.json/.yml).")
    p.add_argument("--bands_preset", choices=list(sds2sam_tool.BAND_PRESETS.keys()), default=None,
                   help="Named preset overrides --minfreq/--maxfreq.")
    p.add_argument("--remove_response", action="store_true", help="Compute VSAM (physical units).")
    p.add_argument("--also_rsam", action="store_true",
                   help="If removing response, also compute RSAM (counts) in parallel.")
    p.add_argument("--stationxml", default=None, help="StationXML file/dir (needed for response removal).")
    p.add_argument("--output", choices=["VEL","DISP","ACC","PA"], default="VEL",
                   help="Physical output when removing response.")
    p.add_argument("--speed", type=int, default=2, choices=[1,2], help="SDS read speed for sds2sam.")
    p.add_argument("--max_rate", type=float, default=250.0, help="Upper bound sample rate during read.")
    p.add_argument("--merge_strategy", default="obspy", help="smart_merge strategy.")
    p.add_argument("--min_rate", type=float, default=None,
                   help="Min sampling rate to keep (Hz). If omitted, auto=2.2×fmax.")
    p.add_argument("--nprocs", default="auto",
                   help="Workers for day-parallel SAM. 'auto' = half of logical CPUs (rounded up).")

    # Reporting
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")

    return p.parse_args(argv)


# ---------------------- main orchestration ----------------------

def main(argv=None):
    args = parse_args(argv)

    # Normalize paths + times
    args.sds_root = _expand(args.sds_root)
    args.sam_root = _expand(args.sam_root)
    if args.stationxml:
        args.stationxml = _expand(args.stationxml)
    start = UTCDateTime(args.start)
    end   = UTCDateTime(args.end)
    if end <= start:
        raise SystemExit("End must be after start.")

    # ---- Validate domain args (quick sanity checks; fdsn2sds will do the real work) ----
    if args.domain == "circle":
        if args.lat is None or args.lon is None:
            raise SystemExit("--domain circle requires --lat and --lon.")
        # At least one radius spec:
        if (args.radius_km is None and
            args.minradius_km is None and args.maxradius_km is None and
            args.minradius_deg is None and args.maxradius_deg is None):
            raise SystemExit("--domain circle requires one of: --radius_km OR (min/max radius in km or deg).")
    elif args.domain == "rect":
        need = [args.minlat, args.maxlat, args.minlon, args.maxlon]
        if any(v is None for v in need):
            raise SystemExit("--domain rect requires --minlat --maxlat --minlon --maxlon.")

    # ---------- 1) Determine missing days from FINAL files ----------
    vsam_missing = None
    rsam_missing = None

    if args.remove_response:
        vsam_missing = VSAM.missing_days(
            startt=start, endt=end, SAM_DIR=args.sam_root,
            trace_ids=None, network=args.network,
            sampling_interval=int(args.sampling_interval),
            ext="pickle", require_full_day=False, tol=0.05,
            verbose=args.verbose,
        )
        if args.also_rsam:
            rsam_missing = RSAM.missing_days(
                startt=start, endt=end, SAM_DIR=args.sam_root,
                trace_ids=None, network=args.network,
                sampling_interval=int(args.sampling_interval),
                ext="pickle", require_full_day=False, tol=0.05,
                verbose=args.verbose,
            )
    else:
        rsam_missing = RSAM.missing_days(
            startt=start, endt=end, SAM_DIR=args.sam_root,
            trace_ids=None, network=args.network,
            sampling_interval=int(args.sampling_interval),
            ext="pickle", require_full_day=False, tol=0.05,
            verbose=args.verbose,
        )

    days_needed = _union_missing_days(vsam_missing, rsam_missing)

    if args.verbose:
        total = len(days_needed)
        if total:
            first = str(days_needed[0].date())
            last  = str(days_needed[-1].date())
            print(f"[wrapper] Missing UTC days: {total}  (range {first} → {last})")
        else:
            print("[wrapper] Nothing to do: all days covered in final files.")
    if not days_needed:
        return 0

    # ---------- 2) Download SDS only for missing day ranges ----------
    for d0, d1 in _contiguous_day_ranges(days_needed):
        dl_start = d0
        dl_end   = d1  # end exclusive for MassDownloader

        fdsn_args = [
            "--service", args.service,
            "--start", str(dl_start), "--end", str(dl_end),
            "--sds_root", args.sds_root,
            "--threads", str(args.threads),
            "--chunk",   str(args.chunk),
            "--minlen",  str(args.minlen),
        ]

        # Selection (honored alongside domain)
        fdsn_args += [
            "--network",  args.network,
            "--station",  args.station,
            "--location", args.location,
            "--channels", args.channels,
        ]

        # Domain passthrough
        fdsn_args += ["--domain", args.domain]
        if args.domain == "circle":
            fdsn_args += ["--lat", str(args.lat), "--lon", str(args.lon)]
            if args.minradius_deg is not None: fdsn_args += ["--minradius_deg", str(args.minradius_deg)]
            if args.maxradius_deg is not None: fdsn_args += ["--maxradius_deg", str(args.maxradius_deg)]
            if args.minradius_km  is not None: fdsn_args += ["--minradius_km",  str(args.minradius_km)]
            if args.maxradius_km  is not None: fdsn_args += ["--maxradius_km",  str(args.maxradius_km)]
            if args.radius_km     is not None: fdsn_args += ["--radius_km",     str(args.radius_km)]
        elif args.domain == "rect":
            fdsn_args += [
                "--minlat", str(args.minlat), "--maxlat", str(args.maxlat),
                "--minlon", str(args.minlon), "--maxlon", str(args.maxlon),
            ]

        # Misc downloader flags
        if args.reject_gaps:
            fdsn_args += ["--reject_gaps"]
        if args.sanitize:
            fdsn_args += ["--sanitize"]
        if args.save_inventory:
            fdsn_args += ["--save_inventory"]
            if args.metadata_dir:
                fdsn_args += ["--metadata_dir", args.metadata_dir]
        if args.id:
            for pat in args.id:
                fdsn_args += ["--id", pat]
        if args.ids_file:
            fdsn_args += ["--ids_file", args.ids_file]
        if args.verbose:
            fdsn_args += ["--verbose"]

        if args.verbose:
            print(f"[wrapper] fdsn2sds → {dl_start} … {dl_end}  ({' '.join(fdsn_args)})")
        fdsn_tool.main(fdsn_args)

    # ---------- 3) Compute SAM for those missing day ranges ----------
    nprocs = _pick_nprocs_auto() if str(args.nprocs).lower() == "auto" else max(1, int(args.nprocs))

    for d0, d1 in _contiguous_day_ranges(days_needed):
        sds2sam_args = [
            "--sds_root", args.sds_root,
            "--sam_root", args.sam_root,
            "--start", str(d0),
            "--end",   str(d1),
            "--network", args.network,
            "--station", args.station,
            "--location", args.location,
            "--channel", "*Z",
            "--sampling_interval", str(int(args.sampling_interval)),
            "--minfreq", str(float(args.minfreq)),
            "--maxfreq", str(float(args.maxfreq)),
            "--corners", str(int(args.corners)),
            "--fill_value", str(float(args.fill_value)),
            "--speed", str(int(args.speed)),
            "--max_rate", str(float(args.max_rate)),
            "--merge_strategy", args.merge_strategy,
            "--nprocs", str(nprocs),
        ]
        # bands or preset
        if args.bands:
            sds2sam_args += ["--bands", args.bands]
        elif args.bands_preset:
            sds2sam_args += ["--bands-preset", args.bands_preset]

        # response removal
        if args.remove_response:
            sds2sam_args += ["--remove_response", "--output", args.output]
            if args.stationxml:
                sds2sam_args += ["--stationxml", args.stationxml]
            if args.also_rsam:
                sds2sam_args += ["--also_rsam"]

        # min_rate (optional)
        if args.min_rate is not None:
            sds2sam_args += ["--min_rate", str(float(args.min_rate))]

        if args.verbose:
            sds2sam_args += ["--verbose"]
            print(f"[wrapper] sds2sam → {d0.date} … {d1.date}  ({' '.join(sds2sam_args)})")

        sds2sam_tool.main(sds2sam_args)

    if args.verbose:
        print("[wrapper] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))