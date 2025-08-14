#!/usr/bin/env python3
"""
fdsn2sds.py — Thin wrapper using ObsPy MassDownloader to build an SDS archive.

- Writes daily MiniSEED files in SDS layout:
    <SDS_ROOT>/<YYYY>/<NET>/<STA>/<CHAN>.D/NET.STA.LOC.CHAN.D.YYYY.JJJ
- Saves StationXML (level=response) next to the archive in:
    <SDS_ROOT>/metadata/<NET>.<STA>.<LOC>.<CHANS>_<YYYYMMDD>-<YYYYMMDD>.xml
- Respects existing files (MassDownloader will skip them).
- Multithreaded downloads via MassDownloader (per FDSN client).

Example:
    python fdsn2sds.py --service IRIS --network IU --station DWPF \
        --start 2011-03-10 --end 2011-03-15 \
        --sds_root ./SDS --threads 4 --save_inventory --verbose
"""

import argparse
import os
from pathlib import Path

from obspy import UTCDateTime

from obspy.clients.fdsn.mass_downloader import (
    Restrictions,
    GlobalDomain,
    CircularDomain,
    RectangularDomain,
)
from obspy.clients.fdsn.mass_downloader.mass_downloader import MassDownloader

from pprint import pprint
from typing import Optional


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None):
  
    p = argparse.ArgumentParser(
        description="Download waveforms + StationXML via ObsPy MassDownloader and store in SDS layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # FDSN
    p.add_argument("--service", default="IRIS", help="FDSN provider or base URL (e.g., IRIS).")
    p.add_argument("--network", default="IU", help="FDSN network code (e.g., IU, II).")
    p.add_argument("--station", default="DWPF", help="FDSN station code.")
    p.add_argument("--location", default="*", help="FDSN location code (e.g., 00, 10, or '*').")
    p.add_argument(
        "--channels",
        default="HH?,BH?,EH?,LH?",
        help="Comma-separated channel families (e.g., 'HH?,BH?'). Z/N/E are auto-expanded.",
    )

    # Time window (end is exclusive)
    p.add_argument("--start", default="2011-03-10", help="Start time (YYYY-MM-DD[THH:MM:SS]).")
    p.add_argument("--end",   default="2011-03-15", help="End time (YYYY-MM-DD[THH:MM:SS], exclusive).")

    # SDS + metadata
    p.add_argument("--sds_root", required=True, help="Path to SDS archive root.")
    p.add_argument("--metadata_dir", default=None, help="Where to store StationXML (default: <sds_root>/metadata).")
    p.add_argument("--save_inventory", action="store_true", help="Also save full StationXML (level=response).")

    # Downloader settings
    p.add_argument("--threads", type=int, default=3, help="Threads per FDSN client.")
    p.add_argument("--chunk", type=int, default=86400, help="Chunk length in seconds (86400 = daily files).")
    p.add_argument("--minlen", type=float, default=0.0, help="Minimum fraction of requested chunk to keep (0.0–1.0).")
    p.add_argument("--reject_gaps", action="store_true", help="Reject channels with gaps in chunk.")
    p.add_argument("--sanitize", action="store_true", help="Drop waveforms lacking matching StationXML.")
    p.add_argument("--verbose", action="store_true", help="Print extra logging.")

    # Domain selection
    p.add_argument("--domain", choices=["global", "circle", "rect"], default="global",
                   help="Spatial domain for station selection.")
    # CircularDomain
    p.add_argument("--lat", type=float, help="Center latitude for --domain circle.")
    p.add_argument("--lon", type=float, help="Center longitude for --domain circle.")
    p.add_argument("--minradius_deg", type=float, default=None,
                   help="Minimum great-circle radius (degrees) for --domain circle.")
    p.add_argument("--maxradius_deg", type=float, default=None,
                   help="Maximum great-circle radius (degrees) for --domain circle.")
    p.add_argument("--minradius_km", type=float, default=None,
                   help="Minimum great-circle radius (km) for --domain circle.")
    p.add_argument("--maxradius_km", type=float, default=None,
                   help="Maximum great-circle radius (km) for --domain circle.")
    p.add_argument("--radius_km", type=float, default=None,
                   help="Shortcut for 0 → radius_km for --domain circle.")

    # RectangularDomain
    p.add_argument("--minlat", type=float, help="Min latitude for --domain rect.")
    p.add_argument("--maxlat", type=float, help="Max latitude for --domain rect.")
    p.add_argument("--minlon", type=float, help="Min longitude for --domain rect.")
    p.add_argument("--maxlon", type=float, help="Max longitude for --domain rect.")

    # Wildcard IDs
    p.add_argument("--id", action="append",
                help="SEED ID pattern 'NET.STA.LOC.CHA' (wildcards allowed). May be given multiple times.")
    p.add_argument("--ids_file",
                help="Path to a text file with one SEED ID pattern per line (wildcards allowed).")
    return p.parse_args(argv)


# -----------------------------
# Helpers
# -----------------------------

def _km_to_deg(km: float) -> float:
    return km / 111.195 if km is not None else None


def parse_seed_id(seed: str):
    parts = seed.strip().split(".")
    if len(parts) != 4:
        raise ValueError(f"Bad SEED pattern (need 4 dot-separated fields): {seed}")
    net, sta, loc, cha = (p if p else "*" for p in parts)
    return net, sta, loc, cha


def expand_channels(families_csv: str) -> str:
    """
    Expand 'HH?,BH?' into 'HHZ,HHN,HHE,BHZ,BHN,BHE' for MassDownloader.
    """
    families = [f.strip() for f in families_csv.split(",") if f.strip()]
    chans = []
    for fam in families:
        if "?" in fam:
            chans += [fam.replace("?", c) for c in ("Z", "N", "E")]
        else:
            chans.append(fam)
    return ",".join(chans)


def sds_waveform_path_builder(sds_root: str):
    """
    Returns a function used by MassDownloader to determine the SDS file path
    for each chunk. If the path already exists, MassDownloader will skip it.
    """
    sds_root = Path(sds_root)

    def _path(network, station, location, channel, starttime, endtime):
        year = f"{starttime.year:04d}"
        jday = f"{starttime.julday:03d}"
        loc = (location or "--").zfill(2)
        subdir = sds_root / year / network / station / f"{channel}.D"
        subdir.mkdir(parents=True, exist_ok=True)
        filename = f"{network}.{station}.{loc}.{channel}.D.{year}.{jday}"
        return str(subdir / filename)

    return _path


def stationxml_path_builder(sds_root: str, subdir: Optional[str] = "stationxml"):
    """
    Returns a callable for MassDownloader that writes one StationXML per (network, station).
    Files go to <sds_root>/<subdir or 'stationxml'>/<NET>.<STA>.xml
    """
    subdir = subdir or "stationxml"         # <-- handle None
    base = os.path.join(os.path.expanduser(sds_root), subdir)
    os.makedirs(base, exist_ok=True)

    def _path(network, station, channels=None, starttime=None, endtime=None):
        return os.path.join(base, f"{network}.{station}.xml")

    return _path


# -----------------------------
# Main
# -----------------------------
def main(argv=None):
    import os
    from obspy import UTCDateTime
    #from obspy.clients.fdsn import MassDownloader
    #from obspy.clients.fdsn.mass_downloader import Restrictions, GlobalDomain

    # Parse either sys.argv[1:] (default) or a provided list
    args = parse_args(argv)

    # Expand user dirs early so downstream paths are absolute
    args.sds_root = os.path.expanduser(args.sds_root)
    if getattr(args, "metadata_dir", None):
        args.metadata_dir = os.path.expanduser(args.metadata_dir)

    start = UTCDateTime(args.start)
    end = UTCDateTime(args.end)
    if end <= start:
        raise SystemExit("End must be after start.")

    # Prepare storage callbacks (you already have these helpers)
    mseed_storage = sds_waveform_path_builder(args.sds_root)

    if args.save_inventory:
        stationxml_storage = stationxml_path_builder(args.sds_root, args.metadata_dir)
    else:
        stationxml_storage = None

    # Build channel list and priorities (Z/N/E)
    expanded_channels = expand_channels(args.channels)
    channel_priorities = [
        f"{fam}[ZNE]" if fam.endswith("?") else f"{fam[0:2]}[ZNE]"
        for fam in [c for c in args.channels.split(",") if c]
    ]

    # Build Restrictions (single wildcard set OR list of SEED ID patterns)
    restrictions_list = []

    if args.id or args.ids_file:
        ids = []
        if args.id:
            ids.extend(args.id)
        if args.ids_file:
            with open(args.ids_file) as f:
                ids.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])

        for pattern in ids:
            net, sta, loc, cha = parse_seed_id(pattern)
            chan = expand_channels(cha) if "?" in cha else cha
            restrictions_list.append(
                Restrictions(
                    starttime=start, endtime=end,
                    network=net, station=sta, location=loc, channel=chan,
                    chunklength_in_sec=int(args.chunk),
                    channel_priorities=channel_priorities,
                    location_priorities=["", "00", "10", "*"],
                    reject_channels_with_gaps=bool(args.reject_gaps),
                    minimum_length=float(args.minlen),
                    sanitize=bool(args.sanitize),
                )
            )
    else:
        restrictions_list.append(
            Restrictions(
                starttime=start, endtime=end,
                network=args.network or None,
                station=args.station or None,
                location=args.location or None,
                channel=expanded_channels or None,
                chunklength_in_sec=int(args.chunk),
                channel_priorities=channel_priorities,
                location_priorities=[args.location] if args.location not in ("*", None) else ["", "00", "10", "*"],
                reject_channels_with_gaps=bool(args.reject_gaps),
                minimum_length=float(args.minlen),
                sanitize=bool(args.sanitize),
            )
        )


    # --- Build domain ---
    if args.domain == "global":
        domain = GlobalDomain()

    elif args.domain == "circle":
        if args.lat is None or args.lon is None:
            raise SystemExit("--lat and --lon are required for --domain circle")

        # Prefer degrees if supplied; otherwise use km; otherwise use --radius_km
        min_deg = args.minradius_deg
        max_deg = args.maxradius_deg

        if min_deg is None and args.minradius_km is not None:
            min_deg = _km_to_deg(args.minradius_km)
        if max_deg is None and args.maxradius_km is not None:
            max_deg = _km_to_deg(args.maxradius_km)

        if min_deg is None and max_deg is None and args.radius_km is not None:
            min_deg, max_deg = 0.0, _km_to_deg(args.radius_km)

        # Sensible defaults if only one bound supplied
        if min_deg is None and max_deg is not None:
            min_deg = 0.0
        if max_deg is None and min_deg is not None:
            max_deg = min_deg

        if min_deg is None or max_deg is None:
            raise SystemExit("Provide --minradius_deg/--maxradius_deg OR km variants (or --radius_km) for --domain circle")

        domain = CircularDomain(latitude=args.lat, longitude=args.lon,
                                minradius=float(min_deg), maxradius=float(max_deg))

    elif args.domain == "rect":
        needed = [args.minlat, args.maxlat, args.minlon, args.maxlon]
        if any(v is None for v in needed):
            raise SystemExit("For --domain rect you must give --minlat --maxlat --minlon --maxlon")
        domain = RectangularDomain(minlatitude=float(args.minlat),
                                   maxlatitude=float(args.maxlat),
                                   minlongitude=float(args.minlon),
                                   maxlongitude=float(args.maxlon))


    if args.verbose:
        print("Using MassDownloader with:")
        print(f"  provider: {args.service}")
        print(f"  time: {start} → {end}  (chunk={args.chunk}s)")
        if isinstance(domain, GlobalDomain):
            print("  domain: GlobalDomain()")
        elif isinstance(domain, CircularDomain):
            print(f"  domain: CircularDomain(lat={args.lat}, lon={args.lon}, "
                  f"minradius={domain.minradius}°, maxradius={domain.maxradius}°)")
        else:
            print(f"  domain: RectangularDomain(lat=[{args.minlat},{args.maxlat}], "
                  f"lon=[{args.minlon},{args.maxlon}])")
        print(f"  filters: net={args.network}, sta={args.station}, loc={args.location}")
        print(f"  channels: {expanded_channels}")
        print(f"  SDS root: {args.sds_root}")
        if stationxml_storage:
            print(f"  StationXML → {stationxml_storage}")

    mdl = MassDownloader(providers=[args.service])

    domain_pos        = domain
    restrictions_pos  = (restrictions_list[0] if len(restrictions_list) == 1 else restrictions_list)
    mseed_storage_pos = mseed_storage
    # Always save StationXML under <SDS_ROOT>/<metadata_dir or "stationxml">/
    stationxml_pos    = stationxml_path_builder(args.sds_root, args.metadata_dir or "stationxml")

    print("Starting download with the following arguments")
    print({
        "domain": domain_pos,
        "restrictions": restrictions_pos,
        "mseed_storage": mseed_storage_pos,
        "stationxml_storage": f"{args.sds_root}/{args.metadata_dir or 'stationxml'}",
        "threads_per_client": max(1, int(args.threads)),
        "download_chunk_size_in_mb": 50,
    })

    mdl.download(
        domain_pos,
        restrictions_pos,
        mseed_storage_pos,
        stationxml_pos,                          # <— always save StationXML
        threads_per_client=max(1, int(args.threads)),
        download_chunk_size_in_mb=50,
    )

    if args.verbose:
        print("Done.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])   # pass real CLI args through

"""
python fdsn2sds.py \
  --service IRIS --network IU --station DWPF --channels BH? \
  --start 2011-03-10 --end 2011-03-15 \
  --domain global \
  --sds_root ~/work/SDS --threads 4 --save_inventory --verbose

python fdsn2sds.py \
  --service IRIS --channels BH? \
  --start 2011-03-10 --end 2011-03-15 \
  --domain circle --lat 19.421 --lon -155.287 --radius_km 15 \
  --sds_root ~/work/SDS --threads 4 --save_inventory --verbose

python fdsn2sds.py \
  --service IRIS --channels BH? \
  --start 2011-03-10 --end 2011-03-15 \
  --domain circle --lat 37.52 --lon 143.04 \
  --minradius_deg 70 --maxradius_deg 90 \
  --sds_root ~/work/SDS --threads 4 --verbose

python fdsn2sds.py \
  --service IRIS --channels BH? \
  --start 2011-03-10 --end 2011-03-15 \
  --domain rect --minlat 19.2 --maxlat 19.6 --minlon -155.6 --maxlon -155.1 \
  --sds_root ~/work/SDS --threads 4 --save_inventory --verbose

"""