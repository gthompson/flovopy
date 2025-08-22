#!/usr/bin/env python3
"""
sds_to_bud_symlinks.py

Create a BUD-style directory of symlinks pointing back to daily SDS MiniSEED files.

Assumptions:
- SDS layout: <SDS_ROOT>/<YYYY>/<NET>/<STA>/<CHAN>.D/NET.STA.LOC.CHAN.D.YYYY.DDD[.ext]
- BUD layout: <BUD_ROOT>/<NET>/<STA>/<CHAN>/STA.LOC.CHAN.YYYY.DDD[.ext]
- We use DAILY BUD filenames (no HH.MM.SS) because symlinks cannot represent hourly slices.
- Location code:
  - In SDS, blank often appears as '--'
  - In BUD filenames, blank is commonly represented as '..' (two dots)
"""

import argparse
import os
from pathlib import Path

def parse_sds_filename(name: str):
    """
    Parse SDS filename: NET.STA.LOC.CHAN.D.YYYY.DDD[.ext]
    Return dict with keys: net, sta, loc, chan, year, doy, ext
    Raise ValueError if it doesn't match the expected pattern.
    """
    parts = name.split(".")
    # Allow extensions like .mseed / .miniseed / .gz (so there may be > 8 parts total)
    # We expect at least: NET STA LOC CHAN D YYYY DDD
    if len(parts) < 7:
        raise ValueError(f"Not enough dot-separated fields: {name}")
    # The last two always should be YYYY and DDD (before any extra suffixes/extensions)
    # Find the 'D' token that precedes YYYY
    try:
        d_index = parts.index("D")
    except ValueError:
        # Some layouts use CHAN.D as a combined token; try to split that
        # e.g., CHAN='BHZ', token 'BHZ', next token 'D'
        # If not found, raise.
        raise ValueError(f"Missing 'D' token in: {name}")

    # Positions:
    # 0: NET
    # 1: STA
    # 2: LOC
    # 3: CHAN
    # 4: D (at d_index)
    # 5: YYYY (d_index+1)
    # 6: DDD  (d_index+2)
    # extensions beyond index 6 (e.g., .mseed, .gz)
    if d_index < 4 or len(parts) < d_index + 3:
        raise ValueError(f"Unexpected token positions in: {name}")

    net = parts[0]
    sta = parts[1]
    loc = parts[2]
    chan = parts[3]
    if parts[d_index] != "D":
        raise ValueError(f"Expected 'D' token, got {parts[d_index]} in: {name}")
    year = parts[d_index + 1]
    doy = parts[d_index + 2]
    ext = ""
    if len(parts) > d_index + 3:
        # Rebuild extension including the leading dot(s)
        ext = "." + ".".join(parts[d_index + 3:])

    # Basic sanity
    if not (len(year) == 4 and year.isdigit()):
        raise ValueError(f"Bad year in: {name}")
    if not (len(doy) == 3 and doy.isdigit()):
        raise ValueError(f"Bad day-of-year in: {name}")

    return dict(net=net, sta=sta, loc=loc, chan=chan, year=year, doy=doy, ext=ext)

def sds_day_dir(root: Path, year: str, net: str, sta: str, chan: str) -> Path:
    # <SDS_ROOT>/<YYYY>/<NET>/<STA>/<CHAN>.D/
    return root / year / net / sta / f"{chan}.D"

def bud_dir(root: Path, net: str, sta: str, chan: str) -> Path:
    # <BUD_ROOT>/<NET>/<STA>/<CHAN>/
    return root / net / sta / chan

def bud_filename(sta: str, net: str, loc: str, chan: str, year: str, doy: str, ext: str, loc_blank_as="..") -> str:
    # BUD filename daily form: STA.LOC.CHAN.YYYY.DDD[.ext]
    loc_field = loc_blank_as if (loc == "" or loc == "--") else loc
    return f"{sta}.{net}.{loc_field}.{chan}.{year}.{doy}{ext}"

def create_symlink(src: Path, dest: Path, overwrite: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if overwrite:
            try:
                dest.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed to remove existing: {dest} ({e})")
        else:
            return  # leave existing link/file in place
    os.symlink(src, dest)

def collect_sds_files(sds_root: Path):
    """
    Yield (src_path, parsed_dict) for SDS files that look like daily MiniSEED.
    """
    # Pattern: <SDS_ROOT>/<YYYY>/<NET>/<STA>/<CHAN>.D/*.*
    for year_dir in sds_root.iterdir():
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        if not (len(year) == 4 and year.isdigit()):
            continue
        for net_dir in year_dir.iterdir():
            if not net_dir.is_dir():
                continue
            for sta_dir in net_dir.iterdir():
                if not sta_dir.is_dir():
                    continue
                for chanD_dir in sta_dir.iterdir():
                    if not chanD_dir.is_dir() or not chanD_dir.name.endswith(".D"):
                        continue
                    for f in chanD_dir.iterdir():
                        if not f.is_file():
                            continue
                        try:
                            parsed = parse_sds_filename(f.name)
                        except ValueError:
                            continue
                        # basic consistency check
                        if parsed["year"] != year:
                            continue
                        yield f, parsed

def main():
    ap = argparse.ArgumentParser(description="Create BUD-style symlinks from SDS daily MiniSEED files.")
    ap.add_argument("sds_root", type=Path, help="Root of SDS tree")
    ap.add_argument("bud_root", type=Path, help="Root where BUD tree (symlinks) will be created")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing symlinks/files at destination")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be linked without making changes")
    ap.add_argument("--loc-blank-as", default="..", help="String to use for blank/-- location code in BUD filename (default: '..')")
    args = ap.parse_args()

    sds_root = args.sds_root.resolve()
    bud_root = args.bud_root.resolve()

    count = 0
    skipped = 0
    for src, p in collect_sds_files(sds_root):
        # Build destination path
        dest_dir = bud_dir(bud_root, p["net"], p["sta"], p["chan"])
        dest_name = bud_filename(p["sta"], p['net'], p["loc"], p["chan"], p["year"], p["doy"], p["ext"], loc_blank_as=args.loc_blank_as)
        dest = dest_dir / dest_name

        rel_src = os.path.relpath(src, dest_dir)  # nice relative symlink
        if args.dry_run:
            print(f"LINK: {dest} -> {rel_src}")
            count += 1
            continue

        try:
            create_symlink(Path(rel_src), dest, overwrite=args.overwrite)
            count += 1
        except Exception as e:
            print(f"SKIP: {dest} ({e})")
            skipped += 1

    print(f"Done. Created {count} link(s), skipped {skipped}.")

if __name__ == "__main__":
    main()