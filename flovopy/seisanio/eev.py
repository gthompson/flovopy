#!/usr/bin/env python3
"""
eev.py — EEV-style event browser for MiniSEED files.

Commands:
  ?            help
  q            quit
  b            back one event
  <ENTER>      next event
  l            list events (compact table)
  <int>        go to event number (1-based)
  t            type event (detailed info)
  tt           header only (less detailed)
  w            show path of waveform file
  p            plot event with ObsPy (equal_scale=False)
  s            open event in Snuffler (CLI)
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional

STATE_FILE = Path.home() / ".eev_like_state.json"

# ------------------------- helpers -------------------------

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    x = float(n); k = 0
    while x >= 1024.0 and k < len(units)-1:
        x /= 1024.0; k += 1
    return f"{x:.1f} {units[k]}"

def parse_evt_times_from_name(name: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    evt_XXXXXX.2016-03-04T23-34-00.000000Z_2016-03-05T01-16-00.000000Z.mseed
    -> (UTC datetime start, UTC datetime end)
    """
    try:
        base = name.split("evt_", 1)[-1]
        parts = base.split(".")
        if len(parts) < 2:
            return None, None
        tail = ".".join(parts[1:])
        t0s, t1s = tail.split("_", 1)
        t1s = t1s.rsplit(".mseed", 1)[0]

        def fix(ts: str) -> str:
            # 2016-03-04T23-34-00.000000Z -> 2016-03-04T23:34:00.000000Z
            return ts.replace("-", ":", 2)

        t0 = fix(t0s).rstrip("Z")
        t1 = fix(t1s).rstrip("Z")
        dt0 = datetime.fromisoformat(t0.replace(" ", "T")).replace(tzinfo=timezone.utc)
        dt1 = datetime.fromisoformat(t1.replace(" ", "T")).replace(tzinfo=timezone.utc)
        return dt0, dt1
    except Exception:
        return None, None

def quick_mseed_header(path: Path):
    """
    Fast header look via ObsPy headonly read.
    Returns (n_traces, ids_preview[list[str]], has_more[bool])
    """
    try:
        from obspy import read
        st = read(str(path), headonly=True)
        ids = sorted({tr.id for tr in st})
        max_show = 8
        return len(st), (ids[:max_show]), (len(ids) > max_show)
    except Exception:
        return None, [], False

def print_info(files: List[Path], idx: int, detailed: bool):
    p = files[idx]
    size = human_bytes(p.stat().st_size) if p.exists() else "?"
    t0, t1 = parse_evt_times_from_name(p.name)
    if t0 and t1:
        dur = (t1 - t0).total_seconds()
        t0s = t0.isoformat().replace("+00:00","Z")
        t1s = t1.isoformat().replace("+00:00","Z")
        t0local = t0.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        t1local = t1.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        dur_str = f"{dur:.0f}s ({dur/60.0:.1f} min)"
    else:
        t0s = t1s = t0local = t1local = dur_str = "?"

    ntr, ids_preview, more = quick_mseed_header(p)
    ids_str = ", ".join(ids_preview) + (" …" if more else "")
    ntr_str = f"{ntr}" if ntr is not None else "?"

    print(f"\n— Event {idx+1}/{len(files)} —")
    print(f"File: {p.name}")
    print(f"Size: {size}")
    if t0 and t1:
        print(f"UTC:   {t0s} → {t1s}  ({dur_str})")
        if detailed:
            print(f"Local: {t0local} → {t1local}")
    else:
        print("UTC:   (unparsed from filename)")

    print(f"Traces: {ntr_str}")
    print(f"NSLC:   {ids_str}")
    if detailed:
        # add on-disk path and mtime in detailed view
        print(f"Path:   {p}")
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"mtime:  {mtime}")
        except Exception:
            pass

def list_events(files: List[Path], around_idx: int, width: int = 20):
    """
    Compact table: index, start time (from filename if parsable), ntraces preview.
    """
    start = max(0, around_idx - width//2)
    end = min(len(files), start + width)
    start = max(0, end - width)  # keep width rows if near end

    rows = []
    for i in range(start, end):
        p = files[i]
        t0, t1 = parse_evt_times_from_name(p.name)
        t0s = t0.isoformat().replace("+00:00","Z") if t0 else "?"
        ntr, ids_preview, more = quick_mseed_header(p)
        ids_str = ", ".join(ids_preview) + (" …" if more else "")
        rows.append((i+1, p.name, t0s, ntr if ntr is not None else "?", ids_str))

    # pretty print
    print("\n#   | File                                                | t0 (UTC)                  | ntr | NSLC preview")
    print("-"*120)
    for (i, name, t0s, ntr, ids_str) in rows:
        mark = "->" if (i-1) == around_idx else "  "
        print(f"{mark} {i:4d} | {name[:52]:52s} | {t0s:25s} | {str(ntr):>3s} | {ids_str}")

def snuffler_cli(path: Path, stationxml: Optional[str]):
    cmd = ["snuffler", "--format=mseed"]
    if stationxml:
        cmd.append(f"--stationxml={stationxml}")
    cmd.append(str(path))
    print("Launching:", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.call(cmd)
    except FileNotFoundError:
        print("✘ 'snuffler' not found on PATH. Install Pyrocko or add snuffler to PATH.")

def plot_obspy(path: Path):
    try:
        from obspy import read
        st = read(str(path))
        print(f"ObsPy stream: {len(st)} traces — plotting… (close the window to return)")
        st.plot(equal_scale=False)  # blocks until closed
    except Exception as e:
        print(f"✘ ObsPy plot failed: {e}")

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_state(root: Path, pattern: str, idx: int):
    state = {"root": str(root), "pattern": pattern, "index": idx}
    STATE_FILE.write_text(json.dumps(state))

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", default="/data/KSC/all_florida_launches", help="Root with event MiniSEEDs")
    ap.add_argument("--pattern", default="**/*.mseed", help="Glob pattern under --root")
    ap.add_argument("--stationxml", default="~/Dropbox/DATA/station_metadata/KSC2.xml", help="StationXML path (optional)")
    ap.add_argument("--resume", action="store_true", help="Resume from last index if state file exists")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.exit(f"✘ root not found: {root}")

    sx = Path(args.stationxml).expanduser()
    stationxml = str(sx) if sx.exists() else None
    if stationxml is None:
        print(f"⚠ StationXML not found at {sx}; continuing without it.")

    files = sorted(p for p in root.glob(args.pattern) if p.is_file())
    if not files:
        sys.exit(f"✘ no files under {root} with pattern {args.pattern}")

    # starting index
    idx = 0
    if args.resume:
        st = load_state()
        if st.get("root") == str(root) and st.get("pattern") == args.pattern:
            idx = max(0, min(int(st.get("index", 0)), len(files)-1))

    # initial summary
    print(f"EEV — {len(files)} events (root={root}, pattern='{args.pattern}')")
    print("Type '?' for help.")
    print_info(files, idx, detailed=False)

    while True:
        try:
            line = input("eev> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            line = "q"

        # ENTER → next
        if line == "":
            idx = (idx + 1) % len(files)
            print_info(files, idx, detailed=False)
            continue

        # integer → jump
        if line.isdigit():
            n = int(line)
            if 1 <= n <= len(files):
                idx = n - 1
                print_info(files, idx, detailed=False)
            else:
                print("out of range")
            continue

        # commands
        if line in ("q", "quit"):
            save_state(root, args.pattern, idx)
            print("Bye.")
            break

        if line in ("?", "h", "help"):
            print(__doc__)
            continue

        if line == "b":
            idx = (idx - 1) % len(files)
            print_info(files, idx, detailed=False)
            continue

        if line == "l":
            list_events(files, idx)
            continue

        if line == "t":
            print_info(files, idx, detailed=True)
            continue

        if line == "tt":
            print_info(files, idx, detailed=False)
            continue

        if line == "w":
            print(files[idx])
            continue

        if line == "p":
            plot_obspy(files[idx])
            continue

        if line == "s":
            snuffler_cli(files[idx], stationxml)
            continue

        print("Unknown command. Type '?' for help.")

if __name__ == "__main__":
    main()