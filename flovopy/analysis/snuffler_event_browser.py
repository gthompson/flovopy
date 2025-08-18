# snuffler_event_browser_cli.py
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", default="/data/KSC/all_florida_launches", help="Root with event MiniSEEDs")
    ap.add_argument("--pattern", default="**/*.mseed", help="Glob under --root")
    ap.add_argument("--stationxml", default="~/Dropbox/DATA/station_metadata/KSC2.xml", help="StationXML path")
    ap.add_argument("--start", type=int, default=0, help="Start index in sorted file list")
    ap.add_argument("--limit", type=int, default=None, help="Max files to open")
    ap.add_argument("--all-in-one", action="store_true", help="Open all selected files in one Snuffler session")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    sx   = Path(args.stationxml).expanduser().resolve()
    if not root.exists(): sys.exit(f"✘ root not found: {root}")
    if not sx.exists():   sys.exit(f"✘ StationXML not found: {sx}")

    files = sorted(root.glob(args.pattern))
    if not files: sys.exit(f"✘ no files under {root} with {args.pattern}")
    sel = files[args.start: (args.start + args.limit) if args.limit else None]
    print(f"✔ opening {len(sel)} of {len(files)} files (start={args.start}"
          f"{'' if args.limit is None else f', limit={args.limit}'})")

    base_cmd = ["snuffler", "--format=mseed", f"--stationxml={sx}"]

    if args.all_in_one:
        # if there are *many* files, consider chunking to avoid OS arg limits
        cmd = base_cmd + [str(p) for p in sel]
        subprocess.call(cmd)
    else:
        for i, p in enumerate(sel, 1):
            print(f"[{i}/{len(sel)}] {p.name}")
            subprocess.call(base_cmd + [str(p)])

if __name__ == "__main__":
    main()