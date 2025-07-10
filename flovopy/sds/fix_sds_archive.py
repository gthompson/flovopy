#!/usr/bin/env python3
import os
import argparse
from obspy import read, UTCDateTime
from flovopy.sds.sds import SDSobj

def is_valid_sds_filename(filename):
    """Check if the filename matches NET.STA.LOC.CHAN.D.YYYY.JJJ format."""
    parts = filename.split('.')
    return len(parts) == 7 and parts[4] == 'D'

def save_processed_dir(log_file, directory):
    with open(log_file, 'a') as f:
        f.write(f"{directory}\n")

def get_processed_dirs(log_file):
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f.readlines())


def fix_sds_archive(
    src_dir,
    dest_dir=None,
    networks='*',
    stations='*',
    start_date=None,
    end_date=None,
    write=False,
    log_file='fix_sds_archive.log',
    metadata_excel_path=None
):
    networks = [networks] if isinstance(networks, str) else networks
    stations = [stations] if isinstance(stations, str) else stations
    start_date = UTCDateTime(start_date) if start_date else None
    end_date = UTCDateTime(end_date) if end_date else None

    sdsout = SDSobj(dest_dir) if dest_dir else SDSobj(src_dir)

    if metadata_excel_path:
        sdsout.load_metadata_from_excel(metadata_excel_path)

    processed_dirs = get_processed_dirs(log_file)

    for root, dirs, files in os.walk(src_dir, topdown=True):
        dirs.sort()
        files.sort()

        if root in processed_dirs:
            print(f"Skipping already processed: {root}")
            continue

        for filename in files:
            if not is_valid_sds_filename(filename):
                continue

            parts = filename.split('.')
            net, sta, loc, chan, _, yyyy, jjj = parts

            if networks != ['*'] and net not in networks:
                continue
            if stations != ['*'] and sta not in stations:
                continue

            try:
                file_path = os.path.join(root, filename)
                stream = read(file_path)

                for tr in stream:
                    if start_date and tr.stats.endtime < start_date:
                        continue
                    if end_date and tr.stats.starttime > end_date:
                        continue

                    if sdsout.metadata is not None:
                        sdsout.match_metadata(tr)

                    if write:
                        sdsout.write(tr, overwrite=True)
                        print(f"✔ Wrote: {tr.id} to {sdsout.get_fullpath(tr)}")

                if write:
                    os.remove(file_path)
            except Exception as e:
                print(f"✘ Error processing {filename}: {e}")

        save_processed_dir(log_file, root)



# ------------------------------
# Command-line interface
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix SDS archive: band codes, paths, midnight splits.")
    parser.add_argument("src_dir", help="Path to original SDS archive")
    parser.add_argument("--dest_dir", help="Optional new archive path", default=None)
    parser.add_argument("--networks", nargs='*', default='*', help="Network(s) to include")
    parser.add_argument("--stations", nargs='*', default='*', help="Station(s) to include")
    parser.add_argument("--start_date", help="Start date (e.g., 2016-01-01)", default=None)
    parser.add_argument("--end_date", help="End date (e.g., 2016-12-31)", default=None)
    parser.add_argument("--write", action='store_true', help="Write corrected files")
    parser.add_argument("--log_file", default="fix_sds_archive.log", help="Log file for resuming")

    args = parser.parse_args()
    fix_sds_archive(
        args.src_dir,
        dest_dir=args.dest_dir,
        networks=args.networks,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        write=args.write,
        log_file=args.log_file
    )