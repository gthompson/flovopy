#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from obspy import read, UTCDateTime, Stream
from flovopy.sds.sds2 import SDSobj #, restore_gaps, ensure_float32
from flovopy.core.preprocessing import fix_trace_id
import traceback
import gc
import tracemalloc
import numpy as np
from flovopy.core.merge import read_mseed_with_gap_masking

tracemalloc.start()
# Initialize an empty DataFrame to track skipped files
files_not_processed_df = pd.DataFrame(columns=["file", "reason"])

def write_sds_archive(
    src_dir,
    dest_dir,
    networks='*',
    stations='*',
    start_date=None,
    end_date=None,
    write=False,
    log_file=None,
    metadata_excel_path=None,
    csv_log_path="fix_sds_archive_log.csv",
    use_sds_structure=True,
    custom_file_list=None
):
    """
    Processes and reorganizes seismic waveform data from an SDS (SeisComP Data Structure) or arbitrary file list.

    This function reads MiniSEED files, filters by network and station, and verifies that each file contains 
    only one trace spanning no more than 24 hours. It trims overflow data, merges valid traces, filters by 
    metadata (e.g., start time, sample rate), optionally writes the output to a destination SDS archive, 
    and logs all actions.

    Parameters:
    -----------
    src_dir : str
        Source directory containing waveform files in SDS or another structure.
    dest_dir : str
        Destination directory where the cleaned SDS archive will be written.
    networks : str or list of str, default='*'
        Network codes to include. Use '*' for all networks.
    stations : str or list of str, default='*'
        Station codes to include. Use '*' for all stations.
    start_date : str or UTCDateTime, optional
        If provided, traces ending before this date will be excluded.
    end_date : str or UTCDateTime, optional
        If provided, traces starting after this date will be excluded.
    write : bool, default=False
        Whether to write the processed traces to the destination SDS archive.
    log_file : str, optional
        Path to a plain-text log file recording processed directories. If None, defaults to a log in dest_dir.
    metadata_excel_path : str, optional
        Path to Excel file containing metadata to assist in writing output SDS files.
    csv_log_path : str, default="fix_sds_archive_log.csv"
        Path to CSV file summarizing all traces processed, including destination paths and metadata.
    use_sds_structure : bool, default=True
        If True, treats src_dir as an SDS archive. If False, processes files from custom_file_list.
    custom_file_list : list of str, optional
        List of full paths to waveform files to process. Only used if use_sds_structure is False.
    """


    if os.path.abspath(src_dir) == os.path.abspath(dest_dir):
        confirm_same_directory()

    networks = [networks] if isinstance(networks, str) else networks
    stations = [stations] if isinstance(stations, str) else stations
    start_date = UTCDateTime(start_date) if start_date else None
    end_date = UTCDateTime(end_date) if end_date else None

    log_file = log_file or os.path.join(dest_dir, 'fix_sds_archive.log')
    sdsin = SDSobj(src_dir)
    sdsout = SDSobj(dest_dir)
    sdsunmatched = SDSobj(os.path.join(dest_dir, 'unmatched'))

    if metadata_excel_path:
        sdsout.load_metadata_from_excel(metadata_excel_path)

    processed_dirs = get_processed_dirs(log_file)
    csv_rows = []
    output_path_set = {}

    file_list = []
    if use_sds_structure:
        for root, dirs, files in os.walk(src_dir, topdown=True):
            dirs.sort()
            files.sort()
            if root in processed_dirs:
                print(f"Skipping already processed: {root}")
                continue
            for filename in files:
                file_list.append((root, filename))
            save_processed_dir(log_file, root)
    else:
        file_list = [(os.path.dirname(f), os.path.basename(f)) for f in custom_file_list]

    for root, filename in file_list:
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1024:.1f} KiB; Peak: {peak / 1024:.1f} KiB")
        file_path = os.path.join(root, filename)

        if use_sds_structure and not is_valid_sds_filename(filename):
            add_to_files_not_processed(file_path, reason="Invalid SDS filename")
            return

        print(f'Processing {file_path}')

        # Try reading the file with ObsPy
        st=Stream()
        try:
            #st = read(file_path, format='MSEED')
            st = read_mseed_with_gap_masking(file_path, fill_value=0.0, zero_gap_threshold=100, split_on_mask=False)
        except Exception:
            print(f'✘ Not a valid MiniSEED file that ObsPy can read: {file_path}')
            add_to_files_not_processed(file_path, reason="Unreadable by ObsPy")
            continue

        if len(st)==0:
            add_to_files_not_processed(file_path, reason="No data in Stream")

        # Extract metadata from filename if needed
        tr = st[0]
        net = tr.stats.network
        sta = tr.stats.station
        loc = tr.stats.location
        chan = tr.stats.channel

        if not net and use_sds_structure:
            try:
                parts = filename.split('.')
                net, sta, loc, chan, _, yyyy, jjj = parts[0:7]
            except Exception:
                add_to_files_not_processed(file_path, reason="Filename parsing failed")
                continue

        # Network and station filtering
        if networks != ['*'] and net not in networks:
            add_to_files_not_processed(file_path, reason=f"Filtered by network {net}")
            continue
        if stations != ['*'] and sta not in stations:
            add_to_files_not_processed(file_path, reason=f"Filtered by station {sta}")
            continue

        # Per-trace filtering
        stream2 = Stream()
        for tr in st:
            if (start_date and tr.stats.endtime < start_date) or \
            (end_date and tr.stats.starttime > end_date):
                add_to_files_not_processed(file_path, reason=f"Outside of time range {start_date} to {end_date}")
                continue
            if tr.stats.sampling_rate < 50.0:
                add_to_files_not_processed(file_path, reason=f"Sampling rate below 50 Hz")
                continue
            if tr.stats.station == 'LLS02':
                add_to_files_not_processed(file_path, reason=f"Station LLS02 ignored")
                continue
            stream2.append(tr)

        if not stream2:
            add_to_files_not_processed(file_path, reason="No valid traces after filtering")
            return
        
        try:
            for tr in stream2:
                print(f'- Processing {tr}')

                restore_gaps(tr, fill_value=0.0)

                # Ensure float dtype to avoid merge fill errors
                ensure_float32(tr)

                fix_trace_id(tr)
                
                metadata_matched = True # default
                if sdsout.metadata is not None:
                    metadata_matched = sdsout.match_metadata(tr)
                    print(f'- Corrected to {tr}')

                full_dest_path = sdsout.get_fullpath(tr)
                print(f'- SDS out path = {full_dest_path}')
                if not metadata_matched:
                    sdsunmatched.stream = Stream(traces=[tr])
                    sdsunmatched.write(overwrite=False)
                    continue

                if full_dest_path in output_path_set:
                    output_path_set[full_dest_path] += 1
                else:
                    output_path_set[full_dest_path] = 1

                csv_rows.append({
                    "source_path": file_path,
                    "dest_path": full_dest_path,
                    "trace_id": tr.id,
                    "starttime": tr.stats.starttime.isoformat(),
                    "endtime": tr.stats.endtime.isoformat(),
                    "sampling_rate": tr.stats.sampling_rate,
                    "npts": tr.stats.npts
                })

                if write:
                    sdsout.stream = Stream(traces=[tr])
                    try:
                        success = sdsout.write(overwrite=False)
                    except Exception as e:
                        traceback.print_exc()
                        print(f"✘ Failed to write: {tr.id} → {full_dest_path}")
                        raise e
                    else:
                        if success:
                            print(f"✔ Wrote: {tr.id} → {full_dest_path}")
                        else:
                            print(f"✘ Failed to write: {tr.id} → {full_dest_path}")
        except Exception as e:
            print(f"Unhandled error: {e}")
            traceback.print_exc()
        finally:
            save_skipped_files_to_csv(os.path.join(dest_dir, "skipped_files.csv"))


    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_log_path, index=False)
    print(f"🔍 Log CSV saved to {csv_log_path}")

    collisions = {k: v for k, v in output_path_set.items() if v > 1}
    if collisions:
        print("⚠️  Detected potential filename collisions:")
        for path, count in collisions.items():
            print(f"  - {path} ({count} traces)")
        print("Review collisions in the CSV log before running with --write.")

    tracemalloc.stop()

def is_valid_sds_filename(filename):
    parts = filename.split('.')
    return len(parts) >= 7 and parts[4] == 'D'

def save_processed_dir(log_file, directory):
    with open(log_file, 'a') as f:
        f.write(f"{directory}\n")

def get_processed_dirs(log_file):
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f.readlines())

def confirm_same_directory():
    msg = (
        "\n⚠️  SAME SOURCE AND DESTINATION SDS ARCHIVE NOT RECOMMENDED.\n"
        "This may overwrite or destroy original data. Proceed only if you're absolutely sure.\n"
        "Type 'YES' and hit ENTER to proceed: "
    )
    response = input(msg)
    if response.strip() != "YES":
        print("Aborting.")
        exit(1)

'''
def fix_sds_archive(
    src_dir,
    dest_dir,
    networks='*',
    stations='*',
    start_date=None,
    end_date=None,
    write=False,
    log_file=None,
    metadata_excel_path=None,
    csv_log_path="fix_sds_archive_log.csv"
):
    if os.path.abspath(src_dir) == os.path.abspath(dest_dir):
        confirm_same_directory()

    networks = [networks] if isinstance(networks, str) else networks
    stations = [stations] if isinstance(stations, str) else stations
    start_date = UTCDateTime(start_date) if start_date else None
    end_date = UTCDateTime(end_date) if end_date else None

    log_file = log_file or os.path.join(dest_dir, 'fix_sds_archive.log')
    sdsin = SDSobj(src_dir)
    sdsout = SDSobj(dest_dir)

    if metadata_excel_path:
        sdsout.load_metadata_from_excel(metadata_excel_path)

    processed_dirs = get_processed_dirs(log_file)
    csv_rows = []
    output_path_set = {}

    for root, dirs, files in os.walk(src_dir, topdown=True):
        dirs.sort()
        files.sort()

        if root in processed_dirs:
            print(f"Skipping already processed: {root}")
            continue

        for filename in files:
            gc.collect()

            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {current / 1024:.1f} KiB; Peak: {peak / 1024:.1f} KiB")

            if not is_valid_sds_filename(filename):
                continue
            print(f'Processing {filename}')

            parts = filename.split('.')
            net, sta, loc, chan, _, yyyy, jjj = parts

            if networks != ['*'] and net not in networks:
                continue
            if stations != ['*'] and sta not in stations:
                continue

            try:
                file_path = os.path.join(root, filename)

                # check if by mistake, the file has more than 24 hours of data
                st_test = read(file_path).merge(fill_value=0, method=1)
                
                if len(st_test)!=1:
                    raise IOError(f'expected 1 trace at {file_path} {st_test}')
                tr_test = st_test[0]
                trace_id = tr_test.id


                #trace_id = f'{net}.{sta}.{loc}.{chan}'  # write helper if needed

                # Compute the expected start and end time from SDS naming
                startt = UTCDateTime(f"{yyyy}-{jjj}")
                endt = startt + 86400  # 1 day later
                if tr_test.stats.starttime < startt - 1:
                    tr_extra_start = tr_test.copy().trim(endtime=startt)
                    sdsout.stream = tr_extra_start
                    sdsout.write()
                if tr_test.stats.endtime > endt + 1:
                    tr_extra_end = tr_test.copy().trim(starttime=endt)
                    sdsout.stream = tr_extra_end
                    sdsout.write() 
                del st_test, tr_test                  
                

                sdsin.read(startt, endt, trace_ids=[trace_id], fill_value=0.0, \
                   skip_low_rate_channels=True, speed=1, verbose=True, merge_method=0, \
                    progress=False,  detect_zero_padding=True)
                stream = sdsin.stream

                stream2 = Stream()
                for tr in stream:
                    if start_date and tr.stats.endtime < start_date:
                        continue
                    if end_date and tr.stats.starttime > end_date:
                        continue
                    if tr.stats.sampling_rate < 50.0 or tr.stats.station == 'LLS02':
                        continue

                    fix_trace_id(tr)
                    stream2.append(tr)
                del stream

                try:
                    stream2.merge(fill_value=None, method=0)
                except Exception:
                    pass


                for tr in stream2:
                    if sdsout.metadata is not None:
                        sdsout.match_metadata(tr)

                    full_dest_path = sdsout.get_fullpath(tr)

                    # Collision detection
                    if full_dest_path in output_path_set:
                        output_path_set[full_dest_path] += 1
                    else:
                        output_path_set[full_dest_path] = 1

                    # Log the trace metadata
                    csv_rows.append({
                        "source_path": file_path,
                        "dest_path": full_dest_path,
                        "trace_id": tr.id,
                        "starttime": tr.stats.starttime.isoformat(),
                        "endtime": tr.stats.endtime.isoformat(),
                        "sampling_rate": tr.stats.sampling_rate,
                        "npts": tr.stats.npts
                    })

                    if write:
                        sdsout.stream = Stream(traces=[tr])
                        try:
                            success = sdsout.write(overwrite=False)
                        except Exception as e:
                            traceback.print_exc()
                            print(f"✘ Failed to write: {tr.id} → {full_dest_path}")
                            raise e
                        else:
                            if success:
                                print(f"✔ Wrote: {tr.id} → {full_dest_path}")
                            else:
                                print(f"✘ Failed to write: {tr.id} → {full_dest_path}")

            except Exception as e:
                print(f"✘ Error processing {filename}: {e}")
                traceback.print_exc()
                raise e

        save_processed_dir(log_file, root)

    # Save log CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_log_path, index=False)
    print(f"🔍 Log CSV saved to {csv_log_path}")

    # Check for filename collisions
    collisions = {k: v for k, v in output_path_set.items() if v > 1}
    if collisions:
        print("⚠️  Detected potential filename collisions:")
        for path, count in collisions.items():
            print(f"  - {path} ({count} traces)")
        print("Review collisions in the CSV log before running with --write.")

    tracemalloc.stop()
'''

def check_for_collisions(log_csv_path):
    # Load the log CSV file
    df = pd.read_csv(log_csv_path)

    # Count how many times each destination file appears
    dest_counts = df['destination_filepath'].value_counts()

    # Filter for collisions (i.e., more than one source per destination)
    collisions = dest_counts[dest_counts > 1]

    if not collisions.empty:
        print("⚠️ Filename collisions detected:")
        for dest_file in collisions.index:
            print(f"\nDestination: {dest_file}")
            sources = df[df['destination_filepath'] == dest_file]['source_filepath'].tolist()
            for src in sources:
                print(f"  ↳ {src}")
    else:
        print("✅ No filename collisions detected. Safe to proceed.")



def add_to_files_not_processed(file_path, reason=None):
    """
    Add a skipped file to the global DataFrame, with an optional reason.

    Parameters
    ----------
    file_path : str
        Full path to the file that could not be processed.
    reason : str, optional
        Explanation for why the file was skipped.
    """
    global files_not_processed_df

    new_entry = pd.DataFrame([{"file": file_path, "reason": reason}])
    if not ((files_not_processed_df["file"] == file_path) & (files_not_processed_df["reason"] == reason)).any():
        files_not_processed_df = pd.concat([files_not_processed_df, new_entry], ignore_index=True)
        print(f"⚠️ Skipped: {file_path} ({reason})" if reason else f"⚠️ Skipped: {file_path}")

def save_skipped_files_to_csv(csv_path="skipped_files.csv"):
    """
    Save the skipped files DataFrame to a CSV file.

    Parameters
    ----------
    csv_path : str
        Destination path for the CSV output.
    """
    if files_not_processed_df.empty:
        print("No skipped files to save.")
        return

    files_not_processed_df.to_csv(csv_path, index=False)
    print(f"📝 Skipped files written to {csv_path}")

# ------------------------------
# Command-line interface
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix SDS archive: band codes, paths, midnight splits.")
    parser.add_argument("src_dir", help="Path to original SDS archive")
    parser.add_argument("dest_dir", help="Path to new fixed SDS archive")
    parser.add_argument("--networks", nargs='*', default='*', help="Network(s) to include")
    parser.add_argument("--stations", nargs='*', default='*', help="Station(s) to include")
    parser.add_argument("--start_date", help="Start date (e.g., 2016-01-01)", default=None)
    parser.add_argument("--end_date", help="End date (e.g., 2016-12-31)", default=None)
    parser.add_argument("--write", action='store_true', help="Write corrected files")
    parser.add_argument("--log_file", default=None, help="Text log file for resuming processing")
    parser.add_argument("--metadata_excel_path", help="Path to Excel file with metadata", default=None)
    parser.add_argument("--csv_log_path", default="fix_sds_archive_log.csv", help="CSV log file path")

    args = parser.parse_args()

    fix_sds_archive(
        src_dir=args.src_dir,
        dest_dir=args.dest_dir,
        networks=args.networks,
        stations=args.stations,
        start_date=args.start_date,
        end_date=args.end_date,
        write=args.write,
        log_file=args.log_file,
        metadata_excel_path=args.metadata_excel_path,
        csv_log_path=args.csv_log_path
    )
