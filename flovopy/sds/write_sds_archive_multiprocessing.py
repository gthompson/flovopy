import os
import glob
import shutil
import multiprocessing as mp
from flovopy.core.miniseed_io import smart_merge
import pandas as pd
from obspy import Stream, UTCDateTime
from flovopy.sds.sds import SDSobj, parse_sds_filename, merge_multiple_sds_archives #is_valid_sds_dir, is_valid_sds_filename
from flovopy.core.preprocessing import fix_trace_id
from flovopy.core.miniseed_io import read_mseed, write_mseed
import traceback

def write_sds_archive(
    src_dir,
    dest_dir,
    networks='*',
    stations='*',
    start_date=None,
    end_date=None,    
    metadata_excel_path=None,
    #log_file=None,
    use_sds_structure=True,
    custom_file_list=None,
    recursive=True,
    file_glob="*.mseed",
    n_processes=1,
    debug=False,
    try_one_more_time=False,
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
        raise ValueError("Source and destination directories must be different.")
    
    networks = [networks] if isinstance(networks, str) else networks
    stations = [stations] if isinstance(stations, str) else stations
    start_date = UTCDateTime(start_date) if isinstance(start_date, str) else None
    end_date = UTCDateTime(end_date) if isinstance(end_date, str) else None 

    # Ability to recover from list of files that were not finished in a previous run of this code
    os.makedirs(dest_dir, exist_ok=True)
    unprocessed_list_of_files_csv = os.path.join(dest_dir, 'unprocessed_file_list.csv')
    if os.path.isfile(unprocessed_list_of_files_csv):
        df = pd.read_csv(unprocessed_list_of_files_csv)
        file_list = df['file'].to_list() 
    else:  
        # Otherwise Build original file list
        if use_sds_structure:
            sdsin = SDSobj(src_dir)
            filterdict = {}
            if networks:
                filterdict['networks']=networks
            if stations:
                filterdict['stations']=stations
            file_list, non_sds_list = sdsin.build_file_list(parameters=filterdict, starttime=start_date, endtime=end_date, return_failed_list_too=True)
            pd.DataFrame(non_sds_list, columns=['file']).to_csv(os.path.join(dest_dir, 'non_sds_file_list.csv'), index=False)
        elif custom_file_list:
            file_list = custom_file_list
        else:
            pattern = os.path.join(src_dir, "**", file_glob) if recursive else os.path.join(src_dir, file_glob)
            file_list = sorted(glob.glob(pattern, recursive=recursive))
        del sdsin

        original_list_of_files_csv = os.path.join(dest_dir, 'original_file_list.csv')
        if not os.path.isfile(original_list_of_files_csv):
            pd.DataFrame(file_list, columns=['file']).to_csv(original_list_of_files_csv, index=False)

    if not file_list:
        print("No list of MiniSEED files found.")
        return
    
    # Split file list among workers
    chunk_size = len(file_list) // n_processes + (len(file_list) % n_processes > 0)
    file_chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    temp_dirs = [os.path.join(dest_dir, f"temp_sds_{i}") for i in range(len(file_chunks))]
    args = [(chunk, temp_dirs[i], networks, stations, start_date, end_date, debug, metadata_excel_path) for i, chunk in enumerate(file_chunks)]
    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(process_partial_file_list, args)
    file_list_remaining = []
    for r in results:
        if isinstance(r, list):
            file_list_remaining.extend(r)
        else:
            print("⚠️ Received non-list result from a worker process, skipping.")
    pd.DataFrame(file_list_remaining, columns=['file']).to_csv(unprocessed_list_of_files_csv, index=False)

    if try_one_more_time:
        # try one more time to process the unprocessed files
        temp_dest = os.path.join(dest_dir, 'dummy')
        
        for sub_dir in temp_dirs:
            unprocessed_dir = os.path.join(sub_dir, 'unprocessed')
            custom_file_list = [
                os.path.join(unprocessed_dir, f)
                for f in os.listdir(unprocessed_dir)
                if os.path.isfile(os.path.join(unprocessed_dir, f))
            ]
            process_partial_file_list(custom_file_list, temp_dest, networks, stations, start_date, end_date, debug, metadata_excel_path) 
        
        temp_dirs.append(temp_dest)
    merge_sds_archives(temp_dirs, dest_dir)
    print(f"Temp dirs being merged: {temp_dirs}")

    #return file_list_remaining


# --- PROCESSING FUNCTION ---
def process_partial_file_list(file_list, temp_dest, networks, stations, start_date, end_date, debug, metadata_excel_path):
    os.makedirs(temp_dest, exist_ok=True)
    sdsout_local = SDSobj(temp_dest)
    if metadata_excel_path:
        sdsout_local.load_metadata_from_excel(metadata_excel_path)
    unmatcheddir = os.path.join(temp_dest, 'unmatched')
    sdsunmatched = SDSobj(unmatcheddir)

    file_list_remaining = file_list.copy()
    pd.DataFrame(file_list, columns=['file']).to_csv(os.path.join(temp_dest, 'original_file_list.csv'), index=False)

    file_logging = []
    trace_logging = []
    unprocessed_dir = os.path.join(temp_dest, 'unprocessed')
    os.makedirs(unprocessed_dir, exist_ok=True)

    def log_file_issue(file_path, reason, n_in, n_out, status='failed'):
        entry = {"file": file_path, "status": status, "reason": reason, "ntraces_in":n_in, "ntraces_out":n_out, "CPU":temp_dest[-1]}
        if entry not in file_logging:
            file_logging.append(entry)
            if status=='failed':
                print(f"⚠️ File skipped: {file_path} ({reason})")
        shutil.copy2(file_path, unprocessed_dir)
        

    def log_trace_issue(trace, file_path, reason, original_id=None, fixed_id=None, outputfile=None, status='failed'):
        entry = {
            "inputfile": file_path,
            "trace_id": trace.id if trace else None,
            "station": trace.stats.station if trace else None,
            "sampling_rate": trace.stats.sampling_rate if trace else None,
            "starttime": trace.stats.starttime.isoformat() if trace else None,
            "endtime": trace.stats.endtime.isoformat() if trace else None,
            "reason": reason,
            "original_id": original_id,
            "fixed_id": fixed_id,
            "outputfile": outputfile,
            "status": status,
            "CPU":temp_dest[-1]
        }
        trace_logging.append(entry)
        if status=='failed':
            print(f"⚠️ Trace skipped in {file_path}: {reason} → {trace.id if trace else 'unknown trace'}")
        
            try:
                outfile=os.path.join(unprocessed_dir,f"{trace.id}_{trace.stats.starttime.strftime('%Y-%m-%dT%H:%M:%S')}_{trace.stats.endtime.strftime('%Y-%m-%dT%H:%M:%S')}.mseed")
                success = write_mseed(trace, outfile, fill_value=0.0, overwrite_ok=False)
                if not success:
                    shutil.copy2(file_path, unprocessed_dir)
            except:
                shutil.copy2(file_path, unprocessed_dir)

    try:
        for file_path in file_list:
            filename = os.path.basename(file_path)

            try:
                st_in = read_mseed(file_path)
            except Exception as e:
                log_file_issue(file_path, f"read_mseed error: {str(e)}", 0, 0)
                continue
            number_traces_in = len(st_in)
            number_traces_out = 0

            if len(st_in) == 0:
                log_file_issue(file_path, "No data in Stream", number_traces_in, number_traces_out)
                continue

            if debug:
                print(f'- Source Stream={st_in}')

            try:
                net, sta, loc, chan = st_in[0].id.split('.')
            except Exception:
                log_file_issue(file_path, "Missing trace ID components", number_traces_in, number_traces_out)
                continue

            if not net:
                try:
                    net, sta, loc, chan, dtype, year, jday = parse_sds_filename(filename)
                except Exception:
                    log_file_issue(file_path, "Got no network and Filename parsing failed", number_traces_in, number_traces_out)
                    continue

            if networks != ['*'] and net not in networks:
                log_file_issue(file_path, f"Filtered by network {net}", number_traces_in, number_traces_out)
                continue
            if stations != ['*'] and sta not in stations:
                log_file_issue(file_path, f"Filtered by station {sta}", number_traces_in, number_traces_out)
                continue

            st_out = Stream()
            
            for tr in st_in:
                if (start_date and tr.stats.endtime < start_date) or \
                   (end_date and tr.stats.starttime > end_date):
                    log_trace_issue(tr, file_path, "Outside of time range")
                    continue
                if tr.stats.sampling_rate < 50.0:
                    log_trace_issue(tr, file_path, "Sampling rate below 50 Hz")
                    continue
                st_out.append(tr)

            if len(st_out) == 0:
                log_file_issue(file_path, "No valid traces after filtering", number_traces_in, number_traces_out)
                continue

            if debug:
                print(f'- Stream after filtering: {st_out}')

            all_traces_written = True
            
            for tr in st_out:
                source_id = tr.id
                print(f'- Processing {source_id}')

                fix_trace_id(tr)
                fixed_id = tr.id

                metadata_matched = True
                if sdsout_local.metadata is not None:
                    metadata_matched = sdsout_local.match_metadata(tr)
                    if debug:
                        print(f'- {source_id} → {fixed_id} → {tr.id}')
                elif debug:
                    print(f'- {source_id} → {fixed_id}')

                if not metadata_matched:
                    
                    try:
                        sdsunmatched.stream = Stream(traces=[tr])
                        trace_written = sdsunmatched.write()
                        if trace_written:
                            log_trace_issue(tr, file_path, "No matching metadata. Written to SDS unmatched", original_id=source_id, fixed_id=fixed_id)
                        else:
                            log_trace_issue(tr, file_path, "No matching metadata. Not written to SDS unmatched", original_id=source_id, fixed_id=fixed_id)
                    except Exception as e:
                        log_trace_issue(tr, file_path, "No matching metadata. Failed to write to SDS unmatched", original_id=source_id, fixed_id=fixed_id)
                    sdsunmatched.stream = Stream()
                    continue

                full_dest_path = sdsout_local.get_fullpath(tr)

                sdsout_local.stream = Stream(traces=[tr])
                try:
                    trace_written = sdsout_local.write(debug=debug)
                    number_traces_out += 1
                    sdsout_local.stream = Stream()
                except Exception as e:
                    log_trace_issue(tr, file_path, f"Write error: {str(e)}", original_id=source_id, fixed_id=fixed_id, outputfile=full_dest_path)
                    sdsout_local.stream = Stream()
                    continue

                all_traces_written = all_traces_written and trace_written

                if trace_written:
                    log_trace_issue(tr, file_path, None, original_id=source_id, fixed_id=fixed_id, outputfile=full_dest_path, status='done')
                    if debug:
                        print(f"  ✔ Wrote: {tr.id} → {full_dest_path}")
                    
                else:
                    log_trace_issue(tr, file_path, f"Failed to write: {tr.id} → {full_dest_path}", original_id=source_id, fixed_id=fixed_id, outputfile=full_dest_path)
                    if debug:
                        print(f"  ✘ Failed to write: {tr.id} → {full_dest_path}")

            if all_traces_written:
                log_file_issue(file_path, None, number_traces_in, number_traces_out, status='done')
                file_list_remaining.remove(file_path)
            else:
                log_file_issue(file_path, None, number_traces_in, number_traces_out, status='incomplete')

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()

    finally:
        pd.DataFrame(file_logging).to_csv(os.path.join(temp_dest, 'file_processing.csv'), index=False)
        pd.DataFrame(trace_logging).to_csv(os.path.join(temp_dest, 'trace_processing.csv'), index=False)
        pd.DataFrame(file_list_remaining, columns=['file']).to_csv(os.path.join(temp_dest, 'unprocessed_file_list.csv'), index=False)

    return file_list_remaining


'''
def merge_sds_archives(sub_dirs, final_dest):
    final_sds = SDSobj(final_dest)
    conflicts_resolved = 0
    conflicts_remaining = 0

    for sub_dir in sub_dirs:
        for root, _, files in os.walk(sub_dir):
            if is_valid_sds_dir(root):
                for file in files:
                    if is_valid_sds_filename(file):
                        temp_path = os.path.join(root, file)
                        rel_path = os.path.relpath(temp_path, sub_dir)
                        dest_path = os.path.join(final_dest, rel_path)

                        if os.path.exists(dest_path):
                            try:
                                st1 = read_mseed(dest_path)
                                st2 = read_mseed(os.path.join(root, file))
                                merged, report = smart_merge(st1+st2)
                                if report['status']=='ok':
                                    final_sds.write(merged)
                                    conflicts_resolved += 1
                                    os.remove(temp_path)
                                else:
                                    # we leave the temp_path file in place
                                    conflicts_remaining += 1
                                    pass

                            except Exception as e:
                                print(f"Error merging {rel_path}: {e}")
                        else:
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.move(temp_path, dest_path)

    print(f"Finished merging. Conflicts resolved/merged: {conflicts_resolved}. Conflicts remaining: {conflicts_remaining}")

    # merge CSV files too
    csv_filenames = ['file_processing.csv', 'trace_processing.csv', 'unprocessed_file_list.csv']
    for csv in csv_filenames:
        merged_df = pd.DataFrame()
        for sub_dir in sub_dirs:
            csv_path = os.path.join(sub_dir, csv)
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                os.remove(csv_path)
                if not 'CPU' in df.columns:
                    df["CPU"] = sub_dir[-1]  # Optional: add a column to track origin
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            else:
                print(f"⚠️ No CSV found in {sub_dir}")

        # Write out the merged DataFrame
        merged_df.to_csv(os.path.join(final_dest, csv), index=False)


    # merged unprocessed directories
    unprocessed_dir = os.path.join(final_dest, 'unprocessed')
    os.makedirs(unprocessed_dir, exist_ok=True)
    for sub_dir in sub_dirs:
        unprocessed_subdir = os.path.join(sub_dir, 'unprocessed')
        if os.path.isdir(unprocessed_subdir):
            for filename in os.listdir(unprocessed_subdir):
                source_path = os.path.join(unprocessed_subdir, filename)
                dest_path = os.path.join(unprocessed_dir, filename)
                # Only move if it's a file and doesn't exist at destination
                if os.path.isfile(source_path) and not os.path.exists(dest_path):
                    shutil.move(source_path, dest_path)
                    print(f"Moved: {filename}")
                else:
                    print(f"Skipped (exists or not a file): {filename}")            

    # merged obsolete directories
    obsolete_dir = os.path.join(final_dest, 'obsolete')
    os.makedirs(obsolete_dir, exist_ok=True)
    for sub_dir in sub_dirs:
        obsolete_subdir = os.path.join(sub_dir, 'obsolete')
        if os.path.isdir(obsolete_subdir):
            for filename in os.listdir(obsolete_subdir):
                source_path = os.path.join(obsolete_subdir, filename)
                dest_path = os.path.join(obsolete_dir, filename)
                # Only move if it's a file and doesn't exist at destination
                if os.path.isfile(source_path) and not os.path.exists(dest_path):
                    shutil.move(source_path, dest_path)
                    print(f"Moved: {filename}")
                else:
                    print(f"Skipped (exists or not a file): {filename}")   

    # remove empty directories
    for sub_dir in sub_dirs:
        for root, dirs, files in os.walk(sub_dir, topdown=False):
            for name in dirs:
                dir_path = os.path.join(root, name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
        if os.path.exists(sub_dir) and not os.listdir(sub_dir):
            os.rmdir(sub_dir)   
'''

def merge_sds_archives(sub_dirs, final_dest):
    """
    Merge multiple SDS archive directories into a final destination.
    Also merges CSV logs and auxiliary directories like 'unprocessed' and 'obsolete'.
    """

    # 1. Merge SDS waveform data
    merge_multiple_sds_archives(sub_dirs, final_dest)

    # 2. Merge standard CSV log files
    csv_filenames = ['file_processing.csv', 'trace_processing.csv', 'unprocessed_file_list.csv']
    for csv in csv_filenames:
        merged_df = pd.DataFrame()
        for sub_dir in sub_dirs:
            csv_path = os.path.join(sub_dir, csv)
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                os.remove(csv_path)
                if 'CPU' not in df.columns:
                    df["CPU"] = os.path.basename(sub_dir)  # use directory name as source label
                merged_df = pd.concat([merged_df, df], ignore_index=True)
            else:
                print(f"⚠️ No CSV found in {sub_dir}")
        if not merged_df.empty:
            merged_df.to_csv(os.path.join(final_dest, csv), index=False)

    # 3. Merge 'unprocessed' directories
    merge_named_directory(sub_dirs, final_dest, 'unprocessed')

    # 4. Merge 'obsolete' directories
    merge_named_directory(sub_dirs, final_dest, 'obsolete')

    # 5. Clean up any empty subdirectories
    for sub_dir in sub_dirs:
        for root, dirs, files in os.walk(sub_dir, topdown=False):
            for name in dirs:
                dir_path = os.path.join(root, name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
        if os.path.exists(sub_dir) and not os.listdir(sub_dir):
            os.rmdir(sub_dir)

    print(f"\n✅ All SDS archives, logs, and auxiliary directories merged into: {final_dest}")


def merge_named_directory(sub_dirs, final_dest, dirname):
    """
    Merge same-named subdirectories (e.g., 'unprocessed', 'obsolete') from each sub_dir into final_dest.
    """
    dest_dir = os.path.join(final_dest, dirname)
    os.makedirs(dest_dir, exist_ok=True)

    for sub_dir in sub_dirs:
        src_subdir = os.path.join(sub_dir, dirname)
        if os.path.isdir(src_subdir):
            for filename in os.listdir(src_subdir):
                src_path = os.path.join(src_subdir, filename)
                dest_path = os.path.join(dest_dir, filename)
                if os.path.isfile(src_path) and not os.path.exists(dest_path):
                    shutil.move(src_path, dest_path)
                    print(f"Moved {dirname}: {filename}")
                else:
                    print(f"Skipped {dirname} (exists or not a file): {filename}")