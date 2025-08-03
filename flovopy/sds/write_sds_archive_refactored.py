import os
import glob
import multiprocessing as mp
import pandas as pd
from obspy import UTCDateTime
from flovopy.sds.sds import SDSobj
from flovopy.core.trace_utils import fix_id_wrapper
from flovopy.core.miniseed_io import read_mseed
import traceback
import sqlite3
import time
import gc
from flovopy.core.computer_health import (
    get_cpu_temperature,
    pause_if_too_hot,
    log_cpu_temperature_to_csv,
    start_cpu_logger,
    log_memory_usage
)
from flovopy.sds.sds_utils import (
    try_lock_output_file,
    release_output_file_lock_safe,
    release_input_file_lock,
    remove_stale_locks,
    remove_empty_dirs,
    sqlite_to_excel,
    setup_database, 
    populate_file_log, 
    get_pending_file_list
)

def write_sds_archive(
    src_dir,
    dest_dir,
    networks='*',
    stations='*',
    start_date=None,
    end_date=None,
    metadata_excel_path=None,
    use_sds_structure=True,
    custom_file_list=None,
    recursive=True,
    file_glob="*.mseed",
    n_processes=1,
    debug=False,
    merge_strategy='obspy'
):
    try:
        if os.path.abspath(src_dir) == os.path.abspath(dest_dir):
            raise ValueError("Source and destination directories must be different.")

        networks = [networks] if isinstance(networks, str) else networks
        stations = [stations] if isinstance(stations, str) else stations
        start_date = UTCDateTime(start_date) if isinstance(start_date, str) else start_date
        end_date = UTCDateTime(end_date) if isinstance(end_date, str) else end_date

        os.makedirs(dest_dir, exist_ok=True)
        db_path = os.path.join(dest_dir, "processing_log.sqlite")

        if os.path.exists(db_path):
            print(f"{UTCDateTime()}: 📂 Resuming from existing database: {db_path}")
            file_list = get_pending_file_list(db_path)
            if not file_list:
                print(f"{UTCDateTime()}: ✅ No pending files left to process.")
                return
        else:
            setup_database(db_path, mode="write")
            if use_sds_structure:
                sdsin = SDSobj(src_dir)
                filterdict = {}
                if networks:
                    filterdict['networks'] = networks
                if stations:
                    filterdict['stations'] = stations
                file_list, non_sds_list = sdsin.build_file_list(
                    parameters=filterdict,
                    starttime=start_date,
                    endtime=end_date,
                    return_failed_list_too=True
                )
                pd.DataFrame(non_sds_list, columns=['file']).to_csv(os.path.join(dest_dir, 'non_sds_file_list.csv'), index=False)
            elif custom_file_list:
                file_list = custom_file_list
            else:
                pattern = os.path.join(src_dir, "**", file_glob) if recursive else os.path.join(src_dir, file_glob)
                file_list = sorted(glob.glob(pattern, recursive=recursive))

            if not file_list:
                print(f"{UTCDateTime()}: No MiniSEED files found to process.")
                return

            populate_file_log(file_list, db_path)
            pd.DataFrame(file_list, columns=['file']).to_csv(os.path.join(dest_dir, 'original_file_list.csv'), index=False)

        start_cpu_logger(interval_sec=60, log_path=os.path.join(dest_dir, "cpu_temperature_log.csv"))

        chunk_size = len(file_list) // n_processes + (len(file_list) % n_processes > 0)
        file_chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

        args = [
            (chunk, dest_dir, networks, stations, start_date, end_date, db_path, str(i), metadata_excel_path, debug, merge_strategy)
            for i, chunk in enumerate(file_chunks)
        ]

        with mp.Pool(processes=n_processes) as pool:
            pool.starmap(process_partial_file_list_db, args)

    except Exception as e:
        traceback.print_exc()
    finally:
        remove_empty_dirs(dest_dir)
        sqlite_to_excel(db_path, db_path.replace('.sqlite', '.xlsx'))

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM file_log WHERE status = 'pending'")
                pending_count = cursor.fetchone()[0]

            if pending_count == 0:
                print(f"{UTCDateTime()}: ✅ All files processed and logged in SQLite database: {db_path}", flush=True)
                print("OK", flush=True)
            else:
                print(f"{UTCDateTime()}: ⚠️ {pending_count} files remain unprocessed. Check logs or rerun script to resume.", flush=True)

        except Exception as e:
            print(f"{UTCDateTime()}: ❌ Could not verify processing completion: {e}", flush=True)

        gc.collect()


def process_partial_file_list_db(file_list, sds_output_dir, networks, stations, start_date, 
                                 end_date, db_path, cpu_id, metadata_excel_path, 
                                 debug, merge_strategy):
    print(f"{UTCDateTime()}: ✅ Started {cpu_id} with {len(file_list)} files")
    os.makedirs(sds_output_dir, exist_ok=True)
    sdsout = SDSobj(sds_output_dir)
    unmatcheddir = os.path.join(sds_output_dir, 'unmatched')
    sdsunmatched = SDSobj(unmatcheddir)


    if metadata_excel_path:
        ext = os.path.splitext(metadata_excel_path)[1].lower()
        if ext in ['.xls', '.xlsx', '.csv']:
            sdsout.load_metadata_from_excel(metadata_excel_path)
        elif ext in ['.xml']:
            sdsout.load_metadata_from_stationxml(metadata_excel_path)
        else:
            raise ValueError(f"Unsupported metadata file extension: {ext}")

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    start_time = UTCDateTime()
    total_files = len(file_list)

    for filenum, file_path in enumerate(file_list):
        try:
            pause_if_too_hot(threshold=75.0)

            cursor.execute("""
                INSERT OR IGNORE INTO locks (filepath, locked_by, locked_at)
                VALUES (?, ?, datetime('now'))
            """, (file_path, cpu_id))
            conn.commit()

            cursor.execute("SELECT locked_by FROM locks WHERE filepath = ?", (file_path,))
            row = cursor.fetchone()
            if not row or row[0] != cpu_id:
                release_input_file_lock(cursor, conn, file_path)
                continue

            try:
                st_in = read_mseed(file_path)
                log_memory_usage(f"[{cpu_id}] After read_mseed: {file_path}")
            except Exception as e:
                cursor.execute("""
                    UPDATE file_log
                    SET status = 'failed', reason = ?, ntraces_in = 0, ntraces_out = 0, cpu_id = ?, timestamp = datetime('now')
                    WHERE filepath = ?
                """, (f'read_mseed error: {str(e)}', cpu_id, file_path))
                conn.commit()
                continue

            ntraces_in = len(st_in)
            ntraces_out = 0
            nmerged = 0

            for tr in st_in:
                if (start_date and tr.stats.endtime < start_date) or (end_date and tr.stats.starttime > end_date):
                    status = 'skipped'
                    reason = 'Outside time range'
                    outputfile = None
                elif tr.stats.sampling_rate < 50:
                    status = 'skipped'
                    reason = 'Low sample rate'
                    outputfile = None
                else:
                    source_id, fixed_id = fix_id_wrapper(tr)
                    metadata_matched = sdsout.match_metadata(tr) if sdsout.metadata is not None else True

                    unmatched = False
                    whichsdsobj = sdsout
                    if not metadata_matched:
                        unmatched = True
                        whichsdsobj = sdsunmatched
                    full_dest_path = whichsdsobj.get_fullpath(tr)
                    output_locked = False
                    try:
                        output_locked = try_lock_output_file(conn, full_dest_path, cpu_id)
                    except Exception as e:
                        print(f"{UTCDateTime()}: ⚠️ Output lock attempt failed for {full_dest_path}: {e}", flush=True)

                    if output_locked:
                        whichsdsobj.stream.traces = [tr]
                        try:
                            results = whichsdsobj.write(debug=debug, merge_strategy=merge_strategy)
                            res = results.get(tr.id, {})
                            status = res.get('status', 'failed')
                            reason = res.get('reason', 'Unknown write error')
                            outputfile = res.get('path', None) if status == 'ok' else None
                            if status == 'ok':
                                ntraces_out += 1
                                if "Merged" in reason:
                                    nmerged += 1
                        except Exception as e:
                            status = 'failed'
                            reason = f"Write exception: {str(e)}"
                            outputfile = None
                        finally:
                            whichsdsobj.stream.clear()
                            release_output_file_lock_safe(conn, full_dest_path)
                    else:
                        status = 'skipped'
                        reason = 'SDS output file locked by another worker'
                        outputfile = None

                    if unmatched:
                        status = 'unmatched ' + status
                    cursor.execute("""
                        INSERT OR REPLACE INTO trace_log
                        (source_id, fixed_id, trace_id, filepath, station, sampling_rate, starttime, endtime, reason, outputfile, status, cpu_id, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """, (
                        source_id, fixed_id, tr.id, file_path, tr.stats.station, tr.stats.sampling_rate,
                        tr.stats.starttime.isoformat(), tr.stats.endtime.isoformat(),
                        reason, outputfile, status, cpu_id
                    ))
                    conn.commit()
                    tr.stats = None
                    del tr

            st_in.clear()
            gc.collect()

            if ntraces_out == ntraces_in:
                file_status = 'done'
                file_reason = None
            elif ntraces_out > 0:
                file_status = 'incomplete'
                file_reason = f"{ntraces_out} of {ntraces_in} written"
                if nmerged:
                    file_reason += f"; {nmerged} merged"
            else:
                file_status = 'failed'
                file_reason = 'All traces failed or skipped'

            cursor.execute("""
                UPDATE file_log
                SET status = ?, reason = ?, ntraces_in = ?, ntraces_out = ?, cpu_id = ?, timestamp = datetime('now')
                WHERE filepath = ?
            """, (file_status, file_reason, ntraces_in, ntraces_out, cpu_id, file_path))
            conn.commit()

        except Exception as e:
            print(f"{UTCDateTime()}: ❌ Worker {cpu_id} crashed on {file_path}: {e}", flush=True)
            traceback.print_exc()
        finally:
            release_input_file_lock(cursor, conn, file_path)

        try:
            if filenum % 10 == 0:
                processed_count = filenum + 1
                elapsed = UTCDateTime() - start_time
                if processed_count > 0:
                    est_total_time = elapsed / processed_count * total_files
                    est_remaining = est_total_time - elapsed
                    est_finish = UTCDateTime() + est_remaining
                    print(f"{UTCDateTime()}: 📊 [{cpu_id}] Progress: {processed_count}/{total_files} files processed, ETA: {est_finish.strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
                remove_stale_locks(cursor, conn, max_age_minutes=2)
        except Exception as e:
            print(f"{UTCDateTime()}: ❌ Worker {cpu_id} crashed on progress logging: {e}", flush=True)
            traceback.print_exc()

        gc.collect()

    conn.close()
    gc.collect()
    log_memory_usage(f"{UTCDateTime()}: [{cpu_id}] Finished all files")
    print(f"{UTCDateTime()}: ✅ Finished {cpu_id}", flush=True)