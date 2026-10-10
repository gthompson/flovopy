"""Restartable MiniSEED-to-SDS ingestion using EnhancedSDSClient.

Writes are deliberately serialized: EnhancedSDSClient(mode='merge') reads and
rewrites existing SDS day files, so independent writers must not race.
"""
from __future__ import annotations

import fnmatch
import glob
import os
from pathlib import Path
import sqlite3
import traceback
from typing import Optional

from obspy import UTCDateTime, read, read_inventory
from flovopy.enhanced.sdsclient import EnhancedSDSClient
from flovopy.sds.sds_utils import discover_files, setup_database, populate_file_log


def _matches(value, patterns):
    if patterns is None:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)


def _has_metadata(inventory, trace):
    if inventory is None:
        return True
    # Select by channel AND time, avoiding an incorrect match to another epoch.
    sel = inventory.select(network=trace.stats.network,
                           station=trace.stats.station,
                           location=trace.stats.location,
                           channel=trace.stats.channel,
                           time=trace.stats.starttime)
    return any(ch for net in sel for sta in net for ch in sta
               if (ch.start_date is None or ch.start_date <= trace.stats.starttime)
               and (ch.end_date is None or ch.end_date >= trace.stats.endtime))


def _record(conn, filepath, status, reason, n_in, n_out):
    conn.execute("""UPDATE file_log SET status=?, reason=?, ntraces_in=?,
        ntraces_out=?, cpu_id='main', timestamp=datetime('now') WHERE filepath=?""",
        (status, reason, n_in, n_out, filepath))
    conn.commit()


def _trace_record(conn, filepath, trace, status, reason, paths):
    # Existing trace_log schema uses (trace_id, filepath) as primary key.
    # Multiple segments with the same ID in one input file cannot be logged
    # separately without a schema migration; preserve the latest status.
    conn.execute("""INSERT OR REPLACE INTO trace_log
        (source_id, fixed_id, trace_id, filepath, station, sampling_rate,
         starttime, endtime, reason, outputfile, status, cpu_id, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,'main',datetime('now'))""",
        (trace.id, trace.id, trace.id, filepath, trace.stats.station,
         trace.stats.sampling_rate, str(trace.stats.starttime),
         str(trace.stats.endtime), reason,
         ';'.join(map(str, paths)) if paths else None, status))
    conn.commit()


def write_sds_archive(
    src_dir, dest_dir, networks='*', stations='*', start_date=None,
    end_date=None, metadata_excel_path=None, use_sds_structure=True,
    custom_file_list=None, recursive=True, file_glob='*.mseed',
    n_processes=1, debug=False, merge_strategy='obspy',
    cpu_temp=False, min_sampling_rate=None, write_mode='merge',
    retry_failed=True,
):
    """Ingest source waveforms into SDS, retaining restartable SQLite logging.

    `metadata_excel_path` is a legacy argument: only StationXML (.xml) is
    supported for metadata matching; spreadsheets require explicit conversion.
    Nonmatching channels are written to DEST/unmatched (not discarded).

    `n_processes` is accepted for compatibility, but writes are serialized.
    `merge_strategy` is forwarded only for merge mode.
    """
    source = Path(src_dir).expanduser().resolve()
    destination = Path(dest_dir).expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate, non-nested directories')
    if write_mode not in {'merge', 'fail', 'overwrite'}:
        raise ValueError('write_mode must be merge, fail or overwrite')
    if n_processes != 1:
        print('NOTE: n_processes is ignored; SDS writes are serialized to prevent collisions.')
    if cpu_temp:
        print('NOTE: CPU-temperature monitoring is not implemented in this writer; no monitoring enabled.')
    start = UTCDateTime(start_date) if start_date is not None else None
    end = UTCDateTime(end_date) if end_date is not None else None
    if start and end and end < start:
        raise ValueError('end_date precedes start_date')
    inventory = None
    if metadata_excel_path:
        if Path(metadata_excel_path).suffix.lower() != '.xml':
            raise ValueError('Metadata matching requires StationXML (.xml); convert Excel/CSV first')
        inventory = read_inventory(metadata_excel_path)
    destination.mkdir(parents=True, exist_ok=True)
    db_path = destination / 'processing_log.sqlite'
    setup_database(str(db_path), mode='write')

    if custom_file_list is not None:
        files = [str(Path(f).expanduser().resolve()) for f in custom_file_list]
    elif use_sds_structure:
        files = discover_files(str(source), use_sds=True,
            filterdict={'networks': networks, 'stations': stations},
            starttime=start, endtime=end)
        # Some older discover_files implementations return (files, rejected).
        if isinstance(files, tuple):
            files = files[0]
    else:
        pattern = str(source / ('**/' if recursive else '') / file_glob)
        files = sorted(glob.glob(pattern, recursive=recursive))
        files = [f for f in files if not any(p.startswith('.') for p in Path(f).relative_to(source).parts)]
    files = sorted(set(map(str, files)))
    populate_file_log(files, str(db_path))
    with sqlite3.connect(db_path) as conn:
        conn.execute('PRAGMA busy_timeout=30000')
        statuses = ('pending', 'incomplete', 'failed') if retry_failed else ('pending', 'incomplete')
        marks = ','.join('?' for _ in statuses)
        pending = [r[0] for r in conn.execute(
            f'SELECT filepath FROM file_log WHERE filepath IN ({",".join("?" for _ in files)}) AND status IN ({marks}) ORDER BY filepath',
            (*files, *statuses))] if files else []
        print(f'Ingesting {len(pending)} pending files of {len(files)} discovered')
        main_client = EnhancedSDSClient(destination)
        unmatched_client = EnhancedSDSClient(destination / 'unmatched') if inventory else None
        for index, filepath in enumerate(pending, 1):
            try:
                st = read(filepath)
                n_in = len(st)
                n_written = n_skipped = n_failed = 0
                for tr in st:
                    if not _matches(tr.stats.network, networks) or not _matches(tr.stats.station, stations):
                        n_skipped += 1
                        _trace_record(conn, filepath, tr, 'skipped', 'NSLC filter', [])
                        continue
                    if (start is not None and tr.stats.endtime < start) or (end is not None and tr.stats.starttime > end):
                        n_skipped += 1
                        _trace_record(conn, filepath, tr, 'skipped', 'Outside time range', [])
                        continue
                    if min_sampling_rate is not None and tr.stats.sampling_rate < min_sampling_rate:
                        n_skipped += 1
                        _trace_record(conn, filepath, tr, 'skipped', 'Below requested sample rate', [])
                        continue
                    tr = tr.copy()
                    if start is not None or end is not None:
                        tr.trim(starttime=start, endtime=end, nearest_sample=False)
                    if not tr.stats.npts:
                        n_skipped += 1
                        continue
                    matched = _has_metadata(inventory, tr)
                    client = main_client if matched else unmatched_client
                    try:
                        kwargs = {'merge_strategy': merge_strategy} if write_mode == 'merge' else {}
                        paths = client.write_trace(tr, mode=write_mode, preprocess=False,
                                                   verbose=debug, **kwargs)
                        if not paths:
                            raise IOError('Client returned no output paths')
                        n_written += 1
                        _trace_record(conn, filepath, tr, 'ok' if matched else 'unmatched ok', '', paths)
                    except Exception as exc:
                        n_failed += 1
                        _trace_record(conn, filepath, tr, 'failed', str(exc), [])
                if n_failed:
                    status = 'incomplete' if n_written else 'failed'
                elif n_skipped and n_written:
                    status = 'partial'
                elif n_skipped:
                    status = 'skipped'
                else:
                    status = 'done'
                _record(conn, filepath, status,
                        f'{n_written} written, {n_skipped} skipped, {n_failed} failed',
                        n_in, n_written)
            except Exception as exc:
                _record(conn, filepath, 'failed', f'{type(exc).__name__}: {exc}', 0, 0)
                if debug:
                    traceback.print_exc()
            if index % 100 == 0 or index == len(pending):
                print(f'Processed {index}/{len(pending)} input files', flush=True)
    print(f'Processing log: {db_path}')
    return str(db_path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src_dir')
    p.add_argument('dest_dir')
    p.add_argument('--network', default='*')
    p.add_argument('--station', default='*')
    p.add_argument('--metadata', default=None, help='StationXML for channel matching')
    p.add_argument('--non-sds', action='store_true')
    p.add_argument('--glob', default='*.mseed')
    p.add_argument('--mode', choices=('merge', 'overwrite', 'fail'), default='merge')
    p.add_argument('--min-sampling-rate', type=float)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()
    write_sds_archive(args.src_dir, args.dest_dir, networks=args.network,
        stations=args.station, metadata_excel_path=args.metadata,
        use_sds_structure=not args.non_sds, file_glob=args.glob,
        write_mode=args.mode, min_sampling_rate=args.min_sampling_rate,
        debug=args.debug)
