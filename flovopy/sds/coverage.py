"""Sample-aware SDS waveform coverage from the existing audit SQLite database.

Uses original MiniSEED IDs. Intervals are half-open [start, end) in UTC.
Filename-only (speed=2) audit records are excluded, not interpreted as data.
"""
import csv
import datetime as dt
import sqlite3
from collections import defaultdict
from pathlib import Path

UTC = dt.timezone.utc


def _time(value):
    value = str(value).replace('Z', '+00:00')
    t = dt.datetime.fromisoformat(value)
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def compute_coverage(db_path, gap_tolerance_samples=0.5, sds_root=None, canonical_only=False):
    """Return coverage rows for each original NSLC and sampling rate.

    A gap <= gap_tolerance_samples / sampling_rate is treated as contiguous.
    Each row includes segment count, summed sample count (not deduplicated),
    and a flag indicating whether any overlap occurred.
    """
    if canonical_only and sds_root is None:
        raise ValueError("sds_root is required when canonical_only=True")
    if canonical_only:
        from flovopy.sds.sds_utils import is_canonical_sds_path
    groups = defaultdict(list)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute('''
            SELECT t.filepath,t.original_id,t.starttime,t.endtime,t.npts,t.sampling_rate
            FROM trace_metadata AS t JOIN file_log AS f ON f.filepath=t.filepath
            WHERE f.status='done' AND t.npts > 0 AND t.sampling_rate > 0
            ORDER BY t.original_id,t.sampling_rate,t.starttime
        ''')
        for filepath, nslc, start, last, npts, sr in rows:
            if canonical_only and not is_canonical_sds_path(filepath, sds_root):
                continue
            # UTCDateTime endtime is the timestamp of the last sample.
            begin = _time(start)
            end = _time(last) + dt.timedelta(seconds=1.0 / sr)
            if end <= begin:
                continue
            groups[(nslc, sr)].append((begin, end, npts))
    result = []
    for (nslc, sr), spans in sorted(groups.items()):
        spans.sort()
        begin = end = None
        count = samples = overlaps = 0
        def append():
            result.append(dict(original_id=nslc, starttime=begin.isoformat(),
                               endtime_exclusive=end.isoformat(), sampling_rate=sr,
                               segments=count, summed_npts=samples, has_overlap=bool(overlaps)))
        for a, b, n in spans:
            if begin is None:
                begin, end, count, samples, overlaps = a, b, 1, n, 0
            elif a <= end + dt.timedelta(seconds=gap_tolerance_samples / sr):
                if a < end - dt.timedelta(seconds=0.25 / sr):
                    overlaps += 1
                end = max(end, b)
                count += 1
                samples += n
            else:
                append()
                begin, end, count, samples, overlaps = a, b, 1, n, 0
        if begin is not None:
            append()
    return result


def export_coverage(db_path, csv_path, gap_tolerance_samples=0.5, sds_root=None, canonical_only=False):
    rows = compute_coverage(db_path, gap_tolerance_samples, sds_root=sds_root, canonical_only=canonical_only)
    columns = ['original_id', 'starttime', 'endtime_exclusive', 'sampling_rate',
               'segments', 'summed_npts', 'has_overlap']
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows
