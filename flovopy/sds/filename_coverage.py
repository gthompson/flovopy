"""Filename-derived SDS day-presence intervals (NOT observed waveform coverage)."""
from __future__ import annotations
import csv
import os
import re
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

_SDS_NAME = re.compile(r'^([^.]+)\.([^.]+)\.([^.]+)\.([^.]+)\.([A-Za-z])\.(\d{4})\.(\d{3})$')


def _canonical_day(filepath: str, sds_root: str):
    """Return (NSLC, date) for a strictly canonical SDS file, else None."""
    root = Path(sds_root).resolve()
    path = Path(filepath).resolve()
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) != 5 or any(p.startswith('.') for p in parts):
        return None
    year_dir, net_dir, sta_dir, chan_dir, filename = parts
    m = _SDS_NAME.fullmatch(filename)
    if not m:
        return None
    net, sta, loc, chan, typ, yyyy, jjj = m.groups()
    if (year_dir != yyyy or net_dir != net or sta_dir != sta or
            chan_dir != f'{chan}.{typ}' or not 1 <= int(jjj) <= 366):
        return None
    try:
        day = date(int(yyyy), 1, 1) + timedelta(days=int(jjj) - 1)
    except ValueError:
        return None
    if day.year != int(yyyy):
        return None
    return f'{net}.{sta}.{loc}.{chan}', day


def export_filename_coverage(db_path: str, output_csv: str, sds_root: str):
    """Export maximal runs of consecutive SDS file-days per original NSLC.

    Uses successful filename-mode records in file_log; never opens MiniSEED.
    Ignores stale records outside the supplied root and noncanonical paths.
    """
    days = defaultdict(set)
    excluded = 0
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT filepath FROM file_log WHERE status='done' AND scan_speed=2")
        for (filepath,) in rows:
            result = _canonical_day(filepath, sds_root)
            if result is None:
                excluded += 1
                continue
            nslc, day = result
            days[nslc].add(day)
    records = []
    for nslc, dayset in sorted(days.items()):
        sorted_days = sorted(dayset)
        if not sorted_days:
            continue
        first = last = sorted_days[0]
        for day in sorted_days[1:]:
            if day == last + timedelta(days=1):
                last = day
            else:
                records.append((nslc, first.isoformat(), last.isoformat(), (last-first).days+1))
                first = last = day
        records.append((nslc, first.isoformat(), last.isoformat(), (last-first).days+1))
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original_id', 'first_sds_day', 'last_sds_day', 'n_file_days'])
        writer.writerows(records)
    print(f'📄 Filename-only SDS day coverage saved to {output_csv}: '
          f'{len(days)} NSLCs, {len(records)} day-runs; '
          f'{excluded} noncanonical/out-of-root DB entries excluded.')
    print('⚠️ Filename coverage is file presence only, not verified waveform coverage.')
    return records
