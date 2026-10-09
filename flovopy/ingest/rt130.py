"""REF TEK RT130 ingestion using EPIC PASSOFT rt2ms (2021.090+).

rt2ms performs proprietary REF TEK decoding; FLOVOpy validates its time-aware
parfile and stages its day-volume MiniSEED without modifying raw CF-card data.
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path

REQUIRED = ('das', 'refchan', 'refstrm', 'netcode', 'station', 'location',
            'channel', 'samplerate', 'gain', 'implement_time')
DAY_FILE = re.compile(r'^[^.]+\.[^.]+\.[^.]*\.[^.]+\.\d{4}\.\d{3}$')


def cf_directories(raw_dir):
    """Return EPIC .cf directories, without guessing at REF TEK card layouts."""
    root = Path(raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(p for p in root.glob('*.cf') if p.is_dir())


def inspect_parfile(path, *, require_final=False):
    """Validate rt2ms parfile structure and summarize IDs, not instrument setup."""
    path = Path(path)
    issues, rows = [], []
    with path.open(newline='') as f:
        reader = csv.reader((line for line in f if line.strip() and not line.lstrip().startswith('#')), delimiter=';')
        try:
            fields = [x.strip().lower() for x in next(reader)]
        except StopIteration:
            raise ValueError(f'Empty parfile: {path}')
        if fields != list(REQUIRED):
            raise ValueError(f'Unexpected parfile columns: {fields}; expected {REQUIRED}')
        for line_no, values in enumerate(reader, 2):
            if len(values) != len(REQUIRED):
                issues.append(f'row {line_no}: expected {len(REQUIRED)} columns, got {len(values)}')
                continue
            row = dict(zip(REQUIRED, (v.strip() for v in values)))
            rows.append(row)
            if require_final and (row['netcode'] in ('', 'XX') or not row['station'] or not row['channel']):
                issues.append(f"row {line_no}: unresolved NSLC {row['netcode']}.{row['station']}.{row['location']}.{row['channel']}")
            try:
                from datetime import datetime
                datetime.fromisoformat(row['implement_time'].replace('Z', '+00:00'))
                float(row['samplerate']); float(row['gain'])
            except (ValueError, TypeError):
                issues.append(f'row {line_no}: invalid implement_time, samplerate or gain')
    if not rows:
        issues.append('no valid channel rows')
    return {'path': str(path), 'rows': len(rows), 'stations': sorted({r['station'] for r in rows}),
            'networks': sorted({r['netcode'] for r in rows}), 'issues': issues}


def run_rt2ms(workspace, *, phase='explore', raw_dir='RAW', parfile='parfile.txt',
              executable='rt2ms', execute=False):
    """Explore (-e), convert (-p), or extract SOH logs (-X) with EPIC rt2ms.

    Work directory contains RAW/*.cf; dir.list is written with absolute paths.
    Conversion requires a manually reviewed parfile. No shell execution.
    """
    if phase not in ('explore', 'convert', 'logs'):
        raise ValueError('phase must be explore, convert or logs')
    work = Path(workspace).expanduser().resolve()
    raw = (work / raw_dir).resolve()
    if not raw.is_dir() or work not in raw.parents:
        raise ValueError(f'RAW directory missing or outside workspace: {raw}')
    cards = cf_directories(raw)
    if cards:
        listing = work / 'dir.list'
        args = ['-D', 'dir.list']
    else:
        listing = None
        args = ['-d', str(raw)]
    if phase == 'explore':
        args += ['-e']
    elif phase == 'logs':
        args += ['-X']
    else:
        parpath = (work / parfile).resolve()
        if not parpath.is_file():
            raise FileNotFoundError(f'Run explore and review parfile first: {parpath}')
        audit = inspect_parfile(parpath, require_final=True)
        if audit['issues']:
            raise ValueError('Parfile review failed: ' + '; '.join(audit['issues']))
        args += ['-p', str(parpath)]
    cmd = [executable, *args]
    report = {'phase': phase, 'command': cmd, 'workspace': str(work),
              'cf_directories': len(cards), 'dir_list': str(listing) if listing else None,
              'executed': execute}
    if not execute:
        return report
    if shutil.which(executable) is None:
        raise FileNotFoundError(f'{executable} not found; install EarthScope PASSOFT')
    if listing:
        listing.write_text(''.join(str(p) + '\n' for p in cards))
    completed = subprocess.run(cmd, cwd=work, text=True, capture_output=True, check=False)
    report.update(returncode=completed.returncode, stdout_tail=completed.stdout[-5000:],
                  stderr_tail=completed.stderr[-5000:], message_log=str(work / 'rt2ms.msg'))
    if completed.returncode:
        raise RuntimeError(f'rt2ms exited {completed.returncode}: {completed.stderr[-1500:]}')
    return report


def stage_rt130(mseed_dir, output_sds, *, dry_run=True, strict=True):
    """Stage rt2ms STA.NET.LOC.CHAN.YEAR.DOY files in daily batches.

    Files without a MiniSEED extension are supported; avoid scanning LOGS or RAW.
    """
    from .miniseed import stage_miniseed_tree
    source = Path(mseed_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    candidates = [p for p in source.rglob('*') if p.is_file() and DAY_FILE.fullmatch(p.name)]
    if not candidates:
        raise ValueError(f'No EPIC rt2ms daily files found under {source}')
    # The generic scanner filters with glob patterns, so a unique filename pattern
    # is supplied for extensionless daily volumes; restrict the root to MSEED/.
    result = stage_miniseed_tree(source, output_sds, patterns=('*.????.???',),
                                 dry_run=dry_run, strict=strict)
    result['rt130_daily_files'] = len(candidates)
    return result
