"""EPIC MiniSEED discovery, staging and validation; raw data are never modified."""
from __future__ import annotations
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SDS_RE = re.compile(r'^(?P<net>[^.]+)\.(?P<sta>[^.]+)\.(?P<loc>[^.]*)\.(?P<chan>[^.]+)\.(?P<type>[DS])\.(?P<year>\d{4})\.(?P<doy>\d{3})$')


def sds_entries(root, *, include_soh=True):
    """Yield (path, header inferred from canonical SDS path); reject inconsistent paths."""
    root = Path(root)
    for p in sorted(root.glob('*/*/*/*.*/*')):
        if not p.is_file() or p.is_symlink():
            continue
        m = SDS_RE.fullmatch(p.name)
        if not m or (m['type'] == 'S' and not include_soh):
            continue
        d = m.groupdict()
        if (p.parent.name != f"{d['chan']}.{d['type']}" or p.parent.parent.name != d['sta']
                or p.parent.parent.parent.name != d['net'] or p.parent.parent.parent.parent.name != d['year']):
            raise ValueError(f'SDS path/header mismatch: {p}')
        try:
            datetime.strptime(d['year'] + d['doy'], '%Y%j')
        except ValueError as exc:
            raise ValueError(f'Invalid SDS date: {p}') from exc
        yield p, d


def discover_miniseed(raw_root, instrument='auto'):
    """Discover MiniSEED candidates without assuming a single firmware archive pattern.

    The caller MUST review the discovered file manifest before staging. Q330 .sdr
    files are included. Non-MiniSEED log/text files are excluded.
    """
    raw = Path(raw_root)
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    accepted = {'.mseed', '.miniseed', '.ms', '.sdr'}
    found = []
    for p in sorted(raw.rglob('*')):
        if p.is_file() and (p.suffix.lower() in accepted or re.fullmatch(r'\d{3}\.D', p.parent.name)):
            found.append(p)
    return found


def stage_miniseed(raw_root, output_sds, *, instrument='auto', dry_run=True,
                   strict=True, verbose=True, write_mode='merge'):
    """Stage Centaur/Pegasus/Q330 MiniSEED in chronological UTC-day batches.

    Reuses :func:`flovopy.ingest.miniseed.stage_miniseed_tree` rather than
    calling ``EnhancedSDSClient.write_stream`` for each input file.

    * Source MiniSEED headers supply NSLC identifiers; no remapping occurs.
    * A SQLite index groups files by actual trace timestamps, including files
      crossing midnight. Each day is read, sorted and merged in memory.
    * Existing staging data are refused: use a fresh staging directory per
      service run, then promote using the transactional SDS archive merger.
    * Dry runs inspect source data but do not create staging files.

    ``strict=False`` allows incomplete days and reports errors; such output
    must not be promoted automatically to a master SDS archive.
    """
    from .miniseed import stage_miniseed_tree

    allowed = {'auto', 'centaur', 'pegasus', 'q330'}
    if instrument not in allowed:
        raise ValueError(f'Unknown instrument {instrument!r}; expected {sorted(allowed)}')
    raw = Path(raw_root).expanduser().resolve()
    output = Path(output_sds).expanduser().resolve()
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError('Input and staging directories must not be nested')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Nonempty stage refused: {output}; use a fresh staging directory')

    # Include Q330 .sdr files; ObsPy must be able to decode them as MiniSEED.
    patterns = ('*.mseed', '*.miniseed', '*.ms', '*.sdr', '*.MSEED',
                '*.MINISEED', '*.MS', '*.SDR')
    report = stage_miniseed_tree(
        raw, output, patterns=patterns, dry_run=dry_run,
        strict=strict, verbose=verbose, write_mode=write_mode,
    )
    counts = report.get('counts', {})
    return {
        'instrument': instrument,
        'files_discovered': counts.get('files_discovered', 0),
        'files_read': counts.get('files_indexed', 0),
        'traces': counts.get('traces_after_merge', 0),
        'daily_batches_written': counts.get('days_written', 0),
        'daily_batches_validated': counts.get('days_validated', 0),
        'errors': report.get('errors', []),
        'dry_run': dry_run,
        'stage': str(output),
        'counts': counts,
    }


def validate_sds(sds_root, *, inspect_headers=True, output_json=None, output_csv=None):
    """Check filename/path/record headers and report daily coverage (no silent fixes)."""
    rows, errors = [], []
    for p, d in sds_entries(sds_root):
        row = {'path': str(p), **d, 'bytes': p.stat().st_size, 'status': 'ok'}
        if row['bytes'] == 0:
            row['status'] = 'empty'
            errors.append({'file': str(p), 'error': 'zero bytes'})
        if inspect_headers and row['status'] == 'ok':
            try:
                from obspy import read
                st = read(str(p), format='MSEED', headonly=True)
                if not st:
                    raise ValueError('No traces')
                for tr in st:
                    s = tr.stats
                    loc = '' if d['loc'] in ('--', '') else d['loc']
                    actual = (s.network, s.station, s.location or '', s.channel)
                    expected = (d['net'], d['sta'], loc, d['chan'])
                    if actual != expected:
                        raise ValueError(f'NSLC mismatch: {actual} != {expected}')
                    start_day = datetime.strptime(d['year'] + d['doy'], '%Y%j').replace(tzinfo=timezone.utc)
                    # A daily file can contain a sample spanning midnight at the end.
                    if s.starttime.datetime.replace(tzinfo=timezone.utc) < start_day - timedelta(seconds=1) or s.starttime.datetime.replace(tzinfo=timezone.utc) >= start_day + timedelta(days=1):
                        raise ValueError(f'Trace start outside filename day: {s.starttime}')
                row['traces'] = len(st)
                row['start'] = str(min(tr.stats.starttime for tr in st))
                row['end'] = str(max(tr.stats.endtime for tr in st))
                row['sample_rates'] = ','.join(str(x) for x in sorted({tr.stats.sampling_rate for tr in st}))
            except Exception as exc:
                row['status'] = 'invalid'
                errors.append({'file': str(p), 'error': str(exc)})
        rows.append(row)
    result = {'files': len(rows), 'errors': errors, 'status_counts': dict(Counter(r['status'] for r in rows)), 'rows': rows}
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, indent=2))
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        fields = ['path','net','sta','loc','chan','type','year','doy','bytes','status','traces','start','end','sample_rates']
        with Path(output_csv).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
    return result
