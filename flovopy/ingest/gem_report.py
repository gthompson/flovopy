"""Gem deployment inventory and UTC-day availability from gemconvert output.

Header-only MiniSEED scan; GPS/telemetry summaries are streamed. Waveform
coverage is the union of sample-support intervals (duplicates do not inflate it).
"""
from __future__ import annotations
import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from obspy import read
from .gem import load_mapping

SERIAL_RE = re.compile(r"^(\d+)(?:gps|metadata)_\d+\.txt$", re.I)
UTC = timezone.utc
DAY = 86400

def _number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def _time(value):
    if not value: return None
    try:
        if str(value).replace('.', '', 1).isdigit():
            return datetime.fromtimestamp(float(value), UTC)
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).astimezone(UTC)
    except (ValueError, OverflowError, OSError):
        return None

def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace('+00:00', 'Z') if timestamp is not None else ''

def _files_by_serial(folder, kind):
    result = defaultdict(list)
    if folder.is_dir():
        for path in folder.glob(f'*{kind}_*.txt'):
            match = SERIAL_RE.match(path.name)
            if match:
                result[str(int(match.group(1)))].append(path)
    return result

def _telemetry(paths):
    values = {'batt': [], 'temp': []}
    keys = ('batt', 'temp', 'maxWriteTime', 'minFifoFree', 'maxFifoUsed', 'maxOverruns')
    result = {'metadata_records': 0, 'metadata_files': len(paths)}
    start = end = None
    maxima, minima = {}, {}
    gps_on = gps_total = 0
    for path in paths:
        with path.open(newline='', errors='replace') as f:
            for row in csv.DictReader(f):
                result['metadata_records'] += 1
                t = _time(row.get('t'))
                if t:
                    ts = t.timestamp()
                    start = min(start, ts) if start is not None else ts
                    end = max(end, ts) if end is not None else ts
                for k in keys:
                    v = _number(row.get(k))
                    if v is None: continue
                    if k in values: values[k].append(v)
                    maxima[k] = max(maxima.get(k, v), v)
                    minima[k] = min(minima.get(k, v), v)
                flag = _number(row.get('gpsOnFlag'))
                if flag is not None:
                    gps_total += 1
                    gps_on += flag > 0
    result.update(metadata_start_utc=_iso(start), metadata_end_utc=_iso(end),
                  battery_min_v=minima.get('batt'), battery_max_v=maxima.get('batt'),
                  battery_median_v=median(values['batt']) if values['batt'] else None,
                  temperature_min_c=minima.get('temp'), temperature_max_c=maxima.get('temp'),
                  temperature_median_c=median(values['temp']) if values['temp'] else None,
                  max_overruns=maxima.get('maxOverruns'), max_write_time=maxima.get('maxWriteTime'),
                  min_fifo_free=minima.get('minFifoFree'), max_fifo_used=maxima.get('maxFifoUsed'),
                  gps_on_fraction=gps_on / gps_total if gps_total else None)
    return result

def _gps(paths):
    lat, lon = [], []
    start = end = None
    for path in paths:
        with path.open(newline='', errors='replace') as f:
            for row in csv.DictReader(f):
                a, b = _number(row.get('lat')), _number(row.get('lon'))
                if a is None or b is None or not (-90 <= a <= 90 and -180 <= b <= 180): continue
                lat.append(a); lon.append(b)
                t = _time(row.get('t'))
                if t:
                    ts = t.timestamp()
                    start = min(start, ts) if start is not None else ts
                    end = max(end, ts) if end is not None else ts
    return dict(gps_files=len(paths), gps_fixes=len(lat),
                gps_lat_median=median(lat) if lat else None,
                gps_lon_median=median(lon) if lon else None,
                gps_start_utc=_iso(start), gps_end_utc=_iso(end),
                gps_lat_range_deg=max(lat)-min(lat) if lat else None,
                gps_lon_range_deg=max(lon)-min(lon) if lon else None)

def _merge(intervals):
    """Merge half-open [start, end) sample-support intervals."""
    merged = []
    for a, b in sorted(intervals):
        if b <= a: continue
        if merged and a <= merged[-1][1] + 1e-7:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged

def _metrics(intervals, start, end):
    merged = _merge([(max(a, start), min(b, end)) for a,b in intervals
                     if a < end and b > start])
    covered = sum(b-a for a,b in merged)
    gaps = sum(merged[i][0]-merged[i-1][1] for i in range(1,len(merged)))
    return covered, gaps, len(merged)

def _write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def report_gem_deployment(converted_root, mapping_csv, output_csv, *, daily_csv=None,
                          pattern='*.mseed', deployment_start=None, deployment_end=None):
    """Generate deployment inventory and daily availability from gemconvert folders.

    deployment_start/end are optional UTC ISO timestamps defining expected uptime.
    If omitted, daily percentages refer to entire UTC days, including partial
    first/last days; inventory coverage refers to first-to-last observed samples.
    """
    root = Path(converted_root)
    if not (root / 'mseed').is_dir():
        raise FileNotFoundError(f'Expected mseed/ in {root}')
    mapping = load_mapping(mapping_csv)
    expected_start = _time(deployment_start)
    expected_end = _time(deployment_end)
    if (deployment_start and expected_start is None) or (deployment_end and expected_end is None):
        raise ValueError('Invalid UTC deployment_start/deployment_end')
    if expected_start and expected_end and expected_start >= expected_end:
        raise ValueError('deployment_start must precede deployment_end')
    start_limit = expected_start.timestamp() if expected_start else None
    end_limit = expected_end.timestamp() if expected_end else None
    waveforms = defaultdict(lambda: dict(intervals=[], rates=set(), files=set(), samples=0))
    errors = []
    found = 0
    for path in sorted((root/'mseed').rglob(pattern)):
        found += 1
        try:
            stream = read(str(path), format='MSEED', headonly=True)
            for tr in stream:
                serial = str(tr.stats.station).lstrip('0') or '0'
                key = (serial, mapping.get(serial, ''))
                fs = float(tr.stats.sampling_rate)
                if fs <= 0: raise ValueError(f'Invalid sampling rate {fs}')
                a = float(tr.stats.starttime.timestamp)
                b = float(tr.stats.endtime.timestamp) + 1/fs
                entry = waveforms[key]
                entry['intervals'].append((a,b))
                entry['rates'].add(fs)
                entry['samples'] += int(tr.stats.npts)
                entry['files'].add(path)
        except Exception as exc:
            errors.append(f'{path}: {type(exc).__name__}: {exc}')
    gps = _files_by_serial(root/'gps', 'gps')
    metadata = _files_by_serial(root/'metadata', 'metadata')
    keys = sorted(set((s, seed) for s,seed in mapping.items()) | set(waveforms),
                  key=lambda x: (int(x[0]), x[1]))
    inventory = []
    daily = []
    for serial, seed in keys:
        item = waveforms.get((serial, seed), dict(intervals=[], rates=set(), files=set(), samples=0))
        intervals = item['intervals']
        merged = _merge(intervals)
        start = merged[0][0] if merged else None
        end = merged[-1][1] if merged else None
        covered = sum(b-a for a,b in merged)
        span = end-start if start is not None else 0
        gaps = sum(merged[i][0]-merged[i-1][1] for i in range(1,len(merged)))
        rates = ';'.join(f'{x:g}' for x in sorted(item['rates']))
        row = dict(gem_serial=serial, seed_id=seed,
                   mapping_status=('UNMAPPED' if not seed else 'UNRESOLVED' if seed.split('.')[1]=='UNK' else 'mapped'),
                   start_utc=_iso(start), end_exclusive_utc=_iso(end),
                   sample_rates_hz=rates, samples_total=item['samples'],
                   waveform_files=len(item['files']),
                   waveform_bytes=sum(p.stat().st_size for p in item['files']),
                   coverage_pct=round(100*covered/span, 5) if span else None,
                   covered_seconds=round(covered, 5), span_seconds=round(span, 5),
                   gap_seconds=round(gaps, 5), continuous_segments=len(merged),
                   coverage_definition='first-to-last observed sample; not deployment uptime')
        row.update(_gps(gps.get(serial, [])))
        row.update(_telemetry(metadata.get(serial, [])))
        inventory.append(row)
        if start is None and (start_limit is None or end_limit is None):
            continue
        first = start_limit if start_limit is not None else start
        last = end_limit if end_limit is not None else end
        if first is None or last is None or last <= first: continue
        day_start = math.floor(first / DAY) * DAY
        while day_start < last:
            day_end = day_start + DAY
            window_start = max(day_start, first)
            window_end = min(day_end, last)
            if window_end <= window_start:
                day_start = day_end; continue
            duration = window_end-window_start
            present, internal_gaps, segments = _metrics(intervals, window_start, window_end)
            # Missing includes leading/trailing gaps within the expected window.
            missing = max(0., duration-present)
            daily.append(dict(gem_serial=serial, seed_id=seed,
                              mapping_status=row['mapping_status'],
                              date_utc=datetime.fromtimestamp(day_start, UTC).date().isoformat(),
                              window_start_utc=_iso(window_start), window_end_exclusive_utc=_iso(window_end),
                              expected_seconds=round(duration, 5), covered_seconds=round(present, 5),
                              missing_seconds=round(missing, 5), completeness_pct=round(100*present/duration, 5),
                              internal_gap_seconds=round(internal_gaps, 5),
                              continuous_segments=segments, sample_rates_hz=rates,
                              window_basis='explicit deployment interval' if start_limit is not None or end_limit is not None else 'observed span (partial boundary days)'))
            day_start = day_end
    inventory_columns = list(inventory[0]) if inventory else ['gem_serial','seed_id']
    daily_columns = ['gem_serial','seed_id','mapping_status','date_utc','window_start_utc',
                     'window_end_exclusive_utc','expected_seconds','covered_seconds',
                     'missing_seconds','completeness_pct','internal_gap_seconds',
                     'continuous_segments','sample_rates_hz','window_basis']
    if daily_csv is None:
        daily_csv = Path(output_csv).with_name('gem_daily_availability.csv')
    _write_csv(output_csv, inventory, inventory_columns)
    _write_csv(daily_csv, daily, daily_columns)
    return dict(deployment_csv=str(output_csv), daily_csv=str(daily_csv),
                deployment_rows=len(inventory), daily_rows=len(daily), mseed_files=found,
                read_errors=errors, unmapped_serials=sorted({s for s,seed in waveforms if not seed}),
                note='Coverage uses union of waveform sample intervals. GPS coordinates are indicative, not surveyed.')
