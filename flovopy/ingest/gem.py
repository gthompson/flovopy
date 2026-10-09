"""Gem hour-MiniSEED to staged SDS; gemconvert remains the upstream raw decoder.

Never write directly to master SDS. Gem serial-to-SEED mapping is campaign-specific.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import json
import subprocess
from obspy import read
from flovopy.enhanced.sdsclient import EnhancedSDSClient

def load_mapping(path):
    """Read REQUIRED deployment CSV: serial,id (full NET.STA.LOC.CHA).

    Also accepts gemlog station_info.txt with SN,Network,Station,Location,Channel.
    Channel is mandatory: it cannot be inferred from the instrument serial.
    """
    if path is None:
        raise ValueError('A deployment-specific --mapping CSV is required; no hardwired Gem IDs')
    path = Path(path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'Empty mapping: {path}')
        rows = list(reader)
    result = {}
    for row in rows:
        serial = (row.get('serial') or row.get('SN') or '').strip().lstrip('0') or '0'
        seed_id = (row.get('id') or '').strip()
        if not seed_id:
            keys = ('Network', 'Station', 'Location', 'Channel')
            if not all(k in row for k in keys):
                raise ValueError('station_info.txt lacks Channel; add Channel or use serial,id CSV')
            seed_id = '.'.join((row.get(k) or '').strip() for k in keys)
        parts = seed_id.split('.')
        if len(parts) != 4 or not parts[0] or not parts[1] or len(parts[3]) != 3:
            raise ValueError(f'Invalid SEED ID {serial}: {seed_id}')
        if serial in result and result[serial] != seed_id:
            raise ValueError(f'Conflicting Gem serial {serial}: {result[serial]} vs {seed_id}')
        result[serial] = seed_id
    if not result:
        raise ValueError('No mapping entries')
    return result

def stage_gem_mseed(input_root, sds_root, *, mapping=None, pattern='*.mseed',
                    recursive=False, dry_run=False, write_mode='merge',
                    verbose=True, strict=True):
    """Read gemconvert hourly files, map IDs and write staged SDS with FLOVOpy.

    Unmapped serials are errors by default, never silently stored as station numbers.
    Errors stop before a write for that source file. Report counts and paths.
    """
    source = Path(input_root).expanduser()
    target = Path(sds_root).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source.resolve() == target.resolve() or source.resolve() in target.resolve().parents:
        raise ValueError('SDS staging root must not be inside the MiniSEED source')
    ids = load_mapping(mapping)
    # Placeholder assignment is provenance, not a defensible station ID.
    unresolved = {k: v for k, v in ids.items() if v.split('.')[1] == 'UNK'}
    files = sorted(source.rglob(pattern) if recursive else source.glob(pattern))
    if not files:
        raise ValueError(f'No Gem MiniSEED files matched {pattern!r} in {source}')
    counts = Counter(files_found=len(files))
    errors = []
    client = None if dry_run else EnhancedSDSClient(str(target))
    for index, path in enumerate(files, 1):
        if path.stat().st_size == 0:
            counts['skipped_empty'] += 1
            if verbose: print(f'SKIP EMPTY {path}', flush=True)
            continue
        try:
            # Dry run still reads headers to verify mappings; no waveform arrays needed.
            stream = read(str(path), format='MSEED', headonly=dry_run)
            if not stream:
                raise ValueError('No readable traces')
            unknown = sorted({str(tr.stats.station).lstrip('0') or '0' for tr in stream
                              if (str(tr.stats.station).lstrip('0') or '0') not in ids})
            if unknown and strict:
                raise ValueError(f'Unmapped Gem serial(s) {unknown}; mapping required')
            placeholder = sorted({str(tr.stats.station).lstrip('0') or '0' for tr in stream
                                  if (str(tr.stats.station).lstrip('0') or '0') in unresolved})
            if placeholder and strict:
                raise ValueError(f'Unresolved Gem station assignment {placeholder}; supply corrected mapping')
            for tr in stream:
                serial = str(tr.stats.station).lstrip('0') or '0'
                if serial in ids:
                    tr.id = ids[serial]
                else:
                    counts['unmapped_traces'] += 1
            counts['traces'] += len(stream)
            if not dry_run:
                client.write_stream(stream, mode=write_mode, preprocess=False, verbose=False)
                counts['files_written'] += 1
            else:
                counts['files_validated'] += 1
        except Exception as exc:
            counts['errors'] += 1
            errors.append(f'{path}: {type(exc).__name__}: {exc}')
            if verbose: print(f'ERROR {errors[-1]}', flush=True)
        if verbose and (index == 1 or index % 50 == 0 or index == len(files)):
            print(f'GEM {index}/{len(files)} validated={counts["files_validated"]} '
                  f'written={counts["files_written"]} errors={counts["errors"]}', flush=True)
    return dict(source=str(source), target=str(target), dry_run=dry_run,
                counts=dict(counts), errors=errors)

def run_gemconvert(workspace, *, executable='gemconvert'):
    """Run upstream gemconvert in a prepared workspace containing raw/.

    The caller prepares raw/ and decides whether rerunning is safe: gemconvert
    overwrites existing hourly MiniSEED and appends logs/metadata.
    """
    workspace = Path(workspace)
    if not (workspace / 'raw').is_dir():
        raise FileNotFoundError(f'Expected raw/ inside {workspace}')
    subprocess.run([executable], cwd=str(workspace), check=True)
    return workspace / 'mseed'


def summarize_gem_gps(gps_root, output_csv, *, mapping=None):
    """Summarize raw gemconvert GPS logs without discarding provenance.

    Delegates GPS estimation to gemlog, which reports medians/uncertainties
    and the time interval of fixes. The output CSV is for QC, not a substitute
    for a surveyed station location or StationXML.
    """
    try:
        import gemlog
    except ImportError as exc:
        raise RuntimeError('GPS summary requires gemlog (pip install gemlog)') from exc
    gps_root = Path(gps_root)
    if not gps_root.is_dir():
        raise FileNotFoundError(gps_root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # gemlog accepts a station_info CSV with SN,Network,Station,Location.
    # Our full NSLC mapping also includes Channel; split it here.
    import tempfile
    ids = load_mapping(mapping)
    with tempfile.TemporaryDirectory() as tempdir:
        station_info = Path(tempdir) / 'station_info.txt'
        with station_info.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['SN', 'Network', 'Station', 'Location'])
            for serial, seed_id in sorted(ids.items(), key=lambda kv: int(kv[0])):
                net, sta, loc, _ = seed_id.split('.')
                writer.writerow([serial.zfill(3), net, sta, loc])
        return gemlog.summarize_gps(
            str(gps_root), station_info=str(station_info),
            output_file=str(output_csv))
