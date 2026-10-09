"""Generic recursive MiniSEED -> staged daily SDS -> transactional master merge.

Input traces must already have correct NET.STA.LOC.CHA identifiers. No
instrument-specific decoding or remapping is performed.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path


def _scan(source, db, patterns, *, strict, errors, counts):
    from obspy import UTCDateTime, read
    seen = set()
    for pattern in patterns:
        for path in source.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            counts['files_discovered'] += 1
            try:
                if not path.stat().st_size:
                    raise ValueError('empty file')
                stream = read(str(path), format='MSEED', headonly=True)
                if not stream:
                    raise ValueError('no traces')
                days = set()
                for tr in stream:
                    if not (tr.stats.network and tr.stats.station and tr.stats.channel):
                        raise ValueError(f'incomplete SEED ID {tr.id}')
                    start = UTCDateTime(tr.stats.starttime.date)
                    end = UTCDateTime(tr.stats.endtime.date)
                    while start <= end:
                        days.add(str(start.date))
                        start += 86400
                for day in days:
                    db.execute('INSERT OR IGNORE INTO files(day,path) VALUES (?,?)', (day, str(path)))
                counts['files_indexed'] += 1
            except Exception as exc:
                counts['scan_errors'] += 1
                errors.append(f'{path}: {exc}')
                if strict:
                    raise RuntimeError(errors[-1]) from exc
    db.commit()


def stage_miniseed_tree(input_root, staging_sds, *, patterns=('*.mseed', '*.miniseed', '*.ms'),
                        dry_run=False, strict=True, verbose=True, write_mode='merge'):
    """Index by actual UTC trace time, read and merge one day at a time.

    Existing staging files are updated by EnhancedSDSClient. Source files are
    never modified. In strict mode scan/read errors abort before a final merge.
    For non-strict mode, affected days may be incomplete: review errors.
    """
    from obspy import Stream, UTCDateTime, read
    from flovopy.enhanced.sdsclient import EnhancedSDSClient

    source = Path(input_root).expanduser().resolve()
    staging = Path(staging_sds).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == staging or source in staging.parents or staging in source.parents:
        raise ValueError('Input and staging SDS directories must not contain one another')
    counts, errors = Counter(), []
    with tempfile.TemporaryDirectory(prefix='flovopy_mseed_index_') as tmp:
        with sqlite3.connect(Path(tmp) / 'index.sqlite') as db:
            db.execute('CREATE TABLE files (day TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(day,path))')
            _scan(source, db, patterns, strict=strict, errors=errors, counts=counts)
            days = [row[0] for row in db.execute('SELECT DISTINCT day FROM files ORDER BY day')]
            if not days:
                raise ValueError(f'No valid MiniSEED found under {source}')
            client = None if dry_run else EnhancedSDSClient(str(staging))
            for day in days:
                start = UTCDateTime(day)
                end = start + 86400
                stream = Stream()
                failed = False
                paths = [r[0] for r in db.execute('SELECT path FROM files WHERE day=? ORDER BY path', (day,))]
                for filename in paths:
                    try:
                        raw = read(filename, format='MSEED')
                        for tr in raw:
                            # Exclusive day end: preserve samples on next UTC day.
                            piece = tr.slice(starttime=start, endtime=end - tr.stats.delta,
                                             nearest_sample=False)
                            if piece.stats.npts and piece.stats.starttime < end and piece.stats.endtime >= start:
                                stream += piece
                    except Exception as exc:
                        failed = True
                        counts['read_errors'] += 1
                        errors.append(f'{filename} ({day}): {exc}')
                        if strict:
                            raise RuntimeError(errors[-1]) from exc
                if failed and not strict:
                    counts['partial_days'] += 1
                if not stream:
                    continue
                stream.sort(keys=['starttime'])
                # method=0 does not silently overwrite conflicting overlapping samples.
                try:
                    stream.merge(method=0, fill_value=None)
                except Exception as exc:
                    counts['merge_errors'] += 1
                    errors.append(f'{day}: {exc}')
                    if strict:
                        raise
                    continue
                if not dry_run:
                    client.write_stream(stream, mode=write_mode, preprocess=False, verbose=False)
                    counts['days_written'] += 1
                else:
                    counts['days_validated'] += 1
                counts['traces_after_merge'] += len(stream)
                if verbose:
                    print(f'{day}: {len(paths)} source files, {len(stream)} merged trace segments')
    return {'input': str(source), 'staging_sds': str(staging), 'dry_run': dry_run,
            'counts': dict(counts), 'errors': errors}


def ingest_miniseed_tree(input_root, target_sds, *, staging_sds, merge_mode='fast',
                         dry_run=False, strict=True, patterns=('*.mseed','*.miniseed','*.ms'),
                         progress_every=50, verbose=True):
    """Stage daily SDS first, then use FLOVOpy's transactional SDS merger.

    The staging path must be persistent and separate from input and target.
    When dry_run=True, no staging files are written and master merge is skipped.
    """
    from flovopy.sds.merge_sds_archives import merge_sds_archives
    source = Path(input_root).expanduser().resolve()
    target = Path(target_sds).expanduser().resolve()
    stage = Path(staging_sds).expanduser().resolve()
    if len({source, target, stage}) != 3:
        raise ValueError('Input, staging and master paths must differ')
    if any(a in b.parents for a in (source, stage, target) for b in (source, stage, target) if a != b):
        raise ValueError('Input, staging and master must not be nested')
    report = stage_miniseed_tree(source, stage, patterns=patterns, dry_run=dry_run,
                                 strict=strict, verbose=verbose)
    if dry_run:
        report['merge'] = 'skipped (dry run)'
        return report
    if report['errors']:
        # Never promote a known-incomplete staging run automatically.
        report['merge'] = 'skipped (staging errors; inspect and retry)'
        return report
    summary = merge_sds_archives(stage, target, mode=merge_mode,
                                 dry_run=False, progress_every=progress_every)
    report['merge'] = summary.as_dict() if hasattr(summary, 'as_dict') else str(summary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir')
    parser.add_argument('target_sds')
    parser.add_argument('--staging-sds', required=True)
    parser.add_argument('--pattern', action='append', dest='patterns')
    parser.add_argument('--mode', choices=('fast','slow'), default='fast')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-errors', action='store_true')
    args = parser.parse_args(argv)
    result = ingest_miniseed_tree(args.input_dir, args.target_sds,
                                  staging_sds=args.staging_sds,
                                  patterns=tuple(args.patterns) if args.patterns else ('*.mseed','*.miniseed','*.ms'),
                                  merge_mode=args.mode, dry_run=args.dry_run,
                                  strict=not args.allow_errors)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
