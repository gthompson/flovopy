"""Create EPIC DAYS/BUD views from canonical daily SDS files.

EPIC layout: DAYS/STA/STA.NET.LOC.CHAN.YEAR.DOY. SDS files must be
validated and made EPIC-compatible (including byte order) before submission.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from .epic import sds_entries


def export_sds_to_bud(sds_root, bud_root, *, mode='symlink', dry_run=True,
                      overwrite_links=False, include_soh=True, validate=False):
    if mode not in ('symlink', 'copy', 'hardlink'):
        raise ValueError(mode)
    src_root = Path(sds_root).expanduser().resolve()
    dest_root = Path(bud_root).expanduser().resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    if dest_root == src_root or dest_root.is_relative_to(src_root) or src_root.is_relative_to(dest_root):
        raise ValueError('Source and destination trees must not overlap')
    summary = Counter()
    issues = []
    destinations = {}
    for src, d in sds_entries(src_root, include_soh=include_soh):
        loc = '' if d['loc'] in ('', '--') else d['loc']
        dest = dest_root / d['sta'] / f"{d['sta']}.{d['net']}.{loc}.{d['chan']}.{d['year']}.{d['doy']}"
        if dest in destinations:
            summary['collision'] += 1
            issues.append(f'BUD destination collision: {dest} from {destinations[dest]} and {src}')
            continue
        destinations[dest] = src
        if src.stat().st_size == 0:
            summary['empty'] += 1; issues.append(f'Empty source: {src}'); continue
        if validate:
            from obspy import read
            try:
                st = read(str(src), format='MSEED', headonly=True)
                if not st or any((tr.stats.network, tr.stats.station, tr.stats.location or '', tr.stats.channel) !=
                                 (d['net'], d['sta'], loc, d['chan']) for tr in st):
                    raise ValueError('MiniSEED NSLC mismatch')
            except Exception as exc:
                summary['invalid'] += 1; issues.append(f'{src}: {exc}'); continue
        if dest.is_symlink():
            if dest.resolve() == src and mode == 'symlink':
                summary['already_correct'] += 1; continue
            if not overwrite_links:
                summary['conflict'] += 1; issues.append(f'Conflicting symlink: {dest}'); continue
            if not dry_run:
                dest.unlink()
        elif dest.exists():
            summary['conflict'] += 1; issues.append(f'Existing regular file: {dest}'); continue
        if dry_run:
            summary['would_create'] += 1; continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == 'symlink':
            os.symlink(os.path.relpath(src, dest.parent), dest)
        elif mode == 'hardlink':
            os.link(src, dest)
        else:
            import shutil
            shutil.copy2(src, dest)
        summary['created'] += 1
    return {'summary': dict(summary), 'issues': issues, 'mode': mode, 'dry_run': dry_run,
            'source': str(src_root), 'destination': str(dest_root)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sds_root', type=Path)
    p.add_argument('bud_root', type=Path)
    p.add_argument('--mode', choices=('symlink','hardlink','copy'), default='symlink')
    p.add_argument('--execute', action='store_true', help='Default is dry run')
    p.add_argument('--overwrite-links', action='store_true')
    p.add_argument('--no-soh', action='store_true')
    p.add_argument('--validate', action='store_true', help='Read MiniSEED headers to check NSLC')
    p.add_argument('--report', type=Path)
    a = p.parse_args(argv)
    result = export_sds_to_bud(a.sds_root, a.bud_root, mode=a.mode, dry_run=not a.execute,
                               overwrite_links=a.overwrite_links, include_soh=not a.no_soh,
                               validate=a.validate)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 1 if result['issues'] else 0

if __name__ == '__main__':
    raise SystemExit(main())
