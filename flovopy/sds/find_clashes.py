"""Compare two canonical SDS archives. Read-only unless --merge is explicitly set."""
from __future__ import annotations
import argparse
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from flovopy.core.miniseed_io import compare_mseed_files, read_mseed, smart_merge, write_mseed


def _canonical(relative: Path) -> bool:
    parts = relative.parts
    if len(parts) != 5 or any(p.startswith('.') for p in parts):
        return False
    year, net, sta, channel_dir, filename = parts
    if not (year.isdigit() and len(year) == 4 and 1 <= int(year) <= 9999):
        return False
    fields = filename.split('.')
    if len(fields) != 7:
        return False
    fn_net, fn_sta, loc, chan, typ, fn_year, jday = fields
    return (net == fn_net and sta == fn_sta and channel_dir == f"{chan}.{typ}"
            and year == fn_year and len(jday) == 3 and jday.isdigit()
            and 1 <= int(jday) <= (366 if int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0) else 365)
            and bool(net and sta and chan) and typ == 'D')


def _files(root: Path):
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith('.') and not (Path(folder) / d).is_symlink()]
        for name in names:
            if name.startswith('.'):
                continue
            p = Path(folder) / name
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(root)
                if _canonical(rel):
                    yield str(rel)


def process_file(args):
    rel, src_root, dst_root = args
    src = Path(src_root) / rel
    dst = Path(dst_root) / rel
    try:
        src_size = src.stat().st_size
        if not dst.is_file():
            return {'path': rel, 'status': 'missing', 'source_size': src_size, 'destination_size': ''}
        dst_size = dst.stat().st_size
        same, err = compare_mseed_files(str(src), str(dst))
        return {'path': rel, 'status': 'error' if err else ('same' if same else 'clash'),
                'source_size': src_size, 'destination_size': dst_size, 'error': err or ''}
    except Exception as exc:
        return {'path': rel, 'status': 'error', 'source_size': '', 'destination_size': '', 'error': str(exc)}


def _csv(path, headers, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def find_clashes_parallel(source_root, dest_root, outdir='.', nproc=6, sample_size=None, merge=False):
    src, dst, output = Path(source_root).resolve(), Path(dest_root).resolve(), Path(outdir).resolve()
    if not src.is_dir() or not dst.is_dir():
        raise ValueError('Source and destination must be existing directories')
    if merge and (output == src or output == dst or output in src.parents or output in dst.parents):
        raise ValueError('Merge staging output must not be an archive root or ancestor')
    paths = list(_files(src))
    if sample_size is not None:
        if sample_size < 0: raise ValueError('sample_size must be >= 0')
        random.Random(0).shuffle(paths)
        paths = paths[:sample_size]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    args = ((p, str(src), str(dst)) for p in paths)
    with ProcessPoolExecutor(max_workers=max(1, nproc)) as executor:
        for result in executor.map(process_file, args, chunksize=16):
            results.append(result)
    clashed = [r for r in results if r['status'] == 'clash']
    missing = [r for r in results if r['status'] == 'missing']
    same = [r for r in results if r['status'] == 'same']
    errors = [r for r in results if r['status'] == 'error']
    _csv(output/'find_clashes_clashed.csv', ['Relative Path','Source Size (bytes)','Destination Size (bytes)'],
         [(r['path'],r['source_size'],r['destination_size']) for r in clashed if r['source_size'] != r['destination_size']])
    _csv(output/'find_clashes_samesize.csv', ['Relative Path','Size (bytes)'],
         [(r['path'],r['source_size']) for r in results if r['destination_size'] != '' and r['source_size'] == r['destination_size']])
    _csv(output/'find_clashes_same_contents.csv', ['Relative Path','Size (bytes)'],
         [(r['path'],r['source_size']) for r in same])
    _csv(output/'find_clashes_clashed_contents.csv', ['Relative Path','Source Size (bytes)','Destination Size (bytes)'],
         [(r['path'],r['source_size'],r['destination_size']) for r in clashed])
    _csv(output/'find_clashes_safetocopy.csv', ['Relative Path','Source Size (bytes)'],
         [(r['path'],r['source_size']) for r in missing])
    _csv(output/'find_clashes_read_errors.csv', ['Relative Path','Error Message'],
         [(r['path'],r.get('error','')) for r in errors])
    print(f'Compared {len(results)} SDS files: {len(same)} identical, {len(missing)} missing, '
          f'{len(clashed)} content clashes, {len(errors)} errors')
    if not merge:
        print('Read-only comparison complete; use --merge to stage merged clash files.')
        return results
    staging = output / 'merged_good'
    conflicts = []
    for r in clashed:
        rel = r['path']
        try:
            st = read_mseed(str(src/rel)) + read_mseed(str(dst/rel))
            merged = smart_merge(st, skip_low_rate_channels=False)
            # Current FLOVOpy smart_merge returns a Stream by default.
            if not hasattr(merged, 'write'):
                raise RuntimeError(f'smart_merge returned {type(merged).__name__}, not a Stream')
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            write_mseed(merged, str(target), overwrite_ok=False)
        except Exception as exc:
            conflicts.append((rel, str(exc)))
    _csv(output/'find_clashes_merge_conflicts.csv', ['Relative Path','Error Message'], conflicts)
    print(f'Merged outputs staged at {staging}; {len(conflicts)} merge failures. Original archives unchanged.')
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare canonical SDS archives without modifying either archive.')
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('-o','--outdir',default='.')
    parser.add_argument('-n','--sample-size',type=int)
    parser.add_argument('-p','--processes',type=int,default=6)
    parser.add_argument('--merge',action='store_true',help='Stage merged content clashes under outdir/merged_good')
    args=parser.parse_args()
    find_clashes_parallel(args.source,args.destination,args.outdir,args.processes,args.sample_size,merge=args.merge)

if __name__ == '__main__':
    main()
