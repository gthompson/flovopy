"""Recover SiliconAudio Gecko SD cards and convert minute MiniSEED to staging SDS.

Recovery accepts an SD-card root or its data/ directory, and copies YYYY/MM/DD/HH
files plus histogram CSVs into a local download directory. Conversion accepts
MM/DD/HH/*.ms (with year=) or YYYY/MM/DD/HH/*.ms.
Uses FLOVOpy EnhancedSDSClient.write_stream; the canonical SDS is merged
separately using flovopy.sds.merge_sds_archives.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _numeric_dirs(path):
    return sorted((p for p in path.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))


def _days(root: Path, year: int | None):
    """Yield (UTC day, directory) for MM/DD or YYYY/MM/DD trees."""
    top = _numeric_dirs(root)
    if not top:
        return
    if all(len(p.name) == 4 for p in top):
        for yr in top:
            for mo in _numeric_dirs(yr):
                for dy in _numeric_dirs(mo):
                    yield datetime(int(yr.name), int(mo.name), int(dy.name), tzinfo=timezone.utc), dy
    else:
        if year is None:
            raise ValueError(f"Year required for MM/DD/HH SiliconAudio directory: {root}")
        for mo in top:
            for dy in _numeric_dirs(mo):
                yield datetime(int(year), int(mo.name), int(dy.name), tzinfo=timezone.utc), dy


def convert_siliconaudio(input_root, sds_root, *, year=None, pattern='*.ms',
                         write_mode='merge', preprocess=False, dry_run=False,
                         verbose=True, strict=True, **write_kwargs):
    """Convert one station's minute files to day-based SDS files.

    This is restartable via EnhancedSDSClient's merge mode, but writing into
    a dedicated staging SDS root is recommended. Existing master files are
    never modified by this function unless explicitly supplied as sds_root.

    Returns a dictionary with day/file counts, errors and written paths.
    """
    from obspy import Stream, UTCDateTime, read
    from flovopy.enhanced.sdsclient import EnhancedSDSClient

    source = Path(input_root).expanduser()
    target = Path(sds_root).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source.resolve() == target.resolve() or target.resolve().is_relative_to(source.resolve()):
        raise ValueError('SDS output must not be inside the source directory')
    days = list(_days(source, year))
    if not days:
        raise ValueError(f'No dated subdirectories found under {source}')
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        client = EnhancedSDSClient(str(target))
    stats = Counter()
    written_paths = []
    errors = []
    for date, day_dir in days:
        files = [f for hour in _numeric_dirs(day_dir) for f in sorted(hour.glob(pattern)) if f.is_file()]
        stats['days_seen'] += 1
        stats['files_found'] += len(files)
        if verbose:
            print(f'{date.date()} | {len(files)} minute files', flush=True)
        if not files:
            continue
        if dry_run:
            stats['days_with_files'] += 1
            continue
        day_stream = Stream()
        day_failed = False
        for f in files:
            if f.stat().st_size == 0:
                stats['empty_files'] += 1
                continue
            try:
                day_stream += read(str(f), format='MSEED')
                stats['files_read'] += 1
            except Exception as exc:
                day_failed = True
                stats['read_errors'] += 1
                errors.append(f'{f}: {exc}')
                if verbose:
                    print(f'  ERROR reading {f}: {exc}', flush=True)
        if not day_stream:
            continue
        # Never publish an incomplete day silently when a minute file failed.
        if day_failed and strict:
            stats['days_skipped'] += 1
            continue
        try:
            day_stream.merge(method=0, fill_value=None)
            # End boundary is inclusive in ObsPy; subtract one sample per trace
            # so samples at midnight belong to the following SDS day.
            start = UTCDateTime(date)
            for tr in day_stream:
                tr.trim(starttime=start, endtime=start + 86400 - tr.stats.delta,
                        nearest_sample=False)
            day_stream = Stream(tr for tr in day_stream if tr.stats.npts)
            if not day_stream:
                continue
            written = client.write_stream(day_stream, mode=write_mode,
                                          preprocess=preprocess, verbose=verbose,
                                          **write_kwargs)
            written_paths.extend(str(p) for p in (written or []))
            stats['days_written'] += 1
            stats['sds_files_reported'] += len(written or [])
        except Exception as exc:
            stats['write_errors'] += 1
            errors.append(f'{date.date()}: {exc}')
            if verbose:
                print(f'  ERROR writing {date.date()}: {exc}', flush=True)
    return {'source': str(source), 'target': str(target), 'dry_run': dry_run,
            **dict(stats), 'errors': errors, 'written': written_paths}


# SD-card recovery helpers, preserved from recover_gecko_sd.py

import argparse
import hashlib
import os
import re
import shutil
import sys
from typing import Optional, Tuple

YEAR_RE = re.compile(r"^\d{4}$")
MONTH_RE = re.compile(r"^(0[1-9]|1[0-2])$")
DAY_RE = re.compile(r"^(0[1-9]|[12]\d|3[01])$")
HOUR_RE = re.compile(r"^([01]\d|2[0-3])$")

DATA_FILE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<hhmm>\d{4}) "
    r"(?P<ss>\d{2}) "
    r"(?P<station>[A-Za-z0-9_-]+)\."
    r"(?P<ext>ms|ss)$",
    re.ASCII,
)

HISTOGRAM_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})\.csv$", re.ASCII)


class Logger:
    def __init__(self, path: Path, dry_run: bool):
        self.path = path
        self.dry_run = dry_run

    def write(self, message: str = "") -> None:
        print(message)
        # A dry run should not create anything in the destination.
        if self.dry_run:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", errors="backslashreplace") as f:
            f.write(message + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_name(name: str) -> str:
    """Printable/loggable representation even for malformed Unicode."""
    return name.encode("utf-8", "backslashreplace").decode("utf-8", "replace")


def safe_scandir(path: Path, log: Logger, counts: Counter) -> list[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            return list(it)
    except (OSError, UnicodeError) as exc:
        counts["scan_errors"] += 1
        log.write(f"SCAN ERROR: {safe_name(str(path))}: {exc}")
        return []


def entry_is_dir(entry: os.DirEntry, log: Logger, counts: Counter) -> bool:
    try:
        return entry.is_dir(follow_symlinks=False)
    except (OSError, UnicodeError) as exc:
        counts["entry_errors"] += 1
        log.write(f"BAD DIRECTORY ENTRY: {safe_name(entry.path)}: {exc}")
        return False


def entry_is_file(entry: os.DirEntry, log: Logger, counts: Counter) -> bool:
    try:
        return entry.is_file(follow_symlinks=False)
    except (OSError, UnicodeError) as exc:
        counts["entry_errors"] += 1
        log.write(f"BAD FILE ENTRY: {safe_name(entry.path)}: {exc}")
        return False


def valid_calendar_date(year: str, month: str, day: str) -> bool:
    try:
        datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
        return True
    except ValueError:
        return False


def parse_valid_data_filename(
    name: str,
    station_filter: Optional[str],
    year: str,
    month: str,
    day: str,
    hour: str,
) -> bool:
    """
    Validate filename structure and also require the timestamp encoded in the
    filename to agree with the directory YYYY/MM/DD/HH containing it.
    """
    m = DATA_FILE_RE.fullmatch(name)
    if not m:
        return False

    if station_filter and m.group("station") != station_filter:
        return False

    date_text = m.group("date")
    hhmm = m.group("hhmm")
    sec = m.group("ss")

    try:
        dt = datetime.strptime(
            f"{date_text} {hhmm} {sec}",
            "%Y-%m-%d %H%M %S",
        )
    except ValueError:
        return False

    if dt.strftime("%Y") != year:
        return False
    if dt.strftime("%m") != month:
        return False
    if dt.strftime("%d") != day:
        return False
    if dt.strftime("%H") != hour:
        return False

    return True


def valid_histogram_filename(name: str) -> bool:
    m = HISTOGRAM_RE.fullmatch(name)
    if not m:
        return False
    try:
        datetime.strptime(m.group("date"), "%Y-%m-%d")
        return True
    except ValueError:
        return False


def determine_source_layout(source_arg: Path) -> Tuple[Path, Optional[Path], Path]:
    """
    Return:
        data_source, histogram_source_or_none, card_root

    Accept either:
        /Volumes/NO NAME
    or:
        /Volumes/NO NAME/data
    """
    source = source_arg.expanduser().resolve()

    if source.name == "data":
        data_source = source
        card_root = source.parent
        histogram_source = card_root / "histogram"
    elif (source / "data").is_dir():
        card_root = source
        data_source = source / "data"
        histogram_source = source / "histogram"
    else:
        raise ValueError(
            f"Could not find a Gecko data directory. Expected either:\n"
            f"  {source}/data\n"
            f"or for the supplied source itself to be named 'data'."
        )

    if not histogram_source.is_dir():
        histogram_source = None

    return data_source, histogram_source, card_root


def copy_one(
    src: Path,
    dst: Path,
    rel_display: str,
    args: argparse.Namespace,
    log: Logger,
    counts: Counter,
) -> None:
    """
    Safely copy one file.

    New/replacement data are written to a .partial file in the destination
    directory first and atomically renamed only after a complete size check.
    """
    try:
        src_size = src.stat().st_size
    except OSError as exc:
        counts["source_stat_errors"] += 1
        log.write(f"SOURCE STAT ERROR: {safe_name(str(src))}: {exc}")
        return

    if dst.exists():
        try:
            dst_size = dst.stat().st_size
        except OSError as exc:
            counts["destination_stat_errors"] += 1
            log.write(f"DEST STAT ERROR: {safe_name(str(dst))}: {exc}")
            return

        if dst_size == src_size:
            if args.verify:
                try:
                    source_hash = sha256_file(src)
                    dest_hash = sha256_file(dst)
                except OSError as exc:
                    counts["verify_errors"] += 1
                    log.write(f"VERIFY ERROR: {rel_display}: {exc}")
                    return

                if source_hash != dest_hash:
                    counts["checksum_mismatches"] += 1
                    log.write(
                        f"CHECKSUM MISMATCH: {rel_display} "
                        f"(same size, different SHA256)"
                    )
                    if not args.overwrite_mismatched:
                        return
                else:
                    counts["existing_verified"] += 1
                    if not args.quiet_existing:
                        print(f"EXISTS VERIFIED: {rel_display}")
                    return
            else:
                counts["existing_same_size"] += 1
                if not args.quiet_existing:
                    print(f"EXISTS: {rel_display}")
                return
        else:
            counts["size_mismatches"] += 1
            log.write(
                f"SIZE MISMATCH: {rel_display}: "
                f"source={src_size}, destination={dst_size}"
            )
            if not args.overwrite_mismatched:
                return

    if args.dry_run:
        action = "WOULD REPLACE" if dst.exists() else "WOULD COPY"
        print(f"{action}: {rel_display}")
        counts["would_copy"] += 1
        return

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        counts["destination_errors"] += 1
        log.write(f"MKDIR ERROR: {safe_name(str(dst.parent))}: {exc}")
        return

    tmp = dst.with_name(dst.name + ".partial")

    try:
        if tmp.exists():
            tmp.unlink()

        with src.open("rb") as fin, tmp.open("wb") as fout:
            shutil.copyfileobj(fin, fout, length=1024 * 1024)
            fout.flush()
            os.fsync(fout.fileno())

        copied_size = tmp.stat().st_size
        if copied_size != src_size:
            raise IOError(
                f"size mismatch after copy: source={src_size}, copied={copied_size}"
            )

        if args.verify:
            source_hash = sha256_file(src)
            copied_hash = sha256_file(tmp)
            if source_hash != copied_hash:
                raise IOError("SHA256 mismatch after copy")

        # Atomic rename on the destination filesystem.
        os.replace(tmp, dst)

        counts["copied"] += 1
        print(f"COPIED: {rel_display}")

    except (OSError, IOError) as exc:
        counts["copy_errors"] += 1
        log.write(f"COPY ERROR: {safe_name(str(src))}: {exc}")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def recover_data_tree(
    data_source: Path,
    destination: Path,
    args: argparse.Namespace,
    log: Logger,
    counts: Counter,
    seen_paths: set[str],
) -> None:
    """
    Traverse only a strict YYYY/MM/DD/HH tree. No generic recursive walk is used.
    This is intentional: malformed/corrupt directory names are never descended.
    """

    for year_entry in safe_scandir(data_source, log, counts):
        counts["directory_entries_seen"] += 1

        if not entry_is_dir(year_entry, log, counts):
            counts["rejected_entries"] += 1
            continue

        year = year_entry.name
        if not YEAR_RE.fullmatch(year):
            counts["rejected_directories"] += 1
            log.write(f"REJECT DIR: {safe_name(year_entry.path)}")
            continue

        year_path = Path(year_entry.path)

        for month_entry in safe_scandir(year_path, log, counts):
            counts["directory_entries_seen"] += 1

            if not entry_is_dir(month_entry, log, counts):
                counts["rejected_entries"] += 1
                continue

            month = month_entry.name
            if not MONTH_RE.fullmatch(month):
                counts["rejected_directories"] += 1
                log.write(f"REJECT DIR: {safe_name(month_entry.path)}")
                continue

            month_path = Path(month_entry.path)

            for day_entry in safe_scandir(month_path, log, counts):
                counts["directory_entries_seen"] += 1

                if not entry_is_dir(day_entry, log, counts):
                    counts["rejected_entries"] += 1
                    continue

                day = day_entry.name
                if not DAY_RE.fullmatch(day) or not valid_calendar_date(year, month, day):
                    counts["rejected_directories"] += 1
                    log.write(f"REJECT DIR: {safe_name(day_entry.path)}")
                    continue

                day_path = Path(day_entry.path)

                for hour_entry in safe_scandir(day_path, log, counts):
                    counts["directory_entries_seen"] += 1

                    if not entry_is_dir(hour_entry, log, counts):
                        counts["rejected_entries"] += 1
                        continue

                    hour = hour_entry.name
                    if not HOUR_RE.fullmatch(hour):
                        counts["rejected_directories"] += 1
                        log.write(f"REJECT DIR: {safe_name(hour_entry.path)}")
                        continue

                    hour_path = Path(hour_entry.path)

                    for file_entry in safe_scandir(hour_path, log, counts):
                        counts["file_entries_seen"] += 1

                        if not entry_is_file(file_entry, log, counts):
                            counts["rejected_entries"] += 1
                            continue

                        filename = file_entry.name

                        if not parse_valid_data_filename(
                            filename,
                            args.station,
                            year,
                            month,
                            day,
                            hour,
                        ):
                            counts["rejected_files"] += 1
                            log.write(f"REJECT FILE: {safe_name(file_entry.path)}")
                            continue

                        relative = Path(year, month, day, hour, filename)
                        key = relative.as_posix()

                        if key in seen_paths:
                            counts["duplicate_path_entries"] += 1
                            continue

                        seen_paths.add(key)
                        counts["valid_data_files"] += 1

                        copy_one(
                            Path(file_entry.path),
                            destination / relative,
                            key,
                            args,
                            log,
                            counts,
                        )


def recover_histograms(
    histogram_source: Optional[Path],
    destination: Path,
    args: argparse.Namespace,
    log: Logger,
    counts: Counter,
    seen_paths: set[str],
) -> None:
    if histogram_source is None:
        counts["histogram_directory_missing"] += 1
        log.write("NOTE: histogram/ directory not found; skipping histograms.")
        return

    for entry in safe_scandir(histogram_source, log, counts):
        counts["histogram_entries_seen"] += 1

        if not entry_is_file(entry, log, counts):
            counts["rejected_entries"] += 1
            continue

        if not valid_histogram_filename(entry.name):
            counts["rejected_histogram_files"] += 1
            log.write(f"REJECT HISTOGRAM: {safe_name(entry.path)}")
            continue

        relative = Path("histogram", entry.name)
        key = relative.as_posix()

        if key in seen_paths:
            counts["duplicate_path_entries"] += 1
            continue

        seen_paths.add(key)
        counts["valid_histogram_files"] += 1

        copy_one(
            Path(entry.path),
            destination / relative,
            key,
            args,
            log,
            counts,
        )


def print_summary(
    counts: Counter,
    data_source: Path,
    histogram_source: Optional[Path],
    destination: Path,
    log: Logger,
) -> None:
    log.write("")
    log.write("=" * 72)
    log.write("RECOVERY SUMMARY")
    log.write("=" * 72)
    log.write(f"Data source:      {data_source}")
    log.write(
        f"Histogram source: {histogram_source if histogram_source else '(not found)'}"
    )
    log.write(f"Destination:      {destination}")
    log.write("")

    order = [
        "valid_data_files",
        "valid_histogram_files",
        "copied",
        "would_copy",
        "existing_same_size",
        "existing_verified",
        "size_mismatches",
        "checksum_mismatches",
        "duplicate_path_entries",
        "rejected_directories",
        "rejected_files",
        "rejected_histogram_files",
        "rejected_entries",
        "scan_errors",
        "entry_errors",
        "source_stat_errors",
        "destination_stat_errors",
        "destination_errors",
        "verify_errors",
        "copy_errors",
    ]

    for key in order:
        if counts[key]:
            log.write(f"{key:28s}: {counts[key]}")

    # Always print the main totals, even if zero.
    log.write("")
    log.write(
        f"Accepted data files:          {counts['valid_data_files']}"
    )
    log.write(
        f"Accepted histogram CSV files: {counts['valid_histogram_files']}"
    )
    log.write(
        f"Duplicate pathname entries:   {counts['duplicate_path_entries']}"
    )
    log.write(
        f"Copy errors:                  {counts['copy_errors']}"
    )



def recover_sd_card(source, destination, *, station=None, dry_run=False,
                    verify=False, overwrite_mismatched=False, quiet_existing=False):
    """Copy Gecko SD-card files to a local download directory without modifying source.

    Accepts either SD-card root (containing data/) or the data/ directory.
    Preserves original recovery validation, partial-copy, SHA256 and logging logic.
    Returns counters and a status code (0 success, 1 recoverable scan/copy errors).
    """
    source_arg = Path(source).expanduser()
    target = Path(destination).expanduser().resolve()
    data_source, histogram_source, card_root = determine_source_layout(source_arg)
    if not data_source.is_dir():
        raise FileNotFoundError(data_source)
    if target == card_root or card_root in target.parents:
        raise ValueError('Destination must not be inside source SD card')
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(station=station, dry_run=dry_run, verify=verify,
                              overwrite_mismatched=overwrite_mismatched,
                              quiet_existing=quiet_existing)
    log = Logger(target / 'gecko_recovery.log', dry_run)
    counts = Counter()
    seen_paths = set()
    log.write('=' * 72)
    log.write('SiliconAudio Gecko SD recovery')
    log.write(f'Card root: {card_root}')
    log.write(f'Data source: {data_source}')
    log.write(f'Histogram source: {histogram_source or "(not found)"}')
    log.write(f'Destination: {target}')
    log.write(f'Station filter: {station or "(all stations)"}')
    log.write(f'Dry run: {dry_run}; SHA256 verify: {verify}; overwrite mismatched: {overwrite_mismatched}')
    recover_data_tree(data_source, target, args, log, counts, seen_paths)
    recover_histograms(histogram_source, target, args, log, counts, seen_paths)
    print_summary(counts, data_source, histogram_source, target, log)
    error_keys = ('copy_errors', 'scan_errors', 'entry_errors', 'source_stat_errors')
    return {'source': str(source_arg), 'destination': str(target), 'dry_run': dry_run,
            'status': 1 if any(counts[k] for k in error_keys) else 0,
            **dict(counts), 'log': str(target / 'gecko_recovery.log') if not dry_run else None}


def convert_to_sds(input_root, sds_root, **kwargs):
    """Alias for convert_siliconaudio, the second ingestion stage."""
    return convert_siliconaudio(input_root, sds_root, **kwargs)


def main(argv=None):
    """Standalone two-stage CLI: python -m flovopy.ingest.siliconaudio ..."""
    parser = argparse.ArgumentParser(description='SiliconAudio Gecko recovery and SDS staging')
    sub = parser.add_subparsers(dest='command', required=True)
    rec = sub.add_parser('recover', help='SD card -> local download')
    rec.add_argument('source')
    rec.add_argument('destination')
    rec.add_argument('--station')
    rec.add_argument('--dry-run', action='store_true')
    rec.add_argument('--verify', action='store_true')
    rec.add_argument('--overwrite-mismatched', action='store_true')
    rec.add_argument('--quiet-existing', action='store_true')
    arc = sub.add_parser('archive', help='local download -> staging SDS')
    arc.add_argument('source')
    arc.add_argument('destination')
    arc.add_argument('--year', type=int)
    arc.add_argument('--pattern', default='*.ms')
    arc.add_argument('--dry-run', action='store_true')
    arc.add_argument('--no-strict', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'recover':
        result = recover_sd_card(args.source, args.destination, station=args.station,
                                 dry_run=args.dry_run, verify=args.verify,
                                 overwrite_mismatched=args.overwrite_mismatched,
                                 quiet_existing=args.quiet_existing)
        return result['status']
    result = convert_to_sds(args.source, args.destination, year=args.year,
                            pattern=args.pattern, dry_run=args.dry_run,
                            strict=not args.no_strict)
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
