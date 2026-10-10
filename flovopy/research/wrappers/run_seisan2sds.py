"""Convert SEISAN WAV archives to SDS using :class:`EnhancedSDSClient`.

Days in ``start``..``end`` are inclusive UTC calendar days.  A file starting
on the preceding day can contribute samples to the requested day.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterator

from obspy import Stream, UTCDateTime, read

from flovopy.enhanced.sdsclient import EnhancedSDSClient
from flovopy.core.trace_utils import fix_trace_mvo

LOG = logging.getLogger(__name__)
SECONDS_PER_DAY = 86400


def _midnight(value) -> UTCDateTime:
    t = UTCDateTime(value)
    return UTCDateTime(t.year, t.month, t.day)


def _days(start, end) -> Iterator[UTCDateTime]:
    day, last = _midnight(start), _midnight(end)
    if last < day:
        raise ValueError("end precedes start")
    while day <= last:
        yield day
        day += SECONDS_PER_DAY


def _files_for_day(wavroot: Path, day: UTCDateTime) -> list[Path]:
    """Include previous-day files starting at 23:40–23:59 UTC."""
    previous = day - SECONDS_PER_DAY
    previous_dir = wavroot / previous.strftime("%Y/%m")
    current_dir = wavroot / day.strftime("%Y/%m")
    previous_pattern = previous.strftime("%Y-%m-%d") + "-23[45]*S.MVO___*"
    current_pattern = day.strftime("%Y-%m-%d") + "*S.MVO___*"
    return sorted(set(previous_dir.glob(previous_pattern)) | set(current_dir.glob(current_pattern)))


def _clip_to_day(stream: Stream, day: UTCDateTime) -> Stream:
    """Retain samples whose actual timestamps fall in [day, day+86400)."""
    stop = day + SECONDS_PER_DAY
    output = Stream()
    for original in stream:
        if not original.stats.npts or original.stats.sampling_rate <= 0:
            continue
        tr = original.copy()
        # Trim to the requested day.  An exclusive end is enforced below.
        tr.trim(starttime=day, endtime=stop, nearest_sample=False, pad=False)
        if not tr.stats.npts:
            continue
        # ObsPy trim endpoints are inclusive; explicitly drop a sample at midnight.
        if tr.stats.endtime >= stop:
            tr.trim(endtime=stop - tr.stats.delta, nearest_sample=False, pad=False)
        if tr.stats.npts and tr.stats.starttime >= day and tr.stats.endtime < stop:
            output.append(tr)
    return output


def _export_datascope(sdsdir: Path, dbout: str, day: UTCDateTime) -> None:
    year, jday = day.strftime("%Y"), day.strftime("%j")
    # SDS layout: YEAR/NET/STA/CHAN.D/NET.STA.LOC.CHAN.D.YEAR.JDAY
    files = sorted(sdsdir.glob(f"{year}/*/*/*.D/*.{year}.{jday}"))
    if not files:
        LOG.info("No SDS files for %s; skipping Datascope export", day.date)
        return
    destination = f"{dbout}{day.strftime('%Y%m%d')}"
    subprocess.run(["miniseed2db", *(str(p) for p in files), destination], check=True)


def seisan_to_sds(
    seisandbdir,
    sdsdir,
    startt0,
    endt0,
    net,
    dbout=None,
    round_sampling_rate=True,
    MBWHZ_only=False,
    *,
    database="DSNC_",
    skip_unreadable=True,
):
    """Import SEISAN waveforms into SDS, merging existing day files.

    Parameters are compatible with the legacy wrapper. ``round_sampling_rate``
    is accepted but deliberately ignored: changing the sampling rate requires
    explicit resampling. ``database`` selects the SEISAN ``WAV`` subdirectory.

    Returns a count dictionary for logging/verification.
    """
    if not net:
        raise ValueError("net must be a nonempty SEED network code")
    wavroot = Path(seisandbdir).expanduser() / "WAV" / database
    if not wavroot.is_dir():
        raise FileNotFoundError(f"SEISAN WAV database not found: {wavroot}")
    sdsroot = Path(sdsdir).expanduser()
    sdsroot.mkdir(parents=True, exist_ok=True)
    client = EnhancedSDSClient(str(sdsroot))
    counts = Counter(days=0, files=0, unreadable=0, traces=0)
    if round_sampling_rate:
        LOG.debug("round_sampling_rate is a legacy no-op; sample rates are unchanged")

    for day in _days(startt0, endt0):
        counts["days"] += 1
        for path in _files_for_day(wavroot, day):
            try:
                stream = read(str(path), format="SEISAN")
            except Exception:
                counts["unreadable"] += 1
                if not skip_unreadable:
                    raise
                LOG.warning("Skipping unreadable SEISAN file: %s", path, exc_info=True)
                continue
            counts["files"] += 1
            if net == "MV":
                for trace in stream:
                    fix_trace_mvo(trace, legacy=False, netcode=net)
            if MBWHZ_only:
                stream = stream.select(station="MBWH", component="Z")
            stream = _clip_to_day(stream, day)
            if stream:
                client.write_stream(stream, mode="merge", preprocess=False)
                counts["traces"] += len(stream)
        if dbout:
            _export_datascope(sdsroot, str(dbout), day)
        LOG.info("Completed SEISAN import for %s", day.strftime("%Y-%m-%d"))
    return dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First UTC calendar day (inclusive)")
    parser.add_argument("--end", required=True, help="Last UTC calendar day (inclusive)")
    parser.add_argument("--seisan", required=True, help="SEISAN database root")
    parser.add_argument("--sds", required=True, help="Output SDS root")
    parser.add_argument("--net", required=True, help="Network code (MV applies MVO trace fixes)")
    parser.add_argument("--database", default="DSNC_", help="SEISAN WAV database (default: DSNC_)")
    parser.add_argument("--dbout", help="Optional Datascope output prefix (requires miniseed2db)")
    parser.add_argument("--round_sampling_rate", action="store_true", help="Legacy no-op")
    parser.add_argument("--MBWHZ_only", action="store_true", help="Only MBWH vertical traces")
    parser.add_argument("--strict", action="store_true", help="Fail on unreadable SEISAN files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    counts = seisan_to_sds(
        args.seisan, args.sds, args.start, args.end, args.net,
        dbout=args.dbout, round_sampling_rate=args.round_sampling_rate,
        MBWHZ_only=args.MBWHZ_only, database=args.database,
        skip_unreadable=not args.strict,
    )
    LOG.info("Import summary: %s", counts)
    return counts


if __name__ == "__main__":
    main()
