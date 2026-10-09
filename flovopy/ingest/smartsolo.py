"""SmartSolo: stage SoloLite/Harvester MiniSEED as SDS.

Raw DLD extraction and MiniSEED conversion are performed externally with
the vendor tools. This module never modifies the source MiniSEED.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from obspy import Stream, UTCDateTime, read

from flovopy.enhanced.sdsclient import EnhancedSDSClient


def load_mapping(mapping_csv):
    """Load serial,seed_prefix, e.g. 453013783,1R.B14..DH."""
    result = {}
    with Path(mapping_csv).expanduser().open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"serial", "seed_prefix"}.issubset(reader.fieldnames or []):
            raise ValueError("Mapping CSV requires serial,seed_prefix columns")
        for row in reader:
            serial = row["serial"].strip()
            prefix = row["seed_prefix"].strip()
            parts = prefix.split(".")
            if not serial.isdigit() or len(parts) != 4 or not parts[0] or not parts[1] or len(parts[3]) != 2:
                raise ValueError(f"Invalid mapping row: {row}")
            if serial in result:
                raise ValueError(f"Duplicate SmartSolo serial: {serial}")
            result[serial] = prefix
    if not result:
        raise ValueError("Empty SmartSolo mapping")
    return result


def _serial_from_trace(tr, filename, mapping):
    """Resolve either full serial or vendor short station code, checking filename."""
    station = str(tr.stats.station).strip()
    candidates = [station]
    if station.isdigit() and not station.startswith("4530"):
        candidates.append("4530" + station)
    first = filename.name.split(".")[0]
    if first.isdigit():
        candidates.append(first)
    hits = {s for s in candidates if s in mapping}
    if len(hits) != 1:
        raise ValueError(f"Cannot uniquely map station={station!r} filename={filename.name!r}: {sorted(hits)}")
    return hits.pop()


def _component(tr, filename):
    """Require E/N/Z from channel or filename, rejecting conflicting evidence."""
    ch = str(tr.stats.channel).upper()
    from_channel = ch[-1] if ch and ch[-1] in "ENZ" else None
    match = re.search(r"\\.([ENZ])\\.miniseed$", filename.name, re.IGNORECASE)
    from_filename = match.group(1).upper() if match else None
    if from_channel and from_filename and from_channel != from_filename:
        raise ValueError(f"Channel/filename component mismatch: {ch}, {filename.name}")
    component = from_channel or from_filename
    if component is None:
        raise ValueError(f"Cannot identify E/N/Z component in {filename.name}")
    return component


def stage_smartsolo_mseed(input_root, sds_root, *, mapping_csv, pattern="*.miniseed",
                         recursive=False, dry_run=False, strict=True, verbose=True,
                         write_mode="merge"):
    """Group daily files by NET.STA.LOC.CHA and UTC day, then write staging SDS.

    The input is vendor-converted daily MiniSEED, not raw DLD.
    With strict=True (default), any bad file in a day prevents that day's write.
    Existing staging SDS is merged by EnhancedSDSClient. Never point sds_root
    at the master archive; merge staging into master as a separate step.
    """
    source = Path(input_root).expanduser().resolve()
    target = Path(sds_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target == source or source in target.parents:
        raise ValueError("Staging SDS must not be inside the input directory")
    mapping = load_mapping(mapping_csv)
    files = sorted(source.rglob(pattern) if recursive else source.glob(pattern))
    counts = Counter(files_found=len(files))
    errors = []
    daily = defaultdict(Stream)
    bad_days = set()

    for path in files:
        try:
            if not path.stat().st_size:
                counts["empty_files"] += 1
                continue
            # Full waveform read: headonly=True would produce no usable samples.
            st = read(str(path), format="MSEED")
            if not st:
                raise ValueError("No traces")
            for tr in st:
                serial = _serial_from_trace(tr, path, mapping)
                component = _component(tr, path)
                tr.id = mapping[serial] + component
                # Split at UTC day boundaries even if a file crosses midnight.
                start_day = UTCDateTime(tr.stats.starttime.date)
                end_day = UTCDateTime(tr.stats.endtime.date)
                day = start_day
                while day <= end_day:
                    # inclusive last sample for this UTC day
                    piece = tr.slice(starttime=day, endtime=day + 86400 - tr.stats.delta,
                                     nearest_sample=False)
                    if piece.stats.npts:
                        daily[(tr.id, str(day.date))] += piece
                    day += 86400
                counts["traces_read"] += 1
            counts["files_read"] += 1
        except Exception as exc:
            counts["read_or_mapping_errors"] += 1
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            # Filename may identify affected date, but without reliable metadata
            # do not risk writing partial output in strict mode.
            if strict:
                raise ValueError(errors[-1]) from exc

    client = None if dry_run else EnhancedSDSClient(str(target))
    for (seed_id, day), st in sorted(daily.items()):
        try:
            st.sort(keys=["starttime"])
            st.merge(method=0, fill_value=None)
            if not dry_run:
                client.write_stream(st, mode=write_mode, preprocess=False, verbose=False)
                counts["daily_groups_written"] += 1
            else:
                counts["daily_groups_validated"] += 1
            if verbose:
                print(f"{day} {seed_id}: {len(st)} continuous segment(s)")
        except Exception as exc:
            counts["write_errors"] += 1
            errors.append(f"{day} {seed_id}: {type(exc).__name__}: {exc}")
            if strict:
                raise
    return {"source": str(source), "staging_sds": str(target),
            "dry_run": dry_run, "counts": dict(counts), "errors": errors}
