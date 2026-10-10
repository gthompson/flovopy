"""Nordic S-file catalog export with optional FLOVOpy enhanced metadata.

No database writes, waveform reads, or modifications to source S-files.
QuakeML contains standard ObsPy event fields; nonstandard MVO metadata
is exported to a separate JSONL sidecar.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from obspy.core.event import Catalog
from flovopy.seisanio.core.sfile import Sfile


@dataclass
class CatalogConversionResult:
    catalog: Catalog = field(default_factory=Catalog)
    records: list = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)


def _jsonable(value: Any):
    """Convert nested ObsPy / NumPy metadata to JSON without silent dropping."""
    from datetime import date, datetime
    from obspy import UTCDateTime
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (UTCDateTime, date, datetime)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'item'):
        return _jsonable(value.item())
    if hasattr(value, 'to_dict'):
        return _jsonable(value.to_dict())
    raise TypeError(f'Cannot serialize metadata type {type(value).__name__}')


def _enhance(parsed):
    """Wrap an existing ObsPy Event using the *current* EnhancedEvent API.

    Do not call MVOSfile.to_enhancedevent(): the supplied MVO implementation
    passes legacy kwargs unsupported by EnhancedEvent.__init__.
    """
    from flovopy.enhanced.event import EnhancedEvent, EnhancedEventMeta

    waveforms = [str(w.path) for w in getattr(parsed, 'wavfileobjs', [])
                 if w is not None and getattr(w, 'path', None)]
    # Prefer the parser’s native EnhancedEvent conversion where available.
    # This preserves specialized metadata without reimplementing it here.
    native = getattr(parsed, "to_enhancedevent", None)
    if callable(native):
        return native()
    aef = getattr(parsed, 'aeffileobj', None)
    metrics = {name: getattr(parsed, name, None) for name in
               ('filetime', 'mainclass', 'subclass', 'analyst', 'analyst_delay')}
    if aef is not None:
        metrics['aefrows'] = getattr(aef, 'aefrows', None)
    meta = EnhancedEventMeta(
        sfile_path=str(parsed.path), wav_paths=waveforms,
        aef_path=str(aef.path) if getattr(aef, 'path', None) else None,
        trigger_window=getattr(aef, 'trigger_window', None),
        average_window=getattr(aef, 'average_window', None),
        metrics=metrics,
    )
    return EnhancedEvent.wrap(parsed.eventobj, meta=meta)


def convert_archive_catalog(archive, starttime, endtime, *, db=None,
                            parser=Sfile, mainclass='*', enhanced=False,
                            strict=False):
    """Parse S-files once, returning catalog, enhanced records and failures.

    Parser defaults to generic Sfile; pass MVOSfile explicitly for MVO.
    Failures are reported rather than silently discarded. No data is written.
    """
    result = CatalogConversionResult()
    for path in archive.iter_sfiles(starttime, endtime, db=db,
                                    mainclass=mainclass):
        path = str(path)
        try:
            parsed = parser(path)
            event = getattr(parsed, 'eventobj', None)
            if event is None:
                raise ValueError('Parser returned no ObsPy Event')
            record = _enhance(parsed) if enhanced else None
            result.catalog.events.append(event)
            result.source_paths.append(path)
            if record is not None:
                result.records.append(record)
        except Exception as exc:
            result.failures.append({'path': path, 'error': f'{type(exc).__name__}: {exc}'})
            if strict:
                raise
    return result


def iter_parsed_sfiles(archive, starttime, endtime, *, db=None,
                       parser=Sfile, mainclass='*', strict=False):
    """Backward-compatible iterator over parsed S-files."""
    for path in archive.iter_sfiles(starttime, endtime, db=db,
                                    mainclass=mainclass):
        try:
            parsed = parser(str(path))
            if getattr(parsed, 'eventobj', None) is None:
                raise ValueError(f'No ObsPy Event parsed from {path}')
            yield parsed
        except Exception:
            if strict:
                raise


def seisan_to_catalog(archive, starttime, endtime, *, db=None,
                      parser=Sfile, mainclass='*', strict=False):
    """Backward-compatible ObsPy Catalog conversion."""
    return convert_archive_catalog(
        archive, starttime, endtime, db=db, parser=parser,
        mainclass=mainclass, strict=strict).catalog


def write_quakeml(catalog, path):
    """Write standards-compliant QuakeML; never overwrite source S-files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(path), format='QUAKEML')
    return path


def write_catalog_result(result, quakeml_path, *, metadata_path=None,
                         failures_path=None):
    """Export QuakeML and optional JSONL sidecars.

    Metadata sidecar requires result.records (enhanced=True at conversion).
    QuakeML cannot preserve MVO AEF and other nonstandard fields.
    """
    if metadata_path is not None and len(result.records) != len(result.catalog.events):
        raise ValueError('Metadata sidecar requires enhanced=True for all events')
    if metadata_path is not None:
        rows = []
        for event, record in zip(result.catalog.events, result.records):
            rows.append({'event_id': str(event.resource_id),
                         'metadata': _jsonable(record.meta.to_json_dict())})
        payload = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows)
    else:
        payload = None
    write_quakeml(result.catalog, quakeml_path)
    if metadata_path is not None:
        p = Path(metadata_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload, encoding='utf-8')
    if failures_path is not None:
        p = Path(failures_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(json.dumps(row) + '\n' for row in result.failures), encoding='utf-8')
    return Path(quakeml_path)


def to_enhanced_catalog(result):
    """Create an EnhancedCatalog with events and records sharing identity.

    Requires an enhanced conversion so no metadata is silently lost.
    """
    from flovopy.enhanced.catalog import EnhancedCatalog
    if len(result.records) != len(result.catalog.events):
        raise ValueError("EnhancedCatalog requires enhanced=True for all events")
    return EnhancedCatalog(events=list(result.records), records=list(result.records))
