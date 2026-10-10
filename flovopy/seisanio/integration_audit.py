"""Read-only SEISAN catalog integration audit.

This module does not alter source files or depend on Montserrat-specific paths.
It compares QuakeML round-trip fields and records EnhancedEvent sidecar
metadata; waveform/SDS verification remains a separate conversion concern.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from obspy import read_events
from obspy.core.event import Catalog


def event_signature(event):
    """Stable structural counts and IDs for metadata round-trip comparisons."""
    return {
        'event_id': str(event.resource_id),
        'picks': sorted(str(p.resource_id) for p in event.picks),
        'origins': sorted(str(o.resource_id) for o in event.origins),
        'magnitudes': sorted(str(m.resource_id) for m in event.magnitudes),
        'arrivals_per_origin': sorted(len(o.arrivals) for o in event.origins),
        'event_type': str(event.event_type) if event.event_type is not None else None,
    }


def audit_catalog_roundtrip(catalog: Catalog):
    """Return discrepancies after writing/reading standard QuakeML."""
    with TemporaryDirectory() as folder:
        target = Path(folder) / 'roundtrip.xml'
        catalog.write(str(target), format='QUAKEML')
        restored = read_events(str(target))
    before = [event_signature(ev) for ev in catalog]
    after = [event_signature(ev) for ev in restored]
    return {'passed': before == after, 'before': before, 'after': after,
            'event_count': len(before)}


def audit_sfiles(paths, *, parser, enhanced=True):
    """Audit explicitly supplied S-files using the production parser API.

    ``parser`` is a class/callable accepting an S-file path. A parser's native
    ``to_enhancedevent`` is used when available. Failures remain visible.
    """
    from flovopy.seisanio.catalog import _enhance, _jsonable
    events, rows, failures = [], [], []
    for source in paths:
        path = str(source)
        try:
            parsed = parser(path)
            event = parsed.eventobj
            if event is None:
                raise ValueError('No ObsPy event parsed')
            record = _enhance(parsed) if enhanced else None
            events.append(event)
            row = {'source_path': path, 'event_id': str(event.resource_id),
                   'signature': event_signature(event)}
            if record is not None:
                row['enhanced_metadata'] = _jsonable(record.meta.to_json_dict())
            rows.append(row)
        except Exception as exc:
            failures.append({'source_path': path,
                             'error': f'{type(exc).__name__}: {exc}'})
    catalog = Catalog(events=events)
    try:
        roundtrip = audit_catalog_roundtrip(catalog)
    except Exception as exc:
        roundtrip = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
    return {'files_attempted': len(rows) + len(failures),
            'events_parsed': len(rows), 'failures': failures,
            'events': rows, 'quakeml_roundtrip': roundtrip}


def write_audit_json(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n',
                    encoding='utf-8')
    return path
