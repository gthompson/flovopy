"""Catalog-to-REA export using ObsPy's public Nordic writer.

Originless-event anchors and dataset comments are caller policies. A manifest
makes reruns idempotent for stable event IDs, verifies reused output hashes,
and records standard-filename collision shifts. Do not run concurrent writers
against the same root/manifest.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json
import os
import tempfile

from obspy import UTCDateTime
from obspy.core.event import Catalog, Origin, ResourceIdentifier


def _atomic_json(path, data):
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix='.rea-manifest-', delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write((json.dumps(data, indent=2, sort_keys=True)+'\n').encode())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(event, parameters):
    buffer = io.BytesIO()
    Catalog(events=[event], resource_id=ResourceIdentifier('smi:local/flovopy/fingerprint')).write(buffer, format='QUAKEML')
    return hashlib.sha256(buffer.getvalue()+json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def export_catalog_to_rea(
    catalog, rea_root, *, database=None, author='OBSP', event_type='L',
    nordic_format='OLD', high_accuracy=True, waveform_files=None,
    anchor_time=None, prepare_event=None, collision='error', max_shift_seconds=30,
    overwrite=False, continue_on_error=True, write_select=False,
):
    """Export an ObsPy Catalog to [database/]YYYY/MM standard Nordic S-files.

    waveform_files(event) -> list[str] supplies waveform links (default none).
    anchor_time(event) -> (UTCDateTime, source_label) or None is used ONLY if
    no existing origin has a time. The caller chooses that policy; no generic
    pick/waveform time inference is performed. Derived anchors have no invented
    coordinates/depth and deterministic, event-specific resource IDs.
    prepare_event(copy) -> Event optionally adds dataset-specific metadata.
    Input events are never mutated. Timeless events are reported as skipped.

    collision='error' reports events needing the same second's filename.
    'shift' tries later filename seconds, up to max_shift_seconds; origin/pick
    times are unchanged. The shift is recorded in the returned rows/manifest.
    Only overwrite=True permits replacing this exporter's own tracked changed
    event. Untracked or other-event files are never overwritten. Stable event
    IDs and identical export settings are required for reuse on reruns.

    Return written/reused paths, skipped/failed records, per-event rows and
    manifest_path. The manifest contains the exact SHA256 of each output.
    Optional select_all.out concatenates verified successful export files.
    """
    if nordic_format not in ('OLD', 'NEW') or (not callable(event_type) and event_type not in ('L','R','D')):
        raise ValueError('Invalid Nordic format or event type')
    if not isinstance(author, str) or len(author)>4:
        raise ValueError('author must have at most four characters')
    if collision not in ('error', 'shift'):
        raise ValueError("collision must be 'error' or 'shift'")
    if not isinstance(max_shift_seconds, int) or isinstance(max_shift_seconds, bool) or max_shift_seconds<0:
        raise ValueError('max_shift_seconds must be a nonnegative integer')
    if database is not None and (not database or Path(database).name!=database or database in ('.','..')):
        raise ValueError('database must be a single directory name')
    root = Path(rea_root).expanduser().resolve()
    if database is not None:
        root /= database
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'.flovopy_rea_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'version':1,'events':{}}
    if manifest.get('version')!=1 or not isinstance(manifest.get('events'),dict):
        raise ValueError('Unsupported REA manifest')
    entries = manifest['events']
    def tracked_path(entry):
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path escapes REA root')
        return path
    owners = {tracked_path(entry):key for key,entry in entries.items()}
    keys=[str(event.resource_id) for event in catalog]
    if len(keys)!=len(set(keys)):
        raise ValueError('Catalog must have distinct event resource IDs')
    result=dict(written=[],reused=[],skipped=[],failed=[],rows=[],manifest_path=manifest_path)
    for original,key in zip(catalog,keys):
        try:
            event=original.copy()
            if prepare_event is not None:
                event=prepare_event(event)
                if event is None:
                    raise ValueError('prepare_event must return an Event')
            preferred=event.preferred_origin()
            origin=preferred if preferred is not None and preferred.time is not None else next(
                (o for o in event.origins if o.time is not None),None)
            source='existing_origin'
            if origin is None and anchor_time is not None:
                anchor=anchor_time(event)
                if anchor is not None:
                    time,source=anchor
                    uid=hashlib.sha256(key.encode()).hexdigest()
                    origin=Origin(time=UTCDateTime(time),resource_id=ResourceIdentifier('smi:local/flovopy/anchor/'+uid))
                    event.origins.append(origin)
            if origin is None:
                result['skipped'].append(dict(event_id=key,reason='no_time_anchor'))
                continue
            event.preferred_origin_id=origin.resource_id
            evtype = event_type(event) if callable(event_type) else event_type
            if evtype not in ('L', 'R', 'D'):
                raise ValueError('Invalid event type')
            waves=list(waveform_files(event)) if waveform_files is not None else []
            if any(not isinstance(w,str) for w in waves):
                raise ValueError('waveform_files must supply strings')
            waves=list(dict.fromkeys(Path(w).name for w in waves if w))
            parameters=dict(author=author,event_type=evtype,nordic_format=nordic_format,
                            high_accuracy=high_accuracy,waveforms=waves,collision=collision,
                            max_shift_seconds=max_shift_seconds)
            fingerprint=_fingerprint(event,parameters)
            old=entries.get(key)
            target=None;shift=0
            if old is not None:
                previous=tracked_path(old)
                if old['fingerprint']==fingerprint and previous.is_file():
                    if hashlib.sha256(previous.read_bytes()).hexdigest()==old['sha256']:
                        result['reused'].append(previous)
                        result['rows'].append(dict(event_id=key,path=str(previous),status='reused',
                            origin_source=old['origin_source'],filename_offset_s=old['filename_offset_s']))
                        continue
                    if not overwrite:
                        raise ValueError('Tracked output was modified; explicit overwrite required')
                elif old['fingerprint']!=fingerprint and not overwrite:
                    raise ValueError('Event/settings changed; explicit overwrite required')
                if old.get('origin_time') != str(origin.time):
                    raise ValueError('Origin time changed; use a new export root to avoid stale filenames')
                if previous.exists():
                    # Keep the already assigned standard filename for updates.
                    target=previous;shift=old['filename_offset_s']
            if target is None:
                for shift in range((max_shift_seconds if collision=='shift' else 0)+1):
                    time=origin.time+shift
                    name=f'{time.day:02d}-{time.hour:02d}{time.minute:02d}-{time.second:02d}{evtype}.S{time.year:04d}{time.month:02d}'
                    candidate=root/f'{time.year:04d}'/f'{time.month:02d}'/name
                    if candidate not in owners and not candidate.exists():
                        target=candidate;break
                if target is None:
                    raise FileExistsError('No free standard S-file name within collision policy')
            target.parent.mkdir(parents=True,exist_ok=True)
            handle=tempfile.NamedTemporaryFile(dir=target.parent,prefix='.nordic-',delete=False)
            temporary=Path(handle.name);handle.close()
            try:
                # Public writer writes only the requested file. No os.chdir.
                Catalog(events=[event]).write(str(temporary),format='NORDIC',userid=author,
                    evtype=evtype,wavefiles=[waves],nordic_format=nordic_format,
                    high_accuracy=high_accuracy)
                if not temporary.stat().st_size:
                    raise OSError('Nordic writer produced an empty file')
                if target.exists() and owners.get(target)!=key:
                    raise FileExistsError(f'Output belongs to another event: {target}')
                if target.exists():
                    os.replace(temporary,target)
                else:
                    # Exclusive creation avoids clobbering an untracked competing file.
                    os.link(temporary,target)
            finally:
                temporary.unlink(missing_ok=True)
            entries[key]=dict(path=str(target.relative_to(root)),fingerprint=fingerprint,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),origin_source=source,
                filename_offset_s=shift,origin_time=str(origin.time))
            owners[target]=key
            _atomic_json(manifest_path,manifest)
            result['written'].append(target)
            result['rows'].append(dict(event_id=key,path=str(target),status='written',
                                      origin_source=source,filename_offset_s=shift))
        except Exception as exc:
            result['failed'].append(dict(event_id=key,error=f'{type(exc).__name__}: {exc}'))
            if not continue_on_error:
                raise
    if write_select:
        select=root/'select_all.out'
        # Only include verified outputs for this invocation; do not re-encode events.
        content='\n'.join(Path(row['path']).read_text() for row in result['rows'])
        temp=tempfile.NamedTemporaryFile(dir=root,prefix='.select-',delete=False)
        tmp=Path(temp.name)
        try:
            with temp:temp.write(content.encode())
            os.replace(tmp,select)
        finally:
            tmp.unlink(missing_ok=True)
        result['select_path']=select
    return result
