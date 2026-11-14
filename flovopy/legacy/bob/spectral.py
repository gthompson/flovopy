"""SSAM readings, source-documented band definitions and precomputed adapters."""
import re
import numpy as np
import pandas as pd
from .tabular import read_table


def read_ssam_band_definitions(documents, *, bands=16):
    """Parse archived BNnn low Hz to high Hz descriptions, with attribution.

    documents maps source names to text. Every band must be documented, and
    conflicting numeric definitions raise. Descriptive question marks remain
    verbatim rather than becoming confirmed physical interpretations.
    """
    definitions={}
    pattern=re.compile(r'^\s*BN(\d+)\s+([0-9.]+)\s*Hz\s+to\s+([0-9.]+)\s*Hz(.*)$', re.I)
    for source,text in documents.items():
        for line in text.splitlines():
            match=pattern.match(line)
            if not match: continue
            number=int(match[1])
            if not 1 <= number <= bands: continue
            low,high=float(match[2]),float(match[3])
            if not np.isfinite([low,high]).all() or not 0 <= low < high:
                raise ValueError(f'Invalid band limits in {source}: {line}')
            if number in definitions and (low,high)!=(definitions[number]['low_hz'],definitions[number]['high_hz']):
                raise ValueError(f'Conflicting BN{number:02d} limits in {source}')
            item=definitions.setdefault(number,dict(name=f'BN{number:02d}',low_hz=low,high_hz=high,sources=[]))
            item['sources'].append(dict(source=source,line=line,description=match[4].strip()))
    missing=set(range(1,bands+1))-set(definitions)
    if missing: raise ValueError(f'Missing documented SSAM bands: {sorted(missing)}')
    return [definitions[i] for i in range(1,bands+1)]


def read_ssam(source, *, bands=16, band_edges_hz=None, band_definitions=None, **kwargs):
    """Read positional band values, optionally using archived band descriptions.

    band_definitions='documents' explicitly parses source TXT documentation.
    A list of {name,low_hz,high_hz} dictionaries also supports overlapping bands.
    band_edges_hz remains supported for strictly contiguous, nonoverlapping bins.
    """
    kwargs.setdefault('expected_fields',bands)
    result=read_table(source,**kwargs)
    if band_edges_hz is not None and band_definitions is not None:
        raise ValueError('Choose band_edges_hz or band_definitions, not both')
    if band_edges_hz is not None:
        edges=np.asarray(band_edges_hz,dtype=float)
        if edges.shape!=(bands+1,) or not np.isfinite(edges).all() or (edges<0).any() or (np.diff(edges)<=0).any():
            raise ValueError('band_edges_hz must contain bands+1 finite increasing boundaries')
        result.metadata['band_edges_hz']=edges.tolist()
        band_definitions=[dict(name=f'BN{i+1:02d}',low_hz=edges[i],high_hz=edges[i+1]) for i in range(bands)]
    if isinstance(band_definitions,str):
        if band_definitions!='documents': raise ValueError("Use 'documents' or an explicit band-definition list")
        band_definitions=read_ssam_band_definitions(result.metadata.get('documents',{}),bands=bands)
    if band_definitions is not None:
        if len(band_definitions)!=bands: raise ValueError('One band definition is required per positional value')
        limits=np.array([[d['low_hz'],d['high_hz']] for d in band_definitions],dtype=float)
        names=[d['name'] for d in band_definitions]
        if (not np.isfinite(limits).all() or (limits[:,0]<0).any() or
            (limits[:,1]<=limits[:,0]).any() or len(set(names))!=bands):
            raise ValueError('Invalid or duplicated band definitions')
        result.metadata['band_definitions']=band_definitions
    result.metadata.update(observable='SSAM band values',bands=bands,
        band_policy='Explicit source definitions when supplied; no frequencies inferred from column count',
        quantity='reported_band_amplitude',units='arbitrary units; amplitude scaling unverified')
    return result


def ssam_to_spectrograms(dataset, *, interval_seconds, time_anchor='start', series=None, duplicate_policy='raise'):
    """Adapt SSAM to FLOVOpy precomputed band results, without FFT or resampling.

    interval_seconds and time_anchor are explicit display conventions until the
    acquisition support window is verified. Invalid times raise. duplicate_policy
    is 'raise' (default), or 'mask': preserve one timestamp with all bands missing
    and retain every conflicting record in metadata. Never average/select a
    preferred measurement silently; original dataset rows remain unchanged.
    """
    from flovopy.processing.spectrograms import BandSpectrogramResult
    definitions=dataset.metadata.get('band_definitions')
    if definitions is None: raise ValueError('Supply documented band definitions to read_ssam first')
    if duplicate_policy not in ('raise','mask'): raise ValueError('duplicate_policy must be raise or mask')
    frame=dataset.data
    selected=list(frame['series'].unique()) if series is None else ([series] if isinstance(series,str) else list(series))
    results=[]
    for name in selected:
        group=frame.loc[frame.series.eq(name)].copy()
        if group.empty: raise ValueError(f'No SSAM series: {name}')
        if group.recorded_time.isna().any(): raise ValueError(f'Invalid recorded times in {name}; review parser errors')
        duplicate=group.recorded_time.duplicated(keep=False)
        audit=group.loc[duplicate].to_dict('records')
        if duplicate.any() and duplicate_policy=='raise':
            raise ValueError(f'Duplicate times in {name}; review source versions or explicitly use duplicate_policy=mask')
        ambiguous_times=group.loc[duplicate,'recorded_time'].unique()
        group=group.sort_values('recorded_time',kind='stable')
        columns=[f'value_{i+1:02d}' for i in range(len(definitions))]
        missing=set(columns)-set(group)
        if missing: raise ValueError(f'Missing value columns: {missing}')
        if duplicate.any():
            group=group.drop_duplicates('recorded_time',keep='first').copy()
            group.loc[group.recorded_time.isin(ambiguous_times),columns]=np.nan
        metadata=dict(duplicate_policy=duplicate_policy,duplicate_time_records=audit,
            ambiguous_timestamp_count=len(ambiguous_times),source_format='legacy_BOB_SSAM',band_definitions=definitions,
            source_files=sorted(group.source_file.unique().tolist()),
            source_file_metadata=[f for f in dataset.metadata.get('files',[]) if f['source_file'] in set(group.source_file)]
                if dataset.metadata.get('files') and 'source_file' in dataset.metadata['files'][0] else dataset.metadata.get('files',[]),
            clock_correction_applied=False, averaging_window_seconds=None,
            display_interval_seconds=interval_seconds,
            support_window_status='Display convention, not an established acquisition averaging interval')
        results.append(BandSpectrogramResult(trace_id=str(name),times=group.recorded_time.to_numpy(),
            values=group[columns].to_numpy(dtype=float).T,
            band_limits_hz=[[d['low_hz'],d['high_hz']] for d in definitions],
            band_names=[d['name'] for d in definitions],interval_seconds=interval_seconds,
            time_anchor=time_anchor,time_basis=dataset.metadata.get('time_basis','unresolved recorded labels'),
            metadata=metadata))
    return results
