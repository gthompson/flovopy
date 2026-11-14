"""Validated annual float32 layout shared by some BOB products, not their meanings."""
import calendar
import re
import struct
from pathlib import PurePosixPath
import numpy as np
import pandas as pd
from .common import input_members, file_metadata, finish

def read_annual(source, *, year=None, interval_seconds=600, header_bytes=576,
                byte_order='<', missing_values=(-998,), century=1900,
                time_basis='unresolved recorded calendar labels', channel_metadata=None, strict=True):
    """Read full-year arrays with a leading header-length word.

    Calendar phase is the chosen Jan-1 + n*interval convention, not verified UTC.
    No polarity, amplitude, calibration or counter transformation is applied.
    station/channel interpretations can be passed explicitly in channel_metadata.
    """
    if interval_seconds <= 0 or 86400 % interval_seconds:
        raise ValueError('interval_seconds must divide one day exactly.')
    if byte_order not in ('<','>'):
        raise ValueError('byte_order must be < or >.')
    frames, errors, files, docs = [], [], [], {}
    for name, blob in input_members(source):
        if name.upper().endswith('.TXT'):
            docs[name] = blob.decode('cp1252', errors='replace'); continue
        if not name.upper().endswith('.DAT'):
            continue
        try:
            stem = PurePosixPath(name).stem
            match = re.fullmatch(r'(.+?)(\d{2})', stem)
            file_year = year if year is not None else (century + int(match[2]) if match else None)
            if file_year is None:
                raise ValueError('Cannot infer year; pass year explicitly.')
            if match and (century + int(match[2])) != file_year:
                raise ValueError('Filename year conflicts with requested year/century.')
            count = int((366 if calendar.isleap(file_year) else 365)*86400/interval_seconds)
            if len(blob) < header_bytes or struct.unpack(byte_order+'I', blob[:4])[0] != header_bytes:
                raise ValueError('Unrecognized header-length word.')
            if len(blob) != header_bytes + count*4:
                raise ValueError('File length is not the expected complete annual float32 layout.')
            raw = np.frombuffer(blob, dtype=byte_order+'f4', offset=header_bytes).copy()
            series = (PurePosixPath(name).parent / (match[1] if match else stem)).as_posix()
            frame = pd.DataFrame(dict(series=series, source_file=name, sample_index=np.arange(count),
                recorded_time=pd.date_range(f'{file_year}-01-01', periods=count, freq=pd.to_timedelta(interval_seconds, unit='s')),
                value_01_raw=raw))
            frames.append(frame)
            files.append(dict(**file_metadata(name,blob), year=file_year, series=series, header_hex=blob[:header_bytes].hex()))
        except (ValueError, struct.error) as exc:
            errors.append(dict(source_file=name,source_line=None,raw_line='',reason=str(exc)))
    if not frames:
        if errors: raise ValueError(f'No valid annual series: {errors[0]}')
        raise ValueError('No annual .DAT files found.')
    # finish accepts record dictionaries; annual arrays use a vectorized route to avoid huge temporary dictionaries.
    from .model import BOBDataset
    data = pd.concat(frames, ignore_index=True)
    raw = data['value_01_raw']
    missing = ~np.isfinite(raw) | raw.isin(missing_values)
    data['value_01_missing'] = missing
    data['value_01'] = raw.mask(missing)
    data['duplicate_timestamp'] = data.duplicated(['series','recorded_time'],keep=False)
    error_frame = pd.DataFrame(errors,columns=['source_file','source_line','raw_line','reason'])
    if strict and len(error_frame): raise ValueError(error_frame.to_dict('records'))
    return BOBDataset(data,dict(format='annual_float32', files=files, documents=docs,
        time_basis=time_basis, clock_correction_applied=False, interval_seconds=interval_seconds,
        header_bytes=header_bytes, byte_order=byte_order, missing_values=list(missing_values),
        units='Uncalibrated source values', channel_metadata=dict(channel_metadata or {}),
        calendar_policy='Jan-1 plus sample index * interval; slot phase must be verified independently'),error_frame)

def read_rsam(source, **kwargs):
    result=read_annual(source,**kwargs);result.metadata['observable']='RSAM';return result

def read_rsam_events(source, **kwargs):
    result=read_annual(source,**kwargs);result.metadata['observable']='RSAM event counter (meaning of counts requires source documentation)';return result

def read_tilt(source, **kwargs):
    result=read_annual(source,**kwargs);result.metadata['observable']='Tilt/housekeeping (channel meanings supplied by caller)';return result
