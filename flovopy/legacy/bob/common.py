"""Input access and common record policies; ZIPs are read without extraction."""
from pathlib import Path, PurePosixPath
import hashlib
import zipfile
import numpy as np
import pandas as pd
from .model import BOBDataset

MONTHS = {name: i for i, name in enumerate('JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC'.split(), 1)}

def input_members(source):
    """Yield (relative name, bytes) from one file, a directory, or a ZIP."""
    source = Path(source).expanduser()
    if source.is_dir():
        for path in sorted(source.rglob('*')):
            if path.is_file() and not any(part.startswith('.') or part=='__MACOSX' for part in path.relative_to(source).parts):
                yield path.relative_to(source).as_posix(), path.read_bytes()
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for member in sorted(archive.infolist(), key=lambda x: x.filename):
                if not member.is_dir() and not any(p.startswith('.') or p=='__MACOSX' for p in PurePosixPath(member.filename).parts):
                    yield member.filename, archive.read(member)
    elif source.is_file():
        yield source.name, source.read_bytes()
    else:
        raise FileNotFoundError(source)

def file_metadata(name, blob):
    return dict(source_file=name, source_sha256=hashlib.sha256(blob).hexdigest(), source_bytes=len(blob))

def finish(rows, metadata, errors, *, missing_values, strict=False):
    """Add masks/usable columns without discarding raw values or duplicate rows."""
    data = pd.DataFrame(rows)
    if data.empty:
        data = pd.DataFrame(columns=['series','source_file','source_line','recorded_time','raw_line'])
    data['recorded_time'] = pd.to_datetime(data['recorded_time'], errors='coerce')
    for column in list(data.columns):
        if column.startswith('value_') and column.endswith('_raw'):
            raw = pd.to_numeric(data[column], errors='coerce')
            missing = ~np.isfinite(raw) | raw.isin(missing_values)
            base = column[:-4]
            data[base + '_missing'] = missing
            data[base] = raw.mask(missing)
    data['duplicate_timestamp'] = data.duplicated(['series','recorded_time'], keep=False) & data['recorded_time'].notna()
    metadata.update(missing_values=list(missing_values), time_basis=metadata.get('time_basis','unresolved recorded calendar labels'),
                    clock_correction_applied=False, units='Uncalibrated source values unless explicitly supplied by caller',
                    duplicate_policy='Preserved and flagged within each series; not averaged or dropped')
    error_table = pd.DataFrame(errors, columns=['source_file','source_line','raw_line','reason'])
    if strict and len(error_table):
        raise ValueError(f'{len(error_table)} BOB parse error(s); use strict=False to inspect retained rows and error records.')
    return BOBDataset(data, metadata, error_table)
