"""Strict scientific JSON and non-lossy metric-table cells.

Missing/nonfinite numbers become JSON null. Numeric arrays become nested JSON
arrays, never their truncated string representations. No pickle is used.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
import json
import math
import os
import tempfile
import numpy as np
import pandas as pd
from obspy import UTCDateTime


def json_compatible(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ma.MaskedArray):
        data=np.asarray(value.data,dtype=object).copy()
        data[np.ma.getmaskarray(value)]=None
        return json_compatible(data.tolist())
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {'__flovopy_type__':'complex','real':json_compatible(value.real),'imag':json_compatible(value.imag)}
    if isinstance(value, UTCDateTime):
        return str(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_compatible(asdict(value))
    if isinstance(value, pd.DataFrame):
        return json_compatible(value.to_dict('records'))
    if isinstance(value, Mapping):
        out={}
        for key,item in value.items():
            key=str(key)
            if key in out:raise ValueError('JSON key collision: '+key)
            out[key]=json_compatible(item)
        return out
    if isinstance(value,(list,tuple)):
        return [json_compatible(item) for item in value]
    raise TypeError(f'Unsupported JSON metadata type: {type(value).__name__}')


def atomic_json(path, payload):
    """Serialize first, then atomically replace one sidecar in its directory."""
    path=Path(path);text=json.dumps(json_compatible(payload),indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            handle.write(text);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def flatten_metrics(payload, prefix=''):
    """Flatten mappings; arrays/lists remain complete JSON strings in CSV cells.

    Dotted keys are retained for familiar metric names. Ambiguous dotted-key
    paths raise instead of silently overwriting a value. JSON is authoritative.
    """
    clean=json_compatible(payload);out={}
    def walk(value,path):
        if isinstance(value,dict) and value:
            for key,item in value.items():walk(item,f'{path}.{key}' if path else key)
        else:
            if path in out:raise ValueError('Ambiguous flattened metric path: '+path)
            out[path]=json.dumps(value,allow_nan=False,separators=(',',':')) if isinstance(value,(list,dict)) else value
    walk(clean,prefix)
    return out
