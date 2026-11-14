"""Dated ASCII monitoring tables; columns remain positional unless documented."""
from datetime import datetime
from pathlib import PurePosixPath
import re
from .common import MONTHS, input_members, file_metadata, finish

RECORD = re.compile(r'^\s*(\d{1,2})[ -]([A-Za-z]{3})[ -](\d{4}|\d{2})\s+(\d{1,2})[:. ]\s*(\d{1,2})\s*(.*?)\s*$')

def read_table(source, *, expected_fields=None, missing_values=(-998,-999), century=1900,
               encoding='cp1252', strict=False, time_basis='unresolved recorded calendar labels',
               series_from_path=None, allow_empty_value=False):
    """Read DAT files from a file, ZIP or recursive directory.

    Numeric rows with invalid times remain in the table with NaT and errors;
    structurally malformed lines stay in errors with their original source line.
    No interpolation, aggregation, deduplication or timezone correction is done.
    """
    rows, errors, files, documents = [], [], [], {}
    for name, blob in input_members(source):
        if name.upper().endswith('.TXT'):
            documents[name] = blob.decode(encoding,errors='replace');continue
        if not name.upper().endswith('.DAT'):continue
        file = file_metadata(name,blob);headers=[]
        series = series_from_path(name) if series_from_path else PurePosixPath(name).parent.as_posix()
        for lineno, raw in enumerate(blob.decode(encoding,errors='replace').splitlines(),1):
            clean=raw.replace('\x1a','').strip()
            if not clean:continue
            match=RECORD.fullmatch(clean)
            if not match:
                if not clean[0].isdigit():headers.append(raw);continue
                errors.append(dict(source_file=name,source_line=lineno,raw_line=raw,reason='Unrecognized dated record'));continue
            day,month,year,hour,minute,tail=match.groups();flags=[];stamp=None
            try:
                y=int(year)+(century if len(year)==2 else 0)
                stamp=datetime(y,MONTHS[month.upper()],int(day),int(hour),int(minute))
            except (ValueError,KeyError) as exc:flags.append('invalid_datetime: '+str(exc))
            tokens=tail.split()
            if expected_fields is not None and len(tokens)!=expected_fields and not (allow_empty_value and not tokens):
                flags.append(f'expected_{expected_fields}_fields_found_{len(tokens)}')
            values=[]
            for token in tokens:
                try:values.append(float(token.replace('D','E').replace('d','e')))
                except ValueError:values.append(float('nan'));flags.append('invalid_numeric_token: '+token)
            row=dict(series=str(series),source_file=name,source_line=lineno,recorded_time=stamp,
                     raw_line=raw,field_count=len(tokens),parse_flags=';'.join(flags))
            for i in range(max(len(values),expected_fields or 0)):
                row[f'value_{i+1:02d}_raw']=values[i] if i<len(values) else float('nan')
            rows.append(row)
            if flags:errors.append(dict(source_file=name,source_line=lineno,raw_line=raw,reason=';'.join(flags)))
        file['headers']=headers;files.append(file)
    if not files:raise ValueError('No supported .DAT inputs found.')
    return finish(rows,dict(format='dated_ascii',expected_fields=expected_fields,files=files,documents=documents,time_basis=time_basis),errors,missing_values=missing_values,strict=strict)

def read_rain(source, **kwargs):
    kwargs.setdefault('expected_fields',8);kwargs.setdefault('missing_values',(-998,))
    return read_table(source,**kwargs)

def read_mudflow(source, **kwargs):
    kwargs.setdefault('expected_fields',3)
    return read_table(source,**kwargs)

def read_counts(source, **kwargs):
    kwargs.setdefault('expected_fields',1);kwargs.setdefault('allow_empty_value',True)
    # Distinct count files may be alternative versions; never silently combine their meaning.
    kwargs.setdefault('series_from_path',lambda name: PurePosixPath(name).as_posix())
    return read_table(source,**kwargs)

def read_cospec(source, **kwargs):
    """Preserve repeated observations and fixed-width minute/value concatenation."""
    kwargs.setdefault('expected_fields',1)
    kwargs.setdefault('series_from_path',lambda name: PurePosixPath(name).as_posix())
    return read_table(source,**kwargs)


def read_plot_definitions(path):
    """Read BOB series/file/Fortran descriptors as text; never execute formats."""
    from pathlib import Path
    import pandas as pd
    rows=[]
    for lineno,raw in enumerate(Path(path).read_bytes().decode('cp1252',errors='replace').splitlines(),1):
        match=re.fullmatch(r'(\S+:\S+)\s+(\S+)\s+(\(.*\))',raw.replace('\x1a','').strip())
        if match:
            rows.append(dict(bob_series=match[1],source_file=match[2],fortran_format=match[3],source_line=lineno,raw_line=raw))
    return pd.DataFrame(rows,columns=['bob_series','source_file','fortran_format','source_line','raw_line'])
