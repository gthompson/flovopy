"""Fixed-column BOB station-map snapshots; no inferred frequency units or epochs."""
from datetime import datetime
import re
import pandas as pd
from .common import input_members, file_metadata, MONTHS
from .model import BOBDataset

def read_radio_maps(source, *, century=1900, strict=False):
    rows,errors,files=[],[],[]
    for name,blob in input_members(source):
        if not name.upper().endswith(('.OLD','.CUR','.MAP')):continue
        snapshot=None;attenuation=False;last=None;last_radio='';files.append(file_metadata(name,blob))
        for lineno,raw in enumerate(blob.decode('cp1252',errors='replace').splitlines(),1):
            line=raw.replace('\x1a','')
            date=re.search(r'SEISMIC STATION MAP \((\d{1,2}) ([A-Z]{3}) (\d{2}|\d{4})\)',line)
            if date:
                y=int(date[3])+(century if len(date[3])==2 else 0)
                snapshot=datetime(y,MONTHS[date[2]],int(date[1]));attenuation=False;last=None;last_radio='';continue
            if 'DISC' in line and 'RADIO' in line:attenuation='ATTN' in line;continue
            if re.match(r'^\s*\d{2}\s+',line):
                if snapshot is None:
                    errors.append(dict(source_file=name,source_line=lineno,raw_line=raw,reason='Undated station row'));continue
                station=line[6:10].strip();assigned=station!='----';radio=line[56:].strip() if assigned else '';route=''
                match=re.search(r'\s+to\s+(.+)$',radio,re.I)
                if match:route=match[1];radio=radio[:match.start()].strip()
                ditto=radio=='"';resolved=last_radio if ditto else radio
                if assigned and radio and not ditto:last_radio=radio
                row=dict(source_file=name,source_line=lineno,recorded_time=snapshot,series='radio_map',
                    disc_slot=line[1:3].strip(),legacy_id=station,assigned=assigned,
                    site_as_written=line[12:42 if attenuation else 50].strip() if assigned else '',
                    attenuation_as_written=line[42:50].strip() if assigned and attenuation else '',
                    vco_as_written=line[50:56].strip() if assigned else '',radio_as_written=radio,
                    radio_resolved_same_table=resolved,relay_as_written=route,raw_line=raw)
                rows.append(row);last=row
            elif line.strip().upper().startswith('TO ') and last is not None:last['relay_as_written']=line.strip()[3:].strip()
    if not files:raise ValueError('No .OLD/.CUR/.MAP snapshot files found.')
    if strict and errors:raise ValueError(errors)
    return BOBDataset(pd.DataFrame(rows),dict(format='radio_map',files=files,clock_correction_applied=False,
        units='Not inferred; raw frequency/attenuation strings retained',date_meaning='Snapshot date, not a deployment epoch'),pd.DataFrame(errors))
