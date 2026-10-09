"""Flat, one-row-per-channel-epoch metadata schema and safe migration.

Source data remain authoritative; uncertain channel expansions are flagged and
never treated as verified metadata. Requires openpyxl for XLSX reading.
"""
from __future__ import annotations
import csv
import re
from collections import defaultdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

FIELDS = ('network station location channel start_time end_time latitude longitude elevation_m depth_m '
          'site_name station_type refsta dnorth_km deast_km css_sta spatial_mode enclosure_id '
          'sensor_id sensor_manufacturer sensor_model sensor_serial sensor_azimuth sensor_dip '
          'azimuth dip sample_rate digitizer_manufacturer digitizer_model digitizer_serial '
          'digitizer_input input_range_vpp gain_setting response_id response_source '
          'response_status service_run source_row migration_status review_flags notes').split()
REQUIRED = ('network','station','location','channel','start_time','sample_rate')

def read_flat(path):
    path=Path(path)
    if path.suffix.lower()=='.csv':
        with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    if path.suffix.lower()=='.xlsx':
        from openpyxl import load_workbook
        wb=load_workbook(path,read_only=True,data_only=True)
        sh=wb['Metadata']
        vals=iter(sh.values); keys=next(vals)
        return [dict(zip(keys,row)) for row in vals if any(v is not None for v in row)]
    raise ValueError('Expected .csv or .xlsx')

def _dt(v):
    if v is None or str(v).strip()=='':return None
    if isinstance(v,datetime):return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)
    if isinstance(v,date):return datetime(v.year,v.month,v.day,tzinfo=timezone.utc)
    if isinstance(v,(int,float)) or (isinstance(v,str) and v.strip().isdigit() and len(v.strip())==5):return datetime(1899,12,30,tzinfo=timezone.utc)+timedelta(days=float(v))
    s=str(v).strip().replace('Z','+00:00'); x=datetime.fromisoformat(s)
    return x.replace(tzinfo=timezone.utc) if x.tzinfo is None else x.astimezone(timezone.utc)

def _str(v):return '' if v is None else str(v).strip()

def expand_legacy_channel(code):
    """Conservative syntactic expansion only; no implied instrument calibration."""
    s=_str(code).upper()
    if re.fullmatch(r'[A-Z0-9]{3}',s):return [s],[]
    prefix=s[:2]; suffix=s[2:]
    if len(prefix)==2 and suffix and all(c in 'ZNE123456789F' for c in suffix):
        return [prefix+c for c in suffix],['COMPOSITE_CHANNEL_EXPANDED_REVIEW']
    return [s],['UNRECOGNIZED_CHANNEL_NOT_EXPANDED']

def migrate_legacy(rows):
    out=[]
    for rowno,r in enumerate(rows,2):
        chans,flags=expand_legacy_channel(r.get('channel'))
        for ch in chans:
            d={k:'' for k in FIELDS}
            for old,new in [('network','network'),('station','station'),('location','location'),('ondate','start_time'),('offdate','end_time'),('lat','latitude'),('lon','longitude'),('elev','elevation_m'),('depth','depth_m'),('fsamp','sample_rate'),('vpp','input_range_vpp'),('das_serial','digitizer_serial'),('sensor_serial','sensor_serial'),('notes','notes')]:
                d[new]=r.get(old) if r.get(old) is not None else ''
            d['network']=_str(d['network']);d['station']=_str(d['station']);d['location']=_str(d['location']).zfill(2)
            d['channel']=ch;d['digitizer_model']=_str(r.get('datalogger'));d['sensor_model']=_str(r.get('sensor'))
            d['source_row']=rowno;d['response_status']='unresolved';d['response_source']=''
            d['spatial_mode']='unknown';d['css_sta']=d['station']
            d['azimuth']=r.get('azimuth') if r.get('azimuth') is not None else ''
            # No assumptions about dip/orientation for unconventional channels.
            d['dip']='';d['service_run']='';d['station_type']=''
            f=flags.copy()
            if d['latitude']=='' or d['longitude']=='':f.append('MISSING_COORDINATES')
            if d['azimuth']=='':f.append('MISSING_AZIMUTH')
            f.append('MISSING_DIP')
            if d['digitizer_model']=='':f.append('UNKNOWN_DIGITIZER')
            if d['sensor_serial']=='':f.append('UNKNOWN_SENSOR_SERIAL')
            if d['digitizer_serial']=='':f.append('UNKNOWN_DIGITIZER_SERIAL')
            if d['input_range_vpp']=='':f.append('UNKNOWN_INPUT_RANGE')
            if _str(r.get('notes')):f.append('REVIEW_HISTORICAL_NOTES')
            d['migration_status']='review_required' if f else 'migrated'
            d['review_flags']=';'.join(dict.fromkeys(f))
            out.append(d)
    return out

def validate_flat(rows, strict=False):
    """Return structured errors/warnings; strict treats unresolved metadata as errors."""
    issues=[]; grouped=defaultdict(list)
    for idx,r in enumerate(rows,2):
        def add(severity,code):issues.append({'row':idx,'severity':severity,'code':code,'nslc':'.'.join(_str(r.get(k)) for k in ('network','station','location','channel'))})
        for k in REQUIRED:
            if _str(r.get(k))=='':add('error','MISSING_'+k.upper())
        if not re.fullmatch(r'[A-Z0-9]{3}',_str(r.get('channel'))):add('error','INVALID_CHANNEL')
        for k in ('latitude','longitude','sample_rate'):
            try:
                v=float(r[k]); assert (k!='sample_rate' or v>0)
                if k=='latitude':assert -90<=v<=90
                if k=='longitude':assert -180<=v<=180
            except (KeyError,ValueError,TypeError,AssertionError):add('error','INVALID_'+k.upper())
        try:
            start=_dt(r.get('start_time'));end=_dt(r.get('end_time'))
            if start is None:raise ValueError('missing start')
            if end is not None and end<=start:add('error','INVALID_EPOCH_ORDER')
            grouped[tuple(_str(r.get(k)) for k in ('network','station','location','channel'))].append((start,end,idx))
        except (ValueError,TypeError):add('error','INVALID_EPOCH_DATE')
        for k in ('azimuth','dip','response_id'):
            if _str(r.get(k))=='':add('error' if strict else 'warning','MISSING_'+k.upper())
        if _str(r.get('review_flags')):add('error' if strict else 'warning','MIGRATION_REVIEW_REQUIRED')
    for nslc,spans in grouped.items():
        spans.sort(key=lambda v:v[0])
        for a,b in zip(spans,spans[1:]):
            if a[1] is None or a[1]>b[0]:issues.append({'row':b[2],'severity':'error','code':'OVERLAPPING_EPOCH','nslc':'.'.join(nslc)})
    return issues

def write_csv(rows,path):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader()
        for r in rows:
            w.writerow({k:(v.isoformat() if isinstance(v,(date,datetime)) else v) for k,v in r.items()})

def build_inventory(rows, strict=True, response_templates=None):
    """Build an ObsPy Inventory from flat rows. Responses must be explicit in strict mode.

    response_templates maps response_id to an ObsPy Response object; no fabricated
    gains, coordinates, or orientations. Epochs are not silently merged.
    """
    from obspy import UTCDateTime
    from obspy.core.inventory import Inventory,Network,Station,Channel,Site
    problems=validate_flat(rows,strict=strict)
    errors=[p for p in problems if p['severity']=='error']
    if errors:raise ValueError(f'{len(errors)} metadata validation errors; first: {errors[:5]}')
    response_templates=response_templates or {}
    nets={};stations={}
    for r in rows:
        rid=_str(r.get('response_id'))
        response=response_templates.get(rid)
        if strict and response is None:raise ValueError(f'Missing response object {rid!r} for {r["station"]}.{r["channel"]}')
        net=_str(r['network']);sta=_str(r['station']);key=(net,sta)
        if net not in nets:nets[net]=Network(code=net,stations=[])
        if key not in stations:
            st=Station(code=sta,latitude=float(r['latitude']),longitude=float(r['longitude']),elevation=float(r.get('elevation_m') or 0),site=Site(name=_str(r.get('site_name')) or sta))
            stations[key]=st;nets[net].stations.append(st)
        chan=Channel(code=_str(r['channel']),location_code=_str(r['location']),latitude=float(r['latitude']),longitude=float(r['longitude']),elevation=float(r.get('elevation_m') or 0),depth=float(r.get('depth_m') or 0),azimuth=float(r['azimuth']) if _str(r.get('azimuth')) else None,dip=float(r['dip']) if _str(r.get('dip')) else None,sample_rate=float(r['sample_rate']),start_date=UTCDateTime(_dt(r['start_time'])),end_date=UTCDateTime(_dt(r['end_time'])) if _dt(r.get('end_time')) else None,response=response)
        stations[key].channels.append(chan)
    return Inventory(networks=list(nets.values()),source='FLOVOpy stationmetadata flat schema')
