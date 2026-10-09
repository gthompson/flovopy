"""Deployment-oriented metadata: spreadsheet rows -> StationXML and CSS3.0.

A physical sensor can have multiple co-located component channels. Array elements
are separate physical sensor locations with explicit reference-station offsets.
Response-free StationXML is intentionally marked preliminary; EPIC submission
requires valid responses for every waveform channel.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict


def rows(path, sheet):
    path = Path(path)
    if path.is_dir():
        f = path / (sheet + '.csv')
        with f.open(newline='', encoding='utf-8-sig') as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == '.xlsx':
        # Importing spreadsheets is optional; CSV directory needs no spreadsheet dependency.
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        if sheet not in wb.sheetnames: return []
        values = wb[sheet].values
        keys = [str(x).strip() if x is not None else '' for x in next(values)]
        return [dict(zip(keys, v)) for v in values if any(x is not None for x in v)]
    raise ValueError('Provide an .xlsx workbook or a directory of sheet-named CSVs')


def when(value):
    if value is None or str(value).strip() == '': return None
    if isinstance(value, datetime): return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit() and len(value.strip()) == 5):
        return datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=float(value))
    s = str(value).strip().replace('Z', '+00:00')
    d = datetime.fromisoformat(s)
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def asfloat(value, default=None):
    return default if value is None or str(value).strip()=='' else float(value)


def nonempty(value): return value is not None and str(value).strip() != ''


def get_tables(path):
    return {name: rows(path,name) for name in ('Stations','Deployments','Sensors','Channels','Responses','ServiceRuns')}


def validate(tables):
    errors=[]
    deployments={str(r.get('deployment_id')):r for r in tables['Deployments'] if nonempty(r.get('deployment_id'))}
    sensors={str(r.get('sensor_id')):r for r in tables['Sensors'] if nonempty(r.get('sensor_id'))}
    stations={(str(r.get('network')),str(r.get('station'))):r for r in tables['Stations']}
    seen=set(); intervals=defaultdict(list)
    for i,r in enumerate(tables['Channels'],2):
        d=deployments.get(str(r.get('deployment_id')))
        s=sensors.get(str(r.get('sensor_id')))
        if not d: errors.append(f'Channels row {i}: unknown deployment_id {r.get("deployment_id")}');continue
        if not s: errors.append(f'Channels row {i}: unknown sensor_id {r.get("sensor_id")}');continue
        key=(str(d.get('network')),str(d.get('station')))
        if key not in stations: errors.append(f'Channels row {i}: station {key} missing')
        for field in ('location','channel','sample_rate'):
            if not nonempty(r.get(field)): errors.append(f'Channels row {i}: missing {field}')
        try:
            start=when(r.get('start_time') or d.get('start_time'))
            end=when(r.get('end_time') or d.get('end_time'))
            if start is None: raise ValueError('missing start_time')
            if end and end<=start: raise ValueError('end_time must follow start_time')
            asfloat(r.get('sample_rate'))
            if str(r.get('channel','')).endswith(('Z','N','E','1','2','3')) and not nonempty(r.get('dip')):
                errors.append(f'Channels row {i}: missing dip (explicit orientation required)')
            if not nonempty(r.get('azimuth')): errors.append(f'Channels row {i}: missing azimuth')
            if not nonempty(s.get('latitude')) or not nonempty(s.get('longitude')):
                errors.append(f'Channels row {i}: sensor {r.get("sensor_id")} lacks coordinates')
            nslc=(*key,str(r['location']),str(r['channel']))
            epoch=(nslc,start,end)
            if epoch in seen: errors.append(f'Channels row {i}: duplicate NSLC epoch {epoch}')
            seen.add(epoch); intervals[nslc].append((start,end,i))
        except (ValueError,TypeError) as exc: errors.append(f'Channels row {i}: {exc}')
    for nslc, spans in intervals.items():
        spans.sort()
        for prev,cur in zip(spans,spans[1:]):
            if prev[1] is None or prev[1]>cur[0]: errors.append(f'Overlapping epochs {nslc}: rows {prev[2]}, {cur[2]}')
    return errors


def _subset(tables,run_id):
    runs=[r for r in tables['ServiceRuns'] if str(r.get('run_id'))==str(run_id)]
    if len(runs)!=1: raise ValueError(f'Expected exactly one ServiceRuns row for {run_id}')
    run=runs[0]; lo=when(run['start_time']); hi=when(run['end_time'])
    if not lo or not hi or lo>=hi: raise ValueError('Invalid service-run window')
    dep={str(d['deployment_id']):d for d in tables['Deployments']}
    out=[]
    for c in tables['Channels']:
        d=dep.get(str(c.get('deployment_id')))
        if not d: continue
        start=when(c.get('start_time') or d.get('start_time'))
        end=when(c.get('end_time') or d.get('end_time'))
        if start and start<hi and (end is None or end>lo): out.append((c,d,start,end))
    return run,out


def stationxml_for_run(tables,run_id,output,response_templates=None,require_responses=False):
    """Create one StationXML for the service-run window, retaining actual epoch dates.

    Response templates map response_id -> ObsPy Response or a single-channel
    ObsPy Inventory (which is cloned). No synthetic instrument response is used.
    """
    from copy import deepcopy
    from obspy.core.inventory import Inventory,Network,Station,Channel,Site
    from obspy import UTCDateTime
    errors=validate(tables)
    if errors: raise ValueError('\n'.join(errors))
    run, entries=_subset(tables,run_id)
    sensors={str(s['sensor_id']):s for s in tables['Sensors']}
    sites={(str(s['network']),str(s['station'])):s for s in tables['Stations']}
    networks={}; stations={}
    templates=response_templates or {}
    missing=[]
    for c,d,start,end in entries:
        net=str(d['network']);sta=str(d['station']);s=sensors[str(c['sensor_id'])];site=sites[(net,sta)]
        if net not in networks: networks[net]=Network(code=net,stations=[])
        if (net,sta) not in stations:
            stations[(net,sta)]=Station(code=sta,latitude=float(site['latitude']),longitude=float(site['longitude']),elevation=asfloat(site.get('elevation_m'),0),site=Site(name=str(site.get('site_name') or sta)),start_date=UTCDateTime(when(d['start_time'])))
            networks[net].stations.append(stations[(net,sta)])
        ch=Channel(code=str(c['channel']),location_code=str(c['location']),latitude=float(s['latitude']),longitude=float(s['longitude']),elevation=asfloat(s.get('elevation_m'),asfloat(site.get('elevation_m'),0)),depth=asfloat(s.get('depth_m'),0),azimuth=float(c['azimuth']),dip=float(c['dip']),sample_rate=float(c['sample_rate']),start_date=UTCDateTime(start),end_date=UTCDateTime(end) if end else None)
        rid=str(c.get('response_id') or '').strip()
        response=templates.get(rid)
        if response is not None:
            if hasattr(response,'networks'):
                chans=[x for n in response.networks for st in n.stations for x in st.channels]
                if len(chans)!=1: raise ValueError(f'Response template {rid} must have exactly one channel')
                response=chans[0].response
            ch.response=deepcopy(response)
        else: missing.append((net,sta,str(c['location']),str(c['channel']),rid))
        stations[(net,sta)].channels.append(ch)
    if require_responses and missing: raise ValueError(f'Missing responses: {missing}')
    inv=Inventory(networks=list(networks.values()),source='FLOVOpy stationmetadata: deployment workbook')
    inv.write(str(output),format='STATIONXML',validate=True)
    return {'channels':len(entries),'missing_responses':missing,'output':str(output)}


def _lddate(): return datetime.now(timezone.utc).timestamp()

def _jdate(t): return int(t.strftime('%Y%j')) if t else -1


def export_css(tables,run_id,folder):
    """Export CSS3 site/sitechan/snetsta relations as pipe-separated import tables.

    CSS3 site refsta/dnorth/deast represent array geometry; co-located components
    share a physical site. CSS sitechan hang/vang are not StationXML azimuth/dip:
    hang=azimuth, vang=90+dip (degrees down from upward vertical).
    """
    errors=validate(tables)
    if errors: raise ValueError('\n'.join(errors))
    _, entries=_subset(tables,run_id)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    sensors={str(s['sensor_id']):s for s in tables['Sensors']}
    sites={(str(s['network']),str(s['station'])):s for s in tables['Stations']}
    site_rows={}; chan_rows=[]; net_rows={}
    for idx,(c,d,start,end) in enumerate(entries,1):
        net,sta=str(d['network']),str(d['station']);s=sensors[str(c['sensor_id'])]; st=sites[(net,sta)]
        # CSS site STA is an actual *physical sensor location* identifier.
        # Require explicit css_sta to prevent collisions when an array has multiple sensors.
        css_sta=str(s.get('css_sta') or '').strip()
        if not css_sta: raise ValueError(f'Sensor {c["sensor_id"]} missing css_sta')
        ref=str(s.get('refsta') or css_sta)
        north=asfloat(s.get('dnorth_km'),0);east=asfloat(s.get('deast_km'),0)
        if ref!=css_sta and (not nonempty(s.get('dnorth_km')) or not nonempty(s.get('deast_km'))):
            raise ValueError(f'Array element {css_sta} requires dnorth_km and deast_km')
        k=(css_sta,_jdate(start),_jdate(end))
        site_rows[k]=[css_sta,_jdate(start),_jdate(end),s['latitude'],s['longitude'],asfloat(s.get('elevation_m'),asfloat(st.get('elevation_m'),0))/1000,str(s.get('description') or st.get('site_name') or sta),str(s.get('station_type') or 'ss'),ref,north,east,_lddate()]
        chan_rows.append([css_sta,str(c['channel']),_jdate(start),idx,_jdate(end),str(c.get('channel_type') or 'n'),asfloat(s.get('depth_m'),0)/1000,float(c['azimuth']),90+float(c['dip']),str(c.get('description') or ''),_lddate()])
        net_rows[(net,css_sta)]=[net,css_sta,css_sta,_lddate()]
    specs={'site':(['sta','ondate','offdate','lat','lon','elev','staname','statype','refsta','dnorth','deast','lddate'],list(site_rows.values())), 'sitechan':(['sta','chan','ondate','chanid','offdate','ctype','edepth','hang','vang','descrip','lddate'],chan_rows), 'snetsta':(['snet','fsta','sta','lddate'],list(net_rows.values()))}
    for name,(header,data) in specs.items():
        with (folder/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(data)
    return {name:len(data) for name,(_,data) in specs.items()}
