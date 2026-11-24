"""Antelope CSS3.0 ASCII catalog tables: reading into ObsPy events, and a pure-Python writer."""
from pathlib import Path
from collections import defaultdict
from urllib.parse import quote

import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import (Catalog, Event, Origin, Pick, Arrival, Magnitude,
                              WaveformStreamID, ResourceIdentifier, CreationInfo)

from flovopy.enhanced.catalog import EnhancedCatalog

# Antelope V3 Schema Reference Manual: field order, scalar type and print width.
_LAYOUTS = {
    'arrival': 'sta:s6 time:t17 arid:i8 jdate:y8 stassid:i8 chanid:i8 chan:s8 iphase:s8 stype:s1 deltim:r6 azimuth:r7 delaz:r7 slow:r7 delslo:r7 ema:r7 rect:r7 amp:r10 per:r7 logat:r7 clip:s1 fm:s2 snr:r10 qual:s1 auth:s15 commid:i8 lddate:t17',
    'assoc': 'arid:i8 orid:i8 sta:s6 phase:s8 belief:r4 delta:r8 seaz:r7 esaz:r7 timeres:r8 timedef:s1 azres:r7 azdef:s1 slores:r7 slodef:s1 emares:r7 wgt:r6 vmodel:s15 commid:i8 lddate:t17',
    'event': 'evid:i8 evname:s15 prefor:i8 auth:s15 commid:i8 lddate:t17',
    'lastid': 'keyname:s15 keyvalue:i8 lddate:t17',
    'netmag': 'magid:i8 net:s8 orid:i8 evid:i8 magtype:s6 nsta:i8 magnitude:r7 uncertainty:r7 auth:s15 commid:i8 lddate:t17',
    'origerr': 'orid:i8 sxx:r15 syy:r15 szz:r15 stt:r15 sxy:r15 sxz:r15 syz:r15 stx:r15 sty:r15 stz:r15 sdobs:r9 smajax:r9 sminax:r9 strike:r6 sdepth:r9 stime:r8 conf:r5 commid:i8 lddate:t17',
    'origin': 'lat:r9 lon:r9 depth:r9 time:t17 orid:i8 evid:i8 jdate:y8 nass:i4 ndef:i4 ndp:i4 grn:i8 srn:i8 etype:s2 review:s4 depdp:r9 dtype:s1 mb:r7 mbid:i8 ms:r7 msid:i8 ml:r7 mlid:i8 algorithm:s15 auth:s15 commid:i8 lddate:t17',
    'remark': 'commid:i8 lineno:i8 remark:s80 lddate:t17',
    'stamag': 'magid:i8 sta:s6 arid:i8 orid:i8 evid:i8 phase:s8 magtype:s6 magnitude:r7 uncertainty:r7 auth:s15 commid:i8 lddate:t17',
    'detection': 'srcid:s64 tagname:s8 sta:s6 chan:s8 time:t17 arid:i8 state:s3 filter:s30 snr:r10 lddate:t17',
    'detev': 'sta:s6 chan:s8 filter:s30 time:t17 tron:r12 troff:r12 iphase:s8 snr:r10 lddate:t17',
    # Waveform index and station relations (css3.0 plus Antelope's snetsta/schanloc).
    'wfdisc': 'sta:s6 chan:s8 time:t17 wfid:i8 chanid:i8 jdate:y8 endtime:t17 nsamp:i8 samprate:r11 calib:r16 calper:r16 instype:s6 segtype:s1 datatype:s2 clip:s1 dir:s64 dfile:s32 foff:i10 commid:i8 lddate:t17',
    'snetsta': 'snet:s8 fsta:s10 sta:s6 lddate:t17',
    'schanloc': 'sta:s6 fchan:s8 loc:s8 chan:s8 lddate:t17',
    'site': 'sta:s6 ondate:y8 offdate:y8 lat:r9 lon:r9 elev:r9 staname:s50 statype:s4 refsta:s6 dnorth:r9 deast:r9 lddate:t17',
    'sitechan': 'sta:s6 chan:s8 ondate:y8 chanid:i8 offdate:y8 ctype:s4 edepth:r9 hang:r6 vang:r6 descrip:s50 lddate:t17',
    # Calibration: wfdisc.calib = instrument.ncalib * sensor.calratio (layouts as in the pisces css3 schema).
    'sensor': 'sta:s6 chan:s8 time:t17 endtime:t17 inid:i8 chanid:i8 jdate:y8 calratio:r16 calper:r16 tshift:r6 instant:s1 lddate:t17',
    'instrument': 'inid:i8 insname:s50 instype:s6 band:s1 digital:s1 samprate:r11 ncalib:r16 ncalper:r16 dir:s64 dfile:s32 rsptype:s6 lddate:t17',
}


def read_table(path, table=None):
    """Read a table file, or a database basename and table name, into pandas.

    Columns retain CSS names, epoch seconds, native units and CSS null sentinels.
    Character fields are stripped of padding but preserve interior spaces.
    """
    path = Path(path)
    if table is None:
        table = path.suffix.lstrip(".")
    else:
        path = Path(f"{path}.{table}")
    if table not in _LAYOUTS:
        raise ValueError(f"Unsupported CSS relation {table!r}. Available: {', '.join(_LAYOUTS)}")
    fields = []
    offset = 0
    for definition in _LAYOUTS[table].split():
        name, fmt = definition.split(":")
        end = offset + int(fmt[1:])
        fields.append((name, fmt[0], slice(offset, end)))
        offset = end + 1
    rows = []
    with path.open(encoding="ascii") as stream:
        for number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            if len(line.rstrip()) > end or len(line) < fields[-1][2].start:
                raise ValueError(f"{path}:{number}: unexpected CSS record length {len(line)}")
            row = []
            for name, kind, span in fields:
                value = line[span].strip()
                try:
                    if kind == "s":
                        row.append(value)
                    elif kind in ("i", "y"):
                        row.append(int(value))
                    else:
                        row.append(float(value))
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: invalid {name}: {value!r}") from exc
            rows.append(row)
    return pd.DataFrame.from_records(rows, columns=[name for name, _, _ in fields])


class CSS_Catalog(EnhancedCatalog):
    """ObsPy Catalog retaining CSS3.0 source relations, detector records,
    and unassociated picks.

    Subclasses EnhancedCatalog so that to_event_rate(), to_dataframe(),
    filter helpers, and group_by() are available without duplication.

    CSS-specific extras
    -------------------
    tables : dict
        Raw DataFrames keyed by CSS relation name (origin, arrival, assoc, …),
        exactly as returned by read_table().
    unassociated_picks : list of obspy.core.event.Pick
        Arrivals not linked to any origin via the assoc table.
    issues : list of str
        Unresolved cross-table links logged during read_events().
    """

    def __init__(self, events=None, **kwargs):
        super().__init__(events=events or [])
        self.tables: dict = {}
        self.unassociated_picks: list = []
        self.issues: list = []

    @property
    def detections(self):
        """Detector state changes, as recorded in the detection relation."""
        return self.tables.get("detection", pd.DataFrame(columns=[
            field.split(":")[0] for field in _LAYOUTS["detection"].split()]))


def read_events(path, *, namespace=None):
    """Read sibling Antelope CSS3.0 ASCII relations into a CSS_Catalog.

    Pass the database basename/descriptor or any supported table path.
    namespace defaults to the volume basename and scopes QuakeML resource IDs.
    Missing optional tables are allowed. Inspect catalog.issues for unresolved links.
    """
    base = Path(path)
    if base.suffix.lstrip(".") in _LAYOUTS:
        base = base.with_suffix("")
    catalog = CSS_Catalog()
    for table in _LAYOUTS:
        if Path(f"{base}.{table}").is_file():
            catalog.tables[table] = read_table(base, table)
    if not catalog.tables:
        raise FileNotFoundError(f"No supported CSS tables beside {base}")
    prefix = "smi:local/css/" + quote(namespace or base.name, safe="")

    def rid(kind, key):
        return ResourceIdentifier(f"{prefix}/{kind}/{key}")

    def records(table):
        frame = catalog.tables.get(table)
        return () if frame is None else frame.itertuples(index=False)

    events, origins, picks = {}, {}, {}
    seen_picks = defaultdict(set)

    def event(key):
        if key not in events:
            events[key] = Event(resource_id=rid("event", key))
        return events[key]

    for r in records("event"):
        event(r.evid).creation_info = CreationInfo(author=None if r.auth == "-" else r.auth)
    for r in records("origin"):
        key = r.evid if r.evid != -1 else f"orid-{r.orid}"
        e = event(key)
        o = Origin(resource_id=rid("origin", r.orid),
                   time=UTCDateTime(r.time) if r.time > -9999999999 else None,
                   latitude=None if r.lat == -999 else r.lat,
                   longitude=None if r.lon == -999 else r.lon,
                   depth=None if r.depth == -999 else r.depth * 1000)
        e.origins.append(o)
        origins[r.orid] = (key, o)
    for r in records("event"):
        if r.prefor == -1:
            continue
        if r.prefor in origins and origins[r.prefor][0] == r.evid:
            events[r.evid].preferred_origin_id = origins[r.prefor][1].resource_id
        else:
            catalog.issues.append(f"event {r.evid}: unresolved prefor {r.prefor}")
    for r in records("arrival"):
        if r.time <= -9999999999:
            catalog.issues.append(f"arrival {r.arid}: missing time")
            continue
        pick = Pick(resource_id=rid("pick", r.arid), time=UTCDateTime(r.time),
                    waveform_id=WaveformStreamID(network_code="", station_code=r.sta, channel_code=None if r.chan == "-" else r.chan),
                    phase_hint=None if r.iphase == "-" else r.iphase)
        if r.deltim >= 0:
            pick.time_errors.uncertainty = r.deltim
        picks[r.arid] = pick
    associated = set()
    for r in records("assoc"):
        if r.orid not in origins or r.arid not in picks:
            catalog.issues.append(f"assoc: unresolved orid {r.orid} or arid {r.arid}")
            continue
        key, origin = origins[r.orid]
        pick = picks[r.arid]
        if r.arid not in seen_picks[key]:
            events[key].picks.append(pick)
            seen_picks[key].add(r.arid)
        origin.arrivals.append(Arrival(resource_id=rid("assoc", f"{r.orid}-{r.arid}"),
                                       pick_id=pick.resource_id, phase=None if r.phase == "-" else r.phase,
                                       time_residual=None if r.timeres == -999 else r.timeres))
        associated.add(r.arid)
    catalog.unassociated_picks = [p for arid, p in picks.items() if arid not in associated]
    for r in records("netmag"):
        if r.magnitude == -999:
            continue
        if r.evid not in events:
            catalog.issues.append(f"netmag {r.magid}: unresolved evid {r.evid}")
            continue
        mag = Magnitude(resource_id=rid("magnitude", r.magid), mag=r.magnitude,
                        magnitude_type=r.magtype, station_count=r.nsta if r.nsta >= 0 else None)
        if r.orid in origins and origins[r.orid][0] == r.evid:
            mag.origin_id = origins[r.orid][1].resource_id
        if r.uncertainty >= 0:
            mag.mag_errors.uncertainty = r.uncertainty
        events[r.evid].magnitudes.append(mag)
    catalog.events = list(events.values())
    for e in catalog:
        e.scope_resource_ids()
    return catalog


# CSS3.0 null values (Antelope css3.0 schema). By scalar kind, then per field.
_NULL_BY_KIND = {'s': '-', 'i': -1, 'y': -1, 't': -9999999999.999, 'r': -1.0}
_NULL_BY_FIELD = {name: -999.0 for name in (
    'lat', 'lon', 'depth', 'depdp', 'mb', 'ms', 'ml', 'magnitude', 'logat',
    'seaz', 'esaz', 'timeres', 'azres', 'slores', 'emares')}
_NULL_BY_FIELD.update(conf=0.0, elev=-999.0, endtime=9999999999.999, calib=0.0,
                      dnorth=0.0, deast=0.0, edepth=-9.9999, calratio=1.0, tshift=0.0)
# Decimal places of real fields, from the schema print formats. Epoch times use 5.
_DECIMALS = dict(
    lat=4, lon=4, depth=4, depdp=4, sdepth=4, sdobs=4, smajax=4, sminax=4,
    sxx=4, syy=4, szz=4, stt=4, sxy=4, sxz=4, syz=4, stx=4, sty=4, stz=4,
    deltim=3, rect=3, delta=3, timeres=3, wgt=3, conf=3,
    azimuth=2, delaz=2, slow=2, delslo=2, ema=2, per=2, logat=2, snr=2, belief=2,
    seaz=2, esaz=2, slores=2, magnitude=2, uncertainty=2, mb=2, ms=2, ml=2,
    strike=2, stime=2, tron=3, troff=3,
    amp=1, azres=1, emares=1,
    samprate=7, calib=6, calper=6, calratio=6, ncalib=6, ncalper=6, tshift=2, elev=4, dnorth=4, deast=4, edepth=4, hang=1, vang=1)


def _fields(table):
    if table not in _LAYOUTS:
        raise ValueError(f"Unsupported CSS relation {table!r}. Available: {', '.join(_LAYOUTS)}")
    return [(name, fmt[0], int(fmt[1:])) for name, fmt in
            (definition.split(':') for definition in _LAYOUTS[table].split())]


def _format_field(table, name, kind, width, value):
    """One fixed-width field. Reals drop decimals to fit the width (as a fixed-width
    print would round); anything that still does not fit is refused, never truncated."""
    if kind == 's':
        text = str(value)
        if not text.strip():
            raise ValueError(f'{table}.{name}: empty string; use None for CSS null')
        if text != text.strip():
            # Readers strip padding, so edge whitespace would be lost silently.
            raise ValueError(f'{table}.{name}: leading/trailing whitespace in {text!r}')
        if any(c in text for c in '\r\n\t') or not text.isascii():
            raise ValueError(f'{table}.{name}: non-ASCII or control character in {text!r}')
        if len(text) > width:
            raise ValueError(f'{table}.{name}: {text!r} exceeds {width} characters')
        return text.ljust(width)
    if kind in ('i', 'y'):
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f'{table}.{name}: integer required, got {value!r}')
        text = str(int(value)).rjust(width)
    else:
        value = float(value)
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError(f'{table}.{name}: non-finite value; use None for CSS null')
        decimals = 5 if kind == 't' else _DECIMALS.get(name, 4)
        text = f"{value:{width}.{decimals}f}"
        while len(text) > width and decimals > 0:
            decimals -= 1
            text = f"{value:{width}.{decimals}f}"
    if len(text) > width:
        raise ValueError(f'{table}.{name}: {value!r} does not fit in {width} characters')
    return text


def format_record(table, **values):
    """Format one CSS3.0 record; omitted fields and None become CSS nulls."""
    fields = _fields(table)
    unknown = set(values) - {name for name, _, _ in fields}
    if unknown:
        raise ValueError(f'{table}: unknown fields {sorted(unknown)}')
    parts = []
    for name, kind, width in fields:
        value = values.get(name)
        if value is None:
            value = _NULL_BY_FIELD.get(name, _NULL_BY_KIND[kind]) if kind in 'rt' else _NULL_BY_KIND[kind]
        parts.append(_format_field(table, name, kind, width, value))
    return ' '.join(parts)


class CSSWriter:
    """Pure-Python writer for a CSS3.0 flat-file database (no Antelope needed).

    Mirrors the Datascope calls an exporter needs: ``nextid(key)`` and
    ``add(table, **fields)``. Rows are held in memory and written on ``close()``
    as fixed-width ASCII relations, plus ``lastid`` and a descriptor file
    (``schema css3.0``, and ``dbpath`` lines for ``links``) at ``dbpath``.

    By default refuses an existing database. ``overwrite=True`` replaces it as a
    derived product: every relation file this module knows is rewritten or removed,
    each via a temporary file, and nothing else in the directory is touched.
    Leaving a ``with`` block on an exception writes nothing.

    ``links``: other databases whose tables this one should see, as Antelope
    ``dbpath`` entries, e.g. ``['../WAV/PNTBO/{PNTBO}']`` (relative to this directory).
    """

    def __init__(self, dbpath, *, schema='css3.0', overwrite=False, links=()):
        self.dbpath = Path(dbpath)
        if not overwrite and (self.dbpath.exists() or list(self.dbpath.parent.glob(self.dbpath.name + '.*'))):
            raise FileExistsError(f'Refusing to overwrite CSS database: {self.dbpath}')
        if self.dbpath.is_dir():
            raise IsADirectoryError(f'CSS descriptor path is a directory: {self.dbpath}')
        self.schema = schema
        self.overwrite = overwrite
        self.links = [str(link) for link in links]
        self.rows = defaultdict(list)
        self.lastid = {}
        self.closed = False

    def nextid(self, key):
        self.lastid[key] = self.lastid.get(key, 0) + 1
        return self.lastid[key]

    def add(self, table, **values):
        if self.closed:
            raise ValueError('CSS database already closed')
        self.rows[table].append(format_record(table, **values))

    def counts(self):
        return {t: len(r) for t, r in self.rows.items()}

    def close(self):
        """Write all relations once; returns {table: row count}."""
        if self.closed:
            return self.counts()
        import os, tempfile
        lddate = float(UTCDateTime())
        for key, value in sorted(self.lastid.items()):
            self.rows['lastid'].append(format_record('lastid', keyname=key, keyvalue=value, lddate=lddate))
        self.dbpath.parent.mkdir(parents=True, exist_ok=True)
        descriptor = f'#\nschema {self.schema}\n' + ''.join(f'dbpath {":".join(self.links)}\n' for _ in self.links[:1])
        outputs = {Path(f'{self.dbpath}.{t}'): '\n'.join(r) + '\n' for t, r in self.rows.items() if r}
        outputs[self.dbpath] = descriptor
        for path, text in outputs.items():
            handle = tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.css-', delete=False, encoding='ascii')
            with handle:
                handle.write(text)
            os.replace(handle.name, path)
        if self.overwrite:
            for table in _LAYOUTS:
                stale = Path(f'{self.dbpath}.{table}')
                if stale not in outputs and stale.exists():
                    stale.unlink()
        self.closed = True
        return self.counts()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            self.close()
        return False
