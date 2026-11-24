"""CSS3.0 waveform index (wfdisc) and station relations for miniSEED files, without Antelope.

One wfdisc row per contiguous run of same-channel records (datatype ``sd``): a
new row starts at a gap, overlap, sample-rate change, change of channel or a
record without samples, so every row is one byte range of whole records
(``foff`` to ``foff`` + the records holding ``nsamp`` samples).

Tables are meant to live beside the waveforms, so ``dir`` stays short and relative:

* SEISAN: ``WAV/<DB>/<DB>`` (descriptor) with ``dir`` = ``YYYY/MM``
  (:func:`index_seisan_wav`).
* SDS: ``<SDS>/<name>_<period>`` with ``dir`` = ``YYYY/NET/STA/CHAN.TYPE``, one
  database per day, month, year or the whole archive (:func:`index_sds`).

A catalogue database elsewhere (e.g. ``REA/<DB>``) can see these tables through a
``dbpath`` link; see :class:`flovopy.io.css.CSSWriter`.

With an ObsPy Inventory, ``calib``/``calper`` come from the instrument
sensitivity (nm/count at ``calper`` s) and ``site``/``sitechan`` are written;
otherwise calibration stays CSS null. CSS keeps day-resolution on/off dates only.
"""
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from obspy import UTCDateTime
from obspy.io.mseed.util import get_record_information

from flovopy.io.css import CSSWriter, read_table

__all__ = ['scan_miniseed', 'write_waveform_database', 'index_seisan_wav', 'index_sds',
           'verify_wfdisc', 'calib_from_channel']


def scan_miniseed(path):
    """Contiguous same-channel record runs in one miniSEED file, in file order.

    Each run: dict(net, sta, loc, chan, foff, time, nsamp, samprate, nbytes).
    Raises on truncated records or an unapplied header time correction.
    """
    path = Path(path)
    size = path.stat().st_size
    runs, offset = [], 0
    with path.open('rb') as handle:
        while offset < size:
            info = get_record_information(handle, offset)
            length = info['record_length']
            if not length or offset + length > size:
                raise ValueError(f'{path}: truncated or unreadable record at byte {offset}')
            if info.get('time_correction') and not info.get('activity_flags', 0) & 2:
                raise ValueError(f'{path}: unapplied header time correction at byte {offset}')
            key = (info['network'], info['station'], info['location'], info['channel'])
            npts, rate, start = info['npts'], info['samp_rate'], info['starttime']
            run = runs[-1] if runs else None
            if npts and rate and run is not None and run['key'] == key and run['open'] \
                    and math.isclose(rate, run['samprate'], rel_tol=1e-9) \
                    and abs(start - (run['time'] + run['nsamp'] / rate)) <= 0.5 / rate:
                run['nsamp'] += npts
                run['nbytes'] += length
            elif npts and rate:
                runs.append(dict(key=key, open=True, foff=offset, time=start, nsamp=npts,
                                 samprate=rate, nbytes=length))
            elif run is not None:
                run['open'] = False  # a record without samples ends the byte range
            offset += length
    return [dict(net=r['key'][0], sta=r['key'][1], loc=r['key'][2], chan=r['key'][3],
                 foff=r['foff'], time=r['time'], nsamp=r['nsamp'], samprate=r['samprate'],
                 nbytes=r['nbytes']) for r in runs]


def calib_from_channel(channel):
    """(calib nm/count, calper s, status) from a StationXML channel's sensitivity."""
    response = getattr(channel, 'response', None)
    sens = getattr(response, 'instrument_sensitivity', None)
    if sens is None or not sens.value or not sens.frequency:
        return None, None, 'no_sensitivity'
    units = (sens.input_units or '').upper().replace(' ', '')
    power = {'M': 0, 'NM': 0, 'M/S': 1, 'M/SEC': 1, 'M/S**2': 2, 'M/S/S': 2, 'M/S2': 2, 'M/SEC**2': 2}.get(units)
    if power is None:
        return None, None, f'unsupported_units:{units or "none"}'
    out = (sens.output_units or '').upper().replace(' ', '')
    if out not in ('COUNTS', 'COUNT', 'DU', 'DIGITALCOUNTS'):  # e.g. a sensor-only model ending in V
        return None, None, f'unsupported_output_units:{out or "none"}'
    scale = 1.0 if units == 'NM' else 1e9
    omega = 2 * math.pi * sens.frequency
    return scale / (sens.value * omega ** power), 1.0 / sens.frequency, 'stationxml'


def _jdate(time):
    return int(UTCDateTime(time).strftime('%Y%j'))


def _ascii(text, width):
    text = unicodedata.normalize('NFKD', str(text or '')).encode('ascii', 'ignore').decode()
    text = ' '.join(text.split())[:width].strip()
    return text or None


class _Names:
    """SEED net/sta/loc/chan to CSS sta/chan (Antelope convention: chan_loc)."""

    def __init__(self, station_map=None):
        self.station_map = dict(station_map or {})
        self.owner = {}

    def sta(self, net, sta):
        css = self.station_map.get((net, sta), sta)
        if len(css) > 6:
            raise ValueError(f'{net}.{sta}: station code longer than 6; pass station_map')
        if self.owner.setdefault(css, (net, sta)) != (net, sta):
            raise ValueError(f'CSS sta {css!r} used by {self.owner[css]} and {(net, sta)}; pass station_map')
        return css

    @staticmethod
    def chan(chan, loc):
        css = f'{chan}_{loc}' if loc else chan
        if len(css) > 8:
            raise ValueError(f'CSS chan {css!r} longer than 8')
        return css


class _InventoryIndex:
    def __init__(self, inventory):
        self.epochs = defaultdict(list)
        for net in inventory or []:
            for sta in net:
                for cha in sta:
                    self.epochs[(net.code, sta.code, cha.location_code, cha.code)].append((sta, cha))

    def find(self, net, sta, loc, chan, time):
        for station, channel in self.epochs.get((net, sta, loc, chan), ()):
            if (channel.start_date is None or channel.start_date <= time) and \
                    (channel.end_date is None or time < channel.end_date):
                return station, channel
        return None, None


def write_waveform_database(dbpath, files, *, inventory=None, station_map=None, overwrite=False,
                            links=(), datatype='sd'):
    """Index miniSEED ``files`` into wfdisc, snetsta and schanloc at ``dbpath``.

    ``dir`` is each file's directory relative to ``dbpath``'s directory. Files that
    cannot be indexed are reported in ``errors`` and contribute no rows.
    With ``inventory``: calib/calper from sensitivities, plus site and sitechan.
    Returns a report dict.
    """
    dbpath = Path(dbpath)
    base = dbpath.parent.resolve()
    names = _Names(station_map)
    inv = _InventoryIndex(inventory)
    writer = CSSWriter(dbpath, overwrite=overwrite, links=links)
    lddate = float(UTCDateTime())
    netsta, chanloc, errors = {}, {}, []
    calib_status, gaps = Counter(), Counter()
    sitechan_ids = {}
    if inventory is not None:
        site_keys = Counter()
        for net in inventory:
            for sta in net:
                css_sta = names.sta(net.code, sta.code)
                netsta[(net.code, sta.code)] = css_sta
                ondate = _jdate(sta.start_date) if sta.start_date else None
                site_keys[(css_sta, ondate)] += 1
                writer.add('site', sta=css_sta, ondate=ondate,
                           offdate=_jdate(sta.end_date) if sta.end_date else None,
                           lat=sta.latitude, lon=sta.longitude,
                           elev=None if sta.elevation is None else sta.elevation / 1000,
                           staname=_ascii(sta.site.name if sta.site else None, 50), lddate=lddate)
                for cha in sta:
                    css_chan = names.chan(cha.code, cha.location_code)
                    chanloc[(css_sta, cha.code, cha.location_code)] = css_chan
                    chanid = writer.nextid('chanid')
                    sitechan_ids[id(cha)] = chanid
                    writer.add('sitechan', sta=css_sta, chan=css_chan,
                               ondate=_jdate(cha.start_date) if cha.start_date else None, chanid=chanid,
                               offdate=_jdate(cha.end_date) if cha.end_date else None, ctype='n',
                               edepth=None if cha.depth is None else cha.depth / 1000,
                               hang=cha.azimuth, vang=None if cha.dip is None else cha.dip + 90,
                               lddate=lddate)
        duplicate_sites = sum(n - 1 for n in site_keys.values() if n > 1)
    files = list(dict.fromkeys(Path(f) for f in files))
    for path in files:
        try:
            directory = os.path.relpath(path.parent.resolve(), base)
            if len(directory) > 64 or len(path.name) > 32:
                raise ValueError(f'dir/dfile too long for wfdisc: {directory}/{path.name}')
            runs = scan_miniseed(path)
            keys = Counter((r['net'], r['sta'], r['loc'], r['chan']) for r in runs)
            rows = []
            for run in runs:
                css_sta = names.sta(run['net'], run['sta'])
                css_chan = names.chan(run['chan'], run['loc'])
                station, channel = inv.find(run['net'], run['sta'], run['loc'], run['chan'], run['time'])
                calib, calper, status = (calib_from_channel(channel) if channel is not None else
                                         (None, None, 'no_epoch' if inventory is not None else 'no_stationxml'))
                rows.append((run, css_sta, css_chan, calib, calper, status,
                             sitechan_ids.get(id(channel)) if channel is not None else None))
        except Exception as exc:
            errors.append(dict(file=str(path), error=f'{type(exc).__name__}: {exc}'))
            continue
        gaps.update(n - 1 for n in keys.values() if n > 1)
        for run, css_sta, css_chan, calib, calper, status, chanid in rows:
            netsta[(run['net'], run['sta'])] = css_sta
            chanloc[(css_sta, run['chan'], run['loc'])] = css_chan
            calib_status[status] += 1
            writer.add('wfdisc', sta=css_sta, chan=css_chan, time=float(run['time']),
                       wfid=writer.nextid('wfid'), chanid=chanid, jdate=_jdate(run['time']),
                       endtime=float(run['time']) + (run['nsamp'] - 1) / run['samprate'],
                       nsamp=run['nsamp'], samprate=run['samprate'], calib=calib, calper=calper,
                       datatype=datatype, dir=directory, dfile=path.name, foff=run['foff'], lddate=lddate)
    for (net, sta), css_sta in sorted(netsta.items()):
        writer.add('snetsta', snet=net, fsta=sta, sta=css_sta, lddate=lddate)
    for (css_sta, fchan, loc), css_chan in sorted(chanloc.items()):
        writer.add('schanloc', sta=css_sta, fchan=fchan, loc=loc or None, chan=css_chan, lddate=lddate)
    counts = writer.close()
    report = dict(database=str(dbpath), files=len(files), files_indexed=len(files) - len(errors),
                  tables=counts, errors=errors, calibration=dict(calib_status),
                  extra_rows_from_gaps=sum(k * v for k, v in gaps.items()))
    if inventory is not None:
        report['duplicate_site_day_keys'] = duplicate_sites
    return report


def index_seisan_wav(wav_db_dir, *, dbname=None, **kwargs):
    """Index a SEISAN ``WAV/<DB>`` tree (YYYY/MM/files) into ``WAV/<DB>/<DB>``."""
    wav_db_dir = Path(wav_db_dir)
    dbname = dbname or wav_db_dir.name
    files = sorted(p for p in wav_db_dir.glob('[0-9][0-9][0-9][0-9]/[0-9][0-9]/*')
                   if p.is_file() and not p.name.startswith('.'))
    return write_waveform_database(wav_db_dir / dbname, files, **kwargs)


_SDS_NAME = re.compile(r'^(?P<net>[^.]*)\.(?P<sta>[^.]*)\.(?P<loc>[^.]*)\.(?P<chan>[^.]*)\.(?P<type>[^.])\.(?P<year>\d{4})\.(?P<doy>\d{3})$')


def index_sds(sds_root, *, name='db', group='day', **kwargs):
    """Index an SDS archive into databases at its root, one per day/month/year/total.

    Database names: ``<name>_YYYYMMDD``, ``<name>_YYYYMM``, ``<name>_YYYY`` or
    ``<name>_total``. Files are grouped by the day in their SDS file name.
    Returns {database name: report}.
    """
    if group not in ('day', 'month', 'year', 'total'):
        raise ValueError("group must be 'day', 'month', 'year' or 'total'")
    sds_root = Path(sds_root)
    groups = defaultdict(list)
    for path in sorted(sds_root.glob('[0-9][0-9][0-9][0-9]/*/*/*.*/*')):
        match = _SDS_NAME.match(path.name)
        if not match or not path.is_file():
            continue
        day = UTCDateTime(year=int(match['year']), julday=int(match['doy']))
        label = dict(day=day.strftime('%Y%m%d'), month=day.strftime('%Y%m'),
                     year=day.strftime('%Y'), total='total')[group]
        groups[f'{name}_{label}'].append(path)
    return {db: write_waveform_database(sds_root / db, files, **kwargs) for db, files in sorted(groups.items())}


def verify_wfdisc(dbpath, *, max_rows=None):
    """Re-read each wfdisc row's byte range; return a list of problems (empty if none).

    Checks that the records from ``foff`` hold exactly ``nsamp`` samples of that
    channel starting at ``time``, decoding them with ObsPy.
    """
    import io
    from obspy import read
    dbpath = Path(dbpath)
    table = read_table(dbpath, 'wfdisc')
    if max_rows is not None:
        table = table.head(max_rows)
    problems = []
    for row in table.itertuples(index=False):
        path = dbpath.parent / row.dir / row.dfile
        try:
            with path.open('rb') as handle:
                offset, total = row.foff, 0
                while total < row.nsamp:
                    info = get_record_information(handle, offset)
                    total += info['npts']
                    offset += info['record_length']
                handle.seek(row.foff)
                stream = read(io.BytesIO(handle.read(offset - row.foff)), format='MSEED')
            css_chan = {tr.stats.channel + ('_' + tr.stats.location if tr.stats.location else '') for tr in stream}
            if total != row.nsamp or len(stream) != 1 or css_chan != {row.chan} \
                    or abs(stream[0].stats.starttime.timestamp - row.time) > 0.5 / row.samprate \
                    or stream[0].stats.npts != row.nsamp:
                problems.append(dict(wfid=row.wfid, file=str(path), error=f'decoded {stream} vs nsamp {row.nsamp}'))
        except Exception as exc:
            problems.append(dict(wfid=row.wfid, file=str(path), error=f'{type(exc).__name__}: {exc}'))
    return problems
