import math

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Site, Station
from obspy.core.inventory.response import InstrumentSensitivity, Response

from flovopy.io.css import CSSWriter, read_table
from flovopy.io.wfdisc import (calib_from_channel, index_sds, index_seisan_wav, scan_miniseed,
                               verify_wfdisc, write_waveform_database)

T0 = UTCDateTime(1991, 6, 15, 5, 55, 0)
RATE = 6250 / 62


def _trace(sta, chan, start, npts, loc=''):
    return Trace(np.arange(npts, dtype=np.float32), header=dict(
        network='XB', station=sta, location=loc, channel=chan, starttime=start, sampling_rate=RATE))


def _write(path, traces):
    path.parent.mkdir(parents=True, exist_ok=True)
    Stream(traces).write(str(path), format='MSEED', reclen=512, encoding='FLOAT32')
    return path


@pytest.fixture
def seisan(tmp_path):
    db = tmp_path / 'WAV' / 'PNTBO'
    _write(db / '1991' / '06' / '1991-06-15-0555-00M.PNTBO_003', [
        _trace('CAB', 'EHZ', T0, 1000),
        _trace('CAB', 'EHZ', T0 + 20, 500),  # gap: second row for this channel
        _trace('PIEZ', 'EHZ', T0, 800, loc='00')])
    (db / '1991' / '06' / 'junk.txt').write_text('not miniseed')
    return db


def _inventory():
    def channel(code, loc, start, end, value):
        resp = Response(instrument_sensitivity=InstrumentSensitivity(
            value=value, frequency=1.0, input_units='M/S', output_units='COUNTS'))
        return Channel(code, loc, 15.1, 120.3, 500.0, 0.0, azimuth=0.0, dip=-90.0, sample_rate=RATE,
                       start_date=start, end_date=end, response=resp)
    sta = Station('CAB', 15.1, 120.3, 500.0, site=Site(name='Clark Air Base'), start_date=T0 - 86400,
                  channels=[channel('EHZ', '', T0 - 86400, None, 1e9)])
    return Inventory(networks=[Network('XB', stations=[sta])], source='test')


def test_scan_splits_runs_at_gaps(seisan):
    runs = scan_miniseed(next(seisan.glob('1991/06/*PNTBO_003')))
    assert [(r['sta'], r['loc'], r['nsamp']) for r in runs] == [('CAB', '', 1000), ('CAB', '', 500), ('PIEZ', '00', 800)]
    assert runs[0]['foff'] == 0 and runs[1]['foff'] == runs[0]['nbytes']
    assert math.isclose(runs[0]['samprate'], RATE)


def test_seisan_index_round_trip(seisan):
    report = index_seisan_wav(seisan)
    assert report['files'] == 2 and report['files_indexed'] == 1 and len(report['errors']) == 1
    assert report['extra_rows_from_gaps'] == 1
    wf = read_table(seisan / 'PNTBO', 'wfdisc')
    assert list(wf.chan) == ['EHZ', 'EHZ', 'EHZ_00'] and set(wf.dir) == {'1991/06'} and set(wf.datatype) == {'sd'}
    assert math.isclose(wf.endtime[0], T0.timestamp + 999 / RATE, abs_tol=1e-5)
    assert (wf.calib == 0).all() and (wf.calper == -1).all()  # CSS nulls without StationXML
    assert verify_wfdisc(seisan / 'PNTBO') == []
    assert list(read_table(seisan / 'PNTBO', 'snetsta').itertuples(index=False, name=None))[0][:3] == ('XB', 'CAB', 'CAB')
    loc = read_table(seisan / 'PNTBO', 'schanloc')
    assert set(zip(loc.fchan, loc['loc'], loc.chan)) == {('EHZ', '-', 'EHZ'), ('EHZ', '00', 'EHZ_00')}
    with pytest.raises(FileExistsError):
        index_seisan_wav(seisan)
    assert index_seisan_wav(seisan, overwrite=True)['tables']['wfdisc'] == 3


def test_verify_detects_wrong_offset(seisan):
    index_seisan_wav(seisan)
    path = seisan / 'PNTBO.wfdisc'
    lines = path.read_text().splitlines()
    lines[1] = lines[1][:-37] + '0'.rjust(10) + lines[1][-27:]  # foff (before commid, lddate) of row 2 -> 0
    path.write_text('\n'.join(lines) + '\n')
    assert [p['wfid'] for p in verify_wfdisc(seisan / 'PNTBO')] == [2]


def test_inventory_calib_site_sitechan(seisan):
    report = index_seisan_wav(seisan, inventory=_inventory())
    assert report['calibration'] == {'stationxml': 2, 'no_epoch': 1}
    wf = read_table(seisan / 'PNTBO', 'wfdisc')
    assert math.isclose(wf.calib[0], 1e9 / (1e9 * 2 * math.pi), rel_tol=1e-5) and wf.calper[0] == 1.0
    assert wf.chanid[0] == 1 and wf.chanid[2] == -1
    site = read_table(seisan / 'PNTBO', 'site')
    assert site.staname[0] == 'Clark Air Base' and math.isclose(site.elev[0], 0.5)
    sc = read_table(seisan / 'PNTBO', 'sitechan')
    assert (sc.vang[0], sc.hang[0], sc.ctype[0]) == (0.0, 0.0, 'n')


def test_calib_units():
    cha = _inventory()[0][0][0]
    cha.response.instrument_sensitivity.input_units = 'PA'
    assert calib_from_channel(cha)[2] == 'unsupported_units:PA'
    cha = _inventory()[0][0][0]
    cha.response.instrument_sensitivity.output_units = 'V'  # sensor-only model: no counts calibration
    assert calib_from_channel(cha) == (None, None, 'unsupported_output_units:V')


def test_sds_groups(tmp_path):
    for day, doy in ((T0, 166), (T0 + 86400, 167)):
        _write(tmp_path / '1991' / 'XB' / 'CAB' / 'EHZ.D' / f'XB.CAB..EHZ.D.1991.{doy}', [_trace('CAB', 'EHZ', day, 300)])
    assert sorted(index_sds(tmp_path, name='pin', group='day')) == ['pin_19910615', 'pin_19910616']
    reports = index_sds(tmp_path, name='pin', group='month')
    assert reports['pin_199106']['tables']['wfdisc'] == 2
    assert set(read_table(tmp_path / 'pin_199106', 'wfdisc').dir) == {'1991/XB/CAB/EHZ.D'}
    assert verify_wfdisc(tmp_path / 'pin_199106') == []


def test_descriptor_links_and_failed_block(tmp_path):
    with CSSWriter(tmp_path / 'REA' / 'PNTBO', links=['../../WAV/PNTBO/{PNTBO}']) as w:
        w.add('event', evid=w.nextid('evid'))
    assert (tmp_path / 'REA' / 'PNTBO').read_text() == '#\nschema css3.0\ndbpath ../../WAV/PNTBO/{PNTBO}\n'
    with pytest.raises(RuntimeError):
        with CSSWriter(tmp_path / 'X') as w:
            w.add('event', evid=1)
            raise RuntimeError
    assert not list(tmp_path.glob('X*'))
