import pytest
from obspy import UTCDateTime

from flovopy.io.css import CSSWriter, format_record, read_events, read_table, _LAYOUTS


def _width(table):
    return sum(int(f.split(':')[1][1:]) for f in _LAYOUTS[table].split()) + len(_LAYOUTS[table].split()) - 1


def test_nulls_and_widths():
    line = format_record('origin', lat=15.13, lon=120.35, time=float(UTCDateTime(1991, 6, 15)), orid=1, evid=1)
    assert len(line) == _width('origin')
    assert line.startswith('  15.1300  120.3500 -999.0000')
    assert format_record('event', evid=3).split()[1] == '-'


@pytest.mark.parametrize('values', [dict(sta='TOOLONGX'), dict(sta=''), dict(sta=' A'), dict(sta='Ä'),
                                    dict(deltim=float('nan')), dict(arid=1.5), dict(bogus=1)])
def test_rejects_unrepresentable(values):
    with pytest.raises(ValueError):
        format_record('arrival', **values)


def test_round_trip(tmp_path):
    db = tmp_path / 'pin'
    t = UTCDateTime(1991, 6, 15, 5, 55, 12.345)
    with CSSWriter(db) as w:
        evid, orid, arid = w.nextid('evid'), w.nextid('orid'), w.nextid('arid')
        w.add('origin', orid=orid, evid=evid, lat=15.1429, lon=120.3496, depth=2.5, time=float(t),
              jdate=int(t.strftime('%Y%j')), auth='PINATUBO')
        w.add('event', evid=evid, prefor=orid, auth='PINATUBO')
        w.add('arrival', arid=arid, sta='PIEZ', chan='EHZ', iphase='P', time=float(t) + 1.5,
              jdate=int(t.strftime('%Y%j')), deltim=0.05, auth='PINATUBO')
        w.add('assoc', arid=arid, orid=orid, sta='PIEZ', phase='P', timedef='n')
        w.add('netmag', magid=w.nextid('magid'), evid=evid, orid=orid, magtype='md', magnitude=2.3)
        w.add('event', evid=w.nextid('evid'))  # unlocated event: prefor null
    assert db.read_text() == '#\nschema css3.0\n'
    assert dict(read_table(db, 'lastid')[['keyname', 'keyvalue']].values) == dict(evid=2, orid=1, arid=1, magid=1)
    cat = read_events(db)
    assert not cat.issues and len(cat) == 2
    e = cat[0]
    o = e.preferred_origin()
    assert (o.latitude, o.longitude, o.depth) == (15.1429, 120.3496, 2500.0)
    assert abs(o.time - t) < 1e-5
    assert e.picks[0].waveform_id.station_code == 'PIEZ' and abs(e.picks[0].time - (t + 1.5)) < 1e-5
    assert e.magnitudes[0].mag == 2.3
    assert cat[1].preferred_origin_id is None


def test_refuses_existing(tmp_path):
    (tmp_path / 'pin.origin').write_text('')
    with pytest.raises(FileExistsError):
        CSSWriter(tmp_path / 'pin')
