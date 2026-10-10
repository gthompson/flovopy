"""Pass 7: catalog integration contracts; optional real S-file smoke test."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from obspy.core.event import (Arrival, Catalog, Event, Magnitude, Origin,
                              Pick, ResourceIdentifier, WaveformStreamID)
from obspy import UTCDateTime

from flovopy.seisanio.integration_audit import (
    audit_catalog_roundtrip, audit_sfiles, write_audit_json,
)


def make_event():
    event = Event(resource_id=ResourceIdentifier('smi:local/mvo/test1'))
    pick = Pick(resource_id=ResourceIdentifier('smi:local/mvo/pick1'),
                time=UTCDateTime('2001-01-01T00:00:00'),
                waveform_id=WaveformStreamID(network_code='MV', station_code='MBLG',
                                             channel_code='BHZ'))
    origin = Origin(resource_id=ResourceIdentifier('smi:local/mvo/origin1'),
                    time=UTCDateTime('2001-01-01T00:00:01'), latitude=16.7,
                    longitude=-62.2, depth=1000,
                    arrivals=[Arrival(pick_id=pick.resource_id, phase='P')])
    magnitude = Magnitude(resource_id=ResourceIdentifier('smi:local/mvo/mag1'),
                          mag=2.1, magnitude_type='ML', origin_id=origin.resource_id)
    event.picks.append(pick)
    event.origins.append(origin)
    event.magnitudes.append(magnitude)
    return event


def test_quakeml_roundtrip_picks_origins_magnitudes_arrivals():
    result = audit_catalog_roundtrip(Catalog(events=[make_event()]))
    assert result['passed'], result
    assert result['event_count'] == 1


def test_sfile_audit_keeps_source_and_failures(tmp_path):
    class Parser:
        def __init__(self, path):
            if path.endswith('bad'):
                raise ValueError('unreadable')
            self.eventobj = make_event()
    report = audit_sfiles(['good', 'bad'], parser=Parser, enhanced=False)
    assert report['files_attempted'] == 2
    assert report['events_parsed'] == 1
    assert report['quakeml_roundtrip']['passed']
    assert report['failures'][0]['source_path'] == 'bad'
    out = write_audit_json(report, tmp_path / 'audit.json')
    assert out.exists()


def test_empty_catalog_roundtrip():
    result = audit_catalog_roundtrip(Catalog())
    assert result['passed']
    assert result['event_count'] == 0


@pytest.mark.skipif(not os.environ.get('FLOVOPY_MVO_SFILE'),
                    reason='Set FLOVOPY_MVO_SFILE to a representative real S-file')
def test_real_mvo_sfile(tmp_path):
    from flovopy.research.mvo.mvosfile import MVOSfile
    source = Path(os.environ['FLOVOPY_MVO_SFILE'])
    if not source.is_file():
        pytest.skip(f'S-file unavailable: {source}')
    report = audit_sfiles([source], parser=MVOSfile, enhanced=True)
    write_audit_json(report, tmp_path / 'real_mvo_audit.json')
    assert not report['failures'], report['failures']
    assert report['quakeml_roundtrip']['passed'], report['quakeml_roundtrip']
