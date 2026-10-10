import json
from pathlib import Path

import pytest
from obspy import read_events, UTCDateTime
from obspy.core.event import Event, Origin, Magnitude, Pick, WaveformStreamID

from flovopy.seisanio.catalog import (convert_archive_catalog, seisan_to_catalog,
                                      write_catalog_result)


def sample_event():
    e = Event()
    e.origins.append(Origin(time=UTCDateTime('2001-01-01T00:00:00'), latitude=16.7,
                            longitude=-62.2))
    e.magnitudes.append(Magnitude(mag=2.1, magnitude_type='ML'))
    e.picks.append(Pick(time=UTCDateTime('2001-01-01T00:00:01'),
                        waveform_id=WaveformStreamID(network_code='MV', station_code='MBLG',
                                                      channel_code='BHZ')))
    return e


class Archive:
    def iter_sfiles(self, *args, **kwargs):
        yield Path('ok.S200101')
        yield Path('bad.S200101')


class Parser:
    def __init__(self, path):
        if path.startswith('bad'):
            raise ValueError('broken Nordic file')
        self.path = path
        self.eventobj = sample_event()
        self.filetime = UTCDateTime('2001-01-01')
        self.mainclass = 'L'
        self.subclass = 'LV'
        self.analyst = 'AB'
        self.analyst_delay = 20
        self.wavfileobjs = []
        self.aeffileobj = None


def test_quakeml_roundtrip_and_failures(tmp_path):
    result = convert_archive_catalog(Archive(), '2001-01-01', '2001-01-02', parser=Parser)
    assert len(result.catalog) == 1
    assert len(result.failures) == 1
    dest = tmp_path / 'catalog.xml'
    write_catalog_result(result, dest, failures_path=tmp_path / 'failures.jsonl')
    loaded = read_events(str(dest))
    assert len(loaded[0].picks) == 1
    assert len(loaded[0].origins) == 1
    assert loaded[0].magnitudes[0].mag == 2.1
    assert loaded[0].resource_id == result.catalog[0].resource_id
    assert json.loads((tmp_path / 'failures.jsonl').read_text())['path'] == 'bad.S200101'


def test_enhanced_sidecar(tmp_path):
    result = convert_archive_catalog(Archive(), 0, 1, parser=Parser, enhanced=True)
    assert result.records[0].meta.metrics['subclass'] == 'LV'
    write_catalog_result(result, tmp_path / 'catalog.xml',
                         metadata_path=tmp_path / 'meta.jsonl')
    row = json.loads((tmp_path / 'meta.jsonl').read_text())
    assert row['metadata']['metrics']['mainclass'] == 'L'
    assert row['metadata']['sfile_path'] == 'ok.S200101'


def test_strict_mode_raises():
    with pytest.raises(ValueError, match='broken Nordic'):
        convert_archive_catalog(Archive(), 0, 1, parser=Parser, strict=True)


def test_legacy_catalog_interface():
    catalog = seisan_to_catalog(Archive(), 0, 1, parser=Parser)
    assert len(catalog) == 1


def test_sidecar_requires_enhanced(tmp_path):
    result = convert_archive_catalog(Archive(), 0, 1, parser=Parser)
    with pytest.raises(ValueError, match='enhanced=True'):
        write_catalog_result(result, tmp_path / 'cat.xml', metadata_path=tmp_path / 'meta.jsonl')
