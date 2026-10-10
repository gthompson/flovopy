"""Offline tests; no network, no SDS data required."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from obspy import Stream, Trace, UTCDateTime
import numpy as np
from flovopy.processing import archive_processor as ap


def _config(tmp_path, **kwargs):
    values = dict(source=ap.SourceSpec('files', {'paths': [str(tmp_path / '*.mseed')]}),
                  output_root=str(tmp_path / 'out'), start='2026-01-01T00:00:03',
                  end='2026-01-01T00:00:23', window_seconds=10,
                  processor='flovopy.processing.archive_processor:MiniSeedWindowWriter')
    values.update(kwargs)
    return ap.ArchiveConfig(**values)


def test_windows_clipped_to_half_open_request(tmp_path):
    windows = list(ap._windows(_config(tmp_path)))
    assert [(str(a), str(b)) for _, a, b in windows] == [
        ('2026-01-01T00:00:03.000000Z', '2026-01-01T00:00:10.000000Z'),
        ('2026-01-01T00:00:10.000000Z', '2026-01-01T00:00:20.000000Z'),
        ('2026-01-01T00:00:20.000000Z', '2026-01-01T00:00:23.000000Z')]


def test_source_spec_validation():
    with pytest.raises(ValueError, match='root'):
        ap.SourceSpec('sds', {}).validate()
    with pytest.raises(ValueError, match='factory'):
        ap.SourceSpec('plugin', {'factory': 'broken'}).validate()


def test_hash_ignores_worker_and_resume(tmp_path):
    a = _config(tmp_path)
    assert ap._fingerprint(a) == ap._fingerprint(replace(a, workers=3, resume=False))
    assert ap._fingerprint(a) != ap._fingerprint(replace(a, algorithm_version='2'))


def test_files_source_and_resume(tmp_path):
    trace = Trace(data=np.arange(30, dtype=np.int32))
    trace.stats.network = 'XX'; trace.stats.station = 'TEST'
    trace.stats.channel = 'BHZ'; trace.stats.starttime = UTCDateTime('2026-01-01')
    trace.write(str(tmp_path / 'data.mseed'), format='MSEED')
    cfg = _config(tmp_path)
    first = ap.run(cfg)
    assert (first['processed'], first['windows']) == (3, 3)
    second = ap.run(cfg)
    assert second['skipped'] == 3
    markers = sorted(Path(first['output_dir']).glob('windows/*/complete.json'))
    assert len(markers) == 3
    assert all(json.loads(p.read_text())['artifacts'] == ['waveforms.mseed'] for p in markers)


def test_missing_artifact_reprocesses(tmp_path):
    tr = Trace(data=np.arange(30, dtype=np.int32))
    tr.stats.starttime = UTCDateTime('2026-01-01')
    tr.write(str(tmp_path / 'data.mseed'), format='MSEED')
    cfg = _config(tmp_path)
    result = ap.run(cfg)
    artifact = Path(result['output_dir']) / 'windows/000000000/waveforms.mseed'
    artifact.unlink()
    result = ap.run(cfg)
    assert result['processed'] == 1 and result['skipped'] == 2
