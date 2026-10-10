"""Focused tests for archive task orchestration. Requires installed FLOVOpy and ObsPy."""
import pytest
from obspy import UTCDateTime
from flovopy.processing.archive_metrics import Config, _days, _config_hash


def config(tmp_path, **kwargs):
    root = tmp_path / 'SDS'
    root.mkdir(exist_ok=True)
    values = dict(sds_root=str(root), output_root=str(tmp_path / 'out'),
                  start='2026-01-01', end='2026-01-03', products=('RSAM',))
    values.update(kwargs)
    return Config(**values)


def test_exclusive_end_day_schedule():
    assert len(list(_days('2026-01-01', '2026-01-03'))) == 2
    assert len(list(_days('2026-01-01T12:00:00', '2026-01-02T12:00:00'))) == 2


def test_product_validation(tmp_path):
    with pytest.raises(ValueError, match='stationxml'):
        config(tmp_path, products=('VSAM',)).validate()


def test_window_divides_day(tmp_path):
    with pytest.raises(ValueError, match='divide 86400'):
        config(tmp_path, sampling_interval=7).validate()


def test_configuration_fingerprint(tmp_path):
    a = config(tmp_path)
    assert _config_hash(a) == _config_hash(config(tmp_path, workers=3))
    assert _config_hash(a) != _config_hash(config(tmp_path, sampling_interval=120))
