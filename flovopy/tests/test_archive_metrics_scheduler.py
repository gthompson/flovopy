"""Adapter tests; run in the installed FLOVOpy/ObsPy environment."""
import json
from pathlib import Path
from unittest.mock import patch

from flovopy.processing.archive_metrics import Config, _config_hash, _scheduler_config, run, consolidate
from flovopy.processing.archive_processor import _windows


def cfg(tmp_path, **kw):
    sds = tmp_path / 'SDS'
    sds.mkdir(exist_ok=True)
    args = dict(sds_root=str(sds), output_root=str(tmp_path / 'products'),
                start='2026-01-01', end='2026-01-03', products=('RSAM',))
    args.update(kw)
    return Config(**args)


def test_scheduler_adapter(tmp_path):
    c = cfg(tmp_path, workers=3, halo_seconds=120, shard_format='csv')
    sc = _scheduler_config(c, _config_hash(c))
    assert sc.source.kind == 'sds'
    assert sc.processor.endswith(':SAMWindowProcessor')
    assert sc.window_seconds == 86400
    assert sc.halo_seconds == 120
    assert sc.workers == 3
    assert sc.processor_kwargs['shard_format'] == 'csv'
    assert len(list(_windows(sc))) == 2


def test_date_ranges_cannot_share_positional_task_ids(tmp_path):
    a = cfg(tmp_path)
    b = cfg(tmp_path, start='2026-01-02', end='2026-01-04')
    assert _scheduler_config(a, _config_hash(a)).algorithm_version != \
           _scheduler_config(b, _config_hash(b)).algorithm_version


def test_scheduler_is_invoked_and_summary_preserved(tmp_path):
    c = cfg(tmp_path)
    result = dict(run_id='abc', windows=2, processed=2, skipped=0,
                  output_dir=str(tmp_path / 'scheduler-run'))
    with patch('flovopy.processing.archive_metrics.run_scheduler', return_value=result) as mock:
        summary = run(c, consolidate_yearly=False)
    mock.assert_called_once()
    assert summary['days'] == 2
    assert summary['processed'] == 2
    assert summary['scheduler_run_id'] == 'abc'


def test_consolidation_requires_complete_markers(tmp_path):
    c = cfg(tmp_path)
    root = tmp_path / 'scheduler-run'
    with __import__('pytest').raises(RuntimeError, match='Missing completed task'):
        consolidate(c, {'output_dir': str(root)})


def test_empty_task_markers_consolidate(tmp_path):
    c = cfg(tmp_path)
    root = tmp_path / 'scheduler-run'
    sc = _scheduler_config(c, _config_hash(c))
    for index, a, b in _windows(sc):
        d = root / 'windows' / f'{index:09d}'
        d.mkdir(parents=True)
        (d / 'complete.json').write_text(json.dumps({
            'start': str(a), 'end': str(b), 'metadata': {'outputs': []}}))
    assert consolidate(c, {'output_dir': str(root)}) == {'RSAM': 0}
