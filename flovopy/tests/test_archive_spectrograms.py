import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from flovopy.processing.archive_processor import WindowContext, SourceSpec
from flovopy.processing.archive_spectrograms import (
    SpectrogramArchiveConfig, SpectrogramWindowProcessor, _combine_fragments,
    iter_spectrograms, load_spectrogram,
)
from flovopy.processing.spectrograms import compute_spectrogram_result


def trace(x, start=0, fs=100):
    tr = Trace(np.asarray(x, dtype=np.float64))
    tr.stats.network = 'XX'
    tr.stats.station = 'TEST'
    tr.stats.channel = 'BHZ'
    tr.stats.starttime = UTCDateTime(start)
    tr.stats.sampling_rate = fs
    return tr


def context(a, b):
    return WindowContext(str(UTCDateTime(a)), str(UTCDateTime(b)),
                         str(UTCDateTime(a-2)), str(UTCDateTime(b+2)), 0, 'test')


def test_config_halo_and_source_independence():
    cfg = SpectrogramArchiveConfig(SourceSpec('fdsn', {'base_url': 'IRIS'}),
                                   '/tmp/spectra', '2026-01-01', '2026-01-02')
    scheduler = cfg.scheduler_config()
    assert scheduler.source.kind == 'fdsn'
    assert scheduler.halo_seconds == cfg.fft_seconds
    assert scheduler.processor.endswith(':SpectrogramWindowProcessor')
    with pytest.raises(ValueError):
        SpectrogramArchiveConfig(cfg.source, '/tmp/s', cfg.start, cfg.end,
                                 halo_seconds=0).scheduler_config()


def test_chunk_equivalence_and_retrieval(tmp_path):
    fs = 100
    x = np.sin(2*np.pi*7*np.arange(1200)/fs)
    whole = compute_spectrogram_result(trace(x), fft_seconds=2, step_seconds=1)
    proc = SpectrogramWindowProcessor(fft_seconds=2, step_seconds=1)
    root = tmp_path/'runs'/'test'/'windows'
    for index, (a,b,lo,hi) in enumerate([(0,6,0,8),(6,12,4,12)]):
        d = root/f'{index:09d}'
        meta = proc.process(Stream([trace(x[int(lo*fs):int(hi*fs)], start=lo)]),
                            context(a,b), d)
        import json
        (d/'complete.json').write_text(json.dumps({'metadata':meta}))
    shards = list(iter_spectrograms(tmp_path, 'test'))
    assert len(shards) == 2
    np.testing.assert_allclose(np.concatenate([s.times for s in shards]), whole.times, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.concatenate([s.values for s in shards], axis=1), whole.values, equal_nan=True)
    assert len(list(iter_spectrograms(tmp_path, 'test', start=6))) == 1
    assert load_spectrogram(next(root.glob('*/spectrum_*.npz'))).trace_id == trace(x).id


def test_gaps_and_fragments(tmp_path):
    x = np.sin(2*np.pi*4*np.arange(1000)/100)
    stream = Stream([trace(x[:300]), trace(x[500:], start=5)])
    merged = _combine_fragments(stream)
    assert len(merged) == 1
    assert np.ma.isMaskedArray(merged[0].data)
    proc = SpectrogramWindowProcessor(fft_seconds=1, step_seconds=1)
    out = tmp_path/'task'
    meta = proc.process(stream, context(0,10), out)
    result = load_spectrogram(out/meta['artifacts'][0])
    assert np.isnan(result.values).any()
    assert np.any(result.coverage == 0)


def test_overlaps_rejected():
    x = np.arange(300.)
    with pytest.raises(ValueError, match='overlapping'):
        _combine_fragments(Stream([trace(x), trace(x, start=2)]))


def test_empty_stream(tmp_path):
    meta = SpectrogramWindowProcessor().process(Stream(), context(0,10), tmp_path)
    assert meta['artifacts'] == []
