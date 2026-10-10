import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from flovopy.processing.spectrograms import (
    SpectrogramResult, compute_spectrogram_result, compute_stream_spectrograms,
)


def trace(data, fs=100, start=0):
    tr = Trace(np.asarray(data, dtype=float))
    tr.stats.sampling_rate = fs
    tr.stats.starttime = UTCDateTime(start)
    tr.stats.network, tr.stats.station, tr.stats.channel = 'XX', 'AAA', 'BHZ'
    return tr


def test_sine_peak():
    fs = 100
    tr = trace(np.sin(2 * np.pi * 10 * np.arange(1000) / fs))
    result = compute_spectrogram_result(tr, fft_seconds=2, step_seconds=1)
    assert np.allclose(result.frequencies[np.nanargmax(result.values, axis=0)], 10)
    assert result.values.shape == (101, 9)


def test_psd_variance_parseval():
    fs = 100
    tr = trace(np.sin(2 * np.pi * 10 * np.arange(200) / fs))
    result = compute_spectrogram_result(tr, fft_seconds=2, step_seconds=1, window='boxcar')
    df = result.frequencies[1] - result.frequencies[0]
    assert np.sum(result.values[:, 0]) * df == pytest.approx(0.5, rel=1e-8)


def test_asd_square_psd():
    tr = trace(np.sin(np.arange(400)))
    kw = dict(fft_seconds=2, step_seconds=1)
    psd = compute_spectrogram_result(tr, quantity='psd', **kw)
    asd = compute_spectrogram_result(tr, quantity='asd', **kw)
    assert np.allclose(asd.values ** 2, psd.values)


def test_nan_gap_and_mask():
    tr = trace(np.ones(400))
    tr.data[120] = np.nan
    result = compute_spectrogram_result(tr, fft_seconds=1, step_seconds=1)
    assert result.coverage.tolist() == pytest.approx([1, .99, 1, 1])
    assert np.isnan(result.values[:, 1]).all()
    masked = trace(np.ones(400))
    masked.data = np.ma.array(masked.data, mask=np.arange(400) == 120)
    result2 = compute_spectrogram_result(masked, fft_seconds=1, step_seconds=1)
    assert np.array_equal(np.isnan(result.values), np.isnan(result2.values))


def test_chunk_equivalence():
    fs = 100
    x = np.sin(2 * np.pi * 7 * np.arange(1200) / fs)
    whole = compute_spectrogram_result(trace(x), fft_seconds=2, step_seconds=1)
    # Adjacent [0,6) and [6,12) output windows, each with sufficient halo.
    left = compute_spectrogram_result(trace(x[:800]), fft_seconds=2, step_seconds=1)
    right = compute_spectrogram_result(trace(x[400:], start=4), fft_seconds=2, step_seconds=1)
    left_keep = left.times < 6
    right_keep = right.times >= 6
    assert np.array_equal(np.r_[left.times[left_keep], right.times[right_keep]], whole.times)
    assert np.allclose(np.column_stack((left.values[:, left_keep], right.values[:, right_keep])), whole.values)


def test_utc_phase():
    x = np.arange(1000, dtype=float)
    whole = compute_spectrogram_result(trace(x), fft_seconds=2, step_seconds=1)
    shifted = compute_spectrogram_result(trace(x[50:], start=.5), fft_seconds=2, step_seconds=1)
    assert np.array_equal(shifted.times, whole.times[1:])
    assert np.allclose(shifted.values, whole.values[:, 1:])


def test_short_trace():
    result = compute_spectrogram_result(trace(np.ones(10)), fft_seconds=2, step_seconds=1)
    assert result.values.shape == (101, 0)


def test_no_mutation():
    tr = trace(np.arange(400.))
    original = tr.data.copy()
    compute_spectrogram_result(tr, fft_seconds=2, step_seconds=1)
    assert np.array_equal(tr.data, original)


def test_validation():
    tr = trace(np.ones(400))
    with pytest.raises(ValueError):
        compute_spectrogram_result(tr, fft_seconds=1.005)
    with pytest.raises(ValueError):
        compute_spectrogram_result(tr, fft_seconds=1, step_seconds=.005)
    with pytest.raises(ValueError):
        compute_spectrogram_result(tr, fft_seconds=1, min_coverage=2)


def test_stream():
    tr = trace(np.ones(400))
    tr2 = tr.copy()
    tr2.stats.station = 'BBB'
    out = compute_stream_spectrograms(Stream([tr, tr2]), fft_seconds=2, step_seconds=1)
    assert len(out) == 2
    with pytest.raises(ValueError):
        compute_stream_spectrograms(Stream([tr, tr.copy()]))


def test_shape_validation():
    with pytest.raises(ValueError):
        SpectrogramResult('a', np.array([0]), np.array([1]), np.zeros((2, 1)), np.array([1]))
