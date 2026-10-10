import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from obspy import Trace, Stream, UTCDateTime
from flovopy.processing.spectrograms import compute_stream_spectrograms, icewebSpectrogram
from flovopy.plotting.spectrograms import plot_waveform_spectrograms


def make_stream(n=2, start=UTCDateTime(2020, 1, 1)):
    st = Stream()
    for i in range(n):
        fs = 40.0
        t = np.arange(400) / fs
        tr = Trace(np.sin(2 * np.pi * (3 + i) * t))
        tr.stats.network = 'XX'; tr.stats.station = f'S{i:02d}'
        tr.stats.channel = 'BHZ'; tr.stats.sampling_rate = fs
        tr.stats.starttime = start + i * 0.5
        st += tr
    return st


def test_single_channel():
    fig = plot_waveform_spectrograms(make_stream(1), fft_seconds=1, step_seconds=.5)
    assert len(fig._flovopy_axes) == 1
    plt.close(fig)


def test_multi_channel_time_alignment():
    st = make_stream()
    fig = plot_waveform_spectrograms(st, fft_seconds=1, step_seconds=.5)
    axes = fig._flovopy_axes
    assert axes[0][0].get_shared_x_axes().joined(axes[0][0], axes[1][0])
    assert axes[1][0].lines[0].get_xdata()[0] > axes[0][0].lines[0].get_xdata()[0]
    plt.close(fig)


def test_precomputed_and_no_mutation():
    st = make_stream()
    before = st.copy()
    spectra = compute_stream_spectrograms(st, fft_seconds=1, step_seconds=.5)
    fig = plot_waveform_spectrograms(st, spectra, color_scale='shared')
    plt.close(fig)
    for a, b in zip(st, before):
        assert np.array_equal(a.data, b.data)
        assert 'spectrogramdata' not in a.stats


def test_masked_gap():
    st = make_stream(1)
    st[0].data = np.ma.array(st[0].data, mask=np.arange(400) < 20)
    fig = plot_waveform_spectrograms(st, fft_seconds=1, step_seconds=.5)
    plt.close(fig)


@pytest.mark.parametrize('scale', ['linear', 'log'])
def test_frequency_scale(scale):
    fig = plot_waveform_spectrograms(make_stream(1), fft_seconds=1,
                                    frequency_limits=(.5, 15), frequency_scale=scale)
    plt.close(fig)


def test_fixed_clim_and_channel_order():
    st = make_stream()
    order = [st[1].id, st[0].id]
    fig = plot_waveform_spectrograms(st, fft_seconds=1, clim=(-80, 0), channel_order=order)
    assert fig._flovopy_axes[0][0].get_ylabel() == st[1].id
    plt.close(fig)


def test_compatibility_wrapper():
    fig = icewebSpectrogram(make_stream(1)).plot_modern(fft_seconds=1)
    plt.close(fig)


def test_validation():
    with pytest.raises(ValueError):
        plot_waveform_spectrograms(Stream())
    with pytest.raises(ValueError):
        plot_waveform_spectrograms(make_stream(1), frequency_scale='bad')


def test_utc_time_fix():
    st = make_stream(1, start=UTCDateTime(0))
    a = compute_stream_spectrograms(st, fft_seconds=2, step_seconds=1)[st[0].id]
    tr = st[0].copy(); tr.stats.starttime += 4; tr.data = np.r_[tr.data[160:], np.zeros(160)]
    b = compute_stream_spectrograms(Stream([tr]), fft_seconds=2, step_seconds=1)[tr.id]
    assert np.array_equal(a.times[4:], b.times[:len(a.times[4:])])
