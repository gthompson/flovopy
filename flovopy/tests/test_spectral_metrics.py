import numpy as np
import pytest
from flovopy.processing.spectrograms import SpectrogramResult
from flovopy.processing.spectral_metrics import (
    band_power, peak_frequency, spectral_centroid, spectral_bandwidth,
    spectral_ratio, compute_spectral_metrics,
)


def fixture(quantity='psd'):
    f = np.arange(0., 11.)
    s = np.zeros((11, 2))
    s[2, :] = 2.
    s[8, :] = 6.
    s[:, 1] = np.nan
    return SpectrogramResult('XX.A..BHZ', np.array([1., 2.]), f, s,
                             np.array([1., .5]), quantity=quantity)


def test_peak_and_gap():
    a = peak_frequency(fixture())
    assert a[0] == 8 and np.isnan(a[1])


def test_band_power():
    a = band_power(fixture(), 1, 3)
    assert a[0] == pytest.approx(2.) and np.isnan(a[1])


def test_total_power():
    assert band_power(fixture())[0] == pytest.approx(8.)


def test_centroid():
    assert spectral_centroid(fixture())[0] == pytest.approx(6.5)


def test_bandwidth():
    assert spectral_bandwidth(fixture())[0] == pytest.approx(np.sqrt(6.75))


def test_ratio():
    assert spectral_ratio(fixture(), (1, 3), (7, 9))[0] == pytest.approx(np.log2(3))


def test_asd_equivalence():
    psd = fixture()
    asd = fixture('asd')
    asd.values = np.sqrt(asd.values)
    np.testing.assert_allclose(band_power(psd), band_power(asd), equal_nan=True)


def test_reject_magnitude():
    with pytest.raises(ValueError, match='PSD or ASD'):
        band_power(fixture('legacy_magnitude'))


def test_metrics_frame():
    df = compute_spectral_metrics(
        fixture(),
        low_band=(0.5, 3.0),
        high_band=(3.0, 6.0),
    )

    assert len(df) == 2
    assert "peak_frequency" in df.columns
    assert "spectral_centroid" in df.columns
    assert "spectral_bandwidth" in df.columns
    assert "low_band_power" in df.columns
    assert "high_band_power" in df.columns
    assert "fratio" in df.columns

def test_reject_out_of_range():
    with pytest.raises(ValueError):
        band_power(fixture(), -1, 3)


def test_zero_power_nan_centroid():
    r = fixture()
    r.values[:, 0] = 0
    assert np.isnan(spectral_centroid(r)[0])
    assert np.isnan(peak_frequency(r)[0])

def test_peak_frequency_with_known_spectrum():
    from flovopy.processing.spectrograms import SpectrogramResult
    from flovopy.processing.spectral_metrics import peak_frequency

    frequencies = np.array([0., 1., 2., 3., 4., 5., 6.])

    # Two time windows, with peaks at 2 Hz and 5 Hz.
    values = np.array([
        [0., 0.],
        [1., 1.],
        [10., 2.],
        [3., 3.],
        [2., 4.],
        [1., 20.],
        [0., 0.],
    ])

    result = SpectrogramResult(
        trace_id="XX.TEST..BHZ",
        times=np.array([1., 2.]),
        frequencies=frequencies,
        values=values,
        coverage=np.array([1., 1.]),
        quantity="psd",
    )

    peaks = peak_frequency(result)

    np.testing.assert_array_equal(peaks, [2., 5.])

    # No returned peak can exceed the frequency grid.
    assert np.all(peaks <= frequencies.max())