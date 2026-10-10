"""Run with pytest in a FLOVOpy environment with ObsPy installed."""
import numpy as np
import pytest
obspy = pytest.importorskip('obspy')
from obspy import Stream, Trace, UTCDateTime
from flovopy.processing.sam import RSAM, VSAM, VSEM


def waveform(data, fs=10, start='2020-01-01T00:00:00', units=None):
    tr = Trace(data=np.ma.asarray(data))
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = 'XX', 'TEST', '', 'BHZ'
    tr.stats.sampling_rate = fs
    tr.stats.starttime = UTCDateTime(start)
    if units is not None:
        tr.stats.units = units
    return Stream([tr])


def test_signed_std_and_zeros():
    x = np.array([0., 0., 1., -1.] * 10)
    result = RSAM(stream=waveform(x), sampling_interval=2, filter=None, bands={})
    df = next(iter(result.dataframes.values()))
    assert df['coverage'].eq(1).all()
    assert np.isclose(df['rms'].iloc[0], np.std(x[:20]))
    assert np.isclose(df['min'].iloc[0], 0)


def test_nan_coverage_and_filtering():
    x = np.arange(20., dtype=float)
    x[2:7] = np.nan
    obj = RSAM(stream=waveform(x), sampling_interval=1, filter=None, bands={})
    df = next(iter(obj.dataframes.values()))
    assert df['coverage'].tolist() == [0.5, 1.0]
    masked = obj.filter_by_coverage(0.8)
    m = next(iter(masked.dataframes.values()))
    assert np.isnan(m['mean'].iloc[0])
    assert m['coverage'].iloc[0] == 0.5
    assert np.isfinite(df['mean'].iloc[0])


def test_utc_alignment_and_start_alignment():
    st = waveform(np.ones(30), start='2020-01-01T00:00:00.500')
    a = RSAM(stream=st, sampling_interval=1, filter=None, bands={})
    b = RSAM(stream=st, sampling_interval=1, filter=None, bands={}, align='start')
    ta = next(iter(a.dataframes.values()))['time'].iloc[0]
    tb = next(iter(b.dataframes.values()))['time'].iloc[0]
    assert ta % 1 == 0
    assert np.isclose(tb % 1, 0.5)


def test_vsem_integral_and_units():
    st = waveform(np.array([1., -1.] * 10), units='m/s')
    obj = VSEM(stream=st, sampling_interval=1, filter=None, bands={})
    df = next(iter(obj.dataframes.values()))
    assert df['coverage'].eq(1).all()
    assert np.allclose(df['energy'], 1.0)
    with pytest.raises(ValueError, match='calibrated'):
        VSAM(stream=waveform(np.ones(20)), filter=None, bands={})


def test_vsem_downsample_adds_integrals():
    obj = VSEM(stream=waveform(np.array([1., -1.] * 20), units='m/s'),
               sampling_interval=1, filter=None, bands={})
    coarse = obj.downsample(2)
    df = next(iter(coarse.dataframes.values()))
    assert np.allclose(df['energy'], 2.0)
    assert df['coverage'].eq(1).all()
