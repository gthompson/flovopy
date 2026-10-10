"""Extended scientific regression tests for flovopy.processing.sam.

Run from the FLOVOpy repository root: pytest -q tests/test_sam_refactor.py tests/test_sam_extended.py
These tests require ObsPy, NumPy, pandas and SciPy.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('obspy')
from obspy import Stream, Trace, UTCDateTime
from flovopy.processing.sam import RSAM, VSAM, DSAM, VSEM


def stream(data, *, fs=10, start='2020-01-01T00:00:00', units=None, channel='BHZ'):
    tr = Trace(data=np.ma.asarray(data))
    tr.stats.network = 'XX'
    tr.stats.station = 'TEST'
    tr.stats.location = ''
    tr.stats.channel = channel
    tr.stats.starttime = UTCDateTime(start)
    tr.stats.sampling_rate = fs
    if units is not None:
        tr.stats.units = units
    return Stream([tr])


def frame(obj):
    assert len(obj.dataframes) == 1
    return next(iter(obj.dataframes.values()))


def rsam(data, **kw):
    return RSAM(stream=stream(data, **kw), sampling_interval=1, filter=None, bands={})


def test_masked_samples_are_missing_not_zero():
    x = np.ma.array([1., -1., 5., 6., 0., 0., 0., 0., 0., 0.],
                    mask=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    df = frame(rsam(x))
    assert df.coverage.iloc[0] == pytest.approx(.8)
    assert np.isfinite(df['rms'].iloc[0])
    assert np.isfinite(df['mean'].iloc[0])


def test_all_valid_zero_samples_have_zero_amplitude():
    df = frame(rsam(np.zeros(20)))
    assert (df.coverage == 1).all()
    for col in ('min', 'mean', 'max', 'median', 'rms'):
        assert (df[col] == 0).all(), col


def test_missing_windows_do_not_poison_later_windows():
    x = np.tile([1., -1.], 15)
    x[10:20] = np.nan
    df = frame(rsam(x))
    assert df.coverage.tolist() == pytest.approx([1., 0., 1.])
    assert np.isnan(df['rms'].iloc[1])
    assert np.isfinite(df['rms'].iloc[2])


def test_infinite_values_count_as_missing():
    x = np.tile([1., -1.], 10)
    x[0], x[1] = np.inf, -np.inf
    df = frame(rsam(x))
    assert df.coverage.iloc[0] == pytest.approx(.8)
    assert np.isfinite(df['mean'].iloc[1])


def test_coverage_filter_is_nonmutating_and_retains_time():
    x = np.tile([1., -1.], 10)
    x[:4] = np.nan
    obj = rsam(x)
    before = frame(obj).copy(deep=True)
    result = obj.filter_by_coverage(.8)
    pd.testing.assert_frame_equal(frame(obj), before)
    assert np.isnan(frame(result)['rms'].iloc[0])
    assert frame(result)['coverage'].iloc[0] == pytest.approx(.6)
    assert np.array_equal(frame(result)['time'], before['time'])
    assert np.isfinite(frame(result)['rms'].iloc[1])


def test_coverage_filter_drop_and_inplace():
    x = np.tile([1., -1.], 10)
    x[:4] = np.nan
    obj = rsam(x)
    returned = obj.filter_by_coverage(.8, inplace=True, drop=True)
    assert returned is obj
    assert len(frame(obj)) == 1
    assert frame(obj)['coverage'].iloc[0] == pytest.approx(1.)


def test_coverage_filter_invalid_threshold():
    with pytest.raises(ValueError, match='coverage'):
        rsam(np.ones(10)).filter_by_coverage(1.1)


def test_utc_midnight_split_matches_whole_trace_without_filtering():
    fs = 10
    x = np.sin(np.arange(40) * 2 * np.pi / 10)
    start = UTCDateTime('2020-01-01T23:59:58')
    whole = frame(rsam(x, fs=fs, start=start))
    left = frame(rsam(x[:20], fs=fs, start=start))
    right = frame(rsam(x[20:], fs=fs, start=start + 2))
    combined = pd.concat([left, right], ignore_index=True)
    pd.testing.assert_frame_equal(whole, combined, check_exact=False, rtol=1e-12, atol=1e-12)
    assert len(whole) == 4
    assert whole['time'].is_unique


def test_one_second_windows_supported_at_100_and_500_hz():
    for fs in (100, 500):
        t = np.arange(3 * fs) / fs
        x = np.sin(2 * np.pi * 5 * t)
        df = frame(rsam(x, fs=fs))
        assert len(df) == 3
        assert (df.coverage == 1).all()
        assert np.allclose(df['rms'], 1 / np.sqrt(2), atol=1e-10)


def test_fractional_trace_start_utc_vs_event_alignment():
    start = UTCDateTime('2020-01-01T00:00:00.5')
    x = np.ones(20)
    utc = frame(RSAM(stream=stream(x, start=start), sampling_interval=1,
                     filter=None, bands={}, align='utc'))
    event = frame(RSAM(stream=stream(x, start=start), sampling_interval=1,
                       filter=None, bands={}, align='start'))
    assert utc.time.iloc[0] % 1 == pytest.approx(0.)
    assert event.time.iloc[0] % 1 == pytest.approx(.5)
    assert utc.coverage.iloc[0] == pytest.approx(.5)
    assert (event.coverage == 1).all()


def test_band_metrics_do_not_propagate_nan_across_gap():
    fs = 100
    t = np.arange(40 * fs) / fs
    x = np.sin(2 * np.pi * 5 * t)
    x[20 * fs:21 * fs] = np.nan
    obj = RSAM(stream=stream(x, fs=fs), sampling_interval=1,
               filter=None, bands={'FIVE_HZ': (4., 6.)})
    df = frame(obj)
    assert df.coverage.iloc[20] == 0
    assert np.isnan(df['FIVE_HZ'].iloc[20])
    assert np.isfinite(df['FIVE_HZ'].iloc[10])
    assert np.isfinite(df['FIVE_HZ'].iloc[30])


def test_short_vlp_record_is_unavailable_not_fabricated():
    fs = 100
    x = np.sin(2 * np.pi * .1 * np.arange(fs * 10) / fs)
    obj = RSAM(stream=stream(x, fs=fs), sampling_interval=1,
               filter=None, bands={'VLP': (.02, .2)})
    df = frame(obj)
    assert df.coverage.eq(1).all()
    assert df['VLP'].isna().all()  # <3 periods at 0.02 Hz


def test_vsem_gap_coverage_and_energy_integral():
    x = np.tile([1., -1.], 15)
    x[10:15] = np.nan
    obj = VSEM(stream=stream(x, units='m/s'), sampling_interval=1,
               filter=None, bands={})
    df = frame(obj)
    assert df.coverage.tolist() == pytest.approx([1., .5, 1.])
    assert df.energy.iloc[0] == pytest.approx(1.)
    assert df.energy.iloc[1] == pytest.approx(.5)
    assert df.energy.iloc[2] == pytest.approx(1.)


def test_physical_units_must_be_explicit():
    for cls, units in ((VSAM, 'm/s'), (DSAM, 'm'), (VSEM, 'm/s')):
        with pytest.raises(ValueError, match='calibrated'):
            cls(stream=stream(np.ones(20)), sampling_interval=1,
                filter=None, bands={})
        obj = cls(stream=stream(np.ones(20), units=units),
                  sampling_interval=1, filter=None, bands={})
        assert len(frame(obj)) == 2


def test_duplicate_trace_ids_raise_instead_of_overwriting():
    st = stream(np.ones(10))
    st += st.copy()
    with pytest.raises(ValueError, match='multiple traces'):
        RSAM(stream=st, sampling_interval=1, filter=None, bands={})


def test_write_and_read_yearly_pickle_roundtrip(tmp_path):
    start = UTCDateTime('2020-01-01T00:00:00')
    obj = RSAM(stream=stream(np.tile([1., -1.], 20), start=start),
               sampling_interval=1, filter=None, bands={})
    obj.write(str(tmp_path), ext='pickle', overwrite=True)
    recovered = RSAM.read(start, start + 4, str(tmp_path),
                          trace_ids=list(obj.dataframes), sampling_interval=1,
                          ext='pickle')
    original = frame(obj)
    loaded = frame(recovered)
    for col in ('time', 'coverage', 'mean', 'rms'):
        assert np.allclose(original[col], loaded[col], equal_nan=True), col


def test_write_and_read_yearly_csv_roundtrip(tmp_path):
    start = UTCDateTime('2020-01-01T00:00:00')
    obj = RSAM(stream=stream(np.tile([1., -1.], 20), start=start),
               sampling_interval=1, filter=None, bands={})
    obj.write(str(tmp_path), ext='csv', overwrite=True)
    recovered = RSAM.read(start, start + 4, str(tmp_path),
                          trace_ids=list(obj.dataframes), sampling_interval=1,
                          ext='csv')
    for col in ('time', 'coverage', 'mean', 'rms'):
        assert np.allclose(frame(obj)[col], frame(recovered)[col], equal_nan=True), col
