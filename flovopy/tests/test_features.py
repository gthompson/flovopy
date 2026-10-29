import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from obspy import Trace, Stream
from flovopy.core.features import FeatureConfig, extract_features, extract_trace_features
from flovopy.enhanced.trace import EnhancedTrace
from flovopy.enhanced.stream import EnhancedStream


def signal(fs=100, seconds=30):
    t = np.arange(int(fs*seconds))/fs
    return t, np.exp(-.5*((t-8)/.7)**2)*np.sin(2*np.pi*12*t) + np.exp(-.5*((t-14)/.7)**2)*np.sin(2*np.pi*2*t)


def test_delayed_lf_and_sample_rate_invariance():
    for fs in (80, 100, 200):
        t, x = signal(fs)
        r = extract_features(x, fs)
        v = r['values']
        assert v['hf_lf.peak_delay_s'] == pytest.approx(6, abs=.15)
        assert v['hf_lf.centroid_delay_s'] == pytest.approx(6, abs=.15)
        assert v['hf_lf.xcorr_lag_s'] == pytest.approx(6, abs=.15)
        assert .8 < v['hf_lf.xcorr_coefficient'] <= 1
        assert v['full.energy_t95_s'] > v['full.energy_t05_s']


def test_tone_power_and_scale_invariance():
    fs=100
    t=np.arange(3000)/fs
    x=np.sin(2*np.pi*5*t)
    a=extract_features(x,fs)['values']
    b=extract_features(x*1e-100+2e-100,fs)['values']
    assert a['spectral.power_peak_hz'] == 5
    assert a['spectral_power.centroid'] == pytest.approx(5, abs=.01)
    assert a['spectral.power_median_hz'] == 5
    assert a['time.crest_factor'] == pytest.approx(np.sqrt(2))
    for k in ('full.energy_t50_s','spectral_power.centroid','time.crest_factor','cepstral.peak_quefrency_s'):
        assert a[k] == pytest.approx(b[k],abs=.02)


def test_energy_duration_uses_squared_samples():
    x=np.zeros(1000);x[200]=1;x[700]=-1
    v=extract_features(x,100)['values']
    assert v['full.duration_90_s'] == 5
    assert v['full.energy_centroid_s'] == 4.5
    assert v['full.energy_spread_s'] == 2.5
    assert v['time_energy.excess_kurtosis'] == -2


@pytest.mark.parametrize('data,flag',[(np.zeros(100),'zero_variance'),(np.ones(100),'zero_variance'),([], 'too_short'),([1,np.nan]*50,'invalid_waveform'),(np.ma.array(np.arange(100),mask=np.arange(100)==20),'invalid_waveform')])
def test_bad_inputs_stable_schema(data,flag):
    r=extract_features(data,100)
    assert flag in r['qc_flags']
    assert set(r['values']) == set(extract_features(signal()[1],100)['values'])
    assert all(np.isnan(v) for v in r['values'].values())


def test_invalid_configs_and_nyquist():
    for kwargs in ({'lf_band':(4,1)}, {'hf_band':(3,20)}, {'smoothing_seconds':0}, {'filter_order':2.5}, {'log_floor_relative':0}):
        with pytest.raises(ValueError): FeatureConfig(**kwargs)
    with pytest.raises(ValueError): extract_features([1]*100,0)
    r=extract_features(np.sin(np.arange(1000)),20)
    assert 'hf:invalid_nyquist' in r['qc_flags']
    assert np.isnan(r['values']['hf_lf.peak_delay_s'])


def test_noise_duration_and_censoring():
    rng=np.random.default_rng(42)
    noise=rng.normal(0,.001,2000)
    t,x=signal()
    r=extract_features(x,100,noise=noise)
    assert r['values']['full.threshold_span_s'] > 6
    assert r['values']['full.threshold_active_s'] <= r['values']['full.threshold_span_s']
    assert np.isfinite(r['values']['hf_lf.onset_delay_s'])
    r=extract_features(np.sin(2*np.pi*5*t),100,noise=noise)
    assert 'full:duration_censored' in r['qc_flags']
    r=extract_features(x,100,noise=np.zeros(2000))
    assert 'full:zero_noise' in r['qc_flags']
    r=extract_features(x,100,noise=rng.normal(0,100,2000))
    assert 'full:no_threshold_detection' in r['qc_flags']
    assert 'full:low_snr' in r['qc_flags']


def test_cepstrum_echo():
    rng=np.random.default_rng(5)
    source=rng.normal(size=10000)
    x=source.copy();x[50:]+=.8*source[:-50]
    r=extract_features(x,100,config=FeatureConfig(cepstrum_min_seconds=.2,cepstrum_max_seconds=.8))
    assert r['values']['cepstral.peak_quefrency_s'] == pytest.approx(.5,abs=.01)


def test_trace_stream_preserve_legacy_and_input():
    x=signal()[1]
    tr=EnhancedTrace(x.copy(),header={'sampling_rate':100})
    tr.stats.metrics={'sample_rms':123,'spectral':{'meanf':42}}
    before=tr.copy()
    r=tr.extract_features()
    assert tr.stats.metrics.sample_rms == 123
    assert tr.stats.metrics.spectral.meanf == 42
    np.testing.assert_array_equal(tr.data,before.data)
    es=EnhancedStream(Stream([tr.copy(),tr.copy()]))
    assert len(es.extract_features(attach=False)) == 2
    noise=Trace(x.copy(),header={'sampling_rate':50})
    with pytest.raises(ValueError): extract_trace_features(tr,noise)
    assert r['provenance']['npts'] == len(x)


def test_pipeline_opt_in_and_legacy(tmp_path):
    directory=Path(__file__).resolve().parents[2]/'pinatubo_features_v6'
    sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('feature_test_pipeline',directory/'pipeline.py')
    pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
    cfg=json.loads((directory/'config.example.json').read_text())
    x=signal()[1]+np.random.default_rng(2).normal(0,.001,3000)
    path=tmp_path/'test.mseed'
    Trace(x,header={'sampling_rate':100,'station':'TEST','channel':'EHZ'}).write(str(path),format='MSEED')
    _,old,q,_=pipeline.process_file(path,cfg)
    assert len(old) == 2, q
    cfg['processing']['compute_reusable_features']=True
    _,new,q,_=pipeline.process_file(path,cfg)
    assert len(new) == 2,q
    assert 'features.values.hf_lf.peak_delay_s' in new[1]
    for a,b in zip(old,new):
        for k,v in a.items():
            if isinstance(v,float) and np.isnan(v):
                assert np.isnan(b[k])
            else: assert b[k] == v, k
    json.dumps(pipeline._plain(new),allow_nan=False)
    pipeline.write_csv(tmp_path/'metrics.csv',new)
    assert 'features.version' in (tmp_path/'metrics.csv').read_text()


def test_uncalibrated_opt_in_has_no_physical_metrics():
    tr=Trace(signal()[1],header={'sampling_rate':100,'channel':'EHZ'})
    tr.stats.units='counts'
    rejected=EnhancedStream(Stream([tr.copy()]))
    rejected.ampengfft(compute_sam=False)
    assert 'metric_error' in rejected[0].stats.metrics
    accepted=EnhancedStream(Stream([tr.copy()]))
    accepted.ampengfft(allow_uncalibrated=True,compute_sam=False)
    m=accepted[0].stats.metrics
    assert m['uncalibrated']
    assert m['sample_rms'] > 0
    assert not any(k in m for k in ('pgv','pga','pgd','pap','pgv_units'))


def test_short_band_and_missing_noise():
    r=extract_features(np.sin(np.arange(50)),100)
    assert 'lf:too_short_for_band' in r['qc_flags']
    assert np.isnan(r['values']['lf.peak_time_s'])
    r=extract_features(signal()[1],100,noise=np.array([np.nan]*100))
    assert 'invalid_noise' in r['qc_flags']
    assert np.isnan(r['values']['full.threshold_onset_s'])
    with pytest.raises(ValueError): extract_features(np.ones((3,5)),100)
    with pytest.raises(ValueError): extract_features(np.ones(20,dtype=complex),100)


def test_endpoint_only_and_noise_filter_padding():
    x=np.zeros(100);x[0]=1;x[-1]=-1
    with np.errstate(all='raise'):
        r=extract_features(x,100)
    assert 'cepstral:zero_windowed_amplitude' in r['qc_flags']
    assert np.isnan(r['values']['cepstral.peak_coefficient'])
    cfg=FeatureConfig(lf_band=(20,30),hf_band=(31,40),edge_cycles=.01,smoothing_seconds=.01)
    r=extract_features(signal()[1],100,noise=np.arange(16.),config=cfg)
    assert 'lf:noise_too_short_for_filter' in r['qc_flags']
