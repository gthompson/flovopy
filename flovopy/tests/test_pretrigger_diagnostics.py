import unittest
import numpy as np
from flovopy.processing.event_windows import PretriggerSettings as Settings,hilbert_diagnostic,frequency_checks,_summarize_hilbert

class HilbertTests(unittest.TestCase):
    def test_known_sinusoid_and_sample_midpoints(self):
        fs=100;t=np.arange(3000)/fs;x=np.cos(2*np.pi*6*t)
        out=hilbert_diagnostic(x,fs,Settings())
        self.assertEqual(len(out),len(x)-1);self.assertAlmostEqual(out.time_trace_s.iloc[0],.005)
        self.assertAlmostEqual(out.instantaneous_hz_valid.median(),6,places=2)
        self.assertAlmostEqual(out.envelope_counts.median(),1,places=2)
        self.assertTrue(out.edge_excluded.iloc[0]);self.assertTrue(out.edge_excluded.iloc[-1])
    def test_chirp_tracks_frequency(self):
        fs=100;t=np.arange(3000)/fs;phase=2*np.pi*(3*t+.15*t*t)
        out=hilbert_diagnostic(np.cos(phase),fs,Settings())
        valid=out[(out.time_trace_s>2)&(out.time_trace_s<28)]
        expected=3+.3*valid.time_trace_s
        self.assertLess(np.nanmedian(np.abs(valid.instantaneous_hz_valid-expected)),.1)
    def test_gaps_segment_independently(self):
        fs=100;t=np.arange(3000)/fs;x=np.cos(2*np.pi*6*t);x[900:1100]=np.nan
        out=hilbert_diagnostic(x,fs,Settings())
        self.assertFalse(((out.time_trace_s>=8.995)&(out.time_trace_s<=11)).any())
        around=out[(out.time_trace_s>8)&(out.time_trace_s<12)]
        self.assertTrue(around.instantaneous_hz_valid.isna().all())
    def test_constant_data_screened(self):
        out=hilbert_diagnostic(np.zeros(2000),100,Settings())
        self.assertTrue(out.low_envelope.all());self.assertTrue(out.instantaneous_hz_valid.isna().all())
        empty=hilbert_diagnostic(np.zeros(3),100,Settings())
        self.assertTrue(empty.empty);self.assertTrue(all(v.dtype.kind!='O' for _,v in empty.items()))
        t=np.arange(3000)/100
        mixed=hilbert_diagnostic(np.cos(2*np.pi*5*t)+.8*np.cos(2*np.pi*12*t),100,Settings())
        negative=mixed.instantaneous_hz_raw<0
        eligible=negative&~mixed.edge_excluded&~mixed.low_envelope
        self.assertTrue(eligible.any());self.assertTrue(mixed.loc[eligible,'instantaneous_hz_valid'].notna().all())
    def test_no_input_mutation_and_disable_smoothing(self):
        fs=100;t=np.arange(3000)/fs;x=np.cos(2*np.pi*6*t);before=x.copy()
        out=hilbert_diagnostic(x,fs,Settings(hilbert_smooth_s=0))
        np.testing.assert_equal(x,before)
        np.testing.assert_equal(out.instantaneous_hz_valid.to_numpy(),out.instantaneous_hz_smoothed.to_numpy())
    def test_summary_matches_screened_samples(self):
        fs=100;t=np.arange(3000)/fs;x=np.cos(2*np.pi*6*t);cfg=Settings()
        h=hilbert_diagnostic(x,fs,cfg);f,_=frequency_checks(x,fs,cfg);summary=_summarize_hilbert(f,h,fs)
        self.assertTrue((summary.hilbert_valid_fraction<=1).all())
        self.assertAlmostEqual(summary.hilbert_median_hz.median(),6,places=2)
    def test_invalid_config(self):
        with self.assertRaises(ValueError):Settings(hilbert_envelope_floor_fraction=2)
        with self.assertRaises(ValueError):Settings(hilbert_edge_s=-1)
