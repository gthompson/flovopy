"""Single-trace pretrigger analysis: frequency checks, onsets, one-second metrics and QC flags.

Moved from the Pinatubo analysis tests; these exercise the internal per-trace engine
behind detect_pretrigger_window.
"""
import unittest
import numpy as np
from obspy import Trace,UTCDateTime
from flovopy.processing.event_windows import PretriggerSettings as Settings,_analyse_trace as analyse_trace,frequency_checks

class FrequencyTests(unittest.TestCase):
    def test_tone_frequency_and_crossings(self):
        fs=100;t=np.arange(2000)/fs;x=np.sin(2*np.pi*6*t)
        f,q=frequency_checks(x,fs,Settings())
        self.assertAlmostEqual(f.centroid_hz.median(),6,places=1)
        self.assertLess(abs(f.zero_crossing_hz.median()-6),.6)
        self.assertTrue(np.isnan(q['frequency_change_trace_s']))
    def test_frequency_only_change_without_amplitude_increase(self):
        fs=100;t=np.arange(3000)/fs;phase=2*np.pi*np.cumsum(np.where(t<10,3,12))/fs
        tr=Trace(np.sin(phase),header={'sampling_rate':fs})
        r=analyse_trace(tr)
        self.assertTrue(np.isnan(r['qc']['amplitude_onset_trace_s']))
        self.assertTrue(10<=r['qc']['frequency_change_trace_s']<=12)
        self.assertEqual(r['qc']['proposed_noise_end_trace_s'],5)
    def test_amplitude_increase_noise_guard(self):
        fs=100;t=np.arange(3000)/fs;x=np.sin(2*np.pi*6*t)*np.where(t<10,1,10)
        r=analyse_trace(Trace(x,header={'sampling_rate':fs}));q=r['qc']
        self.assertEqual(q['proposed_noise_end_trace_s'],q['amplitude_onset_trace_s']-1)
        self.assertGreaterEqual(q['proposed_noise_end_trace_s'],5)
        self.assertTrue(np.isnan(q['frequency_change_trace_s']))
    def test_guard_conflict_not_silently_forced(self):
        fs=100;t=np.arange(2000)/fs;x=np.sin(2*np.pi*6*t)*np.where(t<5.25,1,10)
        q=analyse_trace(Trace(x,header={'sampling_rate':fs}))['qc']
        self.assertTrue(np.isnan(q['proposed_noise_end_trace_s']))
        self.assertIn('CONFLICT',q['noise_proposal_status'])
    def test_frequency_gaps_remain_missing(self):
        fs=100;t=np.arange(2000)/fs;x=np.sin(2*np.pi*6*t);x[900:1100]=np.nan
        f,_=frequency_checks(x,fs,Settings())
        self.assertTrue(f.loc[(f.window_start_s<11)&(f.window_end_s>9),'centroid_hz'].isna().all())

class TraceAnalysisTests(unittest.TestCase):
    def trace(self,x):return Trace(np.array(x,dtype=float),header=dict(sampling_rate=100.,network='XB',station='BUG',channel='EHZ',starttime=UTCDateTime(1991,6,3)))
    def test_step_at_ten_seconds(self):
        x=np.random.default_rng(5).normal(size=3000);x[1000:]*=15
        r=analyse_trace(self.trace(x));q=r['qc']
        self.assertTrue(9.75<=q['amplitude_onset_trace_s']<=10.5)
        self.assertTrue(10<=q['sta_lta_onset_trace_s']<=11)
        self.assertEqual(q['trigger_status'],'NOT_MEASURED')
        self.assertEqual(q['reference_status'],'USER_ASSUMPTION_FIRST_5_SECONDS_NOISE')
    def test_metric_definitions_and_partial_tail(self):
        x=np.tile([-1,1],125);r=analyse_trace(self.trace(x));m=r['one_second']
        np.testing.assert_allclose(m.rsam[:2],1);np.testing.assert_allclose(m.rms[:2],1)
        self.assertFalse(m.iloc[-1].complete);self.assertTrue(np.isnan(m.iloc[-1].rms))
    def test_masks_not_silently_filled(self):
        tr=self.trace(np.tile([-1,1],1000));tr.data=np.ma.array(tr.data,mask=False);tr.data.mask[500:600]=True
        r=analyse_trace(tr);self.assertTrue(np.isnan(r['one_second'].iloc[5].rsam))
        self.assertIn('masked_or_nonfinite',r['qc']['qc_flags'])
    def test_flat_prefix_and_constant_unresolved(self):
        tr=self.trace(np.zeros(2000));r=analyse_trace(tr)
        self.assertEqual(r['qc']['flat_prefix_s'],20);self.assertTrue(np.isnan(r['qc']['amplitude_onset_trace_s']))
        self.assertIn('constant_trace',r['qc']['qc_flags'])
    def test_padding_flag(self):
        x=np.random.default_rng(9).normal(size=2000);x[:100]=0
        self.assertIn('possible_padding',analyse_trace(self.trace(x))['qc']['qc_flags'])
    def test_no_time_or_data_changes(self):
        tr=self.trace(np.random.default_rng(1).normal(size=3000));tr.data[100]=np.inf;before=tr.copy();analyse_trace(tr)
        np.testing.assert_equal(tr.data,before.data);self.assertEqual(tr.stats.starttime,before.stats.starttime)
    def test_invalid_settings(self):
        with self.assertRaises(ValueError):Settings(reference_end_s=0)

if __name__=='__main__':unittest.main()
