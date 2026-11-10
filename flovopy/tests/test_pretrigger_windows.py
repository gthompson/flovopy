import unittest
import numpy as np
import pandas as pd
from obspy import Trace,Stream,UTCDateTime
from flovopy.processing.detection import PretriggerSettings as Settings,detect_pretrigger_window

class WindowStrategyTests(unittest.TestCase):
    def tr(self,station,onset=None,late=None,channel='EHZ',seed=1):
        x=np.random.default_rng(seed).normal(size=8000)
        if onset is not None:x[int(onset*100):]*=10
        if late is not None:x[int(late*100):]*=10
        return Trace(x,header=dict(sampling_rate=100.,network='XB',station=station,channel=channel,starttime=UTCDateTime(1991,6,3)))
    def run_stream(self,traces):
        result=detect_pretrigger_window(Stream(traces))
        result['network']=pd.DataFrame([result['network']])
        return result
    def test_clear_multi_station_event(self):
        r=self.run_stream([self.tr('AAA',10,seed=3),self.tr('BBB',11,seed=4)])
        self.assertEqual(r['network'].iloc[0].network_status,'candidate_cluster')
        self.assertEqual(r['network'].iloc[0].network_station_support,2)
        for q in r['qc'].itertuples():
            self.assertTrue(9<=q.selected_onset_record_s<=12)
            self.assertLessEqual(q.proposed_noise_end_record_s,q.selected_onset_record_s-1+1e-8)
            self.assertGreaterEqual(q.proposed_noise_end_record_s,5)
            self.assertEqual(q.noise_acceptance,'UNREVIEWED')
    def test_weak_station_late_excursion_falls_back_to_five(self):
        r=self.run_stream([self.tr('AAA',10,seed=3),self.tr('BBB',11,seed=4),self.tr('CAB',late=70,seed=5)])
        q=r['qc'].set_index('trace_id').loc['XB.CAB..EHZ']
        self.assertTrue(q.amplitude_onset_record_s>60)
        self.assertTrue(np.isnan(q.selected_onset_record_s));self.assertEqual(q.proposed_noise_end_record_s,5)
    def test_same_station_components_cannot_create_network(self):
        r=self.run_stream([self.tr('CAB',10,seed=3),self.tr('CAB',10,channel='EHN',seed=4)])
        self.assertEqual(r['network'].iloc[0].network_status,'unresolved')
        self.assertTrue((r['qc'].proposed_noise_end_record_s==5).all())
    def test_earlier_weak_rise_beats_later_strong_phase(self):
        early=self.tr('CCC',seed=6);early.data[850:1400]*=2.2;early.data[1400:]*=12
        r=self.run_stream([self.tr('AAA',9,seed=3),self.tr('BBB',9.5,seed=4),early])
        q=r['qc'].set_index('trace_id').loc['XB.CCC..EHZ']
        self.assertTrue(q.amplitude_onset_record_s>12)
        self.assertLess(q.selected_onset_record_s,10);self.assertLess(q.proposed_noise_end_record_s,9)
    def test_short_pulse_limits_noise_prefix(self):
        pulse=self.tr('CCC',10,seed=6);pulse.data[750]=50
        r=self.run_stream([self.tr('AAA',10,seed=3),self.tr('BBB',11,seed=4),pulse])
        q=r['qc'].set_index('trace_id').loc['XB.CCC..EHZ']
        self.assertLessEqual(q.proposed_noise_end_record_s,6.5)
        self.assertIn('earlier_',q.qc_flags)
    def test_initial_padding_is_not_valid_noise(self):
        bad=self.tr('CCC',10,seed=6);bad.data[:100]=0
        r=self.run_stream([self.tr('AAA',10,seed=3),self.tr('BBB',11,seed=4),bad])
        q=r['qc'].set_index('trace_id').loc['XB.CCC..EHZ']
        self.assertTrue(np.isnan(q.proposed_noise_end_record_s));self.assertEqual(q.noise_proposal_status,'INVALID_INITIAL_NOISE')
    def test_initial_pulse_is_flagged_despite_five_second_assumption(self):
        bad=self.tr('CCC',10,seed=6);bad.data[200]=50
        r=self.run_stream([self.tr('AAA',10,seed=3),self.tr('BBB',11,seed=4),bad])
        q=r['qc'].set_index('trace_id').loc['XB.CCC..EHZ']
        self.assertTrue(np.isnan(q.proposed_noise_end_record_s));self.assertIn('initial_reference_contains_large_transient',q.qc_flags)
    def test_minimum_and_guard_conflict_stays_invalid(self):
        r=self.run_stream([self.tr('AAA',5.25,seed=3),self.tr('BBB',5.25,seed=4)])
        self.assertTrue(r['qc'].proposed_noise_end_record_s.isna().all())
        self.assertTrue(r['qc'].noise_proposal_status.str.startswith('CONFLICT').all())
    def test_noise_only_record_is_unresolved(self):
        r=self.run_stream([self.tr('AAA',seed=3),self.tr('BBB',seed=4)])
        self.assertEqual(r['network'].iloc[0].network_status,'unresolved')
        self.assertTrue((r['qc'].proposed_noise_end_record_s==5).all())
    def test_hilbert_signed_noise_is_retained_and_smoothed(self):
        fs=100;t=np.arange(3000)/fs
        tr=Trace(np.cos(2*np.pi*5*t)+.8*np.cos(2*np.pi*12*t),header={'sampling_rate':fs})
        r=detect_pretrigger_window(tr);h=r['details'][0]['hilbert'];negative=(h.instantaneous_hz_raw<0)&~h.edge_excluded&~h.low_envelope
        self.assertTrue(negative.any());self.assertTrue(h.loc[negative,'instantaneous_hz_valid'].notna().all())
        noise=h[(h.time_trace_s>=1)&(h.time_trace_s<5)]
        self.assertGreater(noise.instantaneous_hz_smoothed.notna().mean(),.8)


class PublicPretriggerAPITests(unittest.TestCase):
    def trace(self, station='AAA', onset=10, fs=100, seed=3):
        tr=WindowStrategyTests().tr(station,onset,seed=seed)
        tr.stats.sampling_rate=fs
        return tr

    def test_vdap_entry_point_uses_same_engine_and_settings(self):
        from flovopy.core import vdap
        tr=self.trace();cfg=Settings(network_min_stations=1,compute_hilbert=False)
        generic=detect_pretrigger_window(tr,settings=cfg)
        wrapped=vdap.detect_pretrigger_window(tr,settings=cfg)
        pd.testing.assert_frame_equal(generic['qc'],wrapped['qc'])
        self.assertEqual(generic['algorithm_version'],wrapped['algorithm_version'])

    def test_single_station_is_explicit_and_shorter_reference_allowed(self):
        tr=self.trace(onset=4)
        cfg=Settings(minimum_noise_s=2,reference_end_s=2,lta_s=1.5,
                     network_min_stations=1,compute_hilbert=False)
        result=detect_pretrigger_window(tr,settings=cfg)
        q=result['qc'].iloc[0]
        self.assertEqual(result['network']['network_station_support'],1)
        self.assertTrue(3<=q.selected_onset_trace_s<=5)
        self.assertIn('FIRST_2_SECONDS',q.reference_status)
        self.assertGreaterEqual(q.proposed_noise_end_trace_s,2)
        self.assertEqual(detect_pretrigger_window(tr)['network']['network_status'],'unresolved')

    def test_onset_support_is_finite_first_evidence_not_entire_event(self):
        result=detect_pretrigger_window(self.trace(),settings=Settings(network_min_stations=1))
        q=result['qc'].iloc[0]
        self.assertLess(q.onset_support_start_trace_s,q.onset_support_end_trace_s)
        self.assertLessEqual(q.onset_support_end_trace_s-q.onset_support_start_trace_s,1)
        self.assertTrue(np.isnan(q.proposed_signal_end_trace_s))

    def test_offset_alignment_and_no_mutation(self):
        a=self.trace();b=self.trace('BBB',onset=9,seed=4);b.stats.starttime+=1
        stream=Stream([a,b]);before=stream.copy()
        result=detect_pretrigger_window(stream)
        for q in result['qc'].itertuples():
            self.assertAlmostEqual(q.selected_onset_record_s,q.selected_onset_trace_s+q.trace_start_offset_record_s)
            self.assertAlmostEqual(q.onset_support_end_record_s,q.onset_support_end_trace_s+q.trace_start_offset_record_s)
        for original,saved in zip(stream,before):
            np.testing.assert_array_equal(original.data,saved.data);self.assertEqual(original.stats,saved.stats)

    def test_optional_diagnostics_and_fifty_hz(self):
        tr=self.trace(fs=50)
        result=detect_pretrigger_window(tr,settings=Settings(network_min_stations=1,compute_hilbert=False,compute_frequency=False))
        self.assertTrue(result['details'][0]['hilbert'].empty)
        self.assertTrue(result['details'][0]['frequency'].empty)
        self.assertAlmostEqual(result['qc'].iloc[0].fine_window_actual_s,.24)
        self.assertTrue(np.isfinite(result['qc'].iloc[0].selected_onset_trace_s))
        # Default diagnostics also support rounded, whole-sample steps at 50 Hz.
        self.assertFalse(detect_pretrigger_window(tr)['details'][0]['frequency'].empty)

    def test_contaminated_reference_cannot_vote_for_network(self):
        a=self.trace();b=self.trace('BBB',seed=4);b.data[200]=100
        result=detect_pretrigger_window(Stream([a,b]))
        self.assertEqual(result['network']['network_status'],'unresolved')
        q=result['qc'].set_index('trace_id').loc['XB.BBB..EHZ']
        self.assertTrue(np.isnan(q.selected_onset_trace_s))
        self.assertEqual(q.noise_proposal_status,'INVALID_INITIAL_NOISE')

    def test_empty_stream_and_invalid_types(self):
        result=detect_pretrigger_window(Stream())
        self.assertTrue(result['qc'].empty);self.assertTrue(result['evidence'].empty)
        self.assertIsNone(result['reference_time'])
        with self.assertRaises(TypeError):detect_pretrigger_window('waveform.mseed')
        with self.assertRaises(TypeError):detect_pretrigger_window(Stream(),settings={})
        with self.assertRaises(ValueError):Settings(network_min_stations=0)

    def test_gradual_rise_precedes_large_amplitude_phase(self):
        traces=[]
        for station,seed in [('AAA',3),('BBB',4)]:
            tr=self.trace(station,onset=None,seed=seed)
            t=np.arange(len(tr))/tr.stats.sampling_rate
            tr.data*=1+14*np.clip((t-8)/12,0,1)
            traces.append(tr)
        result=detect_pretrigger_window(Stream(traces))
        for q in result['qc'].itertuples():
            self.assertTrue(8<=q.selected_onset_trace_s<q.amplitude_onset_trace_s)
            self.assertLessEqual(q.proposed_noise_end_trace_s,q.selected_onset_trace_s-1)

    def test_gap_before_arrival_limits_proposal_without_filling_input(self):
        a=self.trace();b=self.trace('BBB',seed=4)
        a.data=np.ma.array(a.data,mask=False);a.data.mask[750:775]=True
        result=detect_pretrigger_window(Stream([a,b]))
        q=result['qc'].set_index('trace_id').loc['XB.AAA..EHZ']
        self.assertLessEqual(q.proposed_noise_end_trace_s,6.5)
        self.assertTrue(a.data.mask[750:775].all())

if __name__=='__main__':unittest.main()
