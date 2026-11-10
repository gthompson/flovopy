from pathlib import Path
from tempfile import TemporaryDirectory
import json,unittest
import numpy as np
from obspy import UTCDateTime
from obspy.core.event import Origin
from flovopy.enhanced.event import EnhancedEvent,EnhancedEventMeta
from flovopy.enhanced.catalog import EnhancedCatalog
from flovopy.enhanced.eventrate import EventRateConfig
from flovopy.enhanced.serialization import json_compatible,flatten_metrics

class EnhancedMetricTests(unittest.TestCase):
    def test_strict_json_preserves_arrays_nulls_and_duplicate_trace_ids(self):
        event=EnhancedEvent(resource_id='smi:local/test',meta=EnhancedEventMeta(metrics={'zero':np.float64(0)},traces=[
            dict(id='XB.BUG..EHZ',trace_index=i,window='noise' if i==0 else 'event',
                 metrics={'new_key':np.array([0,1,np.nan,np.inf]),'flag':np.bool_(False)}) for i in range(2)]))
        with TemporaryDirectory() as root:
            event.save(root,'record',write_quakeml=False)
            text=(Path(root)/'record.json').read_text();self.assertNotIn('NaN',text);self.assertNotIn('Infinity',text)
            loaded=EnhancedEvent.load(str(Path(root)/'record'),require_quakeml=False)
            self.assertEqual(loaded.meta.traces[0]['metrics']['new_key'],[0.,1.,None,None])
            self.assertIs(loaded.meta.traces[0]['metrics']['flag'],False)
            self.assertEqual(loaded.event_id,event.event_id);self.assertEqual(len(loaded.origins),0)
            with self.assertRaises(FileNotFoundError):EnhancedEvent.load(str(Path(root)/'record'))
            catalog=EnhancedCatalog.from_sidecars([Path(root)/'record.json'])
            frame=catalog.metrics_dataframe();self.assertEqual(len(frame),2)
            self.assertEqual(json.loads(frame['trace.metrics.new_key'].iloc[0]),[0.,1.,None,None])
            self.assertEqual(frame['trace.metrics.flag'].iloc[0],False)
    def test_quakeml_preserves_real_origin_but_metrics_stay_in_json(self):
        event=EnhancedEvent(resource_id='smi:local/test',origins=[Origin(time=UTCDateTime(1991,6,1),latitude=15.,longitude=120.)],meta=EnhancedEventMeta(metrics={'custom':3}))
        with TemporaryDirectory() as root:
            event.save(root,'event');loaded=EnhancedEvent.load(str(Path(root)/'event'))
            self.assertEqual(loaded.meta.metrics['custom'],3);self.assertEqual(len(loaded.origins),1)
    def test_dynamic_new_fields_export_without_schema_allowlist(self):
        event=EnhancedEvent(meta=EnhancedEventMeta(traces=[{'window':'event','metrics':{'unanticipated_metric':7,'nested':{'x':[1,2,3]}}}]))
        catalog=EnhancedCatalog(events=[event],records=[event]);frame=catalog.metrics_dataframe()
        self.assertEqual(frame['trace.metrics.unanticipated_metric'].iloc[0],7)
        self.assertEqual(json.loads(frame['trace.metrics.nested.x'].iloc[0]),[1,2,3])
        self.assertEqual(len(catalog.metrics_dataframe(level='event')),1)
    def test_json_mask_and_collision_are_explicit(self):
        self.assertEqual(json_compatible(np.ma.array([1.,2.],mask=[False,True])),[1.,None])
        with self.assertRaises(ValueError):flatten_metrics({'a.b':1,'a':{'b':2}})
        with self.assertRaises(TypeError):json_compatible(object())
    def test_missing_magnitudes_are_not_imputed_for_record_counts(self):
        events=[]
        for i in range(2):
            events.append(EnhancedEvent(resource_id=f'smi:local/test{i}',meta=EnhancedEventMeta(metrics={'record_reference_recorded_time':'1991-06-01T12:00:00Z'})))
        catalog=EnhancedCatalog(events=events,records=events)
        self.assertTrue(catalog.to_event_rate().df.empty)
        frame=catalog.to_event_rate(EventRateConfig(time_source='recorded_file',include_energy=False)).df
        self.assertEqual(frame.event_count.sum(),2);self.assertEqual(frame.magnitude_count.sum(),0)
        self.assertTrue(frame.average_magnitude.isna().all());self.assertTrue(frame.cumulative_energy.isna().all())
    def test_explicit_sidecar_list_rejects_duplicate_event_versions(self):
        event=EnhancedEvent(resource_id='smi:local/repeated')
        with TemporaryDirectory() as root:
            event.save(root,'one',write_quakeml=False);event.save(root,'two',write_quakeml=False)
            with self.assertRaises(ValueError):EnhancedCatalog.from_sidecars([Path(root)/'one.json',Path(root)/'two.json'])

    def test_enhancedstream_forwards_configured_spectral_bands(self):
        from obspy import Trace,Stream
        from flovopy.enhanced.stream import EnhancedStream
        fs=100;t=np.arange(3000)/fs
        tr=Trace(np.sin(2*np.pi*6*t),header={'sampling_rate':fs,'units':'counts'})
        stream=EnhancedStream(Stream([tr]));stream.ampengfft(allow_uncalibrated=True,compute_spectral=True,
            compute_bandratios=True,compute_sam=False,bands=(1.,3.,3.,15.))
        self.assertEqual(list(stream[0].stats.metrics.spectral.bandratio.bands),[1.,3.,3.,15.])

    def test_eventrate_counts_missing_magnitude_without_inventing_energy(self):
        from obspy.core.event import Magnitude
        events=[EnhancedEvent(origins=[Origin(time=UTCDateTime(1991,6,1))],magnitudes=[Magnitude(mag=2.)]),
                EnhancedEvent(origins=[Origin(time=UTCDateTime(1991,6,1))])]
        catalog=EnhancedCatalog(events=events,records=events)
        rate=catalog.to_event_rate();row=rate.df.iloc[0]
        self.assertEqual(row.event_count,2);self.assertEqual(row.magnitude_count,1)
        self.assertAlmostEqual(row.average_magnitude,2.)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();rate.plot_event_count(ax=ax,use_smoothed=False);plt.close(fig)

if __name__=='__main__':unittest.main()
