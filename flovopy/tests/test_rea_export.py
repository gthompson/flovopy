from pathlib import Path
import tempfile,unittest,os,json
from unittest.mock import patch
from obspy import UTCDateTime,read_events
from obspy.core.event import Catalog,Event,Origin,Pick,WaveformStreamID,ResourceIdentifier
from flovopy.seisanio.rea_export import export_catalog_to_rea

T=UTCDateTime('1991-06-03T12:00:00')
def event(key,time=T,origin=True):
    e=Event(resource_id=ResourceIdentifier('smi:local/event/'+key))
    if origin:
        e.origins=[Origin(time=time,latitude=15,longitude=120,depth=3000,
                         resource_id=ResourceIdentifier('smi:local/origin/'+key))]
        e.preferred_origin_id=e.origins[0].resource_id
    return e

class NordicRepairTests(unittest.TestCase):
    def test_seconds_magnitude_type_and_rms(self):
        from obspy.core.event import Magnitude, OriginQuality
        cases=[('1991-06-03T12:00:58.08','58.1','58.080'),('1991-06-03T12:59:59.96','0.0','59.960'),
               ('1991-06-03T12:00:05.012','5.0','5.012')]
        for when,sec1,secH in cases:
            with self.subTest(when=when), tempfile.TemporaryDirectory() as tmp:
                e=event('a',UTCDateTime(when));e.origins[0].quality=OriginQuality(standard_error=0.12)
                e.magnitudes=[Magnitude(mag=1.4,origin_id=e.origins[0].resource_id)]
                r=export_catalog_to_rea(Catalog(events=[e]),tmp,high_accuracy=True)
                lines=r['written'][0].read_text().splitlines()
                one=next(l for l in lines if l[79:80]=='1');h=next(l for l in lines if l[79:80]=='H')
                self.assertEqual(one[16:20].strip(),sec1)
                self.assertEqual(h[16:22].strip(),secH)
                self.assertEqual(one[59],' ')                  # unknown magnitude type, not 'W'
                self.assertEqual(one[51:55].strip(),'0.1')      # RMS from origin.quality
                back=read_events(str(r['written'][0]),format='NORDIC')[0].origins[0].time
                self.assertLess(abs(back-UTCDateTime(when)),0.051)

    def test_ties_round_half_up(self):
        from flovopy.seisanio.rea_export import _round_time
        for when,want in [('1991-08-16T17:17:31.55','1991-08-16T17:17:31.6'),('1991-05-24T14:22:30.85','1991-05-24T14:22:30.9')]:
            self.assertEqual(_round_time(UTCDateTime(when),100_000_000),UTCDateTime(want))

    def test_minute_carry(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=export_catalog_to_rea(Catalog(events=[event('a',UTCDateTime('1991-06-03T23:59:59.97'))]),tmp)
            one=next(l for l in r['written'][0].read_text().splitlines() if l[79:80]=='1')
            self.assertEqual(one[1:20],'1991  6 4  0 0  0.0')


class REAExportTests(unittest.TestCase):
    def test_write_reuse_select_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=event('a');cat=Catalog(events=[e]);before=e.copy();cwd=os.getcwd()
            first=export_catalog_to_rea(cat,tmp,write_select=True,waveform_files=lambda e:['/tmp/wave.M.PNTBO_001'])
            self.assertFalse(first['failed']);self.assertEqual(len(first['written']),1)
            path=first['written'][0];self.assertEqual(path.name,'03-1200-00L.S199106')
            self.assertIn('wave.M.PNTBO_001',path.read_text())
            self.assertEqual(len(read_events(str(path),format='NORDIC')),1)
            second=export_catalog_to_rea(cat,tmp,write_select=True,waveform_files=lambda e:['/tmp/wave.M.PNTBO_001'])
            self.assertEqual(second['reused'],[path]);self.assertFalse(second['written'])
            self.assertEqual(e,before);self.assertEqual(os.getcwd(),cwd)
            self.assertTrue(first['select_path'].is_file())

    def test_collision_policies_and_rollover(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat=Catalog(events=[event('a'),event('b')])
            result=export_catalog_to_rea(cat,Path(tmp)/'strict')
            self.assertEqual(len(result['written']),1);self.assertEqual(len(result['failed']),1)
            shifted=export_catalog_to_rea(cat,Path(tmp)/'shift',collision='shift')
            self.assertEqual(len(shifted['written']),2)
            self.assertEqual(shifted['rows'][1]['filename_offset_s'],1)
            self.assertTrue(all(p.suffix=='.S199106' for p in shifted['written']))
            reread=read_events(str(shifted['written'][1]),format='NORDIC')
            self.assertEqual(reread[0].origins[0].time,T)
            again=export_catalog_to_rea(cat,Path(tmp)/'shift',collision='shift')
            self.assertEqual(len(again['reused']),2)

    def test_anchors_timeless_and_unique_origin_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat=Catalog(events=[event('a',origin=False),event('b',origin=False),event('c',origin=False)])
            sources={str(cat[0].resource_id):T,str(cat[1].resource_id):T+1}
            callback=lambda e:(sources[str(e.resource_id)],'caller_anchor') if str(e.resource_id) in sources else None
            result=export_catalog_to_rea(cat,tmp,anchor_time=callback)
            self.assertEqual(len(result['written']),2);self.assertEqual(len(result['skipped']),1)
            self.assertTrue(all(not e.origins for e in cat))
            reused=export_catalog_to_rea(cat,tmp,anchor_time=callback)
            self.assertEqual(len(reused['reused']),2)
            self.assertTrue(all(r['origin_source']=='caller_anchor' for r in result['rows']))

    def test_changed_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat=Catalog(events=[event('a')]);first=export_catalog_to_rea(cat,tmp);path=first['written'][0]
            path.write_text('modified')
            rejected=export_catalog_to_rea(cat,tmp)
            self.assertEqual(len(rejected['failed']),1);self.assertEqual(path.read_text(),'modified')
            repaired=export_catalog_to_rea(cat,tmp,overwrite=True)
            self.assertEqual(len(repaired['written']),1)
            cat[0].origins[0].depth=4000
            self.assertEqual(len(export_catalog_to_rea(cat,tmp)['failed']),1)
            self.assertEqual(len(export_catalog_to_rea(cat,tmp,overwrite=True)['written']),1)

    def test_untracked_files_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'1991/06';directory.mkdir(parents=True)
            existing=directory/'03-1200-00L.S199106';existing.write_text('untracked')
            result=export_catalog_to_rea(Catalog(events=[event('a')]),root,overwrite=True)
            self.assertEqual(len(result['failed']),1);self.assertEqual(existing.read_text(),'untracked')
            with patch.object(Catalog,'write',side_effect=OSError('simulated failure')):
                failed=export_catalog_to_rea(Catalog(events=[event('b',T+3)]),root)
            self.assertEqual(len(failed['failed']),1)
            self.assertFalse((directory/'03-1200-03L.S199106').exists())
            self.assertFalse(list(directory.glob('.nordic-*')))

    def test_new_format_and_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=export_catalog_to_rea(Catalog(events=[event('a')]),tmp,database='PNTBO',nordic_format='NEW')
            self.assertFalse(result['failed']);self.assertIn('PNTBO',str(result['written'][0]))
            self.assertEqual(len(read_events(str(result['written'][0]),format='NORDIC')),1)
            with self.assertRaises(ValueError):export_catalog_to_rea(Catalog(),tmp,database='../escape')


class ManifestAndWarningTests(unittest.TestCase):
    def _catalog(self,n,picks=False):
        events=[event(f'e{i}',T+10*i) for i in range(n)]
        if picks:
            for e in events:
                e.picks=[Pick(time=e.origins[0].time+1,phase_hint='P',
                              waveform_id=WaveformStreamID('XB','PPO','','EHZ'))]
        return Catalog(events=events)

    def test_manifest_written_once_per_run(self):
        import flovopy.seisanio.rea_export as rea
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rea,'_atomic_json',wraps=rea._atomic_json) as save:
                result=export_catalog_to_rea(self._catalog(5),tmp)
            self.assertEqual(len(result['written']),5);self.assertEqual(save.call_count,1)
            entries=json.loads(result['manifest_path'].read_text())['events']
            self.assertEqual(len(entries),5)
            with patch.object(rea,'_atomic_json',wraps=rea._atomic_json) as save:
                again=export_catalog_to_rea(self._catalog(5),tmp)
            self.assertEqual(len(again['reused']),5);save.assert_not_called()

    def test_checkpoints_while_running(self):
        import flovopy.seisanio.rea_export as rea
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rea,'_atomic_json',wraps=rea._atomic_json) as save:
                export_catalog_to_rea(self._catalog(3),tmp,checkpoint_seconds=0)
            self.assertEqual(save.call_count,3)

    def test_interrupt_still_records_written_files(self):
        import flovopy.seisanio.rea_export as rea
        real=rea.repair_nordic_header_times;calls=[]
        def stop_on_third(text,origins):
            calls.append(1)
            if len(calls)==3:
                raise KeyboardInterrupt
            return real(text,origins)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rea,'repair_nordic_header_times',side_effect=stop_on_third):
                with self.assertRaises(KeyboardInterrupt):
                    export_catalog_to_rea(self._catalog(5),tmp)
            manifest=Path(tmp).resolve()/'.flovopy_rea_manifest.json'
            self.assertEqual(len(json.loads(manifest.read_text())['events']),2)
            rerun=export_catalog_to_rea(self._catalog(5),tmp)
            self.assertEqual((len(rerun['reused']),len(rerun['written']),len(rerun['failed'])),(2,3,0))

    def test_writer_warning_issued_once_with_count(self):
        import warnings
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=export_catalog_to_rea(self._catalog(4,picks=True),tmp)
        self.assertEqual(len(result['written']),4)
        mode=[str(w.message) for w in caught if 'Evaluation mode' in str(w.message)]
        self.assertEqual(mode,['Evaluation mode None is not mappable (Nordic writer; 4 events)'])

if __name__=='__main__':unittest.main()
