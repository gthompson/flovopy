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

if __name__=='__main__':unittest.main()
