"""Independent format fixtures and preservation checks for legacy BOB readers."""
import tempfile,struct,zipfile,unittest
from pathlib import Path
import numpy as np
from flovopy.legacy.bob import read_ssam,read_mudflow,read_cospec,read_counts,read_tilt,read_radio_maps

class BOBReaderTests(unittest.TestCase):
    def text(self,folder,name,body):
        path=Path(folder)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body);return path
    def test_ssam_folder_zip_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'input';file=self.text(folder,'PPE1/day.DAT','SSAM\n01 JUN 91 00:00 '+ ' '.join(['-998']+['2']*15)+'\n01 JUN 91 00:01 '+' '.join(['-999']*16)+'\n')
            before=file.read_bytes();a=read_ssam(folder)
            self.assertEqual(len(a.data),2);self.assertEqual(a.data.value_01_raw.iloc[0],-998)
            self.assertTrue(a.data.value_01_missing.all());self.assertEqual(a.coverage().nonmissing_records.sum(),1)
            self.assertIsNone(a.data.recorded_time.dt.tz)
            archive=root/'data.zip'
            with zipfile.ZipFile(archive,'w') as z:z.write(file,'PPE1/day.DAT')
            b=read_ssam(archive);np.testing.assert_allclose(a.data.value_02,b.data.value_02,equal_nan=True)
            self.assertEqual(file.read_bytes(),before)
    def test_mudflow_bad_records_and_duplicates_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.text(tmp,'day.DAT','01 JUL 91 00:00 1 2 3\n01 JUL 91 00:00 4 5 6\n01 JUL 91 00:01 7 8\n01 JUL 91 25:00 1 2 3\n01 JUL malformed\n')
            result=read_mudflow(p)
            self.assertEqual(len(result.data),4);self.assertEqual(len(result.errors),3)
            self.assertEqual(result.data.duplicate_timestamp.sum(),2)
            self.assertEqual(result.data.recorded_time.isna().sum(),1)
            with self.assertRaises(ValueError):read_mudflow(p,strict=True)
    def test_cospec_adjacent_time_value_and_counts_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.text(tmp,'cos.DAT','13 MAY 91 00:0013361.0\n13 MAY 91 00:00 493.0\n')
            d=read_cospec(p).data;self.assertEqual(d.value_01.tolist(),[13361.,493.]);self.assertTrue(d.duplicate_timestamp.all())
            p=self.text(tmp,'counts.DAT','13 MAY 91 00:00\n13 MAY 91 01:00 0\n')
            d=read_counts(p).data;self.assertTrue(d.value_01_missing.iloc[0]);self.assertEqual(d.value_01.iloc[1],0)
    def test_annual_leap_year_sign_and_unmapped_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'TEST20.DAT';raw=np.full(366*144,-998,dtype='<f4');raw[0]=-2.;raw[1]=3.
            p.write_bytes(struct.pack('<I',576)+bytes(572)+raw.tobytes())
            result=read_tilt(p,year=2020,century=2000)
            self.assertEqual(len(result.data),366*144);self.assertEqual(result.data.value_01.iloc[0],-2)
            self.assertEqual(result.coverage().nonmissing_records.iloc[0],2)
            self.assertEqual(result.metadata['channel_metadata'],{})
            with self.assertRaises(ValueError):read_tilt(p,year=2020)
    def test_export_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.text(tmp,'one.DAT','01 JUL 91 00:00 1 2 3\n');result=read_mudflow(p)
            paths=result.write(Path(tmp)/'out');self.assertTrue(all(p.exists() for p in paths.values()))
            with self.assertRaises(FileExistsError):result.write(Path(tmp)/'out')
    def test_radio_undated_is_error_not_fake_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.text(tmp,'SEIS_MAP.OLD',' 01   PI2Z  station\n')
            result=read_radio_maps(p);self.assertEqual(len(result.errors),1);self.assertTrue(result.data.empty)

class BOBPlotTests(unittest.TestCase):
    def test_ssam_cells_preserve_gaps_and_refuse_duplicates(self):
        from flovopy.legacy.bob import plot_ssam
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'one.DAT'
            path.write_text('01 JUL 91 00:00 '+' '.join(['1']*16)+'\n01 JUL 91 00:05 '+' '.join(['2']*16)+'\n')
            dataset=read_ssam(path)
            ax,collection=plot_ssam(dataset,series='.',interval_seconds=60)
            self.assertEqual(len(collection.get_paths()),32)
            plt.close(ax.figure)
            dataset.data=__import__('pandas').concat([dataset.data,dataset.data.iloc[:1]],ignore_index=True)
            with self.assertRaises(ValueError):plot_ssam(dataset,series='.',interval_seconds=60)

if __name__=='__main__':unittest.main()
