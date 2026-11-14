import tempfile,unittest,struct,warnings
from pathlib import Path
import numpy as np
from obspy import UTCDateTime
from obspy.io.dmx.core import descript_trace_dtypes
from flovopy.core.vdap import read_dmx_file,convert_dmx_to_seisan
from flovopy.core.dmx_recovery import inspect_prefix

def record(station='AAAZ'):
    desc=np.zeros(1,dtype=descript_trace_dtypes)
    desc['network']=b'unk';desc['st_name']=station.encode();desc['component']=b'Z';desc['begintime']=float(UTCDateTime('1991-06-01'));desc['datatype']=b's';desc['length']=4;desc['rate']=100
    data=np.array([2048,2049,2050,2047],dtype='<u2')
    return struct.pack('<cchii',b'S',b'6',7,64,len(data.tobytes()))+desc.tobytes()+data.tobytes()

class RecoveryTests(unittest.TestCase):
    def test_regular_read_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'file.DMX';p.write_bytes(record())
            s=read_dmx_file(p,fix=False)
            self.assertEqual(len(s),1);self.assertFalse(hasattr(s[0].stats,'dmx_recovery'))
            np.testing.assert_array_equal(s[0].data,[2048,2049,2050,2047])
    def test_prefix_guard_warning_and_single_offset(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'file.DMX';blob=record()+record('BBBZ')+record('CCCZ')+b'\xff'*12;p.write_bytes(blob)
            with warnings.catch_warnings(record=True) as w:
                s=read_dmx_file(p,defaultnet='XB')
            self.assertEqual(len(s),2);self.assertTrue(w);self.assertTrue(s[0].stats.dmx_recovery.partial)
            self.assertTrue(s[0].stats.dmx_recovery.discarded_guard_trace)
            self.assertEqual(s[0].stats.network,'XB');np.testing.assert_array_equal(s[0].data,[0,1,2,-1]);self.assertEqual(p.read_bytes(),blob)
            self.assertFalse(read_dmx_file(p,recover_partial=False))
    def test_no_resynchronization(self):
        stop,traces,reason=inspect_prefix(b'\xff'*12+record())
        self.assertEqual(stop,0);self.assertFalse(traces)
    def test_truncated_payload_excluded(self):
        blob=record()+record('BBBZ')[:-1]
        stop,traces,reason=inspect_prefix(blob)
        self.assertEqual(len(traces),1);self.assertEqual(reason,'structure_exceeds_file')
    def test_converter_flags_partial_exports(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);raw=p/'raw';raw.mkdir();(raw/'91060100.DMX').write_bytes(record()+record('BBBZ')+record('CCCZ')+b'\xff'*12)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');r=convert_dmx_to_seisan(raw,p/'WAV',network='XB',database='TEST',recover_partial=True)
            self.assertEqual(len(r['waveform_index']),1);self.assertTrue(r['waveform_index'].partial_recovery.iloc[0]);self.assertIn('validated_prefix',r['waveform_index'].recovery_details.iloc[0])
if __name__=='__main__':unittest.main()
