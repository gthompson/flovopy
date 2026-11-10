from pathlib import Path
import tempfile,unittest
from obspy import UTCDateTime
from flovopy.core.vdap import parse_hypo71_line,parse_hypo71_file,parse_vdap_phase_line,read_vdap_pha_file

FIXTURES=Path(__file__).parent/'data/vdap'
class VDAPIngestTests(unittest.TestCase):
    def test_historical_pha(self):
        picks,errors=read_vdap_pha_file(FIXTURES/'9106030M.PHA',network='XB',return_diagnostics=True)
        self.assertEqual(len(picks),9);self.assertFalse(errors)
        byid={p['seed_id']:p for p in picks}
        self.assertIn('XB.PPO..EHE',byid);self.assertIn('XB.PPO..EHN',byid)
        self.assertEqual(byid['XB.PIE..EHZ']['time'],UTCDateTime('1991-06-03T12:56:36.37'))
        self.assertEqual(byid['XB.PIE..EHZ']['raw_lineno'],4)
        other=read_vdap_pha_file(FIXTURES/'910610A6.PHA',network='XB')
        p=next(p for p in other if p['station']=='UBO')
        self.assertEqual((p['first_motion'],p['weight']),('D',0))

    def test_gain_orientation_and_blank_orientation(self):
        for sta,channel in [('PI2L','ELZ'),('PPON','EHN'),('PPOE','EHE'),('PI2Z','EHZ')]:
            p=parse_vdap_phase_line(f'{sta} P PU0 910603120001.00',network='XB')[0]
            self.assertEqual(p['seed_id'],f'XB.{sta[:3]}..{channel}')
        p=parse_vdap_phase_line('PI2 IPU0 910603120001.00',network='XB',dialect='fixed')[0]
        self.assertEqual(p['channel'],'EHZ')
        self.assertEqual(p['onset'],'I')

    def test_fixed_s_minute_rollover(self):
        line=list('PI2LIPU0 910603125958.00'.ljust(42))
        line[30:36]='  2.00';line[37]='S'
        picks=parse_vdap_phase_line(''.join(line),network='XB',dialect='fixed')
        self.assertEqual(len(picks),2)
        self.assertEqual(picks[1]['time'],UTCDateTime('1991-06-03T13:00:02'))
        self.assertEqual(picks[1]['channel'],'ELZ')

    def test_errors_and_multievent_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'event.PHA';path.write_text('garbage\nPI2L P 0 910603120001.00\n')
            with self.assertRaises(ValueError):read_vdap_pha_file(path,network='XB')
            picks,errors=read_vdap_pha_file(path,network='XB',strict=False,return_diagnostics=True)
            self.assertEqual(len(picks),1);self.assertEqual(errors[0]['source_line'],1)
            path.write_text('10\n')
            with self.assertRaises(ValueError):read_vdap_pha_file(path,strict=False)
            with self.assertRaises(FileNotFoundError):read_vdap_pha_file(Path(tmp)/'missing')

    def test_historical_hypo(self):
        lines=(FIXTURES/'hypo71_summary_excerpt.sum').read_text().splitlines()
        values=[parse_hypo71_line(line) for line in lines]
        self.assertTrue(all(values))
        self.assertEqual(values[0]['origin_time'],UTCDateTime('1991-05-07T03:21:14.29'))
        self.assertAlmostEqual(values[0]['latitude'],15+10.12/60)
        self.assertEqual(values[0]['n_ass'],7)
        cat,errors=parse_hypo71_file(FIXTURES/'hypo71_summary_excerpt.sum',verbose=False)
        self.assertEqual(len(cat),5);self.assertFalse(errors)

    def test_clock_rollover_and_optional_quality(self):
        line=list((FIXTURES/'hypo71_summary_excerpt.sum').read_text().splitlines()[0])
        line[7:9]='23';line[9:11]='60';line[12:17]='61.25'
        line[51:53]='  ';line[62:]=[]
        value=parse_hypo71_line(''.join(line))
        self.assertEqual(value['origin_time'],UTCDateTime('1991-05-08T00:01:01.25'))
        self.assertIsNone(value['n_ass']);self.assertIsNone(value['time_residual'])
        line[20]='S';line[30]='W'
        value=parse_hypo71_line(''.join(line));self.assertLess(value['latitude'],0);self.assertLess(value['longitude'],0)
        line[20]='Q';self.assertIsNone(parse_hypo71_line(''.join(line)))

if __name__=='__main__':unittest.main()
