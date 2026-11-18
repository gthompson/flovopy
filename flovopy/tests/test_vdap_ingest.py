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
        # An S in the next minute is written as seconds >= 60 (HYPO71).
        line=list('PI2LIPU0 910603125958.00'.ljust(42))
        line[30:36]=' 62.00';line[37]='S'
        picks=parse_vdap_phase_line(''.join(line),network='XB',dialect='fixed')
        self.assertEqual(len(picks),2)
        self.assertEqual(picks[1]['time'],UTCDateTime('1991-06-03T13:00:02'))
        self.assertEqual(picks[1]['channel'],'ELZ')
        # S seconds below P seconds are not moved a minute later: the S is skipped, the P kept.
        for card in ['GRNZ P 4 910603171748.23       48.22 S 3',      # 91060310: S 0.01 s before P
                     'BUGZEP 1 91 8 9231436.76       31.13ES?2']:     # S 5.6 s before P
            issues=[]
            picks=parse_vdap_phase_line(card,network='XB',issues=issues)
            self.assertEqual([p['phase'] for p in picks],['P'])
            self.assertTrue(any('earlier than P' in i for i in issues))

    def test_unified_card_rules(self):
        # Real cards from the 1991 Pinatubo monthly files.
        def parse(line,issues=None):
            return [(p['phase'],p['time'],p['onset'],p['first_motion'],p['weight'])
                    for p in parse_vdap_phase_line(line,network='XB',issues=issues)]
        T=UTCDateTime
        # S quality fields are kept.
        self.assertEqual(parse('CABN P 5 91 731163611.90       15.56ESU1')[1],('S',T('1991-07-31T16:36:15.56'),'E','U',1))
        # Minute 60 rolls into the next hour.
        self.assertEqual(parse('FNGZEP+1 91 815 560 2.14        5.69ISU1'),
                         [('P',T('1991-08-15T06:00:02.14'),'E',None,1),('S',T('1991-08-15T06:00:05.69'),'I','U',1)])
        # P seconds >= 60: S counts from the written minute, not the rolled P minute.
        self.assertEqual(parse('CABN P 4 91 722124060.00       62.56 S 1')[1][:2],('S',T('1991-07-22T12:41:02.56')))
        # A bare "P" code is a fixed card, not a tokenized record.
        self.assertEqual(parse('QADZ P   91 818123059.91'),[('P',T('1991-08-18T12:30:59.91'),None,None,None)])
        # Placeholders are skipped.
        self.assertEqual(parse_vdap_phase_line('XXXXIPU0 91 8 3 02752.47',network='XB'),[])
        # A damaged S field keeps the valid P and reports the S problem.
        issues=[]
        self.assertEqual([p[0] for p in parse('CABN P 4 91 726134854.31       5\x00\x008.12 S 1',issues)],['P'])
        self.assertTrue(issues)
        # Damaged timestamps are rejected rather than guessed.
        for bad in ['PPON P 410530154826.01       26.66 S 0','CABZIPD1 91 818 9 5.57        7.75ISD3',
                    'CRWZIPD0 91 8 31457525','CABPD5 91 815112640.10       42.96ISU1']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_vdap_phase_line(bad,network='XB')

    def test_multievent_blocks_and_filters(self):
        from flovopy.core.vdap import read_vdap_pha_blocks,dedupe_picks,filter_picks_near_median
        text='\n'.join(['BUGZIPU0 91 731162835.39       37.49 S 1','BUGZIPU0 91 731162835.39','garbage','10',
                        '10','CABZ PU0 91 7311700 1.00','PI2ZIPU0 91 7311700 2.00','QADZIPU0 91 7311700 3.00',
                        'CRWZIPU0 91 7311710 3.00','100',''])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pinjul91.pha';path.write_text(text)
            blocks,diag=read_vdap_pha_blocks(path,network='XB')
        self.assertEqual([b['block_index'] for b in blocks],[0,1,2,3])
        self.assertEqual([len(b['picks']) for b in blocks],[3,0,4,0])
        self.assertEqual((diag[0]['source_line'],diag[0]['severity']),(3,'error'))
        self.assertEqual(blocks[0]['picks'][0]['raw_lineno'],1)
        self.assertEqual(len(dedupe_picks(blocks[0]['picks'])),2)
        kept,rejected=filter_picks_near_median(blocks[2]['picks'],30)
        self.assertEqual(([p['station'] for p in kept],[p['station'] for p in rejected]),(['CAB','PI2','QAD'],['CRW']))

    def test_split_block_with_two_events(self):
        from flovopy.core.vdap import split_picks_by_time_gap
        T=UTCDateTime('1991-07-10T12:00:00')
        def pk(sta,dt):return dict(station=sta,time=T+dt)
        first=[pk('CAB',0),pk('BUG',1),pk('FNG',2)];second=[pk('CAB',50),pk('BUG',51),pk('QAD',52)]
        groups=split_picks_by_time_gap(first+second+[pk('PI2',400)])
        self.assertEqual([len(g) for g in groups],[3,4])          # stray PI2 joins the nearer event
        self.assertEqual(split_picks_by_time_gap(first+[pk('CAB',50),pk('BUG',51)]),[first+[pk('CAB',50),pk('BUG',51)]])
        self.assertEqual(split_picks_by_time_gap([]),[])

    def test_fixed_s_only_and_shifted_timestamps(self):
        # Lines from the 1991 Pinatubo monthly PHA files. Blank P seconds must
        # not shift the minute, and a P code without P seconds gives no P pick.
        cases=[('CABN     91 8 2 2 6            37.91 S 3',[('S','1991-08-02T02:06:37.91')]),
               ('CABZ     91 7311942            16.45 S  ',[('S','1991-07-31T19:42:16.45')]),
               ('PI2L     91 8 4 2 3            63.19ES?1',[('S','1991-08-04T02:04:03.19')]),
               ('CABN P 4 91 8 823 4            70.13 S 2',[('S','1991-08-08T23:05:10.13')]),
               ('BUGZIPU0  91 8 6 65625.02',[('P','1991-08-06T06:56:25.02')]),
               ('BUGZ PU0 910603073938.49       39.99 S 2',[('P','1991-06-03T07:39:38.49'),('S','1991-06-03T07:39:39.99')])]
        for line,expected in cases:
            with self.subTest(line=line):
                picks=parse_vdap_phase_line(line,network='XB',dialect='fixed')
                self.assertEqual([(p['phase'],p['time']) for p in picks],[(ph,UTCDateTime(t)) for ph,t in expected])

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

    def test_opt_in_terminal_separator_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'event.PHA'
            pick='PI2L P PU0 910603125636.37\n'
            path.write_text(pick+'                 10\n\n\x1a\n')
            with self.assertRaises(ValueError):read_vdap_pha_file(path)
            self.assertEqual(len(read_vdap_pha_file(path,allow_terminal_separator=True)),1)
            for text in [pick+'10\n'+pick,pick+'10\n100\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):read_vdap_pha_file(path,allow_terminal_separator=True)

if __name__=='__main__':unittest.main()
