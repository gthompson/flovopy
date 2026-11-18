import unittest

import pandas as pd
from obspy import UTCDateTime

from flovopy.core.clock import ClockRegimes, dos_mktime_wall_clock

TABLE = pd.DataFrame([
    dict(stream='wav;pick', regime='PHT', start='', end='1991-05-26T07:40', seq_min='', seq_max='', utc_offset_h='-15'),
    dict(stream='wav', regime='PHT_seq', start='1991-05-26T07:40', end='1991-05-26T16:45', seq_min='', seq_max='91052617', utc_offset_h='-15'),
    dict(stream='wav', regime='UTC_seq', start='1991-05-26T07:40', end='1991-05-26T16:45', seq_min='9105261G', seq_max='', utc_offset_h='-7'),
    dict(stream='wav;pick', regime='PDT', start='1991-05-26T16:45', end='1991-10-27T09:00', seq_min='', seq_max='', utc_offset_h='-7'),
    dict(stream='wav;pick', regime='PST', start='1991-10-27T09:00', end='', seq_min='', seq_max='', utc_offset_h='-8'),
])


class ClockTests(unittest.TestCase):
    def test_dos_mktime_wall_clock(self):
        # PDT: epoch is 7 h ahead of the wall clock; PST: 8 h.
        self.assertEqual(dos_mktime_wall_clock('1991-06-03T12:00'), UTCDateTime('1991-06-03T05:00'))
        self.assertEqual(dos_mktime_wall_clock('1991-11-03T12:00'), UTCDateTime('1991-11-03T04:00'))
        # Both readings of the repeated 01:xx hour on 27 Oct come back as 01:xx.
        self.assertEqual(dos_mktime_wall_clock('1991-10-27T08:30'), UTCDateTime('1991-10-27T01:30'))
        self.assertEqual(dos_mktime_wall_clock('1991-10-27T09:30'), UTCDateTime('1991-10-27T01:30'))

    def test_lookup_uses_sequence_where_times_repeat(self):
        c = ClockRegimes(TABLE)
        self.assertEqual(c.to_utc('wav', '1991-05-20T10:00')[0], UTCDateTime('1991-05-19T19:00'))
        self.assertEqual(c.to_utc('wav', '1991-05-26T15:41', '91052617')[0], UTCDateTime('1991-05-26T00:41'))
        self.assertEqual(c.to_utc('wav', '1991-05-26T08:40', '9105261G')[0], UTCDateTime('1991-05-26T01:40'))
        t, info = c.to_utc('wav', '1991-05-26T09:00')
        self.assertIsNone(t); self.assertEqual(info['status'], 'ambiguous')
        t, info = c.to_utc('pick', '1991-05-26T09:00')
        self.assertIsNone(t); self.assertEqual(info['status'], 'unresolved')
        self.assertEqual(c.to_utc('wav', '1991-11-01')[0], UTCDateTime('1991-10-31T16:00'))
        with self.assertRaises(KeyError):
            c.lookup('bob', '1991-06-01')

    def test_regime_offsets_agree_with_mktime(self):
        # After the reset the wall clock was UTC, so H rows must equal the mktime undo.
        c = ClockRegimes(TABLE)
        for t in ['1991-06-01T00:00', '1991-10-27T08:59', '1991-10-27T09:01', '1991-12-31T23:00']:
            self.assertEqual(c.to_utc('wav', t)[0], dos_mktime_wall_clock(t))

    def test_apply_keeps_raw_and_flags(self):
        c = ClockRegimes(TABLE)
        df = pd.DataFrame(dict(t=['1991-06-01T00:00:00', '1991-05-26T09:00:00', ''], seq=['a', 'b', 'c']))
        out = c.apply(df, 'wav', 't')
        self.assertEqual(out.t[0], '1991-05-31T17:00:00')
        self.assertEqual(out.t.isna().tolist(), [False, True, True])
        self.assertEqual(out.raw_t.tolist(), df.t.tolist())
        self.assertEqual(out.clock_status.tolist(), ['resolved', 'ambiguous', 'no_time'])
        with self.assertRaises(ValueError):
            c.apply(out, 'wav', 't')

    def test_marker_and_uncertainty(self):
        table = pd.DataFrame([
            dict(stream='w', regime='old', start='', end='1991-05-26T16:45', marker='0', utc_offset_h='-15', uncertainty_s='5'),
            dict(stream='w', regime='new', start='1991-05-26T07:40', end='', marker='-25200', utc_offset_h='-7', uncertainty_s='5'),
        ])
        c = ClockRegimes(table)
        self.assertEqual(c.to_utc('w', '1991-05-26T15:41', marker=0)[0], UTCDateTime('1991-05-26T00:41'))
        self.assertEqual(c.to_utc('w', '1991-05-26T08:40', marker='-25200')[0], UTCDateTime('1991-05-26T01:40'))
        self.assertEqual(c.lookup('w', '1991-05-26T09:00')['status'], 'ambiguous')
        self.assertEqual(c.lookup('w', '1991-06-26T09:00', marker=0)['status'], 'unresolved')
        out = c.apply(pd.DataFrame(dict(t=['1991-06-01T00:00:00'], m=[-25200])), 'w', 't', marker_column='m')
        self.assertEqual((out.t[0], out.clock_uncertainty_s[0]), ('1991-05-31T17:00:00', 5.0))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ClockRegimes(TABLE.assign(end='1990-01-01').iloc[[1]])
        with self.assertRaises(ValueError):
            ClockRegimes(TABLE.drop(columns='utc_offset_h'))


if __name__ == '__main__':
    unittest.main()
