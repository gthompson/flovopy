import struct
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read
from obspy.io.dmx.core import descript_trace_dtypes

from flovopy.core.vdap import convert_dmx_to_seisan, snap_sampling_rate

VDAP_RATE = float(np.float32(3125 / 31))  # header value, 100.80644989...


def record(station, rate):
    desc = np.zeros(1, dtype=descript_trace_dtypes)
    desc['network'] = b'unk'; desc['st_name'] = station.encode(); desc['component'] = b'Z'
    desc['begintime'] = float(UTCDateTime('1991-06-01')); desc['datatype'] = b's'
    desc['length'] = 4; desc['rate'] = rate
    data = np.array([2048, 2049, 2050, 2047], dtype='<u2')
    return struct.pack('<cchii', b'S', b'6', 7, 64, len(data.tobytes())) + desc.tobytes() + data.tobytes()


class SamplingRateTests(unittest.TestCase):
    def test_snap_recovers_rational_rate(self):
        self.assertEqual(snap_sampling_rate(VDAP_RATE), 3125 / 31)
        self.assertEqual(snap_sampling_rate(100.0), 100.0)
        self.assertEqual(snap_sampling_rate(VDAP_RATE, None), VDAP_RATE)
        # No small fraction nearby: keep the value rather than distort it.
        self.assertEqual(snap_sampling_rate(100.31415), 100.31415)

    def convert(self, **kwargs):
        t = tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        p = Path(t.name); raw = p / 'raw'; raw.mkdir()
        (raw / '91060100.DMX').write_bytes(record('AAAZ', VDAP_RATE) + record('BBBZ', VDAP_RATE))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = convert_dmx_to_seisan(raw, p / 'WAV', network='XB', database='TEST',
                                      continue_on_error=False, **kwargs)
        row = r['waveform_index'].iloc[0]
        return row, read(row.wav_file)

    def test_default_keeps_original_rate(self):
        row, st = self.convert()
        self.assertEqual(row.raw_sampling_rate, repr(VDAP_RATE))
        self.assertEqual(row.sampling_rate, repr(3125 / 31))
        self.assertEqual({tr.stats.sampling_rate for tr in st}, {3125 / 31})

    def test_explicit_override_still_supported(self):
        row, st = self.convert(sampling_rate=100.0)
        self.assertEqual(row.raw_sampling_rate, repr(VDAP_RATE))
        self.assertEqual({tr.stats.sampling_rate for tr in st}, {100.0})

    def test_time_correction_shifts_start_and_name(self):
        seen = []
        def correct(path, event_id, t, header):
            seen.append((path.name, event_id, t, header['localtime']))
            return -7 * 3600.0, 'H_UTC_PDT'
        row, st = self.convert(time_correction=correct)
        self.assertEqual(seen[0][:2], ('91060100.DMX', '91060100'))
        self.assertEqual(seen[0][3], 0.0)  # synthetic descriptor has localtime 0
        self.assertEqual(row.dmx_localtime, '0')
        self.assertEqual(st[0].stats.starttime, UTCDateTime('1991-05-31T17:00:00'))
        self.assertTrue(Path(row.wav_file).name.startswith('1991-05-31-1700-00M'))
        self.assertEqual((row.raw_starttime, row.clock_offset_s, row.clock_regime),
                         ('1991-06-01T00:00:00', -25200.0, 'H_UTC_PDT'))

    def test_unresolved_time_correction_fails_the_file(self):
        t = tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        p = Path(t.name); raw = p / 'raw'; raw.mkdir()
        (raw / '91060100.DMX').write_bytes(record('AAAZ', VDAP_RATE))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = convert_dmx_to_seisan(raw, p / 'WAV', network='XB', database='TEST',
                                      time_correction=lambda *a: (None, 'ambiguous'))
        self.assertEqual(r['failures'].stage.tolist(), ['clock'])
        self.assertFalse((p / 'WAV').exists() and any((p / 'WAV').rglob('*M.*')))


if __name__ == '__main__':
    unittest.main()
