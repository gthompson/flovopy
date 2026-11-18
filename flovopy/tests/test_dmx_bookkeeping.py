"""convert_dmx_to_seisan accounts for every DMX file and every dropped trace."""
import struct, tempfile, unittest, warnings
from pathlib import Path
import numpy as np
from obspy import UTCDateTime
from obspy.io.dmx.core import descript_trace_dtypes
from flovopy.core.vdap import convert_dmx_to_seisan


def record(station, data):
    data = np.asarray(data, dtype='<u2')
    desc = np.zeros(1, dtype=descript_trace_dtypes)
    desc['network'] = b'unk'; desc['st_name'] = station.encode(); desc['component'] = b'Z'
    desc['begintime'] = float(UTCDateTime('1991-06-01')); desc['datatype'] = b's'
    desc['length'] = len(data); desc['rate'] = 100
    return struct.pack('<cchii', b'S', b'6', 7, 64, len(data.tobytes())) + desc.tobytes() + data.tobytes()


class BookkeepingTests(unittest.TestCase):
    def test_every_file_and_dropped_trace_is_logged(self):
        live = [2048, 2049, 2050, 2047]
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); raw = p / 'raw'; raw.mkdir()
            (raw / '91060100.DMX').write_bytes(record('AAAZ', live) + record('BBBZ', [2048] * 4) + record('IRIG', live))
            (raw / '91060101.DMX').write_bytes(record('IRIG', live) + record('CCCZ', [2048] * 4))  # nothing left
            (raw / '91060102.DMX').write_bytes(record('DDDZ', []))                               # header only
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = convert_dmx_to_seisan(raw, p / 'WAV', network='XB', database='TEST',
                                          out_file_log=p / 'log.csv')
            log = r['file_log'].set_index('event_id')
            self.assertEqual(len(log), 3)
            self.assertEqual(log.loc['91060100', 'status'], 'written')
            self.assertEqual(log.loc['91060100', ['traces_read', 'traces_saved', 'traces_flat', 'traces_irig']].tolist(), [3, 1, 1, 1])
            self.assertEqual(log.loc['91060101', 'status'], 'empty')
            self.assertEqual(log.loc['91060102', 'status'], 'empty')
            self.assertEqual(log.loc['91060102', 'traces_no_samples'], 1)
            self.assertTrue((p / 'log.csv').is_file())
            prov = r['trace_id_provenance']
            first = prov[prov.event_id == '91060100'].set_index('trace_status')
            self.assertEqual(sorted(first.index), ['flat', 'irig', 'kept'])
            self.assertEqual(first.loc['flat', 'fixed_id'], '')
            self.assertEqual(first.loc['kept', 'npts'], 4)
            self.assertEqual(set(prov[prov.event_id == '91060101'].trace_status), {'flat', 'irig'})
            index = r['waveform_index'].set_index('event_id')
            self.assertEqual(index.loc['91060100', 'dropped_traces'], 2)
            self.assertIn(':flat', index.loc['91060100', 'dropped_trace_ids'])
            self.assertEqual(r['summary']['empty'], 2)


if __name__ == '__main__':
    unittest.main()
