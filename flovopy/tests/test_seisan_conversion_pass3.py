from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from flovopy.seisanio.conversion import archive_to_sds


def test_transform_audit_and_no_source_mutation(tmp_path):
    t0 = UTCDateTime('2001-01-02')
    trace = Trace(data=np.arange(20, dtype=np.int32), header={
        'network': 'XX', 'station': 'TEST', 'channel': 'BHZ',
        'sampling_rate': 1.0, 'starttime': t0 - 5})
    original = Stream([trace])
    class Archive:
        def iter_waveform_files(self, *a, **kw):
            yield Path('dummy')
        def read_waveform_file(self, *a, **kw):
            return original
    audit, written = [], []
    def transform(st):
        st[0].stats.network = 'MV'
        return st
    def writer(st, **kw):
        written.append(st.copy())
        return [tmp_path / 'file']
    with patch('flovopy.seisanio.conversion.EnhancedSDSClient') as client:
        client.return_value.write_stream.side_effect = writer
        result = archive_to_sds(Archive(), t0, t0 + 10, tmp_path,
            waveform_transform=transform,
            transform_audit=lambda path, changes: audit.append((path, changes)),
            return_counts=True, batch_files=1)
    assert result['files_failed'] == 0
    assert original[0].stats.network == 'XX'
    assert written[0][0].stats.network == 'MV'
    assert written[0][0].stats.starttime == t0
    assert audit[0][1][0]['original_id'].startswith('XX.')
    assert audit[0][1][0]['corrected_id'].startswith('MV.')


def test_transform_error_marks_file_failed(tmp_path):
    t0 = UTCDateTime('2001-01-02')
    class Archive:
        def iter_waveform_files(self, *a, **kw):
            yield Path('bad')
        def read_waveform_file(self, *a, **kw):
            return Stream([Trace(data=np.ones(5), header={'starttime': t0})])
    def fail(st):
        raise ValueError('invalid mapping')
    with patch('flovopy.seisanio.conversion.EnhancedSDSClient') as client:
        result = archive_to_sds(Archive(), t0, t0 + 3, tmp_path,
                                waveform_transform=fail, return_counts=True)
    assert result['files_failed'] == 1
    assert result['failed_files'] == ['bad']
    client.return_value.write_stream.assert_not_called()
