from pathlib import Path
from unittest.mock import patch
import numpy as np
from obspy import Stream, Trace, UTCDateTime
from flovopy.seisanio.conversion import archive_to_sds


def test_conversion_trims_and_batched_writes(tmp_path):
    t0 = UTCDateTime("2001-01-02T00:00:00")
    tr = Trace(data=np.arange(20, dtype=np.int32))
    tr.stats.starttime = t0 - 5
    tr.stats.sampling_rate = 1

    class Archive:
        def iter_waveform_files(self, *args, **kwargs):
            yield Path("dummy")
        def read_waveform_file(self, *args, **kwargs):
            return Stream([tr])

    streams = []
    def write(stream, **kwargs):
        streams.append(stream.copy())
        return [tmp_path / "2001" / "XX" / "a.002"]

    with patch("flovopy.seisanio.conversion.EnhancedSDSClient") as cls:
        cls.return_value.write_stream.side_effect = write
        result = archive_to_sds(Archive(), t0, t0 + 10, tmp_path,
                                return_counts=True, batch_files=1)
    assert result["files_failed"] == 0
    assert len(streams) == 1
    assert streams[0][0].stats.starttime == t0
    assert streams[0][0].stats.npts == 10


def test_overwrite_rejected(tmp_path):
    class Archive: pass
    with patch("flovopy.seisanio.conversion.EnhancedSDSClient"):
        try:
            archive_to_sds(Archive(), "2001-01-01", "2001-01-02",
                           tmp_path, write_mode="overwrite")
        except ValueError as exc:
            assert "overwrite" in str(exc)
        else:
            raise AssertionError("overwrite should be rejected")
