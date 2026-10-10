import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from flovopy.seisanio.conversion import archive_to_sds
from flovopy.seisanio.validation import validate_stream_inventory


def sample():
    tr = Trace(data=np.arange(4, dtype=np.int32),
               header={"network": "MV", "station": "MBLG", "channel": "BHZ",
                       "starttime": UTCDateTime("2001-01-01"), "sampling_rate": 1})
    return Stream([tr])


class Archive:
    def iter_waveform_files(self, *args, **kwargs):
        yield Path("source.WAV")

    def read_waveform_file(self, *args, **kwargs):
        return sample()


def test_stationxml_epoch_and_response_are_advisory():
    class Inventory:
        def get_coordinates(self, trace_id, when):
            if when > UTCDateTime("2001-01-01T00:00:01"):
                raise ValueError("epoch ended")
            return {"latitude": 16.7}

        def get_response(self, trace_id, when):
            raise ValueError("response missing")

    records = validate_stream_inventory(sample(), Inventory())
    assert records[0]["status"] == "missing_channel_or_epoch"
    assert any("response" in item for item in records[0]["details"])


def test_manifest_written_and_correction_audited(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    def rename(stream):
        stream[0].stats.station = "MBWH"
        return stream
    with patch("flovopy.seisanio.conversion.EnhancedSDSClient") as cls:
        cls.return_value.write_stream.return_value = [tmp_path / "output.mseed"]
        result = archive_to_sds(Archive(), "2001-01-01", "2001-01-02", tmp_path,
                                waveform_transform=rename, manifest_path=manifest,
                                return_counts=True, batch_files=1)
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    assert result["files_failed"] == 0
    assert rows[0]["status"] == "written"
    assert rows[0]["corrections"][0]["corrected_id"] == "MV.MBWH..BHZ"
    assert rows[0]["sds_outputs"] == [str(tmp_path / "output.mseed")]


def test_failed_write_recorded(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    with patch("flovopy.seisanio.conversion.EnhancedSDSClient") as cls:
        cls.return_value.write_stream.side_effect = OSError("disk full")
        result = archive_to_sds(Archive(), "2001-01-01", "2001-01-02", tmp_path,
                                manifest_path=manifest, return_counts=True, batch_files=1)
    assert result["files_failed"] == 1
    assert json.loads(manifest.read_text())["status"] == "write_failed"


def test_readback_verification(tmp_path):
    output = tmp_path / "actual.mseed"
    def write(stream, **kwargs):
        stream.write(str(output), format="MSEED")
        return [output]
    with patch("flovopy.seisanio.conversion.EnhancedSDSClient") as cls:
        cls.return_value.write_stream.side_effect = write
        result = archive_to_sds(Archive(), "2001-01-01", "2001-01-02", tmp_path,
                                verify_writes=True, return_counts=True, batch_files=1)
    assert result["files_failed"] == 0
