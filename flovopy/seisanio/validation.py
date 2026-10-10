"""Advisory StationXML checks for SEISAN migration; no inferred corrections."""
from obspy import UTCDateTime


def validate_stream_inventory(stream, inventory):
    """Return one validation record per trace, without modifying waveforms.

    Status is ``valid``, ``missing_channel_or_epoch``, or ``invalid_response``.
    Checks are evaluated at trace start and end (channel epoch must span both).
    """
    records = []
    for tr in stream:
        record = {"trace_id": tr.id, "starttime": str(tr.stats.starttime),
                  "endtime": str(tr.stats.endtime), "status": "valid", "details": []}
        for label, when in (("start", tr.stats.starttime), ("end", tr.stats.endtime)):
            try:
                inventory.get_coordinates(tr.id, UTCDateTime(when))
            except Exception as exc:
                record["status"] = "missing_channel_or_epoch"
                record["details"].append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            try:
                inventory.get_response(tr.id, UTCDateTime(when))
            except Exception as exc:
                if record["status"] == "valid":
                    record["status"] = "invalid_response"
                record["details"].append(f"{label} response: {type(exc).__name__}: {exc}")
        records.append(record)
    return records
