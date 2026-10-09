# SmartSolo ingestion

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)


[Ingestion overview](README.md) · [SDS merging](SDS-Merging.md) · [Gem](Gem.md)

## Workflow

1. Use the dedicated **Harvester / SoloLite** software to copy SmartSolo DLD data and convert to MiniSEED. This FLOVOpy module does **not** decode DLD files.
2. Supply a deployment-specific `serial,seed_prefix` CSV. The prefix includes network, station, location, and the first two channel characters; e.g. `453013783,1R.B14..DH`. The component (`Z`, `N`, `E`) is taken from the trace channel or filename.
3. Stage converted MiniSEED into an independent SDS directory.
4. Merge staged SDS into the master using FLOVOpy's SDS merger.

Vendor output typically resembles `453005764.0001.2026.03.10.20.24.18.000.E.miniseed` (one channel per day). The module reads actual trace timestamps, not just the filename's date. It resolves either full serial or the shortened `4530`-stripped station code seen in historical files.

## Example

```python
from flovopy.ingest.smartsolo import stage_smartsolo_mseed
from flovopy.sds.merge_sds_archives import merge_sds_archives

report = stage_smartsolo_mseed(
    input_root='/data/SOLODATA/ten_stations',
    sds_root='/data/service/20_archive/SmartSolo/SDS',
    mapping_csv='config/smartsolo_mapping.csv',
    dry_run=True,
)
print(report)

# After reviewing dry-run results:
report = stage_smartsolo_mseed(
    '/data/SOLODATA/ten_stations',
    '/data/service/20_archive/SmartSolo/SDS',
    mapping_csv='config/smartsolo_mapping.csv',
)

merge_sds_archives(
    '/data/service/20_archive/SmartSolo/SDS',
    '/data/master/SDS',
    mode='fast',
    dry_run=True,
)
# Rerun with dry_run=False after inspection.
```

The supplied example mapping preserves all eleven IDs from the original Artemis II notebook; copy it into the project repository and update it for each deployment. No KSC mappings are embedded in the library.

**Batching:** files are sorted for deterministic discovery, then grouped by SEED ID and UTC day. Overlapping/adjacent segments are merged in memory (`Stream.merge(method=0)`) and written once per day/channel, not once per one-minute file. The source files are never modified. The staging archive may be updated using `write_mode='merge'`.

**Error handling:** unknown or conflicting serial/component assignments are rejected by default (`strict=True`). A `dry_run` reads waveforms but writes nothing. For large datasets, the current implementation holds grouped daily streams in memory until writing; consider streaming groups if memory is an issue.

## StationXML and response correction

StationXML creation is a **separate metadata task**. The original notebook copied a template SmartSolo response to multiple stations, with coordinates from a project CSV. This can work only after verifying the template's sensor model, sample rate, channel code, gain, units, and deployment epoch against the actual instruments. Do not blindly copy a `DPZ` response onto `DHZ` channels; ObsPy response removal requires matching SEED IDs and valid epochs.

For vertical geophones, StationXML `dip=-90` represents positive-up vertical; `dip=+90` represents positive-down vertical. Verify the actual sensor orientation rather than assuming either. E/N component azimuth/dip and polarity also need checking before creating three-component metadata. Elevation `0` is only a placeholder, not a measured station elevation.

Use `Inventory.get_response(trace.id, trace.stats.starttime)` to check coverage before calling `trace.remove_response(inventory=inv, output='VEL', pre_filt=...)`. Preserve raw MiniSEED and staging SDS; response removal belongs in the analysis pipeline, not in raw ingestion.

## Metadata verification

For experiment-wide delivery, reconcile SmartSolo serial mappings and component responses with [Nexus StationXML](StationXML.md). A template response copied from another channel is not sufficient without checking channel code, sample rate, gain, orientation and response epoch.
