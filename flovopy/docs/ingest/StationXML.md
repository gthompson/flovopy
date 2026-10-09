# StationXML metadata — experiment-wide workflow

[Index](README.md) · [Quick start](Quick-Start.md) · [CLI](Configuration-and-CLI.md) · [StationXML](StationXML.md) · [QC](Validation-and-QC.md) · [SDS merging](SDS-Merging.md)

## Scope and ownership

StationXML records **what the waveform identifiers mean**: network/station/location/channel, coordinates/elevation/depth, sample rates, sensor and digitizer models/serials, component orientation, sensitivity and full response, and valid time epochs. It does **not** convert raw data or replace SDS QC. For EPIC passive experiments, **use Nexus** to assemble and verify the complete experiment-wide StationXML. FLOVOpy can assist with inventories and verification, but the current `flovopy.ingest` snapshot **does not contain a Nexus driver or a generic StationXML builder CLI**.

## Recommended workflow

1. **Inventory waveform identifiers and time ranges.** Run `epic-validate` on staging/master SDS and inspect its CSV. Also check channel sample rates and gaps. [QC guide](Validation-and-QC.md).
2. **Compile a deployment table**, maintained outside FLOVOpy's instrument code, with `network, station, location, channel, start_utc, end_utc, latitude, longitude, elevation_m, depth_m, azimuth_deg, dip_deg, sensor_model, sensor_serial, digitizer_model, digitizer_serial, response_source, notes`. This is a suggested *human-maintained schema*, not a CSV automatically consumed by Nexus.
3. **Collect authoritative response sources.** For Centaur/Pegasus/Q330/RT130, use EPIC/Nexus instrument definitions and calibration information; for Gem/infraBSU/Gecko/SmartSolo, use verified sensor and digitizer sensitivities, gain settings, firmware and appropriate response files. Do not guess calibration constants from the model name alone.
4. **Resolve epochs.** Split metadata at deployment/recovery, sensor swaps, digitizer swaps, channel/recording gain changes, rate changes, location moves and orientation changes. A single station can have multiple channels and overlapping sensor groups; map each channel to the correct sensor.
5. **Build and validate in Nexus** following EPIC's Nexus instructions. Import prior station-specific StationXML only after checking IDs, epoch coverage and response chains. Pegasus Harvester XML should be treated as supporting input, not final EPIC metadata.
6. **Export a single experiment-wide StationXML** (or the submission structure requested by EPIC), then check it with ObsPy against representative raw waveforms and the complete set of NSLC/epoch combinations.
7. **Review with EPIC**, then deliver metadata and daily MiniSEED via the agreed `data2passcal` procedure. Keep versioned metadata snapshots and a change log.

## ObsPy checks (do not remove response during ingestion)

```python
from obspy import read, read_inventory
st = read('/data/sample.mseed')
inv = read_inventory('/data/metadata/experiment.xml')
for tr in st:
    try:
        resp = inv.get_response(tr.id, tr.stats.starttime)
        print(tr.id, tr.stats.starttime, 'response OK', resp is not None)
    except Exception as exc:
        print('MISSING RESPONSE', tr.id, tr.stats.starttime, exc)
```

Check **both** beginning and end of deployment and any known epoch transitions; checking only the first sample does not establish coverage for the whole recording. Response removal is for analysis copies (`trace.remove_response(...)`), **not** for archival raw MiniSEED. Check that `StationXML` channel codes exactly match MiniSEED headers: a response for `DPZ` cannot be silently applied to `DHZ`.

## Orientation caution

In StationXML, vertical **positive up** is conventionally `dip=-90°`, and **positive down** is `dip=+90°`; verify polarity against installation records and calibration, not just the filename's `Z`. Horizontal `N/E` names also require verified azimuth and polarity. Do not use `elevation=0` as though it were a measured value.

## Instrument-specific notes

- [Centaur](Centaur.md), [Pegasus](Pegasus.md), [Q330](Q330.md), [RT130](RT130.md): Nexus and EPIC response/epoch checks.
- [Gem](Gem.md): serial mapping and GPS logs support station history; response calibration is a separate task.
- [SmartSolo](SmartSolo.md): vendor template response is useful only if channel codes, sampling, gain and sensor type match; preserve serials and component orientations.
- [SiliconAudio](SiliconAudio.md): digitizer, sensor, gain and clock history must be assembled from field records.

**Status:** This page documents a *manual + Nexus + ObsPy verification* workflow. It does not claim that FLOVOpy currently creates valid StationXML automatically.
