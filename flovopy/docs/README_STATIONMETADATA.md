# FLOVOpy deployment metadata: phase 1

This is an **additive** update to the April 2026 `stationmetadata` snapshot. Existing modules are preserved; `stationmetadata/deployments.py` is new. The companion expanded workbook retains the original `ksc_stations_master` and `infrastructure` worksheets, and introduces a normalized schema.

## Tables

- `Stations`: one station site (network/station), geographical reference, site description.
- `Deployments`: one digitizer deployment/configuration epoch, `deployment_id`, network, station, recorder, serial, start/end UTC.
- `Sensors`: **physical sensor positions**. `sensor_id` can represent an entire co-located three-component seismic sensor, or an individual array microphone. `css_sta` is a unique CSS physical location code; `refsta`, `dnorth_km`, `deast_km` represent array offsets from the reference. Co-located components share a sensor_id/css_sta and position; separate infrasound array elements have separate sensor_id/css_sta and offsets. `location_group` may identify shared enclosure (not necessarily a CSS station).
- `Channels`: one SEED channel and response epoch, linked to a deployment and sensor; NSLC, sample rate, azimuth, dip, gain configuration, response_id, UTC start/end. Z/N/E are separate channel rows sharing the same sensor_id. Channel dip uses StationXML convention: Z **-90** for upward-positive, horizontal **0**; azimuth N **0**, E **90**. Check actual wiring/polarity.
- `Responses`: metadata and provenance for verified response templates. This release does not invent responses from nominal sensitivity values.
- `ServiceRuns`: explicitly named service-run UTC time windows. Each service-run StationXML contains channels with epochs overlapping that window, retaining their actual epoch boundaries. It is not an authoritative snapshot of all past/future deployments.

## Using the new module

`get_tables(path)` accepts the workbook or a directory of named CSVs. `validate(tables)` detects missing links, invalid intervals, and overlapping NSLC epochs. `stationxml_for_run(tables, run_id, path, response_templates=..., require_responses=True)` produces StationXML using ObsPy. The `response_templates` dictionary maps response_id to verified ObsPy Response or single-channel Inventory objects; missing responses are errors when `require_responses=True`. Use `False` only to create a preliminary response-free XML. `export_css` writes import-oriented CSV tables named `site.csv`, `sitechan.csv`, `snetsta.csv`. These are **not fixed-width CSS flatfiles**; use a subsequent database-specific formatting/import step. The `lddate` fields are placeholders pending database ingestion.

## CSS mappings and caveats

- CSS `site.sta` is the **physical location** (`Sensors.css_sta`), not necessarily the SEED station code. Array elements need distinct CSS station identifiers.
- `site.elev` and `sitechan.edepth` are **kilometers**; workbook elevations/depths are meters.
- `site.refsta` and `dnorth`/`deast` are optional for standalone sites; array members need explicit offsets in kilometers. Coordinates are supplied independently and are not inferred from offsets.
- `sitechan.hang` uses channel azimuth; `sitechan.vang` uses 90 + StationXML dip (CSS angle from upward vertical). Confirm local Antelope convention before importing.
- `snetsta` maps SEED network + physical station code to the local CSS station code. If CSS station codes need an independent alias, extend the schema before import.
- `sitechan.chanid` is a temporary per-export identifier; real Antelope installations need globally managed chanid allocation.
- UTC date-only end dates in legacy records are ambiguous; no automatic inclusive/exclusive conversion is attempted.

## Remaining work

1. Migrate and reconcile historical legacy rows; the legacy `DHZNE` aggregate channel must be split into explicit Z/N/E records. Historical sensor swaps within a row require separate deployment/response epochs.
2. Connect existing NRL, infraBSU, and local response builders to the response_id registry; check overall sensitivity and stage units.
3. Build safe merge logic for verified component/digitizer StationXML with conflict detection; do not silently concatenate duplicate NSLC epochs.
4. Integrate the EarthScope StationXML validator and SDS coverage audit.
5. Export true fixed-width CSS3.0 flatfiles with correct database-assigned IDs and lddate when an Antelope target is specified.
