# Historical BOB readers: `flovopy.legacy.bob`

This package reads surviving BOB monitoring formats without applying a dataset's station interpretations or calibration. It has no organizational namespace and no dependency on the Pinatubo pipeline.

## Public API

| Reader | Supported layout |
|---|---|
| `read_annual` | Full-year float32 arrays with a validated header-length word (576-byte header / 600-second spacing by default) |
| `read_rsam`, `read_rsam_events`, `read_tilt` | Semantic wrappers around that annual layout; callers provide channel interpretations |
| `read_ssam` | Dated text records, 16 positional band values by default; the same reader handles SSAM layouts under DATA directories |
| `read_mudflow` | Dated text records with three positional values by default |
| `read_rain` | Dated text records with eight positional values by default |
| `read_counts` | Dated scalar records, retaining blank values and zero counts |
| `read_cospec` | Dated scalar records, including concatenated minute/value fields and repeated observations |
| `read_radio_maps` | Dated, fixed-column SEIS_MAP-style snapshots; raw frequency strings and ditto resolution retained |
| `read_plot_definitions` | BOB_QPLO series/file/Fortran descriptors, read as metadata without executing formats |

Data readers accept a file, recursive directory or ZIP. They read ZIP members without extraction; input files are never edited. Use a product-specific input, not an entire mixed BOB tree. Nested archives are not recursively unpacked. Layouts outside those listed are not claimed to be supported automatically.

## Result and preservation policy

Readers return `BOBDataset(data, metadata, errors)` (plot-definition reader returns a DataFrame). Tables preserve source filename and text line number (or annual sample index), recorded calendar labels and original numeric values. `value_01_raw` retains a reported value; `value_01_missing` flags sentinels/nonfinite values; `value_01` supplies the corresponding usable value or NaN. Text `raw_line` and file hashes retain provenance.

**Zero is a valid monitoring value.** Missing-value policies are format-specific and configurable. Text wrappers default to -998/-999, except RAIN (-998); annual series default to -998. Negative tilt values are retained. Unknown quantities and units remain positional/uninterpreted. No interpolation, aggregation, counter differencing, polarity change, timezone correction or calibration is performed.

Text parse errors are returned explicitly; `strict=True` raises instead. Numeric records with invalid dates or wrong field counts remain auditable; structurally malformed records are in `errors`. Non-strict reads are not proof of successful decoding. Annual files must have a matching layout/year/size; if no valid annual file remains, the reader raises even in non-strict mode. Two-digit years default to the 1900 century; set `century` explicitly for other periods.

`series` defaults to the relative parent path for daily tables, the individual filename for scalar alternative count/COSPEC files, and parent/channel-prefix for annual arrays. It is a source grouping, not an inferred station identity. Supply `series_from_path` to text readers or explicit `channel_metadata` to annual readers when station/observable mapping is documented. Differences in ZIP root paths remain visible; do not assume names match across folders/archives without an explicit mapping.

Repeated timestamps are preserved and flagged within series. `coverage()` counts rows with at least one nonsentinel finite value and reports unique nonmissing timestamps; it does not establish continuous uptime, sensor validity or signal quality. Radio maps are snapshots, not deployment epochs, and do not have measurement-value coverage statistics.

## Example

```python
from pathlib import Path
from flovopy.legacy.bob import read_ssam, read_mudflow, read_tilt

legacy = Path('/path/to/LEGACY/BOB')
fair = Path('/path/to/FAIR/BOB')

ssam = read_ssam(legacy / 'SSAM' / 'ST01', strict=False)
print(ssam.coverage())
print(ssam.errors)
ssam.write(fair / 'SSAM', prefix='ST01')

mudflow = read_mudflow(legacy / 'MUDFLOW.zip')
# Three raw positional channels remain distinct; no station/calibration guesses.

# For other centuries, specify both the year and filename-century policy.
tilt = read_tilt(legacy / 'TILT', year=2020, century=2000)
```

`write()` creates record/error CSVs and metadata JSON only at an explicit caller-selected destination. It refuses existing output filenames unless `overwrite=True`. The library has no global output-root configuration; application notebooks decide paths.

## Optional plotting

`plot_series()` requires an explicit series if multiple are present. It returns a figure and draws points by default, so absent samples are not bridged by lines. `connect=True` requests lines explicitly; missing values in existing rows remain NaN breaks, but absent timestamps may be bridged. Each positional variable has its own axes/scale. Functions do not select a backend, close the returned figure or save files automatically; callers do that.

`plot_ssam()` draws individual time/band cells, not an interpolated image. The caller must supply `interval_seconds`, the intended display duration of each record. Absent records remain blank. Duplicate timestamps must be resolved explicitly before plotting. No frequency bounds are guessed: without documented `band_edges_hz`, the vertical axis is positional band index. Values are not converted to dB or physical units.

```python
from flovopy.legacy.bob import plot_ssam
ax, image = plot_ssam(ssam, series='ST01', interval_seconds=60)
ax.figure.colorbar(image, ax=ax, label='Uncalibrated source band value')
```

## Migration status and validation

The package recovers/generalizes earlier BOB audit readers. Existing Pinatubo notebooks/local legacy_bob code are left intact until their outputs are compared and migrated; this addition does not silently change an existing pipeline run or remove the earlier TILT class.

Tests cover ZIP/folder parity, missing data, bad dates/widths, duplicate preservation, leap-year annual lengths, signed data, explicit century policy, concatenated COSPEC fields, zero/blank counts and export overwrite protection. Representative real files were read for SSAM, mudflow, tilt, RSAM, rain, COSPEC, counts and radio maps. The SSAM PPE1 record count matches the recovered earlier audit (1,667 rows). A full cloud-backed ABAC archive comparison was stopped while waiting on a file download; one daily mudflow file was decoded successfully (70 rows). Native-format tests and small real-file checks are not a claim that every BOB variant is supported or every measurement is calibrated.


## Direct SSAM and daily multi-channel spectrograms

SSAM can be adapted to precomputed band data without synthesizing a waveform or
calculating another FFT. The generic reader does not assume Pinatubo bands:
`band_definitions="documents"` explicitly ingests archived `BNnn low Hz to high Hz`
TXT descriptions. Missing or conflicting numeric definitions raise. Each band
retains source names, description lines, and tentative wording such as `??`.
Alternatively supply one `{name, low_hz, high_hz}` dictionary per value column.
Overlapping bands are supported; a single vector of contiguous edges cannot
represent all sixteen historical Pinatubo bands.

```python
from flovopy.legacy.bob import read_ssam, ssam_to_spectrograms
from flovopy.processing.spectrograms import (
    plot_band_spectrograms, plot_daily_band_spectrograms,
)

data = read_ssam("/path/to/unpacked/SSAM", band_definitions="documents")
results = ssam_to_spectrograms(
    data, interval_seconds=60, time_anchor="start", duplicate_policy="mask",
)
heatmap = [f"BN{i:02d}" for i in range(1, 11)]
broad = [f"BN{i:02d}" for i in range(11, 17)]
fig = plot_band_spectrograms(
    results, spectral_bands=heatmap, broad_bands=broad,
    start="1991-06-12", end="1991-06-13",
)
paths = plot_daily_band_spectrograms(
    results, "/path/to/external/FAIR/BOB/ssam/day_pages",
    page="day", start="1991-06-01", end="1991-06-17",
    spectral_bands=heatmap, broad_bands=broad,
)
```

`BandSpectrogramResult` lives in `flovopy.processing.spectrograms`. Its times are
recorded calendar labels, not an assertion of UTC/FFT centres. Each station/source
series keeps all band values, independent low/high limits, missing values and
provenance. It deliberately differs from waveform `SpectrogramResult`, whose
frequency bins and UTC FFT-centred timestamps have different semantics. The
modern band plotting API is the compatibility path; it does not use `TFS2trace`
or construct a synthetic seismogram. Existing waveform and legacy IceWeb APIs
are unchanged.

For Pinatubo, BN01–10 form a contiguous 0.5–10.5 Hz heatmap. BN11–16 overlap and
are separate labelled line plots. Attempting to draw overlapping bands as one
frequency-axis heatmap raises. Values remain reported amplitude-like measures in
arbitrary units: no division by band width, PSD/ASD conversion, dB transformation,
calibration or automatic station-gain normalization. `log_colors=True` offers
positive-value logarithmic color normalization; zero/nonpositive cells are blank
in that display, but the underlying measurements are unchanged.

The explicit 60-second display support in the example reflects observed record
spacing. It does not establish the acquisition averaging window or whether the
stamp is its start, centre or end. `time_anchor` records that plotting convention.
The API rejects display widths longer than adjacent record spacing, avoiding
painted-over observations. No time corrections, interpolation, gap filling or
station-name translation occur. The hourly/daily boundaries follow recorded
calendar labels, with exclusive `end`. Missing rows leave blank cells and broken
broad-band traces.

Duplicate timestamps raise by default. Explicit `duplicate_policy="mask"` blanks
all bands at ambiguous timestamps in the plotting copy, retaining every original
duplicate record in `metadata["duplicate_time_records"]`. The source DataFrame
and its CSV are unchanged. This also masks identical duplicates conservatively.
No first/last/mean policy is applied. Invalid timestamps always raise and require
source review. SSAM and DATA source versions should be plotted separately until
station identities and overlapping records have been reconciled.

Daily/hourly batch output defaults to no overwrites, closes each generated figure,
and writes `ssam_pages.json` with band definitions, source metadata, clock basis,
display-support convention and plotting settings. Pages are generated only for
calendar windows containing records. Individual page/channel scales are the
default; provide explicit `clim=(low, high)` for comparisons across pages. Shared
color scales require matching quantity/units, and still do not establish equal
physical station gains.
