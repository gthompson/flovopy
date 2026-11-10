# Quiet prefixes and onset support in triggered records

Use `detect_pretrigger_window()` to propose trace-specific noise prefixes and onset evidence within an **already recorded event waveform**. It works with an ObsPy Trace or Stream; it does not require continuous data.

```python
from obspy import read
from flovopy.processing.detection import PretriggerSettings, detect_pretrigger_window

st = read('event.mseed')  # exclude timing/auxiliary channels before analysis
settings = PretriggerSettings(
    reference_end_s=5,
    minimum_noise_s=5,
    noise_guard_s=1,
    network_min_stations=2,
)
result = detect_pretrigger_window(st, settings=settings)
print(result['qc'][[
    'trace_id', 'selected_onset_record_s',
    'onset_support_start_record_s', 'onset_support_end_record_s',
    'proposed_noise_end_record_s', 'noise_proposal_status', 'qc_flags',
]])
```

The same implementation is available from VDAP for discoverability:

```python
from flovopy.core import vdap
result = vdap.detect_pretrigger_window(st, settings=settings)
```

There are no separate VDAP defaults or duplicate algorithms. The implementation lives in `flovopy.processing.event_windows`, exposed through `processing.detection`. It neither writes files nor modifies input samples, metadata or timestamps. It does not silently exclude named stations; callers decide which channels are waveforms. File discovery, plots, exports and manual acceptance belong in the caller.

## What it estimates

The default working assumption is that the first five seconds of each trace are noise. After reference checks, sustained strong amplitude and amplitude-supported STA/LTA evidence establish the first event supported by two distinct stations within three seconds. Multiple components at one network/station count once. The local search interval spans two seconds before to eight seconds after the anchor.

Within that interval, the earliest weak/strong amplitude or STA/LTA rise proposes an onset. A changed one-second RSAM bin may backtrack an existing fine/STA candidate by at most two seconds, but cannot independently create an arrival. Reference-centred amplitude uses a fixed initial baseline, which can help expose emergent rises that a moving LTA follows. Noise ends one second before the selected evidence, subject to an audit for earlier sustained increases, strong isolated transients, gaps or incomplete bins. Reference contamination prevents a trace from voting for a network cluster. Inconsistent minimum and guard requirements leave the noise end unresolved.

A trace without local event evidence keeps only the assumed minimum. A late secondary burst cannot extend its noise through the main network event. With no association, defaults retain the minimum and flag the record unresolved. Shared interference can still coincide; station support is not an earthquake classification.

All results remain **UNREVIEWED**. Quiet-looking data, low-amplitude activity already in the reference, zero-filled historical gaps, mixed events and gradual onsets require inspection. The first five seconds are not independently proven quiet. No acquisition trigger or configured buffer duration is recovered. Event/coda ends remain unset.

## Results and timing

- `qc`: per-trace proposals, onset-support bounds, measured reference properties, actual rounded window lengths and QC flags.
- `network`: distinct-station association evidence.
- `evidence`: all candidate crossings and their methods/support, including unselected candidates.
- `details`: per-trace waveform-centred arrays, one-second/fine metrics, STA/LTA and optional frequency/Hilbert tables.
- `reference_time`: earliest recorded trace start. It uses the supplied clock, with no UTC correction.
- `settings` and `algorithm_version`: provenance to include in saved products.

`_trace_s` columns are seconds from each trace start; `_record_s` columns are seconds from `reference_time`. Account for offsets when comparing channels. Proposed noise intervals are half-open `[start, end)`. Invalid/missing endpoints are NaN. Empty Streams return empty tables and an unresolved network context.

`onset_support_start_*` and `onset_support_end_*` cover the first bin/sustained measurements that support the candidate, rather than the entire event. They are **not confidence intervals** and do not assert that physical onset lies within the bounds. In particular, gradual rockfall/PDC transitions may begin below a detector's threshold. This is an onset screening tool, not a validated classifier or precise phase picker. Manually labelled examples and holdout evaluation are needed for those applications.

## Configuration for other datasets

For an explicitly assumed shorter quiet prefix at one station:

```python
settings = PretriggerSettings(
    reference_end_s=2,
    minimum_noise_s=2,
    noise_guard_s=0.5,
    lta_s=1.5,
    network_min_stations=1,  # explicit single-station policy
    compute_frequency=False,
    compute_hilbert=False,
)
result = detect_pretrigger_window(trace, settings=settings)
```

The reference must start at trace time zero and cover the minimum proposed noise duration. Do not shorten it without a defensible acquisition assumption. Defaults require two stations even when passed one Trace: pass `network_min_stations=1` deliberately to permit extension from a single station.

Amplitude thresholds use robust reference scatter and sustained crossings; they are configurable screening heuristics, not significance tests. Metric window lengths/steps are rounded to whole sample counts without resampling data; actual durations are recorded in QC. Choose windows resolvable at the sampling rate (fine bins need at least one sample; PSD windows need at least four and steps at least one). Tune the network gate for dataset geometry rather than treating it as a travel-time calculation.

Frequency diagnostics do not determine the onset. Band-limited PSD centroid/divergence and broadband signed Hilbert instantaneous frequency measure different properties. Hilbert estimates retain negative values, exclude low-envelope/edge samples and report coverage. Smoothing is confined to finite runs. Missing diagnostic evidence does not establish noise. Disable either diagnostic using the switches above when it is not needed.

## Validation

Tests cover distinct-station association, weak late channels, earlier rises, transients, initial padding/contamination, guard conflicts, single-station policy, shorter references, sample-rate rounding, timing offsets, input preservation, support intervals, optional diagnostics and VDAP equivalence. These verify mechanics; they do not measure onset accuracy for a scientific dataset. Review a stratified labelled sample, reserve a holdout set, and evaluate noise contamination and unresolved rates before automatically using windows for SNR or spectral comparisons.
