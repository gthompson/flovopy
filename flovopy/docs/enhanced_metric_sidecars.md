# Complete enhanced-event metrics and ensemble exports

`EnhancedEvent` is the per-event persistence container. Store every trace/window's complete output dictionaries in `event.meta.traces`; event-wide provenance belongs in `event.meta.metrics`. `event.save(..., write_quakeml=False)` writes a JSON-only sidecar, and `EnhancedEvent.load(base, require_quakeml=False)` loads it without waveform access. Default loading still requires the paired QuakeML for backward compatibility.

Scientific JSON now preserves numeric arrays as arrays, booleans as booleans, zeros as zeros, nested dictionaries and timestamp strings. Nonfinite/missing numbers become JSON null. Unsupported objects raise rather than silently becoming a truncated string. JSON writes are atomic. Complex numbers use an explicit `__flovopy_type__` object; arrays load as numeric nested lists. A JSON-only load reconstructs enhanced metadata and resource ID, not absent core origins/picks.

```python
from flovopy.enhanced.event import EnhancedEvent, EnhancedEventMeta
from flovopy.enhanced.catalog import EnhancedCatalog

record = EnhancedEvent(
    resource_id='smi:local/record/example',
    meta=EnhancedEventMeta(
        wav_paths=['event.mseed'],
        metrics={'processing_version': 'example-1'},
        traces=[{'id': 'XB.BUG..EHZ', 'trace_index': 0, 'window': 'noise',
                 'metrics': all_noise_metrics, 'features': complete_noise_feature_result},
                {'id': 'XB.BUG..EHZ', 'trace_index': 0, 'window': 'event',
                 'metrics': all_event_metrics, 'features': complete_event_feature_result}],
    ),
)
record.save('sidecars', 'example', write_quakeml=False)
cat = EnhancedCatalog.from_sidecars(['sidecars/example.json'])
cat.export_metrics_csv('all_trace_window_metrics.csv')
cat.export_metrics_csv('all_event_metadata.csv', level='event')
```

`EnhancedCatalog.from_sidecars()` takes an explicit list so callers can select exactly one active cache version per record. Duplicate event resource IDs raise, preventing old/new versions from silently being pooled. `metrics_dataframe(level='trace')` / `export_metrics_csv()` export every sidecar key dynamically. New metric fields require no whitelist update. Dictionaries become dotted column paths; arrays/lists remain complete JSON-valued CSV cells. Ambiguous flattened paths raise. Trace IDs can repeat across windows/components; trace entry index and caller-supplied trace index/window preserve identity.

The existing `to_dataframe()` / `export_csv()` remain conventional event summaries; they do **not** export all waveform features. `level='event'` exports every enhanced field with the complete trace list as a JSON cell. JSON sidecars are the authoritative nested representation; CSV is a derived ensemble view.

QuakeML remains standards-compliant: `cat.write('events.qml', format='QUAKEML')` exports actual core Event/Origin/Pick/Magnitude objects. Arbitrary waveform metric dictionaries remain in JSON. Never invent an earthquake origin/time/location/magnitude from a waveform-file start merely to populate QuakeML.

## Incremental workflow pattern

Use a stable record identity and a computation signature based on the source fingerprint, accepted windows, processing options and relevant algorithm versions. Keep the cache outside selection-specific run folders. Save immutable versions such as `<record-id>/<signature>/event.json`, and maintain an active-version index for the selected ensemble. A new file adds one record; changed windows invalidate that record; CSV/QuakeML exports can be rebuilt from JSON without waveform or feature processing. Notes-only changes can refresh metadata without numeric recalculation.

The Pinatubo implementation follows this pattern in `20_analysis/analysis_workflow.py`. It preserves all EnhancedStream metrics, original spectral/spectrum arrays, SSAM/band ratios, station summaries, complete FLOVOpy feature objects, PSD arrays/settings, artifact audits and quality flags for both accepted window types. The original computed outputs remain available even where its plotting view masks unreliable short-band values. Physical motion metrics still require a defensible end-to-end instrument response.

## EventRate

EventRate aggregates catalog records; it does not persist full metrics or replace sidecars. Default `EventRateConfig(time_source='origin')` uses actual origin times. To count waveform records explicitly from sidecar metadata:

```python
from flovopy.enhanced.eventrate import EventRateConfig
rate = cat.to_event_rate(EventRateConfig(
    interval='D', time_source='recorded_file', include_energy=False,
))
```

`recorded_file` reads `event.meta.metrics['record_reference_recorded_time']`. Such counts are selected recording-file rates, not earthquake occurrence/completeness or station uptime. Events without magnitudes still count; `magnitude_count` records the known-magnitude subset. Missing magnitudes stay missing instead of becoming fabricated small earthquakes, and disabled energy calculations remain missing. Catalogs with no eligible timestamps return an empty table.
