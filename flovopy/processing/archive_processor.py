"""Source-independent, restartable processing of consecutive waveform windows.

A WaveformSource supplies ObsPy Streams; a WindowProcessor writes artifacts into
its unique task directory. The scheduler owns time partitioning, retries,
checkpoints, and parallel execution. It never silently interpolates/merges data.

All times UTC; processing intervals are half-open [start, end). A task reads
[start-halo, end+halo] and passes its *owned* interval separately to the
processor, which must crop its output to owned windows to prevent duplicates.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from obspy import Stream, UTCDateTime, read


@dataclass(frozen=True)
class Selectors:
    network: str = '*'
    station: str = '*'
    location: str = '*'
    channel: str = '*'

    def matches(self, trace) -> bool:
        return all(fnmatchcase(str(getattr(trace.stats, key)), getattr(self, key))
                   for key in ('network', 'station', 'location', 'channel'))


def _utc(value):
    return UTCDateTime(value)


def _atomic_json(path: Path, obj: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class WaveformSource(Protocol):
    def read(self, start: UTCDateTime, end: UTCDateTime, selectors: Selectors) -> Stream: ...


class WindowProcessor(Protocol):
    def process(self, stream: Stream, context: 'WindowContext', output_dir: Path) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SourceSpec:
    """Serializable source configuration for multiprocessing and checkpoints.

    kind: sds, fdsn, earthworm, files, or plugin.
    'files' supports any formats readable by ObsPy read(), including many
    SEISAN waveform files and CSS waveform descriptor files. This is NOT a
    native SEISAN S-file index or Antelope/Datascope database query interface.
    """
    kind: str
    options: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.kind not in {'sds', 'fdsn', 'earthworm', 'files', 'plugin'}:
            raise ValueError(f'Unsupported source kind: {self.kind}')
        needed = {'sds': ('root',), 'fdsn': ('base_url',),
                  'earthworm': ('host', 'port'), 'files': ('paths',),
                  'plugin': ('factory',)}[self.kind]
        for key in needed:
            if key not in self.options:
                raise ValueError(f'{self.kind} source requires option {key!r}')
        if self.kind == 'files' and not self.options['paths']:
            raise ValueError('files source requires at least one path/glob')
        if self.kind == 'plugin' and ':' not in self.options['factory']:
            raise ValueError('plugin factory must be "module:callable"')


def _load_factory(name: str):
    module, attr = name.split(':', 1)
    value = importlib.import_module(module)
    for part in attr.split('.'):
        value = getattr(value, part)
    return value


def _source_read(spec: SourceSpec, start, end, selectors: Selectors) -> Stream:
    spec.validate()
    opt = spec.options
    if spec.kind == 'sds':
        from flovopy.enhanced.sdsclient import EnhancedSDSClient
        result = EnhancedSDSClient(opt['root']).read(
            start, end, net=selectors.network, sta=selectors.station,
            loc=selectors.location, chan=selectors.channel,
            skip_low_rate_channels=False, merge=None,
            postprocess=False, final_smart_merge=False)
    elif spec.kind == 'fdsn':
        from obspy.clients.fdsn import Client
        client = Client(opt['base_url'], **opt.get('client_kwargs', {}))
        result = client.get_waveforms(selectors.network, selectors.station,
                                      selectors.location, selectors.channel, start, end,
                                      **opt.get('waveform_kwargs', {}))
    elif spec.kind == 'earthworm':
        from obspy.clients.earthworm import Client
        client = Client(opt['host'], int(opt['port']), **opt.get('client_kwargs', {}))
        result = client.get_waveforms(selectors.network, selectors.station,
                                      selectors.location, selectors.channel, start, end)
    elif spec.kind == 'files':
        from glob import glob
        paths = sorted({p for pattern in opt['paths'] for p in glob(str(pattern), recursive=True)})
        result = Stream()
        for path in paths:
            if not Path(path).is_file():
                continue
            # ObsPy read() supports many formats, but not every Antelope database.
            st = read(path, starttime=start, endtime=end, **opt.get('read_kwargs', {}))
            result += st
    else:
        factory = _load_factory(opt['factory'])
        result = factory(**opt.get('kwargs', {})).read(start, end, selectors)
    if not isinstance(result, Stream):
        raise TypeError(f'{spec.kind} source must return obspy.Stream, got {type(result)!r}')
    result = result.select()  # shallow copy before trimming
    result = Stream(tr.copy() for tr in result if selectors.matches(tr))
    # ObsPy trim includes the end sample by default; owner must crop its outputs.
    result.trim(starttime=start, endtime=end, nearest_sample=False, pad=False)
    return result


@dataclass(frozen=True)
class WindowContext:
    start: str
    end: str
    read_start: str
    read_end: str
    index: int
    run_id: str

    @property
    def starttime(self): return _utc(self.start)

    @property
    def endtime(self): return _utc(self.end)


@dataclass(frozen=True)
class ArchiveConfig:
    source: SourceSpec
    output_root: str
    start: str
    end: str
    processor: str  # importable "module:factory", instantiated with processor_kwargs
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    selectors: Selectors = field(default_factory=Selectors)
    window_seconds: float = 86400.0
    halo_seconds: float = 0.0
    workers: int = 1
    resume: bool = True
    # Increment when changing science algorithms without changing kwargs.
    algorithm_version: str = '1'

    def validate(self):
        self.source.validate()
        if _utc(self.end) <= _utc(self.start):
            raise ValueError('end must be greater than start (exclusive end)')
        if self.window_seconds < 1 or self.halo_seconds < 0:
            raise ValueError('window_seconds >= 1 and halo_seconds >= 0 required')
        if self.workers < 1:
            raise ValueError('workers must be >= 1')
        if ':' not in self.processor:
            raise ValueError('processor must be an importable "module:factory"')
        if self.source.kind == 'files' and self.workers > 1:
            # Allowed but each task may scan all files; user should supply time-scoped plugin for large DBs.
            pass


def _fingerprint(config: ArchiveConfig) -> str:
    value = asdict(config)
    for key in ('resume', 'workers', 'start', 'end', 'output_root'):
        value.pop(key)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _windows(config: ArchiveConfig):
    """UTC-epoch aligned windows, clipped to requested interval."""
    start, end = _utc(config.start), _utc(config.end)
    step = config.window_seconds
    origin = _utc(0)
    k = int((start - origin) // step)
    t = origin + k * step
    index = 0
    while t < end:
        a, b = max(start, t), min(end, t + step)
        if a < b:
            yield index, a, b
            index += 1
        t += step


def _task(config: ArchiveConfig, context: WindowContext):
    root = Path(config.output_root) / 'runs' / context.run_id / 'windows' / f'{context.index:09d}'
    marker = root / 'complete.json'
    if config.resume and marker.is_file():
        old = json.loads(marker.read_text())
        if (old.get('run_id') == context.run_id and
            old.get('start') == context.start and old.get('end') == context.end and
            all((root / name).is_file() for name in old.get('artifacts', []))):
            return {'index': context.index, 'status': 'skipped', 'artifacts': old['artifacts']}
    root.mkdir(parents=True, exist_ok=True)
    # No stale completion marker may survive a failed retry.
    marker.unlink(missing_ok=True)
    source_stream = _source_read(config.source, _utc(context.read_start),
                                 _utc(context.read_end), config.selectors)
    processor = _load_factory(config.processor)(**config.processor_kwargs)
    metadata = processor.process(source_stream, context, root)
    if not isinstance(metadata, Mapping):
        raise TypeError('processor.process() must return a mapping')
    artifacts = metadata.get('artifacts', [])
    if not isinstance(artifacts, (list, tuple)):
        raise TypeError('processor artifacts must be a list of relative paths')
    for name in artifacts:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or not (root / p).is_file():
            raise ValueError(f'Invalid or missing processor artifact: {name!r}')
    _atomic_json(marker, {'run_id': context.run_id, 'index': context.index,
                          'start': context.start, 'end': context.end,
                          'artifacts': artifacts, 'metadata': dict(metadata),
                          'completed_utc': datetime.now(timezone.utc).isoformat()})
    return {'index': context.index, 'status': 'processed', 'artifacts': artifacts}


def run(config: ArchiveConfig) -> dict[str, Any]:
    """Run independent windows. Failed tasks raise; completed tasks resume.

    Processor instances are created per task. Each task owns its own directory.
    No shared output is written by workers. Consolidation is a separate phase.
    """
    config.validate()
    run_id = _fingerprint(config)
    base = Path(config.output_root) / 'runs' / run_id
    base.mkdir(parents=True, exist_ok=True)
    _atomic_json(base / 'config.json', {'run_id': run_id, 'config': asdict(config)})
    contexts = [WindowContext(str(a), str(b), str(a - config.halo_seconds),
                              str(b + config.halo_seconds), i, run_id)
                for i, a, b in _windows(config)]
    results = []
    if config.workers == 1:
        for ctx in contexts:
            results.append(_task(config, ctx))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {executor.submit(_task, config, ctx): ctx for ctx in contexts}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item['index'])
    return {'run_id': run_id, 'windows': len(contexts),
            'processed': sum(x['status'] == 'processed' for x in results),
            'skipped': sum(x['status'] == 'skipped' for x in results),
            'output_dir': str(base)}


class MiniSeedWindowWriter:
    """Demonstration processor; writes raw fetched waveforms per owned interval.

    This is a test/example processor, not a SAM/spectrogram engine. Output
    includes the right boundary sample only when ObsPy's trim includes it;
    sample ownership is enforced by explicitly using an end-minus-one-sample
    selection for each trace.
    """
    def process(self, stream: Stream, context: WindowContext, output_dir: Path):
        st = stream.copy()
        a, b = context.starttime, context.endtime
        for tr in list(st):
            tr.trim(starttime=a, endtime=b - tr.stats.delta,
                    nearest_sample=False, pad=False)
            if tr.stats.npts == 0:
                st.remove(tr)
        if not st:
            return {'artifacts': [], 'traces': 0}
        filename = 'waveforms.mseed'
        # Write to a temporary file, then atomically publish.
        tmp = output_dir / 'waveforms.partial.mseed'
        st.write(str(tmp), format='MSEED')
        os.replace(tmp, output_dir / filename)
        return {'artifacts': [filename], 'traces': len(st)}


class SAMWindowProcessor:
    """SAM metric adapter; scientific computation is owned by sam.py.

    Each task writes independent per-trace shards. The scheduler writes the
    completion marker; archive_metrics.py performs serial yearly consolidation.
    """
    def __init__(self, products=('RSAM',), sampling_interval=60.0,
                 filter_band=(0.5, 18.0), bands=None, corners=4,
                 stationxml=None, pre_filt=None, shard_format='pickle', source=None,
                 reduction_q=None, reduction_wavespeed_kms=None, reduction_peak_frequency=None):
        self.products = tuple(p.upper() for p in products)
        self.sampling_interval = float(sampling_interval)
        self.filter_band = filter_band
        self.bands = bands
        self.corners = corners
        self.stationxml = stationxml
        self.pre_filt = pre_filt
        self.shard_format = shard_format
        self.source = source
        self.reduction_q = reduction_q
        self.reduction_wavespeed_kms = reduction_wavespeed_kms
        self.reduction_peak_frequency = reduction_peak_frequency
        if shard_format not in ('pickle', 'csv', 'parquet'):
            raise ValueError('Unsupported SAM shard format')
        if any(p in ('DR', 'DRS', 'VR', 'VRS', 'ER') for p in self.products) and not source:
            raise ValueError('Reduced metrics require source coordinates')
        if any(p != 'RSAM' for p in self.products) and not stationxml:
            raise ValueError('Calibrated SAM products require stationxml')

    def process(self, stream: Stream, context: WindowContext, output_dir: Path):
        from flovopy.processing.archive_metrics import (
            METRICS, _merge_stream, _inventory, _calibrated_stream, _atomic_frame)
        import pandas as pd
        if any(p not in METRICS for p in self.products):
            raise ValueError(f'Unknown SAM product(s): {self.products}')
        raw = _merge_stream(stream)
        inv = _inventory(self.stationxml) if self.stationxml and raw else None
        from flovopy.processing.archive_metrics import REDUCED_PARENTS
        artifacts, outputs = [], []
        cache = {}

        def get_metric(product):
            if product in cache:
                return cache[product]
            if product in REDUCED_PARENTS:
                parent = get_metric(REDUCED_PARENTS[product])
                kwargs = dict(Q=self.reduction_q,
                              wavespeed_kms=self.reduction_wavespeed_kms,
                              peakf=self.reduction_peak_frequency)
                if product in ('DR', 'DRS'):
                    obj = parent.compute_reduced_displacement(
                        inv, self.source, surfaceWaves=(product == 'DRS'), **kwargs)
                elif product in ('VR', 'VRS'):
                    obj = parent.compute_reduced_velocity(
                        inv, self.source, surfaceWaves=(product == 'VRS'), **kwargs)
                else:
                    kwargs.pop('peakf')
                    obj = parent.compute_reduced_energy(
                        inv, self.source, fixpeakf=self.reduction_peak_frequency,
                        verbose=False, **kwargs)
            else:
                st = raw if product == 'RSAM' else _calibrated_stream(
                    raw, inv, product, self.pre_filt)
                if not st:
                    return None
                obj = METRICS[product](stream=st, sampling_interval=self.sampling_interval,
                                       filter=self.filter_band, bands=self.bands,
                                       corners=self.corners, align='utc')
            cache[product] = obj
            return obj

        for product in self.products:
            obj = get_metric(product)
            if obj is None:
                continue
            for trace_id, frame in obj.dataframes.items():
                times = pd.to_numeric(frame['time'], errors='coerce')
                keep = ((times >= float(context.starttime.timestamp) - 1e-7) &
                        (times + self.sampling_interval <= float(context.endtime.timestamp) + 1e-7))
                frame = frame.loc[keep].copy()
                if frame.empty:
                    continue
                token = hashlib.sha256((product + '\0' + trace_id).encode()).hexdigest()[:20]
                suffix = {'pickle': '.pkl', 'csv': '.csv', 'parquet': '.parquet'}[self.shard_format]
                name = f'{product}_{token}{suffix}'
                _atomic_frame(frame, output_dir / name, self.shard_format)
                artifacts.append(name)
                outputs.append({'filename': name, 'product': product, 'trace_id': trace_id})
        return {'artifacts': artifacts, 'outputs': outputs, 'products': list(self.products)}
