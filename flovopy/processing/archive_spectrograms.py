"""Restartable numerical spectrogram archives using archive_processor.

Spectra are stored as compressed, per-task/per-channel NPZ shards. This module
is independent of IceWeb and of the underlying waveform source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from obspy import Stream, UTCDateTime

from flovopy.processing.archive_processor import (
    ArchiveConfig, Selectors, SourceSpec, WindowContext, run as run_scheduler,
)
from flovopy.processing.spectrograms import SpectrogramResult, compute_spectrogram_result


def _combine_fragments(stream: Stream) -> Stream:
    """Combine nonoverlapping fragments of the same ID, retaining masked gaps.

    Overlapping fragments are rejected: choosing one would silently alter PSD.
    Sampling-rate changes within a trace ID are also rejected.
    """
    groups = {}
    for tr in stream:
        groups.setdefault(tr.id, []).append(tr.copy())
    out = Stream()
    for trace_id, traces in groups.items():
        traces.sort(key=lambda t: t.stats.starttime)
        fs = float(traces[0].stats.sampling_rate)
        for left, right in zip(traces, traces[1:]):
            if not np.isclose(right.stats.sampling_rate, fs):
                raise ValueError(f'{trace_id}: inconsistent sampling rates')
            # Adjacent samples are separated by 1/fs; anything earlier overlaps.
            if right.stats.starttime <= left.stats.endtime + 0.1 / fs:
                raise ValueError(f'{trace_id}: overlapping waveform fragments')
        merged = Stream(traces)
        merged.merge(method=0, fill_value=None)
        if len(merged) != 1:
            raise ValueError(f'{trace_id}: fragments could not be merged')
        out += merged
    return out


def _safe_name(trace_id: str) -> str:
    return hashlib.sha256(trace_id.encode()).hexdigest()[:20]


def _write_npz(path: Path, result: SpectrogramResult) -> None:
    """Write via a unique temporary file then atomically publish."""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as fh:
        tmp = Path(fh.name)
    try:
        np.savez_compressed(
            tmp, trace_id=np.array(result.trace_id), times=result.times,
            frequencies=result.frequencies, values=result.values,
            coverage=result.coverage, quantity=np.array(result.quantity),
            units=np.array(result.units if result.units is not None else ''),
            metadata=np.array(json.dumps(result.metadata, sort_keys=True, default=str)),
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_spectrogram(path: str | Path) -> SpectrogramResult:
    """Load a shard written by SpectrogramWindowProcessor (no pickle)."""
    with np.load(path, allow_pickle=False) as data:
        units = str(data['units'].item())
        return SpectrogramResult(
            trace_id=str(data['trace_id'].item()),
            times=data['times'].copy(), frequencies=data['frequencies'].copy(),
            values=data['values'].copy(), coverage=data['coverage'].copy(),
            quantity=str(data['quantity'].item()), units=units or None,
            metadata=json.loads(str(data['metadata'].item())),
        )


class SpectrogramWindowProcessor:
    """Archive scheduler adapter: owns FFT centres in [start, end)."""

    def __init__(self, *, fft_seconds=2.56, step_seconds=0.64,
                 window='hann', quantity='psd', detrend='constant',
                 min_coverage=1.0, units=None):
        self.kwargs = dict(fft_seconds=fft_seconds, step_seconds=step_seconds,
                           window=window, quantity=quantity, detrend=detrend,
                           align='utc', min_coverage=min_coverage, units=units)

    def process(self, stream: Stream, context: WindowContext, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        a, b = float(UTCDateTime(context.start)), float(UTCDateTime(context.end))
        artifacts = []
        manifest = []
        for tr in _combine_fragments(stream):
            result = compute_spectrogram_result(tr, **self.kwargs)
            keep = (result.times >= a) & (result.times < b)
            if not np.any(keep):
                continue
            result.times = result.times[keep]
            result.values = result.values[:, keep]
            result.coverage = result.coverage[keep]
            name = f'spectrum_{_safe_name(tr.id)}.npz'
            _write_npz(output_dir / name, result)
            artifacts.append(name)
            manifest.append({'trace_id': tr.id, 'file': name,
                             'columns': len(result.times),
                             'frequency_bins': len(result.frequencies)})
        return {'artifacts': artifacts, 'spectrograms': manifest}


@dataclass(frozen=True)
class SpectrogramArchiveConfig:
    source: SourceSpec
    output_root: str
    start: str
    end: str
    selectors: Selectors = field(default_factory=Selectors)
    fft_seconds: float = 2.56
    step_seconds: float = 0.64
    window: str = 'hann'
    quantity: str = 'psd'
    detrend: str | bool = 'constant'
    min_coverage: float = 1.0
    units: str | None = None
    window_seconds: float = 86400.0
    halo_seconds: float | None = None
    workers: int = 1
    resume: bool = True
    algorithm_version: str = 'spectrogram-pass4-v1'

    def scheduler_config(self) -> ArchiveConfig:
        # A full FFT-length halo is conservative and covers all centre-owned FFTs.
        halo = self.fft_seconds if self.halo_seconds is None else self.halo_seconds
        if halo < self.fft_seconds:
            raise ValueError('halo_seconds must be >= fft_seconds for chunk equivalence')
        if self.step_seconds <= 0 or self.fft_seconds <= 0:
            raise ValueError('fft_seconds and step_seconds must be positive')
        return ArchiveConfig(
            source=self.source, output_root=self.output_root,
            start=self.start, end=self.end, selectors=self.selectors,
            processor='flovopy.processing.archive_spectrograms:SpectrogramWindowProcessor',
            processor_kwargs=dict(fft_seconds=self.fft_seconds,
                                  step_seconds=self.step_seconds,
                                  window=self.window, quantity=self.quantity,
                                  detrend=self.detrend,
                                  min_coverage=self.min_coverage, units=self.units),
            window_seconds=self.window_seconds, halo_seconds=halo,
            workers=self.workers, resume=self.resume,
            algorithm_version=self.algorithm_version,
        )


def run(config: SpectrogramArchiveConfig) -> dict[str, Any]:
    return run_scheduler(config.scheduler_config())


def iter_spectrograms(output_root: str | Path, run_id: str, *, trace_id=None,
                      start=None, end=None):
    """Yield owned shards chronologically, optionally clipped to [start, end).

    No full-archive consolidation is performed; consumers may stream shards.
    """
    base = Path(output_root) / 'runs' / run_id / 'windows'
    a = -np.inf if start is None else float(UTCDateTime(start))
    b = np.inf if end is None else float(UTCDateTime(end))
    for marker in sorted(base.glob('*/complete.json')):
        manifest = json.loads(marker.read_text())
        for item in manifest.get('metadata', {}).get('spectrograms', []):
            if trace_id is not None and item['trace_id'] != trace_id:
                continue
            result = load_spectrogram(marker.parent / item['file'])
            keep = (result.times >= a) & (result.times < b)
            if not np.any(keep):
                continue
            result.times = result.times[keep]
            result.values = result.values[:, keep]
            result.coverage = result.coverage[keep]
            yield result
