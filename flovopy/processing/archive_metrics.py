"""Restartable SDS -> windowed SAM metrics. Future spectrogram engines can reuse the task layer.

Half-open processing interval [start, end); daily tasks are UTC-aligned. Each task
reads a configurable halo, computes metrics, and retains only fully-contained
windows. No interpolation or implicit downsampling is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream, read_inventory

from flovopy.processing.sam import RSAM, VSAM, DSAM, VSEM, DR, DRS, VR, VRS, ER
from flovopy.processing.archive_processor import (
    ArchiveConfig, SourceSpec, Selectors, run as run_scheduler, _windows,
)

METRICS = {'RSAM': RSAM, 'VSAM': VSAM, 'DSAM': DSAM, 'VSEM': VSEM,
           'DR': DR, 'DRS': DRS, 'VR': VR, 'VRS': VRS, 'ER': ER}
REDUCED_PARENTS = {'DR': 'DSAM', 'DRS': 'DSAM', 'VR': 'VSAM',
                   'VRS': 'VSAM', 'ER': 'VSEM'}
BAND_PRESETS = {
    'volcano': {'VLP': [0.02, 0.2], 'LP': [0.5, 4], 'VT': [4, 18]},
    'storm_seismo': {'PRI': [0.05, 0.10], 'SEC': [0.10, 0.35], 'HI': [1, 5]},
    'storm_infra': {'TC': [0.01, 0.10], 'MB': [0.15, 0.35], 'TH': [1, 10]},
    'storm': {'PRI': [0.02, 0.10], 'SEC': [0.10, 0.35], 'HI': [1, 10]},
}


@dataclass(frozen=True)
class Config:
    sds_root: str
    output_root: str
    start: str
    end: str
    products: tuple[str, ...] = ('RSAM',)
    network: str = '*'
    station: str = '*'
    location: str = '*'
    channel: str = '*'
    sampling_interval: float = 60.0
    filter_band: Optional[tuple[float, float]] = (0.5, 18.0)
    bands: Optional[dict] = None
    corners: int = 4
    stationxml: Optional[str] = None
    pre_filt: Optional[tuple[float, float, float, float]] = None
    source_latitude: Optional[float] = None
    source_longitude: Optional[float] = None
    source_elevation: float = 0.0
    reduction_q: Optional[float] = None
    reduction_wavespeed_kms: Optional[float] = None
    reduction_peak_frequency: Optional[float] = None
    halo_seconds: float = 300.0
    shard_format: str = 'pickle'
    workers: int = 1
    resume: bool = True

    def validate(self):
        if UTCDateTime(self.end) <= UTCDateTime(self.start):
            raise ValueError('end must be later than start (end is exclusive)')
        if self.sampling_interval < 1:
            raise ValueError('sampling_interval must be >= 1 s')
        if self.halo_seconds < 0:
            raise ValueError('halo_seconds must be >= 0')
        if self.workers < 1:
            raise ValueError('workers must be >= 1')
        if self.shard_format not in ('pickle', 'csv', 'parquet'):
            raise ValueError('unsupported shard format')
        if not self.products or any(p not in METRICS for p in self.products):
            raise ValueError(f'products must be selected from {list(METRICS)}')
        if any(p != 'RSAM' for p in self.products) and not self.stationxml:
            raise ValueError('VSAM, DSAM and VSEM require --stationxml for response correction')
        if any(p in REDUCED_PARENTS for p in self.products):
            if self.source_latitude is None or self.source_longitude is None:
                raise ValueError('Reduced metrics require source latitude and longitude')
            if not (-90 <= self.source_latitude <= 90 and -180 <= self.source_longitude <= 180):
                raise ValueError('Invalid source coordinates')
        if self.reduction_q is not None and self.reduction_q <= 0:
            raise ValueError('reduction_q must be positive')
        if self.reduction_wavespeed_kms is not None and self.reduction_wavespeed_kms <= 0:
            raise ValueError('reduction_wavespeed_kms must be positive')
        if self.reduction_peak_frequency is not None and self.reduction_peak_frequency <= 0:
            raise ValueError('reduction_peak_frequency must be positive')
        if self.stationxml and not Path(self.stationxml).exists():
            raise FileNotFoundError(self.stationxml)
        if self.bands:
            for name, band in self.bands.items():
                if len(band) != 2 or not 0 < band[0] < band[1]:
                    raise ValueError(f'invalid band {name}: {band}')
        if self.filter_band is not None and not 0 < self.filter_band[0] < self.filter_band[1]:
            raise ValueError('filter_band must have 0 < low < high')
        if not Path(self.sds_root).is_dir():
            raise NotADirectoryError(self.sds_root)
        # UTC alignment requires integral number of windows per day for unambiguous daily ownership.
        if not np.isclose(86400 / self.sampling_interval, round(86400 / self.sampling_interval)):
            raise ValueError('sampling_interval must divide 86400 s for daily UTC tasks')


def _inventory(path):
    if not path:
        return None
    p = Path(path)
    paths = sorted(p.glob('*.xml')) if p.is_dir() else [p]
    if not paths:
        raise ValueError(f'no StationXML found at {path}')
    inv = read_inventory(str(paths[0]))
    for other in paths[1:]:
        inv += read_inventory(str(other))
    return inv


def _config_hash(config):
    d = asdict(config)
    for key in ('workers', 'resume', 'start', 'end', 'output_root'):
        d.pop(key)
    if config.stationxml:
        p = Path(config.stationxml)
        files = sorted(p.glob('*.xml')) if p.is_dir() else [p]
        d['stationxml_fingerprint'] = [(str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns) for f in files]
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _atomic_frame(df, path, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if fmt == 'pickle':
            df.to_pickle(tmp_path)
        elif fmt == 'csv':
            df.to_csv(tmp_path, index=False)
        else:
            df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _read_frame(path, fmt):
    if fmt == 'pickle':
        return pd.read_pickle(path)
    if fmt == 'csv':
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _merge_stream(stream):
    """Resolve same-NSLC fragments; retain gaps as masks rather than filling zeros."""
    if not stream:
        return Stream()
    st = stream.copy()
    st.sort(keys=['network', 'station', 'location', 'channel', 'starttime'])
    st.merge(method=0, fill_value=None)
    return st


def _calibrated_stream(raw, inventory, product, pre_filt):
    output = 'DISP' if product == 'DSAM' else 'VEL'
    units = 'm' if product == 'DSAM' else 'm/s'
    st = Stream()
    for tr in raw:
        t = tr.copy()
        try:
            # Never silently reinterpret uncorrected counts as physical units.
            t.remove_response(inventory=inventory, output=output,
                              pre_filt=pre_filt, water_level=60,
                              zero_mean=True, taper=True)
        except Exception as exc:
            raise RuntimeError(f'{tr.id}: response removal to {units} failed: {exc}') from exc
        t.stats.units = units
        st.append(t)
    return st


def _days(start, end):
    t = UTCDateTime(UTCDateTime(start).date)
    limit = UTCDateTime(end)
    while t < limit:
        yield str(t)
        t += 86400


def _scheduler_config(config: Config, digest: str) -> ArchiveConfig:
    """Translate the stable archive_metrics API into a generic scheduler job.

    The date range is deliberately part of algorithm_version: generic scheduler
    task IDs are positional, so overlapping jobs must not share checkpoints.
    """
    interval_id = hashlib.sha256(
        (str(UTCDateTime(config.start)) + '/' + str(UTCDateTime(config.end))).encode()
    ).hexdigest()[:12]
    return ArchiveConfig(
        source=SourceSpec(kind='sds', options={'root': config.sds_root}),
        output_root=str(Path(config.output_root) / 'scheduler'),
        start=config.start, end=config.end,
        processor='flovopy.processing.archive_processor:SAMWindowProcessor',
        processor_kwargs={
            'products': list(config.products),
            'sampling_interval': config.sampling_interval,
            'filter_band': config.filter_band,
            'bands': config.bands, 'corners': config.corners,
            'stationxml': config.stationxml, 'pre_filt': config.pre_filt,
            'shard_format': config.shard_format,
            'source': ({'lat': config.source_latitude, 'lon': config.source_longitude,
                        'elev': config.source_elevation} if config.source_latitude is not None else None),
            'reduction_q': config.reduction_q,
            'reduction_wavespeed_kms': config.reduction_wavespeed_kms,
            'reduction_peak_frequency': config.reduction_peak_frequency,
        },
        selectors=Selectors(network=config.network, station=config.station,
                            location=config.location, channel=config.channel),
        window_seconds=86400, halo_seconds=config.halo_seconds,
        workers=config.workers, resume=config.resume,
        algorithm_version=f'archive_metrics_v2:{digest}:{interval_id}',
    )


def consolidate(config: Config, scheduler_summary: dict):
    """Serially consolidate scheduler task shards into the existing SAM format.

    Consolidation reads only validated completion markers and uses the trace-ID
    mapping in their metadata. No worker writes shared yearly output files.
    """
    run_dir = Path(scheduler_summary['output_dir'])
    by_product = {p: {} for p in config.products}
    scheduler_config = _scheduler_config(config, _config_hash(config))
    for index, start, end in _windows(scheduler_config):
        task_dir = run_dir / 'windows' / f'{index:09d}'
        marker = task_dir / 'complete.json'
        if not marker.is_file():
            raise RuntimeError(f'Missing completed task {marker}')
        record = json.loads(marker.read_text())
        if record.get('start') != str(start) or record.get('end') != str(end):
            raise RuntimeError(f'Task interval mismatch at {marker}')
        metadata = record.get('metadata', {})
        for entry in metadata.get('outputs', []):
            path = task_dir / entry['filename']
            if not path.is_file():
                raise RuntimeError(f'Missing metric shard: {path}')
            df = _read_frame(path, config.shard_format)
            by_product[entry['product']].setdefault(entry['trace_id'], []).append(df)
    for product, per_trace in by_product.items():
        for trace_id, frames in per_trace.items():
            df = pd.concat(frames, ignore_index=True).sort_values('time')
            df = df.drop_duplicates(subset='time', keep='last').reset_index(drop=True)
            METRICS[product](dataframes={trace_id: df}).write(
                config.output_root, ext='pickle', overwrite=False)
    return {p: len(d) for p, d in by_product.items()}


def run(config: Config, *, consolidate_yearly=True):
    """Run SDS metrics using the shared source-independent scheduler."""
    config.validate()
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    digest = _config_hash(config)
    owner = output_root / '.archive_metrics_owner.json'
    if owner.exists():
        existing = json.loads(owner.read_text())
        if existing.get('digest') != digest:
            raise ValueError('Output root belongs to a different metric configuration; '
                             'use a new --output-root.')
    else:
        owner.write_text(json.dumps({'digest': digest}, indent=2))
    manifest = output_root / f'config_{digest}.json'
    manifest.write_text(json.dumps({'digest': digest, 'config': asdict(config),
                                    'created_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    result = run_scheduler(_scheduler_config(config, digest))
    summary = {'digest': digest, 'days': result['windows'],
               'processed': result['processed'], 'skipped': result['skipped'],
               'scheduler_run_id': result['run_id'],
               'scheduler_output_dir': result['output_dir']}
    if consolidate_yearly:
        summary['trace_ids_by_product'] = consolidate(config, result)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description='Compute RSAM/VSAM/DSAM/VSEM from an SDS archive')
    p.add_argument('--sds-root', '--sds_root', required=True)
    p.add_argument('--output-root', '--sam_root', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True, help='Exclusive end (UTC)')
    p.add_argument('--products', nargs='+', choices=list(METRICS), default=['RSAM'])
    for key in ('network', 'station', 'location', 'channel'):
        p.add_argument('--' + key, default='*')
    p.add_argument('--sampling-interval', '--sampling_interval', type=float, default=60.)
    p.add_argument('--filter', nargs=2, type=float, metavar=('LOW', 'HIGH'), default=[0.5, 18.])
    p.add_argument('--no-filter', action='store_true')
    p.add_argument('--bands', help='Inline JSON or JSON file; use {} for no bands')
    p.add_argument('--bands-preset', choices=sorted(BAND_PRESETS))
    p.add_argument('--corners', type=int, default=4)
    p.add_argument('--stationxml')
    p.add_argument('--source-latitude', type=float)
    p.add_argument('--source-longitude', type=float)
    p.add_argument('--source-elevation', type=float, default=0.)
    p.add_argument('--reduction-q', type=float)
    p.add_argument('--reduction-wavespeed-kms', type=float)
    p.add_argument('--reduction-peak-frequency', type=float)
    p.add_argument('--pre-filt', nargs=4, type=float, help='Response-removal prefilter corners in Hz')
    p.add_argument('--halo-seconds', type=float, default=300.)
    p.add_argument('--workers', '--nprocs', type=int, default=1)
    p.add_argument('--shard-format', choices=['pickle', 'csv', 'parquet'], default='pickle')
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--no-consolidate', action='store_true')
    args = p.parse_args(argv)
    if args.bands is not None:
        bands = json.loads(Path(args.bands).read_text() if Path(args.bands).is_file() else args.bands)
    elif args.bands_preset:
        bands = BAND_PRESETS[args.bands_preset]
    else:
        bands = None  # SAM's established default bands
    config = Config(sds_root=str(Path(args.sds_root).expanduser().resolve()),
                    output_root=str(Path(args.output_root).expanduser().resolve()),
                    start=args.start, end=args.end, products=tuple(args.products),
                    network=args.network, station=args.station, location=args.location, channel=args.channel,
                    sampling_interval=args.sampling_interval,
                    filter_band=None if args.no_filter else tuple(args.filter), bands=bands,
                    corners=args.corners, stationxml=args.stationxml,
                    source_latitude=args.source_latitude, source_longitude=args.source_longitude,
                    source_elevation=args.source_elevation, reduction_q=args.reduction_q,
                    reduction_wavespeed_kms=args.reduction_wavespeed_kms,
                    reduction_peak_frequency=args.reduction_peak_frequency,
                    pre_filt=tuple(args.pre_filt) if args.pre_filt else None,
                    halo_seconds=args.halo_seconds, workers=args.workers,
                    shard_format=args.shard_format, resume=not args.no_resume)
    print(json.dumps(run(config, consolidate_yearly=not args.no_consolidate), indent=2))


if __name__ == '__main__':
    main()
