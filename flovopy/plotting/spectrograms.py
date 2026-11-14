"""UTC-aligned multichannel waveform/spectrogram plots (IceWeb layout).

Rendering is independent of numerical spectral estimation. No input mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from obspy import Stream
from flovopy.processing.spectrograms import (
    SpectrogramResult, compute_stream_spectrograms,
)


def _edges(centres, fallback_width):
    centres = np.asarray(centres, dtype=float)
    if len(centres) == 0:
        return np.empty(0)
    if len(centres) == 1:
        width = float(fallback_width)
        return np.array([centres[0] - width / 2, centres[0] + width / 2])
    mid = (centres[:-1] + centres[1:]) / 2
    return np.r_[centres[0] - (mid[0] - centres[0]), mid,
                 centres[-1] + (centres[-1] - mid[-1])]


def _datetime_x(unix_seconds):
    # Matplotlib's floating-day representation; UTC is explicit.
    return mdates.epoch2num(unix_seconds) if hasattr(mdates, 'epoch2num') else np.asarray(unix_seconds) / 86400.0 + mdates.date2num(__import__('datetime').datetime(1970, 1, 1, tzinfo=__import__('datetime').timezone.utc))


def plot_waveform_spectrograms(
    stream: Stream,
    spectrograms: Mapping[str, SpectrogramResult] | None = None,
    *,
    fft_seconds: float = 2.56,
    step_seconds: float | None = None,
    quantity: str = 'psd',
    frequency_limits: tuple[float, float] | None = (0.5, 25.0),
    frequency_scale: str = 'linear',
    color_scale: str = 'individual',
    clim: tuple[float, float] | None = None,
    percentiles: tuple[float, float] = (5.0, 99.0),
    db: bool = True,
    reference: float = 1.0,
    cmap: str = 'viridis',
    channel_order: list[str] | None = None,
    title: str | None = None,
    colorbar: bool = True,
    figsize: tuple[float, float] | None = None,
    outfile: str | None = None,
):
    """Plot one waveform and spectrogram pair per ObsPy trace.

    ``spectrograms`` may contain precomputed results keyed by SEED trace ID.
    If omitted, compute PSD spectrograms from the input Stream. All panels
    share absolute UTC time. Gaps remain masked, never filled with zeros.

    ``db=True`` uses 10 log10(PSD/reference) or 20 log10(ASD/reference).
    ``legacy_magnitude`` is also amplitude-like and uses 20 log10.
    ``clim`` is expressed in *display units* (dB when db=True).
    Returns a Matplotlib Figure. Does not call plt.show() or close the figure.
    """
    if not isinstance(stream, Stream):
        raise TypeError('stream must be an ObsPy Stream')
    if len(stream) == 0:
        raise ValueError('stream must contain at least one trace')
    if frequency_scale not in ('linear', 'log'):
        raise ValueError("frequency_scale must be 'linear' or 'log'")
    if color_scale not in ('individual', 'shared', 'robust'):
        raise ValueError("color_scale must be individual, shared, or robust")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('reference must be positive')
    if frequency_limits is not None:
        lo, hi = frequency_limits
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo < hi):
            raise ValueError('invalid frequency_limits')
        if frequency_scale == 'log' and lo <= 0:
            raise ValueError('logarithmic frequency limits must be positive')
    if not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError('invalid percentiles')
    if clim is not None and not (np.isfinite(clim).all() and clim[0] < clim[1]):
        raise ValueError('clim must be an increasing finite pair')

    traces = list(stream)
    ids = [tr.id for tr in traces]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate trace IDs; merge/select segments before plotting')
    if channel_order is not None:
        if len(set(channel_order)) != len(channel_order):
            raise ValueError('duplicate IDs in channel_order')
        missing = set(channel_order) - set(ids)
        if missing:
            raise ValueError(f'unknown channels: {sorted(missing)}')
        by_id = {tr.id: tr for tr in traces}
        traces = [by_id[key] for key in channel_order]
    if not traces:
        raise ValueError('channel_order selects no traces')
    if spectrograms is None:
        kwargs = {'fft_seconds': fft_seconds, 'quantity': quantity}
        if step_seconds is not None:
            kwargs['step_seconds'] = step_seconds
        spectrograms = compute_stream_spectrograms(Stream(traces), **kwargs)
    for tr in traces:
        if tr.id not in spectrograms:
            raise ValueError(f'missing spectrogram for {tr.id}')
        if not isinstance(spectrograms[tr.id], SpectrogramResult):
            raise TypeError(f'{tr.id}: expected SpectrogramResult')
        if spectrograms[tr.id].trace_id != tr.id:
            raise ValueError(f'{tr.id}: spectrogram trace_id mismatch')

    prepared = []
    for tr in traces:
        result = spectrograms[tr.id]
        freqs = result.frequencies
        select = np.ones(len(freqs), dtype=bool)
        if frequency_limits is not None:
            select &= (freqs >= frequency_limits[0]) & (freqs <= frequency_limits[1])
        if frequency_scale == 'log':
            select &= freqs > 0
        f = freqs[select]
        z = np.asarray(result.values[select], dtype=float).copy()
        z[~np.isfinite(z)] = np.nan
        if db:
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (10 if result.quantity == 'psd' else 20) * np.log10(z / reference)
            z[~np.isfinite(z)] = np.nan
        prepared.append((tr, result, f, z))

    finite = [z[np.isfinite(z)] for _, _, _, z in prepared if np.isfinite(z).any()]
    combined = np.concatenate(finite) if finite else np.array([])
    def limits(values, robust):
        if values.size == 0:
            return (0.0, 1.0)
        low, high = (np.percentile(values, percentiles) if robust
                     else (float(np.min(values)), float(np.max(values))))
        if low == high:
            pad = max(abs(low) * 0.01, 1e-6)
            return (low - pad, high + pad)
        return (float(low), float(high))
    common = (clim if clim is not None else
              limits(combined, color_scale == 'robust'))

    n = len(prepared)
    fig = plt.figure(figsize=figsize or (12, max(3.0, 2.7 * n)), constrained_layout=True)
    grid = fig.add_gridspec(2 * n, 1, height_ratios=[v for _ in range(n) for v in (1, 2)], hspace=0.07)
    axes = []
    for i, (tr, result, f, z) in enumerate(prepared):
        ax_wave = fig.add_subplot(grid[2*i, 0], sharex=axes[0][0] if axes else None)
        ax_spec = fig.add_subplot(grid[2*i+1, 0], sharex=ax_wave)
        axes.append((ax_wave, ax_spec))
        raw = np.ma.asarray(tr.data, dtype=float)
        y = np.asarray(raw.filled(np.nan), dtype=float)
        y[~np.isfinite(y)] = np.nan
        t = float(tr.stats.starttime.timestamp) + np.arange(len(y)) / float(tr.stats.sampling_rate)
        ax_wave.plot(_datetime_x(t), y, lw=0.55, color='black')
        ax_wave.set_ylabel(tr.id, fontsize=8, rotation=0, ha='right', va='center', labelpad=6)
        ax_wave.grid(alpha=0.2)
        ax_wave.tick_params(labelbottom=False)
        norm_limits = (clim if clim is not None else
                       common if color_scale != 'individual' else
                       limits(z[np.isfinite(z)], False))
        if f.size and result.times.size:
            te = _datetime_x(_edges(result.times, result.metadata.get('step_seconds', 1.0)))
            fe = _edges(f, float(np.median(np.diff(f))) if f.size > 1 else 1.0)
            if frequency_scale == 'log':
                fe = np.maximum(fe, np.finfo(float).tiny)
            mesh = ax_spec.pcolormesh(te, fe, np.ma.masked_invalid(z),
                                      shading='flat', cmap=cmap,
                                      norm=Normalize(*norm_limits))
            if colorbar:
                fig.colorbar(mesh, ax=ax_spec, pad=0.005, fraction=0.018,
                             label=f"{result.quantity.upper()}" + (' (dB)' if db else
                             f" ({result.units})" if result.units else ''))
        if frequency_scale == 'log':
            ax_spec.set_yscale('log')
        if frequency_limits is not None:
            ax_spec.set_ylim(*frequency_limits)
        ax_spec.set_ylabel('Hz')
        ax_spec.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz=__import__('datetime').timezone.utc))
        if i != n-1:
            ax_spec.tick_params(labelbottom=False)
    axes[-1][1].set_xlabel('Time (UTC)')
    if title:
        fig.suptitle(title)
    fig._flovopy_axes = axes  # convenient access; not part of stable public API
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
    return fig


# Precomputed legacy band data: deliberately independent of the waveform/UTC API.
def plot_band_spectrograms(results, *, spectral_bands=None, broad_bands=None,
                          start=None, end=None, color_scale='individual', clim=None,
                          log_colors=False, cmap='viridis', title=None, figsize=None,
                          outfile=None):
    """IceWeb-style rows for precomputed, possibly overlapping band amplitudes.

    Select contiguous nonoverlapping spectral_bands for the Hz heatmap; overlapping
    broad_bands are line panels with explicit frequency labels. No FFT, density
    scaling, dB conversion, interpolation or UTC assumption. start/end select
    [start, end) recorded labels. Missing records remain blank. Positive-only
    logarithmic color normalization changes display, not measurement units.
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm
    results=list(results)
    if not results: raise ValueError('No band spectrograms supplied')
    ids=[r.trace_id for r in results]
    if color_scale=='shared' and len({(r.units,r.quantity) for r in results})!=1:
        raise ValueError('Shared color scales require matching quantity and units')
    if len(set(ids))!=len(ids): raise ValueError('Duplicate channel IDs; keep source versions separate')
    if len({r.time_basis for r in results})!=1:
        raise ValueError('Mixed clock conventions; plot separately or reconcile explicitly')
    if color_scale not in ('individual','shared'): raise ValueError('color_scale must be individual or shared')
    start=np.datetime64(start,'ns') if start is not None else None
    end=np.datetime64(end,'ns') if end is not None else None
    if (start is not None and np.isnat(start)) or (end is not None and np.isnat(end)):
        raise ValueError('Invalid date limit')
    if start is not None and end is not None and end<=start: raise ValueError('end must exceed start')
    prepared=[]
    for r in results:
        names=list(spectral_bands) if spectral_bands is not None else list(r.band_names)
        broad=list(broad_bands or [])
        if not names or len(set(names))!=len(names) or len(set(broad))!=len(broad):
            raise ValueError('Band selections must be nonempty/unique')
        try: indexes=[r.band_names.index(n) for n in names];bi=[r.band_names.index(n) for n in broad]
        except ValueError as exc: raise ValueError(f'Unknown band in {r.trace_id}') from exc
        bounds=r.band_limits_hz[indexes]
        if len(bounds)>1 and not np.allclose(bounds[:-1,1],bounds[1:,0],rtol=0,atol=1e-10):
            raise ValueError('Heatmap bands must be ordered, contiguous and nonoverlapping; plot other bands separately')
        keep=np.ones(len(r.times),dtype=bool)
        if start is not None: keep &= r.times>=start
        if end is not None: keep &= r.times<end
        times=r.times[keep];values=r.values[:,keep]
        # Never let nominal support rectangles overwrite adjacent observations.
        duration=float(r.interval_seconds)
        if len(times)>1 and (np.diff(times).astype('timedelta64[ns]').astype(float)/1e9 < duration-1e-9).any():
            raise ValueError('Display intervals overlap; choose a shorter support duration explicitly')
        prepared.append((r,indexes,bi,bounds,times,values))
    if not any(len(item[4]) for item in prepared): raise ValueError('No records in selected window')
    def norm_for(values):
        values=np.asarray(values);valid=values[np.isfinite(values)]
        if log_colors:valid=valid[valid>0]
        if clim is not None:
            if len(clim)!=2 or not np.isfinite(clim).all() or clim[1]<=clim[0] or (log_colors and clim[0]<=0):
                raise ValueError('Invalid color limits')
            lo,hi=clim
        elif len(valid):
            lo,hi=float(valid.min()),float(valid.max())
            if hi==lo:
                if log_colors:lo,hi=lo/1.01,hi*1.01
                else:lo,hi=lo-.5,hi+.5
        else:lo,hi=(1.,10.) if log_colors else (0.,1.)
        return LogNorm(lo,hi) if log_colors else Normalize(lo,hi)
    shared=None
    if color_scale=='shared':
        shared=norm_for(np.concatenate([v[ix,:].ravel() for _,ix,_,_,_,v in prepared]))
    with_broad=bool(broad_bands)
    fig,axes=plt.subplots(len(results),2 if with_broad else 1,squeeze=False,
        figsize=figsize or (14 if with_broad else 11,max(3,3*len(results))),sharex=True)
    for row,(r,indexes,bi,bounds,times,values) in enumerate(prepared):
        ax=axes[row,0];vertices=[];colors=[]
        duration=r.interval_seconds/86400
        shift={'start':0,'center':-.5,'end':-1}[r.time_anchor]*duration
        xs=mdates.date2num(times)+shift
        for j,x in enumerate(xs):
            for k,index in enumerate(indexes):
                value=values[index,j]
                if not np.isfinite(value) or (log_colors and value<=0):continue
                lo,hi=bounds[k]
                vertices.append([(x,lo),(x+duration,lo),(x+duration,hi),(x,hi)]);colors.append(value)
        norm=shared if shared is not None else norm_for(values[indexes,:])
        collection=PolyCollection(vertices,array=np.asarray(colors),norm=norm,cmap=cmap,edgecolors='none')
        ax.add_collection(collection);ax.autoscale_view();ax.set_ylim(bounds[0,0],bounds[-1,1]);ax.xaxis_date()
        ax.set(ylabel='Frequency (Hz)',title=str(r.trace_id)+' — SSAM bands')
        fig.colorbar(collection,ax=ax,label=r.units)
        if with_broad:
            lineax=axes[row,1]
            # Break lines at gaps, retaining the original samples unchanged.
            for index in bi:
                tx=[];vy=[]
                for j,t in enumerate(times):
                    if j and (times[j]-times[j-1])/np.timedelta64(1,'s')>r.interval_seconds*1.01:
                        tx.append(t);vy.append(np.nan)
                    tx.append(t);vy.append(values[index,j])
                lo,hi=r.band_limits_hz[index]
                lineax.plot(np.asarray(tx,dtype='datetime64[ns]'),vy,'.-',linewidth=.7,markersize=2,
                            label=f'{r.band_names[index]}: {lo:g}–{hi:g} Hz')
            lineax.set(ylabel=r.units,title=str(r.trace_id)+' — broad bands')
            lineax.legend(fontsize='small',ncol=2);lineax.grid(alpha=.2)
        for axis in axes[row]:
            locator=mdates.AutoDateLocator(minticks=3,maxticks=6)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            axis.set_xlabel('Recorded calendar time (no clock correction)')
    if start is not None and end is not None:axes[0,0].set_xlim(mdates.date2num(start),mdates.date2num(end))
    elif start is not None:axes[0,0].set_xlim(left=mdates.date2num(start))
    elif end is not None:axes[0,0].set_xlim(right=mdates.date2num(end))
    fig.suptitle(title or 'SSAM band amplitudes — recorded calendar labels')
    fig._flovopy_band_axes=axes
    fig.tight_layout()
    if outfile is not None:fig.savefig(outfile,dpi=180)
    return fig


def plot_daily_band_spectrograms(results, output_dir, *, page='day', start=None,
                                 end=None, overwrite=False, **plot_kwargs):
    """Write one multi-channel plot per nonempty calendar day or hour.

    Returns generated paths. end is exclusive. Creates no zero-filled pages.
    Files are preflighted before writing; existing products require overwrite=True.
    A manifest retains band definitions, source metadata and unresolved clock policy.
    """
    from pathlib import Path
    import json
    results=list(results)
    if not results: raise ValueError('No band spectrograms supplied')
    if page not in ('day','hour'):raise ValueError('page must be day or hour')
    if 'outfile' in plot_kwargs:raise ValueError('Batch writer manages outfile')
    unit='D' if page=='day' else 'h'
    start=np.datetime64(start,'ns') if start is not None else None
    end=np.datetime64(end,'ns') if end is not None else None
    if (start is not None and np.isnat(start)) or (end is not None and np.isnat(end)):
        raise ValueError('Invalid date limit')
    if start is not None and end is not None and end<=start:raise ValueError('end must exceed start')
    pages=set()
    for r in results:
        times=r.times
        if start is not None:times=times[times>=start]
        if end is not None:times=times[times<end]
        pages.update(times.astype(f'datetime64[{unit}]').astype('int64').tolist())
    if not pages:raise ValueError('No records in selected window')
    output=Path(output_dir)
    plan=[]
    for value in sorted(pages):
        lo=np.datetime64(value,unit).astype('datetime64[ns]')
        hi=lo+np.timedelta64(1,unit)
        label=np.datetime_as_string(lo,unit=unit).replace(':','').replace('-','')
        plan.append((max(lo,start) if start is not None else lo,
                     min(hi,end) if end is not None else hi,output/f'ssam_{page}_{label}.png'))
    manifest_path=output/'ssam_pages.json'
    if not overwrite and any(p.exists() for p in [manifest_path,*[p for _,_,p in plan]]):
        raise FileExistsError('Spectrogram page/manifest exists; choose another directory or overwrite=True')
    output.mkdir(parents=True,exist_ok=True)
    generated=[]
    for lo,hi,path in plan:
        fig=plot_band_spectrograms(results,start=lo,end=hi,**plot_kwargs)
        try:fig.savefig(path,dpi=180)
        finally:plt.close(fig)
        generated.append(path)
    manifest=dict(page=page,paths=[str(p) for p in generated],start=str(start),end_exclusive=str(end),
        plot_options=plot_kwargs,channels=[dict(trace_id=r.trace_id,time_basis=r.time_basis,
        units=r.units,quantity=r.quantity,time_anchor=r.time_anchor,display_interval_seconds=r.interval_seconds,
        band_names=r.band_names,band_limits_hz=r.band_limits_hz.tolist(),metadata=r.metadata) for r in results])
    manifest_path.write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    return generated
