"""Noise-prefix proposals and onset evidence for already recorded event waveforms.

Public entry points are also exposed through :mod:`flovopy.processing.detection`.
No file access, waveform mutation, instrument correction or clock correction is
performed. All intervals are proposals requiring review, not acquisition triggers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from obspy import Trace, Stream
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import periodogram, detrend, hilbert

ALGORITHM_VERSION = 'pretrigger-windows-1'
__all__ = ['PretriggerSettings', 'detect_pretrigger_window']

@dataclass(frozen=True)
class PretriggerSettings:
    """Heuristic parameters for noise-prefix and onset proposals.

    Reference must start at zero and cover ``minimum_noise_s``. Its duration
    is an assumption, not a learned acquisition setting. Station support may
    be one only when explicitly configured. ``plot_seconds`` is a diagnostic
    display preference; it does not restrict the analysis interval.
    """
    reference_start_s: float = 0.0
    reference_end_s: float = 5.0
    minimum_noise_s: float = 5.0
    noise_guard_s: float = 1.0
    frequency_window_s: float = 1.0
    frequency_step_s: float = 0.25
    frequency_low_hz: float = 0.8
    frequency_high_hz: float = 20.0
    frequency_shift_hz: float = 2.0
    spectral_divergence_floor: float = 0.15
    hilbert_edge_s: float = 1.0
    hilbert_envelope_floor_fraction: float = 0.1
    hilbert_smooth_s: float = 0.25
    refine_amplitude_factor: float = 1.5
    refine_mad_factor: float = 3.0
    sta_seed_amplitude_factor: float = 1.3
    network_min_stations: int = 2
    network_coincidence_s: float = 3.0
    network_lead_s: float = 2.0
    network_lag_s: float = 8.0
    reference_peak_factor: float = 10.0
    transient_peak_factor: float = 4.0
    detector_agreement_s: float = 1.0
    fine_window_s: float = 0.25
    amplitude_factor: float = 3.0
    mad_factor: float = 6.0
    sustain_s: float = 0.5
    sta_s: float = 0.5
    lta_s: float = 3.0
    sta_lta_threshold: float = 3.0
    min_coverage: float = 0.95
    plot_seconds: float = 30.0  # diagnostic display preference, not a detection parameter
    compute_frequency: bool = True
    compute_hilbert: bool = True
    def __post_init__(self):
        values=asdict(self)
        if any(type(values[k]) is not bool for k in ('compute_frequency','compute_hilbert')):raise ValueError('Diagnostic switches must be boolean')
        values={k:v for k,v in values.items() if k not in ('compute_frequency','compute_hilbert')}
        if not all(math.isfinite(v) for v in values.values()):raise ValueError('All settings must be finite')
        if not 0 <= self.reference_start_s < self.reference_end_s:raise ValueError('Invalid reference interval')
        if self.minimum_noise_s <= 0:raise ValueError('Minimum assumed noise duration must be positive')
        if self.reference_end_s < self.minimum_noise_s or self.reference_start_s != 0:raise ValueError('Reference must start at zero and cover the minimum noise interval')
        if not 0 < self.frequency_low_hz < self.frequency_high_hz:raise ValueError('Invalid frequency band')
        if not 0 < self.sta_s < self.lta_s:raise ValueError('Require 0 < STA < LTA')
        if not 0 < self.min_coverage <= 1:raise ValueError('Invalid coverage')
        if not 0 < self.hilbert_envelope_floor_fraction <= 1:raise ValueError('Hilbert envelope fraction must be in (0, 1]')
        if self.hilbert_edge_s<0 or self.hilbert_smooth_s<0:raise ValueError('Hilbert edge and smoothing durations must be nonnegative')
        if type(self.network_min_stations) is not int or self.network_min_stations<1:raise ValueError('Station support must be a positive integer')
        for key in values:
            if key not in ('reference_start_s','hilbert_edge_s','hilbert_smooth_s') and values[key]<=0:raise ValueError(key+' must be positive')


def _runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def _prefix(x):
    if not len(x) or not np.isfinite(x[0]):return 0
    different=np.flatnonzero(~np.isfinite(x)|(x!=x[0]))
    return int(different[0]) if len(different) else len(x)


def _bins(x,fs,seconds,minimum):
    n=int(round(fs*seconds))
    if n<1:raise ValueError('Metric window must span at least one sample')
    rows=[]
    for start in range(0,len(x),n):
        block=x[start:start+n];valid=block[np.isfinite(block)];coverage=len(valid)/n
        good=coverage>=minimum and len(block)==n
        rows.append(dict(window_start_s=start/fs,window_end_s=(start+len(block))/fs,
            centre_s=(start+len(block)/2)/fs,coverage=coverage,complete=len(block)==n,
            rsam=float(np.mean(np.abs(valid))) if good else np.nan,
            rms=float(np.sqrt(np.mean(valid**2))) if good else np.nan))
    return pd.DataFrame(rows,columns=['window_start_s','window_end_s','centre_s','coverage','complete','rsam','rms'])


def _sustained(times,values,threshold,after,required):
    mask=np.isfinite(values)&(values>=threshold)&(times>=after)
    for a,b in _runs(mask):
        if b-a>=required:return float(times[a])
    return np.nan


def frequency_checks(x, fs, cfg):
    """Trailing-window PSD descriptors and raw detrended zero-crossing proxy.

    JSD compares normalized power within the selected band. Centroids are Hz,
    JSD is natural-log units. Timing is reported at window END (availability),
    not as an exact onset. Overlapping windows are correlated. No zero-padding,
    endpoint padding, time-domain bandpass or Hilbert phase differentiation.
    """
    if len(x)==0:
        return pd.DataFrame(columns=['window_start_s','window_end_s','centroid_hz','zero_crossing_hz','js_divergence']),dict(
            frequency_change_trace_s=np.nan,frequency_support_start_trace_s=np.nan,
            centroid_baseline_hz=np.nan,frequency_shift_threshold_hz=np.nan,spectral_divergence_threshold=np.nan)
    n=int(round(cfg.frequency_window_s*fs));step=int(round(cfg.frequency_step_s*fs))
    if n<4 or step<1:
        raise ValueError('Frequency window needs at least four samples and step at least one')
    rows=[];powers=[]
    for a in range(0,len(x)-n+1,step):
        segment=x[a:a+n];row=dict(window_start_s=a/fs,window_end_s=(a+n)/fs,
            centroid_hz=np.nan,zero_crossing_hz=np.nan,js_divergence=np.nan)
        if np.all(np.isfinite(segment)):
            y=detrend(segment,type='linear')
            nonzero=y[y!=0]  # exact zero runs are not extra crossings
            row['zero_crossing_hz']=float(np.count_nonzero(np.diff(np.signbit(nonzero)))/(2*(n-1)/fs)) if len(nonzero)>1 else np.nan
            f,power=periodogram(segment,fs=fs,window='hann',detrend='linear',scaling='density')
            mask=(f>=cfg.frequency_low_hz)&(f<=min(cfg.frequency_high_hz,fs/2))
            if mask.sum()>=3 and power[mask].sum()>0:
                power=power[mask]/power[mask].sum();row['centroid_hz']=float(np.sum(f[mask]*power));powers.append(power)
            else:powers.append(None)
        else:powers.append(None)
        rows.append(row)
    frame=pd.DataFrame(rows,columns=['window_start_s','window_end_s','centroid_hz','zero_crossing_hz','js_divergence'])
    reference=[v for r,v in zip(rows,powers) if r['window_end_s']<=cfg.reference_end_s and v is not None]
    if not reference:return frame,dict(frequency_change_trace_s=np.nan,frequency_support_start_trace_s=np.nan,centroid_baseline_hz=np.nan,frequency_shift_threshold_hz=np.nan,spectral_divergence_threshold=np.nan)
    baseline=np.mean(reference,axis=0);baseline/=baseline.sum()
    def js(a,b):
        m=(a+b)/2
        def kl(v):
            good=v>0;return float(np.sum(v[good]*np.log(v[good]/m[good])))
        return .5*(kl(a)+kl(b))
    frame['js_divergence']=[js(v,baseline) if v is not None else np.nan for v in powers]
    ref=frame[frame.window_end_s<=cfg.reference_end_s]
    centre=float(ref.centroid_hz.median());spread=float((ref.centroid_hz-centre).abs().median())
    jcentre=float(ref.js_divergence.median());jspread=float((ref.js_divergence-jcentre).abs().median())
    ct=max(cfg.frequency_shift_hz,cfg.mad_factor*1.4826*spread)
    jt=max(cfg.spectral_divergence_floor,jcentre+cfg.mad_factor*1.4826*jspread)
    frame['centroid_shift_hz']=(frame.centroid_hz-centre).abs()
    changed=((frame.centroid_shift_hz>=ct)&(frame.js_divergence>=jt)).to_numpy()
    # Require fully post-reference windows so reference spectra are not reused as events.
    changed = changed & (frame.window_start_s>=cfg.reference_end_s).to_numpy()
    change=_sustained(frame.window_end_s.to_numpy(),changed.astype(float),1.,cfg.reference_end_s,
        max(1,int(np.ceil(cfg.sustain_s/(step/fs)))))
    return frame,dict(frequency_change_trace_s=change,
        frequency_support_start_trace_s=change-n/fs if np.isfinite(change) else np.nan,
        centroid_baseline_hz=centre,frequency_shift_threshold_hz=ct,spectral_divergence_threshold=jt)


def hilbert_diagnostic(x, fs, cfg):
    """Hilbert analytic phase and first-difference IF on finite contiguous runs.

    Each run is linearly detrended; no bandpass, interpolation or gap filling.
    IF time lies halfway between the two samples used by its phase difference.
    FFT Hilbert is OFFLINE/nonlocal. One-second edge exclusions are heuristic,
    not a guarantee of freedom from end effects. Signed raw IF is retained.
    """
    if not np.isfinite(fs) or fs<=0:raise ValueError('Invalid sampling rate')
    data=np.array(np.ma.asarray(x,dtype=float).filled(np.nan),copy=True)
    columns=['time_trace_s','envelope_counts','phase_midpoint_rad','instantaneous_hz_raw',
             'instantaneous_hz_valid','instantaneous_hz_smoothed',
             'edge_excluded','low_envelope','nonpositive_or_nonfinite_frequency','envelope_floor_counts']
    pieces=[]
    for a,b in _runs(np.isfinite(data)):
        if b-a<4:continue
        y=detrend(data[a:b],type='linear');z=hilbert(y)
        phase=np.unwrap(np.angle(z));envelope=np.abs(z)
        times=(np.arange(a,b-1)+.5)/fs
        raw=np.diff(phase)*fs/(2*np.pi)
        mid_envelope=(envelope[:-1]+envelope[1:])/2
        ref=mid_envelope[(times>=cfg.reference_start_s)&(times<cfg.reference_end_s)]
        ref=ref[np.isfinite(ref)&(ref>0)]
        scale=float(np.median(ref)) if len(ref) else float(np.median(mid_envelope))
        floor=cfg.hilbert_envelope_floor_fraction*scale
        # Requiring BOTH endpoints avoids trusting a derivative across a node.
        low=(np.minimum(envelope[:-1],envelope[1:])<=max(floor,np.finfo(float).tiny))
        edge=((times-a/fs)<cfg.hilbert_edge_s)|(((b-1)/fs-times)<cfg.hilbert_edge_s)
        bad=(raw<=0)|~np.isfinite(raw)
        valid=~(low|edge|~np.isfinite(raw))  # signed estimates remain in summaries
        measured=np.where(valid,raw,np.nan);smooth=measured.copy()
        width=max(1,int(round(cfg.hilbert_smooth_s*fs)))
        if width%2==0:width+=1
        if cfg.hilbert_smooth_s>0 and width>1:
            smooth=pd.Series(measured).rolling(width,center=True,
                min_periods=max(1,int(np.ceil(width*.5)))).median().to_numpy(copy=True)
            smooth[~valid]=np.nan  # no estimate at an unreliable centre; no gap crossing
        frame=pd.DataFrame(dict(time_trace_s=times,envelope_counts=mid_envelope,
            phase_midpoint_rad=(phase[:-1]+phase[1:])/2,instantaneous_hz_raw=raw,
            instantaneous_hz_valid=measured,instantaneous_hz_smoothed=smooth,
            edge_excluded=edge,low_envelope=low,nonpositive_or_nonfinite_frequency=bad,
            envelope_floor_counts=floor))
        pieces.append(frame)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame({c:pd.Series(dtype=bool if c in ['edge_excluded','low_envelope','nonpositive_or_nonfinite_frequency'] else float) for c in columns})


def _summarize_hilbert(frequency,samples,fs):
    """Aggregate only reliability-screened IF in existing trailing PSD windows."""
    records=[]
    for row in frequency.itertuples():
        block=samples[(samples.time_trace_s>=row.window_start_s)&(samples.time_trace_s<row.window_end_s)]
        values=block.instantaneous_hz_valid.dropna().to_numpy(dtype=float)
        records.append(dict(hilbert_median_hz=float(np.median(values)) if len(values) else np.nan,
            hilbert_q10_hz=float(np.quantile(values,.1)) if len(values) else np.nan,
            hilbert_q90_hz=float(np.quantile(values,.9)) if len(values) else np.nan,
            hilbert_valid_fraction=len(values)/max(1,int(round((row.window_end_s-row.window_start_s)*fs))),
            hilbert_negative_fraction=float((block.instantaneous_hz_raw<0).mean()) if len(block) else np.nan))
    out=frequency.copy()
    for key in ['hilbert_median_hz','hilbert_q10_hz','hilbert_q90_hz','hilbert_valid_fraction','hilbert_negative_fraction']:
        out[key]=[r[key] for r in records]
    return out


def _analyse_trace(trace, settings=None):
    cfg=settings or PretriggerSettings();fs=float(trace.stats.sampling_rate)
    if not math.isfinite(fs) or fs<=0:raise ValueError('Invalid sampling rate')
    raw=np.array(np.ma.asarray(trace.data,dtype=float).filled(np.nan),dtype=float,copy=True);raw[~np.isfinite(raw)]=np.nan
    t=np.arange(len(raw))/fs
    reference=raw[(t>=cfg.reference_start_s)&(t<cfg.reference_end_s)]
    expected=(cfg.reference_end_s-cfg.reference_start_s)*fs
    reference_ok=np.count_nonzero(np.isfinite(reference))/expected>=cfg.min_coverage
    dc=float(np.nanmedian(reference)) if np.any(np.isfinite(reference)) else 0.
    x=raw-dc;one=_bins(x,fs,1.,cfg.min_coverage);fine=_bins(x,fs,cfg.fine_window_s,cfg.min_coverage)
    ref=fine[(fine.window_start_s>=cfg.reference_start_s)&(fine.window_end_s<=cfg.reference_end_s)]
    level=float(ref.rsam.median()) if len(ref) else np.nan
    mad=float(np.nanmedian(np.abs(ref.rsam-level))) if np.isfinite(level) else np.nan
    threshold=max(cfg.amplitude_factor*level,level+cfg.mad_factor*1.4826*mad) if reference_ok and level>0 else np.nan
    onset=_sustained(fine.window_start_s.to_numpy(),fine.rsam.to_numpy(),threshold,cfg.reference_end_s,
        max(1,int(np.ceil(cfg.sustain_s/(round(cfg.fine_window_s*fs)/fs))))) if np.isfinite(threshold) else np.nan
    fine_width=round(cfg.fine_window_s*fs)/fs
    cft=np.full(len(raw),np.nan);nsta=max(1,int(round(cfg.sta_s*fs)));nlta=max(nsta+1,int(round(cfg.lta_s*fs)))
    for a,b in _runs(np.isfinite(x)):
        if b-a>=nlta:
            v=classic_sta_lta(x[a:b],nsta,nlta);v[:nlta-1]=np.nan;cft[a:b]=v
    sta_onset=_sustained(t,cft,cfg.sta_lta_threshold,max(cfg.reference_end_s,cfg.lta_s),max(1,int(np.ceil(cfg.sustain_s*fs))))
    finite=raw[np.isfinite(raw)];flat_prefix=_prefix(raw)/fs;flat_suffix=_prefix(raw[::-1])/fs
    extreme_run=0;extreme_fraction=np.nan
    if len(finite) and np.ptp(finite)>0:
        extremes=np.isfinite(raw)&((raw==finite.min())|(raw==finite.max()))
        extreme_run=max((b-a for a,b in _runs(extremes)),default=0);extreme_fraction=float(extremes.sum()/len(finite))
    flags=[]
    if not reference_ok:flags.append('reference_incomplete')
    if not np.isfinite(level) or level<=0:flags.append('reference_zero_or_unusable')
    if np.isfinite(raw).sum()!=len(raw):flags.append('masked_or_nonfinite_samples')
    if flat_prefix>=0.5:flags.append('leading_flat_samples_possible_padding')
    if flat_suffix>=0.5:flags.append('trailing_flat_samples_possible_padding')
    if len(finite) and np.ptp(finite)==0:flags.append('constant_trace')
    if extreme_run>=3:flags.append('repeated_extreme_values_possible_clipping')
    frequency,fq=frequency_checks(x if cfg.compute_frequency else np.array([]),fs,cfg)
    hilbert_samples=hilbert_diagnostic(x if cfg.compute_hilbert else np.array([]),fs,cfg)
    frequency=_summarize_hilbert(frequency,hilbert_samples,fs)
    noise_end=onset-cfg.noise_guard_s if np.isfinite(onset) else cfg.minimum_noise_s
    proposal_status='extended_from_amplitude_candidate' if np.isfinite(onset) else 'minimum_only_no_amplitude_candidate'
    if noise_end<cfg.minimum_noise_s:
        noise_end=np.nan;proposal_status='CONFLICT_minimum_vs_guard';flags.append('amplitude_change_too_early_for_noise_rule')
    if np.isfinite(noise_end):
        noise=x[t<noise_end]
        if len(noise)<int(round(cfg.minimum_noise_s*fs)) or np.isfinite(noise).mean()<cfg.min_coverage or flat_prefix>=.5:
            proposal_status='QC_REVIEW_REQUIRED';flags.append('proposed_noise_has_invalid_or_flat_samples')
        if np.isfinite(fq['frequency_change_trace_s']) and fq['frequency_support_start_trace_s']<noise_end:
            flags.append('spectral_change_may_overlap_proposed_noise')
    agreement=bool(np.isfinite(onset) and np.isfinite(sta_onset) and abs(onset-sta_onset)<=1.)
    qc=dict(fine_window_actual_s=fine_width,frequency_window_actual_s=round(cfg.frequency_window_s*fs)/fs,
        frequency_step_actual_s=round(cfg.frequency_step_s*fs)/fs,trace_id=trace.id,npts=len(raw),sampling_rate_hz=fs,duration_s=len(raw)/fs,
        recorded_starttime=str(trace.stats.starttime),valid_fraction=len(finite)/len(raw) if len(raw) else 0,
        raw_min=float(finite.min()) if len(finite) else np.nan,raw_max=float(finite.max()) if len(finite) else np.nan,
        flat_prefix_s=flat_prefix,flat_suffix_s=flat_suffix,longest_extreme_run_samples=extreme_run,
        extreme_fraction=extreme_fraction,reference_median_counts=dc,reference_rsam_counts=level,
        amplitude_threshold_counts=threshold,amplitude_onset_trace_s=onset,sta_lta_onset_trace_s=sta_onset,
        methods_agree_within_1s=agreement,qc_flags=';'.join(flags),
        candidate_status='candidate_review_required' if np.isfinite(onset) or np.isfinite(sta_onset) else 'unresolved_no_sustained_detection',
        hilbert_valid_fraction=float(hilbert_samples.instantaneous_hz_valid.notna().mean()) if len(hilbert_samples) else 0.,
        hilbert_negative_frequency_fraction=float((hilbert_samples.instantaneous_hz_raw<0).mean()) if len(hilbert_samples) else np.nan,
        proposed_noise_start_trace_s=0.,proposed_noise_end_trace_s=noise_end,noise_proposal_status=proposal_status,
        proposed_signal_start_trace_s=onset,proposed_signal_end_trace_s=np.nan,
        reference_status=f'USER_ASSUMPTION_FIRST_{cfg.reference_end_s:g}_SECONDS_NOISE',trigger_status='NOT_MEASURED',**fq)
    return dict(qc=qc,one_second=one,fine=fine,t=t,centred=x,sta_lta=cft,frequency=frequency,hilbert=hilbert_samples,trace=trace)


def _candidate_evidence(record,cfg):
    """Keep all sustained crossings, not just the first high-threshold one."""
    fine=record['fine'];q=record['qc'];level=q['reference_rsam_counts']
    if not np.isfinite(level) or level<=0 or 'reference_incomplete' in q['qc_flags']:return []
    ref=fine[fine.window_end_s<=cfg.reference_end_s].rsam.dropna()
    scatter=1.4826*float((ref-ref.median()).abs().median()) if len(ref) else np.nan
    weak=max(cfg.refine_amplitude_factor*level,level+cfg.refine_mad_factor*scatter)
    records=[]
    for method,values,threshold,times,dt in [
        ('weak_amplitude',fine.rsam.to_numpy(),weak,fine.window_start_s.to_numpy(),q['fine_window_actual_s']),
        ('strong_amplitude',fine.rsam.to_numpy(),q['amplitude_threshold_counts'],fine.window_start_s.to_numpy(),q['fine_window_actual_s']),
        ('sta_lta',record['sta_lta'],cfg.sta_lta_threshold,record['t'],1/q['sampling_rate_hz'])]:
        mask=np.isfinite(values)&(values>=threshold)&(times>=cfg.reference_end_s)
        for a,b in _runs(mask):
            if (b-a)*dt+1e-9<cfg.sustain_s:continue
            when=float(times[a]);local=fine[(fine.window_start_s>=when)&(fine.window_start_s<when+1.)].rsam
            ratio=float(local.median()/level) if len(local) else np.nan
            # STA/LTA spikes require accompanying amplitude evidence for network seeding.
            seed=method=='strong_amplitude' or (method=='sta_lta' and ratio>=cfg.sta_seed_amplitude_factor)
            records.append(dict(method=method,time_trace_s=when,support_end_trace_s=float(times[b-1]+dt),
                seed_eligible=bool(seed),local_amplitude_ratio=ratio,threshold=float(threshold)))
    # Coarser evidence catches an emergent rise diluted in short noisy bins.
    # Use the START of a changed 1-s bin conservatively; it is a support bound,
    # not a precise arrival pick, and cannot seed a network event by itself.
    one=record['one_second'];baseline=one[one.window_end_s<=cfg.reference_end_s].rsam.dropna()
    spread=1.4826*float((baseline-baseline.median()).abs().median()) if len(baseline) else np.inf
    threshold=max(cfg.refine_amplitude_factor*level,float(baseline.median())+cfg.refine_mad_factor*spread) if len(baseline) else np.inf
    for row in one.itertuples():
        if row.window_start_s>=cfg.reference_end_s and row.complete and row.coverage>=cfg.min_coverage and row.rsam>=threshold:
            records.append(dict(method='one_second_amplitude',time_trace_s=float(row.window_start_s),
                support_end_trace_s=float(row.window_end_s),seed_eligible=False,
                local_amplitude_ratio=float(row.rsam/level),threshold=threshold))
    record['weak_amplitude_threshold_counts']=weak
    return sorted(records,key=lambda r:r['time_trace_s'])


def _reference_check(record,cfg):
    q=record['qc'];x=record['centred'];t=record['t'];fs=q['sampling_rate_hz']
    reference=x[(t>=0)&(t<cfg.minimum_noise_s)];ref=reference[np.isfinite(reference)]
    valid=(len(reference)>=int(round(cfg.minimum_noise_s*fs))
        and len(ref)/max(len(reference),1)>=cfg.min_coverage
        and np.isfinite(q['reference_rsam_counts']) and q['reference_rsam_counts']>0
        and q['flat_prefix_s']<.5 and 'reference_incomplete' not in q['qc_flags'])
    flags=[]
    if len(ref):
        centre=float(np.median(np.abs(ref)))
        spread=1.4826*float(np.median(np.abs(np.abs(ref)-centre)))
        limit=max(cfg.reference_peak_factor*centre,centre+cfg.reference_peak_factor*spread)
        if np.max(np.abs(ref))>limit:
            valid=False;flags.append('initial_reference_contains_large_transient')
    return bool(valid),ref,flags


def _network_context(records,cfg):
    """Earliest supported cluster, counting DISTINCT network/station identities.

    This locates the first candidate event in a file, not an acquisition trigger.
    Two components at one station cannot establish network coincidence.
    """
    seeds=[]
    for record in records:
        q=record['qc'];station=(record['trace'].stats.network,record['trace'].stats.station)
        if not _reference_check(record,cfg)[0]:continue
        if any(flag in q['qc_flags'] for flag in ['reference_incomplete','reference_zero_or_unusable','constant_trace','leading_flat_samples_possible_padding']):continue
        for c in record['candidates']:
            if c['seed_eligible']:seeds.append((c['time_trace_s']+record['record_offset_s'],station))
    seeds.sort(key=lambda v:v[0])
    for start,_ in seeds:
        support={station for t,station in seeds if start<=t<=start+cfg.network_coincidence_s}
        if len(support)>=cfg.network_min_stations:
            return dict(network_status='candidate_cluster',network_anchor_record_s=float(start),
                network_station_support=len(support),network_search_start_record_s=max(cfg.minimum_noise_s,start-cfg.network_lead_s),
                network_search_end_record_s=start+cfg.network_lag_s)
    return dict(network_status='unresolved',network_anchor_record_s=np.nan,network_station_support=0,
        network_search_start_record_s=np.nan,network_search_end_record_s=np.nan)


def _finalize_windows(records,cfg):
    """Network-constrained onset refinement and conservative quiet-prefix checks.

    Every output is a PROPOSAL. Missing data/contradictory guards remain invalid,
    no event is forced on a weak channel, and no Hilbert-based onset is used.
    """
    for record in records:record['candidates']=_candidate_evidence(record,cfg)
    context=_network_context(records,cfg)
    for record in records:
        q=record['qc'];offset=record['record_offset_s'];fs=q['sampling_rate_hz'];x=record['centred'];t=record['t']
        oldflags=[v for v in q['qc_flags'].split(';') if v and v not in
            ['spectral_change_may_overlap_proposed_noise','proposed_noise_has_invalid_or_flat_samples','amplitude_change_too_early_for_noise_rule']]
        q.update(context)
        choices=[c for c in record['candidates'] if context['network_status']=='candidate_cluster' and
            context['network_search_start_record_s']<=c['time_trace_s']+offset<=context['network_search_end_record_s']]
        if not _reference_check(record,cfg)[0]:choices=[]
        # Coarse fluctuations alone cannot create an arrival on a weak channel.
        # Use them only to backtrack a nearby fine/STA candidate by at most 2 s.
        primary=[c for c in choices if c['method']!='one_second_amplitude']
        choices=[c for c in choices if c['method']!='one_second_amplitude' or any(
            0<=p['time_trace_s']-c['time_trace_s']<=2. for p in primary)]
        selected=choices[0] if choices else None
        onset=selected['time_trace_s'] if selected else np.nan
        end=onset-cfg.noise_guard_s if selected else cfg.minimum_noise_s
        status='EXTENDED_NETWORK_SUPPORTED_REVIEW' if selected else ('MINIMUM_ONLY_WEAK_OR_UNRESOLVED' if context['network_status']=='candidate_cluster' else 'MINIMUM_ONLY_NO_NETWORK')
        reason=selected['method'] if selected else 'no_supported_main_event_arrival'
        # Check earliest short excursions, including pulses too brief for a sustained detector.
        base_valid,ref,reference_flags=_reference_check(record,cfg)
        oldflags.extend(reference_flags)
        if not base_valid:
            end=np.nan;status='INVALID_INITIAL_NOISE';oldflags.append('initial_reference_invalid_or_flat')
        if selected and np.isfinite(end) and end<cfg.minimum_noise_s:
            end=np.nan;status='CONFLICT_MINIMUM_VS_GUARD';oldflags.append('arrival_too_early_for_minimum_plus_guard')
        limiting=np.nan;limit_reason=''
        if np.isfinite(end) and end>cfg.minimum_noise_s:
            # Scan through the candidate onset: an earlier pulse within its guard
            # still needs its OWN one-second guard, not just exclusion at its peak.
            until=onset if selected else end
            weak=record.get('weak_amplitude_threshold_counts',np.inf)
            fine=record['fine'];quiet=fine[(fine.window_start_s>=cfg.minimum_noise_s)&(fine.window_start_s<until)]
            suspect=quiet[(~quiet.complete)|(quiet.coverage<cfg.min_coverage)]
            limits=[(float(suspect.iloc[0].window_start_s),'incomplete_noise_bin')] if len(suspect) else []
            # Ordinary isolated weak bins do not invalidate quiet noise. Sustained
            # weak increases do; brief transients require the stricter peak check.
            limits.extend((c['time_trace_s'],'earlier_sustained_amplitude_increase')
                for c in record['candidates'] if c['method']=='weak_amplitude'
                and cfg.minimum_noise_s<=c['time_trace_s']<until)
            # An isolated high-amplitude sample need not pass the sustained test.
            peak_limit=cfg.transient_peak_factor*float(np.quantile(np.abs(ref),.995))
            bad=((~np.isfinite(x))|(np.abs(x)>peak_limit))&(t>=cfg.minimum_noise_s)&(t<until)
            indices=np.flatnonzero(bad)
            if len(indices):limits.append((float(t[indices[0]]),'earlier_transient_or_missing_sample'))
            if limits:
                limiting,limit_reason=min(limits,key=lambda z:z[0]);safe=limiting-cfg.noise_guard_s
                if safe<end:
                    if safe<cfg.minimum_noise_s:
                        end=np.nan;status='CONFLICT_MINIMUM_VS_GUARD'
                    else:
                        end=np.floor((safe+1e-9)/q['fine_window_actual_s'])*q['fine_window_actual_s'];status='SHORTENED_FOR_EARLIER_ACTIVITY_REVIEW'
                    oldflags.append(limit_reason)
        if np.isfinite(q['amplitude_onset_trace_s']) and np.isfinite(q['sta_lta_onset_trace_s']) and abs(q['amplitude_onset_trace_s']-q['sta_lta_onset_trace_s'])>cfg.detector_agreement_s:
            oldflags.append('legacy_amplitude_sta_disagreement')
        if selected and np.isfinite(q['amplitude_onset_trace_s']) and q['amplitude_onset_trace_s']-onset>cfg.detector_agreement_s:
            oldflags.append('high_amplitude_threshold_picked_later_phase')
        if not selected and record['candidates']:oldflags.append('no_local_arrival_in_network_gate' if context['network_status']=='candidate_cluster' else 'network_not_established')
        if np.isfinite(end) and np.isfinite(q['frequency_support_start_trace_s']) and cfg.minimum_noise_s<=q['frequency_support_start_trace_s']<end:
            oldflags.append('spectral_change_inside_proposed_noise_review')
        resolution=1. if selected and selected['method']=='one_second_amplitude' else (1/fs if selected and selected['method']=='sta_lta' else q['fine_window_actual_s'])
        support_end=min(selected['support_end_trace_s'],onset+max(cfg.sustain_s,resolution)) if selected else np.nan
        q.update(onset_support_start_trace_s=onset,onset_support_end_trace_s=support_end,
            onset_support_start_record_s=onset+offset,onset_support_end_record_s=support_end+offset,
            onset_support_interpretation='FIRST_SUPPORT_WINDOW_NOT_CONFIDENCE_INTERVAL',
            selected_onset_trace_s=onset,selected_onset_record_s=onset+offset,selected_onset_method=reason,
            selected_onset_status='candidate_review_required' if selected else 'unresolved',
            noise_limiting_activity_trace_s=limiting,noise_limiting_reason=limit_reason,
            proposed_noise_start_trace_s=0.,proposed_noise_end_trace_s=end,noise_proposal_status=status,
            proposed_signal_start_trace_s=onset,proposed_signal_end_trace_s=np.nan,
            qc_flags=';'.join(dict.fromkeys(oldflags)),noise_acceptance='UNREVIEWED')
        for key in ['proposed_noise_start','proposed_noise_end','proposed_signal_start','proposed_signal_end']:
            q[key+'_record_s']=q[key+'_trace_s']+offset
        # Quantify the proposed prefix; this is a stability summary, not a test
        # proving absence of seismic signal below the detection threshold.
        table=record['fine'];prefix=table[table.window_end_s<=end] if np.isfinite(end) else table.iloc[:0]
        q['proposed_noise_max_rsam_ratio']=float(prefix.rsam.max()/q['reference_rsam_counts']) if len(prefix) else np.nan
        q['proposed_noise_valid_fraction']=float(np.isfinite(x[t<end]).mean()) if np.isfinite(end) and (t<end).any() else np.nan
    return context



def detect_pretrigger_window(trace_or_stream, *, settings=None):
    """Propose per-trace noise prefixes and onset support in a triggered record.

    Parameters
    ----------
    trace_or_stream : obspy.Trace or obspy.Stream
        One already recorded event file/record. Channels may have different
        recorded start times. Gaps remain missing; no padding or filtering is
        applied. Supply only waveform channels intended for analysis (for
        example, exclude IRIG timing channels before calling).
    settings : PretriggerSettings, optional
        Initial-noise assumptions and heuristic screening parameters. Defaults
        assume five initial seconds of noise and require two distinct stations.
        Set ``network_min_stations=1`` explicitly for a single-station record.
        Set ``compute_hilbert=False`` and/or ``compute_frequency=False`` to
        omit those optional diagnostics. Neither diagnostic controls the pick.

    Returns
    -------
    dict
        ``qc``: one row per trace, containing selected onset, proposed noise
        prefix, first onset-support interval, evidence method and QC flags.
        ``network``: distinct-station association evidence for the record.
        ``evidence``: all sustained/coarse onset candidates, not just the pick.
        ``details``: per-trace amplitude, STA/LTA and optional frequency/Hilbert
        arrays/tables, including the corresponding QC row and trace.
        ``reference_time``: earliest recorded trace start (or None if empty).
        ``settings`` and ``algorithm_version`` preserve processing provenance.

        Names ending in ``_trace_s`` are seconds from that trace's start;
        ``_record_s`` are seconds from ``reference_time``. Noise intervals are
        half-open [start, end). Onset support spans the first measurement(s)
        supporting a candidate; it is NOT a statistical confidence interval
        or an assertion that the physical onset lies within those bounds.
        Missing/contradictory evidence remains unresolved (NaN).

    Notes
    -----
    Every proposal remains UNREVIEWED. No waveform, timestamp or metadata is
    modified, no files are read/written, and no UTC/response correction occurs.
    Initial quiet time must be an explicit defensible assumption; weak signals
    already present there can escape screening. Coincidence can reflect shared
    interference. This method is a useful candidate screen for emergent signals,
    not a validated rockfall/PDC classifier or a precise phase picker. Event
    ends remain unset. Arrival evidence cannot recover an acquisition trigger
    or prove the recorder's configured pretrigger buffer duration.
    """
    cfg = PretriggerSettings() if settings is None else settings
    if not isinstance(cfg, PretriggerSettings):
        raise TypeError('settings must be a PretriggerSettings instance')
    if isinstance(trace_or_stream, Trace):
        traces = [trace_or_stream]
    elif isinstance(trace_or_stream, Stream):
        traces = list(trace_or_stream)
    else:
        raise TypeError('Expected an ObsPy Trace or Stream')
    origin = min((tr.stats.starttime for tr in traces), default=None)
    records = []
    for index, trace in enumerate(traces):
        record = _analyse_trace(trace, cfg)
        offset = float(trace.stats.starttime - origin)
        row = record['qc']
        row.update(trace_index=index, trace_start_offset_record_s=offset,
                   record_reference_recorded_time=str(origin))
        for key in ['amplitude_onset', 'sta_lta_onset', 'frequency_change',
                    'frequency_support_start']:
            row[key + '_record_s'] = row[key + '_trace_s'] + offset
        record['record_offset_s'] = offset
        records.append(record)
    context = _finalize_windows(records, cfg)
    evidence = []
    for record in records:
        for candidate in record['candidates']:
            evidence.append(dict(trace_index=record['qc']['trace_index'],
                                 trace_id=record['qc']['trace_id'],
                                 time_record_s=candidate['time_trace_s'] + record['record_offset_s'],
                                 **candidate))
    columns = ['trace_index', 'trace_id', 'time_record_s', 'method',
               'time_trace_s', 'support_end_trace_s', 'seed_eligible',
               'local_amplitude_ratio', 'threshold']
    qc = pd.DataFrame([record['qc'] for record in records])
    if qc.empty:
        qc = pd.DataFrame(columns=['trace_index', 'trace_id', 'selected_onset_trace_s',
                                   'selected_onset_record_s', 'proposed_noise_end_trace_s',
                                   'proposed_noise_end_record_s', 'noise_proposal_status',
                                   'noise_acceptance'])
    return dict(qc=qc, network=context, evidence=pd.DataFrame(evidence, columns=columns),
                details=records, reference_time=origin, settings=cfg,
                algorithm_version=ALGORITHM_VERSION)
