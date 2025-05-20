import os
import uuid
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num

from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import coincidence_trigger, classic_sta_lta, trigger_onset
from obspy.core.event import (
    Catalog, Event, Origin, OriginUncertainty, Amplitude, Comment,
    CreationInfo, ResourceIdentifier #, WaveformStreamID
)

from flovopy.processing.sam import RSAM

# All functions (sanitize_stream, despike_and_extract_events, etc.) would go here...
# For brevity, only the refactored `decompose_sam` function is placed now.

def decompose_sam(sam_obj, metric='mean', trigger_type='classicstalta', sta=1200, lta=7200, thresh_on=1.8, thresh_off=1.0, min_trigs=3, outfile=None):
    raw_st = sam_obj.to_stream(metric=metric)
    sampling_interval = raw_st[0].stats.delta
    startt = min(tr.stats.starttime for tr in raw_st)
    endt = max(tr.stats.endtime for tr in raw_st)

    sanitized_st = sanitize_stream(raw_st)
    if len(sanitized_st) == 0:
        print(f"No data found after sanitization for {startt} to {endt}")
        return

    extended_st, transient_catalog = despike_and_extract_events(sanitized_st)

    extended_triggers = coincidence_trigger(
        trigger_type=trigger_type,
        stream=extended_st,
        thr_on=thresh_on,
        thr_off=thresh_off,
        sta=sta,
        lta=lta,
        thr_coincidence_sum=min((len(sanitized_st), min_trigs)),
        details=True
    )
    for trig in extended_triggers:
        trig['endtime'] = trig['time'] + trig['duration']

    transient_st = sanitized_st.copy()
    for tr in transient_st:
        match = extended_st.select(id=tr.id)
        if len(match) > 0:
            tr.data -= match[0].data

    transient_triggers = coincidence_trigger(
        stream=transient_st,
        trigger_type=trigger_type,
        thr_on=thresh_on,
        thr_off=thresh_off,
        sta=sta,
        lta=lta,
        thr_coincidence_sum=min((len(transient_st), min_trigs)),
        details=True
    )
    for trig in transient_triggers:
        trig['endtime'] = trig['time'] + trig['duration']

    transient_catalog = sorted(transient_catalog, key=lambda e: e['time'])
    all_times = [e['time'] for e in transient_catalog]
    t_min, t_max = min(all_times), max(all_times)

    window_sec = 3600
    step_sec = 60
    time_bins = []
    t = t_min
    while t <= t_max:
        time_bins.append(t)
        t += step_sec

    counts, energy = [], []
    for t0 in time_bins:
        t1 = min(t0 + window_sec, t_max)
        window_events = [e for e in transient_catalog if t0 <= e['time'] < t1]
        counts.append(len(window_events))
        energy.append(sum((e.get('max_amplitude') or 0)**2 for e in window_events))

    energy_tr = Trace()
    energy_tr.data = np.array(energy, dtype=np.float64)
    energy_tr.stats.starttime = time_bins[0]
    energy_tr.stats.delta = 60.0

    cft = classic_sta_lta(energy_tr.data, int(sta / sampling_interval), int(lta / sampling_interval))
    on_off = trigger_onset(cft, thresh_on, thresh_off)

    swarm_triggers = []
    for onset, offset in on_off:
        t_on = energy_tr.stats.starttime + onset * energy_tr.stats.delta
        t_off = energy_tr.stats.starttime + offset * energy_tr.stats.delta
        swarm_triggers.append({"time": t_on, "endtime": t_off, "duration": t_off - t_on})

    time_floats = [date2num(t.datetime) for t in time_bins]

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    geom_rsam = compute_geometric_mean_trace(sanitized_st)
    axes[0].fill_between(geom_rsam.times("matplotlib"), geom_rsam.data, color='lightblue')
    axes[0].set_ylabel("Raw RSAM")
    axes[0].set_title("Geometric Mean - Raw RSAM")

    geom_transient = compute_geometric_mean_trace(transient_st)
    axes[1].fill_between(geom_transient.times("matplotlib"), geom_transient.data, color='orange')
    for trig in transient_triggers:
        axes[1].axvline(date2num(trig['time'].datetime), color='r', linestyle='--', lw=1)
        axes[1].axvline(date2num(trig['endtime'].datetime), color='g', linestyle='--', lw=1)
    axes[1].set_ylabel("Transient")
    axes[1].set_title("Geometric Mean - Transient Stream")

    geom_extended = compute_geometric_mean_trace(extended_st)
    axes[2].fill_between(geom_extended.times("matplotlib"), geom_extended.data, color='gray')
    for trig in extended_triggers:
        axes[2].axvline(date2num(trig['time'].datetime), color='r', linestyle='--', lw=1)
        axes[2].axvline(date2num(trig['endtime'].datetime), color='g', linestyle='--', lw=1)
    axes[2].set_ylabel("Tremor")
    axes[2].set_title("Geometric Mean - Extended Stream")

    axes[3].fill_between(time_floats, counts, color='skyblue')
    axes[3].set_ylabel("Count")
    axes[3].set_title("Sliding Hourly Transient Counts")

    axes[4].fill_between(time_floats, energy, color='salmon')
    for trig in swarm_triggers:
        axes[4].axvline(date2num(trig['time'].datetime), color='r', linestyle='--', lw=1)
        axes[4].axvline(date2num(trig['endtime'].datetime), color='g', linestyle='--', lw=1)
    axes[4].set_ylabel("Energy")
    axes[4].set_title("Sliding Hourly Transient Energy")
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

    cat = save_sam_events_to_quakeml(
        transient_catalog=transient_catalog,
        extended_triggers=extended_triggers,
        swarm_triggers=swarm_triggers,
        output_path="sam_events.xml",
        stream=sanitized_st
    )

    plot_sam_event_summary(transient_catalog, extended_triggers, swarm_triggers, rsam_trace=geom_rsam)
    plot_inter_event_intervals(transient_catalog, extended_triggers, swarm_triggers)

    df = build_event_dataframe(transient_catalog, extended_triggers, swarm_triggers)
    analyze_event_correlations(df)    

    return cat

def sanitize_stream(st, expected_sampling_rate=1 / 60.0):
    st_clean = Stream()
    for tr in st:
        if tr.stats.station[-1] == 'L':
            continue
        tr.stats.sampling_rate = expected_sampling_rate
        tr.data = np.nan_to_num(np.array(tr.data, dtype=np.float64), nan=0.0)
        st_clean.append(tr)

    startt = min(tr.stats.starttime for tr in st)
    endt = max(tr.stats.endtime for tr in st)
    st_clean.trim(starttime=startt, endtime=endt, pad=True, fill_value=0.0)
    num_samples = st_clean[0].stats.npts

    for tr in st_clean:
        data = tr.data

        # Remove flat-line segments
        diffs = np.diff(data)
        flat = np.append(diffs == 0, False)
        start = None
        for i, is_flat in enumerate(flat):
            if is_flat and start is None:
                start = i
            elif not is_flat and start is not None:
                if i - start >= 60:
                    data[start:i] = np.nan
                elif start > 0 and i < len(data):
                    data[start:i] = np.interp(
                        np.arange(start, i), [start - 1, i], [data[start - 1], data[i]]
                    )
                start = None

        # Remove low-diversity segments
        noise_indices = set()
        for i in range(0, len(data) - 20 + 1, 10):
            window = data[i:i + 20]
            if len(np.unique(np.round(window, 0))) <= 4:
                noise_indices.update(range(i, i + 21))
        if noise_indices:
            print(f"{tr.id} has low diversity — flagging.")
            data[list(noise_indices)] = np.nan

        if np.isnan(data).all():
            print(f"{tr.id} is completely NaN — skipping.")
            st_clean.remove(tr)
            continue

        valid = np.where((~np.isnan(data)) & (data != 0))[0]
        if len(valid) == 0:
            st_clean.remove(tr)
            continue
        tr.data = data[valid[0]:valid[-1] + 1]
        tr.stats.starttime += valid[0] * tr.stats.delta

        if len(tr.data) < num_samples // 2:
            print(f"{tr.id} has too few points — skipping.")
            st_clean.remove(tr)
            continue

        bad = ~np.isfinite(tr.data) | (tr.data == 0)
        if np.any(bad):
            valid_idx = np.where(~bad)[0]
            tr.data = np.interp(np.arange(len(tr.data)), valid_idx, tr.data[valid_idx])

    for tr in st_clean:
        if tr.stats.starttime > startt or tr.stats.endtime < endt:
            print(f"{tr.id} incomplete — skipping.")
            st_clean.remove(tr)

    return st_clean


def despike_and_extract_events(st, window=9, z_thresh=1.2, max_len=3):
    st_clean = Stream()
    spike_times = defaultdict(list)

    for tr in st:
        data = tr.data.astype(np.float64)
        padded = np.pad(data, window, mode='reflect')
        median = np.array([np.median(padded[i:i + 2 * window + 1]) for i in range(len(data))])
        std = np.array([np.std(padded[i:i + 2 * window + 1]) for i in range(len(data))])
        spike_mask = (data - median) > z_thresh * std

        regions = []
        in_spike = False
        for i, is_spike in enumerate(spike_mask):
            if is_spike and not in_spike:
                start_idx = i
                in_spike = True
            elif not is_spike and in_spike:
                in_spike = False
                if 1 <= (i - start_idx) <= max_len:
                    regions.append((start_idx, i))

        for start, end in regions:
            spike_times[tr.id].append((tr.stats.starttime + start * tr.stats.delta, end - start))
        full_mask = np.zeros(len(data), dtype=bool)
        for start, end in regions:
            full_mask[start:end] = True
        if np.any(full_mask):
            good_idx = np.where(~full_mask)[0]
            data = np.interp(np.arange(len(data)), good_idx, data[good_idx])

        tr_clean = tr.copy()
        tr_clean.data = data
        st_clean.append(tr_clean)

    catalog = []
    all_spikes = [(t, dur, sta) for sta, spikes in spike_times.items() for t, dur in spikes]
    all_spikes.sort()
    used = set()

    for i, (t1, dur1, sta1) in enumerate(all_spikes):
        if i in used:
            continue
        group = [(t1, dur1, sta1)]
        used.add(i)
        for j in range(i + 1, len(all_spikes)):
            t2, dur2, sta2 = all_spikes[j]
            if abs(t2 - t1) <= 60:
                group.append((t2, dur2, sta2))
                used.add(j)
            else:
                break
        if len(group) >= 2:
            t0 = min(t for t, _, _ in group)
            dmax = max(d for _, d, _ in group)
            amps = []
            for t_evt, dur_evt, sta in group:
                tr_match = st.select(id=sta)[0]
                idx = int((t_evt - tr_match.stats.starttime) / tr_match.stats.delta)
                amp_vals = tr_match.data[idx:idx + dur_evt]
                if len(amp_vals):
                    amps.append(np.max(amp_vals))
            catalog.append({
                'time': t0,
                'duration_minutes': dmax,
                'nstations': len(group),
                'stations': [s for _, _, s in group],
                'max_amplitude': max(amps) if amps else None,
                'median_amplitude': np.median(amps) if amps else None
            })

    return st_clean, catalog


def compute_geometric_mean_trace(st, trace_id="GEOMEAN"):
    data_matrix = np.array([tr.data for tr in st])
    safe_data = np.where(data_matrix <= 0, 1e-10, data_matrix)
    geom_mean = np.exp(np.mean(np.log(safe_data), axis=0))
    tr0 = st[0]
    net, sta, loc, cha = tr0.id.split('.')
    return Trace(
        data=geom_mean.astype(np.float32),
        header={
            'network': net, 'station': trace_id, 'location': '', 'channel': '',
            'starttime': tr0.stats.starttime, 'delta': tr0.stats.delta,
            'sampling_rate': tr0.stats.sampling_rate, 'npts': len(geom_mean)
        }
    )




def _create_event(origin_time, author, event_id_prefix, max_amp=0.0, duration=None, station_count=None, stream_slice=None, amp_method=None):
    event_id = str(uuid.uuid4())
    origin = Origin(
        time=origin_time,
        origin_uncertainty=OriginUncertainty(),
        creation_info=CreationInfo(author=author)
    )
    event = Event(
        resource_id=ResourceIdentifier(f"smi:local/sam/{event_id_prefix}/{event_id}"),
        origins=[origin],
        creation_info=CreationInfo(author=author)
    )

    if stream_slice is not None:
        max_amp = max((np.max(tr.data) for tr in stream_slice if len(tr.data)), default=0.0)

    amplitude = Amplitude(
        generic_amplitude=max_amp,
        type="maximum",
        unit="other",
        category="point",
        method_id=ResourceIdentifier(amp_method)
    )
    event.amplitudes.append(amplitude)

    if duration is not None:
        event.comments.append(Comment(text=f"Duration: {duration:.1f} seconds"))
    if station_count is not None:
        event.comments.append(Comment(text=f"Stations: {station_count}"))

    return event


def save_sam_events_to_quakeml(transient_catalog, extended_triggers, swarm_triggers,
                               output_path, stream=None, waveform_output_dir=None):
    """
    Combines events from transient, extended, and swarm detections into a QuakeML file.

    Parameters:
    - transient_catalog: List of dicts with 'time', 'max_amplitude', 'duration_minutes', 'nstations'
    - extended_triggers: List of dicts with 'time', 'endtime'
    - swarm_triggers: List of dicts with 'time', 'endtime'
    - output_path: Path to save the QuakeML file
    - stream: Optional ObsPy Stream to compute amplitude for extended events
    - waveform_output_dir: (not yet implemented)
    """
    cat = Catalog()

    # --- 1. Transient Events ---
    for evt in transient_catalog:
        duration_sec = evt.get('duration_minutes', 0) * 60
        event = _create_event(
            origin_time=evt['time'],
            author="SAM Transient",
            event_id_prefix="transient",
            max_amp=evt.get('max_amplitude', 0.0),
            duration=duration_sec,
            station_count=evt.get('nstations'),
            amp_method="amplitude:sam_transient"
        )
        cat.events.append(event)

    # --- 2. Extended Events ---
    for trig in extended_triggers:
        slice_ = stream.slice(trig['time'], trig['endtime']) if stream else None
        duration_sec = (trig['endtime'] - trig['time']) if 'endtime' in trig else 0
        event = _create_event(
            origin_time=trig['time'],
            author="SAM Tremor",
            event_id_prefix="tremor",
            stream_slice=slice_,
            duration=duration_sec,
            amp_method="amplitude:sam_tremor"
        )
        cat.events.append(event)

    # --- 3. Swarm Events ---
    for trig in swarm_triggers:
        duration_sec = (trig['endtime'] - trig['time']) if 'endtime' in trig else 0
        event = _create_event(
            origin_time=trig['time'],
            author="SAM Swarm",
            event_id_prefix="swarm",
            duration=duration_sec,
            amp_method="amplitude:sam_swarm"
        )
        cat.events.append(event)

    # --- Save ---
    cat.write(output_path, format="QUAKEML")
    print(f"[✓] Wrote {len(cat.events)} SAM events to {output_path}")
    return cat

def plot_sam_event_summary(transient_catalog, extended_triggers, swarm_triggers, rsam_trace=None, figsize=(12, 6)):
    """
    Plots a summary timeline of SAM-detected events.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if rsam_trace:
        t = rsam_trace.times("matplotlib")
        ax.plot(t, rsam_trace.data, color='lightgray', lw=1.0, label="RSAM")

    def label_once(label, label_set):
        return label if label not in label_set else ""

    labels = set()

    for evt in transient_catalog:
        t = date2num(evt['time'].datetime)
        amp = evt.get('max_amplitude', 0.0)
        ax.plot(t, amp, 'o', color='orange', markersize=4, label=label_once('Transient', labels))
        labels.add('Transient')

    for trig in extended_triggers:
        t0, t1 = map(lambda x: date2num(x.datetime), (trig['time'], trig['endtime']))
        ax.axvspan(t0, t1, color='gray', alpha=0.4, label=label_once('Extended', labels))
        labels.add('Extended')

    for trig in swarm_triggers:
        t0, t1 = map(lambda x: date2num(x.datetime), (trig['time'], trig['endtime']))
        ax.axvspan(t0, t1, color='red', alpha=0.3, label=label_once('Swarm', labels))
        labels.add('Swarm')

    ax.set(title="SAM Event Timeline", ylabel="Amplitude (arbitrary units)", xlabel="Time (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_inter_event_intervals(transient_catalog, extended_triggers, swarm_triggers):
    def compute_intervals(times):
        return [(t2 - t1) for t1, t2 in zip(times[:-1], times[1:])], times[1:]

    """
    def plot_intervals(ax, times, label, color, unit='minute', logscale=True, window=3):
        intervals_sec, times_mid = compute_intervals(times)
        df = pd.DataFrame({'time': [t.datetime for t in times_mid], 'interval': intervals_sec})
        df['smoothed'] = df['interval'].rolling(window=window, min_periods=1).mean()
        if unit == 'hour':
            df[['interval', 'smoothed']] /= 3600
        elif unit == 'minute':
            df[['interval', 'smoothed']] /= 60

        ax.plot(df['time'], df['interval'], 'o:', label=f'{label} intervals', color=color, alpha=0.5, linewidth=0.2)
        ax.plot(df['time'], df['smoothed'], '-', label=f'{label} (rolling avg)', color=color, linewidth=2)
        if logscale:
            ax.set_yscale('log')
    """

    def plot_intervals(ax, times, label, color, amplitudes=None, unit='minute', logscale=True, window=3):
        # Compute inter-event intervals and midpoints
        intervals_sec, times_mid = compute_intervals(times)
        df = pd.DataFrame({'time': [t.datetime for t in times_mid], 'interval': intervals_sec})

        # Rolling average
        df['smoothed'] = df['interval'].rolling(window=window, min_periods=1).mean()

        # Convert units
        if unit == 'hour':
            df[['interval', 'smoothed']] /= 3600
        elif unit == 'minute':
            df[['interval', 'smoothed']] /= 60

        # Marker sizes scaled by amplitude (if provided)
        if amplitudes is not None and len(amplitudes) >= len(df):
            amp_array = np.array(amplitudes[1:])  # align with diff() result
            size = np.interp(amp_array, (amp_array.min(), amp_array.max() + 1e-10), (4, 20)) ** 2
        else:
            size = 16  # constant marker size fallback

        # Plot raw and smoothed interval series
        ax.scatter(df['time'], df['interval'], s=size, color=color, alpha=0.5, edgecolors='k', label=f"{label} intervals")
        ax.plot(df['time'], df['smoothed'], '-', color=color, linewidth=2, label=f"{label} (rolling avg)")

        if logscale:
            ax.set_yscale('log')

    # Preprocess event times and amplitudes
    transient_times = sorted([e['time'] for e in transient_catalog])
    transient_amps = [e.get('max_amplitude', 0.0) or 0.0 for e in transient_catalog]

    extended_times = sorted([e['time'] + 0.5 * (e['endtime'] - e['time']) for e in extended_triggers])
    extended_amps = [
        max(tr['endtime'] - tr['time'], 0.1) for tr in extended_triggers  # or replace with amplitudes if available
    ]

    swarm_times = sorted([e['time'] + 0.5 * (e['endtime'] - e['time']) for e in swarm_triggers])
    swarm_amps = [
        max(tr['endtime'] - tr['time'], 0.1) for tr in swarm_triggers  # or replace with amplitudes if available
    ]

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    plot_intervals(ax[0], extended_times, "Extended", "gray", amplitudes=extended_amps, unit='hour', logscale=False)
    plot_intervals(ax[0], swarm_times, "Swarm", "red", amplitudes=swarm_amps, unit='hour', logscale=False)
    ax[0].set_ylabel("Interval (hours)")
    ax[0].set_title("Inter-event Intervals: Extended + Swarm")
    ax[0].legend()
    ax[0].grid(True)

    plot_intervals(ax[1], transient_times, "Transient", "orange", amplitudes=transient_amps, unit='minute', logscale=False, window=10)
    ax[1].set_ylabel("Interval (minutes)")
    ax[1].set_title("Inter-event Intervals: Transient")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax[1].legend()
    ax[1].grid(True)

    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.show()



def build_event_dataframe(transient_catalog, extended_triggers, swarm_triggers):
    """
    Returns a unified DataFrame for all event types, with:
    - event_type
    - time
    - duration (s)
    - max_amplitude
    - energy (if amplitude available)
    - inter_event_interval (optional, computed later)
    """
    rows = []

    for e in transient_catalog:
        amp = e.get('max_amplitude', 0.0) or 0.0
        dur = e.get('duration_minutes', 0.0) * 60.0
        energy = amp**2 if amp else None
        rows.append({
            'event_type': 'transient',
            'time': e['time'].datetime,
            'duration': dur,
            'amplitude': amp,
            'energy': energy
        })

    for e in extended_triggers:
        amp = e.get('max_amplitude', np.nan)
        dur = (e['endtime'] - e['time'])
        energy = amp**2 if not np.isnan(amp) else None
        rows.append({
            'event_type': 'extended',
            'time': (e['time'] + 0.5 * dur).datetime,
            'duration': dur,
            'amplitude': amp,
            'energy': energy
        })

    for e in swarm_triggers:
        dur = (e['endtime'] - e['time'])
        rows.append({
            'event_type': 'swarm',
            'time': (e['time'] + 0.5 * dur).datetime,
            'duration': dur,
            'amplitude': np.nan,
            'energy': np.nan
        })

    df = pd.DataFrame(rows)
    df.sort_values('time', inplace=True)
    df['interval'] = df['time'].diff().dt.total_seconds()
    return df.reset_index(drop=True)  

def analyze_event_correlations(df):
    # Drop rows with any missing numeric values
    numeric_df = df[['duration', 'amplitude', 'energy', 'interval']].dropna()

    # Correlation matrix
    corr = numeric_df.corr()
    print("\nCorrelation Matrix:\n", corr.round(3))

    # Optional: visualise as heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Event Metrics")
    plt.tight_layout()
    plt.show()

    # Pairwise scatter plots
    pd.plotting.scatter_matrix(numeric_df, figsize=(8, 8), diagonal='kde', alpha=0.6)
    plt.suptitle("Scatter Matrix of Duration, Amplitude, Energy, Interval")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dropbox_path = os.path.expanduser('~/Dropbox')
    BINARY_DIR = Path(dropbox_path) / 'DATA' / 'Montserrat' / 'RSAM_1'

    start_date = UTCDateTime(1996, 7, 27)
    end_date = UTCDateTime(1996,8,12)
    STA_SECONDS, LTA_SECONDS = 1200, 7200
    expanded_start = start_date - LTA_SECONDS
    expanded_end = end_date + LTA_SECONDS

    files = list(BINARY_DIR.glob(f'M???{start_date.year}.DAT'))
    stations = sorted(set(p.name[:4] for p in files))
    print(f"Processing {len(stations)} stations: {stations}")

    rsamObj = RSAM.readRSAMbinary(str(BINARY_DIR), stations, expanded_start, expanded_end, verbose=False, convert_legacy_ids_using_this_network='MV')
    if len(rsamObj) > 0:
        cat = decompose_sam(rsamObj, metric='mean', sta=STA_SECONDS, lta=LTA_SECONDS, outfile=None)