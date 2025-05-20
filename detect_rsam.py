from pathlib import Path
BINARY_DIR = Path('/Volumes/DATA/Montserrat/ASN/RSAM/RSAM_1')
import obspy
from flovopy.processing.sam import RSAM

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates

from obspy.signal.trigger import coincidence_trigger
from obspy import UTCDateTime

import numpy as np
from obspy import Stream

def sanitize_stream(st, expected_sampling_rate=1/60.0):
    """
    Cleans and interpolates over gaps in RSAM data.
    
    - Ensures uniform sampling rate
    - Trims to overlapping time range
    - Interpolates gaps and replaces NaNs
    """
    st_clean = Stream()

    # Step 1: Set expected sampling rate and pad/match times
    for tr in st:
        tr.stats.sampling_rate = expected_sampling_rate
        tr.data = np.array(tr.data, dtype=np.float64)
        tr.data = np.nan_to_num(tr.data, nan=0.0)  # temporary replacement
        st_clean.append(tr)

    # Step 2: Trim all traces to overlapping time window
    start = max(tr.stats.starttime for tr in st_clean)
    end = min(tr.stats.endtime for tr in st_clean)
    st_clean.trim(starttime=start, endtime=end, pad=True, fill_value=0)

    # Step 3: Interpolate over missing samples (zero-filled in trim)
    for tr in st_clean:
        times = tr.times()
        data = tr.data

        # Identify zero-filled gaps (RSAM rarely has true zeros)
        gap_mask = data == 0
        if np.any(gap_mask):
            valid_idx = np.where(~gap_mask)[0]
            interp_data = np.interp(np.arange(len(data)), valid_idx, data[valid_idx])
            tr.data = interp_data

    return st_clean

from obspy import Stream
from collections import defaultdict
import numpy as np
from obspy.core import UTCDateTime

def despike_and_extract_events(st, window=5, z_thresh=5.0, max_len=3):
    """
    Remove 1–3 minute RSAM spikes and extract coincident transient events.
    
    Parameters:
    - st: ObsPy Stream
    - window: int, half-window size for median/std filter
    - z_thresh: float, spike threshold in standard deviations
    - max_len: int, maximum consecutive spike length (in minutes)

    Returns:
    - st_clean: Stream with spikes interpolated
    - event_catalog: list of dicts with event time, duration, and station count
    """
    st_clean = Stream()
    spike_times_by_station = defaultdict(list)

    # --- Step 1: Detect and interpolate spikes per Trace
    for tr in st:
        data = tr.data.astype(np.float64)
        cleaned = data.copy()

        # Pad edges
        padded = np.pad(data, window, mode='reflect')
        median = np.array([np.median(padded[i:i+2*window+1]) for i in range(len(data))])
        std = np.array([np.std(padded[i:i+2*window+1]) for i in range(len(data))])

        # Flag spike candidates
        spike_mask = (data - median) > z_thresh * std

        # Group into contiguous spike segments
        spike_regions = []
        in_spike = False
        start_idx = 0
        for i, val in enumerate(spike_mask):
            if val and not in_spike:
                start_idx = i
                in_spike = True
            elif not val and in_spike:
                end_idx = i
                in_spike = False
                if 1 <= (end_idx - start_idx) <= max_len:
                    spike_regions.append((start_idx, end_idx))
        if in_spike:
            end_idx = len(data)
            if 1 <= (end_idx - start_idx) <= max_len:
                spike_regions.append((start_idx, end_idx))

        # Store spike times per Trace
        for start, end in spike_regions:
            spike_start_time = tr.stats.starttime + start * tr.stats.delta
            duration = end - start
            spike_times_by_station[tr.id].append((spike_start_time, duration))

        # Interpolate spikes
        spike_mask_full = np.zeros(len(data), dtype=bool)
        for start, end in spike_regions:
            spike_mask_full[start:end] = True

        if np.any(spike_mask_full):
            good_idx = np.where(~spike_mask_full)[0]
            interp_data = np.interp(np.arange(len(data)), good_idx, data[good_idx])
            cleaned = interp_data

        tr_clean = tr.copy()
        tr_clean.data = cleaned
        st_clean.append(tr_clean)

    # --- Step 2: Identify coincident spikes across traces
    event_catalog = []
    all_spikes = []
    for station, spikes in spike_times_by_station.items():
        for t, dur in spikes:
            all_spikes.append((t, dur, station))
    all_spikes.sort()

    used = set()
    for i, (t1, dur1, sta1) in enumerate(all_spikes):
        if i in used:
            continue
        coincident = [(t1, dur1, sta1)]
        used.add(i)
        for j in range(i + 1, len(all_spikes)):
            t2, dur2, sta2 = all_spikes[j]
            if abs(t2 - t1) <= 60:  # 1-minute tolerance
                coincident.append((t2, dur2, sta2))
                used.add(j)
            else:
                break  # sorted list

        if len(coincident) >= 2:
            avg_time = min([t for t, _, _ in coincident])
            max_dur = max([d for _, d, _ in coincident])

            # Compute max amplitude from the original stream
            amplitudes = []
            for t_evt, dur_evt, station in coincident:
                tr_match = st.select(id=station)[0]
                idx = int((t_evt - tr_match.stats.starttime) / tr_match.stats.delta)
                amp_vals = tr_match.data[idx:idx + dur_evt]
                if len(amp_vals) > 0:
                    amplitudes.append(np.max(amp_vals))

            event_catalog.append({
                'time': avg_time,
                'duration_minutes': max_dur,
                'nstations': len(coincident),
                'stations': [s for _, _, s in coincident],
                'max_amplitude': max(amplitudes) if amplitudes else None,
                'median_amplitude': np.median(amplitudes) if amplitudes else None
            })

    return st_clean, event_catalog

from matplotlib.dates import date2num

def plot_stream_with_triggers(st, triggers, outfile=None, lw=2):
    """
    Plot a stream of RSAM data with trigger ON and OFF times overlaid.
    Trigger ON = red dashed lines
    Trigger OFF = green dashed lines
    """
    print('Plotting stream with triggers...')
    fig, axes = plt.subplots(len(st), 1, figsize=(7, len(st) * 2.5), sharex=True)
    if len(st) == 1:
        axes = [axes]  # ensure iterable

    # Plot each trace
    for ax, tr in zip(axes, st):
        times_array = tr.times("matplotlib")
        ax.plot(times_array, tr.data, 'k-', linewidth=0.8)
        ax.set_ylabel(f"{tr.id}", fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

        # Plot triggers: red for ON, green for OFF
        for trig in triggers:
            t_on = date2num(trig['time'].datetime)
            t_off = date2num(trig['endtime'].datetime)
            ax.axvline(t_on, color='r', linestyle='--', linewidth=lw)
            ax.axvline(t_off, color='g', linestyle='--', linewidth=lw)

    # Final formatting
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
    plt.xlim([date2num(tr.stats.starttime.datetime), date2num(tr.stats.endtime.datetime)])
    plt.suptitle(f"{tr.stats.starttime.date} - {tr.stats.endtime.date}", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()

def plot_stream_with_triggers(st, triggers, outfile=None, lw=2):
    """
    Plot a stream of RSAM data with area filled and trigger ON/OFF times.
    Trigger ON = red dashed lines
    Trigger OFF = green dashed lines
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.dates import date2num

    print('Plotting stream with triggers...')
    fig, axes = plt.subplots(len(st), 1, figsize=(7, len(st) * 2.5), sharex=True)
    if len(st) == 1:
        axes = [axes]  # ensure iterable

    for ax, tr in zip(axes, st):
        times_array = tr.times("matplotlib")
        data = tr.data

        # Fill under curve
        ax.fill_between(times_array, data, color='lightgray', alpha=0.6, step='mid')

        # Plot line over top
        ax.plot(times_array, data, 'k-', linewidth=0.8)

        ax.set_ylabel(f"{tr.id}", fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

        # Plot triggers: red for ON, green for OFF
        for trig in triggers:
            t_on = date2num(trig['time'].datetime)
            t_off = date2num(trig['endtime'].datetime)
            ax.axvline(t_on, color='r', linestyle='--', linewidth=lw)
            ax.axvline(t_off, color='g', linestyle='--', linewidth=lw)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
    plt.xlim([date2num(tr.stats.starttime.datetime), date2num(tr.stats.endtime.datetime)])
    plt.suptitle(f"{tr.stats.starttime.date} – {tr.stats.endtime.date}", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()        

from obspy import Stream, Trace
import numpy as np

def generate_rockfall_stream(event_catalog, template_stream):
    """
    Generate a synthetic Stream with spike amplitudes only at event times.
    
    Parameters:
    - event_catalog: list of detected rockfall events
    - template_stream: original RSAM Stream (used for timing and metadata)

    Returns:
    - synthetic Stream object with spike amplitudes only at event locations
    """
    rock_stream = Stream()
    sampling_rate = template_stream[0].stats.sampling_rate
    delta = 1 / sampling_rate
    npts = len(template_stream[0].data)
    start = template_stream[0].stats.starttime

    # Prepare a blank Trace per station
    traces_by_station = {}
    for tr in template_stream:
        new_data = np.zeros(npts)
        traces_by_station[tr.id] = Trace(data=new_data, header=tr.stats.copy())

    # Fill in spike amplitudes from event catalog
    for evt in event_catalog:
        t = evt['time']
        dur = evt['duration_minutes']
        amp = evt['max_amplitude']
        for station in evt['stations']:
            if station in traces_by_station:
                tr = traces_by_station[station]
                idx = int((t - tr.stats.starttime) / delta)
                idx_end = min(idx + dur, len(tr.data))
                if 0 <= idx < len(tr.data):
                    tr.data[idx:idx_end] = amp  # spike amplitude over 1–3 samples

    # Assemble into stream
    for tr in traces_by_station.values():
        rock_stream.append(tr)

    return rock_stream

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from obspy import UTCDateTime

def plot_cumulative_rockfall_stats(event_catalog):
    if not event_catalog:
        print("No events to plot.")
        return

    # Sort by time
    catalog = sorted(event_catalog, key=lambda e: e['time'])

    times = [e['time'].datetime for e in catalog]
    amplitudes = [e['max_amplitude'] or 0 for e in catalog]
    energies = np.cumsum([a**2 for a in amplitudes])
    counts = np.arange(1, len(times)+1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot cumulative count
    ax1.plot(times, counts, 'b-', label='Cumulative Count')
    ax1.set_ylabel("Cumulative Event Count", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
    ax1.set_xlabel("Date")

    # Plot cumulative energy on second y-axis
    ax2 = ax1.twinx()
    ax2.plot(times, energies, 'r-', label='Cumulative Energy (Amplitude²)')
    ax2.set_ylabel("Cumulative Energy Proxy", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Cumulative Rockfall Events and Energy")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
from obspy import UTCDateTime

def plot_hourly_counts_by_duration(event_catalog, max_duration=6):
    from collections import defaultdict, Counter
    from datetime import timedelta

    # Bin structure: {duration: Counter({hour_timestamp: count})}
    bins_by_duration = defaultdict(Counter)

    for evt in event_catalog:
        dur = evt.get('duration_minutes')
        if dur is None or dur < 1 or dur > max_duration:
            continue
        t = evt['time'].datetime.replace(minute=0, second=0, microsecond=0)
        bins_by_duration[dur][t] += 1

    # Get complete hourly range
    all_times = [evt['time'].datetime.replace(minute=0, second=0, microsecond=0) for evt in event_catalog]
    t_min = min(all_times)
    t_max = max(all_times)
    time_bins = []
    t = t_min
    while t <= t_max:
        time_bins.append(t)
        t += timedelta(hours=1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for dur in range(1, max_duration + 1):
        counts = [bins_by_duration[dur].get(t, 0) for t in time_bins]
        ax.plot(time_bins, counts, label=f"{dur} min")

    ax.set_xlabel("Date (hourly bins)")
    ax.set_ylabel("Rockfall Count per Hour")
    ax.set_title("Hourly Rockfall Counts by Duration")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Hh'))
    plt.legend(title="Duration")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np

def plot_hourly_cumulative_by_duration(event_catalog, max_duration=6):
    """
    Plot hourly-binned cumulative rockfall counts grouped by duration (1–max_duration minutes).
    """
    # Bin structure: {duration: Counter({hour_timestamp: count})}
    bins_by_duration = defaultdict(Counter)

    # Step 1: Bin events to the hour
    for evt in event_catalog:
        dur = evt.get('duration_minutes')
        if dur is None or dur < 1 or dur > max_duration:
            continue
        t = evt['time'].datetime.replace(minute=0, second=0, microsecond=0)
        bins_by_duration[dur][t] += 1

    # Step 2: Determine overall time range
    all_times = [evt['time'].datetime for evt in event_catalog]
    t_min = min(all_times).replace(minute=0, second=0, microsecond=0)
    t_max = max(all_times).replace(minute=0, second=0, microsecond=0)
    time_bins = []
    t = t_min
    while t <= t_max:
        time_bins.append(t)
        t += timedelta(hours=1)

    # Step 3: Plot cumulative sums
    fig, ax = plt.subplots(figsize=(10, 5))

    for dur in range(1, max_duration + 1):
        hourly_counts = [bins_by_duration[dur].get(t, 0) for t in time_bins]
        cumulative = np.cumsum(hourly_counts)
        ax.plot(time_bins, cumulative, label=f"{dur} min")

    ax.set_xlabel("Date (hourly bins)")
    ax.set_ylabel("Cumulative Count")
    ax.set_title("Hourly-Binned Cumulative Rockfall Counts by Duration")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Hh'))
    plt.legend(title="Duration")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

def plot_hourly_energy_by_duration(event_catalog, max_duration=6):
    from collections import defaultdict
    from datetime import timedelta

    energy_by_duration = defaultdict(lambda: defaultdict(float))

    for evt in event_catalog:
        dur = evt.get('duration_minutes')
        amp = evt.get('max_amplitude', 0.0)
        if dur is None or dur < 1 or dur > max_duration or amp is None:
            continue
        t = evt['time'].datetime.replace(minute=0, second=0, microsecond=0)
        energy_by_duration[dur][t] += amp ** 2

    all_times = [evt['time'].datetime.replace(minute=0, second=0, microsecond=0) for evt in event_catalog]
    t_min = min(all_times)
    t_max = max(all_times)
    time_bins = []
    t = t_min
    while t <= t_max:
        time_bins.append(t)
        t += timedelta(hours=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for dur in range(1, max_duration + 1):
        energies = [energy_by_duration[dur].get(t, 0.0) for t in time_bins]
        ax.plot(time_bins, energies, label=f"{dur} min")

    ax.set_xlabel("Date (hourly bins)")
    ax.set_ylabel("Energy Proxy (Amplitude²) per Hour")
    ax.set_title("Hourly Rockfall Energy by Duration")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Hh'))
    plt.legend(title="Duration")
    plt.tight_layout()
    plt.show()

def plot_sliding_hourly_counts_by_duration(event_catalog, max_duration=6):
    from datetime import timedelta
    from collections import defaultdict

    # Sort catalog
    catalog = sorted(event_catalog, key=lambda e: e['time'])
    all_times = [e['time'].datetime for e in catalog]
    t_min = min(all_times)
    t_max = max(all_times)

    # Prepare time steps (1-minute step, 1-hour window)
    step = timedelta(minutes=1)
    window = timedelta(hours=1)
    time_steps = []
    t = t_min
    while t + window <= t_max:
        time_steps.append(t)
        t += step

    # Prepare series for each duration (1–6) and combined
    counts_by_duration = defaultdict(list)

    for t0 in time_steps:
        t1 = t0 + window
        events_in_window = [e for e in catalog if t0 <= e['time'].datetime < t1]
        duration_counts = defaultdict(int)
        for e in events_in_window:
            dur = e.get('duration_minutes')
            if dur and 1 <= dur <= max_duration:
                duration_counts[dur] += 1
                duration_counts['all'] += 1
        for dur in list(range(1, max_duration + 1)) + ['all']:
            counts_by_duration[dur].append(duration_counts.get(dur, 0))

    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 5))
    for dur in range(1, max_duration + 1):
        ax.plot(time_steps, counts_by_duration[dur], label=f"{dur} min")
    ax.plot(time_steps, counts_by_duration['all'], label="All durations", color='k', linewidth=1.5)

    ax.set_xlabel("Date (sliding 1h windows, 1-min step)")
    ax.set_ylabel("Rockfall Count")
    ax.set_title("Sliding-Hour Rockfall Counts by Duration")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.legend(title="Duration")
    plt.tight_layout()
    plt.show()

def plot_sliding_hourly_energy_by_duration(event_catalog, max_duration=6):
    from datetime import timedelta
    from collections import defaultdict

    # Sort catalog
    catalog = sorted(event_catalog, key=lambda e: e['time'])
    all_times = [e['time'].datetime for e in catalog]
    t_min = min(all_times)
    t_max = max(all_times)

    # Define 1-minute step sliding 1-hour window
    step = timedelta(minutes=1)
    window = timedelta(hours=1)
    time_steps = []
    t = t_min
    while t + window <= t_max:
        time_steps.append(t)
        t += step

    # Compute energy per duration per window
    energy_by_duration = defaultdict(list)

    for t0 in time_steps:
        t1 = t0 + window
        events_in_window = [e for e in catalog if t0 <= e['time'].datetime < t1]
        duration_energy = defaultdict(float)
        for e in events_in_window:
            dur = e.get('duration_minutes')
            amp = e.get('max_amplitude', 0.0)
            if dur and amp and 1 <= dur <= max_duration:
                duration_energy[dur] += amp**2
                duration_energy['all'] += amp**2
        for dur in list(range(1, max_duration + 1)) + ['all']:
            energy_by_duration[dur].append(duration_energy.get(dur, 0.0))

    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 5))
    for dur in range(1, max_duration + 1):
        ax.plot(time_steps, energy_by_duration[dur], label=f"{dur} min")
    ax.plot(time_steps, energy_by_duration['all'], label="All durations", color='k', linewidth=1.5)

    ax.set_xlabel("Date (sliding 1h windows, 1-min step)")
    ax.set_ylabel("Energy Proxy (Amplitude²)")
    ax.set_title("Sliding-Hour Rockfall Energy by Duration")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.legend(title="Duration")
    plt.tight_layout()
    plt.show()

def plot_sliding_hourly_counts_and_energy(event_catalog, window_minutes=60, step_minutes=1):
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Sort events
    catalog = sorted(event_catalog, key=lambda e: e['time'])
    all_times = [e['time'].datetime for e in catalog]
    t_min = min(all_times)
    t_max = max(all_times)

    # Setup sliding windows
    step = timedelta(minutes=step_minutes)
    window = timedelta(minutes=window_minutes)
    time_steps = []
    t = t_min
    while t + window <= t_max:
        time_steps.append(t)
        t += step

    # Compute counts and energy
    counts = []
    energies = []
    for t0 in time_steps:
        t1 = t0 + window
        events = [e for e in catalog if t0 <= e['time'].datetime < t1]
        counts.append(len(events))
        energies.append(sum((e.get('max_amplitude') or 0)**2 for e in events))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Subplot 1: Rockfall Count
    ax1.fill_between(time_steps, counts, color='skyblue', alpha=0.7)
    ax1.set_ylabel("Rockfall Count")
    ax1.set_title("Sliding-Hour Rockfall Count")

    # Subplot 2: Rockfall Energy
    ax2.fill_between(time_steps, energies, color='salmon', alpha=0.7)
    ax2.set_ylabel("Energy Proxy (Amplitude²)")
    ax2.set_title("Sliding-Hour Rockfall Energy")
    ax2.set_xlabel("Date (1h window, 1-min step)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

    plt.tight_layout()
    plt.show()

from obspy import Trace
import numpy as np

def compute_geometric_mean_trace(st, trace_id="GEOMEAN"):
    """
    Compute a geometric mean Trace from a Stream of aligned Traces.

    Parameters:
    - st: ObsPy Stream (assumed aligned and same length)
    - trace_id: str, ID for the new trace

    Returns:
    - ObsPy Trace containing the geometric mean across all traces
    """
    if len(st) == 0:
        raise ValueError("Stream is empty")

    # Stack data into 2D array [n_traces, n_samples]
    data_matrix = np.array([tr.data for tr in st])

    # Avoid log(0) errors: set zeros to small positive number (or mask)
    safe_data = np.where(data_matrix <= 0, 1e-10, data_matrix)

    # Compute geometric mean across axis 0 (time)
    log_mean = np.mean(np.log(safe_data), axis=0)
    geom_mean = np.exp(log_mean)

    # Create new Trace
    tr0 = st[0]
    geom_trace = Trace(
        data=geom_mean.astype(np.float32),
        header={
            'network': 'XX',
            'station': trace_id,
            'location': '',
            'channel': 'RSAM',
            'starttime': tr0.stats.starttime,
            'delta': tr0.stats.delta,
            'sampling_rate': tr0.stats.sampling_rate,
            'npts': len(geom_mean)
        }
    )
    return geom_trace







startt = obspy.core.UTCDateTime(1996,8,2)
endt = obspy.core.UTCDateTime(1996,8,6)

miniseedfile = BINARY_DIR / '19960802.miniseed'
if miniseedfile.exists():
    st = obspy.read(str(miniseedfile))
else:
    print(f"File {miniseedfile} does not exist. Loading RSAM binary files.")
    files = list(BINARY_DIR.glob(f'M???{startt.year}.DAT'))
    #print(files)
    stations = list(set([path.name[0:4] for path in files]))
    stations.remove('MCPE')
    stations.remove('MCPN')
    stations.remove('MCPZ')
    stations.remove('MGT2')
    stations.remove('MSPT')
    print(stations)
    rsamObj = RSAM.readRSAMbinary(str(BINARY_DIR), stations, startt, endt, verbose=True)
    #print(rsamObj)
    #rsamObj.plot()

    # Convert RSAM object to ObsPy Stream
    st = rsamObj.to_stream()
    st.write(str(miniseedfile), format='MSEED')

geom_tr = compute_geometric_mean_trace(st)
plot_stream_with_triggers(Stream([geom_tr]), [])
st = sanitize_stream(st)
# Check for gaps in the data
import numpy as np

for tr in st:
    has_nan = not np.isfinite(tr.data).all()
    print(f"{tr.id} has NaNs:", has_nan)

st, rockfall_catalog = despike_and_extract_events(st, window=9, z_thresh=1.5, max_len=6)
print(rockfall_catalog)

print(f"Detected {len(rockfall_catalog)} potential RSAM transient events:")
for evt in rockfall_catalog:
    print(evt['time'], f"({evt['duration_minutes']} min)", evt['nstations'], "stations")    

# Define STA/LTA parameters (in samples, since RSAM is at 1 sample/minute)
sta = 1200       # 30 minutes STA
lta = 7200      # 3 hours LTA
thr_on = 2.0   # Trigger when STA/LTA exceeds this
thr_off = 1.0  # Trigger off when STA/LTA falls below this

# Run network coincidence detector
triggers = coincidence_trigger(
    trigger_type="classicstalta",
    stream=st,
    thr_on=thr_on,
    thr_off=thr_off,
    sta=sta,
    lta=lta,
    thr_coincidence_sum=3,             # <-- REQUIRED
    trigger_off_extension=0.0,
    details=True
)

# View or print results
for trig in triggers:
    print(trig)
    trig['endtime']= trig['time'] + trig['duration']
    #print(trig['time'], trig['coincidence_sum'], trig['stations'])

plot_stream_with_triggers(st, triggers)
st.write(str(BINARY_DIR / 'tremor_stream.mseed'), format='MSEED')

# Generate synthetic rockfall stream
rockfall_stream = generate_rockfall_stream(rockfall_catalog, st)
rockfall_stream.write(str(BINARY_DIR / 'rockfall_stream.mseed'), format='MSEED')
rockfall_stream.plot(equal_scale=False)#, outfile=str(BINARY_DIR / 'rockfall_stream.png'))

#plot_cumulative_rockfall_stats(rockfall_catalog)

'''

plot_hourly_counts_by_duration(rockfall_catalog, max_duration=6)
plot_hourly_energy_by_duration(rockfall_catalog, max_duration=6)

plot_sliding_hourly_counts_by_duration(rockfall_catalog, max_duration=6)
plot_sliding_hourly_energy_by_duration(rockfall_catalog, max_duration=6)
'''

plot_sliding_hourly_counts_and_energy(rockfall_catalog)

geom_tr = compute_geometric_mean_trace(st)
plot_stream_with_triggers(Stream([geom_tr]), triggers)