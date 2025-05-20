"""
Cleaned RSAM Processing Script
Author: Glenn Thompson (adapted)
Description:
    - Reads RSAM binary or MiniSEED data
    - Cleans and interpolates traces
    - Detects short-duration transients (rockfalls)
    - Computes geometric mean trace
    - Runs STA/LTA network trigger
    - Plots and saves results
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from collections import defaultdict
#from datetime import timedelta

from obspy import Stream, Trace, UTCDateTime, read
from obspy.signal.trigger import coincidence_trigger
from flovopy.processing.sam import RSAM

# ------------------------ FUNCTIONS ------------------------


def sanitize_stream(st, expected_sampling_rate=1/60.0):
    """
    Cleans an ObsPy Stream object by:
    - enforcing a uniform sampling rate,
    - trimming to overlapping time windows,
    - interpolating over gaps and flat segments,
    - removing windows with suspiciously low value diversity.

    Parameters:
    - st: ObsPy Stream
    - expected_sampling_rate: float (samples per second)

    Returns:
    - Cleaned ObsPy Stream
    """
    from obspy import Stream
    import numpy as np

    st_clean = Stream()
    for tr in st:

        # remove low gain channels
        if tr.stats.station[-1] == 'L':
            continue

        tr.stats.sampling_rate = expected_sampling_rate
        tr.data = np.array(tr.data, dtype=np.float64)
        tr.data = np.nan_to_num(tr.data, nan=0.0)
        st_clean.append(tr)

    # Trim and pad the stream to the same start and end time
    startt = min(tr.stats.starttime for tr in st)
    endt = max(tr.stats.endtime for tr in st)
    st_clean.trim(starttime=startt, endtime=endt, pad=True, fill_value=0.0)
    print(f'zeroed stream: {st_clean}')
    num_samples = st_clean[0].stats.npts


    # at this point, bad data is set to 0.0. but hereafter it will be marked with np.nan
    for tr in st_clean:
        bit_noise=False
        data = tr.data

        
        # Detect and handle flat-line segments
        diffs = np.diff(data)
        flat = np.append(diffs == 0, False)
        start = None
        for i, is_flat in enumerate(flat):
            if is_flat and start is None:
                start = i
            elif not is_flat and start is not None:
                duration = i - start
                if duration >= 60:  # 1 hour at 1 sample/min
                    data[start:i] = np.nan
                else:
                    if start > 0 and i < len(data):
                        data[start:i] = np.interp(
                            np.arange(start, i),
                            [start - 1, i],
                            [data[start - 1], data[i]]
                        )
                start = None
        

        # Detect and mask low-diversity segments (moving window)
        window_size = 20
        stride = window_size // 2
        noise_indices = []
        for i in range(0, len(data) - window_size + 1, stride):
            window = data[i:i + window_size]
            unique_values = np.unique(np.round(window, decimals=0))
            if len(unique_values) <= min((window_size // 5, 10)):
                bit_noise=True
                noise_indices.extend(range(i, i + window_size + 1))
        if bit_noise:      
            print(f"{tr.id} has low diversity in data — skipping or flagging.")
            #st_clean.remove(tr)
            #continue
            data[list(set(noise_indices))] = np.nan

        if np.isnan(tr.data).all():
            print(f"{tr.id} is completely NaN — skipping or flagging.")
            st_clean.remove(tr)
            continue

        def trim_trace_data(tr):
            """
            Trims leading/trailing NaNs or zeros from a Trace's data array
            and adjusts tr.stats.starttime accordingly.

            Modifies the Trace in-place.

            Returns:
            - True if trimmed
            - False if entire trace is invalid and should be discarded
            """
            data = tr.data
            valid = np.where((~np.isnan(data)) & (data != 0))[0]

            if valid.size == 0:
                return False  # all NaNs or zeros

            start_idx, end_idx = valid[0], valid[-1]
            tr.data = data[start_idx:end_idx + 1]

            # Adjust start time
            tr.stats.starttime += start_idx * tr.stats.delta
            return True


        success = trim_trace_data(tr)
        if not success or tr.stats.npts < num_samples // 2:
            print(f"{tr.id} has too few points after trimming — skipping.")
            st_clean.remove(tr)
            continue
        else:
            print(f"{tr.id} has {len(tr.data)} points after trimming. Interpolating...")
            # Interpolate gaps and zeros/NaNs
            bad_mask = ~np.isfinite(data) | (data == 0)
            if np.any(bad_mask):
                valid_idx = np.where(~bad_mask)[0]
                data = np.interp(np.arange(len(data)), valid_idx, data[valid_idx])
                tr.data = data

    startt = min(tr.stats.starttime for tr in st)
    endt = max(tr.stats.endtime for tr in st)
    for tr in st_clean:
        if tr.stats.starttime > startt or tr.stats.endtime < endt:
            print(f"{tr.id} is not complete for the expected time range — skipping.")
            st_clean.remove(tr)
            continue
    #st_clean.trim(starttime=startt, endtime=endt)
    return st_clean


def despike_and_extract_events(st, window=9, z_thresh=1.2, max_len=3):
    """
    Detects and removes short-duration spikes (e.g. rockfalls) from RSAM data,
    and extracts events that appear on multiple stations within a short window.

    Parameters:
    - st: ObsPy Stream
    - window: int, rolling window size for median filtering
    - z_thresh: float, threshold for spike detection
    - max_len: int, max allowed spike length in samples

    Returns:
    - Cleaned Stream with spikes removed
    - Event catalog as a list of dictionaries
    """
    st_clean = Stream()
    spike_times_by_station = defaultdict(list)
    for tr in st:
        data = tr.data.astype(np.float64)
        padded = np.pad(data, window, mode='reflect')
        median = np.array([np.median(padded[i:i+2*window+1]) for i in range(len(data))])
        std = np.array([np.std(padded[i:i+2*window+1]) for i in range(len(data))])
        spike_mask = (data - median) > z_thresh * std
        spike_regions = []
        in_spike = False
        for i, val in enumerate(spike_mask):
            if val and not in_spike:
                start_idx = i
                in_spike = True
            elif not val and in_spike:
                end_idx = i
                in_spike = False
                if 1 <= (end_idx - start_idx) <= max_len:
                    spike_regions.append((start_idx, end_idx))
        for start, end in spike_regions:
            spike_start_time = tr.stats.starttime + start * tr.stats.delta
            duration = end - start
            spike_times_by_station[tr.id].append((spike_start_time, duration))
        spike_mask_full = np.zeros(len(data), dtype=bool)
        for start, end in spike_regions:
            spike_mask_full[start:end] = True
        if np.any(spike_mask_full):
            good_idx = np.where(~spike_mask_full)[0]
            data = np.interp(np.arange(len(data)), good_idx, data[good_idx])
        tr_clean = tr.copy()
        tr_clean.data = data
        st_clean.append(tr_clean)
    event_catalog = []
    all_spikes = [(t, dur, sta) for sta, spikes in spike_times_by_station.items() for t, dur in spikes]
    all_spikes.sort()
    used = set()
    for i, (t1, dur1, sta1) in enumerate(all_spikes):
        if i in used:
            continue
        coincident = [(t1, dur1, sta1)]
        used.add(i)
        for j in range(i+1, len(all_spikes)):
            t2, dur2, sta2 = all_spikes[j]
            if abs(t2 - t1) <= 60:
                coincident.append((t2, dur2, sta2))
                used.add(j)
            else:
                break
        if len(coincident) >= 2:
            avg_time = min(t for t, _, _ in coincident)
            max_dur = max(d for _, d, _ in coincident)
            amps = []
            for t_evt, dur_evt, sta in coincident:
                tr_match = st.select(id=sta)[0]
                idx = int((t_evt - tr_match.stats.starttime) / tr_match.stats.delta)
                amp_vals = tr_match.data[idx:idx + dur_evt]
                if len(amp_vals) > 0:
                    amps.append(np.max(amp_vals))
            event_catalog.append({
                'time': avg_time,
                'duration_minutes': max_dur,
                'nstations': len(coincident),
                'stations': [s for _, _, s in coincident],
                'max_amplitude': max(amps) if amps else None,
                'median_amplitude': np.median(amps) if amps else None
            })
    return st_clean, event_catalog

def compute_geometric_mean_trace(st, trace_id="GEOMEAN"):
    """
    Computes the geometric mean of multiple aligned Trace objects in a Stream.

    Parameters:
    - st: ObsPy Stream
    - trace_id: Identifier for the new geometric mean trace

    Returns:
    - ObsPy Trace containing the geometric mean
    """
    data_matrix = np.array([tr.data for tr in st])
    safe_data = np.where(data_matrix <= 0, 1e-10, data_matrix)
    geom_mean = np.exp(np.mean(np.log(safe_data), axis=0))
    tr0 = st[0]
    network, station, location, channel = tr0.id.split('.')
    return Trace(
        data=geom_mean.astype(np.float32),
        header={
            'network': network, 'station': trace_id, 'location': '', 'channel': '',
            'starttime': tr0.stats.starttime, 'delta': tr0.stats.delta,
            'sampling_rate': tr0.stats.sampling_rate, 'npts': len(geom_mean)
        }
    )

def generate_rockfall_stream(event_catalog, template_stream):
    rock_stream = Stream()
    sampling_rate = template_stream[0].stats.sampling_rate
    delta = 1 / sampling_rate
    npts = len(template_stream[0].data)
    traces_by_station = {
        tr.id: Trace(data=np.zeros(npts), header=tr.stats.copy())
        for tr in template_stream
    }
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
                    tr.data[idx:idx_end] = amp
    for tr in traces_by_station.values():
        rock_stream.append(tr)
    return rock_stream

def plot_stream_with_triggers(st, triggers, outfile=None, lw=2):
    fig, axes = plt.subplots(len(st), 1, figsize=(7, len(st) * 2.5), sharex=True)
    if len(st) == 1:
        axes = [axes]
    for ax, tr in zip(axes, st):
        times_array = tr.times("matplotlib")
        ax.fill_between(times_array, tr.data, color='lightgray', alpha=0.6, step='mid')
        ax.plot(times_array, tr.data, 'k-', linewidth=0.8)
        ax.set_ylabel(f"{tr.id}", fontsize=8)
        for trig in triggers:
            ax.axvline(date2num(trig['time'].datetime), color='r', linestyle='--', linewidth=lw)
            ax.axvline(date2num(trig['endtime'].datetime), color='g', linestyle='--', linewidth=lw)
    #axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()

'''
def plot_sliding_hourly_counts_and_energy(event_catalog, window_minutes=60, step_minutes=1):
    """
    Plots sliding-hour rockfall counts and energy (amplitude^2) from an event catalog.

    Parameters:
    - event_catalog: List of rockfall events
    - window_minutes: Size of the sliding window in minutes
    - step_minutes: Step between each window in minutes
    """
    from datetime import timedelta
    catalog = sorted(event_catalog, key=lambda e: e['time'])
    all_times = [e['time'].datetime for e in catalog]
    t_min = min(all_times)
    t_max = max(all_times)
    step = timedelta(minutes=step_minutes)
    window = timedelta(minutes=window_minutes)
    time_steps = []
    t = t_min
    while t + window <= t_max:
        time_steps.append(t)
        t += step
    counts = []
    energies = []
    print(f"Time steps [{len(time_steps)}]: {time_steps}")
    for t0 in time_steps:
        t1 = t0 + window
        events = [e for e in catalog if t0 <= e['time'].datetime < t1]
        counts.append(len(events))
        energies.append(sum((e.get('max_amplitude') or 0)**2 for e in events))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.fill_between(time_steps, counts, color='skyblue', alpha=0.7)
    ax1.set_ylabel("Rockfall Count")
    ax1.set_title("Sliding-Hour Rockfall Count")
    ax2.fill_between(time_steps, energies, color='salmon', alpha=0.7)
    ax2.set_ylabel("Energy Proxy (Amplitude²)")
    ax2.set_title("Sliding-Hour Rockfall Energy")
    ax2.set_xlabel("Date (1h window, 1-min step)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.tight_layout()
    plt.show()
'''
def plot_summary_figure(rsam_stream, rockfall_stream, tremor_stream, triggers, rockfall_catalog, outfile=None):
    """
    Plot 5-panel summary figure with:
    1. Geometric mean of raw RSAM
    2. Geometric mean of rockfall stream
    3. Geometric mean of tremor stream + triggers
    4. Sliding hourly rockfall counts
    5. Sliding hourly rockfall energy

    Parameters:
    - rsam_stream: Stream of original RSAM traces
    - rockfall_stream: Stream with rockfall spikes only
    - tremor_stream: Stream used for STA/LTA detection
    - triggers: List of detected tremor triggers
    - rockfall_catalog: List of extracted rockfall events
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Geometric mean of raw RSAM
    geom_rsam = compute_geometric_mean_trace(rsam_stream)
    t = geom_rsam.times("matplotlib")
    axes[0].fill_between(t, geom_rsam.data, color='lightblue')
    axes[0].set_ylabel("Raw RSAM")
    axes[0].set_title("Geometric Mean - Raw RSAM")

    # Panel 2: Geometric mean of rockfall stream
    geom_rock = compute_geometric_mean_trace(rockfall_stream)
    t = geom_rock.times("matplotlib")
    axes[1].fill_between(t, geom_rock.data, color='orange')
    axes[1].set_ylabel("Rockfall")
    axes[1].set_title("Geometric Mean - Rockfall Stream")

    # Panel 3: Geometric mean of tremor stream
    geom_tremor = compute_geometric_mean_trace(tremor_stream)
    t = geom_tremor.times("matplotlib")
    axes[2].fill_between(t, geom_tremor.data, color='gray')
    for trig in triggers:
        axes[2].axvline(date2num(trig['time'].datetime), color='r', linestyle='--', lw=1)
        axes[2].axvline(date2num(trig['endtime'].datetime), color='g', linestyle='--', lw=1)
    axes[2].set_ylabel("Tremor")
    axes[2].set_title("Geometric Mean - Tremor Stream w/ Triggers")

    # Panel 4: Sliding hourly counts
    from datetime import timedelta
    catalog = sorted(rockfall_catalog, key=lambda e: e['time'])
    all_times = [e['time'].datetime for e in catalog]
    t_min, t_max = min(all_times), max(all_times)
    print(f"Time range: {t_min} to {t_max}")
    window, step = timedelta(hours=1), timedelta(minutes=1)
    time_bins = []
    t = t_min
    while t <= t_max:
        time_bins.append(t)
        t += step
    #print(f"Time bins [{len(time_bins)}]: {time_bins}")

    counts, energy = [], []
    for t0 in time_bins:
        t1 = min((t0 + window, t_max))
        window_events = [e for e in catalog if t0 <= e['time'].datetime < t1]
        counts.append(len(window_events))
        energy.append(sum((e.get('max_amplitude') or 0)**2 for e in window_events))
    axes[3].fill_between(time_bins, counts, color='skyblue')
    axes[3].set_ylabel("Count")
    axes[3].set_title("Sliding Hourly Rockfall Counts")

    # Panel 5: Sliding hourly energy
    axes[4].fill_between(time_bins, energy, color='salmon')
    axes[4].set_ylabel("Energy")
    axes[4].set_title("Sliding Hourly Rockfall Energy")
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()    


    tr = Trace()
    tr.data = np.array(energy, dtype=np.float64)
    tr.stats.starttime = time_bins[0]
    tr.stats.delta = 60.0   
    return tr

from obspy.core.event import Catalog, Event, Origin, OriginUncertainty, Amplitude, WaveformStreamID, ResourceIdentifier, CreationInfo

import uuid
from obspy.core.event import Amplitude, WaveformStreamID, Comment

def save_tremor_triggers_to_quakeml(triggers, output_path, stream=None, waveform_output_dir=None):
    """
    Saves tremor detections from STA/LTA to a QuakeML file, including:
    - Unique event ID
    - Maximum amplitude
    - Duration
    - Optionally: sliced waveform stream as MiniSEED file with URI reference

    Parameters:
    - triggers: List of trigger dictionaries with 'time' and 'endtime'
    - output_path: Path to save the QuakeML file
    - stream: Optional ObsPy Stream to extract amplitude and save waveforms
    - waveform_output_dir: Optional Path to directory where sliced waveforms are saved
    """
    cat = Catalog()

    for trig in triggers:
        event_id = str(uuid.uuid4())
        origin = Origin(
            time=trig['time'],
            origin_uncertainty=OriginUncertainty(),
            creation_info=CreationInfo(author="RSAM Trigger")
        )
        event = Event(
            resource_id=ResourceIdentifier(f"smi:local/rsam/{event_id}"),
            origins=[origin],
            creation_info=CreationInfo(author="RSAM Trigger")
        )

        # Extract amplitude and waveform
        if stream:
            t_start = trig['time']
            t_end = trig['endtime']
            trimmed = stream.slice(t_start, t_end)
            max_amp = max([np.max(tr.data) for tr in trimmed if len(tr.data)]) if len(trimmed) else 0.0

            amplitude = Amplitude(
                generic_amplitude=max_amp,
                type="maximum",
                unit="other",
                category="point",
                method_id=ResourceIdentifier("amplitude:rsam")
            )

            if waveform_output_dir:
                miniseed_path = waveform_output_dir / f"tremor_{event_id}.mseed"
                trimmed.write(str(miniseed_path), format="MSEED")

                amplitude.waveform_id = WaveformStreamID(
                    network_code="XX",
                    station_code="GEO",
                    channel_code="RSAM",
                    resource_uri=f"file://{miniseed_path}"
                )

            event.amplitudes.append(amplitude)

        # Add duration as a comment
        duration_sec = (trig['endtime'] - trig['time']) if 'endtime' in trig else 0
        event.comments.append(Comment(text=f"Duration: {duration_sec:.1f} seconds"))

        cat.events.append(event)

    cat.write(output_path, format="QUAKEML")

import uuid
from obspy.core.event import Amplitude, Comment

def save_rockfalls_to_quakeml(rockfall_catalog, output_path):
    """
    Saves rockfall detections to a QuakeML file with:
    - Unique event IDs
    - Max amplitude (as Amplitude object)
    - Duration and station count (as Comments)

    Parameters:
    - rockfall_catalog: List of event dicts from spike detection
    - output_path: Path to save QuakeML
    """
    cat = Catalog()

    for evt in rockfall_catalog:
        event_id = str(uuid.uuid4())
        origin = Origin(
            time=evt['time'],
            origin_uncertainty=OriginUncertainty(),
            creation_info=CreationInfo(author="RSAM Rockfall")
        )
        event = Event(
            resource_id=ResourceIdentifier(f"smi:local/rsam/{event_id}"),
            origins=[origin],
            creation_info=CreationInfo(author="RSAM Rockfall")
        )

        max_amp = evt.get('max_amplitude', 0.0)
        amplitude = Amplitude(
            generic_amplitude=max_amp,
            type="maximum",
            unit="other",
            category="point",
            method_id=ResourceIdentifier("amplitude:rsam")
        )
        event.amplitudes.append(amplitude)

        # Add duration and number of stations as comments
        duration_sec = evt.get('duration_minutes', 0) * 60
        nstations = evt.get('nstations', 1)
        event.comments.append(Comment(text=f"Duration: {duration_sec} seconds"))
        event.comments.append(Comment(text=f"Stations: {nstations}"))

        cat.events.append(event)

    cat.write(output_path, format="QUAKEML")


import matplotlib.pyplot as plt
from obspy.core.event import read_events
from pathlib import Path
import numpy as np

def plot_combined_detections(tremor_file, swarm_file, rockfall_file, outfile=None):
    """
    Plots start time, end time, and amplitude of tremor, swarm, and rockfall events.

    Parameters:
    - tremor_file: Path to QuakeML file with tremor detections
    - swarm_file: Path to QuakeML file with swarm detections
    - rockfall_file: Path to QuakeML file with rockfall detections
    - outfile: Optional path to save the plot
    """
    def parse_quakeml(file):
        events = read_events(str(file)).events
        onsets, offsets, amps = [], [], []
        for e in events:
            t_start = e.origins[0].time.datetime
            duration = 0
            for c in e.comments:
                if "Duration" in c.text:
                    duration = float(c.text.split(":")[1].split()[0])
            t_end = t_start + np.timedelta64(int(duration * 1000), 'ms')
            amp = e.amplitudes[0].generic_amplitude if e.amplitudes else 0.0
            onsets.append(t_start)
            offsets.append(t_end)
            amps.append(amp)
        return onsets, offsets, amps

    fig, ax = plt.subplots(figsize=(12, 6))

    if Path(tremor_file).exists():
        on, off, amp = parse_quakeml(tremor_file)
        for t1, t2, a in zip(on, off, amp):
            ax.plot([t1, t2], [a, a], color='gray', lw=2, label="Tremor" if 'Tremor' not in ax.get_legend_handles_labels()[1] else "")

    if Path(swarm_file).exists():
        on, off, amp = parse_quakeml(swarm_file)
        for t1, t2, a in zip(on, off, amp):
            ax.plot([t1, t2], [a, a], color='green', lw=2, label="Swarm" if 'Swarm' not in ax.get_legend_handles_labels()[1] else "")

    if Path(rockfall_file).exists():
        on, off, amp = parse_quakeml(rockfall_file)
        for t1, t2, a in zip(on, off, amp):
            ax.plot([t1, t2], [a, a], color='orange', lw=2, label="Rockfall" if 'Rockfall' not in ax.get_legend_handles_labels()[1] else "")

    ax.set_ylabel("Amplitude")
    ax.set_title("Detected Tremor, Swarm, and Rockfall Events")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d\n%H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()

def compute_energy_from_rockfalls(rockfall_catalog, starttime, endtime, bin_seconds=60):
    """
    Bins rockfall energy (amplitude squared) over time.

    Parameters:
    - rockfall_catalog: List of rockfall events with 'time' and 'max_amplitude'
    - starttime: UTCDateTime, start of binning interval
    - endtime: UTCDateTime, end of binning interval
    - bin_seconds: int, width of each bin in seconds

    Returns:
    - time_bins: list of UTCDateTime, start time of each bin
    - energy: list of float, total squared amplitude in each bin
    """
    num_bins = int((endtime - starttime) / bin_seconds)
    energy = np.zeros(num_bins, dtype=np.float64)
    time_bins = [starttime + i * bin_seconds for i in range(num_bins)]

    for evt in rockfall_catalog:
        t = evt.get('time')
        amp = evt.get('max_amplitude', 0.0) or 0.0
        if not (starttime <= t < endtime):
            continue
        idx = int((t - starttime) / bin_seconds)
        if 0 <= idx < num_bins:
            energy[idx] += amp ** 2

    tr = Trace()
    tr.data = np.array(energy, dtype=np.float64)
    tr.stats.starttime = time_bins[0]
    tr.stats.delta = bin_seconds
    return tr

def detect_triggers_to_stream(stream, triggers, reference_trace=None):
    """
    Applies triggers to create an output stream with the same structure,
    inserting detected amplitude during triggered periods and zero elsewhere.

    Parameters:
    - stream: ObsPy Stream used to detect triggers
    - triggers: List of trigger dictionaries with 'time' and 'endtime'
    - reference_trace: Optional Trace to use as amplitude source (e.g. geometric mean)

    Returns:
    - ObsPy Stream containing masked data in trigger windows
    """
    from obspy import Stream

    if reference_trace is None:
        reference_trace = compute_geometric_mean_trace(stream)

    tr_template = reference_trace.copy()
    tr_template.data[:] = 0.0

    for trig in triggers:
        trig['endtime'] = trig['time'] + trig['duration']
        idx0 = int((trig['time'] - tr_template.stats.starttime) / tr_template.stats.delta)
        idx1 = int((trig['endtime'] - tr_template.stats.starttime) / tr_template.stats.delta)
        tr_template.data[idx0:idx1] = reference_trace.data[idx0:idx1]

    return Stream([tr_template])

def rsam_decompose(rsam, use_geom_mean=True, sta=1200, lta=7200):
    """
    Decomposes an RSAM object into separate Streams representing:
    - Short-duration events (rockfalls)
    - Long-duration tremor
    - Energy-based swarms

    Parameters:
    - rsam: RSAM object
    - use_geom_mean: bool, whether to detect tremor/swarm using geometric mean trace
    - sta: int, STA window in seconds
    - lta: int, LTA window in seconds

    Returns:
    - Tuple of (rockfall_stream, tremor_stream, swarm_stream) as ObsPy Streams
    """
    from obspy.signal.trigger import coincidence_trigger

    stream = rsam.to_stream()
    stream = sanitize_stream(stream)

    # Step 1: Rockfall detection
    tremor_stream, rockfall_catalog = despike_and_extract_events(stream)
    rockfall_stream = generate_rockfall_stream(rockfall_catalog, stream)

    # Step 2: Tremor detection using STA/LTA
    geom_tremor_trace = compute_geometric_mean_trace(tremor_stream) if use_geom_mean else tremor_stream
    geom_tremor_stream = Stream(traces=[geom_tremor_trace])
    this_stream = geom_tremor_stream if use_geom_mean else tremor_stream
    triggers = coincidence_trigger(
        trigger_type="classicstalta",
        stream=this_stream,
        thr_on=1.8,
        thr_off=1.0,
        sta=sta,
        lta=lta,
        thr_coincidence_sum=min((len(stream), 3)),
        details=True
    )
    tremor_stream = detect_triggers_to_stream(this_stream, triggers, reference_trace=geom_tremor_trace)

    # Step 3: Swarm detection from energy
    energy_trace = compute_energy_from_rockfalls(rockfall_catalog, stream[0].stats.starttime, stream[0].stats.endtime)
    swarm_triggers = detect_energy_swarm(energy_trace, sta_seconds=sta, lta_seconds=lta, sampling_interval=60)
    swarm_stream = detect_triggers_to_stream(Stream(traces=[energy_trace]), swarm_triggers, reference_trace=energy_trace)

    return rockfall_stream, tremor_stream, swarm_stream


def main(startt, endt, BINARY_DIR, OUTPUT_DIR):
    
    STA_SECONDS = 1200
    LTA_SECONDS = 7200
    expanded_start = startt - LTA_SECONDS
    expanded_end = endt + LTA_SECONDS

    miniseedfile = OUTPUT_DIR / f"{startt.strftime('%Y%m%d')}_raw.miniseed"
    if miniseedfile.exists():
        st = read(str(miniseedfile))
        rsamObj = None
    else:
        files = list(BINARY_DIR.glob(f'M???{startt.year}.DAT'))
        stations = sorted(set(path.name[:4] for path in files))
        print(f"Processing {len(stations)} stations: {stations}")

        rsamObj = RSAM.readRSAMbinary(str(BINARY_DIR), stations, expanded_start, expanded_end, verbose=False, convert_legacy_ids_using_this_network='MV')
        if len(rsamObj) > 0:
            st = rsamObj.to_stream()
            st.write(str(miniseedfile), format='MSEED')
        else:
            st = Stream()

    if len(st) == 0:
        print(f"No data found for {startt} to {endt}")
        return
    print(f"Raw stream: {st}")
    st.plot(equal_scale=False, outfile=str(miniseedfile).replace('_raw.miniseed', '_raw.png'))

    st = sanitize_stream(st)
    if len(st) > 0:
        st = sanitize_stream(st)
    if len(st) == 0:
        print(f"No data found after sanitization for {startt} to {endt}")
        return

    plot_stream_with_triggers(st, [], outfile=str(miniseedfile).replace('_raw.miniseed', '_sanitized.png'))
    tremor_stream, rockfall_catalog = despike_and_extract_events(st)
    tremor_stream.plot(equal_scale=False, outfile=str(miniseedfile).replace('_raw.miniseed', '_tremor.png'))
    geom_tremor_tr = compute_geometric_mean_trace(tremor_stream)
    tremor_stream.write(str(miniseedfile).replace('_raw.miniseed', '_tremor.miniseed'), format='MSEED')

    triggers = coincidence_trigger(
        trigger_type="classicstalta",
        stream=tremor_stream,
        thr_on=1.8,
        thr_off=1.0,
        sta=STA_SECONDS,
        lta=LTA_SECONDS,
        thr_coincidence_sum=min((len(st), 3)),
        details=True
    )
    for trig in triggers:
        trig['endtime'] = trig['time'] + trig['duration']

    plot_stream_with_triggers(Stream([geom_tremor_tr]), triggers, outfile=str(miniseedfile).replace('_raw.miniseed', '_geom_tremor_triggers.png'))

    rockfall_stream = generate_rockfall_stream(rockfall_catalog, st)
    rockfall_stream.write(str(miniseedfile).replace('_raw.miniseed', '_events.miniseed'), format='MSEED')
    rockfall_stream.plot(equal_scale=False, outfile=str(miniseedfile).replace('_raw.miniseed', '_events.png'))
    rockfall_geom_tr = compute_geometric_mean_trace(rockfall_stream, trace_id="RF")
    rockfall_geom_tr.plot(equal_scale=False, outfile=str(miniseedfile).replace('_raw.miniseed', '_events_geom.png'))

    energy_trace = plot_summary_figure(
        rsam_stream=st,
        rockfall_stream=rockfall_stream,
        tremor_stream=tremor_stream,
        triggers=triggers,
        rockfall_catalog=rockfall_catalog,
        outfile=str(miniseedfile).replace('_raw.miniseed', '_summary.png')
    )


    # Energy-based swarm detection
    swarm_triggers = detect_energy_swarm(energy_trace, sta_seconds=STA_SECONDS, lta_seconds=LTA_SECONDS)
    for trig in swarm_triggers:
        trig['endtime'] = trig['time'] + trig['duration']

    save_tremor_triggers_to_quakeml(swarm_triggers, str(miniseedfile).replace('_raw.miniseed', "_swarm_detections.xml"))

    save_tremor_triggers_to_quakeml(triggers, str(miniseedfile).replace('_raw.miniseed', "_tremor_detections.xml"))
    save_rockfalls_to_quakeml(rockfall_catalog, str(miniseedfile).replace('_raw.miniseed', "_event_detections.xml"))
    
    plot_combined_detections(
        tremor_file="merged_tremor_detections.xml",
        swarm_file="merged_swarm_detections.xml",
        rockfall_file="merged_rockfall_detections.xml",
        outfile="all_detections_timeseries.png"
    )

    if rsamObj:
        rockfall_stream, tremor_stream, swarm_stream = rsam_decompose(rsamObj, use_geom_mean=True, sta=1200, lta=7200)
        st_final=Stream(traces=[rockfall_stream[0], tremor_stream[0], swarm_stream[0]])
        st_final.plot(equal_scale=False, outfile=str(miniseedfile).replace('_raw.miniseed', '_final_stream.png'))

def detect_energy_swarm(tr, sta_seconds=1200, lta_seconds=7200, sampling_interval=60):
    """
    Applies a classic STA/LTA detector to binned energy time series (e.g. from rockfalls)
    and returns a list of trigger dictionaries with 'time', 'endtime', and 'duration'.

    Parameters:
    - tr: Trace object
    - sta_seconds: int, short-term average window in seconds
    - lta_seconds: int, long-term average window in seconds
    - sampling_interval: int, number of seconds per energy bin

    Returns:
    - List of trigger dictionaries
    """
    from obspy import Trace
    from obspy.signal.trigger import classic_sta_lta, trigger_onset

    cft = classic_sta_lta(tr.data, int(sta_seconds / sampling_interval), int(lta_seconds / sampling_interval))
    on_off = trigger_onset(cft, 1.8, 1.0)

    triggers = []
    for onset, offset in on_off:
        t_on = tr.stats.starttime + onset * tr.stats.delta
        t_off = tr.stats.starttime + offset * tr.stats.delta
        triggers.append({
            "time": t_on,
            "endtime": t_off,
            "duration": (t_off - t_on)
        })

    return triggers

def merge_quakeml_files(directory, pattern, output_filename="merged_catalog.xml", threshold_seconds=29):
    from obspy.core.event import read_events, Catalog
    from obspy import UTCDateTime
    from pathlib import Path

    all_files = sorted(Path(directory).glob(pattern))
    all_events = []

    for file in all_files:
        try:
            cat = read_events(str(file))
            all_events.extend(cat.events)
        except Exception as e:
            print(f"Could not read {file}: {e}")

    print(f"Merging {len(all_events)} events from {len(all_files)} files")
    all_events.sort(key=lambda e: e.origins[0].time)

    merged = Catalog()
    for evt in all_events:
        if len(merged) == 0:
            merged.events.append(evt)
        else:
            last_evt = merged.events[-1]
            t1 = last_evt.origins[0].time
            t2 = evt.origins[0].time
            if abs(t2 - t1) > threshold_seconds:
                merged.events.append(evt)

    output_path = Path(directory) / output_filename
    merged.write(str(output_path), format="QUAKEML")
    print(f"Merged catalog saved to {output_path}")

if __name__ == '__main__':
    from obspy import UTCDateTime
    BINARY_DIR = Path('/Volumes/DATA/Montserrat/ASN/RSAM/RSAM_1')
    OUTPUT_DIR = Path('/Users/GlennThompson/Dropbox')
    start_date = UTCDateTime(1996, 8, 1)
    end_date = start_date + 86400 * 3
    current_date = start_date
    while current_date < end_date:
        main(current_date, current_date + 86400, BINARY_DIR, OUTPUT_DIR)
        current_date += 86400

    merge_quakeml_files(OUTPUT_DIR, pattern="*_tremor_detections.xml", output_filename="merged_tremor_detections.xml")
    merge_quakeml_files(OUTPUT_DIR, pattern="*_event_detections.xml", output_filename="merged_rockfall_detections.xml")
    merge_quakeml_files(OUTPUT_DIR, pattern="*_swarm_detections.xml", output_filename="merged_swarm_detections.xml")