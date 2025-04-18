# Core scientific stack
import numpy as np
import pandas as pd
import random

# GUI and plotting
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



# ObsPy
from obspy import read, Stream, UTCDateTime
from obspy.signal.trigger import (
    coincidence_trigger,
    z_detect,
    classic_sta_lta,
    recursive_sta_lta,
    delayed_sta_lta,
    carl_sta_trig,
    trigger_onset
)

# Custom or local functions assumed from other modules
from flovopy.processing.metrics import (
    estimate_snr,
    compute_amplitude_ratios,
    plot_amplitude_ratios,
)

from flovopy.core.preprocessing import _pad_trace

from flovopy.core.enhanced import get_bandwidth


# for picker - everything else is for detection
from obspy.core import read, UTCDateTime, Stream
from obspy.signal.trigger import pk_baer, ar_pick, aic_simple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class SeismicGUI:
    def __init__(self, master, stream):
        self.master = master
        self.master.title("Seismic Event Detection")

        self.stream = stream
        self.selected_traces = stream.copy()  # Default: All traces selected
        self.picked_times = []
        self.mode = "select_traces"

        self.frame_left = tk.Frame(master)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.vars = []
        self.checkboxes = []
        for i, tr in enumerate(stream):
            var = tk.BooleanVar(value=True)  # Default to checked
            chk = tk.Checkbutton(self.frame_left, text=tr.id, variable=var)
            chk.pack(anchor="w")
            self.checkboxes.append(chk)
            self.vars.append(var)

        self.fig, self.axs = plt.subplots(len(stream), 1, figsize=(10, 6), sharex=True)
        plt.subplots_adjust(hspace=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.plot_traces()

        self.button = tk.Button(self.frame_left, text="Select Traces", command=self.select_traces)
        self.button.pack(pady=10)

        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

    def plot_traces(self):
        """Plots traces separately in linked subplots, ensuring correct time scaling and linking."""
        for ax in self.axs:
            ax.clear()

        start_time = min(tr.stats.starttime for tr in self.stream)
        end_time = max(tr.stats.endtime for tr in self.stream)

        start_time_num = mdates.date2num(start_time.datetime)
        end_time_num = mdates.date2num(end_time.datetime)

        if self.mode == "select_traces":
            self.fig.suptitle("Select traces using checkboxes, then click 'Select Traces'")

            for i, tr in enumerate(self.stream):
                absolute_times = mdates.date2num([start_time + t for t in tr.times()])
                data_norm = tr.data / max(abs(tr.data)) if max(abs(tr.data)) != 0 else tr.data
                self.axs[i].plot(absolute_times, data_norm, label=tr.id)
                self.axs[i].legend(loc="upper right")
                self.axs[i].set_ylabel(tr.id)

            self.axs[-1].set_xlabel("Time (UTC)")

        elif self.mode == "pick_times":
            self.fig.suptitle("Click to select event start and end times")

            for i, tr in enumerate(self.selected_traces):
                absolute_times = mdates.date2num([tr.stats.starttime + t for t in tr.times()])
                data_norm = tr.data / max(abs(tr.data)) if max(abs(tr.data)) != 0 else tr.data
                self.axs[i].plot(absolute_times, data_norm, label=tr.id)
                self.axs[i].legend(loc="upper right")
                self.axs[i].set_ylabel(tr.id)

            self.axs[-1].set_xlabel("Time (UTC)")

            self.axs[0].set_xlim(start_time_num, end_time_num)

            self.cursor_lines = [ax.axvline(x=start_time_num, color='r', linestyle='dotted', lw=1) for ax in self.axs]

        for ax in self.axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))

        self.canvas.draw()

    def select_traces(self):
        """Handles trace selection and switches to event picking mode."""
        self.selected_traces = [self.stream[i] for i, var in enumerate(self.vars) if var.get()]

        if not self.selected_traces:
            messagebox.showerror("Error", "Please select at least one trace!")
            return

        self.mode = "pick_times"
        self.button.config(text="Pick Start and End Times", state=tk.DISABLED)
        self.plot_traces()

    def on_mouse_move(self, event):
        """Move dotted cursor in pick_times mode across all subplots."""
        if self.mode == "pick_times" and event.xdata:
            for cursor_line in self.cursor_lines:
                if not isinstance(event.xdata, list): # is not None:  # Ensure valid xdata
                    cursor_line.set_xdata([event.xdata]) 
                else:
                    cursor_line.set_xdata(event.xdata)
            self.canvas.draw()

    def on_mouse_click(self, event):
        """Handles event start and end time selection."""
        if self.mode == "pick_times" and event.xdata:
            picked_time = UTCDateTime(mdates.num2date(event.xdata))
            for ax in self.axs:
                ax.axvline(x=event.xdata, color='r', linestyle='solid', lw=2)

            self.picked_times.append(picked_time)
            print(f" Picked event time: {picked_time}")

            if len(self.picked_times) == 1:
                self.fig.suptitle("Click to select event end time")
            elif len(self.picked_times) == 2:
                plt.close(self.fig)  #  Closes only the current figure
                plt.close('all')  #  Ensures all figures are closed                
                self.master.quit()

            self.canvas.draw()

def run_monte_carlo(stream, event_start, event_end, n_trials=5000, verbose=False, max_allowed_misfit=4.0):
    """Runs Monte Carlo simulations with different trigger parameters."""
    print('Finding best autodetect parameters by running Monte Carlo simulations...')
    algorithms = ["recstalta", "classicstalta", "zdetect", "carlstatrig", "delayedstalta"]
    # classicstalta: Classic STA/LTA algorithm
        # classic_sta_lta(tr.data, nsta, nlta) nsta is number of samples
    # recstalta: Recursive STA/LTA algorithm
        # recursive_sta_lta(tr.data, nsta, nlta)
    # zdetect: Z-detect algorithm
        # z_detect(tr.data, nsta) https://docs.obspy.org/packages/autogen/obspy.signal.trigger.z_detect.html#obspy.signal.trigger.z_detect
    # carlstatrig: Carl-Sta-Trig algorithm
        # carl_sta_trig(tr.data, nsta, nlta, ratio, quiet) https://docs.obspy.org/packages/autogen/obspy.signal.trigger.carl_sta_trig.html#obspy.signal.trigger.carl_sta_trig
    # delayed_sta_lta: Delayed STA/LTA algorithm
        # delayed_sta_lta(tr.data, nsta, nlta)
    best_params = None
    trials_complete = 0
    lod = []
    minchans = len(stream)
    best_misfit = np.Inf
    stream_duration = stream[0].stats.endtime - stream[0].stats.starttime
    while trials_complete < n_trials and best_misfit > max_allowed_misfit:
        algorithm = random.choice(algorithms)
        sta = random.uniform(0.1, 5)
        if sta > stream_duration/20:
            sta = stream_duration/20       
        lta = random.uniform(sta * 4, sta * 25)
        if lta > stream_duration/2:
            lta = stream_duration/2
        thr_on = random.uniform(1.5, 5)
        thr_off = random.uniform(thr_on / 10, thr_on / 2)
        #ratio = random.uniform(1, 2)
        best_trig=None
        if verbose:
            print(f"Trying {algorithm} with STA={sta}, LTA={lta}, THR_ON={thr_on}, THR_OFF={thr_off}")

        # Run coincidence_trigger
        if True:
            best_trig = detect_network_event(stream, minchans=minchans, threshon=thr_on, threshoff=thr_off, \
                         sta=sta, lta=lta, pad=0.0, best_only=True, verbose=False, freq=None, algorithm=algorithm, criterion='cft')

        if not best_trig:   # No triggers found
            if verbose:
                print("No triggers found.")
            continue
        
        trials_complete += 1
        best_trig["algorithm"] = algorithm
        best_trig["sta"] = sta
        best_trig["lta"] = lta
        best_trig["thr_on"] = thr_on
        best_trig["thr_off"] = thr_off
        best_trig['endtime'] = best_trig["time"] + best_trig["duration"]
        best_trig['misfit'] = abs(best_trig['time'] - event_start) + abs(best_trig['endtime'] - event_end)
        best_trig['event_start'] = event_start
        best_trig['event_end'] = event_end
        lod.append(best_trig)
        if trials_complete % 10 == 0:
            df = pd.DataFrame(lod)
            best_params = df.loc[df['misfit'].idxmin()].to_dict()
            best_misfit = best_params['misfit']
            print(f'{trials_complete} of {n_trials} trials complete, lowest misfit is {best_misfit}')

    df = pd.DataFrame(lod)
    best_params = df.loc[df['misfit'].idxmin()].to_dict()
    best_misfit = best_params['misfit']
  
    return df, best_params

def run_event_detection(stream, n_trials=50):
    """Runs GUI and returns selected traces, event times, and best parameters."""
    root = tk.Tk()
    app = SeismicGUI(root, stream)
    root.mainloop()

    if len(app.picked_times) != 2:
        print("⚠️ Error: No valid event times selected.")
        return None, None, None, None

    event_start, event_end = app.picked_times
    out_stream = Stream(traces=app.selected_traces)
    plt.close('all')  # Ensures all figures are closed
    root.withdraw()  # Hide the main Tkinter window
    root.quit()  # Quit the Tkinter mainloop
    root.destroy()  # Forcefully close all Tk windows    
    df, best_params = run_monte_carlo(out_stream, event_start, event_end, n_trials)


    return out_stream, best_params, df

def plot_detected_stream(detected_st, best_trig, outfile=None):
    """Plots the detected stream with trigger start and end times."""

    # plot detected stream
    trig_time = best_trig['time'].matplotlib_date
    trig_end = (best_trig['time'] + best_trig['duration']).matplotlib_date
    fig = detected_st.plot(show=False, equal_scale=False)  # show=False prevents it from auto-displaying
    for ax in fig.axes:  # Stream.plot() returns multiple axes (one per trace)
        ax.axvline(trig_time, color='r', linestyle='--', label="Trigger Start")
        ax.axvline(trig_end, color='b', linestyle='--', label="Trigger End")
        ax.legend()
    if outfile:  
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def detection_snr(detected_st, best_trig, outfile=None, method='std',
                 spectral_smooth=5, spectral_average='geometric',
                 spectral_threshold=0.5, freq_band=None):
    """
    Estimate time-domain and spectral-domain SNR from a triggered waveform stream.

    Parameters
    ----------
    detected_st : obspy.Stream
        The full stream of waveform data containing signal and noise.
    best_trig : dict
        Dictionary with keys 'time' (UTCDateTime) and 'duration' (float).
    outfile : str, optional
        If provided, a spectral amplitude ratio plot will be saved to this file.
    method : str, optional
        Time-domain SNR method: 'std', 'max', 'rms', or 'spectral'.
    spectral_smooth : int, optional
        Moving average window for smoothing amplitude ratios.
    spectral_average : str, optional
        Spectral averaging method: 'geometric', 'mean', 'median'.
    spectral_threshold : float, optional
        Bandwidth threshold as fraction of peak for get_bandwidth().
    freq_band : tuple(float, float), optional
        Frequency band (Hz) to average spectral SNR over.

    Returns
    -------
    snr_list : list of float
        SNR values computed using the specified method.
    fmetrics_dict : dict or None
        Bandwidth metrics from the average spectral ratio, or None if unavailable.
    """
    trig_stime = best_trig['time']
    trig_etime = trig_stime + best_trig['duration']

    # Split into signal and noise windows
    signal_st = detected_st.copy().trim(starttime=trig_stime, endtime=trig_etime)
    noise_st = detected_st.copy().trim(endtime=trig_stime)

    # Estimate time-domain SNR
    snr_list, _, _ = estimate_snr(
        detected_st, method=method, split_time=(trig_stime, trig_etime),
        spectral_kwargs={
            'smooth_window': spectral_smooth,
            'average': spectral_average
        },
        freq_band=freq_band
    )

    # Estimate spectral amplitude ratios
    avg_freqs, avg_spectral_ratio, indiv_ratios, freqs_list, trace_ids, _ = compute_amplitude_ratios(
        signal_st, noise_st,
        smooth_window=spectral_smooth,
        average=spectral_average,
        verbose=True
    )

    # Optional plot
    if isinstance(avg_freqs, np.ndarray) and isinstance(avg_spectral_ratio, np.ndarray):
        if outfile:
            plot_amplitude_ratios(
                avg_freqs, avg_spectral_ratio,
                individual_ratios=indiv_ratios,
                freqs_list=freqs_list,
                trace_ids=trace_ids,
                log_scale=True,
                outfile=outfile,
                threshold=spectral_threshold
            )

        # Bandwidth metrics
        fmetrics_dict = get_bandwidth(avg_freqs, avg_spectral_ratio, threshold=spectral_threshold)
        return snr_list, fmetrics_dict

    else:
        return snr_list, None
  
def real_time_optimization(band='all'):
    corners = 2
    if band=='VT':
        # VT + false
        sta_secs = 1.4
        lta_secs = 7.0
        threshON = 2.4
        threshOFF = 1.2
        freqmin = 3.0
        freqmax = 18.0
    elif band=='LP':
        # LP + false
        sta_secs = 2.3
        lta_secs = 11.5
        threshON = 2.4
        threshOFF = 1.2
        freqmin = 0.8
        freqmax = 10.0        
    elif band=='all':
        # all = LP + VT + false
        sta_secs = 2.3
        lta_secs = 11.5
        threshON = 2.4
        threshOFF = 1.2
        freqmin = 1.5
        freqmax = 12.0        
    threshOFF = threshOFF / threshON
        
    return sta_secs, lta_secs, threshON, threshOFF, freqmin, freqmax, corners


def detect_network_event(st_in, minchans=None, threshon=3.5, threshoff=1.0, 
                         sta=0.5, lta=5.0, pad=0.0, best_only=False, verbose=False, 
                         freq=None, algorithm='recstalta', criterion='longest'):
    """
    Detects and associates seismic events across a network of stations using coincidence triggering.

    This function applies a short-term average / long-term average (STA/LTA) or other 
    triggering algorithm to detect seismic events recorded across multiple stations. 
    It identifies events based on a coincidence threshold and returns a list of detected 
    triggers, including event start time, duration, number of stations that triggered, and 
    characteristic function peak values.

    If `best_only=True`, the function returns only the most significant event based on the 
    specified `criterion`. Otherwise, it returns all detected events along with onset and 
    offset times.

    Parameters:
    ----------
    st_in : obspy.Stream
        The input Stream object containing waveform data from multiple stations.
    minchans : int, optional
        Minimum number of stations required to trigger an event (default: half the number of traces in `st_in`, minimum 2).
    threshon : float, optional
        Trigger threshold for STA/LTA or other detection algorithm (default: 3.5).
    threshoff : float, optional
        De-trigger threshold for STA/LTA or other detection algorithm (default: 1.0).
    sta : float, optional
        Short-term average window length in seconds (default: 0.5s).
    lta : float, optional
        Long-term average window length in seconds (default: 5.0s).
    pad : float, optional
        Time (in seconds) to pad the traces before processing (default: 0.0s, no padding).
    best_only : bool, optional
        If `True`, returns only the most significant detected event based on `criterion` (default: False).
    verbose : bool, optional
        If `True`, prints debug information during processing (default: False).
    freq : list of float, optional
        If specified, applies a bandpass filter before processing using `[freq_min, freq_max]` in Hz (default: None).
    algorithm : str, optional
        The detection algorithm to use. Options: `"recstalta"`, `"zdetect"`, `"carlstatrig"` (default: `"recstalta"`).
    criterion : str, optional
        Criterion for selecting the "best" event when `best_only=True`:
        - `"longest"` (default): Selects the event with the highest product of `coincidence_sum * duration`.
        - `"cft"`: Selects the event with the highest sum of `cft_peaks`.
        - `"cft_duration"`: Uses `cft_peaks * duration` to determine the best event.

    Returns:
    -------
    If `best_only=True`:
        dict or None
            A dictionary containing information about the most significant detected event, 
            or `None` if no event was found.

    If `best_only=False`:
        tuple:
            - **trig (list of dicts)**: Each dict contains information about a detected event, 
              with keys including:
              ```
              {
                  'cft_peak_wmean': float,   # Mean peak characteristic function value
                  'cft_peaks': list,         # Characteristic function peak values for each station
                  'cft_std_wmean': float,    # Mean standard deviation of characteristic function
                  'cft_stds': list,          # Standard deviation values per station
                  'coincidence_sum': float,  # Number of stations triggering simultaneously
                  'duration': float,         # Event duration (s)
                  'stations': list,          # Stations that contributed to the event
                  'time': UTCDateTime,       # Event start time
                  'trace_ids': list          # List of trace IDs contributing to the event
              }
              ```
            - **ontimes (list of UTCDateTime)**: List of event start times.
            - **offtimes (list of UTCDateTime)**: List of event end times.

    If no events are detected:
        - If `best_only=True`: Returns `None`.
        - If `best_only=False`: Returns `(None, None, None)`.

    Notes:
    ------
    - If `sta` and `lta` are chosen such that `lta` is too long relative to the signal, 
      event detection may fail.
    - The function applies optional bandpass filtering if `freq` is specified.
    - The `pad` option allows extending the trace duration before filtering or triggering.
    - Detected events can be further refined using `trim_to_event()`.

    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data from multiple stations
    st = read("network_data.mseed")

    # Run event detection
    events, onsets, offsets = detect_network_event(st, freq=[1, 20], verbose=True)

    if events:
        print(f"Detected {len(events)} events")
        for event in events:
            print(f"Event at {event['time']} with duration {event['duration']} s")

    # Retrieve only the best event
    best_event = detect_network_event(st, best_only=True, criterion="cft")
    print(f"Best event: {best_event}")
    ```
    """
    st = st_in.copy()
    if pad>0.0:
        for tr in st:
            _pad_trace(tr, pad)

    if freq:
        if verbose:
            print('Filtering traces')
        st.filter('bandpass', freqmin=freq[0], freqmax=freq[1], corners=4, zerophase=True)
            
    if not minchans:
        minchans = max(( int(len(st)/2), 2)) # half the channels or 2, whichever is greater
    if verbose:
        print('minchans=',minchans)
    #trig = coincidence_trigger(algorithm, threshon, threshoff, st, minchans, sta=sta, lta=lta, max_trigger_length=180, delete_long_trigger=True, details=True) # 0.5s, 10s
    if algorithm == "zdetect":
        trig = coincidence_trigger(algorithm, threshon, threshoff, st, minchans, sta=sta, details=True)
    elif algorithm == "carlstatrig":
        trig = coincidence_trigger(algorithm, threshon, threshoff, st, minchans, sta=sta, lta=lta, ratio=1, quiet=True, details=True)
    else:
        trig = coincidence_trigger(algorithm, threshon, threshoff, st, minchans, sta=sta, lta=lta, details=True)
    if trig:

        if best_only:
            best_trig = {}
            best_product = 0

            for this_trig in trig:
                #print(this_trig)
                thistime = UTCDateTime(this_trig['time'])
                if thistime > st[0].stats.starttime:
                    if criterion=='longest':
                        this_product = this_trig['coincidence_sum']*this_trig['duration']
                    elif criterion=='cft':
                        this_product = sum(this_trig["cft_peaks"])
                    else:
                        this_product = sum(this_trig["cft_peaks"])*this_trig['duration']
                    if this_product > best_product:
                        best_trig = this_trig
                        best_product = this_product
            return best_trig  
        else:
            ontimes = []
            offtimes = []
            for this_trig in trig:
                thistime = UTCDateTime(this_trig['time'])
                if thistime > st[0].stats.starttime:
                    ontimes.append(this_trig['time'])
                    offtimes.append(this_trig['time']+this_trig['duration'])
            return trig, ontimes, offtimes
    else:
        if best_only:
            return None
        else:
            return None, None, None
    
def add_channel_detections(st, lta=5.0, threshon=0.5, threshoff=0.0, max_duration=120):
    """
    Runs single-channel event detection on each Trace in a Stream using the STA/LTA method.

    This function applies a `z_detect` characteristic function to each trace and detects 
    events based on STA/LTA triggering. It does **not** perform multi-station coincidence 
    triggering or event association.

    Detected triggers are stored in `tr.stats.triggers` as a list of two-element numpy arrays, 
    where each array contains the **trigger on** and **trigger off** times as `UTCDateTime` objects.

    Parameters:
    ----------
    st : obspy.Stream
        The input Stream object containing multiple seismic traces.
    lta : float, optional
        Long-term average (LTA) window length in seconds (default: 5.0s).
    threshon : float, optional
        Trigger-on threshold for detection (default: 0.5).
    threshoff : float, optional
        Trigger-off threshold for ending a detection (default: 0.0).
    max_duration : float, optional
        Maximum event duration in seconds before forcing a trigger-off (default: 120s).

    Returns:
    -------
    None
        The function modifies `tr.stats.triggers` in-place for each trace.

    Stores:
    -------
    - `tr.stats.triggers` : list of lists
        Each trace will have a list of **trigger onset and offset times** as `UTCDateTime` pairs.
        Example:
        ```python
        tr.stats.triggers = [
            [UTCDateTime("2025-03-10T12:00:00"), UTCDateTime("2025-03-10T12:00:15")],
            [UTCDateTime("2025-03-10T12:30:05"), UTCDateTime("2025-03-10T12:30:20")]
        ]
        ```

    Notes:
    ------
    - If `lta=5.0`, at least **5 seconds of noise** before the signal is required.
    - This function only detects events **on individual channels**—it does not associate detections across stations.
    - Uses `z_detect()` as the characteristic function and `trigger_onset()` to extract events.

    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data
    st = read("example.mseed")

    # Run single-channel detections
    add_channel_detections(st, lta=5.0, threshon=0.5, threshoff=0.1, max_duration=60)

    # Print detected triggers
    for tr in st:
        print(f"Triggers for {tr.id}: {tr.stats.triggers}")
    ```
    """
    for tr in st:
        tr.stats['triggers'] = []
        Fs = tr.stats.sampling_rate
        cft = z_detect(tr.data, int(lta * Fs)) 
        triggerlist = trigger_onset(cft, threshon, threshoff, max_len = max_duration * Fs)
        for trigpair in triggerlist:
            trigpairUTC = [tr.stats.starttime + samplenum/Fs for samplenum in trigpair]
            tr.stats.triggers.append(trigpairUTC)

def get_event_window(st, pretrig=30, posttrig=30):
    """
    Determines the time window encompassing all detected triggers in a Stream, with optional pre/post-event padding.

    This function scans all traces in the given `Stream`, extracts the earliest **trigger-on**
    and the latest **trigger-off** time from `tr.stats.triggers`, and returns a **time window**
    that includes an additional buffer (`pretrig` and `posttrig`) before and after the event.

    **Assumptions:**
    - The function assumes that `add_channel_detections(st, ...)` has already been run,
      so that each `Trace` contains `tr.stats.triggers`.
    - If no triggers are found, the function returns `(None, None)`.

    Parameters:
    ----------
    st : obspy.Stream
        The input Stream object containing seismic traces with precomputed triggers.
    pretrig : float, optional
        Extra time in seconds to include **before** the first trigger-on time (default: 30s).
    posttrig : float, optional
        Extra time in seconds to include **after** the last trigger-off time (default: 30s).

    Returns:
    -------
    tuple:
        - **mintime (UTCDateTime)**: The earliest trigger-on time minus `pretrig` seconds.
        - **maxtime (UTCDateTime)**: The latest trigger-off time plus `posttrig` seconds.
        - If no triggers exist, returns `(None, None)`.

    Notes:
    ------
    - The function uses the **median** trigger times (`N/2` index) instead of min/max to reduce
      sensitivity to outliers.
    - The returned `mintime` and `maxtime` can be used with `trim_to_event()` to extract
      the event from the full waveform dataset.

    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data and run single-channel detection
    st = read("example.mseed")
    add_channel_detections(st)

    # Get event time window with 20s padding
    mintime, maxtime = get_event_window(st, pretrig=20, posttrig=20)

    print(f"Detected event from {mintime} to {maxtime}")
    ```
    """

    mintime = []
    maxtime = []
    
    for tr in st:
        if 'triggers' in tr.stats:
            if len(tr.stats.triggers)==0:
                continue
            trigons = [thistrig[0] for thistrig in tr.stats.triggers]
            trigoffs = [thistrig[1] for thistrig in tr.stats.triggers]   
            mintime.append(min(trigons))
            maxtime.append(max(trigoffs))           
    
    N = int(len(mintime)/2)
    if len(mintime)>0:
        return sorted(mintime)[N], sorted(maxtime)[N]
    else:
        return None, None


def trim_to_event(st, mintime, maxtime, pretrig=10, posttrig=10):
    """
    Trims a Stream to a specified event window, adding a pre-trigger and post-trigger buffer.

    This function trims a `Stream` object to the time window defined by `mintime` (first trigger-on)
    and `maxtime` (last trigger-off), with optional extra time added before and after the event.

    **Use Cases:**
    - Focuses analysis on the main event while preserving relevant pre/post-event data.
    - Can be used after running `detect_network_event` or `get_event_window` to refine the dataset.

    Parameters:
    ----------
    st : obspy.Stream
        The input Stream object to be trimmed.
    mintime : UTCDateTime
        The earliest trigger-on time (e.g., from `get_event_window` or `detect_network_event`).
    maxtime : UTCDateTime
        The latest trigger-off time (e.g., from `get_event_window` or `detect_network_event`).
    pretrig : float, optional
        Extra time in seconds to keep **before** `mintime` (default: 10s).
    posttrig : float, optional
        Extra time in seconds to keep **after** `maxtime` (default: 10s).

    Returns:
    -------
    None
        The function modifies the `Stream` **in place**, trimming all traces to the specified time window.

    Notes:
    ------
    - If `mintime` or `maxtime` is `None`, the function will not trim the Stream.
    - If `pretrig` or `posttrig` exceeds the available data range, trimming may result in an empty Stream.
    - This function does not return a new `Stream` object; it modifies the input `st` directly.

    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data and detect events
    st = read("example.mseed")
    add_channel_detections(st)

    # Get event time window
    mintime, maxtime = get_event_window(st, pretrig=20, posttrig=20)

    # Trim the Stream to focus on the detected event
    if mintime and maxtime:
        trim_to_event(st, mintime, maxtime, pretrig=5, posttrig=15)
        print("Stream trimmed to event window.")
    ```
    """
    st.trim(starttime=mintime-pretrig, endtime=maxtime+posttrig)


def picker(st):

    # Group by station code
    station_groups = defaultdict(list)
    for tr in st:
        station = tr.stats.station
        station_groups[station].append(tr)

    # Convert to ObsPy Stream objects per station
    station_streams = {sta: Stream(traces) for sta, traces in station_groups.items()}
    print(station_streams)
    for station, stream in station_streams.items():
        if len(stream)==3:
            p_pick, s_pick = ar_pick(stream.select(component='Z')[0].data, stream.select(component='N')[0].data, stream.select(component='E')[0].data, stream[0].stats.sampling_rate, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
            print(f'ar_pick P for {station}: ', p_pick, 'S: ', s_pick)
        else:

            for tr in stream:
                p_pick, phase_info = pk_baer(tr.data, tr.stats.sampling_rate, 20, 60, 7.0, 12.0, 100, 100)
                print(f'pk_baer P for {tr.id}: ', p_pick, phase_info)
                aic_f = aic_simple(tr.data)
                p_idx = aic_f.argmin()
                print(f'aic_simple P for {tr.id}: ',tr.stats.starttime + p_idx / tr.stats.sampling_rate)

def filter_picks(tr, p_sec, s_sec, starttime):
    picks = {}
    p_time = starttime + p_sec
    s_time = starttime + s_sec    
    if p_sec < 1.0:
        p_sec = None
    if s_sec < 1.0:
        s_sec = None
    #print(p_sec, s_sec, p_time, s_time)
    if s_sec and p_sec:
        if s_sec > p_sec:
            picks[tr.id] = {'P': p_time, 'S': s_time} 
    elif p_sec:
        picks[tr.id] = {'P': p_time}
    elif s_sec:
        picks[tr.id] = {'S': s_time}
    return picks
    
def stream_picker(stream):
    picks = {}

    if len(stream) == 3:
        try:
            z = stream.select(component='Z')[0]
            n = stream.select(component='N')[0]
            e = stream.select(component='E')[0]
        except:
            return picks
        p_sec, s_sec = ar_pick(z.data, n.data, e.data, z.stats.sampling_rate, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
        new_picks = filter_picks(z, p_sec, s_sec, z.stats.starttime)
        picks.update(new_picks)
    else:
        try:
            tr = stream.select(component='Z')[0]
        except:
            return picks
        '''
        p_baer, _ = pk_baer(tr.data, tr.stats.sampling_rate, 20, 60, 7.0, 12.0, 100, 100)
        aic_f = aic_simple(tr.data)
        p_baer = tr.stats.starttime + p_baer/tr.stats.sampling_rate
        aic_idx = aic_f.argmin()
        p_aic = tr.stats.starttime + aic_idx / tr.stats.sampling_rate
        p_sec, s_sec = ar_pick(tr.data, tr.data, tr.data, tr.stats.sampling_rate,
                            1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
        p_time = tr.stats.starttime + p_sec
        s_time = tr.stats.starttime + s_sec                
        # Use pk_baer if it returns something usable, else fallback to AIC
        if p_baer:
            picks[tr.id] = {'P': p_baer}
        else:
            picks[tr.id] = {'P': p_aic}
        print(tr.id, p_baer-tr.stats.starttime, p_aic-tr.stats.starttime, p_time-tr.stats.starttime, s_time-tr.stats.starttime)
        '''
        p_sec, s_sec = ar_pick(tr.data, tr.data, tr.data, tr.stats.sampling_rate, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
        new_picks = filter_picks(tr, p_sec, s_sec, tr.stats.starttime)
        picks.update(new_picks)

    return picks


def picker_with_plot(st, plot=False):
    from collections import defaultdict
    picks = {}

    station_groups = defaultdict(list)
    for tr in st:
        station = tr.stats.station
        station_groups[station].append(tr)

    station_streams = {sta: Stream(traces) for sta, traces in station_groups.items()}

    for station, stream in station_streams.items():
        new_picks = stream_picker(stream)
        picks.update(new_picks)

    p_threshold = 3.0
    s_threshold = p_threshold * 1.7
    p_outliers, p_median_time = identify_outlier_picks(picks, phase='P', threshold=p_threshold)
    s_outliers, s_median_time = identify_outlier_picks(picks, phase='S', threshold=s_threshold)
    all_outliers = list(set(p_outliers+s_outliers))
    if all_outliers:
        for trace_id in all_outliers:
            if p_median_time:
                stime = UTCDateTime(p_median_time)-p_threshold
            else:
                stime = st[0].stats.starttime
            if s_median_time:
                etime = UTCDateTime(s_median_time)+s_threshold
            else:
                etime = st[0].stats.endtime
            this_stream = st.select(id=trace_id).copy().trim(starttime=stime, endtime=etime)
            new_picks = stream_picker(this_stream)
            picks.update(new_picks)
    if plot:
        plot_picks_on_stream(st, picks, title="P & S Picks Overlayed on Waveform")
    return picks

def identify_outlier_picks(picks, phase='P', threshold=2.0):
    if len(picks)<3:
        return [], None
    pick_times = []
    pick_ids = []
    for tr_id, phases in picks.items():
        if phase in phases:
            pick_times.append(phases[phase].timestamp)
            pick_ids.append(tr_id)

    if len(pick_times) < 3:
        return [], 0.0  # Not enough data to determine outliers

    pick_times = np.array(pick_times)
    median_time = np.median(pick_times)
    abs_diff = np.abs(pick_times - median_time)

    outliers = [pick_ids[i] for i, diff in enumerate(abs_diff) if diff > threshold]

    return outliers, median_time


def plot_picks_on_stream(stream: Stream, picks: dict, title="Waveform with Picks (Relative Time)"):
    """
    Plot traces with vertical lines at pick times (relative time in seconds).

    Parameters
    ----------
    stream : obspy.Stream
        The stream of traces to plot.
    picks : dict
        Dictionary of picks: {trace.id: {'P': UTCDateTime, 'S': UTCDateTime (optional)}}
    title : str
        Title for the plot.
    """
    fig, axes = plt.subplots(len(stream), 1, figsize=(12, 2.5 * len(stream)), sharex=True)
    if len(stream) == 1:
        axes = [axes]  # Ensure iterable

    for i, tr in enumerate(stream):
        t = tr.times("relative")
        axes[i].plot(t, tr.data, "k-", linewidth=0.8)
        axes[i].set_ylabel(tr.id, fontsize=9)

        # Add vertical lines for picks if availnew_pick['']]able
        if tr.id in picks:
            tr_picks = picks[tr.id]
            if "P" in tr_picks:
                p_rel = (tr_picks["P"] - tr.stats.starttime)
                axes[i].axvline(p_rel, color="b", linestyle="--", label="P")
            if "S" in tr_picks:
                s_rel = (tr_picks["S"] - tr.stats.starttime)
                axes[i].axvline(s_rel, color="r", linestyle="--", label="S")

            axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s since trace start)", fontsize=10)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()