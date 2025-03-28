import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, convolve
from obspy import Stream, Trace
from cycler import cycler
import cartopy.crs as crs
import cartopy.feature as cf

def get_envelope(trace, seconds=0.1):
    """Compute a smoothed envelope of a Trace using a Hilbert transform.

    Parameters:
    - trace (obspy.Trace): The input trace
    - seconds (float): Smoothing window in seconds (default=0.1)

    Returns:
    - np.ndarray: Smoothed envelope
    """
    envelope = np.abs(hilbert(trace.data))
    window_size = int(trace.stats.sampling_rate * seconds)
    window = np.ones(window_size) / window_size
    return convolve(envelope, window, mode='same')


def mulplt(st, outfile=None, bottomlabel=None, ylabels=None, units=None,
                group_by_station=True, remove_offset=True, use_envelope=False,
                show_legend=True, show_grid=True, summary='vector', channels='ZNE'):
    """Plot a Stream object similar to Seisan's mulplt, with optional enhancements.

    Parameters:
    - st (obspy.Stream): Stream object to plot
    - outfile (str): Save path; if None, plot is shown
    - bottomlabel (str): X-axis label for bottom subplot
    - ylabels (list): Custom Y labels
    - units (str): Units string for Y-axis
    - group_by_station (bool): Group traces by station
    - remove_offset (bool): Subtract median from each trace
    - use_envelope (bool): Plot envelope instead of raw data
    - show_legend (bool): Show legend
    - show_grid (bool): Show grid
    - summary (str): 'vector', 'median', or None for 3C traces
    - channels (str): Expected component characters (e.g., 'ZNE')

    Returns:
    - fh (matplotlib.figure.Figure): Figure handle
    - axh (list): List of Axes handles

    Example:
        >>> from obspy import read
        >>> st = read("*.mseed")
        >>> mulplt(st)
    """
    from cycler import cycler
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    fh = plt.figure(figsize=(8, 12))
    startepoch = st[0].stats.starttime.timestamp
    axh = []

    if group_by_station:
        stations = sorted(set(tr.stats.station for tr in st))
        n = len(stations)
        for i, sta in enumerate(stations):
            ax = fh.add_subplot(n, 1, i + 1, sharex=axh[0] if axh else None)
            axh.append(ax)
            these_traces = st.select(station=sta)
            all_ys = []
            for tr in these_traces:
                t = np.linspace(tr.stats.starttime.timestamp - startepoch,
                                tr.stats.endtime.timestamp - startepoch, tr.stats.npts)
                y = get_envelope(tr) if use_envelope else tr.data
                if remove_offset:
                    y = y - np.median(y)
                all_ys.append(y)
                chan = tr.stats.channel[2]
                if chan in channels:
                    ax.plot(t, y, label=chan, lw=0.5 if len(these_traces) > 1 else 2)
            if summary == 'vector' and len(all_ys) == 3:
                ax.plot(t, np.sqrt(sum(y**2 for y in all_ys)), 'r', label='vector', lw=2)
            elif summary == 'median' and len(all_ys) > 1:
                ax.plot(t, np.nanmedian(np.array(all_ys), axis=0), 'r', label='median', lw=2)
            if show_grid:
                ax.grid()
            if show_legend:
                ax.legend()
            ylabel = ylabels[i] if ylabels else sta + ('\n' + units if units else '')
            ax.set_ylabel(ylabel)
            if i == n - 1:
                ax.set_xlabel(bottomlabel or f"Seconds from {st[0].stats.starttime}")
    else:
        n = min(6, len(st))
        for i in range(n):
            ax = fh.add_subplot(n, 1, i + 1, sharex=axh[0] if axh else None)
            axh.append(ax)
            tr = st[i]
            t = np.linspace(tr.stats.starttime.timestamp - startepoch,
                            tr.stats.endtime.timestamp - startepoch, tr.stats.npts)
            y = get_envelope(tr) if use_envelope else tr.data
            if remove_offset:
                offset = np.median(y)
                y -= offset
            else:
                offset = 0
            ax.plot(t, y)
            if show_grid:
                ax.grid()
            if show_legend:
                ax.text(0, 1, f"max={np.max(np.abs(y)):.1e} offset={offset:.1e}",
                        transform=ax.transAxes, va='top')
            ylabel = ylabels[i] if ylabels else f"{tr.stats.station}.{tr.stats.channel}"
            if units:
                ylabel += f"\n{units}"
            ax.set_ylabel(ylabel)
            if i == n - 1:
                ax.set_xlabel(bottomlabel or f"Seconds from {st[0].stats.starttime}")

    plt.rcParams.update({'font.size': 9})
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()

    return fh, axh


def plot_envelope(st, window_size=1.0, percentile=99, outfile=None, units=None):
    """Plot the Nth percentile envelope for each trace in a Stream over fixed time windows.

    Parameters:
    - st (obspy.Stream): Stream to analyze
    - window_size (float): Window duration in seconds (default=1.0)
    - percentile (float): Percentile to compute within each window (default=99)
    - outfile (str): Output path for saved figure
    - units (str): Units label for Y-axis
    """
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange', 'black', 'grey', 'purple', 'cyan']
    for i, tr in enumerate(st):
        samples_per_window = int(window_size * tr.stats.sampling_rate)
        data = np.abs(tr.data)
        times = tr.times()
        num_windows = len(data) // samples_per_window
        reshaped_data = data[:num_windows * samples_per_window].reshape((num_windows, samples_per_window))
        percentiles = np.nanpercentile(reshaped_data, percentile, axis=1)
        window_times = times[:num_windows * samples_per_window:samples_per_window] + window_size / 2
        plt.plot(window_times, percentiles, label=tr.id, color=colors[i % len(colors)], lw=1)
    plt.title(f"{percentile}th Percentile in {window_size}-Second windows")
    plt.xlabel(f"Time (s) from {st[0].stats.starttime}")
    if units:
        plt.ylabel(units)
    plt.grid(True)
    plt.legend()
    if outfile:
        plt.savefig(fname=outfile)
    else:
        plt.show()

def plot_seismograms_basic(st, outfile=None, units=None):
    """Quick-look plotting for a Stream object with default settings.

    Wraps the more configurable `mulplt()` function using common defaults:
    - Envelope view with smoothing
    - Group by station
    - Remove DC offset
    - Show legend and grid
    - Add vector or median summary

    Parameters:
    - st (obspy.Stream): The Stream to plot
    - outfile (str): Path to save figure; if None, the figure is shown
    - units (str): Units string for Y-axis (optional)

    Example:
        >>> from obspy import read
        >>> st = read("*.mseed")
        >>> plot_seismograms_basic(st)
    """
    mulplt(
        st,
        outfile=outfile,
        units=units,
        bottomlabel=None,
        ylabels=None,
        group_by_station=True,
        remove_offset=True,
        use_envelope=True,
        show_legend=True,
        show_grid=True,
        summary='vector',
        channels='ZNE'
    )

def plot_station_amplitude_map(st, station0hypfile=None, outfile=None, cmap='Reds'):
    """Plot a map of station amplitudes using coordinates from a STATION0.HYP file.

    Parameters:
    - st (obspy.Stream): Stream with peak amplitudes in `tr.stats.metrics.peakamp`
    - station0hypfile (str): Path to STATION0.HYP file
    - outfile (str): Path to save figure; if None, figure is shown
    - cmap (str): Matplotlib colormap for amplitude scaling (default='Reds')

    Notes:
    - Volcano location is hardcoded to Soufrière Hills, Montserrat
    - Station coordinates and amplitudes must be present in each Trace
    - Uses `parse_STATION0HYP()` and `add_station_locations()` from `seisan_classes`
    """
    import cartopy.crs as crs
    import cartopy.feature as cf
    from seisan_classes import parse_STATION0HYP, add_station_locations

    if not station0hypfile:
        raise ValueError("STATION0.HYP file path must be provided.")

    station_locationsDF = parse_STATION0HYP(station0hypfile)
    add_station_locations(st, station_locationsDF)

    extent = [-62.27, -62.12, 16.67, 16.82]  # Montserrat
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())
    ax.set_extent(extent)
    ax.add_feature(cf.COASTLINE)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    lons, lats, amps, labels = [], [], [], []
    for tr in st:
        try:
            lons.append(tr.stats.lon)
            lats.append(tr.stats.lat)
            amps.append(tr.stats.metrics.peakamp)
            labels.append(tr.stats.station)
        except AttributeError:
            continue  # Skip traces without location/amp

    amps = np.array(amps)
    norm_sizes = (amps / amps.max()) * 300

    scatter = ax.scatter(lons, lats, s=norm_sizes, c=amps, cmap=cmap,
                         alpha=0.8, transform=crs.PlateCarree(), edgecolor='k')

    for lon, lat, label in zip(lons, lats, labels):
        ax.text(lon, lat, label, transform=crs.PlateCarree(), fontsize=8)

    # Volcano location
    ax.scatter(-62.1833326, 16.7166638, s=300, marker='*', color='orange',
               edgecolor='k', transform=crs.PlateCarree(), zorder=5)

    plt.colorbar(scatter, ax=ax, label='Peak Amplitude')
    plt.title('Station Peak Amplitudes')

    if outfile:
        try:
            plt.savefig(outfile, bbox_inches='tight')
        except Exception:
            plt.show()
    else:
        plt.show()


def downsample_and_plot_1_trace_per_location(stream, target_sampling_rate=1.0, plot=True, outfile=None):
    """Downsample and subset a Stream to one trace per station-location pair, prioritizing cleanest channels.

    Parameters:
    - stream (obspy.Stream): Input stream
    - target_sampling_rate (float): Desired sampling rate in Hz (default=1.0)
    - plot (bool): Whether to plot the selected stream (default=True)
    - outfile (str): Optional path to save the plot

    Returns:
    - selected_stream (obspy.Stream): Stream containing one trace per station-location
    """
    print(f'Downsampling traces to ≤{target_sampling_rate} Hz', end=' ')
    for trace in stream:
        print(trace.id, end=' ')
        if trace.stats.sampling_rate > target_sampling_rate:
            decimation_factor = int(trace.stats.sampling_rate / target_sampling_rate)
            if decimation_factor > 1:
                trace.decimate(decimation_factor, no_filter=True)
    print('\n')

    # Group traces by station-location and select the one with most non-zero samples
    station_location_data = {}
    print('Subsetting one trace per station-location')
    for trace in stream:
        key = (trace.stats.station, trace.stats.location)
        channel = trace.stats.channel
        nonzero_count = np.count_nonzero(trace.data)

        if key not in station_location_data:
            station_location_data[key] = {"seismic": None, "infrasound": None}

        if channel[1] == 'H':  # Seismic channel
            if (
                station_location_data[key]["seismic"] is None or
                nonzero_count > np.count_nonzero(station_location_data[key]["seismic"].data)
            ):
                station_location_data[key]["seismic"] = trace

def super_stream_plot(st, Nmax=6, rank='ZFODNE123456789', use_envelope=False,
                      remove_offset=True, show_legend=True, show_grid=True,
                      summary='vector', units=None, dpi=100, outfile=None,
                      figsize=(8, 8)):
    """Plot a ranked subset of a Stream with mulplt-style enhancements.

    Parameters:
    - st (obspy.Stream): Input Stream
    - Nmax (int): Max number of traces to include
    - rank (str): Channel rank priority, e.g. 'ZNE123'
    - use_envelope (bool): Plot Hilbert envelopes instead of raw data
    - remove_offset (bool): Subtract median offset from each trace
    - show_legend (bool): Show component/channel legend
    - show_grid (bool): Draw grid
    - summary (str): 'vector', 'median', or None for 3C traces
    - units (str): Units for y-axis label (optional)
    - dpi (int): DPI for output
    - outfile (str): Path to save figure
    - figsize (tuple): Figure size in inches

    Returns:
    - st2 (Stream): The plotted subset of the stream
    - fh (Figure): Matplotlib figure object

    Example:
        >>> super_stream_plot(st, Nmax=4, rank='ZNE', use_envelope=True)
    """
    from cycler import cycler
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    # Rank-based trace selection
    st2 = obspy.Stream()
    if len(st) > Nmax:
        rankpos = 0
        while len(st2) < Nmax and rankpos < len(rank):
            for tr in st.select(channel=f'*{rank[rankpos]}'):
                if len(st2) < Nmax:
                    st2.append(tr)
            rankpos += 1
    else:
        st2 = st.copy()

    # Plotting
    fh = plt.figure(figsize=figsize)
    startepoch = st2[0].stats.starttime.timestamp
    axh = []

    for i, tr in enumerate(st2):
        ax = fh.add_subplot(len(st2), 1, i + 1, sharex=axh[0] if axh else None)
        axh.append(ax)
        t = np.linspace(tr.stats.starttime.timestamp - startepoch,
                        tr.stats.endtime.timestamp - startepoch, tr.stats.npts)
        y = get_envelope(tr) if use_envelope else tr.data
        if remove_offset:
            offset = np.median(y)
            y = y - offset
        else:
            offset = 0

        ax.plot(t, y, label=tr.stats.channel)
        if show_grid:
            ax.grid()
        if show_legend:
            ax.text(0.01, 0.95, f"max={np.max(np.abs(y)):.1e} offset={offset:.1e}",
                    transform=ax.transAxes, va='top', fontsize=8)

        # Y-axis label
        ylabel = f"{tr.stats.station}.{tr.stats.channel}"
        if units:
            ylabel += f"\n{units}"
        ax.set_ylabel(ylabel)

        # Summary amplitude if next 2 traces exist and same station
        if summary and i <= len(st2) - 3:
            chs = [st2[i].stats.channel[2], st2[i + 1].stats.channel[2], st2[i + 2].stats.channel[2]]
            if all(ch in rank for ch in chs):
                ys = [get_envelope(tr_) if use_envelope else tr_.data for tr_ in st2[i:i+3]]
                if summary == 'vector':
                    vector = np.sqrt(sum(y**2 for y in ys))
                    ax.plot(t, vector, 'r', label='vector', lw=1)
                elif summary == 'median':
                    ax.plot(t, np.nanmedian(np.vstack(ys), axis=0), 'r', label='median', lw=1)
                ax.legend(fontsize=6)

        if i == len(st2) - 1:
            ax.set_xlabel(f"Seconds from {tr.stats.starttime}")

    plt.rcParams.update({'font.size': 9})
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    if outfile:
        plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    return st2, fh


def plot_record_section(st2, reftime, outfile=None, starttime=0, endtime=0, km_min=0, km_max=20037.5, slopes=[], scale_factor=1.0, plot_arrivals=False, figsize=(16, 16), min_spacing_km=0, reduced_speed=None, normalize=False):
    r = np.array(get_distance_vector(st2))
    st = order_traces_by_distance(st2, r)
    if reduced_speed:
        for tr in st:
            tr.stats.starttime -= tr.stats.distance / reduced_speed
    if endtime > starttime:
        st.trim(starttime=starttime, endtime=endtime)
    if normalize:
        st.normalize(global_max=False)

    rmax = np.max(r)
    l = len(st)
    max_trace_height = scale_factor / l * (km_max - km_min)
    if min_spacing_km == 0:
        min_spacing_km = 111.0 * km_max / 20037.5

    fh = plt.figure(figsize=figsize)
    ah = fh.add_subplot(1, 1, 1)
    last_r = -9999999
    last_asecs = -99999999
    max_pascals = 1500

    for i, tr in enumerate(st):
        tr.detrend('linear')
        if np.max(np.abs(tr.data)) > max_pascals:
            continue
        offset_km = tr.stats.distance / 1000
        if 'arrival' in tr.stats:
            asecs = tr.stats.arrival.arrivaltime - reftime
            if plot_arrivals:
                ah.plot(asecs / 3600, offset_km, 'b.', zorder=len(st) * 2 + 11 - i)
            last_asecs = asecs

        t0 = tr.stats.starttime - reftime
        t = t0 + np.arange(0, tr.stats.delta * len(tr.data), tr.stats.delta)
        h = t / 3600.0
        diff_r = tr.stats.distance - last_r
        brightness = 0
        if diff_r < min_spacing_km * 1000:
            if diff_r < min_spacing_km / 2 * 1000:
                continue
            brightness = 1.0 - (diff_r / 1000) / min_spacing_km
            last_r = tr.stats.distance
        else:
            last_r = tr.stats.distance
        col = [brightness] * 3
        ah.plot(h, offset_km + (tr.data * scale_factor), color=col, linewidth=0.5, zorder=(len(st) * 2 + 10 - i) * (1.0 - brightness))

    if slopes:
        for slope in slopes:
            slope_speed, slope_origintime = slope[:2]
            linestyle = slope[2] if len(slope) == 3 else 'r-'
            d_prime = np.array([km_min, km_max])
            t_prime = d_prime * 1000 / slope_speed + (slope_origintime - st[0].stats.starttime)
            h_prime = t_prime / 3600
            ah.plot(h_prime, d_prime, linestyle, zorder=900)

    ah.set_ylabel('Distance (km)')
    ah.set_xlabel(f'Time (hours) after {reftime.strftime("%Y-%m-%d %H:%M:%S")}')
    ah.set_ylim(km_min, km_max)
    h = t / 3600.0
    ah.set_xlim(h[0], h[-1])
    numhours = h[-1] - h[0]
    stephours = round(numhours / 20, 1 if numhours <= 20 else 0)
    ah.set_xticks(np.arange(h[0], h[-1], stephours))
    ah.grid(axis='x', color='green', linestyle=':', linewidth=0.5)

    m_per_deg = 111195.0
    ah2 = ah.twinx()
    ah2.set_ylabel('Distance (degrees)')
    ah2.set_ylim(1000 * km_min / m_per_deg, 1000 * km_max / m_per_deg)
    ah2.plot([0, 0], [1000 * km_min / m_per_deg, 1000 * km_max / m_per_deg], 'k-')

    if outfile:
        fh.savefig(outfile)
    else:
        plt.show()

    return
