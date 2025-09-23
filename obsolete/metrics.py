

def export_trace_metrics_to_dataframe(stream, include_id=True, include_starttime=True):
    """
    Export all trace.stats.metrics and associated metadata into a pandas DataFrame.

    Parameters
    ----------
    stream : obspy.Stream
        Stream with populated trace.stats.metrics dictionaries.
    include_id : bool
        Whether to include trace.id as a column.
    include_starttime : bool
        Whether to include trace.stats.starttime as a column.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of metrics, one row per trace.
    """
    rows = []

    for tr in stream:
        s = tr.stats
        row = {}

        if include_id:
            row['id'] = tr.id
        if include_starttime:
            row['starttime'] = s.starttime

        row['Fs'] = s.sampling_rate
        row['calib'] = getattr(s, 'calib', None)
        row['units'] = getattr(s, 'units', None)
        row['quality'] = getattr(s, 'quality_factor', None)

        if hasattr(s, 'spectrum'):
            for item in ['medianF', 'peakF', 'peakA', 'bw_min', 'bw_max']:
                row[item] = s.spectrum.get(item, None)

        if hasattr(s, 'metrics'):
            m = s.metrics
            for item in [
                'snr', 'signal_level', 'noise_level', 'twin',
                'peakamp', 'peaktime', 'energy', 'RSAM_high', 'RSAM_low',
                'sample_min', 'sample_max', 'sample_mean', 'sample_median',
                'sample_lower_quartile', 'sample_upper_quartile', 'sample_rms',
                'sample_stdev', 'percent_availability', 'num_gaps', 'skewness', 'kurtosis'
            ]:
                row[item] = m.get(item, None)
            for key, value in m.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        row[f"{key}_{subkey}"] = subval

        if 'bandratio' in m:
            for dictitem in m['bandratio']:
                label = 'bandratio_' + "".join(str(dictitem['freqlims'])).replace(', ', '_')
                row[label] = dictitem['RSAM_ratio']

        if 'lon' in s:
            row['lon'] = s['lon']
            row['lat'] = s['lat']
            row['elev'] = s['elev']

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.round({
        'Fs': 2, 'secs': 2, 'quality': 2, 'medianF': 1, 'peakF': 1,
        'bw_max': 1, 'bw_min': 1, 'peaktime': 2, 'twin': 2,
        'skewness': 2, 'kurtosis': 2
    })
    return df

        
def load_trace_metrics_from_dataframe(df, stream, match_on='id'):
    """
    Load metrics from a DataFrame into each trace.stats.metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with one row per trace and metrics as columns.
    stream : obspy.Stream
        Stream of Trace objects to update.
    match_on : str
        Column to match traces by ('id' or 'starttime').

    Returns
    -------
    stream : obspy.Stream
        Stream with updated .stats.metrics per trace.
    """
    for tr in stream:
        key = tr.id if match_on == 'id' else tr.stats.starttime
        row = df[df[match_on] == key]
        if row.empty:
            continue

        tr.stats.metrics = {}

        for col in row.columns:
            if col in ['id', 'starttime']:
                continue
            val = row.iloc[0][col]

            # Optional: split compound keys like 'ssam_A' back into nested dicts
            if '_' in col:
                main_key, sub_key = col.split('_', 1)
                if main_key not in tr.stats.metrics:
                    tr.stats.metrics[main_key] = {}
                tr.stats.metrics[main_key][sub_key] = val
            else:
                tr.stats.metrics[col] = val

    return stream

def save_enhanced_stream(st, enhanced_wavpath, save_pickle=False):
    """
    Save a stream to MiniSEED along with a CSV of enhanced metrics.

    Parameters
    ----------
    st : obspy.Stream
        Stream with .stats.metrics, .stats.spectrum, etc.
    enhanced_wavpath : str
        Path without extension (.mseed, .csv will be added).
    save_pickle : bool
        Whether to also save as ObsPy .pickle format (optional).
    """
    if enhanced_wavpath.endswith('.mseed'):
        enhanced_wavpath = enhanced_wavpath[:-6]

    os.makedirs(os.path.dirname(enhanced_wavpath), exist_ok=True)

    # Save MiniSEED
    st.write(enhanced_wavpath + '.mseed', format='MSEED')

    # Save metrics CSV
    df = export_trace_metrics_to_dataframe(st)
    df.to_csv(enhanced_wavpath + '.csv', index=False)

    # Optional: also save full stream with attributes
    if save_pickle:
        st.write(enhanced_wavpath + '.pickle', format='PICKLE')


def read_enhanced_stream(enhanced_wavpath):
    """
    Read a stream from MiniSEED + CSV metrics and restore enhanced trace.stats.

    Parameters
    ----------
    enhanced_wavpath : str
        Path without extension (.mseed, .csv will be added).

    Returns
    -------
    st : obspy.Stream
        Stream with restored .stats.metrics and other attributes.
    """
    if enhanced_wavpath.endswith('.mseed'):
        enhanced_wavpath = enhanced_wavpath[:-6]

    st = read(enhanced_wavpath + '.mseed')
    df = pd.read_csv(enhanced_wavpath + '.csv')

    # First pass: restore metrics
    st = load_trace_metrics_from_dataframe(df, st)

    # Second pass: restore extra fields not in .metrics
    for tr in st:
        row = df[df['id'] == tr.id]
        if row.empty:
            continue
        row = row.iloc[0]

        s = tr.stats
        s.units = row.get('units', None)
        s.calib = row.get('calib', None)
        s.quality_factor = row.get('quality', None)

        # Restore spectrum if present
        spectrum_keys = ['medianF', 'peakF', 'peakA', 'bw_min', 'bw_max']
        s.spectrum = {k: row[k] for k in spectrum_keys if k in row and not pd.isna(row[k])}

        # Restore bandratio (if any)
        bandratios = []
        for col in row.index:
            if col.startswith('bandratio_') and not pd.isna(row[col]):
                try:
                    freqlims_str = col.replace('bandratio_', '').replace('[', '').replace(']', '')
                    freqlims = [float(x) for x in freqlims_str.split('_')]
                    bandratios.append({'freqlims': freqlims, 'RSAM_ratio': row[col]})
                except Exception as e:
                    print(f"Warning: couldn't parse {col}: {e}")
        if bandratios:
            s.bandratio = bandratios

        for coord in ['lon', 'lat', 'elev']:
            if coord in row:
                s[coord] = row[coord]

    return st



    



def stream_to_event(stream, source_coords, df=None,
                    event_id=None, creation_time=None,
                    event_type=None, mainclass=None, subclass=None):
    """
    Convert a Stream and source metadata into an ObsPy Event object.

    Parameters
    ----------
    stream : obspy.Stream
        Stream of traces with metrics in .stats.metrics
    source_coords : dict
        {'latitude': ..., 'longitude': ..., 'depth': ...} in meters
    df : pandas.DataFrame, optional
        Output from summarize_magnitudes()
    event_id : str, optional
        Resource ID for the Event
    creation_time : str or UTCDateTime, optional
        Event creation time
    event_type : str, optional
        QuakeML-compatible event type (e.g. "earthquake", "volcanic eruption")
    mainclass : str, optional
        High-level classification (e.g., "volcano-seismic", "tectonic")
    subclass : str, optional
        Detailed classification (e.g., "hybrid", "long-period")

    Returns
    -------
    event : obspy.core.event.Event
    """
    event = Event()

    # Resource ID and creation time
    if event_id:
        event.resource_id = ResourceIdentifier(id=event_id)
    if creation_time:
        event.creation_info = {'creation_time': UTCDateTime(creation_time)}

    # Event type
    if event_type:
        event.event_type = event_type

    # Add origin
    origin = Origin(
        latitude=source_coords['latitude'],
        longitude=source_coords['longitude'],
        depth=source_coords.get('depth', 0)
    )
    event.origins.append(origin)

    # Add network-averaged magnitudes from DataFrame
    if df is not None and hasattr(df, 'iloc'):
        net_ml = df.iloc[-1].get('network_mean_ML')
        net_me = df.iloc[-1].get('network_mean_ME')

        if pd.notnull(net_ml):
            mag_ml = Magnitude(
                mag=net_ml,
                magnitude_type="ML",
                origin_id=origin.resource_id
            )
            event.magnitudes.append(mag_ml)

        if pd.notnull(net_me):
            mag_me = Magnitude(
                mag=net_me,
                magnitude_type="Me",
                origin_id=origin.resource_id
            )
            event.magnitudes.append(mag_me)

    # Add station magnitudes
    for tr in stream:
        ml = tr.stats.metrics.get('local_magnitude')
        if ml is None:
            continue

        wid = WaveformStreamID(
            network_code=tr.stats.network,
            station_code=tr.stats.station,
            location_code=tr.stats.location,
            channel_code=tr.stats.channel
        )

        smag = StationMagnitude(
            mag=ml,
            magnitude_type="ML",
            station_magnitude_type="ML",
            waveform_id=wid
        )
        event.station_magnitudes.append(smag)

        if event.magnitudes:
            contrib = StationMagnitudeContribution(
                station_magnitude_id=smag.resource_id,
                weight=1.0
            )
            event.magnitudes[0].station_magnitude_contributions.append(contrib)

    # Add trace-level metrics as JSON comment
    try:
        metrics_dict = {tr.id: tr.stats.metrics for tr in stream if hasattr(tr.stats, 'metrics')}
        comment = Comment(
            text=json.dumps(metrics_dict, indent=2),
            force_resource_id=False
        )
        event.comments.append(comment)
    except Exception:
        pass

    # Add mainclass and subclass if given
    if mainclass or subclass:
        class_comment = {
            "mainclass": mainclass,
            "subclass": subclass
        }
        comment = Comment(
            text=json.dumps(class_comment, indent=2),
            force_resource_id=False
        )
        event.comments.append(comment)

    return event


################################################################
'''
def max_3c(st):
    """ max of a 3-component seismogram """
    N = len(st)/3
    m = []

    if N.is_integer():
        st.detrend()
        for c in range(int(N)):
            y1 = st[c*3+0].data
            y2 = st[c*3+1].data
            y3 = st[c*3+2].data
            y = np.sqrt(np.square(y1) + np.square(y2) + np.square(y3))
            m.append(max(y))
    return m 
'''


