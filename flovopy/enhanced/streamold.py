class EnhancedStream(Stream):
    def __init__(self, stream=None, traces=None):
        if traces is not None:
            # Allow ObsPy internal logic to pass 'traces='
            super().__init__(traces=traces)
        elif stream is not None:
            # Fallback for manual use with stream=...
            super().__init__(traces=stream.traces)
        else:
            # Empty stream
            super().__init__()




    def legacy_ampengfft(self, filepath, freq_bins=None, amp_avg_window=2.0,
                                trigger_window=None, average_window=None):
        compute_ampengfft_stream(self, freq_bins=freq_bins, amp_avg_window=amp_avg_window)
        write_aef_file(self, filepath, trigger_window=trigger_window, average_window=average_window)


    def ampengfft(
        self,
        *,
        # time-domain stays on by default (cheap)
        differentiate: bool = False,          # True: input disp -> vel for energy/PGM
        compute_spectral: bool = False,       # False = skip all FFT-based metrics
        compute_ssam: bool = False,           # depends on spectral
        compute_bandratios: bool = False,     # depends on spectral
        # spectral params
        threshold: float = 0.707,
        window_length: int = 9,
        polyorder: int = 2,
    ) -> None:
        """
        Compute metrics per trace. Time-domain metrics are always computed.
        Spectral metrics are optional (compute_spectral=False by default).

        differentiate:
        If True, treat input as DISPLACEMENT, derive VEL (and ACC) for energy/PGM.
        If False, treat input as already VEL, derive DISP (and ACC).
        """
        if len(self) == 0:
            return

        for tr in self:
            # Ensure containers
            if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
                tr.stats.metrics = {}

            # Choose working series (float64 for stability)
            dt = float(tr.stats.delta)
            y0 = np.asarray(tr.data, dtype=np.float64)

            tr.detrend("linear").taper(0.01)

            if differentiate:
                # input assumed DIS -> VEL,ACC
                disp = tr.copy()
                vel  = tr.copy().differentiate()
                acc  = vel.copy().differentiate()
            else:
                # input assumed VEL -> DISP,ACC
                vel  = tr
                disp = tr.copy().integrate()
                acc  = tr.copy().differentiate()

            y = np.asarray(vel.data, dtype=np.float64)

            # ---------- TIME-DOMAIN METRICS (cheap) ----------
            m = tr.stats.metrics
            if y.size:
                m["sample_min"]     = float(np.nanmin(y))
                m["sample_max"]     = float(np.nanmax(y))
                m["sample_mean"]    = float(np.nanmean(y))
                m["sample_median"]  = float(np.nanmedian(y))
                m["sample_rms"]     = float(np.sqrt(np.nanmean(y * y)))
                m["sample_stdev"]   = float(np.nanstd(y))
                from scipy.stats import skew, kurtosis
                m["skewness"]       = float(skew(y, nan_policy="omit"))
                m["kurtosis"]       = float(kurtosis(y, nan_policy="omit"))
            else:
                for k in ("sample_min","sample_max","sample_mean","sample_median",
                        "sample_rms","sample_stdev","skewness","kurtosis"):
                    m[k] = np.nan

            # Peak amplitude/time on |VEL| (consistent with typical PGM use)
            absy = np.abs(y)
            if absy.size:
                m["peakamp"] = float(np.nanmax(absy))
                m["peaktime"] = tr.stats.starttime + (int(np.nanargmax(absy)) * dt)
            else:
                m["peakamp"] = np.nan
                m["peaktime"] = None

            # Energy on working series (VEL unless you changed via differentiate)
            m["energy"] = float(np.nansum(y * y) * dt) if y.size else np.nan

            # PGD / PGV / PGA
            try:
                m["pgv"] = float(np.nanmax(np.abs(vel.data))) if vel.stats.npts else np.nan
                m["pgd"] = float(np.nanmax(np.abs(disp.data))) if disp.stats.npts else np.nan
                m["pga"] = float(np.nanmax(np.abs(acc.data))) if acc.stats.npts else np.nan
            except Exception:
                m.setdefault("pgv", np.nan)
                m.setdefault("pgd", np.nan)
                m.setdefault("pga", np.nan)

            # Dominant frequency (time-domain ratio)
            try:
                num = np.abs(vel.data).astype(np.float64)
                den = 2.0 * np.pi * np.abs(disp.data).astype(np.float64) + 1e-20
                fdom_series = num / den
                m["fdom"] = float(np.nanmedian(fdom_series)) if fdom_series.size else np.nan
            except Exception:
                m["fdom"] = np.nan

            # (Optional) You can add RSAM/bandratio *without* FFT using time-domain filters
            # if you want cheap band features here.

            # ---------- SPECTRAL METRICS (optional; more expensive) ----------
            if not compute_spectral:
                # Cheap band features (no FFT):
                # choose your default bands & statistic once
                _band_pairs = [ (0.5, 2.0, 2.0, 8.0),  # LF vs HF
                                (1.0, 3.0, 3.0, 12.0) ]  # alt split; optional

                for (l1, l2, h1, h2) in _band_pairs[:1]:  # [:1] keeps just the first pair for speed
                    br = _td_band_ratio(tr, low=(l1,l2), high=(h1,h2), stat="mean_abs", log2=True)
                    # Persist RSAM scalars so they’re directly usable downstream
                    tr.stats.metrics[f"rsam_{l1}_{l2}"] = br["RSAM_low"]
                    tr.stats.metrics[f"rsam_{h1}_{h2}"] = br["RSAM_high"]
                    tr.stats.metrics[ br["ratio_key"] ] = br["ratio"]
                continue

            # Use rFFT (one-sided) for efficiency
            from numpy.fft import rfft, rfftfreq

            N = tr.stats.npts
            if N < 2:
                continue

            Y = np.abs(rfft(y0))  # choose raw series or vel; y0 is raw input
            F = rfftfreq(N, d=dt)

            if not hasattr(tr.stats, "spectral") or tr.stats.spectral is None:
                tr.stats.spectral = {}
            tr.stats.spectral["freqs"] = F
            tr.stats.spectral["amplitudes"] = Y

            # Bandwidth / cutoffs (uses your helper; stores into metrics)
            try:
                get_bandwidth(F, Y, threshold=threshold, window_length=window_length,
                            polyorder=polyorder, trace=tr)
            except Exception as e:
                print(f"[{tr.id}] Skipping bandwidth metrics: {e}")

            # SSAM (optional)
            if compute_ssam:
                try:
                    _ssam(tr)
                except Exception as e:
                    print(f"[{tr.id}] SSAM computation failed: {e}")

            # Band ratios (optional)
            if compute_bandratios:
                try:
                    _band_ratio(tr, freqlims=[1.0, 6.0, 11.0])
                    _band_ratio(tr, freqlims=[0.5, 3.0, 18.0])
                except Exception as e:
                    print(f"[{tr.id}] Band ratio computation failed: {e}")

            # Spectral peak/means (store both camel/snake variants for compatibility)
            try:
                A = tr.stats.spectral["amplitudes"]
                if A.size and np.any(A > 0):
                    peak_idx = int(np.nanargmax(A))
                    peakf = float(F[peak_idx])
                    meanf = float(np.nansum(F * A) / np.nansum(A))
                else:
                    peakf = meanf = np.nan
                m["peakf"]   = peakf
                m["meanf"]   = meanf
                m["medianf"] = float(np.nanmedian(F[A > 0])) if np.any(A > 0) else np.nan

                # Back-compat into s.stats.spectrum keys some code expects
                s = getattr(tr.stats, "spectrum", {})
                s["peakF"]   = peakf
                s["medianF"] = m["medianf"]
                s["peakA"]   = float(np.nanmax(A)) if A.size else np.nan
                tr.stats.spectrum = s
            except Exception as e:
                print(f"[{tr.id}] Spectral summary failed: {e}")


    def save(self, basepath, save_pickle=False):
        """
        Save enhanced stream to .mseed, .csv, and optionally .pickle.

        Parameters
        ----------
        basepath : str
            Base path (without extension) to save files.
        save_pickle : bool
            If True, also save the stream as a .pickle file.
        """
        if basepath.endswith('.mseed'):
            basepath = basepath[:-6]

        parentdir = os.path.dirname(basepath)
        if not os.path.exists(parentdir):
            os.makedirs(parentdir)

        self.write(basepath + '.mseed', format='MSEED')


        rows = []

        for tr in self:
            s = tr.stats
            row = {}

            #if include_id:
            row['id'] = tr.id
            #if include_starttime:
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

            if 'bandratio' in s:
                for dictitem in s['bandratio']:
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


        df.to_csv(basepath + '.csv', index=False)

        if save_pickle:
            self.write(basepath + '.pickle', format='PICKLE')

    @classmethod
    def read(cls, basepath, match_on='id'):
        """
        Load EnhancedStream from .mseed and .csv file.

        Parameters
        ----------
        basepath : str
            Path without extension (e.g. '/path/to/event')

        Returns
        -------
        EnhancedStream
        """
        basepath = basepath.replace('.mseed', '')
        st = read(basepath + '.mseed', format='MSEED')
        df = pd.read_csv(basepath + '.csv', index_col=False)

        # Load metrics from a DataFrame into each trace.stats.metrics.
        for tr in st:
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

        return cls(stream=st)

    def attach_station_coordinates_from_inventory(self, inventory):
        """ attach_station_coordinates_from_inventory """
        
        for tr in self:
            for netw in inventory.networks:
                for sta in netw.stations:
                    if tr.stats.station == sta.code and netw.code == tr.stats.network:
                        for cha in sta.channels:
                            if tr.stats.location == cha.location_code:
                                tr.stats.coordinates = AttribDict({
                                    'latitude':cha.latitude,
                                    'longitude':cha.longitude,
                                    'elevation':cha.elevation})
                                
    def compute_station_magnitudes(self, inventory, source_coords,
                                    model='body', Q=50, c_earth=2500, correction=3.7,
                                    a=1.6, b=-0.15, g=0,
                                    use_boatwright=True,
                                    rho_earth=2000, S=1.0, A=1.0,
                                    rho_atmos=1.2, c_atmos=340, z=100000,
                                    attach_coords=True, compute_distances=True):
        """
        Attach coordinates and estimate energy-based and local magnitudes for all traces.

        Parameters
        ----------
        inventory : obspy.Inventory
            Station metadata
        source_coords : dict
            {'latitude': ..., 'longitude': ..., 'depth': ...}
        model : str
            'body' or 'surface' wave geometric spreading
        Q : float
            Attenuation quality factor
        c_earth : float
            Seismic wave speed (m/s)
        correction : float
            Correction factor in Hanks & Kanamori formula
        a, b, g : float
            ML Richter coefficients and station correction
        use_boatwright : bool
            If True, use Boatwright formulation for energy estimation
        rho_earth, S, A : float
            Parameters for seismic Boatwright energy model
        rho_atmos, c_atmos, z : float
            Parameters for infrasound Boatwright energy model
        attach_coords : bool
            Whether to attach station coordinates from inventory
        compute_distances : bool
            Whether to compute and store distance (in meters)

        Returns
        -------
        stream : obspy.Stream
            Updated stream with .stats.metrics['energy_magnitude'] and ['local_magnitude']
        """
        if attach_coords:
            self.attach_station_coordinates_from_inventory(inventory)

        for tr in self:
            try:
                R = estimate_distance(tr, source_coords)

                if compute_distances:
                    tr.stats['distance'] = R

                if not hasattr(tr.stats, 'metrics'):
                    tr.stats.metrics = {}

                # Choose energy model based on data type and user preference
                if use_boatwright:
                    if tr.stats.channel[1].upper() == 'D':  # Infrasound
                        E0 = Eacoustic_Boatwright(tr, R, rho_atmos=rho_atmos, c_atmos=c_atmos, z=z)
                    else:  # Seismic
                        E0 = Eseismic_Boatwright(tr, R, rho_earth=rho_earth, c_earth=c_earth, S=S, A=A)

                        # Apply Q correction using spectral peak frequency if available
                        if 'spectral' in tr.stats and 'freqs' in tr.stats.spectral and 'amplitudes' in tr.stats.spectral:
                            freqs = tr.stats.spectral['freqs']
                            amps = tr.stats.spectral['amplitudes']
                            if np.any(amps > 0):
                                f_peak = freqs[np.argmax(amps)]
                                A_att = np.exp(-np.pi * f_peak * R / (Q * c_earth))
                                E0 /= A_att
                else:
                    E0 = estimate_source_energy(tr, R, model=model, Q=Q, c_earth=c_earth)

                ME = Eseismic2magnitude(E0, correction=correction)

                tr.stats.metrics['source_energy'] = E0
                tr.stats.metrics['energy_magnitude'] = ME

                if tr.stats.channel[1].upper() in ('H', 'L'):
                    R_km = R / 1000
                    ML = estimate_local_magnitude(tr, R_km, a=a, b=b, g=g)
                    tr.stats.metrics['local_magnitude'] = ML

            except Exception as e:
                print(f"[{tr.id}] Magnitude estimation failed: {e}")

    def magnitudes2dataframe(self):
        """
        Summarize trace-level and network-averaged magnitude values.

        Returns
        -------
        df : pandas.DataFrame
            Columns: id, starttime, distance, local_magnitude (ML), energy_magnitude (ME),
        """
        rows = []

        for tr in self:
            row = {
                'id': tr.id,
                'starttime': tr.stats.starttime
            }

            row['distance_m'] = tr.stats.get('distance', np.nan)
            metrics = tr.stats.get('metrics', {})

            row['ML'] = metrics.get('local_magnitude', np.nan)
            row['ME'] = metrics.get('energy_magnitude', np.nan)
            row['source_energy'] = metrics.get('source_energy', np.nan)
            row['peakamp'] = metrics.get('peakamp', np.nan)
            row['energy'] = metrics.get('energy', np.nan)

            rows.append(row)

        df = pd.DataFrame(rows)

        return df

    def estimate_network_magnitude(self, key='local_magnitude'):
        """
        Compute network-average magnitude with uncertainty.

        Parameters
        ----------
        key : str
            Metric to average (e.g., 'local_magnitude', 'energy_magnitude')

        Returns
        -------
        tuple of (mean_mag, std_dev, n)
        """
        mags = []
        for tr in self:
            mag = tr.stats.metrics.get(key)
            if mag is not None:
                mags.append(mag)

        if len(mags) == 0:
            return None, None, 0

        return np.mean(mags), np.std(mags), len(mags)
    
    def ampengfftmag(self, inventory, source_coords,
                     model='body', Q=50, c_earth=2500, correction=3.7,
                     a=1.6, b=-0.15, g=0,
                     threshold=0.707, window_length=9, polyorder=2,
                     differentiate=True, verbose=True, snr_method='std',
                     snr_split_time=None, snr_window_length=1.0,
                     snr_min=None):
        """
        Compute amplitude, spectral metrics, SNR, and magnitudes for the EnhancedStream.

        Returns
        -------
        stream : EnhancedStream
        df : pandas.DataFrame
        """


        if verbose:
            print("[2] Estimating SNR values...")
        for tr in self:
            if not hasattr(tr.stats, 'metrics'):
                tr.stats['metrics'] = {}
            if not 'snr' in tr.stats.metrics:
                try:
                    snr, signal_val, noise_val = estimate_snr(
                        tr, method=snr_method, window_length=snr_window_length,
                        split_time=snr_split_time, verbose=False
                    )
                    tr.stats.metrics['snr'] = snr
                    tr.stats.metrics['signal'] = signal_val
                    tr.stats.metrics['noise'] = noise_val
                except Exception as e:
                    if verbose:
                        print(f"[{tr.id}] SNR estimation failed: {e}")
            

        if snr_min: #is not None:
            if verbose:
                print(f"[3] Filtering traces with SNR < {snr_min:.1f}...")
            filtered = [tr for tr in self if tr.stats.metrics.get('snr', 0) >= snr_min]
            self._traces = filtered
            for tr in self:
                if tr.stats.metrics.snr < snr_min:
                    self.remove(tr)

        if verbose:
            print("[1] Computing amplitude and spectral metrics...")
        self.ampengfft(threshold=threshold, window_length=window_length,
                  polyorder=polyorder, differentiate=differentiate)

        if verbose:
            print("[4] Estimating station magnitudes...")
        self.compute_station_magnitudes(inventory, source_coords,
                                   model=model, Q=Q, c_earth=c_earth, correction=correction,
                                   a=a, b=b, g=g,
                                   attach_coords=True, compute_distances=True)

        if verbose:
            print("[5] Summarizing network magnitude statistics...")
        df = self.magnitudes2dataframe()
        #print(df.describe())

        '''
        if verbose and not df.empty:
            net_ml = df.iloc[-1].get('network_mean_ML', np.nan)
            net_me = df.iloc[-1].get('network_mean_ME', np.nan)
            print(f"    → Network ML: {net_ml:.2f} | Network ME: {net_me:.2f}")
        '''

        return self, df


    def to_pickle(self, outdir, remove_data=True, mseed_path=None):
        """
        Save EnhancedStream to a pickle file with optional data removal.

        Parameters
        ----------
        outdir : str
            Directory to write .pkl file to.
        remove_data : bool
            If True, remove trace.data to reduce file size.
        mseed_path : str
            Path to the original MiniSEED file (for reference).
        """
        os.makedirs(outdir, exist_ok=True)
        stream_copy = self.copy()

        for tr in stream_copy:
            if remove_data:
                tr.data = []
            tr.stats['mseed_path'] = mseed_path

        # Compose filename
        filename = os.path.basename(mseed_path or "stream") + ".pkl"
        outfile = os.path.join(outdir, filename)

        with open(outfile, "wb") as f:
            pickle.dump(stream_copy, f)
        
        print(f"[✓] Saved pickle: {outfile}")


    def to_obspyevent(self, source_coords, event_id=None, creation_time=None,
                    event_type='volcanic eruption', mainclass='LV', subclass=None):
        """
        Convert EnhancedStream and source metadata into ObsPy Event object.

        Parameters
        ----------
        source_coords : dict
            {'latitude': ..., 'longitude': ..., 'depth': ...} in meters
        event_id : str, optional
            Resource ID for the Event
        creation_time : str or UTCDateTime, optional
            Event creation time
        event_type : str, optional
            QuakeML-compatible event type
        mainclass : str, optional
            High-level classification (e.g., "LV")
        subclass : str, optional
            Detailed classification (e.g., "hybrid")

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

        # Event type and origin
        if event_type:
            event.event_type = event_type
        origin = Origin(
            latitude=source_coords['latitude'],
            longitude=source_coords['longitude'],
            depth=source_coords.get('depth', 0)
        )
        event.origins.append(origin)

        # Add network magnitudes
        df = self.magnitudes2dataframe()
        if df is not None and not df.empty:
            net_ml = df.iloc[-1].get('network_mean_ML')
            net_me = df.iloc[-1].get('network_mean_ME')
            if pd.notnull(net_ml):
                event.magnitudes.append(Magnitude(
                    mag=net_ml, magnitude_type="ML", origin_id=origin.resource_id))
            if pd.notnull(net_me):
                event.magnitudes.append(Magnitude(
                    mag=net_me, magnitude_type="Me", origin_id=origin.resource_id))

        # Add station magnitudes
        for tr in self:
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
                event.magnitudes[0].station_magnitude_contributions.append(
                    StationMagnitudeContribution(
                        station_magnitude_id=smag.resource_id, weight=1.0))

        # Add trace-level metrics as a comment
        try:
            metrics_dict = {tr.id: tr.stats.metrics for tr in self if hasattr(tr.stats, 'metrics')}
            event.comments.append(Comment(
                text=json.dumps(metrics_dict, indent=2), force_resource_id=False))
        except Exception:
            pass

        # Add classification
        if mainclass or subclass:
            class_comment = {"mainclass": mainclass, "subclass": subclass}
            event.comments.append(Comment(
                text=json.dumps(class_comment, indent=2), force_resource_id=False))

        return event


    def to_enhancedevent(self, event_id=None, dfile=None, origin_time=None):
        """
        Convert EnhancedStream to an EnhancedEvent object.

        Parameters
        ----------
        event_id : str
            Event ID this stream belongs to.
        dfile : str
            Filename of associated waveform file.
        origin_time : UTCDateTime
            Time of event origin (optional).

        Returns
        -------
        EnhancedEvent
        """
        event = EnhancedEvent()
        event.event = self.to_obspyevent()
        event.event_id = event_id
        event.dfile = dfile
        event.origin_time = origin_time
        event.traces = []

        for tr in self:
            trace_info = {
                'id': tr.id,
                'network': tr.stats.network,
                'station': tr.stats.station,
                'location': tr.stats.location,
                'channel': tr.stats.channel,
                'starttime': str(tr.stats.starttime),
                'sampling_rate': tr.stats.sampling_rate,
                'distance_m': tr.stats.get('distance'),
                'metrics': tr.stats.get('metrics', {}),
                'spectral': tr.stats.get('spectral', {}),
            }
            event.traces.append(trace_info)

        return event


########################## METRICS FUNCTIONS FOLLOW ######################

from collections import defaultdict
import numpy as np
from numpy.fft import rfft, rfftfreq

_COMPONENT_SETS = [
    ("Z","N","E"),
    ("Z","1","2"),
    ("Z","R","T"),
]

def _trim_to_overlap(traces):
    if not traces:
        return []
    t0 = max(tr.stats.starttime for tr in traces)
    t1 = min(tr.stats.endtime   for tr in traces)
    if t1 <= t0:
        return []
    out = []
    for tr in traces:
        trc = tr.copy().trim(t0, t1, pad=False)
        if trc.stats.npts <= 1:
            return []
        out.append(trc)
    return out

def _vector_max_3c(trZ, trN, trE, *, kind="vel"):
    def as_kind(tr):
        if kind == "vel":  return tr.copy()
        if kind == "disp": return tr.copy().integrate()
        if kind == "acc":  return tr.copy().differentiate()
        raise ValueError("kind must be vel|disp|acc")
    trs = _trim_to_overlap([as_kind(trZ), as_kind(trN), as_kind(trE)])
    if len(trs) != 3:
        return np.nan
    z, n, e = (np.asarray(t.data, dtype=np.float64) for t in trs)
    vec = np.sqrt(z*z + n*n + e*e)
    return float(np.nanmax(np.abs(vec))) if vec.size else np.nan

def _group_by_station_band(stream):
    groups = defaultdict(lambda: defaultdict(list))
    for tr in stream:
        net, sta, loc, cha = tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel
        if not cha or len(cha) < 3:
            continue
        band2 = cha[:2].upper()   # e.g., 'HH','BH','HD','BD'
        comp  = cha[-1].upper()
        key = (net, sta, loc, band2)
        groups[key][comp].append(tr)
    return groups

# ---------- time-domain "SSAM-lite" ----------
def _td_rsam(tr, f1, f2, *, stat="mean_abs", corners=2, zerophase=True):
    try:
        trf = tr.copy().filter("bandpass", freqmin=float(f1), freqmax=float(f2),
                               corners=int(corners), zerophase=bool(zerophase))
        y = trf.data.astype(np.float64)
        if not y.size:
            return np.nan
        if stat == "median_abs":
            return float(np.nanmedian(np.abs(y)))
        if stat == "rms":
            return float(np.sqrt(np.nanmean(y*y)))
        return float(np.nanmean(np.abs(y)))
    except Exception:
        return np.nan

def _td_band_ratio(tr, low=(0.5,2.0), high=(2.0,8.0), *, stat="mean_abs", log2=True):
    a1,b1 = low; a2,b2 = high
    r_low  = _td_rsam(tr, a1, b1, stat=stat)
    r_high = _td_rsam(tr, a2, b2, stat=stat)
    if not np.isfinite(r_low) or not np.isfinite(r_high) or r_low <= 0:
        ratio = np.nan
    else:
        ratio = r_high / r_low
        if log2 and ratio > 0:
            ratio = float(np.log2(ratio))
    return {
        "RSAM_low":  r_low,
        "RSAM_high": r_high,
        "ratio":     ratio,
        "ratio_key": f"bandratio_{a1}_{b1}__{a2}_{b2}" + ("_log2" if log2 else ""),
    }

# ---------- spectral block (optional) ----------
def _spectral_block(tr, y, dt, threshold, window_length, polyorder, compute_ssam, compute_bandratios):
    if tr.stats.npts < 2:
        return
    Y = np.abs(rfft(y))
    F = rfftfreq(tr.stats.npts, d=dt)
    if not hasattr(tr.stats, "spectral") or tr.stats.spectral is None:
        tr.stats.spectral = {}
    tr.stats.spectral["freqs"] = F
    tr.stats.spectral["amplitudes"] = Y
    try:
        get_bandwidth(F, Y, threshold=threshold, window_length=window_length,
                      polyorder=polyorder, trace=tr)
    except Exception as e:
        print(f"[{tr.id}] Skipping bandwidth metrics: {e}")
    if compute_ssam:
        try:
            _ssam(tr)
        except Exception as e:
            print(f"[{tr.id}] SSAM computation failed: {e}")
    if compute_bandratios:
        try:
            _band_ratio(tr, freqlims=[1.0, 6.0, 11.0])
            _band_ratio(tr, freqlims=[0.5, 3.0, 18.0])
        except Exception as e:
            print(f"[{tr.id}] Band ratio computation failed: {e}")
    try:
        A = tr.stats.spectral["amplitudes"]
        if A.size and np.any(A > 0):
            peak_idx = int(np.nanargmax(A))
            peakf = float(F[peak_idx])
            meanf = float(np.nansum(F * A) / np.nansum(A))
        else:
            peakf = meanf = np.nan
        m = tr.stats.metrics
        m["peakf"]   = peakf
        m["meanf"]   = meanf
        m["medianf"] = float(np.nanmedian(F[A > 0])) if np.any(A > 0) else np.nan
        s = getattr(tr.stats, "spectrum", {})
        s["peakF"]   = peakf
        s["medianF"] = m["medianf"]
        s["peakA"]   = float(np.nanmax(A)) if A.size else np.nan
        tr.stats.spectrum = s
    except Exception as e:
        print(f"[{tr.id}] Spectral summary failed: {e}")

# ---------- pressure metrics ----------
def _pressure_metrics(tr, band=(1.0, 20.0)):
    y = np.asarray(tr.data, dtype=np.float64)
    pap = float(np.nanmax(np.abs(y))) if y.size else np.nan
    try:
        trf = tr.copy().filter("bandpass", freqmin=band[0], freqmax=band[1],
                               corners=2, zerophase=True)
        yf = np.asarray(trf.data, dtype=np.float64)
        pap_band = float(np.nanmax(np.abs(yf))) if yf.size else np.nan
    except Exception:
        pap_band = np.nan
    return pap, pap_band