import os
import json
import pickle
import numpy as np
import pandas as pd
from math import pi

from scipy.signal import savgol_filter
from scipy.stats import describe
from collections import defaultdict

import matplotlib.pyplot as plt

from obspy import Stream, Trace, read, UTCDateTime, read_events
from obspy.core.event import (
    Event, Origin, Magnitude, ResourceIdentifier, Comment,
    CreationInfo, StationMagnitude, StationMagnitudeContribution,
    WaveformStreamID, Amplitude, Catalog
)
from obspy.core.util.attribdict import AttribDict
from obspy.geodetics.base import gps2dist_azimuth

from flovopy.core.preprocessing import add_to_trace_history
from flovopy.processing.metrics import compute_amplitude_spectra #, compute_stationEnergy
from flovopy.seisanio.core.ampengfft import compute_ampengfft_stream, write_aef_file
class EnhancedStream(Stream):

    def __init__(self, stream=None):
        if stream is None:
            stream = Stream()
        super().__init__(traces=stream.traces)



    def legacy_ampengfft(self, filepath, freq_bins=None, amp_avg_window=2.0,
                                trigger_window=None, average_window=None):
        compute_ampengfft_stream(self, freq_bins=freq_bins, amp_avg_window=amp_avg_window)
        write_aef_file(self, filepath, trigger_window=trigger_window, average_window=average_window)


    def ampengfft(self, threshold=0.707, window_length=9, polyorder=2, differentiate=True):
        """
        Compute amplitude, energy, and frequency metrics for all traces in a stream.
        Note that we generally will want stream to be a displacement seismogram, and 
        then compute amplitude on this, and differentiate to compute energy. This is because
        we usually want to estimate a magnitude based on a displacement amplitude, but we 
        compute energy magnitude based on a velocity seismogram.
        However, this function by default will also then use displacement seismograms for
        all the frequency metrics, which effectively divides by frequency. If this is not
        the behaviour wanted, pass a velocity seismogram and pass differentiate=False.

        For each trace:
        - Computes FFT and amplitude spectrum.
        - Stores frequency and amplitude in trace.stats.spectral.
        - Computes bandwidth metrics using get_bandwidth().
        - Computes SSAM band-averaged amplitudes.
        - Computes dominant and mean frequency.
        - Computes band ratio metrics.
        - Computes amplitude, energy, and scipy.stats metrics.
        - Stores all results in trace.stats.metrics

        Parameters
        ----------
        stream : obspy.Stream
            Stream of Trace objects to analyze.
        threshold : float, optional
            Fraction of peak amplitude to define bandwidth cutoff (default is 0.707 for -3 dB).
        window_length : int, optional
            Length of Savitzky-Golay smoothing window (must be odd).
        polyorder : int, optional
            Polynomial order for smoothing filter.
        differentiate: bool, optional
            If stream contains displacement traces, then we want to differentiate to a velocity traces to compute energy

        Returns
        -------
        stream : obspy.Stream
            Modified stream with frequency metrics stored per trace.
        """
        if not isinstance(stream, Stream):
            if isinstance(stream, Trace):
                stream = Stream([stream])
            else:
                return
            
        if len(stream)==0:
            return
        
        if not 'detrended' in stream[0].stats.history: # change this to use better detrend from libseis?
            for tr in stream:
                tr.detrend(type='linear')
                add_to_trace_history(tr, 'detrended')    

        if not hasattr(stream[0].stats, 'spectral'):
            compute_amplitude_spectra(stream)

        for tr in self:
            dt = tr.stats.delta
            N = len(tr.data)
            fft_vals = np.fft.fft(tr.data)
            freqs = np.fft.fftfreq(N, d=dt)
            amps = np.abs(fft_vals)

            pos_freqs = freqs[:N//2]
            pos_amps = amps[:N//2]

            if not hasattr(tr.stats, 'spectral'):
                tr.stats.spectral = {}
            tr.stats.spectral['freqs'] = pos_freqs
            tr.stats.spectral['amplitudes'] = pos_amps

            if not hasattr(tr.stats, 'metrics'):
                tr.stats.metrics = {}

            # Amplitude and energy metrics
            y = np.abs(tr.data)
            tr.stats.metrics['peakamp'] = np.max(y)
            tr.stats.metrics['peaktime'] = np.argmax(y) / tr.stats.sampling_rate + tr.stats.starttime
            if differentiate:
                y = np.diff(y)
            tr.stats.metrics['energy'] = np.sum(y**2) / tr.stats.sampling_rate

            # Magnitude - should be a separate after function
            # we need to convert tr.stats.metrics['energy'] to a source energy, Eseismic
            # based on source and station location, and an attenuation/decay model
            #tr.stats.metrics['Me'] = Eseismic2magnitude(Eseismic, correction=3.7)

            # Bandwidth
            try:
                get_bandwidth(pos_freqs, pos_amps,
                            threshold=threshold,
                            window_length=window_length,
                            polyorder=polyorder,
                            trace=tr)
            except Exception as e:
                print(f"[{tr.id}] Skipping bandwidth metrics: {e}")

            # SSAM
            try:
                _ssam(tr)
            except Exception as e:
                print(f"[{tr.id}] SSAM computation failed: {e}")

            # Band ratios
            try:
                _band_ratio(tr, freqlims=[1.0, 6.0, 11.0])
                _band_ratio(tr, freqlims=[0.5, 3.0, 18.0])
            except Exception as e:
                print(f"[{tr.id}] Band ratio computation failed: {e}")

            # Dominant and mean frequency
            try:
                f = tr.stats.spectral['freqs']
                A = tr.stats.spectral['amplitudes']
                tr.stats.metrics['peakf'] = f[np.argmax(A)]
                tr.stats.metrics['meanf'] = np.sum(f * A) / np.sum(A) if np.sum(A) > 0 else np.nan
            except Exception as e:
                print(f"[{tr.id}] Dominant/mean frequency computation failed: {e}")

            # Scipy descriptive stats
            try:
                from scipy.stats import describe
                stats = describe(tr.data, nan_policy='omit')._asdict()
                tr.stats.metrics['skewness'] = stats['skewness']
                tr.stats.metrics['kurtosis'] = stats['kurtosis']
            except Exception as e:
                print(f"[{tr.id}] scipy.stats failed: {e}")

            add_to_trace_history(tr, 'ampengfft')


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
    def read(cls, basepath):
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

    def summarize_magnitudes(self, include_network=True):
        """
        Summarize trace-level and network-averaged magnitude values.

        Parameters
        ----------
        include_network : bool
            If True, append a summary row with network-average magnitudes.

        Returns
        -------
        df : pandas.DataFrame
            Columns: id, starttime, distance, local_magnitude (ML), energy_magnitude (ME),
                    plus network_mean_* and network_std_* if include_network is True.
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

        if include_network:
            network_stats = {}
            for col in ['ML', 'ME']:
                valid = df[col].dropna()
                if not valid.empty:
                    network_stats[f'network_mean_{col}'] = valid.mean()
                    network_stats[f'network_mean_{col}'] = valid.median()
                    network_stats[f'network_std_{col}'] = valid.std()
                    network_stats[f'network_n_{col}'] = valid.count()

            # Add a summary row at the bottom
            summary_row = {col: np.nan for col in df.columns}
            summary_row.update(network_stats)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

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

        if snr_min is not None:
            if verbose:
                print(f"[3] Filtering traces with SNR < {snr_min:.1f}...")
            filtered = [tr for tr in self if tr.stats.metrics.get('snr', 0) >= snr_min]
            self._traces = filtered

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
        df = self.summarize_magnitudes(include_network=True)

        if verbose and not df.empty:
            net_ml = df.iloc[-1].get('network_mean_ML', np.nan)
            net_me = df.iloc[-1].get('network_mean_ME', np.nan)
            print(f"    → Network ML: {net_ml:.2f} | Network ME: {net_me:.2f}")

        return self, df

    def build_event(self, source_coords, event_id=None, creation_time=None,
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
        df = self.summarize_magnitudes(include_network=True)
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

########################## METRICS FUNCTIONS FOLLOW ######################

def _ssam(tr, freq_bins=None):
    """
    Compute single-station amplitude measurements (SSAM) by binning
    the amplitude spectrum already stored in tr.stats.spectral.

    Parameters
    ----------
    tr : obspy.Trace
        Trace with .stats.spectral['freqs'] and ['amplitudes'].
    freq_bins : array-like, optional
        Frequency bin edges (e.g., np.arange(0, 16, 1.0)).
        Default is 0–16 Hz in 1-Hz bins.

    Returns
    -------
    None
        Stores {'f': bin_centers, 'A': band_averaged_amplitudes} in tr.stats.metrics.ssam
    """
    if freq_bins is None:
        freq_bins = np.arange(0.0, 16.0, 1.0)

    if not hasattr(tr.stats, 'spectral'):
        print(f"[{tr.id}] Missing tr.stats.spectral — run compute_amplitude_spectra() first.")
        return

    f = tr.stats.spectral.get('freqs')
    A = tr.stats.spectral.get('amplitudes')

    if f is None or A is None:
        print(f"[{tr.id}] Missing spectral frequencies or amplitudes.")
        return

    f = np.asarray(f)
    A = np.asarray(A)

    bin_centers = []
    ssam_values = []

    for i in range(len(freq_bins) - 1):
        fmin = freq_bins[i]
        fmax = freq_bins[i+1]
        idx = np.where((f >= fmin) & (f < fmax))[0]

        bin_centers.append((fmin + fmax) / 2.0)
        ssam_values.append(np.nanmean(A[idx]) if idx.size else np.nan)

    tr.stats.metrics['ssam'] = {
        'f': np.array(bin_centers),
        'A': np.array(ssam_values)
    }


    
def _band_ratio(tr, freqlims=[1, 6, 11]):
    """
    Compute band ratio as log2(amplitude above split frequency / amplitude below),
    using frequency limits defined by freqlims.

    Parameters
    ----------
    tr : obspy.Trace
        Trace object with spectral data.
    freqlims : list of float
        Frequency limits in Hz: [low, split, high]
        Some values used before are:
            [1, 6, 11]
            [0.8, 4, 18]
        Better values might be:
            [0.5, 4.0, 32.0]
            [0.5, 3.0, 18.0]
        Or choose two sets of values to better differentiate LPs from hybrids from VTs?
            [0.5, 2.0, 8.0]  LF vs HF
            [2.0, 6.0, 18.0] hybrid vs VT?
        Need to play around with data to determine     

    Stores
    -------
    tr.stats.metrics.bandratio : list of dict
        Appends a dictionary with 'freqlims', 'RSAM_low', 'RSAM_high', 'RSAM_ratio'.
    """
    A = None
    f = None

    # Preferred: new spectral storage
    if hasattr(tr.stats, 'spectral'):
        f = tr.stats.spectral.get('freqs')
        A = tr.stats.spectral.get('amplitudes')

    # Legacy support
    elif hasattr(tr.stats, 'spectrum'):
        f = tr.stats.spectrum.get('F')
        A = tr.stats.spectrum.get('A')

    elif hasattr(tr.stats, 'ssam'):
        f = tr.stats.ssam.get('f')
        A = np.array(tr.stats.ssam.get('A'))

    # Proceed if valid spectral data found
    if A is not None and f is not None and len(A) > 0:
        f = np.array(f)
        A = np.array(A)

        idx_low = np.where((f > freqlims[0]) & (f < freqlims[1]))[0]
        idx_high = np.where((f > freqlims[1]) & (f < freqlims[2]))[0]

        A_low = A[idx_low]
        A_high = A[idx_high]

        sum_low = np.sum(A_low)
        sum_high = np.sum(A_high)

        ratio = np.log2(sum_high / sum_low) if sum_low > 0 else np.nan

        br = {
            'freqlims': freqlims,
            'RSAM_low': sum_low,
            'RSAM_high': sum_high,
            'RSAM_ratio': ratio
        }

        if not hasattr(tr.stats.metrics, 'bandratio'):
            tr.stats.metrics.bandratio = []

        tr.stats.metrics.bandratio.append(br)

def get_bandwidth(frequencies, amplitudes, threshold=0.707,
                  window_length=9, polyorder=2, trace=None):
    """
    Estimate peak frequency, bandwidth, and cutoff frequencies from
    a smoothed amplitude ratio spectrum. Optionally store results in
    trace.stats.metrics if a Trace is provided.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency values (Hz).
    amplitudes : np.ndarray
        Amplitude (or amplitude ratio) spectrum.
    threshold : float
        Fraction of peak amplitude to define bandwidth cutoff (e.g. 0.707 for -3 dB).
    window_length : int
        Smoothing window length for Savitzky-Golay filter.
    polyorder : int
        Polynomial order for Savitzky-Golay filter.
    trace : obspy.Trace, optional
        If given, store metrics in trace.stats.metrics.

    Returns
    -------
    dict
        Dictionary with keys: 'f_peak', 'A_peak', 'f_low', 'f_high', 'bandwidth'
    """
    smoothed = savgol_filter(amplitudes, window_length=window_length, polyorder=polyorder)

    peak_index = np.argmax(smoothed)
    f_peak = frequencies[peak_index]
    A_peak = smoothed[peak_index]
    cutoff_level = A_peak * threshold

    lower = np.where(smoothed[:peak_index] < cutoff_level)[0]
    f_low = frequencies[lower[-1]] if lower.size else frequencies[0]

    upper = np.where(smoothed[peak_index:] < cutoff_level)[0]
    f_high = frequencies[peak_index + upper[0]] if upper.size else frequencies[-1]

    bandwidth = f_high - f_low

    metrics = {
        "f_peak": f_peak,
        "A_peak": A_peak,
        "f_low": f_low,
        "f_high": f_high,
        "bandwidth": bandwidth
    }

    if trace is not None:
        if not hasattr(trace.stats, 'metrics') or not isinstance(trace.stats.metrics, dict):
            trace.stats.metrics = {}
        for key, val in metrics.items():
            trace.stats.metrics[key] = val

    return metrics



def estimate_distance(trace, source_coords):
    """
    Compute hypocentral distance R (in meters) from trace coordinates to source.

    Parameters
    ----------
    trace : obspy.Trace
        Must have .stats.coordinates = {'latitude': ..., 'longitude': ..., 'elevation': ...}
    source_coords : dict
        {'latitude': ..., 'longitude': ..., 'depth': ...} in meters

    Returns
    -------
    R : float
        Hypocentral distance in meters
    """
    sta = trace.stats.coordinates
    lat1, lon1, elev = sta['latitude'], sta['longitude'], sta.get('elevation', 0)
    lat2, lon2, depth = source_coords['latitude'], source_coords['longitude'], source_coords.get('depth', 0)

    epic_dist, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dz = (depth + elev)  # add elevation since depth is below surface
    return np.sqrt(epic_dist**2 + dz**2)



def estimate_source_energy(trace, R, model='body', Q=50, c_earth=2500):
    """
    Estimate source energy by correcting station energy for geometric spreading and attenuation.

    Parameters
    ----------
    trace : obspy.Trace
        Must have .stats.metrics['energy'] and .stats.spectral['freqs'] / 'peakF'
    R : float
        Distance in meters.
    model : str
        'body' or 'surface'
    Q : float
        Quality factor for attenuation.
    c_earth : float
        Seismic wave speed (m/s)

    Returns
    -------
    Eseismic : float
        Estimated source energy in Joules
    """
    E_obs = trace.stats.metrics.get('energy')
    if E_obs is None:
        return None

    # Use peakF if available
    f_peak = None
    if 'peakF' in trace.stats.spectral:
        f_peak = trace.stats.spectral['peakF']
    else:
        freqs = trace.stats.spectral.get('freqs')
        amps = trace.stats.spectral.get('amplitudes')
        if freqs is not None and amps is not None:
            f_peak = freqs[np.argmax(amps)]

    if f_peak is None:
        return None

    # Attenuation factor
    A_att = np.exp(-np.pi * f_peak * R / (Q * c_earth))

    # Geometric spreading correction
    if model == 'body':
        geom = R**2
    elif model == 'surface':
        wavelength = c_earth / f_peak
        geom = R * wavelength
    else:
        raise ValueError("Model must be 'body' or 'surface'.")

    Eseismic = E_obs * geom / A_att  # undo attenuation and spreading
    return Eseismic

def estimate_local_magnitude(trace, R_km, a=1.6, b=-0.15, g=0):
    """
    Estimate and store ML from peak amplitude and distance.

    Parameters
    ----------
    trace : obspy.Trace
    R_km : float
        Distance from source in km
    a, b, g : float
        Richter scaling parameters

    Returns
    -------
    ml : float
    """
    peakamp = trace.stats.metrics.get('peakamp')
    if peakamp is None or R_km <= 0:
        return None

    ml = Mlrichter(peakamp, R_km, a=a, b=b, g=g)
    trace.stats.metrics['local_magnitude'] = ml
    return ml

def Eseismic_Boatwright(val, R, rho_earth=2000, c_earth=2500, S=1.0, A=1.0):
    # val can be a Stream, Trace, a stationEnergy or a list of stationEnergy
    # R in m
    # Following values assumed by Johnson and Aster, 2005:
    # rho_earth 2000 kg/m^3
    # c_earth 2500 m/s
    # A is attenuation = 1
    # S is site response = 1
    #
    # These equations seem to be valid for body waves only, that spread like hemispherical waves in a flat earth.
    # But if surface waves dominate, they would spread like ripples on a pond, so energy density of wavefront like 2*pi*R
    if isinstance(val,Stream): # Stream
        Eseismic = []
        for tr in val:
            Eseismic.append(Eseismic_Boatwright(tr, R, rho_earth, c_earth, S, A))
    elif isinstance(val,Trace): # Trace
        stationEnergy = compute_stationEnergy(val) 
        Eseismic = Eseismic_Boatwright(stationEnergy, R, rho_earth, c_earth, S, A)
    elif isinstance(val, list): # list of stationEnergy
        Eseismic = []
        for thisval in val:
            Eseismic.append(Eseismic_Boatwright(thisval, R, rho_earth, c_earth, S, A))
    else: # stationEnergy
        Eseismic = 2 * pi * (R ** 2) * rho_earth * c_earth * (S ** 2) * val / A 
    return Eseismic

def Eacoustic_Boatwright(val, R, rho_atmos=1.2, c_atmos=340, z=100000):
    # val can be a Stream, Trace, a stationEnergy or a list of stationEnergy
    # R in m
    # Following values assumed by Johnson and Aster, 2005:
    # rho_atmos 1.2 kg/m^3
    # c_atmos 340 m/s  
    # z is just an estimate of the atmospheric vertical scale length - the height of ripples of infrasound energy spreading globally
    if isinstance(val,Stream): # Stream
        Eacoustic = []
        for tr in val:
            Eacoustic.append(Eacoustic_Boatwright(tr, R, rho_atmos, c_atmos))
    elif isinstance(val, Trace): # Trace
        stationEnergy = compute_stationEnergy(val) 
        Eacoustic = Eacoustic_Boatwright(stationEnergy, R, rho_atmos, c_atmos)
    elif isinstance(val, list): # list of stationEnergy
        for thisval in val:
            Eacoustic.append(Eacoustic_Boatwright(thisval, R, rho_atmos, c_atmos, S, A))
    else:
        if R > 100000: # beyond distance z (e.g. 100 km), assume spreading like 2*pi*R
            E_if_station_were_at_z = 2 * pi * (z ** 2) / (rho_atmos * c_atmos) * val
            Eacoustic = E_if_station_were_at_z* R/1e5
        else:
            Eacoustic = 2 * pi * R ** 2 / (rho_atmos * c_atmos) * val
    return Eacoustic

def Eseismic2magnitude(Eseismic, correction=3.7):
    # after equation 7 in Hanks and Kanamori 1979, where moment is substitute with energy
    # energy in Joules rather than ergs, so correction is 3.7 rather than 10.7
    if isinstance(Eseismic, list): # list of stationEnergy
        mag = [] 
        for thisE in Eseismic:
            mag.append(Eseismic2magnitude(thisE, correction=correction))
    else:
        mag = np.log10(Eseismic)/1.5 - correction
    return mag

def magnitude2Eseismic(mag, correction=3.7):
    # after equation 7 in Hanks and Kanamori 1979, where moment is substitute with energy
    # energy in Joules rather than ergs, so correction is 3.7 rather than 10.7   
    if isinstance(mag, list): # list of stationEnergy
        Eseismic = [] 
        for thismag in mag:
            Eseismic.append(magnitude2Eseismic(thismag, correction=correction))
    else:
        Eseismic = np.power(10, 1.5 * mag + correction)
    return Eseismic

def Mlrichter(peakA, R, a=1.6, b=-0.15, g=0):
    """
    Compute Richter local magnitude (ML) from peak amplitude and distance.

    Parameters
    ----------
    peakA : float
        Peak amplitude (in mm or nm depending on calibration).
    R : float
        Epicentral distance in km.
    a : float
        Log-distance scaling (default 1.6).
    b : float
        Offset (default -0.15).
    g : float
        Station correction (default 0).

    Returns
    -------
    ml : float
        Local magnitude.
    """
    return np.log10(peakA) + a * np.log10(R) + b + g


class EventContainer:
    """
    A container for storing information about a single volcano-seismic event,
    including the ObsPy Event object, waveform Stream, trigger dictionary,
    classification label, and associated miniSEED file path.
    """
    def __init__(self, event, stream=None, trigger=None, classification=None, miniseedfile=None):
        self.event = event
        self.stream = stream
        self.trigger = trigger
        self.classification = classification
        self.miniseedfile = miniseedfile

    def to_enhancedevent(self):
        return EnhancedEvent(
            event=self.event,
            stream=self.stream,
            sfile_path=None,  # Optional
            wav_paths=[self.miniseedfile] if self.miniseedfile else [],
            trigger_window=self.trigger['trigger_window'] if self.trigger else None,
            average_window=self.trigger['average_window'] if self.trigger else None,
            metrics={}  # or extract from stream.stats.metrics
        )


class EnhancedCatalog(Catalog):
    """
    Extended ObsPy Catalog class for volcano-seismic workflows.

    Stores:
    - Event metadata (ObsPy Event objects)
    - Associated waveform streams
    - Trigger metadata and parameters
    - Classifications, QuakeML extras (event_type, mainclass, subclass)
    - Save/load routines and plotting utilities
    """
    def __init__(self, events=None, records=None, triggerParams=None, starttime=None, endtime=None,
                 comments=None, description=None, resource_id=None, creation_info=None):
        super().__init__(events=events or [])
        self.records = records or []
        self.triggerParams = triggerParams or {}
        self.starttime = starttime
        self.endtime = endtime
        self.comments = comments or []
        self.description = description or ""
        self._df_cache = None  # Internal cache for DataFrame representation
        self._set_resource_id(resource_id)
        self._set_creation_info(creation_info)

    @property
    def dataframe(self):
        """
        Cached pandas.DataFrame of catalog event metadata.
        Use `update_dataframe()` to refresh after modifying the catalog.
        """
        if self._df_cache is None:
            self.update_dataframe()
        return self._df_cache

    def update_dataframe(self, force=False):
        """
        Refresh internal DataFrame representation of catalog events.

        Parameters
        ----------
        force : bool, optional
            If True, update even if cache already exists.

        Returns
        -------
        None
            Updates self._df_cache.
        """
        if force or self._df_cache is None:
            self._df_cache = self.to_dataframe()


    def addEvent(self, event, stream=None, trigger=None, classification=None,
                 event_type=None, mainclass=None, subclass=None,
                 author="EnhancedCatalog", agency_id="MVO"):
        """
        Add an ObsPy Event to the catalog with optional metadata and waveform stream.
        """
        self.append(event)
        if event_type:
            event.event_type = event_type
            event.event_type_certainty = "suspected"
        if mainclass:
            event.comments.append(Comment(text=f"mainclass: {mainclass}"))
        if subclass:
            event.comments.append(Comment(text=f"subclass: {subclass}"))
        if classification:
            event.comments.append(Comment(text=f"classification: {classification}"))
        if not event.creation_info:
            event.creation_info = CreationInfo(
                author=author, agency_id=agency_id, creation_time=UTCDateTime()
            )
        #self.records.append(EventContainer(event, stream, trigger, classification))
        self.records.append(EnhancedEvent(
            obspy_event=event,
            stream=stream,
            sfile_path=None,  # or pass it if available
            wav_paths=[s.path for s in stream] if stream else [],
            aef_path=None,
            trigger_window=trigger.get('duration') if trigger else None,
            average_window=trigger.get('average_window') if trigger else None,
            metrics={
                "classification": classification,
                "event_type": event.event_type,
                "mainclass": mainclass,
                "subclass": subclass,
            }
        ))



    def get_times(self):
        """Return a list of origin times for events that have origins."""
        return [ev.origins[0].time for ev in self if ev.origins]

    def plot_streams(self):
        """Plot all waveform streams associated with the catalog."""
        for i, rec in enumerate(self.records):
            if rec.stream:
                print(f"\nEVENT NUMBER: {i+1}  time: {rec.stream[0].stats.starttime}\n")
                rec.stream.plot(equal_scale=False)

    def concat(self, other):
        """Merge another EnhancedCatalog into this one."""
        for ev in other:
            self.append(ev)
        self.records.extend(other.records)

    def merge(self, other):
        """
        Merge another EnhancedCatalog into this one, avoiding duplicate events.
        """
        existing_ids = {e.resource_id.id for e in self}
        for rec in other.records:
            if rec.event.resource_id.id not in existing_ids:
                self.append(rec.event)
                self.records.append(rec)

    def summary(self):
        """
        Print a summary of event counts grouped by event_type, mainclass, and subclass.
        """
        df = self.to_dataframe()
        print("\nEvent Type Summary:")
        print(df['event_type'].value_counts(dropna=False))
        print("\nMainclass Summary:")
        print(df['mainclass'].value_counts(dropna=False))
        print("\nSubclass Summary:")
        print(df['subclass'].value_counts(dropna=False))


    def to_dataframe(self):
        """
        Convert catalog metadata to a pandas DataFrame including magnitude,
        origin coordinates, classification, and QuakeML metadata fields.
        """
        rows = []
        for rec in self.records:
            row = {
                'datetime': rec.event.origins[0].time.datetime if rec.event.origins else None,
                'magnitude': rec.event.magnitudes[0].mag if rec.event.magnitudes else None,
                'latitude': rec.event.origins[0].latitude if rec.event.origins else None,
                'longitude': rec.event.origins[0].longitude if rec.event.origins else None,
                'depth': rec.event.origins[0].depth if rec.event.origins else None,
                'duration': rec.trigger['duration'] if rec.trigger and 'duration' in rec.trigger else None,
                'classification': rec.classification,
                'filename': rec.miniseedfile
            }
            if rec.event.event_type:
                row['event_type'] = rec.event.event_type
            for c in rec.event.comments:
                if 'mainclass:' in c.text:
                    row['mainclass'] = c.text.split(':', 1)[-1].strip()
                elif 'subclass:' in c.text:
                    row['subclass'] = c.text.split(':', 1)[-1].strip()
                elif 'classification:' in c.text and not row.get('classification'):
                    row['classification'] = c.text.split(':', 1)[-1].strip()
            if row['magnitude'] is not None:
                row['energy'] = magnitude2Eseismic(row['magnitude'])
            rows.append(row)
        return pd.DataFrame(rows)

    '''
    def save(self, outdir, outfile, net='MV'):
        """Write QuakeML and associated stream metadata to disk."""
        self.write_events(outdir, net=net, xmlfile=outfile + '.xml')
        pklfile = os.path.join(outdir, outfile + '_vars.pkl')
        picklevars = {
            'records': self.records,
            'triggerParams': self.triggerParams,
            'comments': self.comments,
            'description': self.description,
            'starttime': self.starttime.strftime('%Y/%m/%d %H:%M:%S') if self.starttime else None,
            'endtime': self.endtime.strftime('%Y/%m/%d %H:%M:%S') if self.endtime else None
        }
        try:
            with open(pklfile, "wb") as fileptr:
                pickle.dump(picklevars, fileptr, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)
    '''

    def save(self, outdir, outfile, net='MV'):
        """Write catalog-wide QuakeML + JSON for each event and save summary metadata."""
        self._write_events(outdir, net=net, xmlfile=outfile + '.xml')

        summary_json = os.path.join(outdir, outfile + '_meta.json')
        summary_vars = {
            'triggerParams': self.triggerParams,
            'comments': self.comments,
            'description': self.description,
            'starttime': self.starttime.strftime('%Y/%m/%d %H:%M:%S') if self.starttime else None,
            'endtime': self.endtime.strftime('%Y/%m/%d %H:%M:%S') if self.endtime else None
        }
        try:
            with open(summary_json, "w") as f:
                json.dump(summary_vars, f, indent=2, default=str)
        except Exception as ex:
            print("Error saving catalog summary JSON:", ex)

    def export_csv(self, filepath):
        """
        Export catalog metadata and classification fields to a CSV file.

        Parameters
        ----------
        filepath : str
            Path to save the CSV file.
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def _write_events(self, outdir, net='MV', xmlfile=None):
        """Write event metadata and associated streams to disk."""
        if xmlfile:
            self.write(os.path.join(outdir, xmlfile), format="QUAKEML")
        for rec in self.records:
            if not hasattr(rec, "to_enhancedevent"):
                continue
            enh = rec.to_enhancedevent()
            t = enh.event.origins[0].time if enh.event.origins else None
            if not t:
                continue
            base_path = os.path.join(outdir, 'WAV', net, t.strftime('%Y'), t.strftime('%m'), t.strftime('%Y%m%dT%H%M%S'))
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            enh.save(os.path.dirname(base_path), os.path.basename(base_path))

    def filter_by_event_type(self, event_type):
        """Return new catalog filtered by QuakeML event_type."""
        mask = [rec.event.event_type == event_type for rec in self.records]
        filtered_records = [rec for rec, m in zip(self.records, mask) if m]
        filtered_events = [rec.event for rec in filtered_records]
        return EnhancedCatalog(events=filtered_events, records=filtered_records)

    def filter_by_mainclass(self, mainclass):
        """Return new catalog filtered by 'mainclass' in comments."""
        def match_mainclass(event):
            return any('mainclass:' in c.text and mainclass in c.text for c in event.comments)
        filtered_records = [rec for rec in self.records if match_mainclass(rec.event)]
        filtered_events = [rec.event for rec in filtered_records]
        return EnhancedCatalog(events=filtered_events, records=filtered_records)

    def filter_by_subclass(self, subclass):
        """Return new catalog filtered by 'subclass' in comments."""
        def match_subclass(event):
            return any('subclass:' in c.text and subclass in c.text for c in event.comments)
        filtered_records = [rec for rec in self.records if match_subclass(rec.event)]
        filtered_events = [rec.event for rec in filtered_records]
        return EnhancedCatalog(events=filtered_events, records=filtered_records)

    def group_by(self, field):
        """
        Group catalog into multiple subcatalogs by a classification field.

        Parameters
        ----------
        field : str
            One of 'event_type', 'mainclass', or 'subclass'.

        Returns
        -------
        dict
            Dictionary where keys are unique field values, values are EnhancedCatalogs.
        """
        grouped = defaultdict(list)
        for rec in self.records:
            value = None
            if field == 'event_type':
                value = rec.event.event_type
            else:
                for c in rec.event.comments:
                    if f"{field}:" in c.text:
                        value = c.text.split(':', 1)[-1].strip()
                        break
            grouped[value].append(rec)

        return {
            key: EnhancedCatalog(events=[r.event for r in recs], records=recs)
            for key, recs in grouped.items()
        }

    @classmethod
    def from_dataframe(cls, df, load_waveforms=False):
        """
        Construct a EnhancedCatalog from a DataFrame saved by `to_dataframe()`.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with event metadata.
        load_waveforms : bool
            If True, attempt to load each miniSEED file path.

        Returns
        -------
        catalog : EnhancedCatalog
        """
        records = []
        for _, row in df.iterrows():
            ev = Event()
            if not pd.isna(row['magnitude']):
                ev.magnitudes = [Magnitude(mag=row['magnitude'])]
            if not pd.isna(row['latitude']):
                ev.origins = [Origin(time=UTCDateTime(row['datetime']),
                                     latitude=row['latitude'],
                                     longitude=row['longitude'],
                                     depth=row['depth'])]
            if not pd.isna(row.get('event_type')):
                ev.event_type = row['event_type']
            for key in ['mainclass', 'subclass', 'classification']:
                if key in row and not pd.isna(row[key]):
                    ev.comments.append(Comment(text=f"{key}: {row[key]}"))

            stream = None
            if load_waveforms and isinstance(row.get('filename'), str) and os.path.exists(row['filename']):
                try:
                    stream = read(row['filename'])
                except:
                    pass

            records.append(EventContainer(ev, stream=stream,
                                              classification=row.get('classification'),
                                              miniseedfile=row.get('filename')))

        return cls(events=[r.event for r in records], records=records)

    def plot_magnitudes(self, force_update=False, figsize=(10, 4), fontsize=8):
        """
        Plot local magnitude (ML) and energy magnitude (Me) versus time,
        optionally color-coded by subclass if available.

        Parameters
        ----------
        force_update : bool, optional
            If True, forces a rebuild of the internal event DataFrame.
        figsize : tuple, optional
            Size of the matplotlib figure (default: (10, 4)).
        fontsize : int, optional
            Font size for labels and ticks.
        
        Example
        -------
        >>> cat.plot_magnitudes(force_update=True)
        """
        import matplotlib.pyplot as plt

        if force_update:
            self.update_dataframe(force=True)

        df = self.dataframe
        df = df.dropna(subset=['datetime'])  # Ensure time is available

        fig, ax = plt.subplots(figsize=figsize)

        subclasses = df['subclass'].dropna().unique()
        if len(subclasses) > 0:
            for subclass in subclasses:
                subdf = df[df['subclass'] == subclass]
                ax.plot(subdf['datetime'], subdf['magnitude'], 'o', label=subclass, alpha=0.7)
        else:
            ax.plot(df['datetime'], df['magnitude'], 'o', label='ML', alpha=0.7)

        if 'energy' in df.columns and df['energy'].notna().any():
            df['Me'] = np.log10(df['energy']) / 1.5
            ax.plot(df['datetime'], df['Me'], 'x', label='Me', alpha=0.7, color='gray')

        ax.set_ylabel('Magnitude', fontsize=fontsize)
        ax.set_xlabel('Time', fontsize=fontsize)
        ax.grid(True)
        ax.legend(title='Subclass', fontsize=fontsize - 1)
        plt.xticks(rotation=45, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()
        plt.show()

    def plot_eventrate(self, binsize=pd.Timedelta(days=1), time_limits=None, force_update=False, figsize=(10, 6), fontsize=8):
        """
        Plot histogram and cumulative count of events, and cumulative energy over time.

        Parameters
        ----------
        binsize : pd.Timedelta, optional
            Binning resolution (e.g. daily, hourly). Default is 1 day.
        time_limits : tuple(datetime, datetime), optional
            Optional start and end datetime for filtering the data.
        force_update : bool, optional
            If True, refresh the cached DataFrame before plotting.
        figsize : tuple, optional
            Size of the matplotlib figure (default: (10, 6)).
        fontsize : int, optional
            Font size for labels and ticks.

        Returns
        -------
        dfsum : pd.DataFrame
            Resampled DataFrame with summed and cumulative values.

        Example
        -------
        >>> cat.plot_eventrate(binsize=pd.Timedelta(days=1), force_update=True)
        """
        import matplotlib.pyplot as plt

        if force_update:
            self.update_dataframe(force=True)

        df = self.dataframe.copy()
        df = df.dropna(subset=['datetime'])

        df['counts'] = 1

        if time_limits:
            start, end = time_limits
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]

        if df.empty:
            print("No data to plot.")
            return pd.DataFrame()

        dfsum = df.set_index('datetime').resample(binsize).sum()
        dfsum['cumcounts'] = dfsum['counts'].cumsum()
        if 'energy' in dfsum.columns:
            dfsum['cumenergy'] = dfsum['energy'].cumsum()
        else:
            dfsum['cumenergy'] = np.nan

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=figsize)

        # Plot count and cumulative count
        dfsum.plot(y='counts', ax=axs[0], style='-', color='black', legend=False)
        if 'cumcounts' in dfsum:
            dfsum.plot(y='cumcounts', ax=axs[0].twinx(), style='-', color='green', legend=False)
        axs[0].set_ylabel("Counts", fontsize=fontsize)
        axs[0].tick_params(labelsize=fontsize)

        # Plot energy and cumulative energy
        dfsum.plot(y='energy', ax=axs[1], style='-', color='black', legend=False)
        if 'cumenergy' in dfsum:
            dfsum.plot(y='cumenergy', ax=axs[1].twinx(), style='-', color='green', legend=False)
        axs[1].set_ylabel("Energy", fontsize=fontsize)
        axs[1].tick_params(labelsize=fontsize)
        axs[1].set_xlabel("Time", fontsize=fontsize)

        plt.xticks(rotation=45, fontsize=fontsize)
        plt.tight_layout()
        plt.show()

        return dfsum




def load_catalog(catdir, load_waveforms=False):
    """
    Load an EnhancedCatalog from per-event QuakeML + JSON files.

    Parameters
    ----------
    catdir : str
        Path to directory containing .qml + .json pairs.
    load_waveforms : bool
        Whether to read waveforms from disk.

    Returns
    -------
    EnhancedCatalog
    """
    import glob
    from flovopy.enhanced import EnhancedEvent, EnhancedCatalog, EventContainer

    records = []
    qml_files = sorted(glob.glob(os.path.join(catdir, "**", "*.qml"), recursive=True))

    for qml_file in tqdm(qml_files, desc="Loading events"):
        base_path = os.path.splitext(qml_file)[0]
        try:
            enh = EnhancedEvent.load(base_path)
        except Exception as e:
            print(f"[WARN] Skipping {base_path}: {e}")
            continue

        stream = None
        mseedfile = enh.wav_paths[0] if enh.wav_paths else None
        if load_waveforms and mseedfile and os.path.exists(mseedfile):
            try:
                stream = read(mseedfile)
            except Exception as e:
                print(f"[WARN] Could not load waveform {mseedfile}: {e}")

        rec = EventContainer(
            event=enh.event,
            stream=stream,
            trigger=enh.trigger_window,
            classification=enh.metrics.get("class_label"),
            miniseedfile=mseedfile
        )
        records.append(rec)

    return EnhancedCatalog(events=[r.event for r in records], records=records)



def triggers2catalog(trig_list, threshON, threshOFF, sta_secs, lta_secs, max_secs,
                     stream=None, pretrig=None, posttrig=None):
    """
    Convert a list of trigger dictionaries to a EnhancedCatalog.

    Parameters
    ----------
    trig_list : list of dict
        Triggered events from a coincidence or STA/LTA detector.
    threshON, threshOFF : float
        Trigger thresholds.
    sta_secs, lta_secs : float
        STA/LTA window lengths.
    max_secs : float
        Maximum event duration allowed by trigger logic.
    stream : obspy.Stream, optional
        Continuous stream from which to extract waveforms.
    pretrig, posttrig : float, optional
        Time before/after trigger time to include in waveform.

    Returns
    -------
    EnhancedCatalog
        Catalog of triggered events with metadata and optional waveforms.
    """
    triggerParams = {
        'threshON': threshON,
        'threshOFF': threshOFF,
        'sta': sta_secs,
        'lta': lta_secs,
        'max_secs': max_secs,
        'pretrig': pretrig,
        'posttrig': posttrig
    }

    starttime = stream[0].stats.starttime if stream else None
    endtime = stream[-1].stats.endtime if stream else None

    cat = EnhancedCatalog(triggerParams=triggerParams,
                                starttime=starttime,
                                endtime=endtime)

    for thistrig in trig_list:
        # Create minimal Origin
        origin = Origin(time=thistrig['time'])

        # Collect amplitudes and station magnitudes
        amplitudes = []
        station_magnitudes = []
        sta_mags = []

        if stream:
            # Trim waveform
            this_st = stream.copy().trim(
                starttime=thistrig['time'] - pretrig,
                endtime=thistrig['time'] + thistrig['duration'] + posttrig
            )
            for i, seed_id in enumerate(thistrig['trace_ids']):
                tr = this_st.select(id=seed_id)[0]
                sta_amp = np.nanmax(np.abs(tr.data))
                snr = thistrig['cft_peaks'][i] if 'cft_peaks' in thistrig else None
                wf_id = WaveformStreamID(seed_string=tr.id)

                amplitudes.append(Amplitude(snr=snr,
                                            generic_amplitude=sta_amp,
                                            unit='dimensionless',
                                            waveform_id=wf_id))

                sta_mag = np.log10(sta_amp) - 2.5  # Simple placeholder ML
                station_magnitudes.append(StationMagnitude(mag=sta_mag,
                                                           mag_type='ML',
                                                           waveform_id=wf_id))
                sta_mags.append(sta_mag)

            avg_mag = np.nanmean(sta_mags)
            network_mag = Magnitude(mag=avg_mag, mag_type='ML')
            stream_for_event = this_st
        else:
            stream_for_event = None
            amplitudes = []
            station_magnitudes = []
            network_mag = None

        # Build event
        event = Event(origins=[origin],
                      amplitudes=amplitudes,
                      magnitudes=[network_mag] if network_mag else [],
                      station_magnitudes=station_magnitudes,
                      creation_info=CreationInfo(
                          author='coincidence_trigger',
                          creation_time=UTCDateTime()
                      ),
                      event_type="not reported")

        # Add to catalog
        cat.addEvent(event=event, stream=stream_for_event, trigger=thistrig)

    return cat



class EventFeatureExtractor:
    """
    Extracts features and classification labels for volcano-seismic events.

    Designed to be used with EnhancedCatalog and SAM objects.
    """
    def __init__(self, catalog, rsamObj=None):
        self.catalog = catalog
        self.rsamObj = rsamObj

    def extract_features(self):
        """
        Extracts features like peakf, meanf, duration, and RSAM timing indicators.

        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per event.
        """
        rows = []
        for rec in self.catalog.records:
            feat = {}
            trZ = rec.stream.select(component="Z")[0] if rec.stream else None

            if hasattr(trZ.stats, "metrics"):
                feat["peakf"] = trZ.stats.metrics.get("peakf")
                feat["meanf"] = trZ.stats.metrics.get("meanf")

            if rec.trigger:
                feat["duration"] = rec.trigger.get("duration")

            if self.rsamObj:
                rscore = 0
                for seed_id in self.rsamObj.dataframes:
                    df2 = self.rsamObj.dataframes[seed_id]
                    if df2.empty:
                        continue
                    maxi = np.argmax(df2["median"])
                    r = maxi / len(df2)
                    if 0.3 < r < 0.7:
                        rscore += 3
                    elif r < 0.3:
                        rscore += 1
                    else:
                        rscore -= 2
                feat["rsam_score"] = rscore

            feat["subclass"] = self._get_subclass_label(rec)
            feat["event_id"] = rec.event.resource_id.id if rec.event.resource_id else None
            rows.append(feat)

        return pd.DataFrame(rows)

    def _get_subclass_label(self, rec):
        for c in rec.event.comments:
            if "subclass:" in c.text:
                return c.text.split(":", 1)[-1].strip()
        return None

# --- Classifier scaffold ---
class VolcanoEventClassifier:
    def __init__(self):
        self.classes = ['r', 'e', 'l', 'h', 't']

    def classify(self, feature_dict):
        """
        Basic rule-based classifier.
        Input: dict of features (e.g., 'peakf', 'meanf', 'duration', etc.)
        Output: subclass label and scores
        """
        score = {k: 0.0 for k in self.classes}
        f = feature_dict

        if 'peakf' in f:
            if f['peakf'] < 1.0:
                score['l'] += 2
                score['e'] += 1
            elif f['peakf'] > 5.0:
                score['t'] += 2
            else:
                score['h'] += 1

        if 'meanf' in f:
            if f['meanf'] < 1.0:
                score['l'] += 2
            elif f['meanf'] > 4.0:
                score['t'] += 2

        if 'skewness' in f and f['skewness'] > 2.0:
            score['r'] += 2

        if 'kurtosis' in f and f['kurtosis'] > 10:
            score['t'] += 2
            score['r'] += 1

        if 'duration' in f:
            d = f['duration']
            if d < 5:
                score['t'] += 4; score['h'] += 3; score['r'] -= 5; score['e'] -= 10
            elif d < 10:
                score['t'] += 3; score['h'] += 2; score['e'] -= 2
            elif d < 20:
                score['t'] += 2; score['h'] += 3; score['l'] += 2
            elif d < 30:
                score['h'] += 2; score['l'] += 3; score['r'] += 1
            elif d < 40:
                score['h'] += 1; score['l'] += 2; score['r'] += 2
            else:
                score['r'] += 3; score['e'] += 5

        total = sum([max(0, v) for v in score.values()])
        if total > 0:
            for k in score:
                score[k] = max(0, score[k]) / total

        subclass = max(score, key=score.get)
        return subclass, score

# --- Classification wrapper ---
def classify_and_add_event(stream, catalog, trigger=None, save_stream=False, classifier=None):
    """
    Wrapper to compute metrics, magnitudes, and classify subclass for a single event.

    Parameters
    ----------
    stream : obspy.Stream
        Seismic stream for the event.
    catalog : EnhancedCatalog
        Catalog to which the event should be added.
    trigger : dict, optional
        Dictionary of trigger info, including 'duration'.
    save_stream : bool
        Whether to store stream in the catalog (default False).
    classifier : VolcanoEventClassifier, optional
        Custom classifier to use.

    Returns
    -------
    event : obspy.core.event.Event
        The ObsPy Event object created and added to catalog.
    subclass : str
        Assigned subclass label.
    features : dict
        Dictionary of computed features.
    """
    # --- Compute metrics ---
    for tr in stream:
        ampengfftmag(tr)  # assume this adds metrics into tr.stats.metrics

    # --- Extract features from metrics ---
    features = {}
    for tr in stream:
        m = getattr(tr.stats, 'metrics', {})
        for key in ['peakf', 'meanf', 'skewness', 'kurtosis']:
            if key in m:
                features[key] = m[key]
    if trigger and 'duration' in trigger:
        features['duration'] = trigger['duration']

    # --- Classify ---
    if not classifier:
        classifier = VolcanoEventClassifier()
    subclass, score = classifier.classify(features)

    # --- Build event object ---
    origin = Origin(time=stream[0].stats.starttime)
    magnitudes = []
    for tr in stream:
        m = getattr(tr.stats, 'metrics', {})
        if 'mag' in m:
            magnitudes.append(m['mag'])
    if magnitudes:
        mag = Magnitude(mag=np.nanmean(magnitudes))
        magnitude_list = [mag]
    else:
        magnitude_list = []

    ev = Event(
        origins=[origin],
        magnitudes=magnitude_list,
        creation_info=CreationInfo(author="classify_and_add_event")
    )

    # --- Add to catalog ---
    catalog.addEvent(ev, stream=stream if save_stream else None, trigger=trigger,
                     classification=subclass, event_type='volcanic eruption',
                     mainclass='LV', subclass=subclass)

    return ev, subclass, features


class EnhancedEvent:
    def __init__(self, obspy_event, metrics=None, sfile_path=None, wav_paths=None,
                 aef_path=None, trigger_window=None, average_window=None, stream=None):
        """
        Enhanced representation of a seismic event.

        Parameters
        ----------
        obspy_event : obspy.core.event.Event
            Core ObsPy Event object.
        metrics : dict, optional
            Additional metrics or classifications (e.g., peakf, energy, subclass).
        sfile_path : str, optional
            Path to original SEISAN S-file.
        wav_paths : list of str, optional
            Paths to one or two associated waveform files.
        aef_path : str, optional
            Path to AEF file (if external).
        trigger_window : float, optional
            Trigger window duration in seconds.
        average_window : float, optional
            Averaging window duration in seconds. Used for AEF amplitude
        stream : obspy.Stream or EnhancedStream, optional
            Associated waveform stream.
        """
        self.event = obspy_event
        self.metrics = metrics or {}
        self.sfile_path = sfile_path
        self.wav_paths = wav_paths or []
        self.aef_path = aef_path
        self.trigger_window = trigger_window
        self.average_window = average_window
        self.stream = stream

    '''
    def to_quakeml(self):
        return self.event'
    '''

    def to_json(self):
        return {
            "event_id": str(self.event.resource_id),
            "sfile_path": self.sfile_path,
            "wav_paths": self.wav_paths,
            "aef_path": self.aef_path,
            "trigger_window": self.trigger_window,
            "average_window": self.average_window,
            "metrics": self.metrics,
        }

    def save(self, outdir, base_name):
        """
        Save to QuakeML and JSON.

        Parameters
        ----------
        outdir : str
            Base directory to save outputs.
        base_name : str
            Base filename (no extension).
        """
        qml_path = os.path.join(outdir, base_name + ".qml")
        os.makedirs(os.path.dirname(qml_path), exist_ok=True)
        json_path = os.path.join(outdir, base_name + ".json")
        print(f'Writing obspy event to {qml_path}')
        print(f'Writing json to {json_path}')
        json_dict = self.to_json()
        Catalog(events=[self.event]).write(qml_path, format="QUAKEML")

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2, default=str)
        return qml_path, json_path


    @classmethod
    def load(cls, base_path):
        """
        Load an EnhancedEvent from a base path (no extension).

        Parameters
        ----------
        base_path : str
            Full path to the event file *without* extension.
            Assumes `{base_path}.qml` and `{base_path}.json` exist.

        Returns
        -------
        EnhancedEvent
        """
        qml_file = base_path + ".qml"
        json_file = base_path + ".json"

        if not os.path.exists(qml_file):
            raise FileNotFoundError(f"Missing QuakeML file: {qml_file}")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Missing JSON metadata file: {json_file}")

        event = read_events(qml_file)[0]

        with open(json_file, "r") as f:
            metadata = json.load(f)

        return cls(
            event=event,
            sfile_path=metadata.get("sfile_path"),
            wav_paths=metadata.get("wav_paths", []),
            aef_path=metadata.get("aef_path"),
            trigger_window=metadata.get("trigger_window"),
            average_window=metadata.get("average_window"),
            metrics=metadata.get("metrics", {})
        )


if  __name__ == '__main__':
    cat = EnhancedCatalog()

    # 1. Parse the S-file
    sfile_event_metadata = parse_sfile("path/to/SEISANfile.S")  # your existing function

    # 2. Load waveform
    raw_stream = read("path/to/miniseedfile.mseed")

    # 3. Convert to EnhancedStream
    enh_st = EnhancedStream(raw_stream)

    # 4. Run ampengfftmag
    enh_st.ampengfftmag(inventory=inventory, source_coords=sfile_event_metadata)

    # 5. Build ObsPy Event
    event = create_event_from_sfile_and_stream(sfile_event_metadata, enh_st, inventory, source_coords=sfile_event_metadata)

    # 6. Add to catalog
    cat.addEvent(triggerdict=sfile_event_metadata, stream=enh_st, obspy_event=event)

