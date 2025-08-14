import os
import glob
import math
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy.geodetics.base import gps2dist_azimuth
import fnmatch
from obspy import Trace, Stream, UTCDateTime


class SAM:
    '''
    def __init__(self, dataframes=None, stream=None, sampling_interval=60.0, filter=[0.5, 18.0], bands = {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 'VT':[4.0, 18.0]}, corners=4, clip=None, verbose=False, squash_nans=False):
        """ Create an SAM object 
        
            Optional name-value pairs:
                dataframes: Create an SAM object using these dataframes. Used by downsample() method, for example. Default: None.
                stream: Create an SAM object from this ObsPy.Stream object.
                sampling_interval: Compute SAM data using this sampling interval (in seconds). Default: 60
                filter: list of two floats, representing fmin and fmax. Default: [0.5, 18.0]. Set to None if no filter wanted.
                bands: a dictionary of filter bands and corresponding column names. Default: {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 
                    'VT':[4.0, 18.0]}. For example, the default setting creates 3 additional columns for each DataFrame called 
                    'VLP', 'LP', and 'VT', which contain the mean value for each sampling_interval within the specified filter band
                    (e.g. 0.02-0.2 Hz for VLP). If 'LP' and 'VT' are in this dictionary, an extra column called 'fratio' will also 
                    be computed, which is the log2 of the ratio of the 'VT' column to the 'LP' column, following the definition of
                    frequency ratio by Rodgers et al. (2015).
                corners: number of corners to use in filters
                clip: [min, max] level to use for clipping data. default: None
                verbose: default False
                squash_nans: new behaviour is that if any time window contains a NaN, the whole time window for all metrics will be NaN.
                             if squash_nans=True, then old behaviour restored, where NaNs stripped before computing each metric by using nan-aware averaging functions
        """
        self.dataframes = {} 

        if isinstance(dataframes, dict):
            good_dataframes = {}
            for id, df in dataframes.items():
                if isinstance(df, pd.DataFrame):
                    good_dataframes[id]=df
            if len(good_dataframes)>0:
                self.dataframes = good_dataframes
                #print('dataframes found. ignoring other arguments.')
                return
            else:
                #print('no valid dataframes found')
                pass

        if not isinstance(stream, Stream):
            # empty SAM object
            print('creating blank SAM object')
            return
        
        #good_stream = self.check_units(stream)
        good_stream = stream
        if verbose:
            print('good_stream:\n',good_stream)

        if len(good_stream)>0:
            if good_stream[0].stats.sampling_rate == 1/sampling_interval:
                # no downsampling to do
                for tr in good_stream:
                    df = pd.DataFrame()
                    df['time'] = pd.Series(tr.times('timestamp'))
                    df['mean'] = pd.Series(tr.data) 
                    self.dataframes[tr.id] = df
                return 
            elif good_stream[0].stats.sampling_rate < 1/sampling_interval:
                print('error: cannot compute SAM for a Stream with a tr.stats.delta bigger than requested sampling interval')
                return
            
        for tr in good_stream:
            if tr.stats.npts < tr.stats.sampling_rate * sampling_interval:
                print('Not enough samples for ',tr.id,'. Skipping.')
                continue
            #print(tr.id, 'absolute=',absolute)
            df = pd.DataFrame()
            
            t = tr.times('timestamp') # Unix epoch time
            sampling_rate = tr.stats.sampling_rate
            t = self.reshape_trace_data(t, sampling_rate, sampling_interval)
            df['time'] = pd.Series(np.nanmin(t,axis=1))

            if filter:
                if tr.stats.sampling_rate<filter[1]*2.2:
                    print(f"{tr}: bad sampling rate. Skipping.")
                    continue
                tr2 = tr.copy()
                try:
                    tr2.detrend('demean')
                except Exception as e: # sometimes crashes here because tr2.data is a masked array
                    print(e)
                    if isinstance(tr2.data, np.ma.MaskedArray):
                        try:
                            m = np.ma.getmask(tr2.data)
                            tr2.data = tr2.data.filled(fill_value=0)
                            tr2.detrend('demean')
                            tr2.data = tr2.data.filled(fill_value=0)
                        except Exception as e2:
                            print(e2)
                            continue
                    else: # not a masked array
                        continue
                        
                if clip:
                    tr2.data = np.clip(tr2.data, a_max=clip, a_min=-clip)    
                tr2.filter('bandpass', freqmin=filter[0], freqmax=filter[1], corners=corners)
                y = tr2.data
                #y = self.reshape_trace_data(np.absolute(tr2.data), sampling_rate, sampling_interval)
            else:
                y = tr.data
            y = y.astype('float')
            y[y == 0.0] = np.nan
            y = self.reshape_trace_data(np.absolute(y), sampling_rate, sampling_interval)

            if squash_nans:
                df['min'] = pd.Series(np.nanmin(y,axis=1))   
                df['mean'] = pd.Series(np.nanmean(y,axis=1)) 
                df['max'] = pd.Series(np.nanmax(y,axis=1))
                df['median'] = pd.Series(np.nanmedian(y,axis=1))
                df['rms'] = pd.Series(np.nanstd(y,axis=1))
            else:
                df['min'] = pd.Series(np.min(y,axis=1))   
                df['mean'] = pd.Series(np.mean(y,axis=1)) 
                df['max'] = pd.Series(np.max(y,axis=1))
                df['median'] = pd.Series(np.median(y,axis=1))
                df['rms'] = pd.Series(np.std(y,axis=1))
            if bands:
                for key in bands:
                    tr2 = tr.copy()
                    [flow, fhigh] = bands[key]
                    tr2.filter('bandpass', freqmin=flow, freqmax=fhigh, corners=corners)
                    y = self.reshape_trace_data(np.absolute(tr2.data), sampling_rate, sampling_interval)
                    if squash_nans:
                        df[key] = pd.Series(np.nanmean(y,axis=1))
                    else:
                        df[key] = pd.Series(np.mean(y,axis=1))
                if 'LP' in bands and 'VT' in bands:
                    df['fratio'] = np.log2(df['VT']/df['LP'])

            df.replace(0.0, np.nan, inplace=True)
            self.dataframes[tr.id] = df
    '''


    def __init__(self,
                 dataframes=None,
                 stream=None,
                 sampling_interval: float = 60.0,
                 filter=[0.5, 18.0],
                 bands={'VLP': [0.02, 0.2], 'LP': [0.5, 4.0], 'VT': [4.0, 18.0]},
                 corners: int = 4,
                 clip=None,
                 verbose: bool = False,
                 squash_nans: bool = False):
        """
        Initialize a Seismic Amplitude Measurement (SAM) object.

        The SAM object stores per-trace, windowed metrics (e.g., min, mean, max, median, RMS)
        and optionally filtered-band means, computed from an ObsPy Stream or precomputed
        pandas DataFrames.

        Parameters
        ----------
        dataframes : dict of {str: pandas.DataFrame}, optional
            Precomputed metrics to use directly. Keys are trace IDs
            (e.g., "NET.STA.LOC.CHAN"), and values are DataFrames containing
            at least a 'time' column (epoch seconds) and one or more metric
            columns. If provided, `stream` is ignored.
        stream : obspy.Stream, optional
            ObsPy Stream of waveform data from which to compute metrics.
            If given and `dataframes` is None, metrics will be computed.
        sampling_interval : float, default=60.0
            Output sampling interval in seconds for the computed metrics.
            This is the window length over which min/mean/max/etc. are computed.
        filter : list [fmin, fmax] or None, default=[0.5, 18.0]
            Primary bandpass filter to apply before computing the core metrics.
            If None, no primary bandpass is applied.
        bands : dict {name: [fmin, fmax]}, default={'VLP': [0.02, 0.2], 'LP': [0.5, 4.0], 'VT': [4.0, 18.0]}
            Additional named frequency bands. For each band, the mean absolute
            amplitude per window is computed and stored under the given name.
            Set to None or empty dict to skip band-specific metrics.
        corners : int, default=4
            Number of corners for all bandpass filters (Butterworth design).
        clip : float or None, optional
            If given, clip all trace data to ±`clip` before filtering/metrics.
        verbose : bool, default=False
            If True, print progress messages and diagnostic information.
        squash_nans : bool, optional (deprecated)
            Ignored; retained for backward compatibility. All metrics are always
            computed using NaN-aware reducers (`np.nanmin`, `np.nanmean`, etc.).

        Notes
        -----
        - If `dataframes` is supplied and valid, no computation is performed.
        - If `stream` traces appear to have been merged by `SDSobj.read()` with
          the `flovopy:smart_merge_v1` tag, no further sanitization is done.
          Otherwise, a light `sanitize_stream()` pass removes empties/duplicates.
        - If `stream`'s sampling rate matches exactly 1 / `sampling_interval`,
          the method will skip filtering and directly store time/mean pairs.
        - If the trace sampling rate is insufficient for a given band
          (less than 2.2 × fmax), that band is skipped for that trace.
        - All metrics are computed on the absolute value of the waveform.
          Zero values are replaced with NaN before computation.
        - Computed metric columns:
            'time'   – epoch seconds (window start/left edge)
            'min'    – minimum absolute amplitude in the window
            'mean'   – mean absolute amplitude
            'max'    – maximum absolute amplitude
            'median' – median absolute amplitude
            'rms'    – standard deviation (as RMS) of absolute amplitude
          plus one column per band in `bands`, and optionally 'fratio'
          if both 'LP' and 'VT' bands are present.

        Raises
        ------
        ValueError
            If `stream`'s sampling rate is less than the Nyquist requirement for
            the primary filter or any requested band.
        Exception
            If filtering or detrending fails for a given trace, that trace is skipped.

        Examples
        --------
        >>> from obspy import read
        >>> from flovopy.sam import SAM
        >>> st = read("IU_ANMO.mseed")
        >>> sam = SAM(stream=st, sampling_interval=60.0)
        >>> list(sam.dataframes.keys())
        ['IU.ANMO..BHZ']
        >>> sam.dataframes['IU.ANMO..BHZ'].head()
               time       min      mean       max    median       rms   VLP    LP    VT
        0  1.691040e+09  ...   ...   ...   ...   ...   ...   ...
        """
        self.dataframes = {}

        # 0) Accept prebuilt dataframes (unchanged behavior)
        if isinstance(dataframes, dict):
            good = {k: v for k, v in dataframes.items() if isinstance(v, pd.DataFrame)}
            if good:
                self.dataframes = good
                return

        # 1) No stream → blank object
        if not isinstance(stream, Stream):
            print('creating blank SAM object')
            return

        # Deprecation notice (printed once per call)
        if 'squash_nans' in SAM.__init__.__code__.co_varnames:
            if squash_nans is not None:
                print("NOTE: 'squash_nans' is deprecated; SAM now always uses NaN-aware reducers.")

        # Defensive copy of mutable defaults
        filt = None if filter is None else [float(filter[0]), float(filter[1])]
        band_dict = None if bands is None else {str(k): [float(v[0]), float(v[1])] for k, v in bands.items()}

        st = stream

        # 2) Trust-but-verify: light sanitize if SDS merge tag missing
        has_tag = False
        for tr_chk in st:
            proc = getattr(tr_chk.stats, "processing", []) or []
            if any("flovopy:smart_merge_v1" in p for p in proc):
                has_tag = True
                break
        if not has_tag:
            try:
                from flovopy.core.trace_utils import sanitize_stream
                sanitize_stream(st, drop_empty=True, drop_duplicates=True,
                                unmask_short_zeros=True, min_gap_duration_s=1.0)
            except Exception as e:
                if verbose:
                    print(f"sanitize_stream skipped/failed: {e}")

        if verbose:
            print('good_stream:\n', st)

        if len(st) == 0:
            return

        # 3) Fast path: if fs == 1/Δt, emit time+mean directly
        ref_fs = st[0].stats.sampling_rate
        if np.isclose(ref_fs, 1.0 / sampling_interval):
            for tr in st:
                df = pd.DataFrame({
                    'time': pd.Series(tr.times('timestamp')),
                    'mean': pd.Series(tr.data)
                })
                self.dataframes[tr.id] = df
            return

        # 4) Disallow undersampled streams for requested Δt
        if ref_fs < 1.0 / sampling_interval:
            print('error: cannot compute SAM for a Stream with a tr.stats.delta bigger than requested sampling interval')
            return

        # 5) Per-trace processing
        for tr in st:
            fs = tr.stats.sampling_rate
            if tr.stats.npts < int(fs * sampling_interval):
                if verbose:
                    print(f'Not enough samples for {tr.id}. Skipping.')
                continue

            # Base copy once per trace: handle masking, detrend, optional clip
            tr_base = tr.copy()
            if isinstance(tr_base.data, np.ma.MaskedArray):
                tr_base.data = tr_base.data.filled(fill_value=0)
            try:
                tr_base.detrend('demean')
            except Exception as e:
                if verbose:
                    print(f"{tr.id}: detrend failed ({e})")
                continue
            if clip is not None:
                try:
                    tr_base.data = np.clip(tr_base.data, a_min=-clip, a_max=clip)
                except Exception as e:
                    if verbose:
                        print(f"{tr.id}: clip failed ({e})")
                    continue
            tr_base.data = np.asarray(tr_base.data, dtype=float)

            # Window timestamps (left edge = min timestamp per window)
            t_epoch = tr.times('timestamp')
            T = self.reshape_trace_data(t_epoch, fs, sampling_interval)
            time_col = pd.Series(np.nanmin(T, axis=1))

            # Primary filter (optional) with 2.2× Nyquist guard; cache filtered copies
            filt_cache = {}
            if filt:
                fmin, fmax = float(filt[0]), float(filt[1])
                if fs < 2.2 * fmax:
                    if verbose:
                        print(f"{tr.id}: bad sampling rate for primary band {fmin}-{fmax} Hz. Skipping trace.")
                    continue
                key = (round(fmin, 6), round(fmax, 6), int(corners))
                tr_primary = tr_base.copy()
                try:
                    tr_primary.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=corners)
                except Exception as e:
                    if verbose:
                        print(f"{tr.id}: bandpass {fmin}-{fmax} Hz failed ({e})")
                    continue
                filt_cache[key] = tr_primary
                y = tr_primary.data
            else:
                y = tr_base.data

            # Windowed metrics on |y| with NaN-aware reducers
            y = np.asarray(y, dtype=float)
            y[y == 0.0] = np.nan
            Y = self.reshape_trace_data(np.abs(y), fs, sampling_interval)

            df = pd.DataFrame()
            df['time']   = time_col
            df['min']    = pd.Series(np.nanmin(Y, axis=1))
            df['mean']   = pd.Series(np.nanmean(Y, axis=1))
            df['max']    = pd.Series(np.nanmax(Y, axis=1))
            df['median'] = pd.Series(np.nanmedian(Y, axis=1))
            df['rms']    = pd.Series(np.nanstd(Y, axis=1))

            # Extra bands: mean(|bandpass|) per window
            if band_dict:
                for key_name, (flow, fhigh) in band_dict.items():
                    if fs < 2.2 * fhigh:
                        if verbose:
                            print(f"{tr.id}: bad sampling rate for band {key_name} {flow}-{fhigh} Hz. Skipping band.")
                        continue
                    bkey = (round(float(flow), 6), round(float(fhigh), 6), int(corners))
                    if bkey in filt_cache:
                        tr_band = filt_cache[bkey]
                    else:
                        tr_band = tr_base.copy()
                        try:
                            tr_band.filter('bandpass', freqmin=float(flow), freqmax=float(fhigh), corners=corners)
                        except Exception as e:
                            if verbose:
                                print(f"{tr.id}: bandpass {flow}-{fhigh} Hz failed ({e})")
                            continue
                        filt_cache[bkey] = tr_band
                    Yb = self.reshape_trace_data(np.abs(np.asarray(tr_band.data, dtype=float)), fs, sampling_interval)
                    df[key_name] = pd.Series(np.nanmean(Yb, axis=1))

                # frequency ratio when LP & VT available
                if 'LP' in band_dict and 'VT' in band_dict and 'LP' in df and 'VT' in df:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        df['fratio'] = np.log2(df['VT'] / df['LP'])

            # Normalize zeros to NaN, store
            df.replace(0.0, np.nan, inplace=True)
            self.dataframes[tr.id] = df
    
    def copy(self):
        ''' make a full copy of an SAM object and return it '''
        selfcopy = self.__class__(stream=Stream())
        selfcopy.dataframes = self.dataframes.copy()
        return selfcopy
    
    def despike_old(self, metrics=['mean'], thresh=1.5, reps=1, verbose=False):
        if not isinstance(metrics, list):
            metrics = [metrics]
        if metrics=='all':
            metrics = self.get_metrics()
        for metric in metrics:
            st = self.to_stream(metric=metric)
            for tr in st:
                x = tr.data
                count1 = 0
                count2 = 0
                for i in range(len(x)-3): # remove spikes on length 2
                    if x[i+1]>x[i]*thresh and x[i+2]>x[i]*thresh and x[i+1]>x[i+3]*thresh and x[i+2]>x[i+3]*thresh:
                        count2 += 1
                        x[i+1] = (x[i] + x[i+3])/2
                        x[i+2] = x[i+1]
                for i in range(len(x)-2): # remove spikes of length 1
                    if x[i+1]>x[i]*thresh and x[i+1]>x[i+2]*thresh:
                        x[i+1] = (x[i] + x[i+2])/2  
                        count1 += 1  
                if verbose:
                    print(f'{tr.id}: removed {count2} length-2 spikes and {count1} length-1 spikes')           
                self.dataframes[tr.id][metric]=x
        if reps>1:
            self.despike(metrics=metrics, thresh=thresh, reps=reps-1)     

    def despike(self, metrics=['mean'], z=6.0, window=9, inplace=True, verbose=False):
        """
        MAD-based despike on SAM time-series metrics.
        - metrics: list or 'all' to target all numeric metric columns present in dataframes
        - z: modified z-score threshold (typical: 5–8)
        - window: odd integer rolling window length (samples)
        - inplace: modify this SAM or return a new one
        """
        if not isinstance(metrics, list):
            metrics = [metrics]

        out = {} if not inplace else self.dataframes
        for tid, df in self.dataframes.items():
            d = df.copy() if not inplace else df

            # figure out which columns to touch
            if metrics == ['all']:
                cols = [c for c in d.columns if c not in ('time', 'date')]
            else:
                cols = [c for c in metrics if c in d.columns]

            if not cols:
                if verbose:
                    print(f"{tid}: no matching metric columns")
                if not inplace:
                    out[tid] = d
                continue

            # ensure datetime for resampling use elsewhere
            if 'date' not in d.columns:
                d['date'] = pd.to_datetime(d['time'], unit='s')

            # rolling center median/MAD
            for col in cols:
                x = d[col].astype(float).values
                if np.all(np.isnan(x)) or len(x) < max(5, window):
                    continue

                s = pd.Series(x)
                med = s.rolling(window=window, center=True, min_periods=3).median()
                # MAD with rolling window
                abs_dev = (s - med).abs()
                mad = abs_dev.rolling(window=window, center=True, min_periods=3).median()

                # modified z-score (0.6745 * |x - med| / MAD)
                with np.errstate(divide='ignore', invalid='ignore'):
                    mz = 0.6745 * abs_dev / mad
                spikes = mz > z

                # replace spikes with rolling median
                x_new = x.copy()
                x_new[spikes.values] = med.values[spikes.values]
                d[col] = x_new

                if verbose:
                    n_spikes = int(np.nansum(spikes.values))
                    print(f"{tid}.{col}: replaced {n_spikes} spikes (z>{z})")

            if not inplace:
                out[tid] = d

        if inplace:
            return self
        else:
            return self.__class__(dataframes=out)   

    def downsample_old(self, new_sampling_interval=3600):
        ''' downsample an SAM object to a larger sampling interval(e.g. from 1 minute to 1 hour). Returns a new SAM object.
         
            Optional name-value pair:
                new_sampling_interval: the new sampling interval (in seconds) to downsample to. Default: 3600
        '''

        dataframes = {}
        for id in self.dataframes:
            df = self.dataframes[id]
            df['date'] = pd.to_datetime(df['time'], unit='s')
            old_sampling_interval = self.get_sampling_interval(df)
            if new_sampling_interval > old_sampling_interval:
                freq = '%.0fmin' % (new_sampling_interval/60)
                try:
                    new_df = df.groupby(pd.Grouper(key='date', freq=freq)).mean()
                    new_df.reset_index(drop=True)
                except:
                    print(f'Could not downsample dataframe for {id}')
                else:
                    dataframes[id] = new_df
            else:
                print('Cannot downsample to a smaller sampling interval')
        return self.__class__(dataframes=dataframes) 
    
    def downsample(self, new_sampling_interval=3600, inplace=False):
        """
        Downsample SAM metrics to a coarser interval (seconds).
        Returns a new SAM unless inplace=True.
        """
        if new_sampling_interval <= 0:
            raise ValueError("new_sampling_interval must be positive")

        result = {} if not inplace else self.dataframes

        for tid, df in self.dataframes.items():
            d = df.copy() if not inplace else df

            if 'date' not in d.columns:
                d['date'] = pd.to_datetime(d['time'], unit='s')

            # current cadence (best effort)
            if len(d) < 2:
                if not inplace:
                    result[tid] = d
                continue
            # resample
            freq = f"{int(new_sampling_interval)}S"  # seconds
            # numeric aggregation only; leave non-numeric alone
            numeric = d.select_dtypes(include='number').copy()
            # Ensure 'time' isn't used as numeric input
            numeric = numeric.drop(columns=[c for c in ['time'] if c in numeric.columns], errors='ignore')

            agg = numeric.set_index(d['date']).resample(freq, origin='epoch', label='left').mean()

            # rebuild 'time' and 'date'
            agg['date'] = agg.index
            agg['time'] = (agg['date'].view('int64') // 10**9).astype(float)

            # Put 'time' first, then others (stable order)
            ordered = ['time'] + [c for c in d.columns if c not in ('time', 'date')] + ['date']
            # Some columns might be absent (e.g., non-numeric); filter existing
            ordered = [c for c in ordered if c in agg.columns]

            d2 = agg[ordered].reset_index(drop=True)

            if inplace:
                self.dataframes[tid] = d2
            else:
                result[tid] = d2

        if inplace:
            return self
        else:
            return self.__class__(dataframes=result)
        
    def drop(self, id):
        if id in self.__get_trace_ids():
            del self.dataframes[id]

    def ffm(self):
        # To do. 1/RSAM plots and work out where intersect y-axis.
        pass

    def get_distance_km(self, inventory, source):
        distance_km = {}
        coordinates = {}
        print(f'SCAFFOLD: {inventory}')
        for seed_id in self.dataframes:
            coordinates[seed_id] = inventory.get_coordinates(seed_id)
            if seed_id[0:2]=='MV':
                if coordinates[seed_id]['longitude'] > 0:
                    coordinates[seed_id]['longitude']  *= -1
            #print(coordinates[seed_id])
            #print(source)
            distance_m, az_source2station, az_station2source = gps2dist_azimuth(source['lat'], source['lon'], coordinates[seed_id]['latitude'], coordinates[seed_id]['longitude'])
            #print(distance_m/1000)
            #distance_km[seed_id] = degrees2kilometers(distance_deg)
            distance_km[seed_id] = distance_m/1000
        return distance_km, coordinates

    def __len__(self):
        return len(self.dataframes)
    '''
    def plot(self, metrics=['mean'], kind='stream', logy=False, equal_scale=False, outfile=None, ylims=None):
        """ plot a SAM object 

            Optional name-value pairs:
                metrics: The columns of each SAM DataFrame to plot. Can be one (scalar), or many (a list)
                         If metrics='bands', this is shorthand for metrics=['VLP', 'LP', 'VT', 'specratio']
                         Default: metrics='mean'
                kind:    The kind of plot to make. kind='stream' (default) will convert each of the request 
                         DataFrame columns into an ObsPy.Stream object, and then use the ObsPy.Stream.plot() method.
                         kind='line' will render plots directly using matplotlib.pyplot, with all metrics requested 
                         on a single plot.
                logy:    In combination with kind='line', will make the y-axis logarithmic. No effect if kind='stream'.
                equal_scale: If True, y-axes for each plot will have same limits. Default: False.
        
        """
        self.__remove_empty()
        if isinstance(metrics, str):
            metrics = [metrics]
        if kind == 'stream':
            if metrics == ['bands']:
                metrics = ['VLP', 'LP', 'VT', 'fratio']
            for m in metrics:
                print('METRIC: ',m)
                st = self.to_stream(metric=m, ylims=ylims)
                if outfile:
                    if not m in outfile:
                        this_outfile = outfile.replace('.png', f"_{m}.png")
                        st.plot(equal_scale=equal_scale, outfile=this_outfile);
                    else:
                        st.plot(equal_scale=equal_scale, outfile=outfile);
                else:
                    st.plot(equal_scale=equal_scale);
            return
        for key in self.dataframes:
            df = self.dataframes[key]
            this_df = df.copy()
            this_df['time'] = pd.to_datetime(df['time'], unit='s')
            if isinstance(ylims, list) or isinstance(ylims, tuple):
                for m in metrics:
                    this_df[m] = this_df[m].clip(lower=ylims[0], upper=ylims[1])
            if metrics == ['bands']:
                # plot f-bands only
                if not 'VLP' in this_df.columns:
                    print('no frequency bands data for ',key)
                    continue
                ph2 = this_df.plot(x='time', y=['VLP', 'LP', 'VT'], kind=kind, title=f"{key}, f-bands", logy=logy, rot=45)
                if outfile:
                    this_outfile = outfile.replace('.png', "_bands.png")
                    plt.savefig(this_outfile)
                else:
                    plt.show()
            else:
                for m in metrics:
                    got_all_metrics = True
                    if not m in this_df.columns:
                        print(f'no {m} column for {key}')
                        got_all_metrics = False
                if not got_all_metrics:
                    continue
                if kind == 'line':
                    ph = this_df.plot(x='time', y=metrics, kind=kind, title=key, logy=logy, rot=45)
                elif kind  == 'scatter':
                    fh, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
                    for i, m in enumerate(metrics):
                        this_df.plot(x='time', y=m, kind=kind, ax=ax[i], title=key, logy=logy, rot=45)
                if outfile:
                    this_outfile = outfile.replace('.png', "_metrics.png")
                    plt.savefig(this_outfile)
                else:
                    plt.show()
            plt.close('all')
    
    @classmethod
    def read(classref, startt, endt, SAM_DIR, trace_ids=None, network='*', sampling_interval=60, ext='pickle', verbose=False):
        """ read one or many SAM files from folder specified by SAM_DIR for date/time range specified by startt, endt
            return corresponding SAM object

            startt and endt must be ObsPy.UTCDateTime data types

            Optional name-value pairs:
                trace_ids (list): only load SAM files corresponding to these trace IDs.
                network (str): only load SAM files matching this network code. Ignored if trace_ids given.
                sampling_interval (int): seconds of raw seismic data corresponding to each SAM sample. Default: 60
                ext (str): should be 'csv' or 'pickle' (default). Indicates what type of file format to open.

        """
        #self = classref() # blank SAM object
        dataframes = {}

        if not trace_ids: # make a list of possible trace_ids, regardless of year
            trace_ids = []
            for year in range(startt.year, endt.year+1):
                samfilepattern = classref.get_filename(SAM_DIR, network, year, sampling_interval, ext)
                #print(samfilepattern)
                samfiles = glob.glob(samfilepattern)
                if len(samfiles)==0:
                    if verbose:
                        print(f'No files found matching {samfilepattern}')
                #samfiles = glob.glob(os.path.join(SAM_DIR,'SAM_*_[0-9][0-9][0-9][0-9]_%ds.%s' % (sampling_interval, ext )))
                for samfile in samfiles:
                    parts = samfile.split('_')
                    trace_ids.append(parts[-3])
            trace_ids = list(set(trace_ids)) # makes unique
            #print(trace_ids)
        
        for id in trace_ids:
            df_list = []
            for yyyy in range(startt.year, endt.year+1):
                samfile = classref.get_filename(SAM_DIR, id, yyyy, sampling_interval, ext)
                #samfile = os.path.join(SAM_DIR,'SAM_%s_%4d_%ds.%s' % (id, yyyy, sampling_interval, ext))
                if os.path.isfile(samfile):
                    if verbose:
                        print('Reading ',samfile)
                    if ext=='csv':
                        df = pd.read_csv(samfile, index_col=False)
                    elif ext=='pickle':
                        df = pd.read_pickle(samfile)
                    if df.empty:
                        continue
                    if 'std' in df.columns:
                        df.rename(columns={'std':'rms'}, inplace=True)
                    df['pddatetime'] = pd.to_datetime(df['time'], unit='s')
                    # construct Boolean mask
                    mask = df['pddatetime'].between(startt.isoformat(), endt.isoformat())
                    # apply Boolean mask
                    subset_df = df[mask]
                    subset_df = subset_df.drop(columns=['pddatetime'])
                    df_list.append(subset_df)
                else:
                    print(f"Cannot find {samfile}")
            if len(df_list)==1:
                dataframes[id] = df_list[0]
            elif len(df_list)>1:
                dataframes[id] = pd.concat(df_list)
                
        samObj = classref(dataframes=dataframes) # create SAM object         
        return samObj
    '''

    def plot(self, metrics=['mean'], kind='stream', logy=False, equal_scale=False, outfile=None, ylims=None):
        """Plot a SAM object.

        Parameters
        ----------
        metrics : str or list, default 'mean'
            Columns to plot from each SAM DataFrame. Special value 'bands'
            auto-detects a band set among:
            - PRI, SEC, HI   (storm seismic)
            - TC, MB, TH     (storm infrasound)
            - VLP, LP, VT    (classic volcano)
            If LP and VT are present anywhere, 'fratio' is appended too.
        kind : {'stream','line','scatter'}, default 'stream'
            'stream' → convert each metric to an ObsPy.Stream and call .plot()
            'line'   → pandas line plot (metrics on one axes)
            'scatter'→ two stacked scatter subplots (for 2 metrics)
        logy : bool, default False
            Log-scale y-axis (only for kind != 'stream').
        equal_scale : bool, default False
            If True, same y-limits for all Stream plots (kind='stream').
        outfile : str or None
            If provided, figures are written to disk. For kind='stream',
            the metric name is appended before '.png' unless already present.
        ylims : (low, high) or None
            Optional clipping bounds applied before plotting (non-stream).
        """
        import matplotlib.pyplot as plt
        self.__remove_empty()

        # Normalize metrics input
        if isinstance(metrics, str):
            metrics = [metrics]

        # -------- Auto-detect band set if requested --------
        if metrics == ['bands']:
            # Union of columns across all traces
            all_cols = set()
            for df in self.dataframes.values():
                all_cols.update(df.columns)

            # Preference order of band triads
            candidates = [
                ['PRI', 'SEC', 'HI'],   # storm seismic
                ['TC', 'MB', 'TH'],     # storm infrasound
                ['VLP', 'LP', 'VT'],    # classic volcano
            ]
            chosen = None
            for triad in candidates:
                if set(triad).issubset(all_cols):
                    chosen = triad
                    break

            if not chosen:
                print('no frequency bands data present')
                return

            # Add fratio if available (LP & VT)
            if {'LP', 'VT'}.issubset(all_cols):
                metrics = chosen + ['fratio']
            else:
                metrics = chosen

        # -------- kind='stream' path --------
        if kind == 'stream':
            for m in metrics:
                print('METRIC:', m)
                st = self.to_stream(metric=m, ylims=ylims)
                if outfile:
                    if m not in outfile:
                        this_outfile = outfile.replace('.png', f"_{m}.png")
                        st.plot(equal_scale=equal_scale, outfile=this_outfile)
                    else:
                        st.plot(equal_scale=equal_scale, outfile=outfile)
                else:
                    st.plot(equal_scale=equal_scale)
            return

        # -------- kind='line' / 'scatter' path --------
        for key, df in self.dataframes.items():
            this_df = df.copy()
            this_df['time'] = pd.to_datetime(df['time'], unit='s')

            # Optional clipping
            if isinstance(ylims, (list, tuple)) and len(ylims) == 2:
                lo, hi = ylims
                for m in metrics:
                    if m in this_df.columns:
                        this_df[m] = this_df[m].clip(lower=lo, upper=hi)

            # Ensure requested metrics exist for this trace
            missing = [m for m in metrics if m not in this_df.columns]
            if missing:
                print(f"{key}: missing columns: {', '.join(missing)}")
                continue

            if kind == 'line':
                ax = this_df.plot(x='time', y=metrics, kind='line', title=key, logy=logy, rot=45)
            elif kind == 'scatter':
                if len(metrics) < 2:
                    print(f"{key}: scatter requires at least 2 metrics")
                    continue
                fh, ax = plt.subplots(nrows=min(2, len(metrics)), ncols=1, sharex=True, sharey=False)
                if not isinstance(ax, (list, tuple, np.ndarray)):
                    ax = [ax]
                for i, m in enumerate(metrics[:len(ax)]):
                    this_df.plot(x='time', y=m, kind='scatter', ax=ax[i], title=key, logy=logy, rot=45)
            else:
                print(f"Unknown kind='{kind}'")
                continue

            if outfile:
                suffix = "_bands.png" if set(['PRI','SEC','HI']).issubset(metrics) or set(['VLP','LP','VT']).issubset(metrics) or set(['TC','MB','TH']).issubset(metrics) else "_metrics.png"
                this_outfile = outfile.replace('.png', suffix)
                plt.savefig(this_outfile)
            else:
                plt.show()

            plt.close('all')

    @classmethod
    def read(classref, startt, endt, SAM_DIR, trace_ids=None, network='*',
            sampling_interval=60, ext='pickle', verbose=False):
        """
        Read one or many SAM files from SAM_DIR over [startt, endt) and return a SAM object.

        Parameters
        ----------
        startt, endt : obspy.UTCDateTime
            Start and end times (UTC). End is treated as **exclusive**.
        SAM_DIR : str
            Base output directory that contains RSAM/ or VSAM subfolders (class handles this).
        trace_ids : list[str], optional
            Explicit list of trace IDs to load (e.g., "IU.DWPF.10.BHZ"). If omitted, the
            method discovers trace IDs from files matching the network pattern.
        network : str, default '*'
            Network filter when discovering trace IDs (ignored if trace_ids is provided).
        sampling_interval : int, default 60
            SAM sampling interval (seconds).
        ext : {'pickle','csv'}, default 'pickle'
            File format to read.
        verbose : bool, default False
            Print diagnostics.

        Returns
        -------
        classref
            A SAM/RSAM/VSAM instance with `dataframes` populated for the selected IDs/time span.
        """
        dataframes = {}

        # Convert bounds to pandas Timestamps (naive UTC) for robust masking
        start_ts = pd.to_datetime(startt.datetime)
        end_ts   = pd.to_datetime(endt.datetime)

        # -------- Discover trace_ids if not provided --------
        # -------- Discover trace_ids if not provided --------
        if not trace_ids:
            found_ids = set()
            subdir = classref.__name__.upper()  # 'RSAM' or 'VSAM'
            for year in range(startt.year, endt.year + 1):
                # Files live at: <SAM_DIR>/<RSAM|VSAM>/<NET>/<RSAM|VSAM>_<NET.STA.LOC.CHA>_<YEAR>_<Δs>.<ext>
                pattern = os.path.join(
                    SAM_DIR, subdir, network,
                    f"{subdir}_*_{year}_{int(sampling_interval)}s.{ext}"
                )
                samfiles = glob.glob(pattern)
                if not samfiles and verbose:
                    print(f'No files found matching {pattern}')
                for path in samfiles:
                    base = os.path.basename(path)
                    parts = base.split('_')
                    # e.g., VSAM_IU.DWPF.10.BHZ_2011_60s.pickle  -> parts[-3] == 'IU.DWPF.10.BHZ'
                    if len(parts) >= 3:
                        found_ids.add(parts[-3])
            trace_ids = sorted(found_ids)

            if verbose and not trace_ids:
                print("No trace IDs discovered for the given pattern/time range.")

        # -------- Load frames per id across years, then slice by time --------
        for tid in trace_ids:
            yearly = []
            for yyyy in range(startt.year, endt.year + 1):
                samfile = classref.get_filename(SAM_DIR, tid, yyyy, sampling_interval, ext)
                if os.path.isfile(samfile):
                    if verbose:
                        print('Reading', samfile)
                    if ext == 'csv':
                        df = pd.read_csv(samfile, index_col=False)
                    elif ext == 'pickle':
                        df = pd.read_pickle(samfile)
                    else:
                        raise ValueError(f"Unsupported ext '{ext}'. Use 'pickle' or 'csv'.")
                    if df.empty:
                        continue

                    # Standardize column name
                    if 'std' in df.columns:
                        df.rename(columns={'std': 'rms'}, inplace=True)

                    # Time mask: [start, end)
                    df['pddatetime'] = pd.to_datetime(df['time'], unit='s', utc=False)
                    mask = df['pddatetime'].between(start_ts, end_ts, inclusive='left')
                    subset_df = df.loc[mask].drop(columns=['pddatetime'])
                    if not subset_df.empty:
                        yearly.append(subset_df)
                else:
                    if verbose:
                        print(f"Cannot find {samfile}")

            if len(yearly) == 1:
                dataframes[tid] = yearly[0]
            elif len(yearly) > 1:
                dataframes[tid] = pd.concat(yearly, ignore_index=True)

        # Return a SAM/RSAM/VSAM instance populated with the loaded frames
        return classref(dataframes=dataframes)

    @classmethod
    def missing_days(
        classref,
        startt,
        endt,
        SAM_DIR,
        trace_ids=None,
        network="*",
        sampling_interval=60,
        ext="pickle",
        require_full_day=False,
        tol=0.05,
        verbose=False,
    ):
        """
        Return per-trace lists of *missing* UTC days in [startt, endt), based on FINAL
        per-year RSAM/VSAM files. This **wraps `read()`** to avoid code duplication.

        Coverage rule:
          - require_full_day=False → a day is covered if it has ≥ 1 sample.
          - require_full_day=True  → a day is covered if it has ≥ (1 - tol) * floor(86400/Δ) samples.

        Parameters
        ----------
        startt, endt : obspy.UTCDateTime
            Time window; end is **exclusive**. Days are midnight UTC boundaries.
        SAM_DIR : str
            Base output dir that contains RSAM/ or VSAM (chosen by subclass).
        trace_ids : list[str], optional
            Explicit SEED IDs to consider. If omitted, IDs are discovered by `read()`.
            If provided and some IDs are missing from the loaded object, those IDs
            are treated as **completely missing** in the interval.
        network : str, default '*'
            Network filter when discovering IDs (ignored if `trace_ids` provided).
        sampling_interval : int, default 60
            SAM Δ in seconds (used for expected-per-day).
        ext : {'pickle','csv'}, default 'pickle'
            File format of final files.
        require_full_day : bool, default False
            Enforce near-full-day coverage (see `tol`).
        tol : float, default 0.05
            Shortfall tolerance when `require_full_day=True`.
        verbose : bool, default False
            Print diagnostics.

        Returns
        -------
        dict[str, list[UTCDateTime]]
            Mapping: trace_id → list of missing-day UTC midnights in [startt, endt).
        """
        # Build the day list once
        start_ts = pd.to_datetime(startt.datetime, utc=True)
        end_ts   = pd.to_datetime(endt.datetime,   utc=True)
        all_days = pd.date_range(
            start=start_ts.floor("D"),
            end=end_ts.floor("D"),
            freq="D",
            inclusive="left",
        )
        day_keys = [d.strftime("%Y-%m-%d") for d in all_days]

        if not day_keys:
            return {}

        # Expected samples per full day for this Δ
        exp_per_day = int(math.floor(86400.0 / float(sampling_interval)))
        threshold   = (1.0 - float(tol)) * exp_per_day if require_full_day else 1

        # Reuse the existing loader — no duplication
        loaded = classref.read(
            startt=startt, endt=endt, SAM_DIR=SAM_DIR,
            trace_ids=trace_ids, network=network,
            sampling_interval=int(sampling_interval),
            ext=ext, verbose=verbose,
        )

        # Which IDs are we evaluating?
        loaded_ids = set(loaded.dataframes.keys())
        requested_ids = set(trace_ids) if trace_ids else loaded_ids
        all_ids = sorted(requested_ids.union(loaded_ids))

        missing_by_id = {}

        for tid in all_ids:
            # If an explicitly requested ID didn't load at all, everything is missing
            if tid not in loaded_ids:
                missing_by_id[tid] = [
                    UTCDateTime(pd.Timestamp(k).to_pydatetime().replace(tzinfo=None))
                    for k in day_keys
                ]
                if verbose:
                    print(f"[missing_days] {tid}: no data loaded → {len(day_keys)} missing days")
                continue

            df = loaded.dataframes[tid]
            if df.empty:
                # No samples in the window ⇒ all days are missing
                missing_by_id[tid] = [
                    UTCDateTime(pd.Timestamp(k).to_pydatetime().replace(tzinfo=None))
                    for k in day_keys
                ]
                if verbose:
                    print(f"[missing_days] {tid}: empty frame → {len(day_keys)} missing days")
                continue

            # Count samples per UTC day
            ts = pd.to_datetime(df["time"], unit="s", utc=True)
            # ts is a Series of pandas Timestamps
            # Use the datetime accessor; fall back to .dt.normalize() for older pandas.
            try:
                days = ts.dt.floor("D")
            except AttributeError:  # very old pandas
                days = ts.dt.normalize()
            per_day_counts = days.value_counts()

            # Covered if day in counts and meets threshold
            covered = {
                pd.Timestamp(d).strftime("%Y-%m-%d")
                for d, n in per_day_counts.items() if n >= threshold
            }
            remaining = [k for k in day_keys if k not in covered]

            missing_by_id[tid] = [
                UTCDateTime(pd.Timestamp(k).to_pydatetime().replace(tzinfo=None))
                for k in remaining
            ]

            if verbose:
                print(f"[missing_days] {tid}: covered={len(covered)}  missing={len(remaining)}")

        return missing_by_id

    def select(self, network=None, station=None, location=None, channel=None,
               sampling_interval=None, npts=None, component=None, id=None,
               inventory=None):
        """
        Return new SAM object only with DataFrames that match the given
        criteria (e.g. all DataFrames with ``channel="BHZ"``).

        Alternatively, DataFrames can be selected based on the content of an
        :class:`~obspy.core.inventory.inventory.Inventory` object: DataFrame will
        be selected if the inventory contains a matching channel active at the
        DataFrame start time.

        based on obspy.Stream.select()

        .. rubric:: Examples

        >>> samObj2 = samObj.select(station="R*")
        >>> samObj2 = samObj.select(id="BW.RJOB..EHZ")
        >>> samObj2 = samObj.select(component="Z")
        >>> samObj2 = samObj.select(network="CZ")
        >>> samObj2 = samObj.select(inventory=inv)
    
        All keyword arguments except for ``component`` are tested directly
        against the respective entry in the :class:`~obspy.core.trace.Stats`
        dictionary.

        If a string for ``component`` is given (should be a single letter) it
        is tested against the last letter of the ``Trace.stats.channel`` entry.

        Alternatively, ``channel`` may have the last one or two letters
        wildcarded (e.g. ``channel="EH*"``) to select all components with a
        common band/instrument code.

        All other selection criteria that accept strings (network, station,
        location) may also contain Unix style wildcards (``*``, ``?``, ...).
        """
        if inventory is None:
            dataframes = self.dataframes
        else:
            trace_ids = []
            start_dates = []
            end_dates = []
            for net in inventory.networks:
                for sta in net.stations:
                    for chan in sta.channels:
                        id = '.'.join((net.code, sta.code,
                                       chan.location_code, chan.code))
                        trace_ids.append(id)
                        start_dates.append(chan.start_date)
                        end_dates.append(chan.end_date)
            dataframes = {}
            for thisid, thisdf in self.dataframes.items():
                idx = 0
                while True:
                    try:
                        idx = trace_ids.index(thisid, idx)
                        start_date = start_dates[idx]
                        end_date = end_dates[idx]
                        idx += 1
                        if start_date is not None and\
                                self.__get_starttime(thisdf) < start_date:
                            continue
                        if end_date is not None and\
                                self.__get_endtime(thisdf) > end_date:
                            continue
                        dataframes[thisid]=thisdf
                    except ValueError:
                        break
        dataframes_after_inventory_filter = dataframes

        # make given component letter uppercase (if e.g. "z" is given)
        if component is not None and channel is not None:
            component = component.upper()
            channel = channel.upper()
            if (channel[-1:] not in "?*" and component not in "?*" and
                    component != channel[-1:]):
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise ValueError(msg)

        # For st.select(id=) without wildcards, use a quicker comparison mode:
        quick_check = False
        quick_check_possible = (id is not None
                                #and sampling_rate is None and npts is None
                                and network is None and station is None
                                and location is None and channel is None
                                and component is None)
        if quick_check_possible:
            no_wildcards = not any(['?' in id or '*' in id or '[' in id])
            if no_wildcards:
                quick_check = True
                [net, sta, loc, chan] = id.upper().split('.')

        dataframes = {}
        for thisid, thisdf in dataframes_after_inventory_filter.items():
            [thisnet, thissta, thisloc, thischan] = thisid.upper().split('.')
            if quick_check:
                if (thisnet.upper() == net
                        and thissta.upper() == sta
                        and thisloc.upper() == loc
                        and thischan.upper() == chan):
                    dataframes[thisid]=thisdf
                continue
            # skip trace if any given criterion is not matched
            if id and not fnmatch.fnmatch(thisid.upper(), id.upper()):
                continue
            if network is not None:
                if not fnmatch.fnmatch(thisnet.upper(),
                                       network.upper()):
                    continue
            if station is not None:
                if not fnmatch.fnmatch(thissta.upper(),
                                       station.upper()):
                    continue
            if location is not None:
                if not fnmatch.fnmatch(thisloc.upper(),
                                       location.upper()):
                    continue
            if channel is not None:
                if not fnmatch.fnmatch(thischan.upper(),
                                       channel.upper()):
                    continue
            if sampling_interval is not None:
                if float(sampling_interval) != self.get_sampling_interval(thisdf):
                    continue
            if npts is not None and int(npts) != self.__get_npts(thisdf):
                continue
            if component is not None:
                if not fnmatch.fnmatch(thischan[-1].upper(),
                                       component.upper()):
                    continue
            dataframes[thisid]=thisdf
        return self.__class__(dataframes=dataframes)       
     
    def to_stream(self, metric='mean', ylims=None):
        ''' Convert one column (specified by metric) of each DataFrame in an SAM object to an obspy.Trace, 
            return an ObsPy.Stream that is the combination of all of these Trace objects'''
        st = Stream()
        for key in self.dataframes:
            #print(key)
            df = self.dataframes[key]
            if metric in df.columns:
                dataSeries = df[metric]
                if isinstance(ylims, list) or isinstance(ylims, tuple):
                    dataSeries = dataSeries.clip(lower=ylims[0], upper=ylims[1])
                tr = Trace(data=np.array(dataSeries))
                #timeSeries = pd.to_datetime(df['time'], unit='s')
                tr.stats.delta = self.get_sampling_interval(df)
                tr.stats.starttime = UTCDateTime(df.iloc[0]['time'])
                tr.id = key
                #print(min(tr.data))
                if tr.data.size - np.count_nonzero(np.isnan(tr.data)):
                    st.append(tr)
                
        return st
    
    def trim(self, starttime=None, endtime=None, pad=False, keep_empty=False, fill_value=None):
        ''' trim SAM object based on starttime and endtime. Both must be of type obspy.UTCDateTime 

            based on obspy.Stream.trim()

            keep_empty=True will retain dataframes that are either blank, or full of NaN's or 0's.
                       Default: False
            
            Note:
            - pad option and fill_value option not yet implemented
        '''
        if pad:
            print('pad option not yet supported')
            return
        if fill_value:
            print('fill_value option not yet supported')
            return
        if not starttime or not endtime:
            print('starttime and endtime required as ObsPy.UTCDateTime')
            return
        for id in self.dataframes:
            df = self.dataframes[id]
            mask = (df['time']  >= starttime.timestamp ) & (df['time'] <= endtime.timestamp )
            self.dataframes[id] = df.loc[mask]
        if not keep_empty:
            self.__remove_empty()
        
    def write(self, SAM_DIR, ext='pickle', overwrite=False, verbose=False):
        ''' Write SAM object to CSV or Pickle files (one per net.sta.loc.chan, per year) into folder specified by SAM_DIR

            Optional name-value pairs:
                ext: Should be 'csv' or 'pickle' (default). Specifies what file format to save each DataFrame.   
        '''
        print('write')
        for id in self.dataframes:
            df = self.dataframes[id]
            if df.empty:
                continue
            starttime = df.iloc[0]['time']
            yyyy = UTCDateTime(starttime).year
            samfile = self.get_filename(SAM_DIR, id, yyyy, self.get_sampling_interval(df), ext)
            samdir = os.path.dirname(samfile)
            if not os.path.isdir(samdir):
                os.makedirs(samdir)
            if os.path.isfile(samfile): # COMBINE
                if ext=='csv':
                    original_df = pd.read_csv(samfile)
                elif ext=='pickle':
                    original_df = pd.read_pickle(samfile)
                if not overwrite: # DO NOT COMBINE IF WE ALREADY HAVE DATA FOR THIS TIMEWINDOW
                    endtime = df.iloc[-1]['time']
                    if endtime > starttime:
                        original_df['pddatetime'] = pd.to_datetime(original_df['time'], unit='s')
                        # construct Boolean mask
                        try:
                            mask = original_df['pddatetime'].between(starttime.isoformat(), endtime.isoformat())
                        except:
                            # something went wrong. just try to combine
                            pass
                        else:
                            # apply Boolean mask
                            subset_df = original_df[mask]
                            subset_df = subset_df.drop(columns=['pddatetime'])
                            original_df.drop(columns=['pddatetime'])
                            if len(subset_df)>=len(df):
                                # skip combining
                                continue
                    else:
                        # skip combining
                        continue

                # COMBINING HERE
                combined_df = pd.concat([original_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['time'], keep='last') # overwrite duplicate data
                print(f'Modifying {samfile}')             
                if ext=='csv':
                    combined_df.to_csv(samfile, index=False)
                elif ext=='pickle':
                    combined_df.to_pickle(samfile)
                    
            else: # WRITE FOR FIRST TIME
                # SCAFFOLD: do i need to create a blank file here for whole year? probably not because smooth() is date-aware
                print(f'Writing {samfile}')
                if ext=='csv':
                    df.to_csv(samfile, index=False)
                elif ext=='pickle':
                    df.to_pickle(samfile)

    @staticmethod
    def check_units(st):
        good_st = st
        #print('SAM')
        return good_st
    
    @staticmethod
    def __get_endtime(df):
        ''' return the end time of an SAM dataframe as an ObsPy UTCDateTime'''
        return UTCDateTime(df.iloc[-1]['time'])
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='RSAM'):
        network = id.split('.')[0]
        return os.path.join(SAM_DIR, name, network, '%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))

    @staticmethod
    def __get_npts(df):
        ''' return the number of rows of an SAM dataframe'''
        return len(df)

    @staticmethod
    def get_sampling_interval(df):
        ''' return the sampling interval of an SAM dataframe in seconds '''
        if len(df)>1:
            return df.iloc[1]['time'] - df.iloc[0]['time']  
        else:
            #print(UTCDateTime(df.iloc[0]['time']))
            return 60

    @staticmethod
    def __get_starttime(df):
        ''' return the start time of an SAM dataframe as an ObsPy UTCDateTime'''
        return UTCDateTime(df.iloc[0]['time'])
    
    def __get_trace_ids(self):
        return [id for id in self.dataframes]

            
    def __remove_empty(self):
        ''' remove empty dataframes from an SAM object - these are net.sta.loc.chan for which there are no non-zero data '''
        #print('removing empty dataframes')
        dfs_dict = self.dataframes.copy() # store this so we can delete during loop, otherwise complains about deleting during iteration
        for id in self.dataframes:
            df = self.dataframes[id]
            #print(id, self.dataframes[id]['mean'])
            metrics=self.get_metrics(df=df)
            for metric in metrics:
                if (df[metric] == 0).all() or pd.isna(df[metric]).all():
                    #print('dropping ', metric)
                    dfs_dict[id].drop(columns=[metric])
            if len(df)==0:
                del dfs_dict[id]
        self.dataframes = dfs_dict

    @staticmethod
    def reshape_trace_data(x, sampling_rate, sampling_interval):
        ''' reshape data vector from 1-D to 2-D to support vectorized loop for SAM computation '''
        # reshape the data vector into an array, so we can take matplotlib.pyplot.xticks(fontsize= )advantage of np.mean()
        x = np.absolute(x)
        s = np.size(x) # find the size of the data vector
        nc = int(sampling_rate * sampling_interval) # number of columns
        nr = int(s / nc) # number of rows
        x = x[0:nr*nc] # cut off any trailing samples
        y = x.reshape((nr, nc))
        return y
        
    def get_seed_ids(self):
        seed_ids = list(self.dataframes.keys())
        return seed_ids
        
    def get_metrics(self, df=None):
        if isinstance(df, pd.DataFrame):
            metrics = df.columns[1:]
        else:
            seed_ids = self.get_seed_ids()
            metrics = self.dataframes[seed_ids[0]].columns[1:]
        return metrics
        

    def __str__(self):
        contents=""
        for i, trid in enumerate(self.dataframes):
            df = self.dataframes[trid]
            print(df)
            if i==0:
                contents += f"Metrics: {','.join(df.columns[1:])}" + '\n'
                contents += f"Sampling Interval={self.get_sampling_interval(df)} s" + '\n\n'
            startt = self.__get_starttime(df)
            endt = self.__get_endtime(df)        
            contents += f"{trid}: {startt.isoformat()} to {endt.isoformat()}"
            contents += "\n"
        return contents   
    
    def __len__(self):
        return len(self.dataframes)
    
class RSAM(SAM):
        
    @staticmethod
    def check_units(st):
        #print('RSAM')
        good_st = Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'COUNTS':
                    good_st.append(tr)
            else: # for RSAM, ok if no units
                tr.stats.units = 'Counts'
                good_st.append(tr)
        return good_st
        
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='RSAM'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)

    @classmethod
    def readRSAMbinary(cls, SAM_DIR=None, station=None, stime=None, etime=None,
                    filepath=None, sampling_interval=60, verbose=False,
                    convert_legacy_ids_using_this_network=None):
        """
        Read RSAM binary file(s) and return an RSAM object.
        Either:
        - provide `filepath`, `stime`, and `etime` to read a single file, or
        - provide `SAM_DIR`, `station`, `stime`, and `etime` to read from structured archive.
        `station` may be a string or list.
        """
        
        #from obspy import Trace, Stream, UTCDateTime
        #import os
        import struct
        #import numpy as np
        import re
        from flovopy.core.preprocessing import fix_trace_id #_get_band_code

        def read_single_file(filepath, stime, etime):
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            if verbose:
                print(f"Reading {filepath}")
            records_per_day = int(86400 / sampling_interval)
            values = []
            with open(filepath, "rb") as f:
                f.seek(4 * records_per_day)
                while True:
                    bytes_read = f.read(4)
                    if not bytes_read:
                        break
                    v = struct.unpack("f", bytes_read)[0]
                    values.append(v)

            tr = Trace(data=np.array(values))
            filename = os.path.basename(filepath)
            station_code = ''.join(filter(str.isalpha, filename))

            match = re.search(r'(\d{2,4})', filename)
            if match:
                ystr = match.group(1)
                year = int(ystr) if len(ystr) == 4 else (2000 + int(ystr) if int(ystr) < 50 else 1900 + int(ystr))
            else:
                raise ValueError("No year found in filename")

            tr.stats.starttime = UTCDateTime(year, 1, 1)
            tr.stats.station = station_code
            tr.stats.delta = sampling_interval
            tr.data[tr.data == -998.0] = np.nan
            tr.trim(stime, etime)
            if convert_legacy_ids_using_this_network and not tr.stats.network:
                from flovopy.core.trace_utils import _fix_legacy_id
                fix_trace_id(tr, legacy=True, netcode=convert_legacy_ids_using_this_network)

            return tr

        def read_structured_file(station, year):
            file_path = os.path.join(SAM_DIR, f"{station}{year}.DAT")
            if not os.path.isfile(file_path):
                if verbose:
                    print(f"{file_path} not found")
                return None

            days = 366 if year % 4 == 0 else 365
            records = days * int(86400 / sampling_interval)
            values = []
            with open(file_path, "rb") as f:
                f.seek(4 * int(86400 / sampling_interval))
                for _ in range(records):
                    bytes_read = f.read(4)
                    if not bytes_read:
                        break
                    v = struct.unpack("f", bytes_read)[0]
                    values.append(v)
            tr = Trace(data=np.array(values))
            tr.stats.starttime = UTCDateTime(year, 1, 1)
            tr.stats.station = station
            tr.stats.delta = sampling_interval
            tr.data[tr.data == -998.0] = np.nan
            tr.trim(stime, etime)
            if convert_legacy_ids_using_this_network and not tr.stats.network:
                from flovopy.core.trace_utils import _fix_legacy_id
                fix_trace_id(tr, legacy=True, netcode=convert_legacy_ids_using_this_network)
            return tr

        if filepath:
            tr = read_single_file(filepath, stime, etime)
            return cls(stream=Stream([tr]), sampling_interval=sampling_interval)

        if isinstance(station, list):
            traces = []
            for sta in station:
                try:
                    obj = cls.readRSAMbinary(SAM_DIR=SAM_DIR, station=sta, stime=stime, etime=etime,
                                            sampling_interval=sampling_interval, verbose=verbose,
                                            convert_legacy_ids_using_this_network=convert_legacy_ids_using_this_network)
                    if obj and len(obj)>0:
                        sta_stream = obj.to_stream() # empty if RSAM object contains only  NaN data
                        if len(sta_stream) > 0:    
                            tr = sta_stream[0]
                            if tr.data.size > np.count_nonzero(np.isnan(tr.data)):
                                traces.append(tr)
                except Exception as e:
                    print(f"Error reading station {sta}: {e}")
            return cls(stream=Stream(traces), sampling_interval=sampling_interval)

        if isinstance(station, str):
            traces = []
            for year in range(stime.year, etime.year + 1):
                tr = read_structured_file(station, year)
                if tr:
                    traces.append(tr)
            if traces:
                stream = Stream(traces).merge(method=0, fill_value=np.nan)
                return cls(stream=stream, sampling_interval=sampling_interval)

        raise ValueError("Invalid arguments provided to readRSAMbinary.")


class VSAM(SAM):
    # Before calling, make sure tr.stats.units is fixed to correct units.

    @staticmethod
    def check_units(st):
        #print('VSAM')
        good_st = Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'M/S' or u == 'PA':
                    good_st.append(tr)
        return good_st    

    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VSAM'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)

    @staticmethod
    def compute_geometrical_spreading_correction(this_distance_km, chan, surfaceWaves=False, wavespeed_kms=2000, peakf=2.0):
        #print('peakf =',peakf)
        if surfaceWaves and chan[1]=='H': # make sure seismic channel
            wavelength_km = wavespeed_kms/peakf
            gsc = np.sqrt(np.multiply(this_distance_km, wavelength_km))
        else: # body waves - infrasound always here at local distances
            gsc = this_distance_km
        return gsc
    
    
    @staticmethod
    def compute_inelastic_attenuation_correction(this_distance_km, peakf, wavespeed_kms, Q):
        if Q:
            t = np.divide(this_distance_km, wavespeed_kms) # s
            iac = np.exp(math.pi * peakf * t / Q) 
            return iac
        else:
            return 1.0  
      
    def reduce(self, inventory, source, surfaceWaves=False, Q=None, wavespeed_kms=None, fixpeakf=None):
        # if the original Trace objects had coordinates attached, add a method in SAM to save those
        # in self.inventory. And add to SAM __init___ the possibility to pass an inventory object.
        
        #print(self)
        # Otherwise, need to pass an inventory here.
        if not wavespeed_kms:
            if surfaceWaves:
                wavespeed_kms=2 # km/s
            else:
                wavespeed_kms=3 # km/s
        
        # Need to pass a source too, which should be a dict with name, lat, lon, elev.
        print(f'SCAFFOLD: {inventory}')
        distance_km, coordinates = self.get_distance_km(inventory, source)

        corrected_dataframes = {}
        for seed_id, df0 in self.dataframes.items():
            if not seed_id in distance_km:
                continue
            df = df0.copy()
            this_distance_km = distance_km[seed_id]
            ratio = df['VT'].sum()/df['LP'].sum()
            if fixpeakf:
                peakf = fixpeakf
            else:
                peakf = np.sqrt(ratio) * 4
            net, sta, loc, chan = seed_id.split('.')    
            gsc = self.compute_geometrical_spreading_correction(this_distance_km, chan, surfaceWaves=surfaceWaves, \
                                                           wavespeed_kms=wavespeed_kms, peakf=peakf)
            if Q:
                iac = self.compute_inelastic_attenuation_correction(this_distance_km, peakf, wavespeed_kms, Q)
            else:
                iac = 1.0
            for col in df.columns:
                if col in self.get_metrics(): 
                    if col=='VLP':
                        gscvlp = self.compute_geometrical_spreading_correction(this_distance_km, chan, surfaceWaves=surfaceWaves, \
                                                           wavespeed_kms=wavespeed_kms, peakf=0.06)
                        iacvlp = self.compute_inelastic_attenuation_correction(this_distance_km, 0.06, wavespeed_kms, Q)
                        df[col] = df[col] * gscvlp * iacvlp * 1e7 # convert to cm^2/s (or cm^2 for DR)
                    else:
                        df[col] = df[col] * gsc * iac * 1e7                    
            corrected_dataframes[seed_id] = df
        return corrected_dataframes
    
    def compute_reduced_velocity(self, inventory, source, surfaceWaves=False, Q=None, wavespeed_kms=None, peakf=None):
        corrected_dataframes = self.reduce(inventory, source, surfaceWaves=surfaceWaves, Q=Q, wavespeed_kms=wavspeed_kms, fixpeakf=peakf)
        if surfaceWaves:
            return VRS(dataframes=corrected_dataframes)
        else:
            return VR(dataframes=corrected_dataframes)
        

class DSAM(VSAM):
    
    @staticmethod
    def check_units(st):
        #print('DSAM')
        good_st = Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'M' or u == 'PA':
                    good_st.append(tr)
                else:
                    print(f'DSAM: skipping {tr}: units are wrong {tr.stats.units}')

            elif tr.stats.channel[1]=='H':
                tr.stats['units'] = 'm'
                good_st.append(tr)
            elif tr.stats.channel[1]=='D':
                tr.stats['units'] = 'Pa'
                good_st.append(tr)                
            else:
                print(f'DSAM: skipping {tr}: malformed channel code, not seismometer or pressure sensor')
        return good_st

    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='DSAM'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)

    def compute_reduced_displacement(self, inventory, source, surfaceWaves=False, Q=None, wavespeed_kms=2.0, peakf=None):
        print(f'SCAFFOLD: {inventory}')
        corrected_dataframes = self.reduce(inventory, source, surfaceWaves=surfaceWaves, Q=Q, wavespeed_kms=wavespeed_kms, fixpeakf=peakf)
        if surfaceWaves:
            return DRS(dataframes=corrected_dataframes)
        else:
            return DR(dataframes=corrected_dataframes)


class VSEM(VSAM):

    def __init__(self, dataframes=None, stream=None, sampling_interval=60.0, filter=[0.5, 18.0], bands = {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 'VT':[4.0, 18.0]}, corners=4, verbose=False):
        ''' Create a VSEM object 
        
            Optional name-value pairs:
                dataframes: Create an VSEM object using these dataframes. Used by downsample() method, for example. Default: None.
                stream: Create an VSEM object from this ObsPy.Stream object.
                sampling_interval: Compute VSEM data using this sampling interval (in seconds). Default: 60
                filter: list of two floats, representing fmin and fmax. Default: [0.5, 18.0]. Set to None if no filter wanted.
                bands: a dictionary of filter bands and corresponding column names. Default: {'VLP': [0.02, 0.2], 'LP':[0.5, 4.0], 
                    'VT':[4.0, 18.0]}. For example, the default setting creates 3 additional columns for each DataFrame called 
                    'VLP', 'LP', and 'VT', which contain the mean value for each sampling_interval within the specified filter band
                    (e.g. 0.02-0.2 Hz for VLP). If 'LP' and 'VT' are in this dictionary, an extra column called 'fratio' will also 
                    be computed, which is the log2 of the ratio of the 'VT' column to the 'LP' column, following the definition of
                    frequency ratio by Rodgers et al. (2015).
        '''
        self.dataframes = {} 

        if isinstance(dataframes, dict):
            good_dataframes = {}
            for id, df in dataframes.items():
                if isinstance(df, pd.DataFrame):
                    good_dataframes[id]=df
            if len(good_dataframes)>0:
                self.dataframes = good_dataframes
                if verbose:
                    print('dataframes found. ignoring other arguments.')
                return
            else:
                print('no valid dataframes found')
                pass

        if not isinstance(stream, Stream):
            # empty VSEM object
            print('creating blank VSEM object')
            return

        good_stream = self.check_units(stream)
        if verbose:
            print('good_stream:\n',good_stream)


        if len(good_stream)>0:
            if good_stream[0].stats.sampling_rate == 1/sampling_interval:
                # no downsampling to do
                for tr in good_stream:
                    df = pd.DataFrame()
                    df['time'] = pd.Series(tr.times('timestamp'))
                    df['mean'] = pd.Series(tr.data) 
                    self.dataframes[tr.id] = df
                return 
            elif good_stream[0].stats.sampling_rate < 1/sampling_interval:
                print('error: cannot compute SAM for a Stream with a tr.stats.delta bigger than requested sampling interval')
                return
            
        for tr in good_stream:
            if tr.stats.npts < tr.stats.sampling_rate * sampling_interval:
                print('Not enough samples for ',tr.id,'. Skipping.')
                continue
            #print(tr.id, 'absolute=',absolute)
            df = pd.DataFrame()
            
            t = tr.times('timestamp') # Unix epoch time
            sampling_rate = tr.stats.sampling_rate
            t = self.reshape_trace_data(t, sampling_rate, sampling_interval)
            df['time'] = pd.Series(np.nanmin(t,axis=1))

            if filter:
                if tr.stats.sampling_rate<filter[1]*2.2:
                    print(f"{tr}: Sampling rate must be at least {filter[1]*2.2:.1f}. Skipping.")
                    continue
                tr2 = tr.copy()
                tr2.detrend('demean')
                tr2.filter('bandpass', freqmin=filter[0], freqmax=filter[1], corners=corners)
                y = self.reshape_trace_data(np.absolute(tr2.data), sampling_rate, sampling_interval)
            else:
                y = self.reshape_trace_data(np.absolute(tr.data), sampling_rate, sampling_interval)
 
            df['energy'] = pd.Series(np.nansum(np.square(y),axis=1)/tr.stats.sampling_rate)

            if bands:
                for key in bands:
                    tr2 = tr.copy()
                    [flow, fhigh] = bands[key]
                    tr2.filter('bandpass', freqmin=flow, freqmax=fhigh, corners=corners)
                    y = self.reshape_trace_data(abs(tr2.data), sampling_rate, sampling_interval)
                    df[key] = pd.Series(np.nansum(np.square(y),axis=1)) 
  
            self.dataframes[tr.id] = df

    @staticmethod
    def check_units(st):
        #print('VSEM')
        good_st = Stream()
        for tr in st:
            if 'units' in tr.stats:
                u = tr.stats['units'].upper()
                if u == 'M/S' or u == 'PA':
                #if u == 'M2/S' or u == 'PA2':
                    good_st.append(tr)
            elif tr.stats.channel[1]=='H':
                tr.stats['units'] = 'm/s'
                good_st.append(tr)
                
        return good_st  
    
    def reduce(self, inventory, source, Q=None, wavespeed_kms=None, fixpeakf=None):
        # if the original Trace objects had coordinates attached, add a method in SAM to save those
        # in self.inventory. And add to SAM __init___ the possibility to pass an inventory object.
        
        #print(self)
        # Otherwise, need to pass an inventory here.

        if not wavespeed_kms:
            wavespeed_kms=3 # km/s
        
        # Need to pass a source too, which should be a dict with name, lat, lon, elev.
        distance_km, coordinates = self.get_distance_km(inventory, source)

        corrected_dataframes = {}
        for seed_id, df0 in self.dataframes.items():
            if not seed_id in distance_km:
                continue
            df = df0.copy()
            this_distance_km = distance_km[seed_id]
            ratio = df['VT'].sum()/df['LP'].sum()
            if fixpeakf:
                peakf = fixpeakf
            else:
                peakf = np.sqrt(ratio) * 4

            net, sta, loc, chan = seed_id.split('.') 
            esc = self.Eseismic_correction(this_distance_km*1000) # equivalent of gsc
            if Q:
                iac = self.compute_inelastic_attenuation_correction(this_distance_km, peakf, wavespeed_kms, Q)
            else:
                iac = 1.0
            for col in df.columns:
                if col in self.get_metrics(): 
                    if col=='VLP':
                        iacvlp = self.compute_inelastic_attenuation_correction(this_distance_km, 0.06, wavespeed_kms, Q)
                        df[col] = df[col] * esc * iacvlp # do i need to multiply by iac**2 since this is energy?
                    else:
                        df[col] = df[col] * esc * iac # do i need to multiply by iac**2 since this is energy?
            corrected_dataframes[seed_id] = df
        return corrected_dataframes
       
    def compute_reduced_energy(self, inventory, source, Q=None):
        corrected_dataframes = self.reduce(inventory, source, Q=Q)
        return ER(dataframes=corrected_dataframes)

    @staticmethod
    def Eacoustic_correction(r, c=340, rho=1.2): 
        Eac = (2 * math.pi * r**2) / (rho * c)
        return Eac

    @staticmethod
    def Eseismic_correction(r, c=3000, rho=2500, S=1, A=1): # a body wave formula
        Esc = (2 * math.pi * r**2) * rho * c * S**2/A
        return Esc

    def downsample(self, new_sampling_interval=3600):
        ''' downsample a VSEM object to a larger sampling interval(e.g. from 1 minute to 1 hour). Returns a new VSEM object.
         
            Optional name-value pair:
                new_sampling_interval: the new sampling interval (in seconds) to downsample to. Default: 3600
        '''

        dataframes = {}
        for id in self.dataframes:
            df = self.dataframes[id]
            df['date'] = pd.to_datetime(df['time'], unit='s')
            old_sampling_interval = self.get_sampling_interval(df)
            if new_sampling_interval > old_sampling_interval:
                freq = '%.0fmin' % (new_sampling_interval/60)
                new_df = df.groupby(pd.Grouper(key='date', freq=freq)).sum()
                new_df.reset_index(drop=True)
                dataframes[id] = new_df
            else:
                print('Cannot downsample to a smaller sampling interval')
        return self.__class__(dataframes=dataframes) 
            
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VSEM'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)
	    
	    

class DR(SAM):
    def __init__(self, dataframes=None):
 
        self.dataframes = {} 

        if isinstance(dataframes, dict):
            good_dataframes = {}
            for id, df in dataframes.items():
                if isinstance(df, pd.DataFrame):
                    good_dataframes[id]=df
            if len(good_dataframes)>0:
                self.dataframes = good_dataframes
                #print('dataframes found. ignoring other arguments.')
                return
            else:
                pass
                #print('no valid dataframes found')
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='DR'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)
	    
    def linearplot(st, equal_scale=False, percentile=None, linestyle='-'):
    	hf = st.plot(handle=True, equal_scale=equal_scale, linestyle=linestyle) #, method='full'); # standard ObsPy plot
    	# change the y-axis so it starts at 0
    	allAxes = hf.get_axes()
    	ylimupper = [ax.get_ylim()[1] for ax in allAxes]
    	print(ylimupper)
    	if percentile:
        	ylimupper = np.array([np.percentile(tr.data, percentile) for tr in st])*1.1
    	# if equal_scale True, we set to maximum scale
    	print(ylimupper)
    	ymax=max(ylimupper)
    	for i, ax in enumerate(allAxes):
            if equal_scale==True:
            	ax.set_ylim([0, ymax])
            else:
            	ax.set_ylim([0, ylimupper[i]])  

    def iceweb_plot(self, metric='median', equal_scale=False, type='log', percentile=None, linestyle='-', outfile=None):
        measurement = self.__class__.__name__
        if measurement[1]=='R':
            if measurement[0]=='D':
                units = f"(cm\N{SUPERSCRIPT TWO})"
            elif measurement[0]=='V':
                units = f"(cm\N{SUPERSCRIPT TWO}/s)"
            subscript = "{%s}" % measurement[1:]
            
            measurement = f"${measurement[0]}_{subscript}$"
        st = self.to_stream(metric=metric)
        for tr in st:
            tr.data = np.where(tr.data==0, np.nan, tr.data)
        if type=='linear':
            linearplot(st, equal_scale=equal_scale, percentile=percentile, linestyle=linestyle)
        elif type=='log':

            plt.rcParams["figure.figsize"] = (10,6)
            fig, ax = plt.subplots()
            for tr in st:  
                t = [this_t.datetime for this_t in tr.times("utcdatetime") ]     
                ax.semilogy(t, tr.data, linestyle, label='%s' % tr.id) #, alpha=0.03) 1e7 is conversion from amplitude in m at 1000 m to cm^2
            ax.format_xdata = mdates.DateFormatter('%H')
            ax.legend()
            plt.xticks(rotation=90)
            plt.ylim((0.2, 100)) # IceWeb plots went from 0.05-30
            plt.yticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], \
                ['0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'])
            #plt.ylabel(r'$D_{RS}$ ($cm^{2}$)')
            #plt.xlabel(r'UTC / each point is max $D_{RS}$ in %d minute window' % (st[0].stats.delta/60))
            #plt.title('Reduced Displacement (%s)\n%s to %s' % (r'$D_{RS}$', t[0].strftime('%d-%b-%Y %H:%M:%S UTC'), t[-1].strftime('%d-%b-%Y %H:%M:%S UTC')))
            plt.ylabel(measurement + units)
            plt.xlabel(r'UTC / each point is max %s in %d minute window' % (measurement, st[0].stats.delta/60))
            plt.title('Reduced Displacement (%s)\n%s to %s' % (measurement, t[0].strftime('%d-%b-%Y %H:%M:%S UTC'), t[-1].strftime('%d-%b-%Y %H:%M:%S UTC')))            
            plt.xticks(fontsize=6)
            if outfile:
            	plt.savefig(outfile)
            else:
            	plt.show()
     
    def max(self, metric='rms'):
        lod = []
        #print(type(self))
        if metric=='rms' and not 'rms' in self.get_metrics():
            metric='std'
        allmax = []
        classname = self.__class__.__name__
        for seed_id in self.dataframes:
            df = self.dataframes[seed_id]
            thismax = df[metric].max()
            if thismax == 0 or np.isnan(thismax):
                continue
            #print(f"{seed_id}: {thismax:.1e} m at 1 km" )
            #maxes[seed_id] = thismax
            allmax.append(thismax)
            thisDict = {'seed_id': seed_id, classname:np.round(thismax,2)}
            lod.append(thisDict)
        allmax = np.array(sorted(allmax))
        medianMax = np.median(allmax) 
        #print(f"Network: {medianMax:.1e} m at 1 km" )   
        networkMax = np.round(medianMax,2)
        thisDict = {'seed_id':'Network', classname:networkMax}
        lod.append(thisDict)
        df = pd.DataFrame(lod)
        #display(df)
        return networkMax

    def show_percentiles(self, metric):
        st = self.to_stream(metric=metric)
        #fig, ax = plt.subplots(len(st), 1)
        for idx,tr in enumerate(st):
            if tr.id.split('.')[-1][-1]!='Z':
                continue
            y = tr.data #.sort()
            p = [p for p in range(101)]
            h = np.percentile(y, p)
            #ax[idx].plot(p[1:], np.diff(h))
            #ax[idx].set_title(f'percentiles for {tr.id}')
            plt.figure()
            plt.semilogy(p[1:], np.diff(h))
            plt.title(f'percentiles for {tr.id}')
                    
    def examine_spread(self, low_percentile=50, high_percentile=98):
        medians = {}
        station_corrections = {}
        metrics = self.get_metrics()
        for bad_metric in ['min', 'max', 'fratio']:
            if bad_metric in metrics:
                metrics = metrics.drop(bad_metric)       
        seed_ids = self.get_seed_ids()
        for metric in metrics: 
            st = self.to_stream(metric=metric)
            medians[metric] = {}
            station_corrections[metric] = {}
            m_array = []
            for tr in st:
                y = np.where( (tr.data>np.percentile(tr.data, low_percentile)) & (tr.data<np.percentile(tr.data, high_percentile)), tr.data, np.nan)
                m = np.nanmedian(y)
                medians[metric][tr.id] =m 
                m_array.append(y)
            m_array=np.array(m_array)
            medians[metric]['network_median'] = np.nanmedian(m_array)
            s_array = []
            for tr in st:
                s = medians[metric]['network_median']/medians[metric][tr.id]
                station_corrections[metric][tr.id] = s
                s_array.append(s)
            
            s_array = np.array(s_array)
            for idx, element in enumerate(s_array):
                if element < 0.1 or element > 10.0:
                    s_array[idx]=np.nan
                if element < 1.0:
                    s_array[idx]=1.0/element     
            station_corrections[metric]['network_std'] = np.nanstd(s_array)
 
            print('\nmetric: ', metric)
            for seed_id in seed_ids:
                print(f"{seed_id}, median: {medians[metric][seed_id]:.3e}, station correction: {station_corrections[metric][seed_id]:.3f}")
            print(f"network: median: {medians[metric]['network_median']:.03e}, station correction std: {station_corrections[metric]['network_std']:.03e}") 
        return medians, station_corrections
                
    def apply_station_corrections(self, station_corrections):
        for seed_id in self.dataframes:
            df = self.dataframes[seed_id]
            for metric in self.get_metrics():
                if metric in station_corrections:
                    if seed_id in station_corrections[metric]: 
                        df[metric] = df[metric] * station_corrections[metric][seed_id]
                    
                      
    def compute_average_dataframe(self, average='mean'):
        ''' 
        Average a SAM object across the whole network of seed_ids
        This is primarily a tool for then making iceweb_plot's with just one representative trace
        It is particularly designed to be used after running examine_spread and apply_station_corrections
        '''
        df = pd.DataFrame()
        for metric in self.get_metrics():
            st = self.to_stream(metric)
            df['time'] = self.dataframes[st[0].id]['time']
            all_data_arrays = []
            for tr in st:
                all_data_arrays.append(tr.data)
            twoDarray = np.stac
#import pytzk(all_data_arrays)
            if average=='mean':
                df[metric] = pd.Series(np.nanmean(y,axis=1))  
            elif average=='median':
                df[metric] = pd.Series(np.nanmedian(y,axis=1))
            net, sta, loc, chan = st[0].id.split('.')
            average_id = '.'.join(net, 'AVRGE', loc, chan)
            dataframes[average_id] = df
        return self.__class__(dataframes=dataframes)  
            
        
class DRS(DR):
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='DRS'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)
	    
class VR(DR):
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VR'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)
	    
class VRS(DR):
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='VRS'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)   
	    	    
class ER(DR):

    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='ER'):
        return SAM.get_filename(SAM_DIR, id, year, sampling_interval, ext, name=name)   
       
    def sum_energy(self, startt=None, endt=None, metric='energy', a=-3.2, b=2/3): #, inventory, source):
        st = self.to_stream(metric)
        if startt and endt:
            st.trim(starttime=startt, endtime=endt)
        #r_km, coords = self.get_distance_km(inventory, source)
        lod = []
        allE = []
        allM = []
        for tr in st:
            
            #r = r_km[tr.id] * 1000.0
            e = np.nansum(tr.data)
            if e==0:
                continue
            m = np.round(energy2magnitude(e, a=a, b=b),2)
            allE.append(e)
            allM.append(m)
            print(f"{tr.id}: Joules: {e:.2e}, Magnitude: {m:.1f}")
            thisDict = {'seed_id':tr.id, 'Energy':e, 'EMag':m}
            lod.append(thisDict)

        medianE = np.median(allE)
        medianM = np.round(np.median(allM),2)
        print(f"Network: Joules: {medianE:.2e}, Magnitude: {medianM:.1f}")
        thisDict = {'seed_id':'Network', 'Energy':medianE, 'EMag':medianM}
        lod.append(thisDict)
        df = pd.DataFrame(lod)  
        return medianE, medianM

    #def plot():
    #    pass
    
    @staticmethod
    def get_filename(SAM_DIR, id, year, sampling_interval, ext, name='ER'):
	    return os.path.join(SAM_DIR,'%s_%s_%4d_%ds.%s' % (name, id, year, sampling_interval, ext))	   

def magnitude2energy(ME, a=-3.2, b=2/3):
    ''' 
    Convert (a vector of) magnitude into (a vector of) equivalent energy(ies).
   
    Conversion is based on the equation 7 Hanks and Kanamori (1979):
 
       mag = 2/3 * log10(energy) - 4.7
 
    That is, energy (Joules) is roughly proportional to the peak amplitude to the power of 1.5.
    This obviously is based on earthquake waveforms following a characteristic shape.
    For a waveform of constant amplitude, energy would be proportional to peak amplitude
    to the power of 2.
 
    For Montserrat data, when calibrating against events in the SRU catalog, a factor of
    a=-3.7 was preferred to a=-4.7.
    
    based on https://github.com/geoscience-community-codes/GISMO/blob/master/core/%2Bmagnitude/mag2eng.m

    *** Edited 2024/04/17

    Now based on 2024 SSA Presentation:

    ME = 2/3 log10(E) - 3.7

    ME = b * log10(E) + a

    So:

    E = 10^(1.5 * (ME + 3.7))

    E = 10^(1/b * (ME - a))
    
    E = 3 x 10^5 x 10^(1.5 ME)
    
    *** Edited 2024/04/18

    Now based on Choy & Boatright [1995] who find best value for a=-3.2, so altering default 
    '''

    #eng = np.power(10, (1/b * mag - a ))
    E = np.power(10, (1/b * (ME - a) ))
    
    return E 
    
    
def energy2magnitude(E, a=-3.2, b=2/3):
    '''
    Convert a (vector of) magnitude(s) into a (vector of) equivalent energy(/ies).
    
    Conversion is based on the the following formula from Hanks and Kanamori (1979):
 
       mag = 2/3 * log10(energy) - 4.7
 
    That is, energy (Joules) is roughly proportional to the peak amplitude to the power of 1.5.
    This obviously is based on earthquake waveforms following a characteristic shape.
    For a waveform of constant amplitude, energy would be proportional to peak amplitude
    to the power of 2.
 
    For Montserrat data, when calibrating against events in the SRU catalog, a factor of
    3.7 was preferred to 4.7.

    based on https://github.com/geoscience-community-codes/GISMO/blob/master/core/%2Bmagnitude/eng2mag.m
    
    *** Edited 2024/04/17

    Now based on 2024 SSA Presentation:

    ME = 2/3 log10(E) - 3.7

    ME = b * log10(E) + a

    So:

    E = 10^(1.5 * (ME + 3.7))

    E = 10^(1/b * (ME - a))

    *** Edited 2024/04/18

    Now based on Choy & Boatright [1995] who find best value for a=-3.2, so altering default
     
    '''
    
    #mag = b * (np.log10(eng) + a) 	

    ME = b * np.log10(E) + a
    return ME

import argparse
from obspy.clients.fdsn import Client
from obspy import Stream, UTCDateTime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Z/N/E components from IRIS/EarthScope FDSN and compute RSAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data/service
    parser.add_argument("--service", default="IRIS",
                        help="FDSN service name or base URL.")
    parser.add_argument("--network", default="IU",
                        help="FDSN network code (e.g., IU, II, IC).")
    parser.add_argument("--station", default="DWPF",
                        help="Station code (e.g., ANMO).")
    parser.add_argument("--location", default="*",
                        help="Location code (use '*' for any).")

    # Time range (defaults form a canned example)
    parser.add_argument("--startdate", default="2011-03-10",
                        help="Start ISO UTC: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.")
    parser.add_argument("--enddate", default="2011-03-15",
                        help="End ISO UTC (not inclusive of the last day's end).")

    # RSAM settings
    parser.add_argument("--sampling_interval", type=float, default=60.0,
                        help="RSAM sampling interval (s).")
    parser.add_argument("--minfreq", type=float, default=0.5,
                        help="Bandpass low-cut for RSAM (Hz).")
    parser.add_argument("--maxfreq", type=float, default=18.0,
                        help="Bandpass high-cut for RSAM (Hz).")
    parser.add_argument("--corners", type=int, default=4,
                        help="IIR filter corners for bandpass.")

    # I/O
    parser.add_argument("--sam_dir", default="SAM_OUT",
                        help="Output directory root for SAM/RSAM archive.")
    parser.add_argument("--ext", choices=["pickle", "csv"], default="pickle",
                        help="Output format for RSAM files.")

    # Misc
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging.")
    return parser.parse_args()


def _build_channel_csv():
    # Try these in priority order; request explicit Z/N/E
    families = ["HH?", "BH?", "EH?", "LH?"]
    chans = []
    for fam in families:
        chans += [fam.replace("?", c) for c in ("Z", "N", "E")]
    return ",".join(chans)


def _process_day(client, args, day_start, day_end):
    channel_csv = _build_channel_csv()

    if args.verbose:
        print(f"\n=== {day_start.isoformat()} to {day_end.isoformat()} ===")

    # Fetch waveforms
    try:
        st_all = client.get_waveforms(
            network=args.network,
            station=args.station,
            location=args.location,
            channel=channel_csv,
            starttime=day_start,
            endtime=day_end,
            attach_response=False,
        )
    except Exception as e:
        if args.verbose:
            print(f"Waveform request failed for {day_start.date}: {e}")
        return

    if len(st_all) == 0:
        if args.verbose:
            print("No data returned for this day.")
        return

    # Merge segments; keep Z/N/E only
    try:
        st_all.merge(method=1, fill_value=None)
    except Exception as e:
        if args.verbose:
            print(f"Merge warning: {e}")

    st_clean = Stream([tr for tr in st_all if tr.stats.channel[-1] in ("Z", "N", "E")])
    if len(st_clean) == 0:
        if args.verbose:
            print("No Z/N/E components found after filtering.")
        return

    # Compute and write RSAM
    try:
        rsamObject = RSAM(
            stream=st_clean,
            sampling_interval=args.sampling_interval,
            filter=[args.minfreq, args.maxfreq],
            corners=args.corners,
            verbose=args.verbose,
        )
        if len(rsamObject) > 0:
            rsamObject.write(SAM_DIR=args.sam_dir, ext=args.ext, overwrite=False, verbose=args.verbose)
            if args.verbose:
                print("RSAM written.")
        else:
            if args.verbose:
                print("RSAM object is empty; nothing to write.")
    except Exception as e:
        print(f"RSAM/write failed for {day_start.date}: {e}")


def main():
    args = parse_args()
    client = Client(args.service)

    # Parse to UTCDateTime (accepts date-only or full ISO)
    start_utc = UTCDateTime(args.startdate)
    end_utc = UTCDateTime(args.enddate)

    day = start_utc
    one_day = 24 * 3600
    while day < end_utc:
        day_start = day
        day_end = min(day_start + one_day, end_utc)
        _process_day(client, args, day_start, day_end)
        day += one_day

    if args.verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()