import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from obspy import UTCDateTime, Trace, Stream
from scipy import interpolate
from matplotlib import mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
pqlx = cm.viridis  # or your preferred default colormap


def numpyarrays2dataframe(T,F,S):
    df = pd.DataFrame(np.transpose(S), columns=[str(thisf) for thisf in F])
    df.insert(0, 'time', pd.Series([(spectime + t) for t in T]) )
    return df

def dataframe2numpyarrays(df):
    F = np.array([float(thisf) for thisf in df.columns[1:] ], dtype='float64')
    utcdt = [UTCDateTime(t) for t in df['time'] ]
    T = np.array([t - utcdt[0] for t in utcdt],  dtype='float64')
    S = np.transpose(df.iloc[:, 1:].to_numpy( dtype='float64'))
    return T, F, S, utcdt

def TFS2trace(T, F, S, starttime, seed_id=None):
    tr = Trace()
    if seed_id:
        tr.id = seed_id
    tr.stats.delta = T[1]-T[0]
    tr.stats.starttime = starttime
    f_indexes_all = np.intersect1d(np.where(F>0.5), np.where(F<18.0))
    #tr.data = df.iloc[:,f_indexes_all[0]:f_indexes_all[-1]].sum(axis=1).to_numpy()
    tr.data = np.sum(S[f_indexes_all[0]:f_indexes_all[-1], :], axis=0)
    #tr.data = np.sum(S, axis=0)
    tr.stats['spectrogramdata'] = {'T':T, 'F':F, 'S':S}  
    return tr

class icewebSpectrogram:
    
    def __init__(self, stream=None, secsPerFFT=-1):
        """
        :type st: Stream
        :param st: an ObsPy Stream object. No detrending or filtering is done, so do that before calling this function.
        """
        self.stream = Stream()
        self.precomputed = False
        if isinstance(stream, Stream):
            self.stream = stream
            count = 0
            for tr in self.stream:
                if 'spectrogramdata' in tr.stats:
                    count += 1
    
            if len(self.stream)==count:
                self.precomputed = True
       
    def __str__(self):
        str = '\n\nicewebSpectrogram:\n'
        str += self.stream.__str__()
        #str += '\nF: %d 1-D numpy arrays' % len(self.F)
        #str += '\nT: %d 1-D numpy arrays' % len(self.T)
        #str += '\nS: %d 2-D numpy arrays\n\n' % len(self.S)
        return str
    
    def precompute(self, secsPerFFT=None):
        """
    
        For each Trace in self.stream, call compute_spectrogram. T,F and S arrays will be saved in tr.stats.spectrogramdata
    
        :type secsPerFFT: int or float
        :param secsPerFFT: Window length for fft in seconds. If this parameter is too
            small, the calculation will take forever. If None, it defaults to
            ceil(sampling_rate/100.0).
        """

        # seconds to use for each FFT. 1 second if the event duration is <= 100 seconds, 6 seconds if it is 10-minutes
        if secsPerFFT is None:
            secsPerFFT = np.ceil((self.stream[0].stats.delta * self.stream[0].stats.npts)/100) 
        
        for tr in self.stream:
            [T, F, S] = compute_spectrogram(tr, wlen=secsPerFFT)
            tr.stats['spectrogramdata'] = {'T':T, 'F':F, 'S':S}
        self.precomputed = True
    
        return self    

    
    def plot(self, outfile=None, secsPerFFT=None, fmin=0.5, fmax=20.0, log=False, cmap=pqlx, clim=None, \
                      equal_scale=False, title=None, add_colorbar=True, precompute=False, dbscale=False, trace_indexes=[] ):   
        """
        For each Trace in a Stream, plot the seismogram and spectrogram. This results in 2*N subplots 
        on the figure, where N is the number of Trace objects.
        This is modelled after IceWeb spectrograms, which have been part of the real-time monitoring 
        system at the Alaska Volcano Observatory since March 1998
        MATLAB code for this exists in the GISMO toolbox at 
        https://github.com/geoscience-community-codes/GISMO/blob/master/applications/+iceweb/spectrogram_iceweb.m
    
        :type outfile: str
        :param outfile: String for the filename of output file     
        :type fmin: float
        :param fmin: frequency minimum to plot on spectrograms
        :type fmax: float
        :param fmax: frequency maximum to plot on spectrograms    
        :type log: bool
        :param log: Logarithmic frequency axis if True, linear frequency axis
            otherwise.
        :type cmap: :class:`matplotlib.colors.Colormap`
        :param cmap: Specify a custom colormap instance. If not specified, then the
            pqlx colormap is used. viridis_white_r might be worth trying too.
        :type clim: [float, float]
        :param clim: colormap limits. adjust colormap to clip at lower and/or upper end.
            This overrides equal_scale parameter.
        :type equal_scale: bool
        :param equal_scale: Apply the same colormap limits to each spectrogram if True. 
            This requires more memory since all spectrograms have to be pre-computed 
            to determine overall min and max spectral amplitude within [fmin, fmax]. 
            If False (default), each spectrogram is individually scaled, which is best 
            for seeing details. equal_scale is overridden by clim if given.
        :type add_colorbar: bool
        :param add_colorbar: Add colorbars for each spectrogram (5% space will be created on RHS)        
        :type title: str
        :param title: String for figure super title
        :type dbscale: bool
        :param dbscale: If True 20 * log10 of color values is taken. 
        :type trace_indexes: list of int (or None:default)
        :param trace_indexes: Only plot spectrograms for these trace indexes, if set.
        """

        #print(self.__dict__)
        
        
        if not self.precomputed and precompute:
            self = self.precompute(secsPerFFT=secsPerFFT)
        
        st = self.stream
        if len(trace_indexes)>0: # same logic as metrics.select_by_index_list(st, chosen)
            st = Stream()
            for i, tr in enumerate(self.stream):
                if i in trace_indexes:
                    st.append(tr)

        N = len(st) # number of channels we are plotting        
        if N==0:
            print('Stream object is empty. Nothing to do')
            return
         
        fig, ax = plt.subplots(N*2, 1); # create fig and ax handles with approx positions for now
        fig.set_size_inches(5.76, 7.56); 

        if clim:
            if clim[0]<=clim[1]/100000:
                print('Warning: Lower clim should be at least 1/10000 of Upper clim. This translates to a 100 dB range in amplitude')
                clim[0]=clim[1]/100000
 
        if equal_scale and not clim: # calculate range of spectral amplitudes
            if self.precomputed:
                Smin, Smax = icewebSpectrogram.get_S_range(self, fmin=fmin, fmax=fmax)
            else:
                index_min = np.argmin(st.max()) # find the index of largest Trace object
                [T, F, S] = compute_spectrogram(st[index_min], wlen=secsPerFFT)
                f_indexes = np.intersect1d(np.where(F>=fmin), np.where(F<fmax))
                S_filtered = S[f_indexes, :]
                Smax = np.nanmax(S_filtered)
                S_filtered[S_filtered == 0] = Smax
                Smin = np.nanmin(S_filtered)
                
            if Smin<Smax*1e-6: # impose a dynamic range limit of 1,000,000
                Smin=Smax*1e-6
                
            clim = (Smin, Smax)
        
        for c, tr in enumerate(st):
            if self.precomputed: 
                T = tr.stats.spectrogramdata.T
                F = tr.stats.spectrogramdata.F
                S = tr.stats.spectrogramdata.S          
            else:                
                [T, F, S] = compute_spectrogram(tr, wlen=secsPerFFT)             
        
            # fix the axes positions for this trace and spectrogram, making space for a colorbar at bottom if using a fixed scale
            if add_colorbar and clim:
                spectrogramPosition, tracePosition = icewebSpectrogram.calculateSubplotPositions(N, c, 
                                                                       frameBottom = 0.17, totalHeight = 0.80)
                #cax = fig.add_axes([spectrogramPosition[0], 0.08, spectrogramPosition[2], 0.02])
                #cax.set_xticks([])
                #cax.set_yticks([])


            else:
                spectrogramPosition, tracePosition = icewebSpectrogram.calculateSubplotPositions(N, c)
            ax[c*2].set_position(tracePosition)
            ax[c*2+1].set_position(spectrogramPosition)
        
            # plot the trace
            t = tr.times()
            ax[c*2].plot(t, tr.data, linewidth=0.5);
            ax[c*2].set_yticks(ticks=[]) # turn off yticks
        
            if log:
                # Log scaling for frequency values (y-axis)
                ax[c*2+1].set_yscale('log')
           
            # Plot spectrogram
            vmin = None
            vmax = None
            if clim:
                if dbscale:
                    vmin, vmax = amp2dB(clim)
                else:
                    vmin, vmax = clim
            if dbscale:
                S = amp2dB(S)
            
            #print(vmin, vmax)
            sgram_handle = ax[c*2+1].pcolormesh(T, F, S, vmin=vmin, vmax=vmax, cmap=cmap );
            ax[c*2+1].set_ylim(fmin, fmax)

            # turn off xticklabels, except for bottom panel
            ax[c*2].set_xticklabels([])
            if c<N-1:
                ax[c*2+1].set_xticklabels([])

            # add a ylabel
            ax[c*2+1].set_ylabel('     ' + tr.stats.station + '.' + tr.stats.channel, rotation=90)
            #ax[c*2+1].set_ylabel('     ' + tr.stats.id, rotation=70, fontsize=10)
        
            # Plot colorbar
            if add_colorbar:
                if clim: # Scaled. Add a colorbar at the bottom of the figure. Just do it once.
                    if c==0:
                        #fig.colorbar(sgram_handle, cax=cax, orientation='horizontal'); 
                        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                        #cax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
                        cax = fig.add_axes([spectrogramPosition[0], 0.08, spectrogramPosition[2], 0.02])
                        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', label='Counts/Hz')
                        if dbscale:
                            cax.set_xlabel('dB relative to 1 m/s/Hz')
                            ''' 
                            # Failed attempt to add a second set of labels to top of colorbar
                            # It just wipes out the original xtick labels
                            cax2 = cax.twinx()
                            cax2.xaxis.set_ticks_position("top")
                            xticks = cax.get_xticks()
                            xticks = np.power(xticks/20, 10)
                            cax2.set_xticks(xticks)
                            cax2.set_xlabel('m/s/Hz')
                            '''
                        else:
                            cax.set_xlabel('m/s/Hz')
                else: # Unscaled, so each spectrogram has max resolution. Add a colorbar to the right of each spectrogram.
                    divider = make_axes_locatable(ax[c*2+1])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(sgram_handle, cax=cax);
                    # also add space next to Trace
                    divider2 = make_axes_locatable(ax[c*2])
                    hide_ax = divider2.append_axes("right", size="5%", pad=0.05, visible=False)
                      

            

            
            # increment c and go to next trace (if any left)
            c += 1   
     
        ax[N*2-1].set_xlabel('Time [s]')
    
        if title:
            ax[0].set_title(title)
        
        # set the xlimits for each panel from the min and max time values we kept updating
        min_t = min([tr.stats.starttime for tr in st]) 
        max_t = max([tr.stats.endtime for tr in st]) - min_t        
        for c in range(N): 
            ax[c*2].set_xlim(0, max_t)
            ax[c*2].grid(axis='x', linestyle = ':', linewidth=0.5)
            ax[c*2+1].set_xlim(0, max_t)
            ax[c*2+1].grid(True, linestyle = ':', linewidth=0.5)    

        # change all font sizes
        plt.rcParams.update({'font.size': 8});    
                
        # write plot to file
        if outfile:
            fig.savefig(outfile, dpi=100)

        return fig, ax 
    
    def get_time_range(self):
        min_t = min([tr.stats.starttime for tr in self.stream]) 
        max_t = max([tr.stats.endtime for tr in self.stream]) 
        return min_t, max_t      

    def get_S_range(self, fmin=0.5, fmax=20.0):
        Smin = 999999.9
        Smax = -999999.9
        for tr in self.stream:
            F = tr.stats.spectrogramdata['F']
            S = tr.stats.spectrogramdata['S']
                            
            # filter S between fmin and fmax and then update Smin and Smax
            f_indexes = np.intersect1d(np.where(F>=fmin), np.where(F<fmax))
            try:
                S_filtered = S[f_indexes, :]
            except:
                print('S_range failed. F is ',F.shape, ' S is ',S.shape)
            else:
                Smin = np.nanmin([np.nanmin(S_filtered), Smin])
                Smax = np.nanmax([np.nanmax(S_filtered), Smax])
        #print('S ranges from %e to %e' % (Smin, Smax))
        return Smin, Smax    
        
    def calculateSubplotPositions(numchannels, channelNum, frameLeft=0.12, frameBottom=0.08, \
                              totalWidth = 0.8, totalHeight = 0.88, fractionalSpectrogramHeight = 0.8):
        """ Copied from the MATLAB/GISMO function """
    
        channelHeight = totalHeight/numchannels;
        spectrogramHeight = fractionalSpectrogramHeight * channelHeight;
        traceHeight = channelHeight - spectrogramHeight; 
        spectrogramBottom = frameBottom + (numchannels - channelNum - 1) * channelHeight; 
        traceBottom = spectrogramBottom + spectrogramHeight;
        spectrogramPosition = [frameLeft, spectrogramBottom, totalWidth, spectrogramHeight];
        tracePosition = [frameLeft, traceBottom, totalWidth, traceHeight];
    
        return spectrogramPosition, tracePosition  

    def compute_amplitude_spectrum(self, compute_bandwidth=False):
        for c, tr in enumerate(self.stream):
            tr.stats['spectrum'] = dict()
            A = np.nanmean(tr.stats.spectrogramdata.S, axis=1)
            F = tr.stats.spectrogramdata.F
            max_i = np.argmax(A)
            tr.stats.spectrum['A'] = A
            tr.stats.spectrum['F'] = F
            tr.stats.spectrum['peakF'] = F[max_i]
            tr.stats.spectrum['peakA'] = max(A)
            tr.stats.spectrum['medianF'] = np.sum(np.dot(A, F))/np.sum(A)
            if compute_bandwidth:
                try:
                    Athresh = max(A)*0.707          
                    fn = interpolate.interp1d(F,  A)
                    xnew = np.arange(0, max(F), 0.1)
                    ynew = fn(xnew)
                    ind = np.argwhere(ynew>Athresh)
                    tr.stats.spectrum['bw_min'] = xnew[ind[0]]
                    tr.stats.spectrum['bw_max'] = xnew[ind[-1]]
                    print(f'{tr.id} bandwidth: {tr.stats.spectrum["bw_min"]}-{tr.stats.spectrum["bw_max"]}')
                except:
                    print('Could not compute bandwidth')
            for key in tr.stats.spectrum:
                v = tr.stats.spectrum[key]
                if np.ndim(v)==1:
                    if np.size(v)==1:
                        tr.stats.spectrum[key] = v[0]

                       
    
    def plot_amplitude_spectrum(self, normalize=False, title=None, outfile=None, logx=False, logy=False):
        '''
        fig, ax = plt.subplots(len(self.stream), 1);
        for c, tr in enumerate(self.stream):
            if not 'spectrum' in tr.stats:
                continue
            A = tr.stats.spectrum.A   
            F = tr.stats.spectrum.F
            ax[c].semilogy(F,  A);
            ax[c].set_ylabel('Spectral Amplitude:\n%s/Hz' % tr.stats.units)
            ax[c].set_xlabel('SSAM bin (Hz?)')
            ax[c].set_title(tr.id)       
        A = np.array(tr.stats.ssam.A)
        f = tr.stats.ssam.f
        '''
        
        fig, ax = plt.subplots(1, 1);
        

        for c, tr in enumerate(self.stream):
            if not 'spectrum' in tr.stats:
                continue
            A = tr.stats.spectrum.A   
            F = tr.stats.spectrum.F
            if normalize:
            	A = A/max(A)
            if logy and logx:
                ax.loglog(F, A, label=tr.id)
            elif logy:
                ax.semilogy(F,  A, label=tr.id)
            elif logx:
                ax.semilogx(F, A, label=tr.id)
            else:
                ax.plot(F, A, label=tr.id)
        if normalize:
        	ax.set_ylabel('Normalized Spectral Amplitude')
        else:
        	ax.set_ylabel('Spectral Amplitude')
        ax.set_xlabel('Hz')
        if title:
        	ax.set_title(title)
        ax.grid()
        ax.legend()    
        if outfile:
            fig.savefig(outfile)  
        else:
            fig.show() 
       

    def compute_band_ratio(self, freqlims=[0.8, 4.0, 18.0], plot=True):
        for tr in self.stream:
            S = tr.stats.spectrogramdata.S
            F = tr.stats.spectrogramdata.F
            T = tr.stats.spectrogramdata.T
            f_indexes_low = np.intersect1d(np.where(F>freqlims[0]), np.where(F<freqlims[1]))
            f_indexes_high = np.intersect1d(np.where(F>freqlims[1]), np.where(F<freqlims[2]))       
            S_low = S[f_indexes_low]        
            S_high = S[f_indexes_high] 
            sam_high = sum(S_high)
            sam_low = sum(S_low)
            fratio = np.log2(sum(S_high)/sum(S_low))
            sam = sum(S)
            if plot:
                fh, axh1=plt.subplots(1,1)
                fh.set_size_inches(10, 2)
                axh1.plot(sam, 'C0', alpha=0.3)
                sam60 = pd.Series(sam).rolling(60).mean()
                axh1.plot(sam60, 'C0')
                axh1.tick_params(axis='y', color='C0', labelcolor='C0')
                axh2 = axh1.twinx()
                fratio60 = pd.Series(np.log2(sam_high/sam_low)).rolling(60).mean()
                axh2.plot(fratio, 'C1', alpha=0.3)
                axh2.plot(fratio60, 'C1')
                axh2.grid()
                axh2.set_ylabel('Frequency Ratio\nlog2(VT band/LP band)', color='C1')
                axh2.tick_params(axis='y', color='C1', labelcolor='C1')
                axh2.spines['right'].set_color('C1')
                axh2.spines['left'].set_color('C0')
                #axh1.xaxis.set_major_locator(md.MinuteLocator(byminute = [0]))
                #axh1.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
                #plt.setp(axh1.xaxis.get_majorticklabels(), rotation = 90)
                #for tick in axh1.get_xticklabels():
                #    tick.set_rotation(90)
                plt.show()


def compute_metrics_TFS(T, F, S, freqlims=[0.8, 4.0, 18.0]):
    # SAM within this band
    F = np.array(F)
    stime = T[0]
    T = np.array([t - stime for t in T])
    f_indexes_all = np.intersect1d(np.where(F>freqlims[0]), np.where(F<freqlims[2]))
    S_all = S[f_indexes_all]
    sam = sum(S_all)

    # band ratio metrics
    f_indexes_low = np.intersect1d(np.where(F>freqlims[0]), np.where(F<freqlims[1]))
    f_indexes_high = np.intersect1d(np.where(F>freqlims[1]), np.where(F<freqlims[2]))       
    S_low = S[f_indexes_low]        
    S_high = S[f_indexes_high] 
    sam_high = np.sum(S_high)
    sam_low = np.sum(S_low)
    fratio = np.log2(sum(S_high)/sum(S_low))

    # mean frequency
    #meanf = sum(np.multiply(S, F))/sum(S)
    meanf = np.sum(np.dot(S, F))/np.sum(S)

    # peak frequency
    peak_index = np.argmax(S_all)
    peakf = F[peak_index]

    # bandwidth
    peakS = np.nanmax(S_all) 
    #peakA = peakS / (F[1]-F[0]) # peak amplitude but represented as if over a 1-Hz range. better would be to interpolate onto 1 Hz grid.
    ind_true_false = S_all>peakS/np.sqrt(2)
    ind = np.argwhere(ind_true_false)
    bw_min = F(ind[0]) # crude algorithm where any dips between main peak and an earlier peak could be ignored
    bw_max = F(ind[1])
    bw_sum = np.sum(np.arange(ind[0], ind[1]+1, 1))
    '''
    results = consecutive_true_values(ind_true_false)
    for result in results:
        if peak_index >= result[0] and peak_index <= result[0] + result[1]:
            bw_min = Fresult[0]
    bw_sum = 
    '''
    fmetrics = pd.DataFrame([{'sam':sam, 'VT':sam_high, 'LP':sam_low, 'fratio':fratio, 'meanf':meanf, 'peakf':peakf, 'bw_sum':bw_sum, 'bw_min':bw_min, 'bw_max':bw_max}])
    return fmetrics
          
def _nearest_pow_2(x):
    """
    Return the next power of 2 greater than or equal to x.

    Parameters:
    - x: float or int

    Returns:
    - int: the nearest power of 2 >= x
    """
    return 2 ** int(np.ceil(np.log2(x)))

def compute_spectrogram(tr, per_lap=0.99, wlen=None, mult=8.0):
    """
        Computes spectrogram of the input data.
        Modified from obspy.imaging.spectrogram because we want the plotting part
        of that method in a different function to this.

        :type tr: Trace
        :param tr: Trace object to compute spectrogram for
        :type per_lap: float
        :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
        :type wlen: int or float
        :param wlen: Window length for fft in seconds. If this parameter is too
            small, the calculation will take forever. If None, it defaults to
            (samp_rate/100.0).
        :type mult: float
        :param mult: Pad zeros to length mult * wlen. This will make the
            spectrogram smoother.

    """

    Fs = float(tr.stats.sampling_rate)
    y = tr.data
    npts = tr.stats.npts
        
    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = Fs / 100.0
 
    # nfft needs to be an integer, otherwise a deprecation will be raised
    nfft = int(_nearest_pow_2(wlen * Fs))
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    y = y - y.mean()

    # Here we do not call plt.specgram as that always produces a plot.
    # matplotlib.mlab.specgram should be faster as it computes only the arrays
    S, F, T = mlab.specgram(y, Fs=Fs, NFFT=nfft, pad_to=mult, noverlap=nlap, mode='magnitude')
        
    return T, F, S

def amp2dB(X):
    return 20 * np.log10(X)

def dB2amp(X):
    return np.power(10.0, float(X)/20.0)




def plot_strongest_trace(st, detrend=True, clip_level=1.0, fmin=None, fmax=None, dbscale=True, log=False, cmap='inferno', secsPerFFT=1.0, outfile=None):
    if detrend:
        st.detrend('linear')
    peakA = np.array([max(abs(tr.data)) for tr in st])
    peakA[peakA>clip_level]=0.0 # outlier in wrong units?
    maxA = max(peakA)
    strongest_tr = Stream(traces=st[np.argmax(peakA)])
    spobj = icewebSpectrogram(strongest_tr, secsPerFFT=secsPerFFT)
    spobj.plot(log=log, dbscale=dbscale, cmap=cmap, fmin=fmin, fmax=fmax, clim=[maxA/200, maxA/3], outfile=outfile)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import dpss
from numpy.fft import rfft
from typing import Optional, Tuple, Union

def _pick_strongest_trace(stream):
    # choose the trace with the largest RMS (robust for different gains)
    best = None
    best_rms = -np.inf
    for tr in stream:
        d = np.asarray(tr.data, dtype=float)
        # robust detrend/demean is handled upstream if needed
        rms = np.sqrt(np.mean(d**2))
        if rms > best_rms:
            best_rms = rms
            best = tr
    return best

def _multitaper_spectrogram(x: np.ndarray, fs: float,
                            nperseg: int, noverlap: int,
                            NW: float = 3.0, K: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multitaper (DPSS) spectrogram: returns (freqs, times, Sxx_dB).
    """
    x = np.asarray(x, float)
    step = nperseg - noverlap
    if step <= 0 or len(x) < nperseg:
        raise ValueError("nperseg/noverlap incompatible with data length.")
    n_frames = 1 + (len(x) - nperseg) // step
    if K is None:
        K = max(1, int(2 * NW) - 1)  # e.g., NW=3 -> K=5

    tapers, eigs = dpss(nperseg, NW=NW, Kmax=K, return_ratios=True)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    Sxx = np.zeros((len(freqs), n_frames), dtype=float)

    # simple equal weighting across tapers (adaptive weighting is a further upgrade)
    for i in range(n_frames):
        seg = x[i * step : i * step + nperseg]
        acc = 0.0
        for k in range(K):
            Xk = rfft(seg * tapers[k])
            acc += (np.abs(Xk) ** 2)
        Sxx[:, i] = acc / K

    Sxx = np.maximum(Sxx, np.finfo(float).eps)
    Sxx_db = 10.0 * np.log10(Sxx)
    t = (np.arange(n_frames) * step + 0.5 * nperseg) / fs
    return freqs, t, Sxx_db

def plot_strongest_trace_multitaper(
    stream_or_data,
    fs=None,
    *,
    fmin=0.1, fmax=20.0,
    nperseg_sec=0.64,        # SHORTER window for 75–100 Hz data
    overlap=0.95,            # heavier overlap for better time tracking
    NW=2.0,                  # narrower mainlobe, less smoothing
    K=None,                  # default -> int(2*NW)-1  (e.g., 3 tapers for NW=2)
    weighting="eigen",       # "equal" | "eigen" | "adaptive"
    pad_factor=4,            # zero-padding (frequency interpolation)
    detrend=True,
    db_clip_pct=(5, 98),
    cmap="inferno",
    outfile=None,
    title=None,
):
    """
    Multitaper spectrogram tuned for short seismic events (Fs ~ 75–100 Hz).
    - Shorter window + 95% overlap = better time detail
    - pad_factor gives a denser freq grid (visual sharpness)
    - weighting: eigen/adaptive reduce bias vs equal averaging
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal.windows import dpss
    from numpy.fft import rfft, rfftfreq

    # --- accept Stream or raw array ---
    try:
        from obspy import Stream
    except Exception:
        Stream = None

    if Stream is not None and isinstance(stream_or_data, Stream):
        tr = max(stream_or_data, key=lambda tr: np.sqrt(np.mean(np.asarray(tr.data, float)**2)))
        x = np.asarray(tr.data, float)
        fs0 = float(tr.stats.sampling_rate)
        ch_label = getattr(tr.stats, "channel", "VEL")
    else:
        if fs is None:
            raise ValueError("Provide fs when passing a numpy array.")
        x = np.asarray(stream_or_data, float)
        fs0 = float(fs)
        ch_label = "VEL"

    if detrend:
        x = x - np.mean(x)

    nperseg = int(round(nperseg_sec * fs0))
    if nperseg < 32:
        nperseg = 32
    noverlap = int(round(overlap * nperseg))
    step = nperseg - noverlap
    if step <= 0 or len(x) < nperseg:
        raise ValueError("nperseg/noverlap incompatible with data length.")

    if K is None:
        K = max(1, int(2*NW) - 1)  # NW=2 -> K=3 tapers (good for detail)
    tapers, eigs = dpss(nperseg, NW=NW, Kmax=K, return_ratios=True)

    # zero-padding for a denser frequency grid (visual sharpness)
    # (does not change true resolution, just interpolates bins)
    nfft = 1
    while nfft < pad_factor * nperseg:
        nfft <<= 1

    freqs_full = rfftfreq(nfft, 1.0/fs0)
    t_frames = 1 + (len(x) - nperseg)//step
    S = np.zeros((freqs_full.size, t_frames), float)

    # Optional: simple prewhiten to keep lows from dominating (comment out if not wanted)
    # x = np.diff(np.r_[x[0], x])  # 1st difference

    # Thomson-style *eigen* weighting by default.
    w_eigen = eigs / np.sum(eigs)

    # rough noise level for adaptive weighting
    def _adaptive_weights(Pk):
        # Pk: (K, F) power spectra for this frame
        # simple 1–2 iter scheme; robust and fast
        S_est = np.average(Pk, axis=0, weights=w_eigen)
        N0 = np.median(S_est)  # crude white-noise floor
        for _ in range(2):
            denom = (eigs[:, None] * S_est[None, :]) + (1.0 - eigs)[:, None] * N0
            wk = (eigs[:, None] * S_est[None, :]) / np.maximum(denom, 1e-30)
            wk /= np.sum(wk, axis=0, keepdims=True)
            S_est = np.sum(wk * Pk, axis=0)
        return S_est

    for i in range(t_frames):
        seg = x[i*step:i*step+nperseg]
        Xk = np.empty((K, freqs_full.size), complex)
        for k in range(K):
            Xk[k] = rfft(seg * tapers[k], n=nfft)
        Pk = (np.abs(Xk)**2)

        if weighting == "equal":
            S[:, i] = np.mean(Pk, axis=0)
        elif weighting == "eigen":
            S[:, i] = np.average(Pk, axis=0, weights=w_eigen)
        else:  # "adaptive"
            S[:, i] = _adaptive_weights(Pk)

    S = np.maximum(S, np.finfo(float).eps)
    Sdb = 10*np.log10(S)

    # trim band and compute time vector
    fmask = (freqs_full >= max(0.0, fmin)) & (freqs_full <= fmax)
    freqs = freqs_full[fmask]
    Sdb = Sdb[fmask, :]
    times = (np.arange(t_frames)*step + 0.5*nperseg)/fs0

    vmin, vmax = np.percentile(Sdb, db_clip_pct)
    tt = np.arange(len(x))/fs0

    #fig = plt.figure(figsize=(7, 9), dpi=240)
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(tt, x, lw=0.8)
    ax1.set_xlim(tt[0], tt[-1])
    ax1.set_ylabel(ch_label)

    ax2 = plt.subplot(212, sharex=ax1)
    im = ax2.imshow(
        Sdb, origin="lower", aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        vmin=vmin, vmax=vmax, cmap=cmap,
        interpolation="nearest",  # keeps fine detail
    )
    ax2.set_ylabel("Hz"); ax2.set_xlabel("Time [s]")
    cb = plt.colorbar(im, ax=ax2, pad=0.01)
    cb.set_label("dB (multitaper)")
    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches="tight"); plt.close(fig)
    else:
        plt.show()

def plot_strongest_trace_reassigned(
    stream_or_data,
    fs=None,
    *,
    fmin=0.5, fmax=20.0,
    win_sec=0.64,          # ~0.5–0.8 s for 75–100 Hz data
    hop_frac=0.05,         # 5% hop (95% overlap) -> good time detail
    pad_factor=4,          # frequency interpolation (visual)
    power_db_clip=(5, 99), # percentile clip for color scale
    cmap="inferno",
    outfile=None,
    title=None,
):
    """
    Reassigned (instantaneous-frequency) spectrogram for short seismic events.
    Uses librosa.reassigned_spectrogram if available; otherwise a dense STFT.
    Accepts an ObsPy Stream (picks strongest Z trace) or a 1D numpy array (+fs).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- accept Stream or raw array ---
    try:
        from obspy import Stream
    except Exception:
        Stream = None

    if Stream is not None and isinstance(stream_or_data, Stream):
        # strongest vertical by RMS
        trZ = [tr for tr in stream_or_data if getattr(tr.stats, "channel", "").endswith("Z")]
        trlist = trZ if trZ else list(stream_or_data)
        tr = max(trlist, key=lambda tr: np.sqrt(np.mean(np.asarray(tr.data, float)**2)))
        x = np.asarray(tr.data, float)
        sr = float(tr.stats.sampling_rate)
        ch_label = getattr(tr.stats, "channel", "BHZ")
    else:
        if fs is None:
            raise ValueError("Provide fs when passing a numpy array.")
        x = np.asarray(stream_or_data, float)
        sr = float(fs)
        ch_label = "VEL"

    # detrend/demean light touch
    x = x - np.mean(x)

    # window / hop / fft
    n_win = int(round(win_sec * sr))
    n_win = max(64, n_win | 1)      # odd length ≥64
    hop = max(1, int(round(hop_frac * n_win)))

    # choose n_fft with padding (next pow2)
    n_fft = 1
    target = pad_factor * n_win
    while n_fft < target:
        n_fft <<= 1

    # --- try librosa reassignment ---
    S_db = None
    try:
        import librosa
        import librosa.display  # noqa: F401

        # Hann window; center=False to align with ObsPy-style indexing
        S, freqs, times = librosa.reassigned_spectrogram(
            y=x,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_win,
            center=False,
            power=2.0,  # power spectrogram
        )
        # S is already power; convert to dB
        S_db = librosa.power_to_db(S, ref=np.max)
        f = freqs
        t = times

    except Exception:
        # --- fallback: dense STFT (no reassignment) ---
        from numpy.fft import rfft, rfftfreq
        w = np.hanning(n_win)
        step = hop
        n_frames = 1 + (len(x) - n_win) // step
        f = rfftfreq(n_fft, 1.0 / sr)
        S = np.empty((f.size, n_frames), float)
        for i in range(n_frames):
            seg = x[i*step:i*step+n_win] * w
            X = rfft(seg, n=n_fft)
            S[:, i] = (np.abs(X)**2)
        S = np.maximum(S, np.finfo(float).eps)
        S_db = 10*np.log10(S) - np.max(10*np.log10(S))
        t = (np.arange(n_frames)*step + 0.5*n_win)/sr

    # band-limit
    mask = (f >= max(0.0, fmin)) & (f <= fmax)
    f = f[mask]
    S_db = S_db[mask, :]

    # color limits by percentiles for contrast
    vmin, vmax = np.percentile(S_db, power_db_clip)

    # --- plot waveform + spectrogram ---
    tt = np.arange(len(x))/sr
    #fig = plt.figure(figsize=(7.5, 9), dpi=240)
    fig = plt.figure()

    ax1 = plt.subplot(211)
    ax1.plot(tt, x, lw=0.8)
    ax1.set_xlim(tt[0], tt[-1])
    ax1.set_ylabel(ch_label)

    ax2 = plt.subplot(212, sharex=ax1)
    im = ax2.imshow(
        S_db, origin="lower", aspect="auto",
        extent=[t[0], t[-1], f[0], f[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax2.set_ylabel("Hz"); ax2.set_xlabel("Time [s]")
    cb = plt.colorbar(im, ax=ax2, pad=0.01)
    cb.set_label("dB (reassigned)" if 'librosa' in globals() else "dB")

    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, bbox_inches="tight"); plt.close(fig)
    else:
        plt.show()



class dailySpectrogram:

    def __init__(self):
        """
        """
        
        self.seed_ids = []
        self.dataframes = []
        self.filenames = []
    
    def load(self, infiles, nrows=None): # change from Trace to Stream
        if not isinstance(infiles, list):
            infiles = [infiles]
        for infile in infiles:
            if os.path.isfile(infile): # this should be a pickle file with raw S,F,T values
                if infile[-4:]=='.csv':
                    df = pd.read_csv(infile, nrows=nrows)
                else:
                    df = pd.read_pickle(infile)
                    df = df.iloc[0:nrows, :]
                df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
                
                self.dataframes.append(df)
                self.filenames.append(infile)
                parts1 = os.path.basename(infile).split('_')
                parts2 = parts1[1].split('.')
                self.seed_ids.append('.'.join(parts2[0:4]))
            
    def to_icewebSpectrogram(self, maxpixels=1000):
        st = Stream()
        for i, df in enumerate(self.dataframes):

            # make a stream object from the S matrix
            
            tr = Trace()
            T, F, S, utcdt = dataframe2numpyarrays(df)
            delta = T[1] - T[0]
            dt = pd.Series([pd.Timestamp(t.datetime) for t in utcdt])
            print(dt)
            if T.size > maxpixels:
                crunch_factor = int(np.ceil(T.size/maxpixels))
                delta = np.ceil(crunch_factor * delta)
                print(delta)
                #df.resample(f"{delta:.0f}s", on=dt).mean()
                #df.set_index(dt)
                #df.resample("10s", on=dt, origin='start').asfreq() #.mean()
                df = df.iloc[::crunch_factor]
                T, F, S, utcdt = dataframe2numpyarrays(df)
            # T is a numpy array of float64
            # F is a numpy array of float64
            # S is a 2D numpy array
            # utcdt is a list of UTCDateTime
            tr = TFS2trace(T, F, S, utcdt[0], seed_id='.'.join(parts2[0:4]) )
            st.append(tr)
        iwsobj = icewebSpectrogram()
        iwsobj.stream = st
        iwsobj.precomputed = True
        return iwsobj
            
            
    def compute(self, st, secsPerFFT=None, sampling_interval=60, secsPerSpectrogram=600, outdir='.'):

        #metrics_all_time = None
        for tr in st:
            stime = tr.stats.starttime
            etime = tr.stats.endtime
            YMD = stime.strftime('%Y%m%d')
            outfile = os.path.join(outdir, f'FTS_{tr.id}.{YMD}')
            for spectime in np.arange(stime, etime, secsPerSpectrogram):
                tr2 = tr.copy().trim(starttime=spectime, endtime=spectime+secsPerSpectrogram)
                print(tr2)
                print('- Computing spectrograms')
                if secsPerFFT is None:
                    secsPerFFT = np.ceil((tr2.stats.delta * tr2.stats.npts)/100) 
                    secsPerFFT = 2.56
                    per_lap = 0.56/secsPerFFT
                print(f"secsPerFFT={secsPerFFT}")
                [T, F, S] = compute_spectrogram(tr2, wlen=secsPerFFT, per_lap=per_lap, mult=2.0)
                #print(T.shape, F.shape, S.shape)
                # turn into a dataframe
                df = pd.DataFrame(np.transpose(S), columns=[str(thisf) for thisf in F])
                df.insert(0, 'time', pd.Series([(spectime + t) for t in T]) )
                #print(df)
                
                if os.path.isfile(outfile):
                    df_day = pd.read_pickle(outfile)
                    for col in df_day.columns:
                        if 'Unnamed' in col:
                            df_day.drop(labels=col, inplace=True)
                    df_day = pd.concat([df_day, df])
                else:
                    df_day = df
                df_day.to_pickle(outfile, index=False)
            self.dataframes.append(df_day)
            self.filenames.append(outfile)
            self.seed_ids.append(tr.id)
            df_day.to_csv(outfile + '.csv', index=False)

    def compute_metrics(self, new_sampling_interval=None):
        for df in self.dataframes:
            if isinstance(df, pd.DataFrame):
                # potentially downsample the df here
                print('- Computing metrics')
                F = [float(thisf) for thisf in  df.columns[1:]]
                fmetrics = compute_metrics_TFS(T=df['time'], F=F, S=df.iloc[0:-1][0:-2])
                print(fmetrics)


    '''
            metrics_stream = iwspobj.compute_metrics(sampling_interval=sampling_interval)
            # concatenate metrics
            print('- Concatenating metrics')
            if not metrics_all_time:
                metrics_all_time = metrics_stream
            for seed_id in metrics_stream:
                metrics_trace = metrics_stream[seed_id]
                metrics_all_time[seed_id] = pd.concat(metrics_all_time[seed_id], metrics_trace)
    '''  

               