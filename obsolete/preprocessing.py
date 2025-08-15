import numpy as np
import os
from obspy import read, read_inventory, Stream, Trace, UTCDateTime
from scipy.signal import welch
from obspy.signal.quality_control import MSEEDMetadata 
#from flovopy.core.trace_utils import fix_trace_id
# Leverage your modules
from flovopy.core.gaputils import (
    smart_fill as gap_smart_fill,
    classify_gaps,
    fill_stream_gaps,
    fill_gaps_with_constant,
    fill_gaps_with_linear_interpolation,
    fill_gaps_with_previous_value,
    fill_gaps_with_filtered_noise,
)
from flovopy.core.miniseed_io import unmask_gaps
"""

If you’re processing daily data, call it like:

    preprocess_stream(stream, inv=inv, max_dropout=None, outputType='VEL', ...)

If you’re processing event-sized data:

    preprocess_stream(stream, inv=inv, max_dropout=0.1, ...)

Optional: In _interpolate_small_gaps(), treat 0 as a null value only when you want to.

If you’re padding 0.0 into gaps in daily data, it’s better to exclude 0 from null values (to avoid masking valid data). 
When calling detrend_trace(), do:

    detrend_trace(tr, null_values=[np.nan], ...)

instead of:

    detrend_trace(tr, null_values=[0, np.nan], ...)


This change will:
	•	Preserve event file behavior (aggressively reject bad traces with gaps)
	•	Allow daily SDS files to retain gappy traces and interpolate later if needed
	•	Let you process days with partial data coverage, e.g., 12 hours of usable data in a 24-hour trace


"""

#######################################################################
##                Trace  tools                                       ##
#######################################################################

def preprocess_trace(tr, bool_despike=True, bool_clean=True, inv=None, quality_threshold=-np.inf, taperFraction=0.05, \
                  filterType="bandpass", freq=[0.5, 30.0], corners=6, zerophase=False, outputType='VEL', \
                    miniseed_qc=True, verbose=False, max_dropout=None, units='Counts', bool_detrend=True, min_sampling_rate=20.0):
    """
    Preprocesses a seismic trace by applying quality control, filtering, and instrument response correction.

    This function performs the following operations:
    - Quality control checks, including dropout detection and data gaps.
    - Optional despiking to remove single-sample anomalies.
    - Detrending, tapering, and bandpass filtering.
    - Instrument response removal (if an ObsPy inventory is provided).
    - Scaling data to physical units using the calibration factor (`calib`).
    - Tracks processing steps in `tr.stats.history`.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to process.
    bool_despike : bool, optional
        Whether to remove single-sample spikes from the trace (default: True).
    bool_clean : bool, optional
        Whether to apply detrending, tapering, and filtering (default: True).
    inv : obspy.Inventory, optional
        Instrument response metadata for deconvolution (default: None).
    quality_threshold : float, optional
        Minimum quality factor required to keep the trace (default: -Inf).
    taperFraction : float, optional
        Fraction of the trace length to use for tapering (default: 0.05).
    filterType : str, optional
        Type of filter to apply. Options: "bandpass", "lowpass", "highpass" (default: "bandpass").
    freq : list of float, optional
        Frequency range for filtering: [freq_min, freq_max] (default: [0.5, 30.0] Hz).
    corners : int, optional
        Number of filter corners (default: 6).
    zerophase : bool, optional
        Whether to apply a zero-phase filter (default: False).
    outputType : str, optional
        Type of output after instrument response removal. Options: "VEL" (velocity), "DISP" (displacement), "ACC" (acceleration), "DEF" (default) (default: "VEL").
    miniseed_qc : bool, optional
        Whether to perform MiniSEED quality control checks (default: True).
    verbose : bool, optional
        If True, prints processing steps (default: False).
    max_dropout : float, optional
        Maximum allowable data dropout percentage before rejection (default: 0.0).
    units : str, optional
        Unit of the trace before processing. Defaults to "Counts".

    Returns:
    -------
    bool
        Returns `True` if the trace was successfully processed, `False` if rejected due to poor quality or errors.
    
    Notes:
    ------
    - If `inv` is provided, `remove_response()` is used to convert the waveform to physical units.
    - If `inv` is not provided but `tr.stats.calib` is set, the trace is manually scaled.
    - If the trace fails quality checks (e.g., excessive gaps), it is rejected.
    - `tr.stats.history` keeps track of applied processing steps.

    Example:
    --------
    ```python
    from obspy import read
    from obspy.clients.fdsn import Client

    # Read a seismic trace
    tr = read("example.mseed")[0]

    # Load an inventory for response correction
    client = Client("IRIS")
    inv = client.get_stations(network="IU", station="ANMO", level="response")

    # Process the trace
    success = preprocess_trace(tr, inv=inv, verbose=True)

    if success:
        tr.plot()
    else:
        print("Trace did not meet quality criteria and was rejected.")
    ```
    """
    if verbose:
        print(f'Processing {tr}')
    if not 'history' in tr.stats:
        tr.stats['history'] = list()    

    if not 'units' in tr.stats:
        tr.stats['units'] = units  

    # a good trace has quality factor 3, one with 0s and -1s has 1, bad trace has 0
    if not 'quality_factor' in tr.stats:
        tr.stats['quality_factor'] = 1.0
    #is_bad_trace = False    

    # ignore traces with weirdly low sampling rates
    if tr.stats.sampling_rate < min_sampling_rate and tr.stats.channel[0] != 'L':
        add_to_trace_history(tr, 'Sampling rate too low')
        #is_bad_trace = True   

    # ignore traces with few samples
    if tr.stats.npts < tr.stats.sampling_rate:
        add_to_trace_history(tr, 'Not enough samples')
        #is_bad_trace = True
    
    # Step 1: Detect Empty or Nearly Empty Traces
    if _is_empty_trace(tr):
        add_to_trace_history(tr, 'Trace is blank')
        return 0.0  # Discard trace

    # Step 2: Compute Raw Data Quality Metrics
    tr.stats['metrics'] = {}
    tr.stats.metrics["twin"] = tr.stats.npts * tr.stats.delta  # Compute trace duration
    if miniseed_qc:
        _can_write_to_miniseed_and_read_back(tr, return_metrics=True)
    else:
        _compute_trace_metrics(tr)

    # Adjust quality factor based on gaps, overlaps, and availability
    if 'num_gaps' in tr.stats.metrics:
        tr.stats.quality_factor -= tr.stats.metrics['num_gaps']
    if 'num_overlaps'  in tr.stats.metrics:
        tr.stats.quality_factor -= tr.stats.metrics['num_overlaps']
    if 'percent_availability' in tr.stats.metrics:
        tr.stats.quality_factor *= tr.stats.metrics['percent_availability'] / 100.0

    # Step 3: Detect Bit-Level Noise
    num_unique_values = np.unique(tr.data).size
    if num_unique_values > 10:
        tr.stats.quality_factor += np.log10(num_unique_values)
    else:
        add_to_trace_history(tr, 'Bit-level noise suspected')
        return False  

    # Step 4: Detect and Handle Dropouts
    tr.stats['gap_report'] = []
    if max_dropout is not None:
        if not _detect_and_handle_dropouts(tr, max_dropout=max_dropout, verbose=verbose):
            return False
        if not _detect_and_handle_gaps(tr, gap_threshold=int(max_dropout * tr.stats.sampling_rate), verbose=verbose):
            return False


    # Step 5: Detect and Correct Clipping, Spikes, and Step Functions
    if verbose:
        print('- detecting and correcting artifacts')

    _detect_and_correct_artifacts(tr, amp_limit=1e10, count_thresh=10, spike_thresh=50.0, fill_method="interpolate")

    # Step 6: Adjust Quality Factor Based on Artifacts
    artifacts = tr.stats.get('artifacts', {})
    if artifacts.get("upper_clipped", False):
        tr.stats.quality_factor /= 2.0
    if artifacts.get("lower_clipped", False):
        tr.stats.quality_factor /= 2.0
    if artifacts.get("spike_count", 0) == 0:
        tr.stats.quality_factor += 1.0
    else:
        #add_to_trace_history(tr, f'{artifacts["spike_count"]} outliers found')
        #add_to_trace_history(tr, f'{spike_count} outliers (spikes) found')
        tr.stats['outlier_indices'] = artifacts.get("spike_indices", [])

    # Step 7: Final Quality Check
    if tr.stats.quality_factor < quality_threshold:
        return False

    if verbose:
        print(f'- artifacts processed. qf={tr.stats.quality_factor}')

    if bool_detrend or bool_clean:
        tr = detrend_trace(tr, gap_threshold=3, verbose=True)      

    """ CLEAN (PAD, TAPER, FILTER, CORRECT, UNPAD) TRACE """
    # Step 8: Apply Cleaning Steps (Pad, Taper, Filter, Response Removal)
    tr.stats.metrics['maxamp_raw'] = np.max(np.abs(tr.data))
    if bool_clean:
        _clean_trace(tr, taperFraction, filterType, freq, corners, zerophase, inv, outputType, verbose)

    if verbose:
        print(f'- processing complete. qf={tr.stats.quality_factor}')
    
    return True

# ---- Helper Functions ----


def clean_stream(stream_in, taperFraction=0.05,
                 filterType='bandpass', freq=(1.0, 10.0), corners=4,
                 zerophase=True, inv=None, outputType='VEL', verbose=False):
    """
    Apply padding, tapering, filtering, and instrument response correction
    to each trace in a Stream using `_clean_trace`.

    Parameters:
    -----------
    stream_in : obspy.Stream
        Input stream to clean.

    taperFraction : float
        Fraction of trace length to taper on both ends (e.g., 0.05 = 5%).

    filterType : str
        Type of filter to apply (e.g., 'bandpass', 'highpass').

    freq : float or tuple
        Frequency or frequency range (tuple for bandpass).

    corners : int
        Number of filter corners.

    zerophase : bool
        Whether to use zero-phase filtering.

    inv : obspy.Inventory or None
        Inventory for instrument correction.

    outputType : str
        One of 'DISP', 'VEL', 'ACC', or 'DEF'.

    verbose : bool
        Print detailed processing messages.

    Returns:
    --------
    cleaned_stream : obspy.Stream
        New stream with cleaned traces (failed traces are skipped).
    """
    cleaned = Stream()

    for tr in stream_in.copy():
        if _clean_trace(tr, taperFraction, filterType, freq, corners, zerophase,
                        inv, outputType, verbose):
            cleaned.append(tr)
        elif verbose:
            print(f"[SKIP] Trace {tr.id} was not added to cleaned stream.")

    if verbose:
        print(f"[DONE] Cleaned stream contains {len(cleaned)} traces.")

    return cleaned

def _clean_trace(tr, taperFraction, filterType, freq, corners, zerophase, inv, outputType, verbose):
    """
    Clean a trace by padding, tapering, filtering, and optionally removing the instrument response.

    Parameters:
    -----------
    tr : obspy.Trace
        The input trace to process.

    taperFraction : float
        Fraction of the trace to taper on each end.

    filterType : str
        ObsPy filter type (e.g., 'bandpass', 'highpass').

    freq : float or tuple of float
        Corner frequencies. Tuple for bandpass, float for others.

    corners : int
        Number of filter corners.

    zerophase : bool
        Whether to apply zero-phase filtering.

    inv : obspy.Inventory or None
        Inventory used for instrument correction (if available).

    outputType : str
        One of 'DISP', 'VEL', 'ACC', or 'DEF'.

    verbose : bool
        Whether to print debug messages.

    Returns:
    --------
    success : bool
        True if cleaning succeeded; False if skipped or failed.
    """
    if tr.stats.npts == 0:
        if verbose:
            print(f"[SKIP] Trace {tr.id} has zero length.")
        return False

    if verbose:
        print(f'- cleaning trace {tr.id}')

    try:
        # Padding
        npts_pad = int(taperFraction * tr.stats.npts)
        npts_pad_seconds = max(npts_pad * tr.stats.delta, 1 / (freq[0] if isinstance(freq, (list, tuple)) else freq))
        _pad_trace(tr, npts_pad_seconds)
        max_fraction = npts_pad / tr.stats.npts

        # Taper
        if verbose:
            print('- tapering')
        tr.taper(max_percentage=max_fraction, type="hann")
        add_to_trace_history(tr, 'tapered')

        # Filtering and/or Response Removal
        if inv:
            # Estimate pre_filt based on desired band
            nyquist = 0.5 / tr.stats.delta
            if filterType == "bandpass":
                fmin, fmax = freq
            else:
                fmin = freq
                fmax = nyquist * 0.95

            pre_filt = (
                max(0.01, fmin * 0.5),
                fmin,
                min(fmax, nyquist * 0.95),
                min(fmax * 1.5, nyquist * 0.99)
            )

            # Instrument response correction
            success = _handle_instrument_response(tr, inv, pre_filt, outputType, verbose)
            if not success:
                return False
        else:
            # Only apply filter
            if verbose:
                print('- filtering (no instrument response)')
            if filterType == "bandpass":
                tr.filter(filterType, freqmin=freq[0], freqmax=freq[1], corners=corners, zerophase=zerophase)
            else:
                tr.filter(filterType, freq=freq[0], corners=corners, zerophase=zerophase)
            _update_trace_filter(tr, filterType, freq, zerophase)
            add_to_trace_history(tr, filterType)

        # Remove Padding
        _unpad_trace(tr)
        return True

    except Exception as e:
        print(f"[ERROR] Failed to clean trace {tr.id}: {e}")
        return False

def _handle_instrument_response(tr, inv, pre_filt, outputType, verbose):
    """
    Removes the instrument response using ObsPy's `remove_response()`.

    Note that water_level=60 means never allow the response amplitude to fall below 1/60th of its maximum during spectral division.
    """
    if inv:
        try:
            if verbose:
                print('- removing instrument response')
            if tr.stats.channel[1]=='D':
                outputType='DEF' # for pressure sensor
            tr.remove_response(inventory=inv, output=outputType, \
                pre_filt=pre_filt, water_level=60, zero_mean=True, \
                taper=False, taper_fraction=0.0, plot=False, fig=None)      
            add_to_trace_history(tr, 'calibrated')
            tr.stats.calib = 1.0
            tr.stats['calib_applied'] = _get_calib(tr, inv) # we have to do this, as the calib value is used to scale the data in plots
        except Exception as e:
            print(f'Error removing response: {e}')
            print('remove_response failed for %s' % tr.id)
            return False            
    elif tr.stats['units'] == 'Counts' and not 'calibrated' in tr.stats.history and tr.stats.calib!=1.0:
        tr.data = tr.data * tr.stats.calib
        tr.stats['calib_applied'] = tr.stats.calib
        tr.stats.calib = 1.0 # we have to do this, as the calib value is used to scale the data in plots
        add_to_trace_history(tr, 'calibrated') 
    
    if 'calibrated' in tr.stats.history:           
        if tr.stats.channel[1]=='H':
            if outputType=='VEL':
                tr.stats['units'] = 'm/s'
            elif outputType=='DISP':
                tr.stats['units'] = 'm'
        elif tr.stats.channel[1]=='N':
            tr.stats['units'] = 'm/s2'                
        elif tr.stats.channel[1]=='D':
            tr.stats['units'] = 'Pa' 
        tr.stats.metrics['maxamp_corrected'] = np.max(np.abs(tr.data))

def add_to_trace_history(tr, message):
    """
    Adds a message to the processing history of a seismic trace.

    This function maintains a record of operations applied to a trace 
    by appending descriptive messages to `tr.stats.history`.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to which the history message is added.
    message : str
        A string describing the processing step or modification.

    Returns:
    -------
    None
        The function updates `tr.stats.history` **in place**.

    Notes:
    ------
    - If `tr.stats.history` does not exist, it is initialized as an empty list.
    - Duplicate messages are not added to prevent redundancy.

    Example:
    --------
    ```python
    from obspy import Trace
    import numpy as np

    # Create a sample trace
    tr = Trace(data=np.random.randn(1000))

    # Add processing steps to the trace history
    add_to_trace_history(tr, "Detrended")
    add_to_trace_history(tr, "Applied bandpass filter (0.1 - 10 Hz)")
    add_to_trace_history(tr, "Detrended")  # Duplicate, will not be added

    # Print trace history
    print(tr.stats.history)
    # Output: ['Detrended', 'Applied bandpass filter (0.1 - 10 Hz)']
    ```
    """
    if 'history' not in tr.stats:
        tr.stats['history'] = []
    if message not in tr.stats.history:
        tr.stats.history.append(message)




def _compute_trace_metrics(trace):
    """
    Computes and stores the number of gaps, number of overlaps, 
    and percent availability of data in an ObsPy Trace object.
    
    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to analyze.
    
    Returns:
    -------
    None
        The function modifies `trace.stats.metrics` in place.
    
    Stores:
    -------
    - trace.stats.metrics['num_gaps'] : int
        Number of gaps (missing data).
    - trace.stats.metrics['num_overlaps'] : int
        Number of overlapping time windows.
    - trace.stats.metrics['percent_availability'] : float
        Percentage of available data relative to the total expected time span.
    
    """
    # Ensure metrics dictionary exists
    if not hasattr(trace.stats, "metrics"):
        trace.stats.metrics = {}

    # Detect gaps and overlaps
    gaps = trace.get_gaps()
    
    num_gaps = sum(1 for gap in gaps if gap[6] > 0)  # Count only gaps
    num_overlaps = sum(1 for gap in gaps if gap[6] < 0)  # Count only overlaps

    # Compute percent availability
    total_duration = trace.stats.npts * trace.stats.delta  # Expected duration
    actual_duration = total_duration  # Start with full duration

    for gap in gaps:
        if gap[6] > 0:  # If it's a gap, subtract its duration
            actual_duration -= gap[6]

    percent_availability = (actual_duration / total_duration) * 100 if total_duration > 0 else 0

    # Store metrics in trace.stats
    trace.stats.metrics["num_gaps"] = num_gaps
    trace.stats.metrics["num_overlaps"] = num_overlaps
    trace.stats.metrics["percent_availability"] = percent_availability  



    
def _get_islands(arr, mask):
    """
    Identifies contiguous sequences of the same value in an array.

    Parameters:
    ----------
    arr : np.ndarray
        The input data array.
    mask : np.ndarray
        A boolean mask indicating regions where the values do not change.

    Returns:
    -------
    list
        A list of contiguous segments (flat regions).
    """
    mask_ = np.concatenate(([False], mask, [False]))
    idx = np.flatnonzero(mask_[1:] != mask_[:-1])
    return [arr[idx[i]:idx[i+1] + 1] for i in range(0, len(idx), 2)]

def _FindMaxLength(lst):
    """
    Finds the longest contiguous sequence in a list.

    Parameters:
    ----------
    lst : list
        A list of contiguous sequences.

    Returns:
    -------
    tuple
        - The longest sequence.
        - Its length.
    """
    maxList = max(lst, key=len)
    maxLength = max(map(len, lst))      
    return maxList, maxLength     


def _detect_and_correct_artifacts(tr, amp_limit=1e10, count_thresh=10, spike_thresh=3.5, fill_method="median"):
    """
    Detects and optionally corrects **clipping, spikes, and step-function artifacts** in a seismic trace.

    This function:
    - Identifies **clipped samples** at both **upper and lower** limits.
    - Detects **spikes and step functions** using **Median Absolute Deviation (MAD)**.
    - Records the **number of clipped and spiked samples** in `tr.stats['artifacts']`.
    - Optionally **corrects both** clipping and spikes.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to analyze and correct.
    amp_limit : float, optional
        Absolute amplitude threshold for clipping detection (default: `1e10`).
    count_thresh : int, optional
        Minimum number of consecutive samples at a limit to classify as clipping (default: `10`).
    spike_thresh : float, optional
        MAD-based threshold for detecting spikes (default: `3.5`).
    fill_method : str, optional
        Method for replacing artifacts (`"median"`, `"interpolate"`, `"zero"`, or `"ignore"`).

    Returns:
    -------
    None
        The function modifies `tr` **in place**, detecting and correcting artifacts.

    Notes:
    ------
    - `"median"`: Replaces artifacts with the **median** of non-artifact data.
    - `"interpolate"`: Uses **linear interpolation** for smooth corrections.
    - `"zero"`: Replaces artifacts with `0`.
    - `"ignore"`: Only detects artifacts, without modifying the data.

    Example:
    --------
    ```python
    from obspy import read

    tr = read("example.mseed")[0]
    
    # Detect and correct artifacts
    _detect_and_correct_artifacts(tr, amp_limit=1e6, fill_method="interpolate")

    print(tr.stats['artifacts'])  # View detected issues
    ```
    """
    y = tr.data.copy()

    # --- Step 1: Detect Clipping ---
    upper_limit = min(np.nanmax(y), amp_limit)
    lower_limit = max(np.nanmin(y), -amp_limit)

    count_upper = np.sum(y >= upper_limit)
    count_lower = np.sum(y <= lower_limit)

    upper_clipped = count_upper >= count_thresh
    lower_clipped = count_lower >= count_thresh

    # --- Step 2: Detect Spikes Using MAD ---
    tr_detrended = tr.copy()
    tr_detrended.detrend()
    points = tr_detrended.data

    median = np.median(points)
    diff = np.abs(points - median)
    mad = np.median(diff)

    if mad == 0:
        modified_z_score = np.zeros_like(points)
    else:
        modified_z_score = 0.6745 * diff / mad

    spike_indices = np.where(modified_z_score > spike_thresh)[0]
    spike_count = spike_indices.size

    # --- Step 3: Identify Step Functions ---
    step_threshold = mad * 3  # Define a threshold for sudden level shifts
    diff_signal = np.abs(np.diff(y))
    step_indices = np.where(diff_signal > step_threshold)[0]
    step_count = step_indices.size

    # --- Step 4: Store Artifact Metadata ---
    tr.stats['artifacts'] = {
        "upper_clipped": upper_clipped,
        "lower_clipped": lower_clipped,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
        "count_upper": count_upper,
        "count_lower": count_lower,
        "spike_count": spike_count,
        "spike_indices": spike_indices.tolist(),
        "step_count": step_count,
        "step_indices": step_indices.tolist(),
    }

    # --- Step 5: Log Artifacts in Trace History ---
    msg = f"Trace {tr.id} detected artifacts:"
    if upper_clipped:
        msg += f" Upper limit {upper_limit} (count={count_upper})"
    if lower_clipped:
        msg += f" Lower limit {lower_limit} (count={count_lower})"
    if spike_count > 0:
        msg += f" {spike_count} spike(s) detected"
    if step_count > 0:
        msg += f" {step_count} step function(s) detected"
    
    add_to_trace_history(tr, msg)

    # --- Step 6: Correct Artifacts (if needed) ---
    if fill_method != "ignore":
        # Correct Clipping
        if upper_clipped or lower_clipped:
            fill_value = np.nanmedian(y[(y < upper_limit) & (y > lower_limit)])
            y[y >= upper_limit] = fill_value
            y[y <= lower_limit] = fill_value

        # Correct Spikes
        if spike_count > 0:
            if fill_method == "median":
                y[spike_indices] = np.nanmedian(y)
            elif fill_method == "interpolate":
                y[spike_indices] = np.interp(spike_indices, np.arange(len(y)), y)
            elif fill_method == "zero":
                y[spike_indices] = 0

        # Correct Step Functions
        if step_count > 0:
            for idx in step_indices:
                # Simple fix: Average values before and after step
                if 0 < idx < len(y) - 1:
                    y[idx] = (y[idx - 1] + y[idx + 1]) / 2

        # Update trace data
        tr.data = y

from flovopy.core.gaputils import smart_fill, piecewise_detrend
def detrend_trace(tr: Trace,
                  gap_threshold: int = 10,
                  null_values = (0.0, np.nan),
                  detrend_type: str = 'linear',
                  verbose: bool = False) -> Trace:
    """
    DEPRECATED: Use prepare_stream_for_analysis_v2() or gaputils.piecewise_detrend(_linear).
    Kept for backward compatibility; avoids split/merge and masks-aware.
    """
    # Interpolate small gaps (based on sample count) but keep long gaps masked
    sr = float(tr.stats.sampling_rate or 0.0)
    if sr > 0 and gap_threshold > 0 and isinstance(tr.data, np.ma.MaskedArray):
        # use smart_fill to interpolate short gaps; mark long gaps as NaN
        short_sec = gap_threshold / sr
        tmp = smart_fill(tr, short_thresh=short_sec, long_fill_value=np.nan)
        arr = np.asarray(tmp.data)
        masked = np.ma.masked_array(arr, mask=~np.isfinite(arr))
        work = tr.copy(); work.data = masked
    else:
        work = tr.copy()

    # Piecewise detrend (linear or simple)
    mode = 'linear' if detrend_type == 'linear' else 'simple'
    out = piecewise_detrend(work, mode=mode, use_null_values=True)
   
    # Mark history (to mirror old behavior)
    try:
        if 'history' not in out.stats: out.stats['history'] = []
        if 'detrended' not in out.stats.history:
            out.stats.history.append('detrended')
    except Exception:
        pass

    if verbose:
        print(f"[deprecated] detrend_trace: mode={mode}, gap_threshold={gap_threshold} samples")

    return out

from flovopy.core.gaputils import _generate_spectrally_matched_noise as _gen_spec_noise

def _pad_trace(tr: Trace, seconds: float, method: str = "mirror") -> None:
    """
    Pad both ends of a Trace by `seconds` using one of:
      - "mirror": reflect the existing ends (best for filtering/decon)
      - "zeros" : zero pad
      - "noise" : spectrally matched noise (uses gaputils._generate_spectrally_matched_noise)

    Notes
    -----
    - Works if tr.data is masked or ndarray; masked data are treated with a
      non-destructive filled copy *only* for computing the pad content.
    - Stores pad metadata for clean removal in `_unpad_trace()`:
        tr.stats._pad = {"seconds": seconds, "npts": npts_pad, "method": method}
      and also stores original times:
        tr.stats.originalStartTime / tr.stats.originalEndTime
    - Does not taper; your pad→taper→filter→unpad sequence stays intact.
    """
    if seconds is None or seconds <= 0:
        return
    sr = float(tr.stats.sampling_rate or 0.0)
    if sr <= 0:
        raise ValueError("Cannot pad trace with non-positive sampling_rate.")
    npts_pad = int(round(seconds * sr))
    if npts_pad == 0:
        return

    # Preserve original timing
    tr.stats["originalStartTime"] = tr.stats.starttime
    tr.stats["originalEndTime"] = tr.stats.endtime

    # Build a read-only ndarray view of the data for constructing the pad
    data = tr.data
    if isinstance(data, np.ma.MaskedArray):
        y = np.asarray(data.filled(0.0))  # non-destructive fill for padding only
    else:
        y = np.asarray(data)

    n = y.size
    if n == 0:
        # Degenerate: pad just with zeros or noise
        if method == "noise":
            pre = _gen_spec_noise(tr, taper_percentage=0.2)[:npts_pad]
            post = _gen_spec_noise(tr, taper_percentage=0.2)[:npts_pad]
        elif method == "mirror":
            pre = np.zeros(npts_pad, dtype=y.dtype)
            post = np.zeros(npts_pad, dtype=y.dtype)
        elif method == "zeros":
            pre = np.zeros(npts_pad, dtype=y.dtype)
            post = np.zeros(npts_pad, dtype=y.dtype)
        else:
            raise ValueError("method must be one of {'mirror','zeros','noise'}")
    else:
        if method == "mirror":
            # Reflect edges; if request exceeds available samples, tile the reflected edge
            k = min(npts_pad, n)
            left_edge = np.flip(y[:k])
            right_edge = np.flip(y[-k:])
            if k < npts_pad:
                reps = int(np.ceil(npts_pad / k))
                left_edge = np.resize(np.tile(left_edge, reps), npts_pad)
                right_edge = np.resize(np.tile(right_edge, reps), npts_pad)
            pre = left_edge.astype(y.dtype, copy=False)
            post = right_edge.astype(y.dtype, copy=False)

        elif method == "zeros":
            pre = np.zeros(npts_pad, dtype=y.dtype)
            post = np.zeros(npts_pad, dtype=y.dtype)

        elif method == "noise":
            # Generate noise with the same spectrum as valid samples
            noise_full = _gen_spec_noise(tr, taper_percentage=0.2)
            # In case _gen_spec_noise returns length n, slice/resize as needed
            if noise_full.size < npts_pad:
                reps = int(np.ceil(npts_pad / max(1, noise_full.size)))
                noise_full = np.tile(noise_full, reps)
            pre = noise_full[:npts_pad].astype(y.dtype, copy=False)
            post = noise_full[:npts_pad].astype(y.dtype, copy=False)

        else:
            raise ValueError("method must be one of {'mirror','zeros','noise'}")

    # Concatenate pad + original + pad
    tr.data = np.concatenate([pre, y.astype(pre.dtype, copy=False), post])

    # Update timing and metadata
    tr.stats.starttime -= npts_pad * tr.stats.delta
    tr.stats["_pad"] = {"seconds": float(seconds), "npts": int(npts_pad), "method": str(method)}

    # History (optional, keep concise)
    try:
        if not hasattr(tr.stats, "history"):
            tr.stats.history = []
        tr.stats.history.append(f"padded:{method}({seconds:.3f}s)")
    except Exception:
        pass

def _update_trace_filter(tr, filtertype, freq, zerophase):
    """
    Updates the filter settings applied to a seismic trace.

    This function modifies the **filter metadata** stored in `tr.stats['filter']`, ensuring
    that **frequency bounds and phase settings** are correctly tracked.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace whose filter metadata will be updated.
    filtertype : str
        Type of filter applied (`"highpass"`, `"lowpass"`, or `"bandpass"`).
    freq : float or tuple
        The cutoff frequency (single value for `"highpass"` or `"lowpass"`, tuple for `"bandpass"`).
    zerophase : bool
        Indicates whether the filter is zero-phase.

    Returns:
    -------
    None
        The function modifies `tr.stats['filter']` **in place**.

    Notes:
    ------
    - Ensures the trace has a `filter` dictionary (`tr.stats['filter']`).
    - Updates **frequency bounds** based on the filter type:
      - `"highpass"` → Updates `freqmin` only.
      - `"lowpass"` → Updates `freqmax` only.
      - `"bandpass"` → Updates both `freqmin` and `freqmax`.
    - Tracks whether the filter is **zero-phase**.

    Example:
    --------
    ```python
    from obspy import read

    tr = read("example.mseed")[0]
    
    # Apply a highpass filter and update metadata
    _update_trace_filter(tr, "highpass", freq=1.0, zerophase=True)

    print(tr.stats['filter'])  # Metadata now includes the filter
    ```
    """
    if 'filter' not in tr.stats:
        tr.stats['filter'] = {'freqmin':0, 'freqmax':tr.stats.sampling_rate/2, 'zerophase': False}
    if filtertype == 'highpass':    
        tr.stats.filter["freqmin"] = max([freq, tr.stats.filter["freqmin"]])
    if filtertype == 'bandpass':
        tr.stats.filter["freqmin"] = max([freq[0], tr.stats.filter["freqmin"]]) 
        tr.stats.filter["freqmax"] = min([freq[1], tr.stats.filter["freqmax"]])
    if filtertype == 'lowpass':
        tr.stats.filter["freqmax"] = min([freq, tr.stats.filter["freqmax"]])
    tr.stats.filter['zerophase'] = zerophase

def _get_calib(tr, this_inv):
    """
    Retrieves the overall calibration factor for a given trace from an inventory.

    This function looks up the **instrument response calibration factor** (sensitivity and gain)
    for a specific station and channel from an ObsPy **Inventory**.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace whose calibration factor is needed.
    this_inv : obspy.Inventory
        The station metadata (e.g., from a `StationXML` file).

    Returns:
    -------
    float
        The **calibration value** (overall sensitivity) for the trace.

    Notes:
    ------
    - Searches `this_inv` for a **matching station and channel**.
    - Extracts **overall sensitivity** from `channel.response`.
    - Assumes `this_inv` contains only **one network**.

    Example:
    --------
    ```python
    from obspy import read_inventory, read

    inv = read_inventory("example.xml")
    tr = read("example.mseed")[0]

    calib = _get_calib(tr, inv)
    print(f"Calibration factor: {calib}")
    ```
    """
    calib_value = 1.0
    for station in this_inv.networks[0].stations:
        if station.code == tr.stats.station:
            for channel in station.channels:
                if channel.code == tr.stats.channel:
                    calib_freq, calib_value = channel.response._get_overall_sensitivity_and_gain()
    return calib_value

def _unpad_trace(tr):
    """
    Remove padding applied by `_pad_trace()`, restoring the original extent.

    Strategy:
      1) If originalStartTime/originalEndTime exist -> trim by time (robust to resampling).
      2) Else if stats._pad.npts exists and is plausible -> slice by sample count (fast).
      3) Otherwise do nothing.

    Always clears pad-related metadata and records 'unpadded' in history.
    """
    # Prefer time-based trim (robust even if npts/delta changed)
    t0 = tr.stats.get("originalStartTime", None)
    t1 = tr.stats.get("originalEndTime", None)
    if t0 is not None and t1 is not None:
        try:
            # pad=False ensures we only remove, never add
            tr.trim(starttime=t0, endtime=t1, pad=False)
        except Exception as e:
            # If trim fails for any reason, try sample-slice fallback below
            pass
        else:
            # Success -> cleanup metadata & history
            for k in ("_pad", "originalStartTime", "originalEndTime"):
                if k in tr.stats:
                    try:
                        del tr.stats[k]
                    except Exception:
                        pass
            try:
                if not hasattr(tr.stats, "history"):
                    tr.stats.history = []
                tr.stats.history.append("unpadded")
            except Exception:
                pass
            return

    # Fallback: sample-slice using stored pad size
    meta = getattr(tr.stats, "_pad", None)
    if meta and isinstance(meta, dict):
        npts_pad = int(meta.get("npts", 0))
        if npts_pad > 0 and tr.stats.npts >= 2 * npts_pad:
            tr.data = tr.data[npts_pad: tr.stats.npts - npts_pad]
            # If original times exist, restore starttime exactly; otherwise shift by delta
            if t0 is not None:
                tr.stats.starttime = t0
            else:
                tr.stats.starttime += npts_pad * tr.stats.delta
            # Cleanup
            for k in ("_pad", "originalStartTime", "originalEndTime"):
                if k in tr.stats:
                    try:
                        del tr.stats[k]
                    except Exception:
                        pass
            try:
                if not hasattr(tr.stats, "history"):
                    tr.stats.history = []
                tr.stats.history.append("unpadded")
            except Exception:
                pass
            return
    # Nothing to do
    return


def remove_low_quality_traces(st, quality_threshold=1.0, verbose=False):
    """
    Removes traces from an ObsPy Stream based on a quality factor threshold.

    This function scans all traces in the given Stream and removes those with 
    a `quality_factor` lower than the specified threshold.

    Parameters:
    ----------
    st : obspy.Stream
        The input Stream object containing seismic traces.
    quality_threshold : float, optional
        The minimum `quality_factor` a trace must have to remain in the Stream (default: 1.0).

    Returns:
    -------
    None
        The function **modifies** the input `Stream` in-place, removing low-quality traces.

    Notes:
    ------
    - Assumes `tr.stats.quality_factor` is set for each trace.
    - If `quality_factor` is missing from a trace, the function may raise an error.
    - The function **modifies** `st` in-place instead of returning a new Stream.

    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data
    st = read("example.mseed")

    # Remove traces with quality factor below 1.5
    remove_low_quality_traces(st, quality_threshold=1.5)

    print(f"Remaining traces: {len(st)}")
    ```
    """
    for tr in st:
        if tr.stats.quality_factor < quality_threshold: 
            if verbose:
                print(f'Removing {tr.id} because quality_factor is {tr.stats.quality_factor} and threshold is {quality_threshold}')
            st.remove(tr)




def preprocess_stream(st, bool_despike=True, bool_clean=True, inv=None, \
                      quality_threshold=-np.inf, taperFraction=0.05, \
                    filterType="bandpass", freq=[0.5, 30.0], corners=6, \
                    zerophase=False, outputType='VEL', \
                    miniseed_qc=True, verbose=False, max_dropout=None, \
                    units='Counts', bool_detrend=True):
    """
    Preprocesses a seismic stream by applying quality control, filtering, and instrument response correction.

    This function performs the following operations:
    - Quality control checks, including dropout detection and data gaps.
    - Optional despiking to remove single-sample anomalies.
    - Detrending, tapering, and bandpass filtering.
    - Instrument response removal (if an ObsPy inventory is provided).
    - Scaling data to physical units using the calibration factor (`calib`).
    - Tracks processing steps in `tr.stats.history`.

    Parameters:
    ----------
    st : obspy.Stream
        The seismic stream to process.
    bool_despike : bool, optional
        Whether to remove single-sample spikes from the trace (default: True).
    bool_clean : bool, optional
        Whether to apply detrending, tapering, and filtering (default: True).
    inv : obspy.Inventory, optional
        Instrument response metadata for deconvolution (default: None).
    quality_threshold : float, optional
        Minimum quality factor required to keep the trace (default: -Inf).
    taperFraction : float, optional
        Fraction of the trace length to use for tapering (default: 0.05).
    filterType : str, optional
        Type of filter to apply. Options: "bandpass", "lowpass", "highpass" (default: "bandpass").
    freq : list of float, optional
        Frequency range for filtering: [freq_min, freq_max] (default: [0.5, 30.0] Hz).
    corners : int, optional
        Number of filter corners (default: 6).
    zerophase : bool, optional
        Whether to apply a zero-phase filter (default: False).
    outputType : str, optional
        Type of output after instrument response removal. Options: "VEL" (velocity), "DISP" (displacement), "ACC" (acceleration), "DEF" (default) (default: "VEL").
    miniseed_qc : bool, optional
        Whether to perform MiniSEED quality control checks (default: True).
    verbose : bool, optional
        If True, prints processing steps (default: False).
    max_dropout : float, optional
        Maximum allowable data dropout percentage before rejection (default: 0.0).
    units : str, optional
        Unit of the trace before processing. Defaults to "Counts".

    Returns:
    -------
    bool
        Returns `True` if the trace was successfully processed, `False` if rejected due to poor quality or errors.
    
    Notes:
    ------
    - If `inv` is provided, `remove_response()` is used to convert the waveform to physical units.
    - If `inv` is not provided but `tr.stats.calib` is set, the trace is manually scaled.
    - If the trace fails quality checks (e.g., excessive gaps), it is rejected.
    - `tr.stats.history` keeps track of applied processing steps.

    Example:
    --------
    ```python
    from obspy import read
    from obspy.clients.fdsn import Client

    # Read a seismic trace
    tr = read("example.mseed")[0]

    # Load an inventory for response correction
    client = Client("IRIS")
    inv = client.get_stations(network="IU", station="ANMO", level="response")

    # Process the trace
    success = preprocess_trace(tr, inv=inv, verbose=True)

    if success:
        tr.plot()
    else:
        print("Trace did not meet quality criteria and was rejected.")
    ```
    """
    if len(st) > 0:
        for tr in st:                  
            if preprocess_trace(tr, bool_despike=True, \
                    bool_clean=bool_clean, \
                    inv=inv, \
                    quality_threshold=quality_threshold, \
                    taperFraction=0.05, \
                    filterType="bandpass", \
                    freq=freq, \
                    corners=2, \
                    zerophase=False, \
                    outputType=outputType, \
                    miniseed_qc=True, \
                    max_dropout=max_dropout):
                pass
            else:
                st.remove(tr)
            
        remove_low_quality_traces(st, quality_threshold=quality_threshold)


def order_traces_by_id(st):
    sorted_ids = sorted(tr.id for tr in st)
    return Stream([tr.copy() for id in sorted_ids for tr in st if tr.id == id])

