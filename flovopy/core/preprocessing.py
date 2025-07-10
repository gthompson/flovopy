import numpy as np
import os
from obspy import read, read_inventory, Stream, Trace, UTCDateTime
from scipy.signal import welch
from obspy.signal.quality_control import MSEEDMetadata 
from flovopy.core.legacy import _fix_legacy_id

"""

If you’re processing daily data, call it like:

    preprocess_stream(stream, inv=inv, max_dropout=None, outputType='VEL', ...)

If you’re processing event-sized data:

    preprocess_stream(stream, inv=inv, max_dropout=0.1, ...)

Optional: In _interpolate_small_gaps(), treat 0 as a null value only when you want to.

If you’re padding 0.0 into gaps in daily data, it’s better to exclude 0 from null values (to avoid masking valid data). When calling detrend_trace(), do:

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

def _clean_trace(tr, taperFraction, filterType, freq, corners, zerophase, inv, outputType, verbose):
    """
    Applies padding, tapering, filtering, and instrument response correction.
    """
    if verbose:
        print('- cleaning trace')

    # Padding
    npts_pad = int(taperFraction * tr.stats.npts)
    npts_pad_seconds = max(npts_pad * tr.stats.delta, 1/freq[0])  # Ensure minimum pad length
    _pad_trace(tr, npts_pad_seconds)
    max_fraction = npts_pad / tr.stats.npts

    # Taper
    if verbose:
        print('- tapering')
    tr.taper(max_percentage=max_fraction, type="hann")
    add_to_trace_history(tr, 'tapered')

    # Filtering
    if inv:
        # Estimate pre_filt based on bandpass
        nyquist = 0.5 / tr.stats.delta
        if filterType == "bandpass":
            fmin, fmax = freq
        else:
            fmin = freq
            fmax = 999999 # unrealistically high
        pre_filt = (
            max(0.01, fmin * 0.5),
            fmin,
            min(fmax, nyquist * 0.95),
            min(fmax * 1.5, nyquist * 0.99)
        )

        # Instrument Response Removal
        _handle_instrument_response(tr, inv, pre_filt, outputType, verbose)    

    else: # Filter only, do not remove response
        if verbose:
            print('- filtering')
        if filterType == "bandpass":
            tr.filter(filterType, freqmin=freq[0], freqmax=freq[1], corners=corners, zerophase=zerophase)
        else:
            tr.filter(filterType, freq=freq[0], corners=corners, zerophase=zerophase)
        _update_trace_filter(tr, filterType, freq, zerophase)
        add_to_trace_history(tr, filterType)

    # Remove Padding
    _unpad_trace(tr)

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

def _can_write_to_miniseed_and_read_back(tr, return_metrics=True):
    """
    Tests whether an ObsPy Trace can be written to and successfully read back from MiniSEED format.

    This function attempts to:
    1. **Convert trace data to float** (if necessary) to avoid MiniSEED writing issues.
    2. **Write the trace to a temporary MiniSEED file**.
    3. **Read the file back to confirm integrity**.
    4. If `return_metrics=True`, computes MiniSEED metadata using `MSEEDMetadata()`.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to test for MiniSEED compatibility.
    return_metrics : bool, optional
        If `True`, computes MiniSEED quality control metrics and stores them in `tr.stats['metrics']` (default: True).

    Returns:
    -------
    bool
        `True` if the trace can be written to and read back from MiniSEED successfully, `False` otherwise.

    Notes:
    ------
    - **Converts `tr.data` to float** (`trace.data.astype(float)`) if necessary.
    - **Removes temporary MiniSEED files** after testing.
    - Uses `MSEEDMetadata()` to compute quality metrics similar to **ISPAQ/MUSTANG**.
    - Sets `tr.stats['quality_factor'] = -100` if the trace has **no data**.

    Example:
    --------
    ```python
    from obspy import read

    # Load a trace
    tr = read("example.mseed")[0]

    # Check if it can be written & read back
    success = _can_write_to_miniseed_and_read_back(tr, return_metrics=True)

    print(f"MiniSEED compatibility: {success}")
    if success and "metrics" in tr.stats:
        print(tr.stats["metrics"])  # Print MiniSEED quality metrics
    ```
    """
    if len(tr.data) == 0:
        tr.stats['quality_factor'] = 0.0
        return False

    # Convert data type to float if necessary (prevents MiniSEED write errors)
    if not np.issubdtype(tr.data.dtype, np.floating):
        tr.data = tr.data.astype(float)

    tmpfilename = f"{tr.id}_{tr.stats.starttime.isoformat()}.mseed"

    try:
        # Attempt to write to MiniSEED
        if hasattr(tr.stats, "mseed") and "encoding" in tr.stats.mseed:
            del tr.stats.mseed["encoding"]
        tr.write(tmpfilename)

        # Try reading it back
        _ = read(tmpfilename)

        if return_metrics:
            # Compute MiniSEED metadata
            mseedqc = MSEEDMetadata([tmpfilename])
            tr.stats['metrics'] = mseedqc.meta
            add_to_trace_history(tr, "MSEED metrics computed (similar to ISPAQ/MUSTANG).")

        return True  # Successfully wrote and read back

    except Exception as e:
        if return_metrics:
            tr.stats['quality_factor'] = 0.0
        print(f"Failed MiniSEED write/read test for {tr.id}: {e}")
        return False

    finally:
        # Clean up the temporary file
        if os.path.exists(tmpfilename):
            os.remove(tmpfilename)


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

def _is_empty_trace(trace):
    """
    Determines whether a seismic trace is effectively empty.

    A trace is considered empty if:
    - It has zero data points (`npts == 0`).
    - All samples are identical (e.g., all zeros, all -1s).
    - All values are NaN.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to check.

    Returns:
    -------
    bool
        `True` if the trace is empty or contains only redundant values, otherwise `False`.

    Notes:
    ------
    - The function first checks if the trace has no data (`npts == 0`).
    - Then it checks if all values are identical (suggesting a completely flat signal).
    - Finally, it verifies if all values are NaN.

    Example:
    --------
    ```python
    from obspy import Trace
    import numpy as np

    # Create an empty trace
    empty_trace = Trace(data=np.array([]))
    print(_is_empty_trace(empty_trace))  # True

    # Create a flat-line trace
    flat_trace = Trace(data=np.zeros(1000))
    print(_is_empty_trace(flat_trace))  # True

    # Create a normal trace with random data
    normal_trace = Trace(data=np.random.randn(1000))
    print(_is_empty_trace(normal_trace))  # False
    ```
    """
    if trace.stats.npts == 0:
        return True
    
    # Check for flat trace (e.g. all zero, or all -1)
    if np.all(trace.data == np.nanmean(trace.data)):
        return True

    # Check if all values are NaN
    if np.all(np.isnan(trace.data)):
        return True 

    return False


    
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

def detrend_trace(tr, gap_threshold=10, null_values=[0, np.nan], detrend_type='linear', verbose=False):
    """
    Detrends a seismic trace while handling gaps appropriately.

    This function first checks for gaps marked by NaNs or a specified null value 
    (e.g., 0). It handles small gaps by interpolating them, while large gaps 
    are treated by splitting the trace and detrending each segment separately.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to be detrended.
    gap_threshold : int, optional
        The maximum gap size (in samples) that will be interpolated. 
        Larger gaps will trigger piecewise detrending (default: 10).
    null_values : list, optional
        Values indicating missing data (default: [0, np.nan]).
    detrend_type : str, optional
        Type of detrending to apply ('linear' or 'simple') (default: 'linear').
    verbose : bool, optional
        If `True`, prints debug messages (default: False).

    Returns:
    -------
    tr : obspy.Trace
        The detrended seismic trace.
    """

    # Ensure the trace hasn't already been detrended
    if 'detrended' in tr.stats.get('history', []):
        return tr  # Already detrended

    success = False  # Flag to track successful detrending

    try:
        # Detect and interpolate small gaps (gap ≤ threshold)
        tr = _interpolate_small_gaps(tr, gap_threshold, null_values, verbose)

        # Attempt standard detrending
        tr.detrend(detrend_type)
        success = True

    except Exception:
        if verbose:
            print(f"- Standard detrending failed for {tr.id}, trying piecewise detrending.")

        # Handle larger gaps by piecewise detrending
        tr2 = _piecewise_detrend(tr, null_values=null_values, detrend_type=detrend_type, verbose=verbose)

        if tr2:
            tr = tr2
            success = True
        else:
            # Fall back to simple detrend if all else fails
            if verbose:
                print(f"- Using simple detrend for {tr.id}")
            tr.detrend('simple')
            success = True

    if success:
        add_to_trace_history(tr, 'detrended')

    return tr


def _interpolate_small_gaps(tr, gap_threshold, null_values, verbose=False):
    """
    Identifies and interpolates small gaps in a seismic trace.

    Gaps marked by specified null values (e.g., 0 or NaN) are interpolated 
    if their length does not exceed `gap_threshold`. Larger gaps remain unchanged.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace containing gaps.
    gap_threshold : int
        Maximum number of consecutive null samples to interpolate.
    null_values : list
        Values indicating missing data (e.g., [0, np.nan]).
    verbose : bool, optional
        If `True`, prints debug messages (default: False).

    Returns:
    -------
    tr : obspy.Trace
        The trace with small gaps interpolated.
    """
    data = tr.data.copy()
    mask = np.isin(data, null_values)

    # Identify gap start and end indices
    gap_starts = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0][::2]
    gap_ends = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0][1::2]

    for start, end in zip(gap_starts, gap_ends):
        gap_size = end - start
        if gap_size <= gap_threshold:
            if verbose:
                print(f"- Interpolating small gap ({gap_size} samples) in {tr.id}")
            data[start:end] = np.interp(
                np.arange(start, end),
                [start - 1, end],
                [data[start - 1], data[end]]
            )

    tr.data = data
    return tr


def _piecewise_detrend(tr, null_values=[0, np.nan], detrend_type='linear', verbose=False):
    """
    Applies piecewise detrending to a trace with large gaps.

    This function identifies contiguous segments of valid data, detrends them separately,
    and merges the results back together, filling large gaps with NaNs.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to be processed.
    null_values : list, optional
        Values indicating missing data (default: [0, np.nan]).
    detrend_type : str, optional
        Type of detrending to apply ('linear' or 'simple') (default: 'linear').
    verbose : bool, optional
        If `True`, prints debug messages (default: False).

    Returns:
    -------
    obspy.Trace or None
        - If successful, returns the detrended trace.
        - If unsuccessful, returns `None`.
    """
    if verbose:
        print(f"{tr.id}: Checking for large gaps...")

    # Mask null values (gaps)
    data_masked = np.ma.masked_where(np.isin(tr.data, null_values), tr.data)

    if data_masked.mask.any():  # If there are masked values (gaps)
        if verbose:
            print(f"{tr.id}: Large gaps detected, applying piecewise detrending.")

        # Split into contiguous segments
        st_segments = tr.split()
        if verbose:
            print(f"{tr.id}: Split into {len(st_segments)} segments.")

        # Apply detrending to each segment
        for tr_segment in st_segments:
            tr_segment.detrend(detrend_type)

        # Merge the segments back together
        st_merged = st_segments.merge(method=0, fill_value=np.nan)

        if len(st_merged) == 1:
            return st_merged[0]
        else:
            return None  # Failed to properly merge

    else:
        return tr  # No large gaps, return original trace

def _pad_trace(tr, seconds, method="mirror"):
    """
    Pads a seismic trace by extending it with mirrored data, zeros, or spectrally-matched noise.

    This function extends the **start and end** of a given **ObsPy Trace** by `seconds`, using 
    one of three methods:
    - **"mirror" (default)**: Reverses and appends the first and last `seconds` of data.
    - **"zeros"**: Pads with zeros.
    - **"noise"**: Generates and appends spectrally-matched noise.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to be padded.
    seconds : float
        The number of seconds to extend at both ends.
    method : str, optional
        Padding method (`"mirror"`, `"zeros"`, or `"noise"`). Default: `"mirror"`.

    Returns:
    -------
    None
        The function modifies `tr` **in place**, updating its `starttime`.

    Notes:
    ------
    - Stores original start and end times in `tr.stats['originalStartTime']` and `tr.stats['originalEndTime']`.
    - **Mirror padding** is best for filtering and deconvolution.
    - **Noise padding** helps maintain spectral continuity.

    Example:
    --------
    ```python
    from obspy import read

    tr = read("example.mseed")[0]
    print(f"Before padding: {tr.stats.starttime}")

    _pad_trace(tr, 5.0, method="noise")

    print(f"After padding: {tr.stats.starttime}")
    ```
    """
    if seconds <= 0.0:
        return

    # Store original time range
    tr.stats['originalStartTime'] = tr.stats.starttime
    tr.stats['originalEndTime'] = tr.stats.endtime

    npts_pad = int(tr.stats.sampling_rate * seconds)

    if method == "mirror":
        # Extract waveform segments and reverse them
        y_prepend = np.flip(tr.data[:npts_pad])  # Reverse first N samples
        y_postpend = np.flip(tr.data[-npts_pad:])  # Reverse last N samples
    elif method == "zeros":
        # Create zero-padding
        y_prepend = np.zeros(npts_pad)
        y_postpend = np.zeros(npts_pad)
    elif method == "noise":
        # Generate spectrally-matched noise
        y_prepend = _generate_spectrally_matched_noise(tr, npts_pad)
        y_postpend = _generate_spectrally_matched_noise(tr, npts_pad)
    else:
        raise ValueError(f"Invalid padding method: {method}. Choose 'mirror', 'zeros', or 'noise'.")

    # Concatenate padded trace
    tr.data = np.concatenate([y_prepend, tr.data, y_postpend])

    # Update starttime to reflect new padding
    tr.stats.starttime -= npts_pad * tr.stats.delta
    add_to_trace_history(tr, f'padded using {method}')

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
    Removes padding from a previously padded seismic trace.

    This function trims a trace back to its **original start and end times**, assuming it was 
    previously padded using `_pad_trace()`.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to unpad.

    Returns:
    -------
    None
        The function modifies `tr` **in place**, restoring its original time range.

    Notes:
    ------
    - Uses `tr.stats['originalStartTime']` and `tr.stats['originalEndTime']` for trimming.
    - Calls `tr.trim()` to remove the extra data.
    - Adds `"unpadded"` to the trace's history.

    Example:
    --------
    ```python
    from obspy import read

    tr = read("example.mseed")[0]
    
    _pad_trace(tr, 5.0, method="noise")
    print(f"Padded start time: {tr.stats.starttime}")

    _unpad_trace(tr)
    print(f"Restored start time: {tr.stats.starttime}")
    ```
    """
    if 'originalStartTime' in tr.stats and 'originalEndTime' in tr.stats:
        tr.trim(starttime=tr.stats['originalStartTime'], endtime=tr.stats['originalEndTime'], pad=False)
        add_to_trace_history(tr, 'unpadded')     


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

#######################################################################
##               Fixing IDs                                          ##
#######################################################################

def _get_band_code(sampling_rate):
    """
    Determines the appropriate band code based on the sampling rate.

    The band code is the first letter of the **SEED channel naming convention**, which 
    categorizes seismic channels based on frequency range.

    Parameters:
    ----------
    sampling_rate : float
        The sampling rate of the seismic trace in Hz.

    Returns:
    -------
    str or None
        The appropriate band code (e.g., 'B' for broadband, 'H' for high-frequency broadband).
        Returns `None` if no matching band code is found (should not happen if lookup table is correct).

    Notes:
    ------
    - This function relies on `BAND_CODE_TABLE`, a dictionary defining the mapping 
      between frequency ranges and SEED band codes.

    Example:
    --------
    ```python
    band_code = _get_band_code(100.0)
    print(band_code)  # Output: 'H' (High-frequency broadband)
    ```
    """
    # Band code lookup table based on IRIS SEED convention
    BAND_CODE_TABLE = {
        (0.0001, 0.001): "R",  # Extremely Long Period (0.0001 - 0.001 Hz)   
        (0.001, 0.01): "U",  # Ultra Low Frequency (~0.01 Hz)
        (0.01, 0.1): "V",  # Very Low Frequency (~0.1 Hz)
        (0.1, 2): "L",   # Long Period (~1 Hz)
        (2, 10): "M",  # Mid Period (1 - 10 Hz)
        (10, 80): "B", # Broadband (S if Short Period instrument, corner > 0.1 Hz)
        (80, 250): "H",  # High Frequency (80 - 250 Hz) (E if Short Period instrument, corner > 0.1 Hz)
        (250, 1000): "D",  # Very High Frequency (250 - 1000 Hz) (C if Short Period instrument, corner > 0.1 Hz)
        (1000, 5000): "G",  # Extremely High Frequency (1 - 5 kHz) (F if Short period)
    }

    for (low, high), code in BAND_CODE_TABLE.items():
        if low <= sampling_rate < high:
            return code
    return None  # Should not happen if lookup table is correct

def _adjust_band_code_for_sensor_type(current_band_code, expected_band_code, short_period=False):
    """
    Adjusts the band code if the current trace belongs to a short-period seismometer.

    SEED convention distinguishes between **broadband** and **short-period** seismometers.
    This function adjusts the expected band code based on the current sensor type.

    Mapping:
    - 'B' (Broadband) → 'S' (Short-period)
    - 'H' (High-frequency broadband) → 'E' (Short-period high-frequency)
    - 'D' (Very long period broadband) → 'C' (Short-period very long period)
    - 'G' (Extremely high-frequency broadband) → 'F' (Short-period extremely high-frequency)

    Parameters:
    ----------
    current_band_code : str
        The first character of the current `trace.stats.channel` (e.g., 'S', 'E', 'C', 'F').
    expected_band_code : str
        The computed band code based on the sampling rate.
    short_period : bool, optional
        If `True`, forces short-period band codes even if the current band code is not in the expected mapping.

    Returns:
    -------
    str
        The adjusted band code if applicable.

    Example:
    --------
    ```python
    adjusted_band_code = _adjust_band_code_for_sensor_type('S', 'B')
    print(adjusted_band_code)  # Output: 'S' (Short-period equivalent of 'B')
    ```
    """
    short_period_codes = {'S', 'E', 'C', 'F'}
    
    if current_band_code in short_period_codes or short_period:
        band_code_mapping = {'B': 'S', 'H': 'E', 'D': 'C', 'G': 'F'}
        return band_code_mapping.get(expected_band_code, expected_band_code)
    
    return expected_band_code



def fix_trace_id(trace, legacy=False, netcode=None, verbose=False):
    """
    Standardizes a seismic trace's ID to follow SEED naming conventions.

    This function:
    - Fixes legacy **VDAP/analog telemetry IDs** if `legacy=True`.
    - Ensures a valid **network code** if `netcode` is provided.
    - Adjusts the **band code** based on sampling rate.
    - Ensures the location code is **either empty or two characters**.
    - Fixes known **station name substitutions** (e.g., `CARL1` → `TANK` for KSC data).

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to modify.
    legacy : bool, optional
        If `True`, applies `_fix_legacy_id()` to correct old-style station codes (default: False).
    netcode : str, optional
        Network code to assign if missing (default: None).
    verbose : bool, optional
        If `True`, prints trace ID changes (default: False).

    Returns:
    -------
    bool
        `True` if the trace ID was changed, `False` otherwise.

    Notes:
    ------
    - Calls `_get_band_code()` to determine the correct band code based on sampling rate.
    - Calls `_adjust_band_code_for_sensor_type()` to refine the band code for short-period sensors.
    - Ensures **station names are corrected** for specific networks (e.g., `FL.CARL1 → FL.TANK`).

    Example:
    --------
    ```python
    from obspy import read

    trace = read("example.mseed")[0]
    changed = fix_trace_id(trace, legacy=True, netcode="XX", verbose=True)

    if changed:
        print(f"Updated Trace ID: {trace.id}")
    ```
    """
    changed = False

    if legacy: # indicates an old VDAP/analog telemetry network where 4-character station code includes orientation
        _fix_legacy_id(trace)

    if not trace.stats.network and netcode:
        trace.stats.network = netcode
        changed = True   
    
    current_id = trace.id
    net, sta, loc, chan = current_id.split('.')
    sampling_rate = trace.stats.sampling_rate
    current_band_code = chan[0]

    # KSC network fixes
    if net=='FL' or net=='1R': # KSC
        if sta=='CARL1':
            sta = 'TANK'
        elif sta=='CARL0':
            sta = 'BCHH'
        if trace.stats.station == 'CARL0':
            trace.stats.station = 'BCHH'
        if trace.stats.station == '378':
            trace.stats.station = 'DVEL1'
        if trace.stats.station == 'FIRE' and trace.stats.starttime.year == 2018:
            trace.stats.station = 'DVEL2'

        if trace.stats.network == 'FL':
            trace.stats.network = '1R'

        if trace.stats.location in ['00', '0', '--', '', '10']:
            trace.stats.location = '00'  
        net, sta, loc, chan = trace.id.split('.')  


    # if not an analog QC channel, fix band code
    if chan[0]=='A':
        pass
    else:

        # Determine the correct band code
        expected_band_code = _get_band_code(sampling_rate) # this assumes broadband sensor

        # adjust if short-period sensor
        expected_band_code = _adjust_band_code_for_sensor_type(current_band_code, expected_band_code)
        chan = expected_band_code + chan[1:]

    # make sure location is 0 or 2 characters
    if len(loc)==1:
        loc = loc.zfill(2)

      

    expected_id = '.'.join([net,sta,loc,chan])
    #print(current_id, expected_id)

    if (expected_id != current_id):
        changed = True
        if verbose:
            print(f"Current ID: {current_id}, Expected: {expected_id}) based on fs={sampling_rate}")
        trace.id = expected_id
    #print(trace)
    return changed 

#######################################################################
###                        Gap filling tools                        ###
#######################################################################

def fill_all_gaps(trace, verbose=False):
    """
    Identifies and fills all gaps in a seismic trace using an appropriate method based on gap size.

    This function scans a given seismic **Trace** for gaps and fills them using the best 
    available method for each case:
    - **Tiny gaps** (≤ 5 samples) → **Linear interpolation**
    - **Short gaps** (≤ 1 second) → **Repeat previous data**
    - **Longer gaps** (> 1 second) → **Spectrally-matched noise**

    If the station is **'MBSS'**, the function generates before/after plots of the trace.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace containing gaps to be filled.
    verbose : bool, optional
        If `True`, prints information about detected gaps and selected filling methods (default: False).

    Returns:
    -------
    None
        The function modifies `trace` **in place**, filling all detected gaps.

    Notes:
    ------
    - Uses `trace.get_gaps()` to identify missing data regions.
    - Calls `_fill_gap_with_linear_interpolation()`, `_fill_gap_with_repeat_previous_data()`, 
      or `_fill_gap_with_filtered_noise()` based on gap size.
    - Designed for **robust seismic data preprocessing**.

    Example:
    --------
    ```python
    from obspy import read

    # Load a trace with gaps
    tr = read("example_with_gaps.mseed")[0]

    # Fill all gaps using the best available method
    fill_all_gaps(tr, verbose=True)

    # Plot the corrected trace
    tr.plot()
    ```
    """
    #if trace.stats.station == 'MBSS':
    #    verbose = True
    #    trace.plot(outfile='MBSS_before_gap_filling.png')
    if verbose:
        print(f'filling gaps for {trace}')
    stream = Stream(traces=[trace])  # Wrap the trace in a stream
    gaps = stream.get_gaps()  # Get detected gaps

    for net, sta, loc, chan, t1, t2, delta, samples in gaps:
        gap_start = UTCDateTime(t1)
        gap_end = UTCDateTime(t2)

        # Select appropriate gap-filling method
        if samples <= 5:  # Tiny gaps (few samples) → Linear interpolation
            if verbose:
                print(f'gap start={gap_start}, end={gap_end}, linear interpolation')
            _fill_gap_with_linear_interpolation(trace, gap_start, gap_end)
        elif samples <= trace.stats.sampling_rate:  # Short gaps (≤ 1 sec) → Repeat data
            if verbose:
                print(f'gap start={gap_start}, end={gap_end}, repeating data')            
            _fill_gap_with_repeat_previous_data(trace, gap_start, gap_end)
        else:  # Longer gaps → Fill with spectrally-matched noise
            if verbose:
                print(f'gap start={gap_start}, end={gap_end}, adding spectral noise')            
            _fill_gap_with_filtered_noise(trace, gap_start, gap_end)
    #if trace.stats.station == 'MBSS':
    #    trace.plot(outfile='MBSS_after_gap_filling.png')

def _fill_gap_with_repeat_previous_data(trace, gap_start, gap_end):
    """
    Fills a seismic trace gap by repeating the last valid segment before the gap.

    This method is useful for **short gaps (≤ 1 second)**, where simply extending the last 
    valid data point provides a reasonable approximation.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace containing the gap.
    gap_start : UTCDateTime
        The start time of the gap.
    gap_end : UTCDateTime
        The end time of the gap.

    Returns:
    -------
    None
        The function modifies `trace.data` **in place**, filling the gap.

    Notes:
    ------
    - The function extracts the **last valid segment** before the gap and appends it.
    - Assumes the trace **has continuous data before the gap**.

    Example:
    --------
    ```python
    # Fill a gap by repeating previous data
    _fill_gap_with_repeat_previous_data(trace, gap_start, gap_end)
    ```
    """
    sample_rate = trace.stats.sampling_rate
    num_samples = int((gap_end - gap_start) * sample_rate)

    # Get the last valid segment before the gap
    fill_data = trace.data[-num_samples:]  # Last 'num_samples' of valid data
    trace.data = np.concatenate((trace.data, fill_data))

def _fill_gap_with_linear_interpolation(trace, gap_start, gap_end):
    """
    Fills gaps in a seismic trace using linear interpolation.

    This method is ideal for **tiny gaps (≤ 5 samples)**, where interpolating 
    between adjacent valid data points ensures a smooth transition.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace containing the gap.
    gap_start : UTCDateTime
        The start time of the gap.
    gap_end : UTCDateTime
        The end time of the gap.

    Returns:
    -------
    None
        The function modifies `trace.data` **in place**, replacing the gap with interpolated values.

    Notes:
    ------
    - Uses `numpy.linspace()` to interpolate values between **neighboring valid samples**.
    - Assumes the trace has valid data **before and after** the gap.

    Example:
    --------
    ```python
    # Fill a gap using linear interpolation
    _fill_gap_with_linear_interpolation(trace, gap_start, gap_end)
    ```
    """
    sample_rate = trace.stats.sampling_rate
    num_samples = int((gap_end - gap_start) * sample_rate)
    gap_idx = np.arange(num_samples)

    # Get neighboring valid samples
    prev_value = trace.data[-num_samples]  # Last valid value before the gap
    next_value = trace.data[num_samples]  # First valid value after the gap

    # Linear interpolation
    interp_values = np.linspace(prev_value, next_value, num_samples)
    trace.data[-num_samples:] = interp_values  # Replace gap with interpolated values

def _fill_gap_with_filtered_noise(trace, gap_start, gap_end):
    """
    Fills a seismic trace gap using spectrally-matched noise.

    This method is used for **longer gaps (> 1 second)** where maintaining spectral
    consistency is important. It generates **spectrally-matched noise** that closely 
    resembles the missing signal.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace containing the gap.
    gap_start : UTCDateTime
        The start time of the gap.
    gap_end : UTCDateTime
        The end time of the gap.

    Returns:
    -------
    None
        The function modifies `trace.data` **in place**, filling the gap with spectrally-matched noise.

    Notes:
    ------
    - Extracts the **last valid segment** before the gap to estimate spectral characteristics.
    - Uses `_generate_spectrally_matched_noise()` to synthesize noise with a **matching frequency profile**.
    - Ensures smooth transitions by replacing the gap **in place**.

    Example:
    --------
    ```python
    # Fill a gap using spectrally-matched noise
    _fill_gap_with_filtered_noise(trace, gap_start, gap_end)
    ```
    """
    sample_rate = trace.stats.sampling_rate
    num_samples = int((gap_end - gap_start) * sample_rate)

    # Generate spectrally-matched noise
    noise = _generate_spectrally_matched_noise(trace, num_samples)

    # Replace gap in trace
    start_index = int((gap_start - trace.stats.starttime) * sample_rate)
    trace.data[start_index : start_index + num_samples] = noise


def _generate_spectrally_matched_noise(trace, num_samples):
    """
    Generates noise that matches the spectral characteristics of a given seismic trace.

    This function extracts a segment from the trace, analyzes its frequency spectrum,
    generates noise with a matching power spectrum, and applies an inverse FFT.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to match the noise characteristics.
    num_samples : int
        The number of samples to generate.

    Returns:
    -------
    np.ndarray
        An array of spectrally-matched noise.

    Notes:
    ------
    - Uses **FFT** to analyze the dominant frequency content of the adjacent waveform.
    - Shapes Gaussian white noise to match this **spectral profile**.
    - Applies an **inverse FFT** to reconstruct the time-domain noise.
    - Ensures smooth transitions by applying a **bandpass filter** if necessary.
    """
    sample_rate = trace.stats.sampling_rate
    segment = trace.data[-num_samples:]  # Use last valid segment

    # Compute power spectral density (PSD)
    freqs, psd = welch(segment, fs=sample_rate, nperseg=min(256, num_samples))

    # Generate white noise in frequency domain
    noise_freqs = np.random.normal(size=len(freqs)) * np.sqrt(psd)

    # Convert back to time domain using inverse FFT
    noise_time_domain = np.fft.irfft(noise_freqs, n=num_samples)

    # Ensure the generated noise has the same standard deviation as the original segment
    noise_time_domain *= (np.std(segment) / np.std(noise_time_domain))

    return noise_time_domain

def _detect_and_handle_dropouts(tr, max_dropout, verbose=False):
    """
    Detects and handles dropouts (flat-line sequences) in a seismic trace.

    A dropout is a region where samples have the same value for an extended period,
    often due to telemetry failures. This function:
    - Identifies **continuous flat-line sections**.
    - If dropouts exceed `max_dropout`, the trace is **discarded**.
    - Otherwise, gaps are **filled using**:
      - **Linear interpolation** (tiny dropouts).
      - **Repeating previous data** (short dropouts).
      - **Spectrally-matched noise** (longer dropouts).

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to analyze.
    max_dropout : float
        Maximum allowed dropout duration (seconds). Traces exceeding this threshold are discarded.
    verbose : bool, optional
        If `True`, prints detailed information about detected dropouts.

    Returns:
    -------
    bool
        `True` if the trace is valid, `False` if the trace should be discarded.
    """
    seq = tr.data
    sampling_rate = tr.stats.sampling_rate
    max_samples = int(max_dropout * sampling_rate)

    # Detect contiguous sequences of identical values
    islands = _get_islands(seq, np.r_[np.diff(seq) == 0, False]) 

    try:
        _, max_length = _FindMaxLength(islands)
        add_to_trace_history(tr, f'Longest flat sequence: {max_length} samples')

        if max_length >= max_samples:
            return False  # Dropout is too long, discard trace
        else:
            fill_all_gaps(tr)
            return True
    except:
        return False  # If error occurs, discard trace

def _detect_and_handle_gaps(tr, gap_threshold=10, null_values=[0, np.nan], verbose=False):
    """
    Detects and processes gaps in a seismic trace.

    This function:
    - Uses `get_gaps()` to detect segment gaps if the trace is part of a `Stream`.
    - Identifies internal gaps (marked by NaNs or zero-like values).
    - Interpolates small gaps and applies piecewise detrending for large ones.

    Parameters:
    ----------
    tr : obspy.Trace
        The seismic trace to be processed.
    gap_threshold : int, optional
        Maximum gap length (in samples) to interpolate (default: 10).
    null_values : list, optional
        Values indicating missing data (e.g., [0, np.nan]).
    verbose : bool, optional
        If `True`, prints gap information.

    Returns:
    -------
    obspy.Trace or None
        - Returns the processed trace if successful.
        - Returns `None` if trace is too fragmented or unreliable.
    """
    # 1. Detect gaps **between** traces in a Stream (if available)
    if isinstance(tr, Stream):  # If we somehow pass a Stream
        gaps = tr.get_gaps()
        if gaps:
            if verbose:
                print(f"{tr.id}: Found {len(gaps)} segment gaps. Merging with padding.")
            tr.merge(method=1, fill_value=np.nan)  # Merge with NaN padding

    # 2. Detect internal gaps (NaNs, zeros, or other null values)
    data = tr.data.copy()
    mask = np.isin(data, null_values)

    # Identify contiguous null-value sequences
    gap_starts = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0][::2]
    gap_ends = np.where(np.diff(np.concatenate(([False], mask, [False]))))[0][1::2]

    if len(gap_starts) == 0:
        return tr  # No gaps found

    # Process gaps
    for start, end in zip(gap_starts, gap_ends):
        gap_size = end - start
        if gap_size <= gap_threshold:
            if verbose:
                print(f"- Interpolating gap ({gap_size} samples) in {tr.id}")
            data[start:end] = np.interp(
                np.arange(start, end),
                [start - 1, end],
                [data[start - 1], data[end] if end < len(data) else data[start - 1]]

            )
        else:
            if verbose:
                print(f"- Large gap ({gap_size} samples) in {tr.id}. Piecewise detrending needed.")
            tr = _piecewise_detrend(tr, null_values=null_values, verbose=verbose)
            if tr is None:
                if verbose:
                    print(f"- Trace {tr.id} is too fragmented. Discarding.")
                return None  # Give up on this trace

    tr.data = data
    return tr

#######################################################################
##                Stream tools                                       ##
#######################################################################

def remove_empty_traces(stream):
    """
    Removes empty traces, traces full of zeros, and traces full of NaNs from an ObsPy Stream.

    This function filters out traces that contain **no valid seismic data**, including:
    - **Completely empty traces** (no data points).
    - **Traces filled entirely with zeros**.
    - **Traces containing only NaN (Not-a-Number) values**.

    Parameters:
    ----------
    stream : obspy.Stream
        The input Stream object containing multiple seismic traces.

    Returns:
    -------
    obspy.Stream
        A new Stream object with only valid traces.

    Notes:
    ------
    - This function uses `_is_empty_trace(trace)` to check if a trace is empty or invalid.
    - The function does **not modify the original Stream**, but returns a cleaned copy.

    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data
    st = read("example.mseed")

    # Remove empty or invalid traces
    cleaned_st = remove_empty_traces(st)

    print(f"Original stream had {len(st)} traces, cleaned stream has {len(cleaned_st)} traces.")
    ```
    """
    '''
    cleaned_stream = Stream()  # Create a new empty Stream

    for trace in stream:
        if not _is_empty_trace(trace):
            cleaned_stream += trace'
    '''

    return Stream(tr for tr in stream if not _is_empty_trace(tr))   



def smart_merge(st, verbose=False, interactive=False):
    """
    Merges overlapping or adjacent traces in an ObsPy Stream, handling gaps and conflicts intelligently.

    This function:
    - Groups traces by unique **NSLC ID**.
    - Removes **exact duplicates** (same start/end time, sampling rate).
    - Attempts a **standard ObsPy merge** (`Stream.merge()`).
    - If the standard merge fails, **merges traces in pairs** using `smart_merge_traces()`.
    - If `interactive=True`, prompts the user to manually select traces when merging fails.

    Parameters:
    ----------
    st : obspy.Stream
        The input Stream object containing multiple seismic traces.
    verbose : bool, optional
        If `True`, prints detailed debug output (default: False).
    interactive : bool, optional
        If `True`, prompts the user for input when merge conflicts occur (default: False).

    Returns:
    -------
    obspy.Stream
        A new Stream object with merged traces.

    Notes:
    ------
    - Uses `smart_merge_traces()` for difficult merges.
    - Handles both **gaps and overlaps** intelligently.
    - Preserves **non-zero data points** when merging.

    Example:
    --------
    ```python
    from obspy import read

    # Load a Stream with overlapping traces
    st = read("example_overlapping.mseed")

    # Perform smart merging
    merged_st = smart_merge(st, verbose=True, interactive=False)

    # Print results
    print(merged_st)
    ```
    """
    newst = Stream()
    all_ids = list(set(tr.id for tr in st))

    for this_id in all_ids:
        these_traces = st.select(id=this_id).sort()

        # Remove exact duplicate traces
        traces_to_keep = []
        for i in range(len(these_traces)):
            if i == 0 or (
                these_traces[i].stats.starttime != these_traces[i - 1].stats.starttime or
                these_traces[i].stats.endtime != these_traces[i - 1].stats.endtime or
                these_traces[i].stats.sampling_rate != these_traces[i - 1].stats.sampling_rate
            ):
                traces_to_keep.append(these_traces[i])

        these_traces = Stream(traces=traces_to_keep)

        # If only 1 trace remains, add it to new Stream
        if len(these_traces) == 1:
            newst.append(these_traces[0])
            continue

        # Try standard merge
        try:
            merged_trace = these_traces.copy().merge()
            newst.append(merged_trace[0])
            if verbose:
                print("- Standard merge successful")
            continue
        except:
            if verbose:
                print("- Standard merge failed, attempting pairwise merge")

        # Pairwise smart merge strategy
        merged_trace = these_traces[0]
        for i in range(1, len(these_traces)):
            trace_pair = Stream([merged_trace, these_traces[i]])

            try:
                merged_trace = trace_pair.copy().merge()[0]  # Standard merge
            except:
                try:
                    merged_trace = _smart_merge_traces([merged_trace, these_traces[i]])  # Use smart merge
                    if verbose:
                        print(f"- Smart merged {these_traces[i].id}")
                except:
                    if verbose:
                        print(f"- Failed to smart merge {these_traces[i].id}")

        # Add merged trace to final Stream
        if merged_trace:
            newst.append(merged_trace)

        # If interactive mode is enabled, prompt user to manually choose a trace
        elif interactive:
            print("\nTrace conflict detected:\n")
            trace_pair.plot()
            for idx, tr in enumerate(trace_pair):
                print(f"{idx}: {tr}")
            choice = int(input("Enter the index of the trace to keep: "))
            newst.append(trace_pair[choice])
        else:
            raise ValueError("Unable to merge traces automatically")

    return newst


def _smart_merge_traces(trace_pair):
    """
    Merges two overlapping traces, preserving all non-zero data values.

    This function:
    - Ensures the traces have the **same ID and sampling rate**.
    - If traces cannot be merged, returns the trace with the **most valid data**.
    - Uses **non-zero values** from both traces when merging.

    Parameters:
    ----------
    trace_pair : list of obspy.Trace
        A pair of seismic traces to merge.

    Returns:
    -------
    obspy.Trace
        A single merged trace.

    Notes:
    ------
    - If traces have **different IDs or sampling rates**, the one with the most valid data is returned.
    - Overlapping segments are filled using **non-zero values from both traces**.

    Example:
    --------
    ```python
    from obspy import read

    # Load two traces with overlap
    tr1 = read("trace1.mseed")[0]
    tr2 = read("trace2.mseed")[0]

    # Perform smart merging
    merged_trace = _smart_merge_traces([tr1, tr2])

    # Print results
    print(merged_trace)
    ```
    """
    this_tr, other_tr = trace_pair

    # Check if traces are mergeable
    if this_tr.id != other_tr.id:
        print("Different trace IDs. Cannot merge.")
        return this_tr if np.count_nonzero(this_tr.data) >= np.count_nonzero(other_tr.data) else other_tr

    if this_tr.stats.sampling_rate != other_tr.stats.sampling_rate:
        print("Different sampling rates. Cannot merge.")
        return this_tr if np.count_nonzero(this_tr.data) >= np.count_nonzero(other_tr.data) else other_tr

    if abs(this_tr.stats.starttime - other_tr.stats.starttime) > this_tr.stats.delta / 4:
        print("Different start times. Cannot merge.")
        return this_tr if np.count_nonzero(this_tr.data) >= np.count_nonzero(other_tr.data) else other_tr

    if abs(this_tr.stats.endtime - other_tr.stats.endtime) > this_tr.stats.delta / 4:
        print("Different end times. Cannot merge.")
        return this_tr if np.count_nonzero(this_tr.data) >= np.count_nonzero(other_tr.data) else other_tr

    # Merge traces, using non-zero values from both
    merged_data = np.where(other_tr.data == 0, this_tr.data, other_tr.data)
    merged_trace = this_tr.copy()
    merged_trace.data = merged_data

    return merged_trace
        
def Stream_min_starttime(all_traces):
    """
    Computes the minimum and maximum start and end times for a given Stream.

    This function takes an **ObsPy Stream** containing multiple traces and 
    determines the following time statistics:
    - **Earliest start time** (`min_stime`)
    - **Latest start time** (`max_stime`)
    - **Earliest end time** (`min_etime`)
    - **Latest end time** (`max_etime`)

    Parameters:
    ----------
    all_traces : obspy.Stream
        A Stream object containing multiple seismic traces.

    Returns:
    -------
    tuple:
        - **min_stime (UTCDateTime)**: The earliest start time among all traces.
        - **max_stime (UTCDateTime)**: The latest start time among all traces.
        - **min_etime (UTCDateTime)**: The earliest end time among all traces.
        - **max_etime (UTCDateTime)**: The latest end time among all traces.

    Notes:
    ------
    - Useful for determining the **temporal coverage** of a Stream.
    - Created for the **CALIPSO data archive** (Alan Linde).

    Example:
    --------
    ```python
    from obspy import read

    # Load a Stream of seismic data
    st = read("example.mseed")

    # Compute time bounds
    min_stime, max_stime, min_etime, max_etime = Stream_min_starttime(st)

    print(f"Start Time Range: {min_stime} to {max_stime}")
    print(f"End Time Range: {min_etime} to {max_etime}")
    ```
    """ 
    min_stime = min([tr.stats.starttime for tr in all_traces])
    max_stime = max([tr.stats.starttime for tr in all_traces])
    min_etime = min([tr.stats.endtime for tr in all_traces])
    max_etime = max([tr.stats.endtime for tr in all_traces])    
    return min_stime, max_stime, min_etime, max_etime

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


def clean_velocity_stream(st, max_gap_sec=10.0, verbose=True):
    """
    Clean a Stream already in physical units (e.g., from SDS_VEL).

    Parameters
    ----------
    st : obspy.Stream
        Input Stream (already in velocity units).
    max_gap_sec : float
        Max gap (in seconds) to fill/interpolate.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    st_clean : obspy.Stream
        Cleaned Stream.
    """

    st_clean = st.copy()
    for tr in st_clean:
        if verbose:
            print(f"\n[INFO] Cleaning trace: {tr.id}")

        # Convert max gap to samples
        max_gap_samples = int(max_gap_sec * tr.stats.sampling_rate)

        # Handle gaps and dropouts
        gap_ok = _detect_and_handle_gaps(tr, gap_threshold=max_gap_samples, verbose=verbose)
        drop_ok = _detect_and_handle_dropouts(tr, max_dropout=1.0, verbose=verbose)

        if not (gap_ok and drop_ok):
            print(f"[WARN] Trace {tr.id} had large gaps. Consider trimming.")
            continue

        # Remove artifacts (e.g., clipping, spikes)
        _detect_and_correct_artifacts(tr, amp_limit=1e10, spike_thresh=4.0, fill_method="interpolate")

        # Detrend and taper (optional)
        detrend_trace(tr, gap_threshold=10, detrend_type='linear', verbose=verbose)

        add_to_trace_history(tr, "gap-cleaned without response removal")

    return st_clean

if __name__ == "__main__":
    from obspy.clients.filesystem.sds import Client
    from obspy import UTCDateTime

    sds_path = "/data/SDS_VEL"
    client = Client(sds_path)

    st = client.get_waveforms("MV", "MBLG", "", "BHZ", UTCDateTime(1997, 6, 25, 0), UTCDateTime(1997, 6, 25, 23, 59))
    st_clean = clean_velocity_stream(st, max_gap_sec=15.0, verbose=True)