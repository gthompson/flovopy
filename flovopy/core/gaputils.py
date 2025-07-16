import numpy as np
from obspy import Trace, Stream


def smart_fill(tr: Trace, short_thresh=5.0):
    short, long = classify_gaps(tr, threshold_seconds=short_thresh)
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()
    data = tr.data.copy()

    # Fill short gaps with interpolation
    for s, e in short:
        x = np.arange(len(data))
        valid = ~data.mask
        data[s:e] = np.interp(x[s:e], x[valid], data[valid])

    # Fill long gaps with 0.0 or NaN
    for s, e in long:
        data[s:e] = 0.0

    filled = tr.copy()
    filled.data = data.filled(0.0).astype(np.float32)
    return filled

def classify_gaps(tr: Trace, sampling_rate=None, threshold_seconds=5.0):
    """
    Returns indices of short vs. long masked spans based on duration threshold.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return [], []

    sampling_rate = sampling_rate or tr.stats.sampling_rate
    mask = tr.data.mask
    gaps = []
    in_gap = False
    for i, m in enumerate(mask):
        if m and not in_gap:
            start = i
            in_gap = True
        elif not m and in_gap:
            end = i
            in_gap = False
            duration = (end - start) / sampling_rate
            gaps.append((start, end, duration))

    short = [(s, e) for s, e, d in gaps if d <= threshold_seconds]
    long = [(s, e) for s, e, d in gaps if d > threshold_seconds]
    return short, long


def fill_stream_gaps(stream, method="constant", **kwargs):
    """
    Apply a gap-filling method to all Traces in a Stream.

    Parameters:
    -----------
    stream : Stream
        ObsPy Stream object with Traces (preferably with masked gaps).
    method : str
        Gap filling method. Options: "linear", "previous", "noise".
    kwargs : dict
        Additional keyword arguments passed to the selected method.

    Returns:
    --------
    Stream
        Gap-filled stream.
    """
    methods = {
        "constant": fill_gaps_with_constant,
        "linear": fill_gaps_with_linear_interpolation,
        "previous": fill_gaps_with_previous_value,
        "noise": fill_gaps_with_filtered_noise
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Valid options: {list(methods.keys())}")

    filled_stream = Stream()
    for tr in stream:
        filled_stream.append(methods[method](tr, **kwargs))
    return filled_stream


def fill_gaps_with_constant(tr: Trace, fill_value: float = 0.0, **kwargs) -> Trace:
    """
    Fill masked gaps in a Trace with a constant value (default is 0.0).

    Parameters:
    -----------
    tr : obspy.Trace
        Trace with masked gaps in tr.data.
    fill_value : float
        Value to fill the gaps with.

    Returns:
    --------
    obspy.Trace
        Trace with gaps filled and no masked array.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    new_tr.data = tr.data.filled(fill_value).astype(np.float32)
    return new_tr

def fill_gaps_with_linear_interpolation(tr: Trace, **kwargs) -> Trace:
    """
    Fill masked gaps in a Trace using linear interpolation.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    data = new_tr.data.astype(np.float32)
    mask = data.mask

    if not np.any(mask):
        return new_tr

    x = np.arange(len(data))
    data[mask] = np.interp(x[mask], x[~mask], data[~mask])
    new_tr.data = data.data  # strip the mask
    return new_tr


def fill_gaps_with_previous_value(tr: Trace, **kwargs) -> Trace:
    """
    Fill masked gaps by repeating the previous valid data value.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    data = new_tr.data.astype(np.float32)
    mask = data.mask

    if not np.any(mask):
        return new_tr

    for i in range(1, len(data)):
        if mask[i]:
            data[i] = data[i - 1]
    new_tr.data = data.data
    return new_tr


def fill_gaps_with_filtered_noise(tr: Trace, taper_percentage=0.2, **kwargs) -> Trace:
    """
    Fill masked gaps using spectrally matched noise.
    """
    if not isinstance(tr.data, np.ma.MaskedArray):
        return tr.copy()

    new_tr = tr.copy()
    data = new_tr.data.astype(np.float32)
    mask = data.mask

    if not np.any(mask):
        return new_tr

    noise = _generate_spectrally_matched_noise(new_tr, taper_percentage=taper_percentage)
    data[mask] = noise[mask]
    new_tr.data = data.data
    return new_tr


def _generate_spectrally_matched_noise(tr: Trace, taper_percentage=0.2, **kwargs):
    """
    Generate noise matched to the spectral characteristics of the unmasked signal.
    """
    data = tr.data
    if not isinstance(data, np.ma.MaskedArray):
        data = np.ma.masked_array(data)

    unmasked = data[~data.mask]
    nfft = _npts2nfft(len(data))
    fft_vals = np.fft.rfft(unmasked, n=nfft)
    amplitude_spectrum = np.abs(fft_vals)

    random_phases = np.exp(2j * np.pi * np.random.rand(len(amplitude_spectrum)))
    synthetic_fft = amplitude_spectrum * random_phases

    synthetic_noise = np.fft.irfft(synthetic_fft, n=nfft).real[:len(data)]

    # Apply taper to reduce edge effects
    npts = len(data)
    taper_len = int(npts * taper_percentage / 2)
    taper = np.ones(npts)
    taper[:taper_len] = np.linspace(0, 1, taper_len)
    taper[-taper_len:] = np.linspace(1, 0, taper_len)

    return synthetic_noise * taper