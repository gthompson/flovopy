from obspy import Trace, Stream, UTCDateTime
import numpy as np

def estimate_snr(
    trace_or_stream,
    method: str = 'std',                 # 'max'|'std'|'rms'|'spectral'
    window_length: float = 1.0,          # seconds for auto/onset windowing
    split_time=None,                     # UTCDateTime or (UTCDateTime, UTCDateTime)
    verbose: bool = True,
    spectral_kwargs: dict | None = None, # forwarded to compute_amplitude_ratios
    freq_band: tuple[float,float] | None = None,  # (fmin,fmax) average band for spectral SNR
    bandpass_for_time_methods: tuple[float,float] | None = None, # optional bandpass for time-domain SNR
) -> tuple[float, float, float] | tuple[list, list, list]:
    """
    Estimate SNR for a Trace or Stream.

    Returns
    -------
    (snr, signal_val, noise_val) for a Trace
    or
    ([snr...], [signal_val...], [noise_val...]) for a Stream
    """
    if isinstance(trace_or_stream, Stream):
        out_snr, out_sig, out_noi = [], [], []
        for tr in trace_or_stream:
            s, a, n = estimate_snr(
                tr, method=method, window_length=window_length, split_time=split_time,
                verbose=verbose, spectral_kwargs=spectral_kwargs, freq_band=freq_band,
                bandpass_for_time_methods=bandpass_for_time_methods
            )
            out_snr.append(s); out_sig.append(a); out_noi.append(n)
        return out_snr, out_sig, out_noi

    tr: Trace = trace_or_stream

    def _extract_window(trc: Trace, t1: UTCDateTime, t2: UTCDateTime):
        """Return a trimmed *copy* without padding; None if empty."""
        try:
            c = trc.copy().trim(starttime=t1, endtime=t2, pad=False)
            return c if c.stats.npts > 0 else None
        except Exception:
            return None

    # --- define signal/noise windows ---
    pre_noise_guard = 2.0  # seconds before onset
    sig_tr = noi_tr = None

    if split_time:
        if isinstance(split_time, (list, tuple)) and len(split_time) == 2:
            t_on, t_off = split_time
            dur = float(t_off - t_on)
            if not np.isfinite(dur) or dur <= 0:
                return np.nan, np.nan, np.nan
            sig_tr = _extract_window(tr, t_on, t_off)
            n2 = t_on - pre_noise_guard
            n1 = n2 - dur
            noi_tr = _extract_window(tr, n1, n2)
        else:
            t_on = split_time
            dur = float(window_length)
            if not np.isfinite(dur) or dur <= 0:
                return np.nan, np.nan, np.nan
            sig_tr = _extract_window(tr, t_on, t_on + dur)
            n2 = t_on - pre_noise_guard
            n1 = n2 - dur
            noi_tr = _extract_window(tr, n1, n2)
    else:
        # auto-mode: split into equal windows, pick max-std as signal and min-std as noise
        samples_per_win = int(tr.stats.sampling_rate * window_length)
        if samples_per_win < 1 or tr.stats.npts < 2 * samples_per_win:
            if verbose: print("Trace too short for auto SNR.")
            return np.nan, np.nan, np.nan
        # Build window start indices
        starts = np.arange(0, tr.stats.npts - samples_per_win + 1, samples_per_win, dtype=int)
        stds = []
        for sidx in starts:
            w = tr.copy()
            w.data = tr.data[sidx:sidx + samples_per_win]
            stds.append(np.nanstd(w.data))
        if len(stds) < 2:
            return np.nan, np.nan, np.nan
        sig_idx = int(np.nanargmax(stds))
        noi_idx = int(np.nanargmin(stds))
        # Reconstruct windows as Traces for consistency
        sig_tr = tr.copy(); sig_tr.data = tr.data[starts[sig_idx]:starts[sig_idx]+samples_per_win]
        noi_tr = tr.copy(); noi_tr.data = tr.data[starts[noi_idx]:starts[noi_idx]+samples_per_win]

    if sig_tr is None or noi_tr is None:
        if verbose: print("[SNR] window not covered by trace; skipping (no padding).")
        return np.nan, np.nan, np.nan

    # Optional bandpass for time-domain methods (does not affect 'spectral')
    if bandpass_for_time_methods and method in ('max','std','rms'):
        f1, f2 = bandpass_for_time_methods
        try:
            sig_tr = sig_tr.copy().filter("bandpass", freqmin=float(f1), freqmax=float(f2), corners=2, zerophase=True)
            noi_tr = noi_tr.copy().filter("bandpass", freqmin=float(f1), freqmax=float(f2), corners=2, zerophase=True)
        except Exception:
            pass

    # --- compute SNR ---
    snr = signal_val = noise_val = np.nan
    try:
        if method == 'max':
            signal_val = float(np.nanmax(np.abs(sig_tr.data)))
            noise_val  = float(np.nanmax(np.abs(noi_tr.data)))
            snr = signal_val / noise_val if noise_val not in (0.0, np.nan) else np.inf

        elif method in ('std', 'rms'):
            # std == rms for zero-mean windows; robust enough for our use
            signal_val = float(np.nanstd(sig_tr.data))
            noise_val  = float(np.nanstd(noi_tr.data))
            snr = signal_val / noise_val if noise_val not in (0.0, np.nan) else np.inf

        elif method == 'spectral':
            # Wrap the two windows as Streams and use compute_amplitude_ratios
            from flovopy.core.spectral import compute_amplitude_ratios
            spectral_kwargs = spectral_kwargs or {}
            S = Stream([sig_tr.copy()])
            N = Stream([noi_tr.copy()])
            freqs, avg_ratio, *_ = compute_amplitude_ratios(
                signal_stream=S, noise_stream=N, **spectral_kwargs, verbose=verbose
            )
            if avg_ratio is None or freqs is None:
                snr = signal_val = noise_val = np.nan
            else:
                mask = np.ones_like(avg_ratio, dtype=bool)
                if freq_band:
                    fmin, fmax = map(float, freq_band)
                    mask = (freqs >= fmin) & (freqs <= fmax)
                    if not np.any(mask):
                        mask[:] = True
                vals = np.asarray(avg_ratio[mask], dtype=float)
                # For reporting we keep max/min; SNR summary as mean in-band
                signal_val = float(np.nanmax(vals)) if vals.size else np.nan
                noise_val  = float(np.nanmin(vals)) if vals.size else np.nan
                snr        = float(np.nanmean(vals)) if vals.size else np.nan

        else:
            raise ValueError(f"Unknown SNR method: {method}")

    except Exception as e:
        if verbose:
            print(f"[ERROR] SNR estimation failed: {e}")
        return np.nan, np.nan, np.nan

    # --- store on trace.metrics (canonical + method-specific) ---
    try:
        if not hasattr(tr.stats, "metrics") or tr.stats.metrics is None:
            tr.stats.metrics = {}
        # canonical (most recent call wins)
        tr.stats.metrics["snr"] = snr
        tr.stats.metrics["signal_level"] = signal_val
        tr.stats.metrics["noise_level"]  = noise_val
        # method-specific
        tr.stats.metrics[f"snr_{method}"]          = snr
        tr.stats.metrics[f"signal_level_{method}"] = signal_val
        tr.stats.metrics[f"noise_level_{method}"]  = noise_val
    except Exception:
        pass

    if verbose and np.isfinite(snr):
        print(f"[{tr.id} | {method}] SNR={snr:.2f} (signal={signal_val:.2g}, noise={noise_val:.2g})")

    return snr, signal_val, noise_val



def _check_spectral_qc(freqs, ratios, threshold=2.0, min_fraction_pass=0.5):
    """
    Check if a spectral ratio passes quality control.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins.
    ratios : np.ndarray
        Spectral amplitude ratios.
    threshold : float
        Minimum acceptable amplitude ratio.
    min_fraction_pass : float
        Fraction of frequency bins that must exceed threshold.

    Returns
    -------
    passed : bool
        Whether the QC check passed.
    failed_fraction : float
        Fraction of frequencies below threshold.
    """    
    failed = np.sum(ratios < threshold)
    frac_failed = failed / len(ratios)
    return frac_failed <= (1 - min_fraction_pass), frac_failed



