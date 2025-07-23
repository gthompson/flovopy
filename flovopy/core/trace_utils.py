"""
trace_utils.py

Utility functions for working with ObsPy Trace objects in FLOVOpy.

Includes tools for:
- Ensuring data is in float32 format
- Enforcing masked arrays for data handling
- Comparing traces with tolerance, ignoring zeros/gaps

These helpers support robust preprocessing and testing workflows
when working with volcano-seismic time series data.

Author: [Your Name or FLOVOpy Development Team]
"""

import numpy as np
from obspy import Trace

#from flovopy.core.miniseed_io import sanitize_trace  # Assumes sanitize_trace is implemented in masking.py


def ensure_float32(tr: Trace) -> None:
    """
    Convert trace data to float32 if not already.

    Parameters
    ----------
    tr : Trace
        ObsPy Trace object to convert in-place.
    """
    if not np.issubdtype(tr.data.dtype, np.floating) or tr.data.dtype != np.float32:
        tr.data = tr.data.astype(np.float32)


def ensure_masked(trace: Trace) -> None:
    """
    Ensure the Trace data is a masked array (np.ma.MaskedArray).

    This allows for handling of missing/invalid values (e.g., gaps or artifacts)
    using mask-aware operations like merging, gap-filling, or comparison.

    Parameters
    ----------
    trace : Trace
        ObsPy Trace object to modify in-place.
    """
    if not np.ma.isMaskedArray(trace.data):
        trace.data = np.ma.masked_array(trace.data, mask=False)


def trace_equals(trace1: Trace, trace2: Trace, rtol=1e-5, atol=1e-8) -> bool:
    """
    Compare two ObsPy Trace objects for equality, ignoring gaps and zeros.

    Parameters
    ----------
    trace1 : Trace
        First trace to compare.
    trace2 : Trace
        Second trace to compare.
    rtol : float
        Relative tolerance for floating-point comparison.
    atol : float
        Absolute tolerance for floating-point comparison.

    Returns
    -------
    bool
        True if traces are equal within tolerance (ignoring gaps), False otherwise.
    """
    if trace1.id != trace2.id:
        return False
    if abs(trace1.stats.starttime - trace2.stats.starttime) > trace1.stats.delta / 4:
        return False
    if trace1.stats.sampling_rate != trace2.stats.sampling_rate:
        return False

    t1 = sanitize_trace(trace1.copy())
    t2 = sanitize_trace(trace2.copy())

    if len(t1.data) != len(t2.data):
        return False

    return np.allclose(
        t1.data.filled(np.nan),
        t2.data.filled(np.nan),
        rtol=rtol,
        atol=atol,
        equal_nan=True
    )
