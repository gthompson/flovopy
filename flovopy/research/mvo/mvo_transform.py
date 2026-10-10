"""Opt-in adapter for historical MVO waveform corrections.

No Montserrat-specific corrections are applied by generic SEISAN conversion.
Review correction rules against source metadata before using in bulk migrations.
"""
from __future__ import annotations


def correct_mvo_stream(stream):
    """Apply existing MVO trace corrections to a copy of an ObsPy Stream.

    Existing ``fix_trace_mvo`` mutates traces and may alter times and sample
    rates as well as NSLC codes. This adapter intentionally does not hide that.
    """
    from flovopy.research.mvo.mvo_ids import fix_trace_mvo

    corrected = stream.copy()
    for trace in corrected:
        fix_trace_mvo(trace, verbose=False)
    return corrected
