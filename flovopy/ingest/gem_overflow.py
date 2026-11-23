"""Detect, mask and repair 16-bit overflow in Gem infrasound records.

When the pressure at a Gem logger exceeds its 16-bit range (roughly 100-140 Pa at
high gain), consecutive samples of the gemconvert output jump by more than
~20000 counts and the baseline then walks away by 6.6e4 to 2e6 counts, a step
that persists after the overflow ends. Steep shock fronts (for example rocket
crackle) also produce one-sample jumps of that size but leave no persistent step,
so they are real signal and must not be masked.

Typical use, on raw counts before instrument correction::

    from flovopy.ingest.gem_overflow import (stream_clip_intervals, repair_gem_overflow,
                                             clip_mask)
    intervals = stream_clip_intervals(st_raw)          # {seed_id: [(t0, t1), ...]}
    tr_fixed = repair_gem_overflow(tr_raw)             # bridge intervals, remove step
    mask = clip_mask(tr_corrected, intervals[tr.id])   # True inside overflow

Repaired samples inside an interval are a straight line and carry no signal; mask
them in any analysis. Outside the intervals the record is usable (validated
against a co-located 24-bit microbarometer: correlation ~0.99 above 1 Hz).
"""
from __future__ import annotations

import numpy as np

JUMP_COUNTS = 20000      # one-sample jump that starts a candidate cluster
STEP_COUNTS = 32768      # persistent baseline step that marks a real overflow


def gem_jump_clusters(tr_raw, jump=JUMP_COUNTS, step=STEP_COUNTS, gap_s=2.0, pad_s=0.25, side_s=3.0):
    """Clusters of large one-sample jumps in a raw-count trace, classified.

    Jumps closer than ``gap_s`` are grouped; each group is padded by ``pad_s``.
    The baseline step is the median of the ``side_s`` seconds after the group minus
    the median of the ``side_s`` seconds before it.

    Returns a list of dicts with keys ``t0``, ``t1`` (UTCDateTime), ``n_jumps``,
    ``step`` (counts) and ``kind``: ``"overflow"`` (several jumps and a persistent
    step), ``"reset"`` (one jump and a persistent step, a baseline jump) or
    ``"shock"`` (no persistent step: real signal).
    """
    d = np.asarray(tr_raw.data, dtype=np.int64)
    k = np.flatnonzero(np.abs(np.diff(d)) > jump)
    if not len(k):
        return []
    dt, t0 = tr_raw.stats.delta, tr_raw.stats.starttime
    groups, start, prev = [], k[0], k[0]
    for x in k[1:]:
        if (x - prev) * dt > gap_s:
            groups.append((start, prev))
            start = x
        prev = x
    groups.append((start, prev))
    out = []
    for a, b in groups:
        ta, tb = t0 + a * dt - pad_s, t0 + (b + 1) * dt + pad_s
        pre, post = tr_raw.slice(ta - side_s, ta).data, tr_raw.slice(tb, tb + side_s).data
        stp = float(np.median(post) - np.median(pre)) if len(pre) and len(post) else np.nan
        n = int(((k >= a) & (k <= b)).sum())
        big = abs(stp) > step
        kind = "shock" if not big else ("overflow" if n > 1 else "reset")
        out.append(dict(t0=ta, t1=tb, n_jumps=n, step=stp, kind=kind))
    return out


def gem_clip_intervals(tr_raw, kinds=("overflow", "reset"), **kwargs):
    """Intervals ``[(t0, t1), ...]`` to mask in a raw-count Gem trace."""
    return [(c["t0"], c["t1"]) for c in gem_jump_clusters(tr_raw, **kwargs) if c["kind"] in kinds]


def stream_clip_intervals(st_raw, channel_prefix="HD", **kwargs):
    """``{seed_id: [(t0, t1), ...]}`` for every Gem trace (channel starting ``channel_prefix``)."""
    return {tr.id: gem_clip_intervals(tr, **kwargs) for tr in st_raw
            if tr.stats.channel.startswith(channel_prefix)}


def clip_mask(tr, intervals):
    """Boolean array, True where the trace's samples fall inside any interval."""
    t = tr.stats.starttime.timestamp + tr.times()
    m = np.zeros(tr.stats.npts, bool)
    for a, b in intervals:
        m |= (t >= a.timestamp) & (t <= b.timestamp)
    return m


def repair_gem_overflow(tr_raw, side_s=3.0, **kwargs):
    """Copy of a raw Gem trace with overflow/reset intervals bridged and baseline steps removed.

    Without this, the persistent step leaks through a high-pass filter as a slow
    swing lasting tens of seconds. Samples inside each interval are replaced by a
    straight line (mask them); samples after it are shifted by minus the step.
    """
    t = tr_raw.copy()
    t.data = np.asarray(t.data, dtype=np.float64)
    t0, dt = t.stats.starttime, t.stats.delta
    for c in gem_jump_clusters(tr_raw, side_s=side_s, **kwargs):
        if c["kind"] == "shock":
            continue
        i0 = max(int(round((c["t0"] - t0) / dt)), 1)
        i1 = min(int(round((c["t1"] - t0) / dt)), t.stats.npts - 1)
        pre = np.median(t.data[max(i0 - int(side_s / dt), 0):i0])
        post = np.median(t.data[i1:i1 + int(side_s / dt)])
        t.data[i1:] -= post - pre
        t.data[i0:i1] = np.linspace(t.data[i0 - 1], t.data[i1], i1 - i0, endpoint=False)
    return t
