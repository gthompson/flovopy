import numpy as np
from obspy import Trace, UTCDateTime

from flovopy.ingest.gem_overflow import (clip_mask, gem_jump_clusters, repair_gem_overflow,
                                         stream_clip_intervals)


def _trace(data, fs=100.0):
    return Trace(data=np.asarray(data, dtype=np.float32),
                 header=dict(network='1R', station='TST', channel='HDF', sampling_rate=fs,
                             starttime=UTCDateTime(2026, 4, 1)))


def _signal(n=6000, seed=0):
    return np.random.default_rng(seed).normal(0, 500, n)


def test_overflow_with_persistent_step_is_detected_and_repaired():
    x = _signal()
    # overflow between samples 2000 and 2300: repeated wraps, then a lasting offset
    for i in range(2000, 2300, 20):
        x[i] += 30000 * (1 if (i // 20) % 2 else -1)
    x[2300:] -= 500000
    tr = _trace(x)
    clusters = gem_jump_clusters(tr)
    assert [c['kind'] for c in clusters] == ['overflow']
    fixed = repair_gem_overflow(tr)
    # baseline after the interval is restored to that before it
    assert abs(np.median(fixed.data[2500:]) - np.median(fixed.data[:1900])) < 200
    m = clip_mask(fixed, [(c['t0'], c['t1']) for c in clusters])
    assert m[2100] and not m[1000] and not m[4000]


def test_shock_front_is_not_masked():
    x = _signal()
    x[3000] = -7000
    x[3001] = 19000          # one-sample jump > 20000 counts, no lasting step
    tr = _trace(x)
    assert [c['kind'] for c in gem_jump_clusters(tr)] == ['shock']
    assert stream_clip_intervals([tr]) == {tr.id: []}
