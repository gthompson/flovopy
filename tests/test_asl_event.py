import os
import tempfile
import numpy as np
from obspy import Trace, Stream, UTCDateTime
from flovopy.analysis.asl import asl_event
from flovopy.core.mvo import load_mvo_master_inventory

def test_asl_event_runs_without_error():
    # Create dummy trace
    npts = 5000
    sr = 100
    t = Trace(data=np.random.randn(npts))
    t.stats.network = "XX"
    t.stats.station = "FAKE"
    t.stats.location = ""
    t.stats.channel = "EHZ"
    t.stats.starttime = UTCDateTime("2023-01-01T00:00:00")
    t.stats.sampling_rate = sr

    st = Stream(traces=[t])
    raw_st = st.copy()

    inv = load_mvo_master_inventory()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            asl_event(st, raw_st,
                      inv=inv,
                      outdir=tmpdir,
                      Q=23,
                      surfaceWaveSpeed_kms=1.5,
                      peakf=8.0,
                      metric="rms",
                      min_stations=1,
                      interactive=False)
        except Exception as e:
            assert False, f"asl_event raised an exception: {e}"
