from obspy import Trace

from flovopy.core.trace_utils import fix_location_code, fix_channel_code


def fix_trace_id_ksc(trace: Trace) -> None:
    """
    Normalize Kennedy Space Center (KSC) / Cape Canaveral trace metadata.

    Applies station, network, location, and channel corrections for known
    legacy and non-standard naming conventions used in KSC datasets.

    This function operates in-place on the provided Trace.

    Corrections include:
    - Mapping numeric channel codes (e.g., "2000") to SEED-standard codes
    - Assigning network code "1R" for specific stations
    - Renaming legacy station codes (e.g., CARL1 → TANK)
    - Fixing known station anomalies (e.g., FIRE → DVEL2 in 2018)
    - Normalizing location and channel codes

    Parameters
    ----------
    trace : Trace
        ObsPy Trace object to modify in-place.
    """

    net = (trace.stats.network or "").upper()
    sta = (trace.stats.station or "").upper()
    loc = (trace.stats.location or "").upper()
    chan = (trace.stats.channel or "").upper()

    # ------------------------------------------------------------------
    # Station-based channel remapping and network assignment
    # ------------------------------------------------------------------
    if sta.startswith("BHP") or sta in ("TANKP", "FIREP"):
        if chan.startswith("2"):
            if chan == "2000":
                chan = "EHZ"
            elif chan == "2001":
                chan = "EH1"
            elif chan == "2002":
                chan = "EH2"
            trace.stats.channel = chan

        trace.stats.network = "1R"

    # ------------------------------------------------------------------
    # Station renaming rules
    # ------------------------------------------------------------------
    if sta == "CARL1":
        trace.stats.station = "TANK"
    elif sta == "CARL0":
        trace.stats.station = "BCHH"
    elif sta == "378":
        trace.stats.station = "DVEL1"
    elif sta == "FIRE" and getattr(trace.stats, "starttime", None):
        if trace.stats.starttime.year == 2018:
            trace.stats.station = "DVEL2"

    # ------------------------------------------------------------------
    # Network normalization
    # ------------------------------------------------------------------
    if net == "FL":
        trace.stats.network = "1R"

    # ------------------------------------------------------------------
    # Final normalization of location and channel codes
    # ------------------------------------------------------------------
    trace.stats.location = fix_location_code(loc)
    trace.stats.channel = fix_channel_code(
        trace.stats.channel,
        trace.stats.sampling_rate,
    )