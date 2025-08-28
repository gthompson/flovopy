from collections import defaultdict
from obspy import UTCDateTime

def _sd(ch):  # safe start
    return ch.start_date or UTCDateTime(1900,1,1)
def _ed(ch):  # safe end
    return ch.end_date or UTCDateTime(2100,1,1)

def _response_signature(resp):
    """
    Summarize the response 'shape' so equivalent responses hash the same.
    Light but robust: stage class, stage gain, ref freq, decimation, sample_rate,
    and (for PZ) the number of poles/zeros.
    """
    if not resp or not getattr(resp, "response_stages", None):
        return ("noresp",)
    sig = [len(resp.response_stages)]
    for st in resp.response_stages:
        cls = st.__class__.__name__
        g   = getattr(st, "stage_gain", None)
        gg  = getattr(g, "gain", None)
        gf  = getattr(g, "frequency", None)
        df  = getattr(st, "decimation_factor", None)
        sr  = getattr(st, "sample_rate", None)
        np_ = len(getattr(st, "poles", []) or []) if hasattr(st, "poles") else None
        nz_ = len(getattr(st, "zeros", []) or []) if hasattr(st, "zeros") else None
        sig.append((cls, gg, gf, df, sr, np_, nz_))
    return tuple(sig)

def _merge_channel_epochs_by_signature(inv, *, eps=0.001):
    """
    Per station:
      - bucket channels by (location_code, code, response_signature)
      - sort epochs and merge any that overlap or abut (gap ≤ eps)
      - keep distinct signatures as separate, non-overlapping timelines
    """
    for net in inv.networks:
        for sta in net.stations:
            buckets = defaultdict(list)
            for ch in sta.channels:
                sig = _response_signature(getattr(ch, "response", None))
                buckets[(ch.location_code, ch.code, sig)].append(ch)

            new_channels = []
            for (_loc, _code, _sig), chans in buckets.items():
                chans.sort(key=lambda c: (_sd(c), _ed(c)))
                cur = chans[0]
                for nxt in chans[1:]:
                    # overlap or abut?
                    if _sd(nxt) <= _ed(cur) + eps:
                        # extend end if needed
                        if _ed(nxt) > _ed(cur):
                            # preserve 'open-ended' (None) if nxt is open-ended
                            cur.end_date = nxt.end_date
                        # else nxt fully contained; drop it
                    else:
                        new_channels.append(cur)
                        cur = nxt
                new_channels.append(cur)

            sta.channels = new_channels

def _same_response(a, b) -> bool:
    # lightweight equality: response “shape” + SR/units/az/dip
    try:
        if float(getattr(a, "sample_rate", 0)) != float(getattr(b, "sample_rate", 0)):
            return False
        # Responses can be large; string compare is crude but effective when built consistently
        return str(a.response) == str(b.response)
    except Exception:
        return False

def _dedupe_and_merge_channel_epochs(inv, *, eps=0.001):
    """
    Per station:
      - Drop exact duplicates (same loc, chan, start, end, response)
      - Merge overlapping or abutting epochs (gap <= eps) when responses match
      - Keep distinct, separated epochs as-is
    """
    def _sd(c): return c.start_date or UTCDateTime(1900, 1, 1)
    def _ed(c): return c.end_date or UTCDateTime(2100, 1, 1)

    for net in inv.networks:
        for sta in net.stations:
            # bucket by (loc, chan)
            buckets = {}
            for ch in sta.channels:
                key = (ch.location_code, ch.code)
                buckets.setdefault(key, []).append(ch)

            new_channels = []
            for (_, _), chans in buckets.items():
                # sort by (start, end)
                chans.sort(key=lambda c: (_sd(c), _ed(c)))

                merged = []
                for c in chans:
                    if not merged:
                        merged.append(c)
                        continue

                    last = merged[-1]
                    last_sd, last_ed = _sd(last), _ed(last)
                    c_sd, c_ed = _sd(c), _ed(c)
                    same_resp = _same_response(last, c)

                    # exact duplicate epoch with same response -> drop
                    if (last.location_code == c.location_code and
                        last.code == c.code and
                        str(last.start_date) == str(c.start_date) and
                        str(last.end_date) == str(c.end_date) and
                        same_resp):
                        continue

                    # overlap or abut (≤ eps) ?
                    overlap = (last_ed + eps) >= c_sd

                    if overlap and same_resp:
                        # extend last's end_date to the max
                        if c_ed > last_ed:
                            last.end_date = None if c.end_date is None else c.end_date
                    else:
                        # keep as a separate epoch
                        merged.append(c)

                new_channels.extend(merged)

            sta.channels = new_channels

from collections import defaultdict
from obspy import UTCDateTime
import hashlib

def _response_fingerprint(resp) -> str:
    """
    Create a stable fingerprint for a Channel.response so we can decide if two
    responses are 'the same'. Falls back to 'NO_RESP' if missing/empty.
    """
    if resp is None or not getattr(resp, "response_stages", None):
        return "NO_RESP"
    # The string repr is stable enough for identical responses; hash to keep key short.
    s = str(resp)
    return "RESP:" + hashlib.sha1(s.encode("utf-8")).hexdigest()

def _condense_station_channels(sta, *, touch_tol: float = 0.0):
    """
    For a given Station, merge its channels by (loc, code, response) when epochs overlap
    or *touch* within `touch_tol` seconds. Keeps metadata from the first epoch and
    extends start/end to cover the union.
    """
    def _sd(ch): return ch.start_date or UTCDateTime(1900,1,1)
    def _ed(ch): return ch.end_date or UTCDateTime(2100,1,1)

    buckets = defaultdict(list)
    for ch in sta.channels:
        key = (ch.location_code, ch.code, _response_fingerprint(getattr(ch, "response", None)))
        buckets[key].append(ch)

    new_chans = []
    for (loc, code, _fp), chans in buckets.items():
        # sort by start, then end
        chans.sort(key=lambda ch: (_sd(ch), _ed(ch)))

        merged = []
        for ch in chans:
            if not merged:
                merged.append(ch)
                continue

            last = merged[-1]
            last_sd, last_ed = _sd(last), _ed(last)
            ch_sd,   ch_ed   = _sd(ch),   _ed(ch)

            # touching if last_ed == ch_sd (or within tolerance)
            touching = (last_ed + touch_tol) >= ch_sd
            overlapping = last_ed > ch_sd
            if overlapping or touching:
                # extend the last channel's window to cover this one
                if ch_sd < last_sd:
                    last.start_date = ch.start_date  # keep None iff ch.start_date is None
                if ch_ed > last_ed:
                    last.end_date = ch.end_date      # keep None iff ch.end_date is None
                # (we keep metadata/response from 'last'; they should be equivalent by bucket)
            else:
                merged.append(ch)

        new_chans.extend(merged)

    sta.channels = new_chans
