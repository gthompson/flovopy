# --- EnhancedTrace ----------------------------------------------------
from obspy import Trace

class EnhancedTrace(Trace):
    """Small helpers on top of ObsPy Trace."""
    def __init__(self, data=None, header=None, **kwargs):
        super().__init__(data=data, header=header, **kwargs)

    # convenience accessors
    @property
    def channel(self) -> str:
        return getattr(self.stats, "channel", "") or ""

    @property
    def component(self) -> str:
        return self.channel[-1].upper() if self.channel else ""

    @property
    def band2(self) -> str:
        # e.g., 'HH', 'BH', 'HD', 'BD'...
        return self.channel[:2].upper() if len(self.channel) >= 2 else ""

    def is_infrasound(self) -> bool:
        return len(self.channel) >= 2 and self.channel[1].upper() == "D"

    def is_seismic(self) -> bool:
        return len(self.channel) >= 2 and self.channel[1].upper() in ("H","B","E","S","L")  # adjust to taste

    def station_key(self) -> str:
        # group by NET.STA.LOC (no component)
        n = getattr(self.stats, "network", "")
        s = getattr(self.stats, "station", "")
        l = getattr(self.stats, "location", "")
        return f"{n}.{s}.{l}"

    def ensure_metrics(self):
        if not hasattr(self.stats, "metrics") or self.stats.metrics is None:
            self.stats.metrics = {}
        return self.stats.metrics