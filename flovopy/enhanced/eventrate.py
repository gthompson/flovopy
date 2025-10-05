# flovopy/enhanced/event_rate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.core.event import Catalog, Event

# If you keep these in flovopy.core.physics:
from flovopy.core.physics import magnitude2Eseismic, Eseismic2magnitude


@dataclass
class EventRateConfig:
    """Configuration for turning a catalog into a regular time series."""
    interval: str = "D"                 # 'T','H','D','W','M','Y' or pandas offset alias
    groupby: Optional[str] = None       # e.g., 'subclass' | 'event_type' | None
    threshold_magnitude: float = 1.0
    lowest_magnitude: float = -2.0      # replace placeholders < this with min valid
    include_energy: bool = True         # compute energy & cumulative magnitude
    # rolling/rt helpers
    rolling: Optional[int] = None       # e.g., 7 bins rolling mean on counts
    ema_alpha: Optional[float] = None   # optional exponential moving avg (0..1)


class EventRate:
    """
    Regularly-sampled time series derived from a seismic catalog.

    Data model:
      - self.df is a DataFrame with columns:
          ['time','event_count','average_magnitude','minimum_magnitude',
           'maximum_magnitude', 'cumulative_energy','cumulative_magnitude',
           'thresholded_count','thresholded_energy','thresholded_magnitude',
           (optional group column)]
      - rows are one per time bin (and per group value if groupby provided).
    """

    def __init__(self, df: pd.DataFrame, *, groupby: Optional[str] = None):
        self.df = df.copy()
        self.groupby = groupby

    # ---------- constructors ----------
    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog,
        *,
        config: Optional[EventRateConfig] = None,
        # optional projection of per-event metadata into the builder
        extractor: Optional[callable] = None,
    ) -> "EventRate":
        """
        Build an EventRate from any ObsPy Catalog/EnhancedCatalog.

        extractor(event) -> dict may return custom fields including the `groupby` key.
        If None, we try to read common fields from EnhancedEvent.meta or Event.comments.
        """
        cfg = config or EventRateConfig()

        # -------- collect per-event rows --------
        rows: List[Dict[str, Any]] = []
        for ev in catalog:  # type: Event
            if not ev.origins:
                continue
            o = ev.origins[0]
            t = o.time.datetime if o.time else None
            if t is None:
                continue

            mag = ev.magnitudes[0] if ev.magnitudes else None
            mval = mag.mag if mag else None
            mtype = mag.magnitude_type if mag else None

            row: Dict[str, Any] = {
                "time": t,
                "magnitude": mval,
                "mag_type": mtype,
            }

            # Optional enrichment
            if extractor is not None:
                try:
                    row.update(extractor(ev) or {})
                except Exception:
                    pass
            else:
                # best-effort to pull subclass/event_type if available
                # (works with EnhancedEvent.meta or comment fallbacks)
                val_et = getattr(ev, "event_type", None)
                row["event_type"] = val_et
                # subclass/mainclass via comments
                mc, sc = None, None
                for c in (ev.comments or []):
                    txt = (c.text or "").lower()
                    if "mainclass:" in txt:
                        mc = txt.split(":", 1)[-1].strip()
                    if "subclass:" in txt:
                        sc = txt.split(":", 1)[-1].strip()
                row["mainclass"] = mc
                row["subclass"] = sc

            rows.append(row)

        df = pd.DataFrame(rows)

        # -------- clean & convert --------
        df["time"] = pd.to_datetime(df["time"])
        if df.empty:
            cols = [
                "time","event_count","minimum_magnitude","average_magnitude","maximum_magnitude",
                "cumulative_energy","cumulative_magnitude",
                "thresholded_count","thresholded_energy","thresholded_magnitude",
            ]
            if cfg.groupby:
                cols.insert(1, cfg.groupby)  # after 'time'
            return cls(pd.DataFrame(columns=cols), groupby=cfg.groupby)

        # sanitize magnitudes
        df["imputed"] = df["magnitude"] < cfg.lowest_magnitude
        valid = df["magnitude"] >= cfg.lowest_magnitude
        if valid.any():
            min_valid = df.loc[valid, "magnitude"].min()
            df.loc[~valid, "magnitude"] = min_valid
        else:
            df["magnitude"] = cfg.lowest_magnitude

        if cfg.include_energy:
            df["energy"] = magnitude2Eseismic(df["magnitude"])
            nz = df["energy"] > 0
            if nz.any():
                df.loc[~nz, "energy"] = df.loc[nz, "energy"].min()
        else:
            df["energy"] = np.nan

        # -------- resample/bin --------
        group_cols: List[Any] = [pd.Grouper(key="time", freq=cfg.interval)]
        if cfg.groupby and cfg.groupby in df.columns:
            group_cols.append(cfg.groupby)

        grouped = df.groupby(group_cols).agg(
            event_count=("magnitude", "count"),
            minimum_magnitude=("magnitude", "min"),
            average_magnitude=("magnitude", "mean"),
            maximum_magnitude=("magnitude", "max"),
            cumulative_energy=("energy", "sum"),
        ).reset_index()

        if cfg.include_energy:
            grouped["cumulative_magnitude"] = Eseismic2magnitude(grouped["cumulative_energy"])
        else:
            grouped["cumulative_magnitude"] = np.nan

        # thresholded series
        thr = df[df["magnitude"] >= cfg.threshold_magnitude].groupby(group_cols).agg(
            thresholded_count=("magnitude", "count"),
            thresholded_energy=("energy", "sum"),
        ).reset_index()

        on_keys = ["time"] + ([cfg.groupby] if cfg.groupby and cfg.groupby in grouped.columns else [])
        grouped = grouped.merge(thr, on=on_keys, how="left")

        grouped = grouped.merge(thr, on=[c for c in grouped.columns if c in thr.columns], how="left")
        grouped["thresholded_count"] = grouped["thresholded_count"].fillna(0)
        grouped["thresholded_energy"] = grouped["thresholded_energy"].fillna(0)
        grouped["thresholded_magnitude"] = Eseismic2magnitude(grouped["thresholded_energy"].fillna(0))

        # ensure time col present & sorted
        grouped["time"] = pd.to_datetime(grouped["time"])
        grouped = grouped.sort_values(["time"] + ([cfg.groupby] if cfg.groupby and cfg.groupby in grouped.columns else []))

        # Optional rolling/EMA on counts (per group if present)
        if cfg.rolling or cfg.ema_alpha:
            if cfg.groupby and cfg.groupby in grouped.columns:
                grouped = grouped.groupby(cfg.groupby, group_keys=False).apply(
                    lambda g: _apply_smoothers(g, rolling=cfg.rolling, ema_alpha=cfg.ema_alpha)
                )
            else:
                grouped = _apply_smoothers(grouped, rolling=cfg.rolling, ema_alpha=cfg.ema_alpha)

        return cls(grouped, groupby=cfg.groupby)

    # ---------- convenience accessors ----------
    def pivot(self, value: str, *, fill_value: float = 0.0) -> pd.DataFrame:
        """
        Pivot into wide format when grouped, e.g. columns per subclass.
        value: one of 'event_count', 'thresholded_count', 'cumulative_magnitude', ...
        """
        if self.groupby and self.groupby in self.df.columns:
            return self.df.pivot_table(index="time", columns=self.groupby, values=value, fill_value=fill_value)
        # not grouped: return single-column time-indexed series as DataFrame
        out = self.df[["time", value]].set_index("time").sort_index()
        out.columns = [value]
        return out

    # ---------- plotting ----------
    def plot_event_count(self, *, ax=None, label: Optional[str] = None, use_smoothed: bool = True):
        df = self.df.copy()
        own = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            own = True

        series = "event_count_ema" if use_smoothed and "event_count_ema" in df.columns else "event_count_roll"
        if use_smoothed and series not in df.columns:
            series = "event_count"

        if self.groupby and self.groupby in df.columns:
            for key, g in df.groupby(self.groupby):
                ax.plot(g["time"], g[series], drawstyle="steps-mid", label=str(key))
        else:
            ax.plot(df["time"], df[series], drawstyle="steps-mid", label=label or "count")

        ax.set_ylabel("Event count")
        ax.set_xlabel("Time")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if self.groupby:
            ax.legend()
        if own:
            plt.tight_layout()
            plt.show()

    def plot_cumulative_magnitude(self, *, ax=None, label: Optional[str] = None):
        df = self.df.copy()
        own = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            own = True

        if self.groupby and self.groupby in df.columns:
            for key, g in df.groupby(self.groupby):
                ax.plot(g["time"], g["cumulative_magnitude"], drawstyle="steps-mid", label=str(key))
        else:
            ax.plot(df["time"], df["cumulative_magnitude"], drawstyle="steps-mid", label=label or "cum Me")

        ax.set_ylabel("Cumulative magnitude (from energy)")
        ax.set_xlabel("Time")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if self.groupby:
            ax.legend()
        if own:
            plt.tight_layout()
            plt.show()

    def plot_dual(
        self,
        *,
        secondary: str = "cumulative_magnitude",
        label: Optional[str] = None,
        ax=None
    ):
        """
        Dual-axis: event_count vs secondary (e.g., 'cumulative_magnitude').
        If grouped, overlays lines per group on the right axis, and sums counts on the left.
        """
        own = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
            own = True

        df = self.df.copy()
        # Left axis: counts (if grouped, show total per time)
        if self.groupby and self.groupby in df.columns:
            tot = df.groupby("time", as_index=False)["event_count"].sum()
            ax.plot(tot["time"], tot["event_count"], drawstyle="steps-mid", label="total")
        else:
            ax.plot(df["time"], df["event_count"], drawstyle="steps-mid", label=label or "count")
        ax.set_ylabel("Event count")
        ax.grid(True, linestyle="--", alpha=0.3)

        # Right axis: secondary
        ax2 = ax.twinx()
        if self.groupby and self.groupby in df.columns:
            for key, g in df.groupby(self.groupby):
                ax2.plot(g["time"], g[secondary], drawstyle="steps-mid", label=str(key))
            ax2.legend(title=self.groupby)
        else:
            ax2.plot(df["time"], df[secondary], drawstyle="steps-mid", color="green")
        ax2.set_ylabel(secondary.replace("_", " ").title(), color="green")

        ax.set_xlabel("Time")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if own:
            plt.tight_layout()
            plt.show()

    # ---------- exports ----------
    def to_csv(self, path: str) -> None:
        self.df.to_csv(path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        return self.df.copy()


# ---------- helpers ----------
def _apply_smoothers(g: pd.DataFrame, *, rolling: Optional[int], ema_alpha: Optional[float]) -> pd.DataFrame:
    g = g.copy()
    if rolling and rolling > 1:
        g["event_count_roll"] = g["event_count"].rolling(rolling, min_periods=1).mean()
    if ema_alpha and 0 < ema_alpha < 1:
        g["event_count_ema"] = g["event_count"].ewm(alpha=ema_alpha, adjust=False).mean()
    return g