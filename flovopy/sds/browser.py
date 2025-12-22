#!/usr/bin/env python3
"""
SDS MiniSEED Interactive Browser (EnhancedSDSClient-based)

Refactored version with:
- Centralized SDS access via EnhancedSDSClient
- Robust day-based loading
- Interactive day-to-day navigation

Keys
----
← / →     : scroll window
↑ / ↓     : change window length
[ / ]     : previous / next day
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Set, Dict
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import Stream, UTCDateTime

from flovopy.enhanced.enhanced_sds_client import EnhancedSDSClient

plt.ion()

# =============================================================================
# Constants
# =============================================================================

HIGH_RATE_FIRST: Set[str] = set("SBEHCD")

SOH_MAP: Dict[str, str] = {
    "GLA": "GNSS Latitude (deg * 1e6)",
    "GLO": "GNSS Longitude (deg * 1e6)",
    "GNS": "GNSS Satellites",
    "LCQ": "Clock Quality (%)",
    "VDT": "Digitizer Temp (C*100)",
    "VM1": "Mass Position 1",
    "VM2": "Mass Position 2",
    "VM3": "Mass Position 3",
    "VM4": "Mass Position 4",
    "VM5": "Mass Position 5",
    "VM6": "Mass Position 6",
}

IMPORTANT_SOH: Set[str] = {"LCQ", "VDT", "GNS"}

# =============================================================================
# Config
# =============================================================================

@dataclass
class BrowserConfig:
    sds_root: str = "/data/SDS"
    net: str = ""
    sta: str = "*"
    loc: str = "*"
    chan: str = "*"
    year: str = ""
    jday: str = ""
    window_sec: float = 3600.0
    mode: str = "all"

    @staticmethod
    def config_path() -> Path:
        import __main__
        script = Path(getattr(__main__, "__file__", "interactive")).stem
        return Path.home() / ".config" / "flovopy" / f"{script}.json"

    @classmethod
    def load(cls) -> "BrowserConfig":
        cfg_file = cls.config_path()
        if cfg_file.exists():
            try:
                return cls(**json.loads(cfg_file.read_text()))
            except Exception as e:
                print(f"⚠️ Failed to load config: {e}")
        return cls()

    def save(self) -> None:
        cfg_file = self.config_path()
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(asdict(self), indent=2))



def prompt(msg: str, default: str = "") -> str:
    val = input(f"{msg} [{default}]: ").strip()
    return val or default

# =============================================================================
# Channel classification
# =============================================================================

class ChannelClassifier:
    @staticmethod
    def is_high_rate(ch: str) -> bool:
        return ch and ch[0] in HIGH_RATE_FIRST

    @staticmethod
    def is_seismic(ch: str) -> bool:
        return len(ch) == 3 and ch[1] == "H" and ch[2] in "ZNE12"

    @staticmethod
    def is_infrasound(ch: str) -> bool:
        return len(ch) >= 3 and ch[1] == "D"

    @staticmethod
    def is_soh(tr) -> bool:
        return tr.stats.location == "D0"

# =============================================================================
# Stats
# =============================================================================

class StatsComputer:
    @staticmethod
    def compute(st: Stream, soh_desc=False) -> pd.DataFrame:
        rows = []
        for tr in st:
            d = tr.data.astype(float)
            if not len(d):
                continue
            r = {
                "ID": tr.id,
                "min": d.min(),
                "max": d.max(),
                "mean": d.mean(),
                "median": np.median(d),
                "rms_dev": d.std(),
            }
            if soh_desc:
                r["description"] = SOH_MAP.get(tr.stats.channel, "")
            rows.append(r)
        return pd.DataFrame(rows)

# =============================================================================
# Plotting
# =============================================================================

class PlotManager:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))

    def redraw(self, st: Stream, title: str):
        self.fig.clf()
        if not st:
            self.fig.suptitle("No data")
            self.fig.canvas.draw()
            return

        axes = self.fig.subplots(len(st), 1, sharex=True)
        if len(st) == 1:
            axes = [axes]

        for ax, tr in zip(axes, st):
            ax.plot(tr.times("matplotlib"), tr.data, lw=0.8)
            ax.set_ylabel(tr.stats.channel, rotation=0, labelpad=20)
            ax.grid(True, alpha=0.3)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].set_xlabel("UTC")
        self.fig.suptitle(title)
        self.fig.tight_layout()
        self.fig.canvas.draw()

# =============================================================================
# Browser core
# =============================================================================

class SDSBrowser:
    def __init__(self, cfg: BrowserConfig):
        self.cfg = cfg
        self.sds = None 
        self.plotter = PlotManager()
        self.stats = StatsComputer()
        self.st_full = Stream()
        self.state = None
        self.last_df = pd.DataFrame()

    def run(self):
        self._prompt()
        self._load_day()
        self._init_state()
        self._connect_keys()
        self.refresh()
        self.menu()
        self.cfg.save()

    # ---------------- setup ----------------

    def _prompt(self):
        print("\n=== SDS Browser ===\n")
        self.cfg.sds_root = prompt("SDS root", self.cfg.sds_root)
        self.cfg.net = prompt("Network", self.cfg.net)
        self.cfg.sta = prompt("Station", self.cfg.sta)
        self.cfg.loc = prompt("Location", self.cfg.loc)
        self.cfg.chan = prompt("Channel", self.cfg.chan)
        self.cfg.year = prompt("Year YYYY", self.cfg.year)
        self.cfg.jday = prompt("Julian day DDD", self.cfg.jday)

    def _load_day(self):
        day = UTCDateTime(int(self.cfg.year), julday=int(self.cfg.jday))
        print(f"\n📅 Loading {day.date} ...")


        # Now it is safe
        self.sds = EnhancedSDSClient(self.cfg.sds_root)        

        if not self.sds.has_data_for_browser_day(
            net=self.cfg.net,
            sta=self.cfg.sta,
            loc=self.cfg.loc,
            chan=self.cfg.chan,
            day=day,
        ):


            self.st_full = self.sds.read_day(
                net=self.cfg.net,
                sta=self.cfg.sta,
                loc=self.cfg.loc,
                chan=self.cfg.chan,
                year=self.cfg.year,
                jday=self.cfg.jday,
                merge=1,
            )

    def _init_state(self):
        t0 = min(tr.stats.starttime for tr in self.st_full)
        t1 = max(tr.stats.endtime for tr in self.st_full)
        self.state = dict(
            t0=t0,
            t1=t1,
            win=t0,
            win_sec=self.cfg.window_sec,
            mode=self.cfg.mode,
        )

    # ---------------- navigation ----------------

    def _shift_day(self, delta: int):
        d = UTCDateTime(int(self.cfg.year), julday=int(self.cfg.jday)) + delta * 86400
        self.cfg.year = f"{d.year:04d}"
        self.cfg.jday = f"{d.julday:03d}"
        self._load_day()
        self._init_state()
        self.refresh()

    # ---------------- interaction ----------------

    def _connect_keys(self):
        self.plotter.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, e):
        if e.key == "right":
            self.state["win"] += self.state["win_sec"]
        elif e.key == "left":
            self.state["win"] -= self.state["win_sec"]
        elif e.key == "up":
            self.state["win_sec"] *= 2
        elif e.key == "down":
            self.state["win_sec"] = max(10, self.state["win_sec"] / 2)
        elif e.key == "[":
            self._shift_day(-1)
            return
        elif e.key == "]":
            self._shift_day(1)
            return
        else:
            return
        self.refresh()

    # ---------------- rendering ----------------

    def refresh(self):
        ws = self.state["win"]
        we = ws + self.state["win_sec"]
        st = self._filter(self.st_full).slice(ws, we)

        soh = self.state["mode"].startswith("soh")
        self.last_df = self.stats.compute(st, soh_desc=soh)

        print("\n" + "-" * 80)
        print(f"{ws} → {we} | {self.state['mode']}")
        print(self.last_df.to_string(index=False))

        title = f"{self.cfg.net} {self.cfg.year}:{self.cfg.jday}"
        self.plotter.redraw(st, title)

    def _filter(self, st):
        out = Stream()
        for tr in st:
            ch = tr.stats.channel
            m = self.state["mode"]
            if m == "all":
                out += tr
            elif m == "high" and ChannelClassifier.is_high_rate(ch):
                out += tr
            elif m == "seismic" and ChannelClassifier.is_seismic(ch):
                out += tr
            elif m == "infrasound" and ChannelClassifier.is_infrasound(ch):
                out += tr
            elif m == "soh" and ChannelClassifier.is_soh(tr):
                out += tr
            elif m == "soh_imp" and ChannelClassifier.is_soh(tr) and ch in IMPORTANT_SOH:
                out += tr
        return out

    # ---------------- menu ----------------

    def menu(self):
        while True:
            print("""
Menu:
  a  all channels
  h  high-rate
  s  seismic
  i  infrasound
  o  SOH
  O  important SOH
  [  previous day
  ]  next day
  q  quit
""")
            c = input("Choice: ").strip()
            if c == "q":
                break
            if c == "[":
                self._shift_day(-1)
            elif c == "]":
                self._shift_day(1)
            elif c in {"a", "h", "s", "i", "o", "O"}:
                self.state["mode"] = {
                    "a": "all",
                    "h": "high",
                    "s": "seismic",
                    "i": "infrasound",
                    "o": "soh",
                    "O": "soh_imp",
                }[c]
                self.refresh()

# =============================================================================
# Main
# =============================================================================

def main():
    cfg = BrowserConfig.load()
    SDSBrowser(cfg).run()

if __name__ == "__main__":
    main()
