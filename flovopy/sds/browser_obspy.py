#!/usr/bin/env python3
"""
flovopy.bin.sds_browser
======================

RAW Interactive SDS MiniSEED Browser (ObsPy SDSClient ONLY)

Goals:
- Show waveforms "as-is" from MiniSEED in an SDS archive.
- NO signal processing:
  - no merge
  - no detrend
  - no filter
  - no resample
  - no padding/fill
- Windowing uses Stream.slice() (trim only, no fill).

Features retained:
- Config persistence (~/.config/flovopy/<scriptname>.json)
- Prompts: SDS root, net/sta/loc/chan, year/jday, start HH:MM, window length
- Day load + [ / ] previous/next day navigation
- Persistent interactive Matplotlib window
- Keyboard:
    Left/Right  : scroll window by 1 window
    Up/Down     : increase/decrease window length
    [ / ]       : previous / next day
- Mouse wheel: scroll window by 1 window
- Modes:
    all, high, seismic, infrasound, soh (location D0), soh_imp
- Per-window channel stats:
    p2p, median, std (std about mean)
- SOH summaries (when in SOH modes):
    GNSS lat/lon, clock quality, satellites, temperature
    mean VM1..VM6 table
    optional VM bar plot
- Export:
    PNG of waveform figure
    CSV of current stats table
- Toggle:
    same y-scale across traces

Important note on "raw":
- ObsPy’s SDS client will read MiniSEED records and return Trace objects.
- This script does NOT call Stream.merge() anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import Stream, UTCDateTime
from obspy.clients.filesystem.sds import Client as SDSClient

plt.ion()

# =============================================================================
# Constants / Channel maps
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
    start_hhmm: str = "00:00"
    window_sec: float = 3600.0
    same_y_scale: bool = False
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
                print(f"⚠️ Failed to load config {cfg_file}: {e}")
        return cls()

    def save(self) -> None:
        cfg_file = self.config_path()
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(asdict(self), indent=2))


def prompt(msg: str, default: str = "") -> str:
    val = input(f"{msg} [{default}]: ").strip()
    return val or default


def _parse_year_jday(year: str, jday: str) -> Tuple[int, int]:
    y = int(str(year).strip())
    jd = int(str(jday).strip())
    if not (1 <= jd <= 366):
        raise ValueError("jday must be in 1..366")
    return y, jd


def _parse_hhmm(hhmm: str) -> int:
    """
    Convert 'HH:MM' to seconds since midnight.
    """
    s = (hhmm or "").strip()
    if not s:
        return 0
    try:
        hh, mm = s.split(":")
        h = int(hh)
        m = int(mm)
    except Exception as e:
        raise ValueError("Time must be HH:MM (e.g., 06:30)") from e

    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError("HH must be 0..23 and MM must be 0..59")

    return h * 3600 + m * 60


# =============================================================================
# SDS access: ObsPy SDSClient ONLY (no processing)
# =============================================================================

class RawObsPySDSAccessor:
    """
    Read waveform data from an SDS archive using ObsPy's filesystem SDS Client.

    IMPORTANT:
    - NO merge
    - NO detrend/filter/resample/taper
    - No padding/fill
    """

    def __init__(self, sds_root: str):
        p = Path(sds_root).expanduser()
        if not p.is_dir():
            raise OSError(f"SDS root is not a local directory: {p}")
        self.sds_root = str(p)
        self.client = SDSClient(self.sds_root)
        self.backend = "ObsPySDSClient"

    def read_day(self, year: int, jday: int, net: str, sta: str, loc: str, chan: str) -> Stream:
        t0 = UTCDateTime(year, julday=jday)
        t1 = t0 + 86400
        st = self.client.get_waveforms(
            network=net,
            station=sta,
            location=loc,
            channel=chan,
            starttime=t0,
            endtime=t1,
        )
        # RAW: do not merge, do not split, do not detrend, do not filter
        return st


# =============================================================================
# Classification + stats + SOH summaries
# =============================================================================

class ChannelClassifier:
    @staticmethod
    def is_high_rate(ch: str) -> bool:
        return bool(ch) and ch[0] in HIGH_RATE_FIRST

    @staticmethod
    def is_seismic(ch: str) -> bool:
        return len(ch) == 3 and ch[1] == "H" and ch[2] in "ZNE12"

    @staticmethod
    def is_infrasound(ch: str) -> bool:
        return len(ch) >= 3 and ch[1] == "D"

    @staticmethod
    def is_soh(tr) -> bool:
        # Centaur SOH commonly uses location code D0
        return getattr(tr.stats, "location", "") == "D0"


class StatsComputer:
    """Compute per-trace window statistics and SOH summaries (read-only)."""

    @staticmethod
    def compute_stats(st: Stream, add_soh_description: bool = False) -> pd.DataFrame:
        rows = []
        for tr in st:
            # Raw stats on raw samples
            data = np.asarray(tr.data)
            if data.size == 0:
                continue
            # Use float for numerical stability only (does not change samples in tr.data)
            x = data.astype(np.float64, copy=False)

            row = {
                "NET.STA.LOC.CHAN": tr.id,
                #"min": float(np.min(x)),
                #"max": float(np.max(x)),
                "p2p": int(np.max(x)-np.min(x)),
                #"mean": float(np.mean(x)),
                "median": int(np.median(x)),
                "std": int(np.std(x, ddof=0)),  # std about mean
            }
            if add_soh_description:
                ch = tr.stats.channel.upper()
                row["description"] = SOH_MAP.get(ch, "Unknown SOH Channel")

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("NET.STA.LOC.CHAN").reset_index(drop=True)
        return df

    @staticmethod
    def _median_channel_value(st: Stream, channel: str) -> Optional[float]:
        sel = st.select(channel=channel)
        if not sel:
            return None
        data = np.asarray(sel[0].data)
        if data.size == 0:
            return None
        return float(np.median(data.astype(np.float64, copy=False)))

    @classmethod
    def print_soh_human_lines(cls, st_soh_window: Stream) -> None:
        med_gla = cls._median_channel_value(st_soh_window, "GLA")
        med_glo = cls._median_channel_value(st_soh_window, "GLO")
        if med_gla is not None and med_glo is not None:
            print(f"Median GNSS position (deg): lat={med_gla/1e6:.6f}, lon={med_glo/1e6:.6f}")
        else:
            print("Median GNSS position: (GLA/GLO not present in window)")

        med_lcq = cls._median_channel_value(st_soh_window, "LCQ")
        if med_lcq is not None:
            print(f"Median clock quality (LCQ): {med_lcq:.2f} %")
        else:
            print("Median clock quality: (LCQ not present)")

        med_gns = cls._median_channel_value(st_soh_window, "GNS")
        if med_gns is not None:
            print(f"Median satellites used (GNS): {int(round(med_gns))}")
        else:
            print("Median satellites used: (GNS not present)")

        med_vdt = cls._median_channel_value(st_soh_window, "VDT")
        if med_vdt is not None:
            print(f"Median digitizer temperature (VDT): {med_vdt/100.0:.2f} °C")
        else:
            print("Median digitizer temperature: (VDT not present)")

        vm_traces = []
        for k in ["VM1", "VM2", "VM3", "VM4", "VM5", "VM6"]:
            sel = st_soh_window.select(channel=k)
            if sel:
                vm_traces.append(sel[0])

        if len(vm_traces) >= 6:
            means = [float(np.mean(np.asarray(tr.data).astype(np.float64, copy=False))) for tr in vm_traces[:6]]
            table = np.array(means).reshape(3, 2)
            idx = [tr.stats.channel for tr in vm_traces[:3]]
            df_vm = pd.DataFrame(table, index=idx, columns=["Sensor A", "Sensor B"])
            print("\nMean mass positions (VM*):")
            print(df_vm.to_string(float_format=lambda x: f"{x:10.3f}"))
        else:
            print("\nMean mass positions: (VM1..VM6 not all present)")


# =============================================================================
# Plot + export
# =============================================================================

class PlotManager:
    """Persistent waveform plot manager (read-only)."""

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))

    @staticmethod
    def _format_time_axis(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, alpha=0.3)

    def redraw_waveforms(self, st: Stream, title: str, same_y_scale: bool = False) -> None:
        self.fig.clf()

        n = len(st)
        if n == 0:
            self.fig.suptitle("No channels selected")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            return

        global_min = None
        global_max = None
        if same_y_scale:
            mins = []
            maxs = []
            for tr in st:
                d = np.asarray(tr.data)
                if d.size == 0:
                    continue
                mins.append(np.min(d))
                maxs.append(np.max(d))
            if mins and maxs:
                global_min = float(np.min(mins))
                global_max = float(np.max(maxs))
                if global_min == global_max:
                    pad = 1.0 if global_min == 0 else abs(global_min) * 0.1
                    global_min -= pad
                    global_max += pad

        axes = self.fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]

        for ax, tr in zip(axes, st):
            t = tr.times("matplotlib")
            ax.plot(t, tr.data, linewidth=0.8)

            # Full SEED id label (net.sta.loc.chan)
            ax.set_ylabel(tr.id, rotation=0, labelpad=35, fontsize=8)

            self._format_time_axis(ax)
            if same_y_scale and global_min is not None and global_max is not None:
                ax.set_ylim(global_min, global_max)

        axes[-1].set_xlabel("Time (UTC)")
        self.fig.suptitle(title)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def plot_mean_mass_positions(st_soh_window: Stream) -> None:
        vm = []
        for k in ["VM1", "VM2", "VM3", "VM4", "VM5", "VM6"]:
            sel = st_soh_window.select(channel=k)
            if sel:
                vm.append(sel[0])

        if not vm:
            print("No VM channels found in the current window.")
            return

        means = {tr.stats.channel: float(np.mean(np.asarray(tr.data).astype(np.float64, copy=False))) for tr in vm}

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(list(means.keys()), list(means.values()))
        ax.set_title("Mean Mass Positions (current window)")
        ax.set_ylabel("Counts")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class Exporter:
    @staticmethod
    def safe_prefix(net: str, year: str, jday: str, mode: str, win_start: UTCDateTime, win_end: UTCDateTime) -> str:
        def fmt(t: UTCDateTime) -> str:
            return t.datetime.strftime("%Y%m%dT%H%M%S")
        net_s = (net or "NET").replace("*", "X")
        mode_s = mode.replace("*", "X")
        return f"{net_s}_{year}_{jday}_{mode_s}_{fmt(win_start)}_{fmt(win_end)}"

    @staticmethod
    def save_png(fig, outdir: Path, prefix: str) -> Path:
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{prefix}.png"
        fig.savefig(path, dpi=150)
        return path

    @staticmethod
    def save_csv(df: pd.DataFrame, outdir: Path, prefix: str) -> Path:
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{prefix}.csv"
        df.to_csv(path, index=False)
        return path


# =============================================================================
# Browser
# =============================================================================

class SDSBrowser:
    def __init__(self, cfg: BrowserConfig):
        self.cfg = cfg
        self.sds: Optional[RawObsPySDSAccessor] = None
        self.plot = PlotManager()

        self.st_full = Stream()
        self.st_view = Stream()
        self.stats_df = pd.DataFrame()

        self.day_start: Optional[UTCDateTime] = None
        self.day_end: Optional[UTCDateTime] = None
        self.win: Optional[UTCDateTime] = None
        self.win_sec = float(cfg.window_sec)

        self.plot.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.plot.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def run(self) -> None:
        self._prompt_initial()
        self._load_day()
        self.refresh()
        self.menu()

    def _prompt_initial(self) -> None:
        print("\n--- SDS Browser (RAW / ObsPy SDSClient only) ---")
        self.cfg.sds_root = prompt("SDS root", self.cfg.sds_root)
        self._ensure_accessor()

        self.cfg.net = prompt("Network", self.cfg.net)
        self.cfg.sta = prompt("Station (wildcards ok)", self.cfg.sta)
        self.cfg.loc = prompt("Location (wildcards ok)", self.cfg.loc)
        self.cfg.chan = prompt("Channel (wildcards ok)", self.cfg.chan)
        self.cfg.year = prompt("Year (YYYY)", self.cfg.year)
        self.cfg.jday = prompt("Julian day (1-366)", self.cfg.jday)
        self.cfg.start_hhmm = prompt("Start time (HH:MM)", self.cfg.start_hhmm)

        ws = prompt("Window length (seconds)", str(int(self.cfg.window_sec)))
        try:
            self.cfg.window_sec = float(ws)
        except Exception:
            pass

        md = prompt("Mode (all/high/seismic/infrasound/soh/soh_imp)", self.cfg.mode)
        if md:
            self.cfg.mode = md

        try:
            self.cfg.save()
        except Exception as e:
            print(f"⚠️ Could not save config: {e}")

    def _ensure_accessor(self) -> None:
        while True:
            p = Path(self.cfg.sds_root).expanduser()
            if not p.is_dir():
                print(f"❌ SDS root is not a local directory: {p}")
                self.cfg.sds_root = prompt("SDS root", str(p))
                continue
            try:
                self.sds = RawObsPySDSAccessor(str(p))
                try:
                    self.cfg.save()
                except Exception:
                    pass
                print(f"✅ Using SDS backend: {self.sds.backend} at {p}")
                return
            except Exception as e:
                print(f"❌ Failed to init ObsPy SDSClient at {p}: {e}")
                self.cfg.sds_root = prompt("SDS root", str(p))

    def _load_day(self) -> None:
        if self.sds is None:
            self._ensure_accessor()

        try:
            year, jday = _parse_year_jday(self.cfg.year, self.cfg.jday)
        except Exception as e:
            print(f"❌ Invalid year/jday: {e}")
            self.st_full = Stream()
            self._init_state_empty()
            return

        self.day_start = UTCDateTime(year, julday=jday)
        self.day_end = self.day_start + 86400

        try:
            offset = _parse_hhmm(self.cfg.start_hhmm)
        except Exception as e:
            print(f"⚠️ Invalid start time '{self.cfg.start_hhmm}': {e} (using 00:00)")
            offset = 0
            self.cfg.start_hhmm = "00:00"

        self.win_sec = max(10.0, float(self.cfg.window_sec))
        self.win = self.day_start + offset

        if self.win < self.day_start:
            self.win = self.day_start
        if self.win + self.win_sec > self.day_end:
            self.win = max(self.day_start, self.day_end - self.win_sec)

        try:
            self.st_full = self.sds.read_day(
                year, jday, self.cfg.net, self.cfg.sta, self.cfg.loc, self.cfg.chan
            )
            if len(self.st_full) == 0:
                print("⚠️ Read succeeded but returned an empty Stream (check patterns + jday padding).")
        except Exception as e:
            print(f"⚠️ Failed to read day from SDS ({self.sds.backend}): {e}")
            self.st_full = Stream()

        if len(self.st_full) == 0:
            self._init_state_empty()
        else:
            # Keep user's chosen start time; do NOT reset to earliest trace start.
            # Only clamp to within actual data bounds if needed.
            t0 = min(tr.stats.starttime for tr in self.st_full)
            t1 = max(tr.stats.endtime for tr in self.st_full)

            if self.win < t0:
                self.win = t0
            if self.win + self.win_sec > t1:
                self.win = max(t0, t1 - self.win_sec)

    def _init_state_empty(self) -> None:
        try:
            year, jday = _parse_year_jday(self.cfg.year, self.cfg.jday)
            self.day_start = UTCDateTime(year, julday=jday)
            self.day_end = self.day_start + 86400
            self.win = self.day_start
        except Exception:
            self.day_start = UTCDateTime()
            self.day_end = self.day_start + 86400
            self.win = self.day_start
        self.win_sec = max(10.0, float(self.cfg.window_sec))

    def _filter_stream(self, st: Stream) -> Stream:
        m = (self.cfg.mode or "all").lower()
        out = Stream()
        for tr in st:
            ch = tr.stats.channel.upper()
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

    def refresh(self) -> None:
        if self.win is None:
            self._init_state_empty()

        win_start = self.win
        win_end = self.win + self.win_sec

        if self.day_start is not None and win_start < self.day_start:
            win_start = self.day_start
            win_end = win_start + self.win_sec
        if self.day_end is not None and win_end > self.day_end:
            win_end = self.day_end
            win_start = max(self.day_start, win_end - self.win_sec)

        # Slice only (trim); no merge/pad/fill
        st_mode = self._filter_stream(self.st_full)
        self.st_view = st_mode.slice(win_start, win_end)

        add_desc = (self.cfg.mode.lower() in {"soh", "soh_imp"})
        self.stats_df = StatsComputer.compute_stats(self.st_view, add_soh_description=add_desc)

        title = self._title_line(win_start, win_end)
        print("\n" + "=" * 90)
        print(title)
        if self.stats_df.empty:
            print("(No data in current window after filtering)")
        else:
            print(self.stats_df.to_string(index=False))

        if self.cfg.mode.lower() in {"soh", "soh_imp"}:
            try:
                StatsComputer.print_soh_human_lines(self.st_view)
            except Exception as e:
                print(f"⚠️ SOH summary failed: {e}")

        self.plot.redraw_waveforms(self.st_view, title, same_y_scale=bool(self.cfg.same_y_scale))

        self.cfg.window_sec = float(self.win_sec)
        try:
            self.cfg.save()
        except Exception:
            pass

    def _title_line(self, win_start: UTCDateTime, win_end: UTCDateTime) -> str:
        backend = self.sds.backend if self.sds else "ObsPySDSClient"
        return (
            f"SDS={self.cfg.sds_root} ({backend}) | "
            f"{self.cfg.net}.{self.cfg.sta}.{self.cfg.loc}.{self.cfg.chan} | "
            f"Y{self.cfg.year} J{self.cfg.jday} | "
            f"mode={self.cfg.mode} | "
            f"{win_start.isoformat()} .. {win_end.isoformat()} "
            f"({self.win_sec/60:.1f} min)"
        )

    def _shift_window(self, direction: int) -> None:
        if self.win is None:
            return
        self.win = self.win + direction * self.win_sec
        self.refresh()

    def _change_window_sec(self, factor: float) -> None:
        self.win_sec = max(10.0, float(self.win_sec) * factor)
        self.refresh()

    def _shift_day(self, delta_days: int) -> None:
        try:
            year, jday = _parse_year_jday(self.cfg.year, self.cfg.jday)
        except Exception:
            print("❌ Can't shift day: invalid year/jday.")
            return
        t = UTCDateTime(year, julday=jday) + delta_days * 86400
        self.cfg.year = str(t.year)
        self.cfg.jday = f"{t.julday:03d}"
        self._load_day()
        self.refresh()

    def _on_key(self, event) -> None:
        k = (event.key or "").lower()
        if k in {"left", "right", "up", "down", "[", "]"}:
            if k == "left":
                self._shift_window(-1)
            elif k == "right":
                self._shift_window(+1)
            elif k == "up":
                self._change_window_sec(1.25)
            elif k == "down":
                self._change_window_sec(0.8)
            elif k == "[":
                self._shift_day(-1)
            elif k == "]":
                self._shift_day(+1)

    def _on_scroll(self, event) -> None:
        step = getattr(event, "step", 0)
        if step > 0:
            self._shift_window(-1)
        elif step < 0:
            self._shift_window(+1)

    def menu(self) -> None:
        outdir = Path.cwd() / "sds_browser_exports"

        while True:
            state = "ON" if self.cfg.same_y_scale else "OFF"
            print(
                f"""
Menu:
  Modes:
    a  all channels
    h  high-rate
    s  seismic
    i  infrasound
    o  SOH (location D0)
    O  important SOH (subset)

  Day:
    [  previous day
    ]  next day
    g  goto year/jday
    r  reload current day

  Start hour/minute:
    t  set start time (HH:MM)

  Window:
    w  set window length (seconds)

  Fixed y-scale:
    y  toggle same y-scale (currently: {state})

  Query:
    qy change net/sta/loc/chan (keeps year/jday)
    qs change SDS root

  SOH:
    p  plot mean VM1..VM6 for current window

  Export:
    e  export PNG of waveform figure
    c  export CSV of current stats table

  Other:
    ?  help (reprint menu)
    x  exit
"""
            )
            c = input("Choice: ").strip()

            if c == "x":
                break
            if c == "?":
                continue

            if c == "[":
                self._shift_day(-1)
            elif c == "]":
                self._shift_day(1)
            elif c == "g":
                self.cfg.year = prompt("Year (YYYY)", self.cfg.year)
                self.cfg.jday = prompt("Julian day (1-366)", self.cfg.jday)
                self._load_day()
                self.refresh()
            elif c == "r":
                self._load_day()
                self.refresh()
            elif c == "t":
                self.cfg.start_hhmm = prompt("Start time (HH:MM)", self.cfg.start_hhmm)
                try:
                    offset = _parse_hhmm(self.cfg.start_hhmm)
                    if self.day_start is not None:
                        self.win = self.day_start + offset
                except Exception as e:
                    print(f"⚠️ Invalid HH:MM: {e}")
                self.refresh()
            elif c == "w":
                ws = prompt("Window length (seconds)", str(int(self.win_sec)))
                try:
                    self.win_sec = max(10.0, float(ws))
                except Exception:
                    print("⚠️ Invalid window length; unchanged.")
                self.refresh()

            elif c == "qy":
                self.cfg.net = prompt("Network", self.cfg.net)
                self.cfg.sta = prompt("Station (wildcards ok)", self.cfg.sta)
                self.cfg.loc = prompt("Location (wildcards ok)", self.cfg.loc)
                self.cfg.chan = prompt("Channel (wildcards ok)", self.cfg.chan)
                self._load_day()
                self.refresh()
            elif c == "qs":
                self.cfg.sds_root = prompt("SDS root", self.cfg.sds_root)
                self._ensure_accessor()
                self._load_day()
                self.refresh()

            elif c == "p":
                if self.cfg.mode.lower() not in {"soh", "soh_imp"}:
                    print("⚠️ VM plot is usually meaningful in SOH modes; switching to 'soh' temporarily.")
                try:
                    self.plot.plot_mean_mass_positions(self.st_view)
                except Exception as e:
                    print(f"⚠️ Could not plot VM means: {e}")

            elif c == "y":
                self.cfg.same_y_scale = not bool(self.cfg.same_y_scale)
                print(f"Same y-scale: {'ON' if self.cfg.same_y_scale else 'OFF'}")
                self.refresh()

            elif c in {"a", "h", "s", "i", "o", "O"}:
                self.cfg.mode = {
                    "a": "all",
                    "h": "high",
                    "s": "seismic",
                    "i": "infrasound",
                    "o": "soh",
                    "O": "soh_imp",
                }[c]
                self.refresh()

            elif c == "e":
                if self.win is None:
                    self._init_state_empty()
                prefix = Exporter.safe_prefix(
                    self.cfg.net, self.cfg.year, self.cfg.jday, self.cfg.mode,
                    self.win, self.win + self.win_sec
                )
                try:
                    path = Exporter.save_png(self.plot.fig, outdir, prefix)
                    print(f"✅ Saved: {path}")
                except Exception as e:
                    print(f"⚠️ Export PNG failed: {e}")

            elif c == "c":
                if self.win is None:
                    self._init_state_empty()
                prefix = Exporter.safe_prefix(
                    self.cfg.net, self.cfg.year, self.cfg.jday, self.cfg.mode,
                    self.win, self.win + self.win_sec
                )
                try:
                    path = Exporter.save_csv(self.stats_df, outdir, prefix)
                    print(f"✅ Saved: {path}")
                except Exception as e:
                    print(f"⚠️ Export CSV failed: {e}")

            else:
                print("Unknown choice. Type '?' for help.")


def main() -> None:
    cfg = BrowserConfig.load()
    SDSBrowser(cfg).run()


if __name__ == "__main__":
    main()
