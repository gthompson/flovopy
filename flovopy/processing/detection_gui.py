from __future__ import annotations

"""
flovopy.processing.detection_gui

Interactive / GUI helpers for manually selecting traces, picking event start/end
times, and tuning network trigger parameters.

This module is intentionally separate from flovopy.processing.detection so that
the core detection workflow remains lightweight and usable in scripts, batch
processing, and headless environments.
"""

import random
import tkinter as tk
from tkinter import messagebox

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from obspy import Stream, UTCDateTime

from flovopy.processing.detection_old import detect_network_event


class SeismicGUI:
    """
    Simple Tkinter-based GUI for:
    1. selecting traces from a Stream
    2. manually picking event start and end times
    """

    def __init__(self, master, stream: Stream):
        self.master = master
        self.master.title("Seismic Event Detection")

        self.stream = stream
        self.selected_traces = stream.copy()
        self.picked_times = []
        self.mode = "select_traces"

        self.frame_left = tk.Frame(master)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.vars = []
        self.checkboxes = []
        for tr in stream:
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(self.frame_left, text=tr.id, variable=var)
            chk.pack(anchor="w")
            self.checkboxes.append(chk)
            self.vars.append(var)

        self.fig, self.axs = plt.subplots(len(stream), 1, figsize=(10, 6), sharex=True)
        plt.subplots_adjust(hspace=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.button = tk.Button(self.frame_left, text="Select Traces", command=self.select_traces)
        self.button.pack(pady=10)

        self.cursor_lines = []
        self.plot_traces()

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_click)

    def plot_traces(self):
        """
        Plot traces in linked subplots.
        """
        for ax in self.axs:
            ax.clear()

        start_time = min(tr.stats.starttime for tr in self.stream)
        end_time = max(tr.stats.endtime for tr in self.stream)

        start_time_num = mdates.date2num(start_time.datetime)
        end_time_num = mdates.date2num(end_time.datetime)

        traces_to_plot = self.stream if self.mode == "select_traces" else self.selected_traces

        if self.mode == "select_traces":
            self.fig.suptitle("Select traces using checkboxes, then click 'Select Traces'")
        else:
            self.fig.suptitle("Click to select event start and end times")

        for i, tr in enumerate(traces_to_plot):
            absolute_times = mdates.date2num([tr.stats.starttime + t for t in tr.times()])
            maxabs = np.max(np.abs(tr.data)) if len(tr.data) else 0.0
            data_norm = tr.data / maxabs if maxabs != 0 else tr.data

            self.axs[i].plot(absolute_times, data_norm, label=tr.id)
            self.axs[i].legend(loc="upper right")
            self.axs[i].set_ylabel(tr.id)

        self.axs[-1].set_xlabel("Time (UTC)")
        self.axs[0].set_xlim(start_time_num, end_time_num)

        if self.mode == "pick_times":
            self.cursor_lines = [
                ax.axvline(x=start_time_num, color="r", linestyle="dotted", lw=1)
                for ax in self.axs[: len(traces_to_plot)]
            ]

        for ax in self.axs[: len(traces_to_plot)]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))

        self.canvas.draw()

    def select_traces(self):
        """
        Handle trace selection and switch to picking mode.
        """
        selected = [self.stream[i] for i, var in enumerate(self.vars) if var.get()]

        if not selected:
            messagebox.showerror("Error", "Please select at least one trace.")
            return

        self.selected_traces = Stream(traces=selected)
        self.mode = "pick_times"
        self.button.config(text="Pick Start and End Times", state=tk.DISABLED)
        self.plot_traces()

    def on_mouse_move(self, event):
        """
        Move vertical cursor across all subplots while picking.
        """
        if self.mode != "pick_times" or event.xdata is None:
            return

        for cursor_line in self.cursor_lines:
            cursor_line.set_xdata([event.xdata])

        self.canvas.draw()

    def on_mouse_click(self, event):
        """
        Record event start/end picks.
        """
        if self.mode != "pick_times" or event.xdata is None:
            return

        picked_time = UTCDateTime(mdates.num2date(event.xdata))

        for ax in self.axs[: len(self.selected_traces)]:
            ax.axvline(x=event.xdata, color="r", linestyle="solid", lw=2)

        self.picked_times.append(picked_time)
        print(f"Picked event time: {picked_time}")

        if len(self.picked_times) == 1:
            self.fig.suptitle("Click to select event end time")
        elif len(self.picked_times) == 2:
            plt.close(self.fig)
            plt.close("all")
            self.master.quit()

        self.canvas.draw()


def run_monte_carlo(
    stream: Stream,
    event_start: UTCDateTime,
    event_end: UTCDateTime,
    n_trials: int = 5000,
    verbose: bool = False,
    max_allowed_misfit: float = 4.0,
):
    """
    Search over trigger parameters to find a detection matching manually picked
    event start/end times.
    """
    print("Finding best autodetect parameters by running Monte Carlo simulations...")

    algorithms = ["recstalta", "classicstalta", "zdetect", "carlstatrig", "delayedstalta"]

    best_params = None
    trials_complete = 0
    lod = []
    minchans = len(stream)
    best_misfit = np.inf
    stream_duration = stream[0].stats.endtime - stream[0].stats.starttime

    while trials_complete < n_trials and best_misfit > max_allowed_misfit:
        algorithm = random.choice(algorithms)
        sta = random.uniform(0.1, 5.0)
        sta = min(sta, stream_duration / 20.0)

        lta = random.uniform(sta * 4.0, sta * 25.0)
        lta = min(lta, stream_duration / 2.0)

        thr_on = random.uniform(1.5, 5.0)
        thr_off = random.uniform(thr_on / 10.0, thr_on / 2.0)

        if verbose:
            print(
                f"Trying {algorithm} with "
                f"STA={sta:.3f}, LTA={lta:.3f}, "
                f"THR_ON={thr_on:.3f}, THR_OFF={thr_off:.3f}"
            )

        best_trig = detect_network_event(
            stream,
            minchans=minchans,
            threshon=thr_on,
            threshoff=thr_off,
            sta=sta,
            lta=lta,
            pad=0.0,
            best_only=True,
            verbose=False,
            freq=None,
            algorithm=algorithm,
            criterion="cft",
        )

        if not best_trig:
            if verbose:
                print("No triggers found.")
            continue

        trials_complete += 1
        best_trig["algorithm"] = algorithm
        best_trig["sta"] = sta
        best_trig["lta"] = lta
        best_trig["thr_on"] = thr_on
        best_trig["thr_off"] = thr_off
        best_trig["endtime"] = best_trig["time"] + best_trig["duration"]
        best_trig["misfit"] = abs(best_trig["time"] - event_start) + abs(best_trig["endtime"] - event_end)
        best_trig["event_start"] = event_start
        best_trig["event_end"] = event_end
        lod.append(best_trig)

        if trials_complete % 10 == 0:
            df = pd.DataFrame(lod)
            best_params = df.loc[df["misfit"].idxmin()].to_dict()
            best_misfit = best_params["misfit"]
            print(f"{trials_complete} of {n_trials} trials complete, lowest misfit is {best_misfit}")

    df = pd.DataFrame(lod)
    if len(df) == 0:
        return df, None

    best_params = df.loc[df["misfit"].idxmin()].to_dict()
    return df, best_params


def run_event_detection(stream: Stream, n_trials: int = 50):
    """
    Launch the GUI, let the user pick an event, then run Monte Carlo trigger
    parameter tuning against those picks.

    Returns
    -------
    tuple
        (selected_stream, best_params, dataframe)
    """
    root = tk.Tk()
    app = SeismicGUI(root, stream)
    root.mainloop()

    if len(app.picked_times) != 2:
        print("No valid event times selected.")
        try:
            root.withdraw()
            root.quit()
            root.destroy()
        except Exception:
            pass
        return None, None, None

    event_start, event_end = app.picked_times
    out_stream = Stream(traces=app.selected_traces)

    plt.close("all")
    root.withdraw()
    root.quit()
    root.destroy()

    df, best_params = run_monte_carlo(out_stream, event_start, event_end, n_trials=n_trials)
    return out_stream, best_params, df