import os
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime

TOPDIR = '/data/b18_waveform_processing'
SAVE_INTERVAL = 1000  # Save every 1000 events now

# Paths for periodic save files
timestamp = int(UTCDateTime().timestamp)
trace_outpath_base = '/home/thompsong/Dropbox/Trace_Level_ML_ME_metrics'
event_outpath_base = '/home/thompsong/Dropbox/Event_Level_ML_ME_summary'

trace_outpath_pkl = f'{trace_outpath_base}.pkl'
event_outpath_pkl = f'{event_outpath_base}.pkl'
trace_outpath_pkl_final = f'{trace_outpath_base}_final.pkl'
event_outpath_pkl_final = f'{event_outpath_base}_final.pkl'

# Try to load existing partial results
if os.path.isfile(event_outpath_pkl_final):
    df_events = pd.read_pickle(event_outpath_pkl)
    
# Parse event_time as pandas datetime if needed
if 'event_time' in df_events.columns:
    df_events['event_time'] = pd.to_datetime(df_events['event_time'], errors='coerce')

print(df_events.columns, len(df_events))

# --- Plot Event-level ML and ME versus Time ---
if not df_events.empty:
    fig, ax = plt.subplots(figsize=(14, 7))

    if 'ML_median' in df_events.columns:
        ax.plot(df_events['event_time'], df_events['ML_median'], label='Median ML', marker='o', linestyle='-', alpha=0.7)
    if 'ME_median' in df_events.columns:
        ax.plot(df_events['event_time'], df_events['ME_median'], label='Median ME', marker='s', linestyle='-', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Magnitude')
    ax.set_title('Event-Level ML and ME (Median) versus Time')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()

    plot_outpath = '/home/thompsong/Dropbox/Event_Level_ML_ME_vs_time_plot.png'
    plt.savefig(plot_outpath)
    plt.show()
    print(f"[DONE] Saved event-level plot to {plot_outpath}")