import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import pickle

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
if os.path.isfile(trace_outpath_pkl) and os.path.isfile(event_outpath_pkl):
    print("[INFO] Loading existing autosave pickle files...")
    df_traces = pd.read_pickle(trace_outpath_pkl)
    df_events = pd.read_pickle(event_outpath_pkl)
    existing_event_times = set(pd.to_datetime(df_events['event_time']).astype(str))
    print(f"[INFO] Loaded {len(df_events)} events and {len(df_traces)} traces so far.")
    startrows = len(df_events)
else:
    print("[INFO] No existing autosave found. Starting fresh...")
    exit()
    df_traces = pd.DataFrame()
    df_events = pd.DataFrame()
    existing_event_times = set()

# Lists to collect new data
trace_rows = []
event_rows = []
newrowcounter = 0

# --- Walk through all MVO folders ---
filecounter = 0
for root, dirs, files in os.walk(TOPDIR):
    dirs = sorted(dirs)
    for d in dirs:
        if 'MVO' in d:
            filecounter += 1
            if filecounter % 100 == 0:
                print(filecounter)
            thisdir = os.path.join(root, d)
            
            try:
                magfile = os.path.join(thisdir, 'magnitudes.csv')
                streamfile = os.path.join(thisdir, 'stream_metrics.csv')
                drfile = os.path.join(thisdir, 'database_row.csv')

                mag_exists = os.path.isfile(magfile)
                stream_exists = os.path.isfile(streamfile)
                dr_exists = os.path.isfile(drfile)

                if mag_exists and dr_exists:
                    mag = pd.read_csv(magfile)
                    dr = pd.read_csv(drfile)

                    if not mag.empty and not dr.empty:
                        event_info = dr.iloc[0]
                        event_time = pd.to_datetime(event_info['time'])
                        public_id = event_info.get('public_id', None)
                        mainclass = event_info.get('mainclass', None)
                        subclass = event_info.get('subclass', None)

                        # Check if this event already processed
                        if str(event_time) in existing_event_times:
                            print('_', end='')
                            continue  # Skip already processed
                        print('*', end='')

                        if stream_exists:
                            stream = pd.read_csv(streamfile)
                            if not stream.empty:
                                merged = pd.merge(mag, stream, on='id', suffixes=('_mag', '_metrics'))
                                if merged.empty:
                                    merged = mag.copy()
                        else:
                            merged = mag.copy()

                        merged['event_time'] = event_time
                        merged['public_id'] = public_id
                        merged['mainclass'] = mainclass
                        merged['subclass'] = subclass

                        trace_rows.extend(merged.to_dict(orient='records'))

                        event_row = {
                            'event_time': event_time,
                            'public_id': public_id,
                            'mainclass': mainclass,
                            'subclass': subclass,
                            'ML_median': mag['ML'].median() if 'ML' in mag.columns else None,
                            'ME_median': mag['ME'].median() if 'ME' in mag.columns else None,
                            'num_traces': len(mag)
                        }
                        event_rows.append(event_row)
                        newrowcounter += 1

                        # --- Periodic autosave ---
                        if newrowcounter % SAVE_INTERVAL == 0:
                            print('S', end='')
                            #print(f"[INFO] Processed {counter} new events. Autosaving...")

                            # Combine with previous
                            df_traces = pd.concat([df_traces, pd.DataFrame(trace_rows)], ignore_index=True)
                            df_events = pd.concat([df_events, pd.DataFrame(event_rows)], ignore_index=True)

                            # Save to pickle
                            df_traces.to_pickle(trace_outpath_pkl)
                            df_events.to_pickle(event_outpath_pkl)

                            # Reset incremental rows
                            trace_rows = []
                            event_rows = []
                        else:
                            print('*', end='')

            except Exception as e:
                print(f"[ERROR] Problem processing {thisdir}: {e}")

# --- Final Save ---
if trace_rows:
    df_traces = pd.concat([df_traces, pd.DataFrame(trace_rows)], ignore_index=True)
if event_rows:
    df_events = pd.concat([df_events, pd.DataFrame(event_rows)], ignore_index=True)

# Save final pickles
df_traces.to_pickle(trace_outpath_pkl_final)
df_events.to_pickle(event_outpath_pkl_final)

print(f"[DONE] Saved final trace-level dataframe to {trace_outpath_pkl_final}")
print(f"[DONE] Saved final event-level dataframe to {event_outpath_pkl_final}")

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
