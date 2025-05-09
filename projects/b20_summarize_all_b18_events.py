import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime

TOPDIR = '/data/b18_waveform_processing'
SAVE_INTERVAL = 50  # Save every 50 events

# Lists to collect data
trace_rows = []
event_rows = []
counter = 0

timestamp = int(UTCDateTime().timestamp)
trace_outpath = f'/home/thompsong/Dropbox/Trace_Level_ML_ME_metrics_{timestamp}.csv'
event_outpath = f'/home/thompsong/Dropbox/Event_Level_ML_ME_summary_{timestamp}.csv'

# --- Walk through all MVO folders ---
for root, dirs, files in os.walk(TOPDIR):
    #files = sorted(files)
    dirs = sorted(dirs)
    for d in dirs:
        if 'MVO' in d:
            thisdir = os.path.join(root, d)

            try:
                print(f"\n[CHECKING] {thisdir}")

                # Expected file paths
                magfile = os.path.join(thisdir, 'magnitudes.csv')
                streamfile = os.path.join(thisdir, 'stream_metrics.csv')
                drfile = os.path.join(thisdir, 'database_row.csv')

                # Check if files exist
                mag_exists = os.path.isfile(magfile)
                stream_exists = os.path.isfile(streamfile)
                dr_exists = os.path.isfile(drfile)

                print(f"    magnitudes.csv exists? {mag_exists}")
                print(f"    stream_metrics.csv exists? {stream_exists}")
                print(f"    database_row.csv exists? {dr_exists}")

                if mag_exists and dr_exists:
                    mag = pd.read_csv(magfile)
                    dr = pd.read_csv(drfile)

                    if not mag.empty and not dr.empty:
                        event_info = dr.iloc[0]
                        event_time = pd.to_datetime(event_info['time'])
                        public_id = event_info.get('public_id', None)
                        mainclass = event_info.get('mainclass', None)
                        subclass = event_info.get('subclass', None)

                        # Try to merge if stream_metrics exists
                        if stream_exists:
                            stream = pd.read_csv(streamfile)

                            if not stream.empty:
                                merged = pd.merge(mag, stream, on='id', suffixes=('_mag', '_metrics'))

                                if merged.empty:
                                    print(f"    [WARNING] Merge failed (no matching ids?) at {thisdir}")
                                    # fallback: just use magnitudes.csv alone
                                    merged = mag.copy()
                                else:
                                    print(f"    [INFO] Merged {len(merged)} traces for event.")
                            else:
                                print(f"    [WARNING] stream_metrics.csv empty at {thisdir}")
                                merged = mag.copy()
                        else:
                            print(f"    [WARNING] No stream_metrics.csv at {thisdir}")
                            merged = mag.copy()

                        # Add event-level info to every trace
                        merged['event_time'] = event_time
                        merged['public_id'] = public_id
                        merged['mainclass'] = mainclass
                        merged['subclass'] = subclass

                        trace_rows.extend(merged.to_dict(orient='records'))

                        # Summarize event
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

                        counter += 1

                        # --- Periodic autosave ---
                        if counter % SAVE_INTERVAL == 0:
                            print(f"[INFO] Processed {counter} events. Autosaving...")
                            df_traces = pd.DataFrame(trace_rows)
                            df_events = pd.DataFrame(event_rows)

                            if not df_traces.empty and 'event_time' in df_traces.columns:
                                df_traces = df_traces.sort_values('event_time')
                            if not df_events.empty and 'event_time' in df_events.columns:
                                df_events = df_events.sort_values('event_time')

                            # Save with "-autosave"
                            df_traces.to_csv(trace_outpath.replace('.csv', '-autosave.csv'), index=False)
                            df_events.to_csv(event_outpath.replace('.csv', '-autosave.csv'), index=False)
            except Exception as e:
                print(f"[ERROR] Problem processing {thisdir}: {e}")

# --- Final Save ---
df_traces = pd.DataFrame(trace_rows)
df_events = pd.DataFrame(event_rows)

if not df_traces.empty and 'event_time' in df_traces.columns:
    df_traces = df_traces.sort_values('event_time')
if not df_events.empty and 'event_time' in df_events.columns:
    df_events = df_events.sort_values('event_time')

df_traces.to_csv(trace_outpath, index=False)
df_events.to_csv(event_outpath, index=False)

print(f"[DONE] Saved final trace-level dataframe to {trace_outpath}")
print(f"[DONE] Saved final event-level dataframe to {event_outpath}")

# --- Plot Event-level ML and ME versus Time ---
if not df_events.empty:
    fig, ax = plt.subplots(figsize=(14,7))

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
