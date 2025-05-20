import sqlite3
from obspy import read
from obspy.core import UTCDateTime
from obspy.signal.cross_correlation import correlate, xcorr_max
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flovopy.config_projects import get_config

DB_PATH = "mvo_seisan_index.db"
OUTPUT_TABLE = "timing_lag_analysis"
MAX_SHIFT_SECONDS = 5.0

def round_sr(st):
    for tr in st:
        tr.stats.sampling_rate = round(tr.stats.sampling_rate)
    return st

def run_analysis():
    config = get_config()
    dbfile = config['mvo_seisan_index_db']
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    # Create the output table if not exists
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
            event_path TEXT,
            continuous_path TEXT,
            trace_id TEXT,
            sampling_rate REAL,
            lag_sec REAL,
            cc_max REAL,
            event_start TEXT,
            continuous_start TEXT
        );
    """)
    conn.commit()

    # Load file metadata
    events = pd.read_sql_query("SELECT path, start_time, end_time FROM wav_files WHERE file_type='event'", conn)
    continuous = pd.read_sql_query("SELECT path, start_time, end_time FROM wav_files WHERE file_type='continuous'", conn)

    for _, evt in events.iterrows():
        evt_start = UTCDateTime(evt['start_time'])
        evt_end = UTCDateTime(evt['end_time'])

        overlaps = continuous[
            (continuous['start_time'] <= evt_end.isoformat()) &
            (continuous['end_time'] >= evt_start.isoformat())
        ]

        for _, cont in overlaps.iterrows():
            try:
                st_evt = round_sr(read(evt['path']))
                st_cont = round_sr(read(cont['path']))
            except Exception as e:
                print(f"Read error: {e}")
                continue

            for tr_evt in st_evt:
                tr_id = tr_evt.id
                tr_match = st_cont.select(id=tr_id)
                if not tr_match:
                    continue
                tr_cont = tr_match[0]

                sr = tr_evt.stats.sampling_rate
                max_shift = int(MAX_SHIFT_SECONDS * sr)

                t0 = max(tr_evt.stats.starttime, tr_cont.stats.starttime)
                t1 = min(tr_evt.stats.endtime, tr_cont.stats.endtime)
                if t1 - t0 < 1.0:
                    continue

                te = tr_evt.copy().trim(t0, t1, pad=True, fill_value=0)
                tc = tr_cont.copy().trim(t0, t1, pad=True, fill_value=0)
                min_len = min(len(te.data), len(tc.data))
                te.data = te.data[:min_len]
                tc.data = tc.data[:min_len]

                try:
                    cc = correlate(tc.data, te.data, max_shift)
                    shift_idx, cc_max = xcorr_max(cc)
                    lag_sec = shift_idx / sr
                except Exception as e:
                    print(f"Cross-correlation error: {e}")
                    continue

                cursor.execute(f"""
                    INSERT INTO {OUTPUT_TABLE} (
                        event_path, continuous_path, trace_id, sampling_rate,
                        lag_sec, cc_max, event_start, continuous_start
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evt['path'], cont['path'], tr_id, sr,
                    lag_sec, cc_max, evt_start.isoformat(), UTCDateTime(cont['start_time']).isoformat()
                ))
                conn.commit()

    conn.close()

def plot_lag_histogram():
    config = get_config()
    dbfile = config['mvo_seisan_index_db']
    df = pd.read_sql(f"SELECT * FROM {OUTPUT_TABLE} WHERE cc_max >= 0.9", sqlite3.connect(dbfile))

    if df.empty:
        print("No data with cc_max ≥ 0.9 found.")
        return

    df['year'] = pd.to_datetime(df['event_start']).dt.year

    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='lag_sec', hue='year', bins=100, kde=True, multiple='stack')
    plt.title("Lag (sec) between Event and Continuous Files (cc_max ≥ 0.9)")
    plt.xlabel("Lag (seconds)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_analysis()
    plot_lag_histogram()