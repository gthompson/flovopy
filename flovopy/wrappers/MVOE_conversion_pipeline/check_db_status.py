import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

TABLES = [
    "wav_files", "aef_files", "aef_metrics",
    "sfile_wav_map", "processing_log", "trace_id_corrections",
    "sfiles_mvo", "sfiles_bgs", "wav_processing_errors"
]

START_DATE = pd.Timestamp("1996-10-01", tz='UTC')
END_DATE = pd.Timestamp("2008-09-01", tz='UTC')


def query_table(conn, table_name, limit=10):
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        print(f"\n--- {table_name.upper()} (showing up to {limit}) ---")
        print(df)
    except Exception as e:
        print(f"[ERROR] Could not query table {table_name}: {e}")


def count_table_rows(conn, table_name):
    try:
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        print(f"{table_name}: {result[0]} rows")
    except Exception as e:
        print(f"[ERROR] Count failed for {table_name}: {e}")


def save_time_series_plot(data, title, ylabel, out_prefix, is_mean=False):
    for freq, label in [('D', 'daily'), ('W', 'weekly'), ('MS', 'monthly')]:
        if is_mean:
            df_resampled = data.resample(freq).mean()
        else:
            df_resampled = data.resample(freq).sum()
        df_resampled.plot(title=f"{title} ({label})", figsize=(12, 4))
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{label}.png")
        df_resampled.to_csv(f"{out_prefix}_{label}.csv")
        plt.close()


def save_station_heatmap(df_corr, outdir):
    try:
        df_corr = df_corr.copy()
        df_corr['station'] = df_corr['corrected_id'].str.split('.').str[1]
        df_corr['date'] = pd.to_datetime(df_corr['date'], utc=True)
        df_corr = df_corr[(df_corr['date'] >= START_DATE) & (df_corr['date'] <= END_DATE)]
        pivot_df = df_corr.groupby(['date', 'station']).size().unstack(fill_value=0)
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_df.T, cmap='viridis', cbar_kws={'label': 'Daily Trace Count'})
        plt.title("Station Participation Heatmap")
        plt.xlabel("Date")
        plt.ylabel("Station")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "station_participation_heatmap.png"))
        pivot_df.to_csv(os.path.join(outdir, "station_participation_heatmap.csv"))
        plt.close()
    except Exception as e:
        print(f"[ERROR] Could not create station participation heatmap: {e}")


def show_status_summary(db_path):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    print("\n=== Table Row Counts ===")
    for table in TABLES:
        count_table_rows(conn, table)

    print("\n=== Sample Records ===")
    for table in TABLES:
        query_table(conn, table, limit=5)

    outdir = os.path.dirname(db_path)

    # (1) Aggregate WAV processing errors
    try:
        df_errors = pd.read_sql_query("SELECT error_message FROM wav_processing_errors", conn)
        error_counts = df_errors['error_message'].value_counts()
        print("\n--- WAV PROCESSING ERRORS SUMMARY ---")
        print(error_counts)
        error_counts.to_csv(os.path.join(outdir, "wav_processing_errors_summary.csv"))
    except Exception as e:
        print(f"[ERROR] Could not summarize wav_processing_errors: {e}")

    # (2) WAV files per time unit
    try:
        df_wav = pd.read_sql_query("SELECT start_time FROM wav_files", conn)
        df_wav['datetime'] = pd.to_datetime(df_wav['start_time'], utc=True)
        df_wav = df_wav[(df_wav['datetime'] >= START_DATE) & (df_wav['datetime'] <= END_DATE)]
        df_wav['count'] = 1
        df_wav.set_index('datetime', inplace=True)
        save_time_series_plot(df_wav[['count']], "WAV Files", "File Count", os.path.join(outdir, "wav_files"))
    except Exception as e:
        print(f"[ERROR] Could not plot wav_files time series: {e}")

    # (3) Corrected IDs per time unit
    try:
        df_corr = pd.read_sql_query("SELECT date, corrected_id FROM trace_id_corrections", conn)
        df_corr['date'] = pd.to_datetime(df_corr['date'], utc=True)
        df_corr = df_corr[(df_corr['date'] >= START_DATE) & (df_corr['date'] <= END_DATE)]
        df_corr.set_index('date', inplace=True)
        ids_per_day = df_corr.groupby('date')['corrected_id'].nunique()
        save_time_series_plot(ids_per_day.to_frame(name='unique_ids'), "Corrected Trace IDs", "Unique Corrected IDs", os.path.join(outdir, "corrected_ids"), is_mean=True)
    except Exception as e:
        print(f"[ERROR] Could not plot corrected_id time series: {e}")

    # (4) Stations per time unit
    try:
        df_corr = df_corr.copy()
        df_corr['station'] = df_corr['corrected_id'].str.split('.').str[1]
        stations_per_day = df_corr.groupby('date')['station'].nunique()
        save_time_series_plot(stations_per_day.to_frame(name='station_count'), "Station Count", "Unique Stations", os.path.join(outdir, "station_count"), is_mean=True)
        save_station_heatmap(df_corr, outdir)
    except Exception as e:
        print(f"[ERROR] Could not plot station count time series: {e}")

    # (5) Comparison: WAV, AEF, S-files per day/week/month
    try:
        df_wav = pd.read_sql_query("SELECT parsed_time FROM wav_files", conn)
        df_aef = pd.read_sql_query("SELECT parsed_time FROM aef_files", conn)
        df_sfiles = pd.read_sql_query("SELECT parsed_time FROM sfiles_mvo", conn)

        for df, label in zip([df_wav, df_aef, df_sfiles], ['wav', 'aef', 'sfiles']):
            df['datetime'] = pd.to_datetime(df['parsed_time'], utc=True)
            df.dropna(subset=['datetime'], inplace=True)
            df = df[(df['datetime'] >= START_DATE) & (df['datetime'] <= END_DATE)]
            df['count'] = 1
            df.set_index('datetime', inplace=True)
            df_grouped = df.resample('D').sum()
            df_grouped.rename(columns={'count': f'{label}_count'}, inplace=True)
            if 'combined' not in locals():
                combined = df_grouped
            else:
                combined = combined.join(df_grouped, how='outer')

        combined.fillna(0, inplace=True)
        for freq, label in [('D', 'daily'), ('W', 'weekly'), ('MS', 'monthly')]:
            summary = combined.resample(freq).sum()
            summary.plot(figsize=(14, 5), title=f"Files per {label.capitalize()}")
            plt.xlabel("Date")
            plt.ylabel("Count")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"files_summary_{label}.png"))
            summary.to_csv(os.path.join(outdir, f"files_summary_{label}.csv"))
            plt.close()

    except Exception as e:
        print(f"[ERROR] Could not create files summary plots: {e}")

    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check the status of the MVOE index database")
    parser.add_argument("index_db", help="Path to the SQLite database")
    args = parser.parse_args()
    show_status_summary(args.index_db)
