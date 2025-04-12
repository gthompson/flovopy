import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import subprocess

# Optional: Disable MathJax to suppress warnings
try:
    if pio.kaleido.scope:
        pio.kaleido.scope.mathjax = None
except Exception:
    pass

TABLES = [
    "wav_files", "aef_files", "aef_metrics",
    "sfile_wav_map", "processing_log", "trace_id_corrections",
    "sfiles", "wav_processing_errors", "aef_processing_errors", "sfile_processing_errors"
]

START_DATE = pd.Timestamp("1996-10-01", tz='UTC')
END_DATE = pd.Timestamp("2008-09-01", tz='UTC')

def run_shell_count(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return int(result.stdout.strip())
    except Exception as e:
        print(f"[ERROR] Could not run command: {cmd}\n{e}")
        return -1

def add_filesystem_summary():
    print("\n=== Filesystem Summary ===")

    dsn_wav_count = run_shell_count("find /data/SEISAN_DB/WAV/MVOE_/[12]* -type f -iname '*MVO*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    dsn_aef_count = run_shell_count("find /data/SEISAN_DB/AEF/MVOE_/[12]* -type f -iname '*MVO*.aef' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    dsn_local_count = run_shell_count("find /data/SEISAN_DB/REA/MVOE_/[12]* -type f -iname '*L.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    dsn_regional_count = run_shell_count("find /data/SEISAN_DB/REA/MVOE_/[12]* -type f -iname '*R.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    dsn_teleseismic_count = run_shell_count("find /data/SEISAN_DB/REA/MVOE_/[12]* -type f -iname '*D.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    spn_merged_count = run_shell_count("find /data/SEISAN_DB/WAV/MVOE_/[12]* -type f -iname '*SPN*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    asn_wav_count = run_shell_count("find /data/SEISAN_DB/WAV/ASNE_/[12]* -type f -iname '*ASN*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    asn_rea_count = run_shell_count("find /data/SEISAN_DB/REA/ASNE_/[12]* -type f -iname '*L.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    print("\n--- DSN (MVOE_) ---")
    print(f"Number of MVO (DSN) event WAV files: {dsn_wav_count}")
    print(f"Number of MVO (DSN) event AEF files: {dsn_aef_count}")
    print(f"Number of REA files by type:\n- Local: {dsn_local_count}\n- Regional: {dsn_regional_count}\n- Teleseismic: {dsn_teleseismic_count}\n- TOTAL {dsn_local_count + dsn_regional_count + dsn_teleseismic_count}")

    print("\n--- ASN ---")
    print(f"Number of SPN (ASN) event WAV files in merged MVOE_ db: {spn_merged_count}")
    print(f"Number of ASN event WAV files: {asn_wav_count}")
    print(f"Number of ASN REA files: {asn_rea_count}")

def save_plotly_timeseries(df, x_col, y_col, title, out_prefix, freq, is_mean=False):
    df = df.copy()
    df[x_col] = pd.to_datetime(df[x_col], utc=True)
    df = df[(df[x_col] >= START_DATE) & (df[x_col] <= END_DATE)]
    df.set_index(x_col, inplace=True)
    if is_mean:
        df_resampled = df.resample(freq).mean()
    else:
        df_resampled = df.resample(freq).sum()
    df_resampled.reset_index(inplace=True)

    csv_path = f"{out_prefix}_{freq}.csv"
    png_path = f"{out_prefix}_{freq}.png"
    html_path = f"{out_prefix}_{freq}.html"
    df_resampled.to_csv(csv_path, index=False)

    fig = px.line(df_resampled, x=x_col, y=y_col, title=f"{title} ({freq})")
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, engine="kaleido")
    except Exception as e:
        print(f"[ERROR] Failed to write image for {title} ({freq}): {e}")

def generate_error_summary(df_errors, outdir):
    if 'error_message' in df_errors.columns:
        counts = df_errors['error_message'].value_counts()
        print("\n--- WAV PROCESSING ERRORS SUMMARY ---")
        print(counts)
        counts.to_csv(os.path.join(outdir, "wav_processing_errors_summary.csv"))
        counts_df = counts.reset_index()
        counts_df.columns = ['error_message', 'count']
        fig = px.bar(counts_df, x='error_message', y='count',
             title="Top WAV Processing Errors",
             labels={'error_message': 'Error Message', 'count': 'Count'})

        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(os.path.join(outdir, "wav_processing_errors_summary.html"))
        try:
            fig.write_image(os.path.join(outdir, "wav_processing_errors_summary.png"), engine="kaleido")
        except Exception as e:
            print(f"[ERROR] Could not save error plot: {e}")

def show_status_summary(db_path):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return

    add_filesystem_summary()

    conn = sqlite3.connect(db_path)
    outdir = os.path.dirname(db_path)

    # (1) WAV processing errors summary
    try:
        df_errors = pd.read_sql_query("SELECT error_message FROM wav_processing_errors", conn)
        generate_error_summary(df_errors, outdir)
    except Exception as e:
        print(f"[ERROR] Could not summarize wav_processing_errors: {e}")

    # (2) WAV files per unit time
    try:
        df_wav = pd.read_sql_query("SELECT start_time FROM wav_files", conn)
        df_wav['datetime'] = pd.to_datetime(df_wav['start_time'], utc=True)
        df_wav = df_wav.dropna(subset=['datetime'])
        df_wav['count'] = 1
        for freq in ['D', 'W', 'MS']:
            save_plotly_timeseries(df_wav[['datetime', 'count']], 'datetime', 'count', "WAV Files", os.path.join(outdir, "wav_files"), freq)
    except Exception as e:
        print(f"[ERROR] Could not plot wav_files time series: {e}")

    # (3) Corrected IDs
    try:
        df_corr = pd.read_sql_query("SELECT date, corrected_id FROM trace_id_corrections", conn)
        df_corr['date'] = pd.to_datetime(df_corr['date'], utc=True)
        df_corr = df_corr.dropna(subset=['date'])
        df_corr.set_index('date', inplace=True)
        for freq in ['D', 'W', 'MS']:
            ids = df_corr.groupby(pd.Grouper(freq=freq))['corrected_id'].nunique().reset_index()
            ids.columns = ['date', 'unique_corrected_ids']
            save_plotly_timeseries(ids, 'date', 'unique_corrected_ids', "Corrected IDs", os.path.join(outdir, "corrected_ids"), freq, is_mean=True)
    except Exception as e:
        print(f"[ERROR] Could not plot corrected_id time series: {e}")

    # (4) Stations per unit time
    try:
        df_corr['station'] = df_corr['corrected_id'].str.split('.').str[1]
        for freq in ['D', 'W', 'MS']:
            station_df = df_corr.groupby(pd.Grouper(freq=freq))['station'].nunique().reset_index()
            station_df.columns = ['date', 'unique_stations']
            save_plotly_timeseries(station_df, 'date', 'unique_stations', "Stations Reporting", os.path.join(outdir, "stations"), freq, is_mean=True)
    except Exception as e:
        print(f"[ERROR] Could not plot station count time series: {e}")

    # (5) Files summary comparison
    try:
        df_wav = pd.read_sql_query("SELECT parsed_time FROM wav_files", conn)
        df_aef = pd.read_sql_query("SELECT parsed_time FROM aef_files", conn)
        df_sfiles = pd.read_sql_query("SELECT parsed_time FROM sfiles", conn)

        def prep_df(df, label):
            df['datetime'] = pd.to_datetime(df['parsed_time'], utc=True)
            df = df.dropna(subset=['datetime'])
            df = df[(df['datetime'] >= START_DATE) & (df['datetime'] <= END_DATE)]
            df['count'] = 1
            return df[['datetime', 'count']].set_index('datetime').resample('D').sum().rename(columns={'count': label})

        combined = pd.concat([
            prep_df(df_wav, 'wav_files'),
            prep_df(df_aef, 'aef_files'),
            prep_df(df_sfiles, 'sfiles')
        ], axis=1).fillna(0)

        for freq in ['D', 'W', 'MS']:
            summary = combined.resample(freq).sum().reset_index()
            summary.to_csv(os.path.join(outdir, f"files_summary_{freq}.csv"), index=False)
            fig = px.line(summary, x='datetime', y=['wav_files', 'aef_files', 'sfiles'], title=f"Files Summary ({freq})")
            fig.write_html(os.path.join(outdir, f"files_summary_{freq}.html"))
            try:
                fig.write_image(os.path.join(outdir, f"files_summary_{freq}.png"), engine="kaleido")
            except Exception as e:
                print(f"[ERROR] Could not save plot files_summary_{freq}.png: {e}")
    except Exception as e:
        print(f"[ERROR] Could not create files summary plots: {e}")

    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check the status of the MVOE index database")
    parser.add_argument("index_db", help="Path to the SQLite database")
    args = parser.parse_args()
    show_status_summary(args.index_db)

