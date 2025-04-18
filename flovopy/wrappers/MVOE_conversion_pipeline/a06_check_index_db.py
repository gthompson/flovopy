import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import subprocess
from obspy import read_inventory
import traceback
# Optional: Disable MathJax to suppress warnings
try:
    if pio.kaleido.scope:
        pio.kaleido.scope.mathjax = None
except Exception:
    pass

TABLES = [
    "wav_files", "aef_files", "aef_metrics",
    "sfile_wav_map", "processing_log", "trace_id_corrections",
    "sfiles", "wav_processing_errors", "aef_processing_errors", "sfile_processing_errors",
    "mseed_processing_log", "response_failures_summary"
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

def generate_error_summary(df_errors, outdir, filetype):
    if 'error_message' in df_errors.columns:
        counts = df_errors['error_message'].value_counts()
        print(f"\n--- {filetype} PROCESSING ERRORS SUMMARY ---")
        print(counts)
        counts.to_csv(os.path.join(outdir, f"a06_{filetype}_processing_errors_summary.csv"))
        counts_df = counts.reset_index()
        counts_df.columns = ['error_message', 'count']
        fig = px.bar(counts_df, x='error_message', y='count',
             title=f"Top {filetype} Processing Errors",
             labels={'error_message': 'Error Message', 'count': 'Count'})

        fig.update_layout(xaxis_tickangle=-45)
        fig.write_html(os.path.join(outdir, f"a06_{filetype}_processing_errors_summary.html"))
        try:
            fig.write_image(os.path.join(outdir, f"a06_{filetype}_processing_errors_summary.png"), engine="kaleido")
        except Exception as e:
            print(f"[ERROR] Could not save error plot: {e}")




def export_stationxml_todo_table(conn, outdir):
    print("\n=== Missing Instrument Response Summary ===")
    try:
        df_resp = pd.read_sql_query("SELECT * FROM response_failures_summary", conn)
        if df_resp.empty:
            print("No missing response entries found.")
            return

        # Split trace_id into NET.STA.LOC.CHA components
        parts = df_resp['trace_id'].str.split('.', expand=True)
        df_resp['network'] = parts[0]
        df_resp['station'] = parts[1]
        df_resp['location'] = parts[2]
        df_resp['channel'] = parts[3]

        # Write CSV
        todo_path = os.path.join(outdir, "a06_stationxml_todo.csv")
        df_resp.sort_values(by=["trace_id"], inplace=True)
        df_resp.to_csv(todo_path, index=False)
        print(f"[INFO] Missing response summary written to {todo_path}")

        # Summary counts
        print(f"[INFO] Total missing trace_ids: {df_resp['trace_id'].nunique()}")
        print(f"[INFO] Affected stations: {df_resp['station'].nunique()}")

        # Safely convert date columns
        df_resp['first_seen'] = pd.to_datetime(df_resp['first_seen'], errors='coerce', utc=True)
        df_resp['last_seen'] = pd.to_datetime(df_resp['last_seen'], errors='coerce', utc=True)
        df_resp.dropna(subset=['first_seen', 'last_seen'], inplace=True)

        df_resp['duration_days'] = (df_resp['last_seen'] - df_resp['first_seen']).dt.total_seconds() / 86400.0

        # Plot timeline
        fig = px.timeline(df_resp, x_start="first_seen", x_end="last_seen", y="trace_id",
                          title="Missing Instrument Response Periods", labels={"trace_id": "Trace ID"})
        fig.update_yaxes(autorange="reversed")
        fig.write_html(os.path.join(outdir, "a06_stationxml_missing_timeline.html"))

        try:
            fig.write_image(os.path.join(outdir, "a06_stationxml_missing_timeline.png"), engine="kaleido")
        except Exception as e:
            print(f"[ERROR] Could not save timeline PNG: {e}")

    except Exception as e:
        print(f"[ERROR] Could not query or plot response_failures_summary:\n{traceback.format_exc()}")




def generate_stationxml_needed_list(conn, inv_path, outdir):
    print("\n=== Comparing Missing Trace IDs to Existing StationXML Inventory ===")
    try:
        # Load missing trace IDs
        df_resp = pd.read_sql_query("SELECT trace_id FROM response_failures_summary", conn)
        if df_resp.empty:
            print("[INFO] No missing trace IDs found.")
            return

        parts = df_resp['trace_id'].str.split('.', expand=True)
        df_resp['network'] = parts[0]
        df_resp['station'] = parts[1]

        df_needed = df_resp[['network', 'station']].drop_duplicates()

        # Load inventory
        inv = read_inventory(inv_path)
        available_pairs = {(net.code, sta.code) for net in inv for sta in net.stations}

        df_needed['available'] = df_needed.apply(
            lambda row: (row['network'], row['station']) in available_pairs,
            axis=1
        )

        # Only show those not in current inventory
        df_missing = df_needed[~df_needed['available']].copy()
        df_missing.sort_values(by=['network', 'station'], inplace=True)

        # Suggest StationXML filenames
        df_missing['suggested_xml_file'] = df_missing['network'] + "." + df_missing['station'] + ".xml"

        out_path = os.path.join(outdir, "a06_stationxml_needed.csv")
        df_missing.to_csv(out_path, index=False)
        print(f"[INFO] StationXML needed for {len(df_missing)} network.station pairs")
        print(f"[INFO] Written to {out_path}")

    except Exception as e:
        print(f"[ERROR] Could not generate a06_stationxml_needed.csv: {e}")

def save_combined_plotly_timeseries(df_dict, freq, out_prefix, title):
    """
    df_dict: dictionary with keys as labels (e.g. "WAV", "AEF", "S-file")
             and values as raw DataFrames with 'datetime' and 'count' columns.
    freq: 'D', 'W', or 'MS'
    out_prefix: prefix for output files (CSV, HTML, PNG)
    title: plot title
    """
    df_merged = None
    for label, df in df_dict.items():
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df[(df['datetime'] >= START_DATE) & (df['datetime'] <= END_DATE)]
        df.set_index('datetime', inplace=True)
        resampled = df.resample(freq).sum().rename(columns={'count': label})
        if df_merged is None:
            df_merged = resampled
        else:
            df_merged = df_merged.join(resampled, how='outer')

    df_merged = df_merged.fillna(0).reset_index()

    # Save CSV
    csv_path = f"{out_prefix}_{freq}.csv"
    df_merged.to_csv(csv_path, index=False)

    # Plot
    fig = px.line(
        df_merged,
        x='datetime',
        y=list(df_dict.keys()),
        title=f"{title} ({freq})"
    )
    html_path = f"{out_prefix}_{freq}.html"
    fig.write_html(html_path)

    png_path = f"{out_prefix}_{freq}.png"
    try:
        fig.write_image(png_path, engine="kaleido")
    except Exception as e:
        print(f"[ERROR] Could not save image: {e}")


def show_status_summary(db_path):
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return

    #add_filesystem_summary()

    conn = sqlite3.connect(db_path)
    outdir = os.path.dirname(db_path)

    # (1) WAV processing errors summary
    try:
        df_errors = pd.read_sql_query("SELECT error_message FROM wav_processing_errors", conn)
        generate_error_summary(df_errors, outdir, 'WAV')
    except Exception as e:
        print(f"[ERROR] Could not summarize wav_processing_errors: {e}")
    # (1 CONT.) AEF processing errors summary
    try:
        df_errors = pd.read_sql_query("SELECT error_message FROM aef_processing_errors", conn)
        generate_error_summary(df_errors, outdir, 'AEF')
    except Exception as e:
        print(f"[ERROR] Could not summarize aef_processing_errors: {e}")    
    # (1 CONT.) S-FILE processing errors summary
    try:
        df_errors = pd.read_sql_query("SELECT error_message FROM sfile_processing_errors", conn)
        generate_error_summary(df_errors, outdir, 'REA')
    except Exception as e:
        print(f"[ERROR] Could not summarize sfile_processing_errors: {e}")                

    # (2) WAV files per unit time
    try:
        df_wav = pd.read_sql_query("SELECT start_time FROM wav_files", conn)
        df_wav['datetime'] = pd.to_datetime(df_wav['start_time'], utc=True)
        df_wav = df_wav.dropna(subset=['datetime'])
        df_wav['count'] = 1
        for freq in ['D', 'W', 'MS']:
            save_plotly_timeseries(df_wav[['datetime', 'count']], 'datetime', 'count', "WAV Files", os.path.join(outdir, "a06_wav_files"), freq)
    except Exception as e:
        print(f"[ERROR] Could not plot wav_files time series: {e}")
    # (2 CONT.) AEF-files per unit time
    try:
        df_aeffile = pd.read_sql_query("SELECT filetime FROM aef_files", conn)
        df_aeffile['datetime'] = pd.to_datetime(df_aeffile['filetime'], utc=True)
        df_aeffile = df_aeffile.dropna(subset=['datetime'])
        df_aeffile['count'] = 1
        for freq in ['D', 'W', 'MS']:
            save_plotly_timeseries(df_aeffile[['datetime', 'count']], 'datetime', 'count', "AEF-Files", os.path.join(outdir, "a06_aef_files"), freq)
    except Exception as e:
        print(f"[ERROR] Could not plot sfiles time series: {e}")  
    # (2 CONT.) S-files per unit time
    try:
        df_sfile = pd.read_sql_query("SELECT filetime FROM sfiles", conn)
        df_sfile['datetime'] = pd.to_datetime(df_sfile['filetime'], utc=True)
        df_sfile = df_sfile.dropna(subset=['datetime'])
        df_sfile['count'] = 1
        for freq in ['D', 'W', 'MS']:
            save_plotly_timeseries(df_sfile[['datetime', 'count']], 'datetime', 'count', "S-Files", os.path.join(outdir, "a06_sfiles"), freq)
    except Exception as e:
        print(f"[ERROR] Could not plot sfiles time series: {e}")  
    # (2 CONT.) Combined WAV, AEF, S-file per unit time
    try:
        df_dict = {
            "WAV": df_wav[['datetime', 'count']],
            "AEF": df_aeffile[['datetime', 'count']],
            "S-file": df_sfile[['datetime', 'count']]
        }

        for freq in ['D', 'W', 'MS']:
            save_combined_plotly_timeseries(
                df_dict,
                freq=freq,
                out_prefix=os.path.join(outdir, "a06_combined_wav_aef_sfile"),
                title="WAV, AEF, and S-files"
            )
    except Exception as e:
        print(f"[ERROR] Could not generate combined WAV/AEF/S-file plot: {e}")
                 

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
    """
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
    """

    # (6) Missing StationXML summary
    try:
        export_stationxml_todo_table(conn, outdir)
    except Exception as e:
        print(f"[ERROR] Could not generate StationXML TODO list: {e}")
    try:
        inv_path = "/data/SEISAN_DB/CAL/MV.xml"  # Replace with your actual file
        generate_stationxml_needed_list(conn, inv_path, outdir)
    except Exception as e:
        print(f"[ERROR] Could not compare missing trace IDs to StationXML inventory: {e}")


    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check the status of the MVOE index database")
    parser.add_argument("index_db", help="Path to the SQLite database")
    args = parser.parse_args()
    show_status_summary(args.index_db)

