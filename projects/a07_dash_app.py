import sqlite3
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import subprocess
from dash import Dash, html, dcc, Input, Output

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

START_DATE = pd.Timestamp("1996-10-01")
END_DATE = pd.Timestamp("2008-09-01")
START_DATE_TZ = pd.Timestamp("1996-10-01", tz='UTC')
END_DATE_TZ = pd.Timestamp("2008-09-01", tz='UTC')


DB_PATH = "/home/thompsong/public_html/index_mvoe.sqlite"

app = Dash(__name__, suppress_callback_exceptions=True)


def run_shell_count(command):
    try:
        result = subprocess.check_output(command, shell=True).decode('utf-8').strip()
        return int(result)
    except Exception as e:
        print(f"[ERROR] Failed to run command: {command}\n{e}")
        return 0


def get_filesystem_summary():
    return {
        "MVO WAVs": run_shell_count("find /data/SEISAN_DB/WAV/MVOE_/[12]* -type f -iname '*MVO*' ! -iname '*.png' ! -iname '*.jpg' | wc -l"),
        "MVO AEFs": run_shell_count("find /data/SEISAN_DB/AEF/MVOE_/[12]* -type f -iname '*MVO*.aef' ! -iname '*.png' ! -iname '*.jpg' | wc -l"),
        "MVO REA Local": run_shell_count("find /data/SEISAN_DB/REA/MVOE_/[12]* -type f -iname '*L.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l"),
        "MVO REA Regional": run_shell_count("find /data/SEISAN_DB/REA/MVOE_/[12]* -type f -iname '*R.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l"),
        "MVO REA Teleseismic": run_shell_count("find /data/SEISAN_DB/REA/MVOE_/[12]* -type f -iname '*D.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l"),
        "SPN WAVs in MVOE_": run_shell_count("find /data/SEISAN_DB/WAV/MVOE_/[12]* -type f -iname '*SPN*' ! -iname '*.png' ! -iname '*.jpg' | wc -l"),
        "ASN WAVs": run_shell_count("find /data/SEISAN_DB/WAV/ASNE_/[12]* -type f -iname '*ASN*' ! -iname '*.png' ! -iname '*.jpg' | wc -l"),
        "ASN REA Local": run_shell_count("find /data/SEISAN_DB/REA/ASNE_/[12]* -type f -iname '*L.S*' ! -iname '*.png' ! -iname '*.jpg' | wc -l")
    }


def summarize_db(conn):
    summary = {}
    for table in TABLES:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            summary[table] = count
        except:
            summary[table] = 0
    return summary


def generate_plot(df, xcol, ycols, title):
    if isinstance(ycols, str):
        ycols = [ycols]
    fig = px.line(df, x=xcol, y=ycols, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return dcc.Graph(figure=fig)



def generate_heatmap(df):
    pivot = df.pivot_table(index='station', columns='date', aggfunc='size', fill_value=0)
    fig = px.imshow(pivot.values, 
                    labels=dict(x="Date", y="Station", color="Count"),
                    x=pivot.columns.strftime('%Y-%m-%d'),
                    y=pivot.index,
                    title="Station Participation Heatmap")
    fig.update_layout(height=600)
    return dcc.Graph(figure=fig)


def load_time_series_data(resample_rule):
    conn = sqlite3.connect(DB_PATH)
    data = {}

    try:
        df_wav = pd.read_sql_query("SELECT start_time FROM wav_files", conn, parse_dates=['start_time'])
        df_wav = df_wav[(df_wav['start_time'] >= START_DATE_TZ) & (df_wav['start_time'] < END_DATE_TZ)]
        df_wav['count'] = 1
        df_wav = df_wav.set_index('start_time').resample(resample_rule).count().rename(columns={'count': 'WAV'})

        df_aef = pd.read_sql_query("SELECT parsed_time FROM aef_files WHERE parsed_successfully=1", conn, parse_dates=['parsed_time'])
        df_aef = df_aef[(df_aef['parsed_time'] >= START_DATE) & (df_aef['parsed_time'] < END_DATE)]
        df_aef['count'] = 1
        df_aef = df_aef.set_index('parsed_time').resample(resample_rule).count().rename(columns={'count': 'AEF'})

        df_sfile = pd.read_sql_query("SELECT parsed_time FROM sfiles", conn, parse_dates=['parsed_time'])
        df_sfile = df_sfile[(df_sfile['parsed_time'] >= START_DATE) & (df_sfile['parsed_time'] < END_DATE)]
        df_sfile['count'] = 1
        df_sfile = df_sfile.set_index('parsed_time').resample(resample_rule).count().rename(columns={'count': 'S-file'})

        df_combined = pd.concat([df_wav, df_aef, df_sfile], axis=1).fillna(0).reset_index()
        data['combined_files'] = df_combined

    except Exception as e:
        data['combined_files'] = pd.DataFrame()
        print("[WARN] Could not load combined file time series:", e)


    try:
        #df = pd.read_sql_query("SELECT DISTINCT DATE(parsed_time) as date, corrected_id FROM trace_id_corrections", conn, parse_dates=['date'])
        df = pd.read_sql_query("SELECT date, corrected_id FROM trace_id_corrections", conn, parse_dates=['date'])

        df = df[(df['date'] >= START_DATE) & (df['date'] < END_DATE)]
        data['corrected_ids'] = df.groupby('date').agg({'corrected_id': 'nunique'})
        data['corrected_ids'].rename(columns={'corrected_id': 'unique_corrected_ids'}, inplace=True)
    except Exception as e:
        data['corrected_ids'] = pd.DataFrame()
        print("[WARN] Could not load corrected_ids time series:", e)

    try:
        #df = pd.read_sql_query("SELECT DISTINCT DATE(parsed_time) as date, corrected_id FROM trace_id_corrections", conn, parse_dates=['date'])
        df = pd.read_sql_query("SELECT date, corrected_id FROM trace_id_corrections", conn, parse_dates=['date'])

        df = df[(df['date'] >= START_DATE) & (df['date'] < END_DATE)]
        df['station'] = df['corrected_id'].apply(lambda x: x.split('.')[1] if isinstance(x, str) and '.' in x else None)
        data['station_counts'] = df.groupby('date').agg({'station': pd.Series.nunique})
        data['station_counts'].rename(columns={'station': 'unique_stations'}, inplace=True)
        data['station_heatmap'] = df[['date', 'station']].dropna()
    except Exception as e:
        data['station_counts'] = pd.DataFrame()
        data['station_heatmap'] = pd.DataFrame()
        print("[WARN] Could not load station count time series:", e)

    try:
        df = pd.read_sql_query("SELECT trace_id, first_seen, last_seen FROM response_failures_summary", conn, parse_dates=["first_seen", "last_seen"])
        df = df.dropna(subset=["trace_id", "first_seen", "last_seen"])

        # Expand to daily rows per trace_id
        records = []
        for _, row in df.iterrows():
            trace_id = row['trace_id']
            for dt in pd.date_range(row['first_seen'], row['last_seen'], freq='D'):
                records.append((dt.date(), trace_id))

        df_expanded = pd.DataFrame(records, columns=["date", "trace_id"])
        pivot = df_expanded.pivot_table(index="trace_id", columns="date", aggfunc="size", fill_value=0)
        data["missing_response_heatmap_raw"] = df_expanded
        data["missing_response_heatmap_pivot"] = pivot

    except Exception as e:
        data["missing_response_heatmap"] = pd.DataFrame()
        print("[WARN] Could not generate missing response heatmap:", e)


    conn.close()
    return data


@app.callback(
    Output("time-series-graphs", "children"),
    Input("time-resolution", "value")
)
def update_time_series(resample_value):
    data = load_time_series_data(resample_value)
    plots = []

    for key, df in data.items():
        if key == 'combined_files' and not df.empty:
            plots.append(generate_plot(df, 'start_time', ['WAV', 'AEF', 'S-file'], "Indexed Files: WAV, AEF, S-file"))
        elif key == 'corrected_ids' and not df.empty:
            plots.append(generate_plot(df.reset_index(), 'date', 'unique_corrected_ids', "Unique Corrected IDs"))
        elif key == 'station_counts' and not df.empty:
            plots.append(generate_plot(df.reset_index(), 'date', 'unique_stations', "Unique Stations Reporting"))
        elif key == 'station_heatmap' and not df.empty:
            plots.append(generate_heatmap(df))
        elif key == "missing_response_heatmap" and not df.empty:
            plots.append(generate_missing_response_heatmap(df))
    

    if not plots:
        plots.append(html.Div("No data available for the selected time resolution."))

    return plots

def generate_missing_response_heatmap(df):
    if df.empty:
        return html.Div("No missing response data available.")
    
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="trace_id", columns="date", aggfunc="size", fill_value=0)

    fig = px.imshow(
        pivot.values,
        labels=dict(x="Date", y="Trace ID", color="Missing"),
        x=pivot.columns.strftime('%Y-%m-%d'),
        y=pivot.index,
        title="Missing Instrument Response: Trace ID vs Time"
    )
    fig.update_layout(height=800, margin=dict(l=20, r=20, t=40, b=20))
    return dcc.Graph(figure=fig)

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    Input("time-resolution", "value"),
    prevent_initial_call=True
)
def download_heatmap_csv(n_clicks, resample_value):
    data = load_time_series_data(resample_value)
    pivot = data.get("missing_response_heatmap_pivot")

    if pivot is not None and not pivot.empty:
        return dcc.send_data_frame(pivot.to_csv, "missing_response_heatmap.csv")
    else:
        return None





conn = sqlite3.connect(DB_PATH)
db_summary = summarize_db(conn)
fs_summary = get_filesystem_summary()
conn.close()

app.layout = html.Div([
    html.H1("MVOE Database Dashboard"),

    html.H2("Filesystem Summary"),
    html.Ul([html.Li(f"{k}: {v}") for k, v in fs_summary.items()]),

    html.H2("Database Table Counts"),
    html.Ul([html.Li(f"{table}: {count} rows") for table, count in db_summary.items()]),

    html.H2("Time Series Plots"),
    html.Div([
        html.Label("Select Time Resolution:"),
        dcc.RadioItems(
            id="time-resolution",
            options=[
                {"label": "Daily", "value": "D"},
                {"label": "Weekly", "value": "W"},
                {"label": "Monthly", "value": "MS"}
            ],
            value="D",
            labelStyle={"display": "inline-block", "margin-right": "10px"}
        )
    ], style={"margin-bottom": "20px"}),

    html.Div(id="time-series-graphs")
    html.Div([
        html.Button("Download Missing Response Heatmap as CSV", id="download-btn"),
        dcc.Download(id="download-dataframe-csv")
    ], style={"margin-top": "20px"}),

])

if __name__ == "__main__":
    app.run(debug=True)