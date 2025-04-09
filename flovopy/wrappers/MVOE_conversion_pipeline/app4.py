iimport sqlite3
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import subprocess
from dash import Dash, html, dcc

try:
    if pio.kaleido.scope:
        pio.kaleido.scope.mathjax = None
except Exception:
    pass

TABLES = [
    "wav_files", "aef_files", "aef_metrics",
    "sfile_wav_map", "processing_log", "trace_id_corrections",
    "sfiles_mvo", "sfiles_bgs", "wav_processing_errors"
]

START_DATE = pd.Timestamp("1996-10-01", tz='UTC')
END_DATE = pd.Timestamp("2008-09-01", tz='UTC')

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


def generate_placeholder_plot(title):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Count")
    return dcc.Graph(figure=fig)


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

    html.H2("Time Series Plots (placeholders)"),
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

    html.Div([
        generate_placeholder_plot("WAV Files Indexed Over Time"),
        generate_placeholder_plot("AEF Files Indexed Over Time"),
        generate_placeholder_plot("S-Files (BGS) Indexed Over Time"),
        generate_placeholder_plot("Corrected Trace IDs Per Day"),
        generate_placeholder_plot("Station Count Per Day"),
        generate_placeholder_plot("Station Participation Heatmap")
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
