import os
import sqlite3
import pandas as pd
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "MVOE Data Dashboard"

DB_PATH = "/home/public_html/index_mvoe.sqlite"
START_DATE = pd.Timestamp("1996-10-01", tz='UTC')
END_DATE = pd.Timestamp("2008-09-01", tz='UTC')


def fetch_table_counts(conn):
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    summary = []
    for table in tables:
        name = table[0]
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            summary.append({"table": name, "rows": count})
        except:
            summary.append({"table": name, "rows": "error"})
    return pd.DataFrame(summary)


def fetch_time_series(conn, table, time_col):
    try:
        df = pd.read_sql(f"SELECT {time_col} FROM {table}", conn)
        df['datetime'] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
        df = df.dropna(subset=['datetime'])
        df = df[(df['datetime'] >= START_DATE) & (df['datetime'] <= END_DATE)]
        df['count'] = 1
        return df[['datetime', 'count']].groupby('datetime').sum().resample('W').sum().reset_index()
    except Exception as e:
        return pd.DataFrame(columns=['datetime', 'count'])


def fetch_top_errors(conn, limit=10):
    try:
        df = pd.read_sql("SELECT error_message FROM wav_processing_errors", conn)
        return df['error_message'].value_counts().head(limit).reset_index().rename(columns={'index': 'Error Message', 'error_message': 'Count'})
    except:
        return pd.DataFrame(columns=['Error Message', 'Count'])


conn = sqlite3.connect(DB_PATH)
summary_df = fetch_table_counts(conn)
wav_df = fetch_time_series(conn, "wav_files", "start_time")
aef_df = fetch_time_series(conn, "aef_files", "parsed_time")
sfile_df = fetch_time_series(conn, "sfiles_mvo", "parsed_time")
error_df = fetch_top_errors(conn)
conn.close()

file_fig = px.line()
if not wav_df.empty:
    file_fig.add_scatter(x=wav_df['datetime'], y=wav_df['count'], name="WAV files")
if not aef_df.empty:
    file_fig.add_scatter(x=aef_df['datetime'], y=aef_df['count'], name="AEF files")
if not sfile_df.empty:
    file_fig.add_scatter(x=sfile_df['datetime'], y=sfile_df['count'], name="S files")
file_fig.update_layout(title="Weekly File Counts", xaxis_title="Date", yaxis_title="Count")

layout = dbc.Container([
    html.H1("MVOE Seismic Archive Dashboard", className="mt-4 mb-4"),

    html.H4("Database Table Summary"),
    dash_table.DataTable(
        summary_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in summary_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    ),

    html.Hr(),
    html.H4("Weekly File Upload Timeline"),
    dcc.Graph(figure=file_fig),

    html.Hr(),
    html.H4("Top WAV File Processing Errors"),
    dash_table.DataTable(
        error_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in error_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    ),

    html.Footer("Generated with Dash + SQLite", className="mt-4")
], fluid=True)

app.layout = layout

if __name__ == '__main__':
    app.run(debug=True, port=8051)


