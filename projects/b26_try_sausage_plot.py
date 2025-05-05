# sausage_plot.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read_events, UTCDateTime
from obspy.geodetics import gps2dist_azimuth
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import mplcursors
from flovopy.core.enhanced import bin_events
from flovopy.core.mvo import dome_location

def load_volcanoes(volcano_csv):
    """Load volcano metadata or fallback to dome_location."""
    if os.path.exists(volcano_csv):
        return pd.read_csv(volcano_csv)
    else:
        return pd.DataFrame([{
            'place': 'Soufriere Hills',
            'lat': dome_location['lat'],
            'lon': dome_location['lon'],
            'elev': dome_location.get('elevation', 0),
            'radius_km': 5.0
        }])

def load_catalog(catalog_file, catalog_csv):
    """Load earthquake catalog from QuakeML or CSV."""
    if os.path.exists(catalog_file):
        cat = read_events(catalog_file)
        rows = []
        for ev in cat:
            if ev.origins:
                o = ev.preferred_origin() or ev.origins[0]
                rows.append({
                    'time': o.time.datetime,
                    'latitude': o.latitude,
                    'longitude': o.longitude,
                    'depth_km': o.depth / 1000.0 if o.depth else np.nan,
                    'magnitude': ev.preferred_magnitude().mag if ev.preferred_magnitude() else np.nan,
                    'subclass': 'all'
                })
        return pd.DataFrame(rows)
    elif os.path.exists(catalog_csv):
        df = pd.read_csv(catalog_csv, parse_dates=['time'])
        for col in ['depth_km', 'magnitude', 'subclass']:
            if col not in df.columns:
                df[col] = np.nan if col != 'subclass' else 'all'
        return df
    else:
        raise FileNotFoundError("No catalog file or CSV found!")

def plot_sausage(
    df,
    volcanoes,
    output_png="sausage_plot.png",
    binsize="M",
    months_back=24,
    scale=40,
    color_by="magnitude",
    title="Sausage Plot of Seismic Activity",
):
    """
    Generate a sausage plot showing seismic activity around volcanoes.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog dataframe with at least 'time', 'latitude', 'longitude', 'magnitude'.
    volcanoes : pd.DataFrame
        Volcano dataframe with 'place', 'lat', 'lon', 'radius_km'.
    output_png : str
        Filename to save the output plot.
    binsize : str
        Binning interval (e.g., 'M' for month, 'A' for year).
    months_back : int
        How many months back from latest event to include.
    scale : float
        Size scaling factor for bubbles.
    color_by : str
        Color by 'magnitude' or 'energy'.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = cm.get_cmap("hot_r")
    norm = mcolors.Normalize(vmin=1.0, vmax=6.0)  # Adjust if needed

    df['time'] = pd.to_datetime(df['time'])
    latest_time = df['time'].max()

    # Set true calendar-aligned start time if yearly bins
    if binsize in ["A", "Y", "12M"]:
        start_year = (latest_time - pd.DateOffset(months=months_back)).year
        start_time = pd.Timestamp(f"{start_year}-01-01")
    else:
        start_time = latest_time - pd.DateOffset(months=months_back)

    df_recent = df[df['time'] >= start_time]

    only_one_volcano = len(volcanoes) == 1
    PERCENTILES = {}

    bin_labels = []

    for idx, v in volcanoes.iterrows():
        name, lat0, lon0, radius = v['place'], v['lat'], v['lon'], v['radius_km'] * 1000

        if only_one_volcano:
            dfv_all = df.copy()
            dfv_recent = df_recent.copy()
        else:
            def within_radius(row):
                try:
                    dist_m, _, _ = gps2dist_azimuth(lat0, lon0, row['latitude'], row['longitude'])
                    return dist_m <= radius
                except Exception:
                    return False
            dfv_all = df[df.apply(within_radius, axis=1)].copy()
            dfv_recent = df_recent[df_recent.apply(within_radius, axis=1)].copy()

        if dfv_recent.empty:
            continue

        dfv_all['subclass'] = 'all'
        binned_all = bin_events(dfv_all, interval=binsize, groupby='subclass')
        hist_counts = binned_all.groupby('time').size()

        if name not in PERCENTILES:
            PERCENTILES[name] = np.percentile(hist_counts.values, np.arange(101))

        dfv_recent['subclass'] = 'all'
        binned_recent = bin_events(dfv_recent, interval=binsize, groupby='subclass')
        recent_counts = binned_recent.groupby('time').size()

        if color_by == "magnitude":
            recent_mags = binned_recent.groupby('time')['cumulative_magnitude'].sum()
        elif color_by == "energy":
            mags = binned_recent.groupby('time')['cumulative_magnitude'].sum()
            energies = 10 ** (1.5 * mags)
            cumulative_energies = energies.groupby(level=0).sum()
            recent_mags = (np.log10(cumulative_energies) / 1.5)
        else:
            raise ValueError("color_by must be 'magnitude' or 'energy'.")

        bin_times = list(recent_counts.index)
        bin_labels = [bt.strftime("%Y") if binsize in ['A', 'Y', '12M'] else bt.strftime("%Y-%m") for bt in bin_times]

        for bin_idx, (bin_time, count) in enumerate(recent_counts.items()):
            mag = recent_mags.get(bin_time, np.nan)
            color = cmap(norm(mag))
            percentile = np.searchsorted(PERCENTILES[name], count, side='right')
            if percentile > 50:
                msize = 5 + ((percentile-50) / 50) * scale * 25
            else:
                msize = 5

            sc = ax.scatter(
                idx,
                bin_idx,
                s=msize,
                c=[color],
                marker='s',  # square
                edgecolors='k',
                zorder=3
            )

    # Axis formatting
    ax.set_xticks(range(len(volcanoes)))
    ax.set_xticklabels(volcanoes['place'], rotation=45, ha='right')

    ax.set_yticks(range(len(bin_labels)))
    ax.set_yticklabels(reversed(bin_labels))  # Most recent at bottom
    ax.set_ylabel(f"Time ({binsize} bins)")
    ax.set_xlabel("Volcano")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.invert_yaxis()
    fig.tight_layout()

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, label=f"Cumulative {color_by.title()}")

    # Add hover tooltips
    cursor = mplcursors.cursor(ax.collections, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(
            f"Bin: {bin_labels[len(bin_labels) - sel.index - 1]}"
        )
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    plt.savefig(output_png, dpi=200)
    print(f"[✓] Saved plot to {output_png}")

if __name__ == "__main__":
    # --- CONFIG ---
    catalog_file = "./catalog.xml"
    catalog_csv = "/home/thompsong/Dropbox/AEG_talk/old_energymag_converted.csv"
    volcano_csv = "./volcanoes.csv"
    output_png = "./sausage_plot.png"
    years_back = 14
    months_back = 12 * years_back # How many months to look back
    binsize = "M"
    months_per_string = 12
    #binsize = "A"  # 'A' = Annual
    scale = 50
    color_by = "magnitude"
    title = "Sausage Plot of Seismic Activity"
    # ---------------

    volcanoes = load_volcanoes(volcano_csv)
    df = load_catalog(catalog_file, catalog_csv)

    plot_sausage(
        df=df,
        volcanoes=volcanoes,
        output_png=output_png,
        binsize=binsize,
        months_back=months_back,
        scale=scale,
        color_by=color_by,
        title=title
    )
