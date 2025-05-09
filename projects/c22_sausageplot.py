# Refactored Sausage Plot Code (ObsPy + Pandas centric)
# Now includes multi-volcano support with percentile-based sizing and long-term historical mode

def main():
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from obspy import UTCDateTime, read_events
    from obspy.geodetics import gps2dist_azimuth
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # Sample placeholders for paths
    catalog_file = "./catalog.xml"  # QuakeML file or similar
    volcano_csv = "./volcanoes.csv"  # CSV with columns: place, lat, lon, elev, radius_km
    output_png = "./sausage_plot.png"

    # Read volcano metadata
    volcanoes = pd.read_csv(volcano_csv)

    # Read earthquake catalog
    cat = read_events(catalog_file)

    # Convert catalog to DataFrame
    rows = []
    for ev in cat:
        if ev.origins:
            o = ev.preferred_origin() or ev.origins[0]
            rows.append({
                'time': o.time.datetime,
                'latitude': o.latitude,
                'longitude': o.longitude,
                'depth_km': o.depth / 1000.0 if o.depth else np.nan,
                'magnitude': ev.preferred_magnitude().mag if ev.preferred_magnitude() else np.nan
            })
    df = pd.DataFrame(rows)

    # General parameters
    now = UTCDateTime()
    weeks_back = 12
    start_time = now - weeks_back * 7 * 86400
    df['week'] = df['time'].apply(lambda t: int((UTCDateTime(t) - start_time) / (86400 * 7)))
    df_recent = df[df['week'] >= 0]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = cm.get_cmap('hot_r')
    norm = mcolors.Normalize(vmin=1.0, vmax=5.0)

    SCALE = 25
    PERCENTILES = {}
    longest_weeks = 0

    for idx, v in volcanoes.iterrows():
        name, lat0, lon0, radius = v['place'], v['lat'], v['lon'], v['radius_km'] * 1000

        def within_radius(row):
            dist_m, _, _ = gps2dist_azimuth(lat0, lon0, row['latitude'], row['longitude'])
            return dist_m <= radius

        dfv_all = df[df.apply(within_radius, axis=1)]
        dfv_recent = df_recent[df_recent.apply(within_radius, axis=1)]

        if dfv_recent.empty:
            continue

        # Weekly binning for full history and recent
        counts_all = dfv_all.copy()
        counts_all['hist_week'] = counts_all['time'].apply(lambda t: int((UTCDateTime(t) - UTCDateTime(min(df['time']))) / (86400 * 7)))
        hist_counts = counts_all.groupby('hist_week').size()

        if name not in PERCENTILES:
            PERCENTILES[name] = np.percentile(hist_counts.values, np.arange(101))

        recent_counts = dfv_recent.groupby('week').size()
        recent_mags = dfv_recent.groupby('week')['magnitude'].sum()

        for w, count in recent_counts.items():
            mag = recent_mags.get(w, np.nan)
            color = cmap(norm(mag))
            percentile = np.searchsorted(PERCENTILES[name], count, side='right')
            msize = 5 + (percentile / 100) * SCALE
            ax.scatter(idx, weeks_back - w, s=msize, color=color, edgecolor='k')

        longest_weeks = max(longest_weeks, recent_counts.index.max())

    ax.set_xticks(range(len(volcanoes)))
    ax.set_xticklabels(volcanoes['place'], rotation=45, ha='right')
    ax.set_yticks(range(longest_weeks + 1))
    ax.set_yticklabels([f"Week {-i}" for i in range(longest_weeks + 1)])
    ax.set_title("Sausage Plot of Seismic Activity at Volcanoes")
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"Saved sausage plot to {output_png}")

if __name__ == '__main__':
    main()

