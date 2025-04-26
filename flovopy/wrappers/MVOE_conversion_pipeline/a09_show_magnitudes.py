import sqlite3
import pandas as pd
import os
scriptname = os.path.basename(__file__).replace('.py', '')     
TOP_DIR = os.path.join('/data', scriptname)
os.makedirs(TOP_DIR, exist_ok=True)
DB_PATH = "/home/thompsong/public_html/seiscomp_like.sqlite"
conn = sqlite3.connect(DB_PATH)
include_origins=False
include_magnitudes=False
include_all = True
addstring = '_MAG' if include_magnitudes else ''
if include_origins:
    addstring += '_LOC'
if include_all:
    addstring += '_ALL'
query_result_file = os.path.join(TOP_DIR, f'magnitude_query_result{addstring}2.pkl')
query_dataframe_pkl = os.path.join(TOP_DIR, f'magnitude_query_dataframe{addstring}2.pkl')
N=None
if os.path.isfile(query_dataframe_pkl):
    df = pd.read_pickle(query_dataframe_pkl)
else:

    # === Try loading from .pkl if it exists ===
    if os.path.isfile(query_result_file):
        print(f"[INFO] Loading cached query result from {query_result_file}")
        df = pd.read_pickle(query_result_file)
    else:
        print("[INFO] Running query and saving result to cache.")
        if include_all:
            query = '''
            SELECT e.public_id, o.latitude, o.longitude, o.depth, m.magnitude, m.mag_type, ec.time, ec.author, ec.source, ec.dfile, ec.mainclass, ec.subclass, mfs.dir
            FROM event_classifications ec
            JOIN events e ON ec.event_id = e.public_id
            LEFT JOIN origins o ON o.event_id = e.public_id
            JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
            LEFT JOIN magnitudes m ON m.event_id = e.public_id
            WHERE ec.subclass IS NOT NULL
            GROUP BY e.public_id
            '''            
        elif include_origins and include_magnitudes:
            query = '''
            SELECT e.public_id, o.latitude, o.longitude, o.depth, m.magnitude, m.mag_type, ec.time, ec.author, ec.source, ec.dfile, ec.mainclass, ec.subclass, mfs.dir
            FROM event_classifications ec
            JOIN events e ON ec.event_id = e.public_id
            LEFT JOIN origins o ON o.event_id = e.public_id
            JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
            JOIN magnitudes m ON m.event_id = e.public_id
            WHERE ec.subclass IS NOT NULL
            GROUP BY e.public_id
            '''
        elif include_magnitudes:
            query = '''
            SELECT e.public_id, m.magnitude, m.mag_type, ec.time, ec.author, ec.source, ec.dfile, ec.mainclass, ec.subclass, mfs.dir
            FROM event_classifications ec
            JOIN events e ON ec.event_id = e.public_id
            JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
            JOIN magnitudes m ON m.event_id = e.public_id
            WHERE ec.mainclass = 'LV'
            GROUP BY e.public_id
            '''        
        elif include_origins:
            query = '''
            SELECT e.public_id, o.latitude, o.longitude, o.depth, ec.time, ec.author, ec.source, ec.dfile, ec.mainclass, ec.subclass, mfs.dir
            FROM event_classifications ec
            JOIN events e ON ec.event_id = e.public_id
            LEFT JOIN origins o ON o.event_id = e.public_id
            JOIN mseed_file_status mfs ON ec.dfile = mfs.dfile
            WHERE ec.subclass IS NOT NULL
            GROUP BY e.public_id
            '''            
        if N:
            query += f" LIMIT {N}"


        df = pd.read_sql_query(query, conn)
        df.to_pickle(query_result_file)
    
    # sort dataframe
    df = df.sort_values(by="time")
    df.to_pickle(query_dataframe_pkl)

#df=df[df['subclass']=='r']    
print(f'got {len(df)} rows')

print('[STARTUP] dataframe of database query results created')

conn.close()
#filtered_df = df[df['depth'] < 10000]
#filtered_df = df[df['depth'] < 10000].copy()
filtered_df = df.copy()

#print(filtered_df)
#print(filtered_df.columns)
import pandas as pd
import matplotlib.pyplot as plt
#pd.plotting.scatter_matrix(filtered_df[['depth', 'magnitude', 'latitude', 'longitude']], figsize=(10, 10))
plt.show()


# Make sure 'time' is datetime
filtered_df['time'] = pd.to_datetime(filtered_df['time'])

if include_all:
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    # --- Make sure 'time' column is datetime ---
    filtered_df['time'] = pd.to_datetime(filtered_df['time'])
    filtered_df['year_month'] = filtered_df['time'].dt.to_period('M')

    # --- Calculate per-month percentages ---
    monthly_stats = []

    for period, group in filtered_df.groupby('year_month'):
        total_events = len(group)
        located_events = group[['latitude', 'longitude', 'depth']].dropna().shape[0]
        mag_events = group[['magnitude']].dropna().shape[0]
        ht_events = group[group['subclass'].isin(['h', 't'])].shape[0]

        located_pct = 100.0 * located_events / total_events if total_events > 0 else 0
        mag_pct = 100.0 * mag_events / total_events if total_events > 0 else 0
        ht_pct = 100.0 * ht_events / total_events if total_events > 0 else 0

        monthly_stats.append({
            'year_month': period.to_timestamp(),
            'located_pct': located_pct,
            'mag_pct': mag_pct,
            'ht_pct': ht_pct
        })

    monthly_df = pd.DataFrame(monthly_stats)

    # --- Apply Time Range Filter ---
    start_date = pd.Timestamp('1996-10-01')
    end_date = pd.Timestamp('2004-02-29')
    monthly_df = monthly_df[(monthly_df['year_month'] >= start_date) & (monthly_df['year_month'] <= end_date)]

    # --- Calculate 6-month moving averages ---
    monthly_df['located_pct_ma'] = monthly_df['located_pct'].rolling(window=6, min_periods=1, center=True).mean()
    monthly_df['mag_pct_ma'] = monthly_df['mag_pct'].rolling(window=6, min_periods=1, center=True).mean()
    monthly_df['ht_pct_ma'] = monthly_df['ht_pct'].rolling(window=6, min_periods=1, center=True).mean()

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14,7))

    width = 10  # days wide

    # Plot faded bars
    bars1 = ax.bar(
        monthly_df['year_month'] - pd.Timedelta(days=width/2),
        monthly_df['located_pct'],
        width=width,
        label='Located (%)',
        alpha=0.3
    )
    bars2 = ax.bar(
        monthly_df['year_month'] + pd.Timedelta(days=width/2),
        monthly_df['mag_pct'],
        width=width,
        label='With Magnitude (%)',
        alpha=0.3
    )

    # Get solid colors for moving averages
    located_color = mcolors.to_rgba(bars1.patches[0].get_facecolor(), alpha=1.0)
    mag_color = mcolors.to_rgba(bars2.patches[0].get_facecolor(), alpha=1.0)

    # Plot moving averages (bold and clear)
    ax.plot(
        monthly_df['year_month'],
        monthly_df['located_pct_ma'],
        color=located_color,
        linestyle='-',
        linewidth=3,
        label='Located (6-mo MA)'
    )
    ax.plot(
        monthly_df['year_month'],
        monthly_df['mag_pct_ma'],
        color=mag_color,
        linestyle='-',
        linewidth=3,
        label='Magnitude (6-mo MA)'
    )

    # Setup left y-axis
    ax.set_ylabel('Location & Magnitude Percentage (%)')
    ax.set_ylim(0, 30)
    ax.set_xlim([start_date, end_date])
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # --- Add right y-axis ---
    ax2 = ax.twinx()
    ax2.plot(
        monthly_df['year_month'],
        monthly_df['ht_pct_ma'],
        color='black',
        linestyle='--',
        linewidth=3,
        label="Hybrid+Tremor (6-mo MA)"
    )
    ax2.set_ylabel('Hybrid+Tremor Percentage (%)', color='black')

    # Make right y-axis match left y-axis exactly
    ax2.set_ylim(ax.get_ylim())
    ax2.grid(False)

    # --- Legends ---
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Merge both legends
    ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, framealpha=0.7)

    plt.tight_layout()

    # Save and Show
    plt.savefig(os.path.join(TOP_DIR, 'final_percentages_with_ht_combined_moving_average.png'))
    plt.show()


import matplotlib.dates as mdates
import matplotlib.colors as mcolors

# --- Make sure 'time' column is datetime ---
filtered_df['time'] = pd.to_datetime(filtered_df['time'])
filtered_df['year_month'] = filtered_df['time'].dt.to_period('M')

# --- Calculate per-month percentages ---
monthly_stats = []

for period, group in filtered_df.groupby('year_month'):
    total_events = len(group)
    located_events = group[['latitude', 'longitude', 'depth']].dropna().shape[0]
    mag_events = group[['magnitude']].dropna().shape[0]
    ht_events = group[group['subclass'].isin(['h', 't'])].shape[0]

    located_pct = 100.0 * located_events / total_events if total_events > 0 else 0
    mag_pct = 100.0 * mag_events / total_events if total_events > 0 else 0
    ht_pct = 100.0 * ht_events / total_events if total_events > 0 else 0

    monthly_stats.append({
        'year_month': period.to_timestamp(),
        'located_pct': located_pct,
        'mag_pct': mag_pct,
        'ht_pct': ht_pct
    })

monthly_df = pd.DataFrame(monthly_stats)

# --- Apply Time Range Filter ---
start_date = pd.Timestamp('1996-10-01')
end_date = pd.Timestamp('2004-02-29')
monthly_df = monthly_df[(monthly_df['year_month'] >= start_date) & (monthly_df['year_month'] <= end_date)]

# --- Calculate 3-month Acausal (past) moving averages ---
monthly_df['located_pct_ma'] = monthly_df['located_pct'].rolling(window=3, min_periods=1, center=False).mean()
monthly_df['mag_pct_ma'] = monthly_df['mag_pct'].rolling(window=3, min_periods=1, center=False).mean()
monthly_df['ht_pct_ma'] = monthly_df['ht_pct'].rolling(window=3, min_periods=1, center=False).mean()

# --- Plot ---
fig, ax = plt.subplots(figsize=(14,7))

# Plot moving averages (no bars)
ax.plot(
    monthly_df['year_month'],
    monthly_df['located_pct_ma'],
    color='blue',
    linestyle='-',
    linewidth=3,
    label='Located (3-mo MA)'
)
ax.plot(
    monthly_df['year_month'],
    monthly_df['mag_pct_ma'],
    color='green',
    linestyle='-',
    linewidth=3,
    label='Magnitude (3-mo MA)'
)

# --- Add right y-axis for Hybrid+Tremor ---
ax2 = ax.twinx()
ax2.plot(
    monthly_df['year_month'],
    monthly_df['ht_pct_ma'],
    color='black',
    linestyle='--',
    linewidth=3,
    label="Hybrid+VT (3-mo MA)"
)

# --- Setup left y-axis ---
ax.set_ylabel('Location & Magnitude Percentage (%)')
ax.set_xlim([start_date, end_date])
#ax.set_yscale('' \
#'log')
ax.set_ylim(1.0, 100)
ax.grid(True, which='both', linestyle='--', alpha=0.7)

# --- Setup right y-axis ---
ax2.set_ylabel('Hybrid+VT Percentage (%)', color='black')
#ax2.set_yscale('log')
ax2.set_ylim(1.0, 100)
ax2.grid(False)


# --- Merge Legends ---
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper center', frameon=True, framealpha=0.7)

# --- X-axis formatting ---
# Only label January 1st each year
ax.xaxis.set_major_locator(mdates.YearLocator(1))  # Tick every January 1st
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


# --- Save figure with tight layout first ---
plt.tight_layout()

'''
# --- THEN manually rotate labels ---
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_horizontalalignment('center')
'''
# Save and Show
plt.savefig(os.path.join(TOP_DIR, 'logscale_percentages_3mo_acausal_moving_average.png'))
plt.show()







# Plot by mag_type
if include_magnitudes:
    for mag_type, group in filtered_df.groupby('mag_type'):
        plt.scatter(group['time'], group['magnitude'], label=mag_type, alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title('Magnitude vs Time (colored by mag_type)')
    plt.legend(title='Mag Type')
    plt.grid(True)
    


    # Set up the plot
    plt.figure(figsize=(10,6))

    # Group by mag_type
    for mag_type, group in filtered_df.groupby('mag_type'):
        group = group.sort_values('time')  # sort within each mag_type
        cumulative_count = group['time'].rank(method='first').astype(int)
        plt.plot(group['time'], cumulative_count, label=mag_type)

    plt.xlabel('Time')
    plt.ylabel('Cumulative Number of Events')
    plt.title('Cumulative Event Count by mag_type')
    plt.legend(title='Mag Type')
    plt.grid(True)
    plt.savefig('cumulative_count_v_time_by_magtype.png')


    # Set up the plot
    plt.figure(figsize=(12,7))

    # Group by subclass
    for subclass, group in filtered_df.groupby('subclass'):
        group = group.sort_values('time')
        group['cumulative_number'] = range(1, len(group)+1)
        plt.plot(group['time'], group['cumulative_number'], label=subclass)

    plt.xlabel('Time')
    plt.ylabel('Cumulative Number of Events')
    plt.title('Cumulative Event Count by Subclass')
    plt.legend(title='Subclass', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_count_v_time_by_subclass.png')


"""
import pygmt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if include_origins:
    # Ensure filtered_df is clean
    valid_subclasses = ['r', 'e', 'l', 'h', 't']
    filtered_df = filtered_df[filtered_df['subclass'].isin(valid_subclasses)].copy()

    # Define region tightly around Montserrat
    region = [-62.3, -62.1, 16.67, 16.83]

    # Assign colors to subclasses
    subclass_colors = {
        'r': 'red',
        'e': 'orange',
        'l': 'blue',
        'h': 'purple',
        't': 'green'
    }

    # Create the first figure for all events
    fig = pygmt.Figure()

    fig.grdimage(
        grid="@earth_relief_01s",
        region=region,
        projection="M6i",
        shading=True,
        frame=["af", "+tEpicenters for located events from MVO DSN"]
    )

    fig.coast(
        shorelines="1/0.5p,black",
        water="lightblue",
        resolution="h"
    )

    for subclass, group in filtered_df.groupby('subclass'):
        fig.plot(
            x=group['longitude'],
            y=group['latitude'],
            style="c0.15c",
            fill=subclass_colors[subclass],
            pen="black",
            label=subclass
        )

    fig.legend(position="JTL+o0.2c/0.2c", box=True)

    fig.basemap(
        map_scale="jTL+w5k+o0.5c/0.5c+f+u"
    )

    fig.plot(
        x=[-62.29],
        y=[16.82],
        style="n0.6c",
        pen="1p,black",
        fill="black"
    )
    fig.text(
        text="N",
        x=-62.29,
        y=16.825,
        font="14p,Helvetica-Bold,black",
        justify="CB"
    )

    fig.show()

    # Create the second figure for subclass medians only
    fig_medians = pygmt.Figure()

    tight_region = [-62.25, -62.15, 16.69, 16.79]

    fig_medians.grdimage(
        grid="@earth_relief_01s",
        region=tight_region,
        projection="M6i",
        shading=True,
        frame=["af", "+tMedian locations by subclass"]
    )

    fig_medians.coast(
        shorelines="1/0.5p,black",
        water="lightblue",
        resolution="h"
    )

    for subclass, group in filtered_df.groupby('subclass'):
        median_lon = group['longitude'].median()
        median_lat = group['latitude'].median()
        fig_medians.plot(
            x=[median_lon],
            y=[median_lat],
            style="a0.4c",
            fill=subclass_colors[subclass],
            pen="black",
            label=subclass
        )

    fig_medians.legend(position="JTL+o0.2c/0.2c", box=True)

    fig_medians.basemap(
        map_scale="jTL+w5k+o0.5c/0.5c+f+u"
    )

    fig_medians.plot(
        x=[-62.29],
        y=[16.82],
        style="n0.6c",
        pen="1p,black",
        fill="black"
    )
    fig_medians.text(
        text="N",
        x=-62.29,
        y=16.825,
        font="14p,Helvetica-Bold,black",
        justify="CB"
    )

    fig_medians.show()

    # Timeseries plot: cumulative number of located events by subclass
    plt.figure(figsize=(10, 6))
    filtered_df['time'] = pd.to_datetime(filtered_df['time'])
    filtered_df = filtered_df.sort_values('time')

    for subclass, group in filtered_df.groupby('subclass'):
        group = group.sort_values('time')
        group['cumulative_count'] = range(1, len(group) + 1)
        plt.plot(group['time'], group['cumulative_count'], label=subclass, color=subclass_colors[subclass])

    plt.xlabel('Time')
    plt.ylabel('Cumulative Number of Events')
    plt.title('Cumulative Number of Located Events by Subclass')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Cross-sections: West-East and North-South
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['longitude'], -filtered_df['depth'], c=filtered_df['subclass'].map(subclass_colors), alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Depth (km)')
    plt.title('West-East Cross Section')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['latitude'], -filtered_df['depth'], c=filtered_df['subclass'].map(subclass_colors), alpha=0.6)
    plt.xlabel('Latitude')
    plt.ylabel('Depth (km)')
    plt.title('North-South Cross Section')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # 3D visualization with zoom and median stars
    from mpl_toolkits.mplot3d import Axes3D

    fig_3d = plt.figure(figsize=(12, 10))
    ax = fig_3d.add_subplot(111, projection='3d')

    sc = ax.scatter(
        filtered_df['longitude'],
        filtered_df['latitude'],
        -filtered_df['depth'],
        c=filtered_df['subclass'].map(subclass_colors),
        alpha=0.7
    )

    # Plot median locations as stars
    for subclass, group in filtered_df.groupby('subclass'):
        median_lon = group['longitude'].median()
        median_lat = group['latitude'].median()
        median_depth = group['depth'].median()
        ax.scatter(median_lon, median_lat, -median_depth, marker='*', s=200, c=subclass_colors[subclass], edgecolor='black')

    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.set_zlabel('Depth (km)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('3D Event Cloud Beneath Topography with Medians')
    ax.invert_zaxis()
    plt.tight_layout()
    plt.show()

import pygmt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D

if include_origins:
    # Ensure filtered_df is clean
    valid_subclasses = ['r', 'e', 'l', 'h', 't']
    filtered_df = filtered_df[filtered_df['subclass'].isin(valid_subclasses)].copy()

    # Define region tightly around Montserrat
    region = [-62.3, -62.1, 16.67, 16.83]

    # Assign colors to subclasses
    subclass_colors = {
        'r': 'red',
        'e': 'orange',
        'l': 'blue',
        'h': 'purple',
        't': 'green'
    }

    # Timeseries plot: cumulative number of located events by subclass
    plt.figure(figsize=(10, 6))
    filtered_df['time'] = pd.to_datetime(filtered_df['time'])
    filtered_df = filtered_df.sort_values('time')

    for subclass, group in filtered_df.groupby('subclass'):
        group = group.sort_values('time')
        group['cumulative_count'] = range(1, len(group) + 1)
        plt.plot(group['time'], group['cumulative_count'], label=subclass, color=subclass_colors[subclass])

    plt.xlabel('Time')
    plt.ylabel('Cumulative Number of Events')
    plt.title('Cumulative Number of Located Events by Subclass')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Cross-sections: West-East and North-South
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['longitude'], -filtered_df['depth'], c=filtered_df['subclass'].map(subclass_colors), alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Depth (km)')
    plt.title('West-East Cross Section')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['latitude'], -filtered_df['depth'], c=filtered_df['subclass'].map(subclass_colors), alpha=0.6)
    plt.xlabel('Latitude')
    plt.ylabel('Depth (km)')
    plt.title('North-South Cross Section')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # 3D visualization with zoom, median stars, ellipsoids, and coastline
    fig_3d = plt.figure(figsize=(12, 10))
    ax = fig_3d.add_subplot(111, projection='3d')

    for subclass, group in filtered_df.groupby('subclass'):
        color = subclass_colors[subclass]
        ax.scatter(group['longitude'], group['latitude'], -group['depth'], c=color, alpha=0.3, label=subclass)

        if len(group) > 100:
            coords = group[['longitude', 'latitude', 'depth']].to_numpy()
            coords[:, 2] *= -1
            pca = PCA(n_components=3)
            pca.fit(coords)
            center = pca.mean_
            radii = np.sqrt(chi2.ppf(0.67, df=3)) * np.sqrt(pca.explained_variance_)
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], pca.components_) + center
            ax.plot_wireframe(x, y, z, color=color, alpha=0.4)
        else:
            median_lon = group['longitude'].median()
            median_lat = group['latitude'].median()
            median_depth = group['depth'].median()
            ax.scatter(median_lon, median_lat, -median_depth, marker='*', s=200, c=color, edgecolor='black')

    # Overlay Montserrat coastline (simple rectangle approximation)
    coastline_lon = [-62.25, -62.15, -62.15, -62.25, -62.25]
    coastline_lat = [16.69, 16.69, 16.79, 16.79, 16.69]
    coastline_depth = [-0.5] * len(coastline_lon)
    ax.plot(coastline_lon, coastline_lat, coastline_depth, color='black', linewidth=2)

    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.set_zlabel('Depth (km)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('3D Event Cloud with 67% Ellipsoids, Median Stars, and Coastline')
    ax.invert_zaxis()
    ax.legend()
    plt.tight_layout()
    plt.show()

# Required packages: pygmt, pandas, matplotlib, numpy, scipy
# Install missing packages if needed:
# pip install pygmt pandas matplotlib numpy scipy
# Required packages: pygmt, pandas, matplotlib, numpy, scipy
# Install missing packages if needed:
# pip install pygmt pandas matplotlib numpy scipy

# Required packages: pygmt, pandas, matplotlib, numpy, scipy
# Install missing packages if needed:
# pip install pygmt pandas matplotlib numpy scipy

# Required packages: pygmt, pandas, matplotlib, numpy, scipy
# Install missing packages if needed:
# pip install pygmt pandas matplotlib numpy scipy

import pygmt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

if include_origins:
    # Ensure filtered_df is clean
    valid_subclasses = ['l', 'h', 't']
    filtered_df = filtered_df[filtered_df['subclass'].isin(valid_subclasses)].copy()

    # Define region tightly around Montserrat
    volcano_lon = -62.18
    volcano_lat = 16.72
    region = [volcano_lon - 0.045, volcano_lon + 0.045, volcano_lat - 0.045, volcano_lat + 0.045]

    # Filter events to only those under Montserrat
    filtered_df = filtered_df[
        (filtered_df['longitude'] >= region[0]) &
        (filtered_df['longitude'] <= region[1]) &
        (filtered_df['latitude'] >= region[2]) &
        (filtered_df['latitude'] <= region[3])
    ].copy()

    # Assign colors to subclasses
    subclass_colors = {
        'r': 'red',
        'e': 'orange',
        'l': 'blue',
        'h': 'purple',
        't': 'green'
    }

    # Timeseries plot: cumulative number of located events by subclass
    plt.figure(figsize=(10, 6))
    filtered_df['time'] = pd.to_datetime(filtered_df['time'])
    filtered_df = filtered_df.sort_values('time')

    for subclass, group in filtered_df.groupby('subclass'):
        group = group.sort_values('time')
        group['cumulative_count'] = range(1, len(group) + 1)
        plt.plot(group['time'], group['cumulative_count'], label=subclass, color=subclass_colors[subclass])

    plt.xlabel('Time')
    plt.ylabel('Cumulative Number of Events')
    plt.title('Cumulative Number of Located Events by Subclass')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Cross-sections: West-East and North-South
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['longitude'], -filtered_df['depth'], c=filtered_df['subclass'].map(subclass_colors), alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Depth (km)')
    plt.title('West-East Cross Section')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['latitude'], -filtered_df['depth'], c=filtered_df['subclass'].map(subclass_colors), alpha=0.6)
    plt.xlabel('Latitude')
    plt.ylabel('Depth (km)')
    plt.title('North-South Cross Section')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # 3D voxel histograms colored by number of events
    for subclass, group in filtered_df.groupby('subclass'):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D grid
        xbins = np.linspace(region[0], region[1], 50)
        ybins = np.linspace(region[2], region[3], 50)
        zbins = np.linspace(0, 8.8, 20)  # Depth from 0 to 8.8 km

        hist, edges = np.histogramdd(
            sample=np.column_stack((group['longitude'], group['latitude'], group['depth'])),
            bins=(xbins, ybins, zbins)
        )

        xpos, ypos, zpos = np.meshgrid(
            (xbins[:-1] + xbins[1:]) / 2,
            (ybins[:-1] + ybins[1:]) / 2,
            (zbins[:-1] + zbins[1:]) / 2,
            indexing="ij"
        )

        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = -zpos.ravel()  # Negative because depth increases downward

        dx = dy = (region[1] - region[0]) / 50
        dz = (zbins[1] - zbins[0])
        values = hist.ravel()

        # Only plot nonzero voxels
        nonzero = values > 0

        # Normalize and map to colors
        norm = plt.Normalize(vmin=values[nonzero].min(), vmax=values[nonzero].max())
        colors = cm.viridis(norm(values[nonzero]))

        ax.bar3d(
            xpos[nonzero], ypos[nonzero], zpos[nonzero],
            dx, dy, dz,
            shade=True,
            color=colors,
            alpha=0.8
        )

        # Add colorbar
        mappable = cm.ScalarMappable(cmap='viridis', norm=norm)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, label='Number of Events per Voxel')

        # Plot approximate coastline manually
        coastline_lon = [-62.25, -62.15, -62.15, -62.25, -62.25]
        coastline_lat = [16.69, 16.69, 16.79, 16.79, 16.69]
        coastline_depth = np.full_like(coastline_lon, 0)
        ax.plot(coastline_lon, coastline_lat, coastline_depth, color='black', linewidth=1)

        # Plot summit marker
        ax.scatter(volcano_lon, volcano_lat, 1.2, mark# Required packages: pygmt, pandas, matplotlib, numpy, scipy
# Install missing packages if needed:
# pip install pygmt pandas matplotlib numpy scipy

import pygmt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

if include_origins:
    # Ensure filtered_df is clean
    valid_subclasses = ['r', 'e', 'l', 'h', 't']
    filtered_df = filtered_df[filtered_df['subclass'].isin(valid_subclasses)].copy()

    # Define region tightly around Montserrat
    volcano_lon = -62.18
    volcano_lat = 16.72
    region = [volcano_lon - 0.045, volcano_lon + 0.045, volcano_lat - 0.045, volcano_lat + 0.045]

    # Filter events to only those under Montserrat
    filtered_df = filtered_df[
        (filtered_df['longitude'] >= region[0]) &
        (filtered_df['longitude'] <= region[1]) &
        (filtered_df['latitude'] >= region[2]) &
        (filtered_df['latitude'] <= region[3])
    ].copy()

    # Assign colors to subclasses
    subclass_colors = {
        'r': 'red',
        'e': 'orange',
        'l': 'blue',
        'h': 'purple',
        't': 'green'
    }

    # Create a 3D plot of subclass medians
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for subclass, group in filtered_df.groupby('subclass'):
        median_lon = group['longitude'].median()
        median_lat = group['latitude'].median()
        median_depth = group['depth'].median()

        ax.scatter(
            median_lon,
            median_lat,
            -median_depth,
            marker='*',
            s=200,
            color=subclass_colors[subclass],
            edgecolor='black',
            label=f'{subclass} median'
        )

    # Plot summit marker
    ax.scatter(
        volcano_lon,
        volcano_lat,
        1.2,
        marker='^',
        s=300,
        color='red',
        edgecolor='black',
        label='Summit'
    )

    # Plot approximate coastline manually
    coastline_lon = [-62.25, -62.15, -62.15, -62.25, -62.25]
    coastline_lat = [16.69, 16.69, 16.79, 16.79, 16.69]
    coastline_depth = np.full_like(coastline_lon, 0)
    ax.plot(coastline_lon, coastline_lat, coastline_depth, color='black', linewidth=1)

    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.set_zlim(-8.8, 1.5)
    ax.set_zlabel('Elevation (km)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('3D Plot of Median Epicenter Positions by Subclass')
    ax.invert_zaxis()
    ax.legend()
    plt.tight_layout()
    plt.show()

        plt.show()
"""

# Required packages: pygmt, pandas, matplotlib, numpy, scipy
# Install missing packages if needed:
# pip install pygmt pandas matplotlib numpy scipy

import pygmt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

if include_origins:
    # Ensure filtered_df is clean
    valid_subclasses = ['r', 'e', 'l', 'h', 't']
    filtered_df = filtered_df[filtered_df['subclass'].isin(valid_subclasses)].copy()

    # Define dome location
    dome_location = {'lat': 16.71111, 'lon': -62.17722}
    volcano_lon = dome_location['lon']
    volcano_lat = dome_location['lat']

    # Define 5 km box around dome
    region = [volcano_lon - 0.045, volcano_lon + 0.045, volcano_lat - 0.045, volcano_lat + 0.045]

    # Filter events to only those under Montserrat
    filtered_df = filtered_df[
        (filtered_df['longitude'] >= region[0]) &
        (filtered_df['longitude'] <= region[1]) &
        (filtered_df['latitude'] >= region[2]) &
        (filtered_df['latitude'] <= region[3])
    ].copy()

    # Assign colors to subclasses
    subclass_colors = {
        'r': 'red',
        'e': 'orange',
        'l': 'blue',
        'h': 'purple',
        't': 'green'
    }

    # Create a 3D plot of subclass medians
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for subclass, group in filtered_df.groupby('subclass'):
        if len(group) > 0:
            median_lon = group['longitude'].median()
            median_lat = group['latitude'].median()
            median_depth = group['depth'].median() / 1000.0  # Convert meters to km

            ax.scatter(
                median_lon,
                median_lat,
                median_depth,
                marker='*',
                s=200,
                color=subclass_colors[subclass],
                edgecolor='black',
                label=f'{subclass} median'
            )

    # Plot summit marker
    ax.scatter(
        volcano_lon,
        volcano_lat,
        1.2,
        marker='^',
        s=300,
        color='red',
        edgecolor='black',
        label='Summit'
    )

    # Drop a vertical line from summit to 8.8 km depth
    ax.plot(
        [volcano_lon, volcano_lon],
        [volcano_lat, volcano_lat],
        [1.2, -8.8],
        color='black',
        linestyle='--',
        linewidth=1.5
    )

    # Plot approximate coastline manually
    coastline_lon = [-62.25, -62.15, -62.15, -62.25, -62.25]
    coastline_lat = [16.69, 16.69, 16.79, 16.79, 16.69]
    coastline_depth = np.full_like(coastline_lon, 0)
    ax.plot(coastline_lon, coastline_lat, coastline_depth, color='black', linewidth=1)

    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.set_zlim(1.5, -8.8)
    ax.set_zlabel('Elevation (km)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('3D Plot of Median Epicenter Positions by Subclass')
    ax.legend()
    plt.tight_layout()
    plt.show()
