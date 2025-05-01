from obspy import read_inventory
import sqlite3
import pandas as pd
from obspy import UTCDateTime
from flovopy.analysis.asl import ASL, montserrat_topo_map, plot_heatmap_montserrat_colored  # assuming ASL class is imported
import numpy as np

def build_station_coordinates(inventory):
    """Build a dictionary of station coordinates."""
    coords = {}
    for net in inventory:
        for sta in net:
            for cha in sta:
                coords[f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"] = {
                    'latitude': cha.latitude,
                    'longitude': cha.longitude
                }
    return coords

def reconstruct_asl_from_database(event_id, db_path, inventory):
    global SCALING_FACTOR
    conn = sqlite3.connect(db_path)

    origins = pd.read_sql_query(f"""
        SELECT * FROM origin WHERE event_id = '{event_id}'
    """, conn)

    conn.close()

    if origins.empty:
        raise ValueError(f"No origins found for event_id {event_id}")

    source = {
        't': [UTCDateTime(t) for t in pd.to_datetime(origins['origin_time']).to_list()],
        'lat': origins['latitude'].to_numpy(),
        'lon': origins['longitude'].to_numpy(),
        'DR': origins['amplitude'].fillna(0).to_numpy() * SCALING_FACTOR,  # scale like ASL
        'misfit': origins['rms_error'].fillna(0).infer_objects(copy=False).to_numpy(),
        'azgap': origins['azimuthal_gap'].fillna(180).infer_objects(copy=False).to_numpy(),
        'nsta': origins['num_stations'].fillna(0).infer_objects(copy=False).astype(int).to_numpy()
    }

    asl = ASL.__new__(ASL)
    asl.source = source
    asl.inventory = inventory
    asl.station_coordinates = build_station_coordinates(inventory)
    asl.located = True

    return asl


import pygmt
import numpy as np

import pandas as pd

def plot_day_of_origins(origins_df, day, inventory, outfile, region):
    global SCALING_FACTOR

    # Title for the day
    title = day.strftime("%Y-%m-%d")

    # Create the base topo map
    fig = montserrat_topo_map(
        show=False,
        inv=inventory,
        stations=[],
        topo_color=True,
        add_labels=False,
        zoom_level=1,
        title=title,
        region=region
    )

    if origins_df.empty:
        # If no events, write a centered text
        fig.text(
            x=(region[0] + region[1]) / 2,
            y=(region[2] + region[3]) / 2,
            text=f"No events on {day.strftime('%Y-%m-%d')}",
            font="18p,Helvetica-Bold,black",
            justify="MC"
        )
    else:
        # Build a points table: longitude, latitude, size
        points = pd.DataFrame({
            "longitude": origins_df["longitude"],
            "latitude": origins_df["latitude"],
            "size": origins_df["amplitude"].fillna(0) * SCALING_FACTOR/50
        })

        # Clip absurdly small or large sizes (optional)
        points["size"] = points["size"].clip(lower=0.05, upper=1.0)

        # Plot events as circles (fixed size from 'size' column)
        fig.plot(
            data=points[["longitude", "latitude", "size"]],
            style="c",
            fill="red",
            pen="black"
        )


    # Lock map view to exact region
    fig.basemap(region=region, frame=True)

    # Save figure
    fig.savefig(outfile)



import os

def make_per_event_movie_frames(db_path, inventory, output_dir, region):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    event_ids = pd.read_sql_query("SELECT DISTINCT event_id FROM event", conn)['event_id'].tolist()
    conn.close()

    for idx, event_id in enumerate(event_ids):
        #try:
        asl = reconstruct_asl_from_database(event_id, db_path, inventory)
        if isinstance(asl, ASL):
            #title = f"{asl.source['t'][0].strftime('%Y-%m-%dT%H:%M')} | {asl.source['nsta'][0]} stations | Gap {asl.source['azgap'][0]:.1f}°"
            title = f"{asl.source['t'][0].strftime('%Y-%m-%dT%H:%M')} "
            outfile = os.path.join(output_dir, f"frame_event_{idx:05d}.png")
            try:
                asl.plot(zoom_level=0, scale=0.1, join=True, equal_size=False, add_labels=False, \
                        outfile=outfile, title=title, region=region, normalize=False)
                print('Got here 6')
            except Exception as e:
                print(e)
                continue
            print(f"[✓] Frame {idx} for event {event_id} saved to {outfile}")

        #except Exception as e:
        #    print(f"[!] Failed for event {event_id}: {e}")

import datetime

def make_per_day_movie_frames(db_path, inventory, output_dir, region):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    origins = pd.read_sql_query("SELECT * FROM origin", conn)
    conn.close()

    origins['origin_time'] = pd.to_datetime(origins['origin_time'])
    origins['day'] = origins['origin_time'].dt.date

    days = sorted(origins['day'].unique())

    for idx, day in enumerate(days):
        day_origins = origins[origins['day'] == day]
        outfile = os.path.join(output_dir, f"frame_day_{idx:05d}.png")
        plot_day_of_origins(day_origins, pd.Timestamp(day), inventory, outfile, region)
        print(f"[✓] Frame {idx} for day {day}")

# === START ===

def generate_movies(region, inventory):


    print("[INFO] Making per-event frames...")
    make_per_event_movie_frames(DB_PATH, inventory, EVENT_FRAMES_DIR, region)

    print("[INFO] Making per-day frames...")
    make_per_day_movie_frames(DB_PATH, inventory, DAY_FRAMES_DIR, region)

    print("[INFO] Building per-event movie...")
    os.system(f"ffmpeg -y -framerate {FRAMERATE_EVENT} -pattern_type glob -i '{EVENT_FRAMES_DIR}/frame_event_*.png' -c:v libx264 -pix_fmt yuv420p10le -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 30 {EVENT_MOVIE}")

    print("[INFO] Building per-day movie...")
    os.system(f"ffmpeg -y -framerate {FRAMERATE_DAY} -pattern_type glob -i '{DAY_FRAMES_DIR}/frame_day_*.png' -c:v libx264 -pix_fmt yuv420p10le -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 30 {DAY_MOVIE}")

    print("\n[✓] All done! Movies generated.")


import pandas as pd
import os

def make_weekly_heatmaps(db_path, inventory, output_dir, start_time, end_time, region=None):
    """
    Create weekly heatmaps from ASL origin database.

    Parameters:
    - db_path: Path to SQLite database with 'origin' table
    - inventory: ObsPy Inventory object
    - output_dir: Folder to save weekly heatmaps
    - start_time: UTCDateTime or pandas.Timestamp (start)
    - end_time: UTCDateTime or pandas.Timestamp (end)
    - region: Optional [minlon, maxlon, minlat, maxlat] to lock the view
    """
    from flovopy.analysis.asl import montserrat_topo_map  # make sure this is imported earlier
    os.makedirs(output_dir, exist_ok=True)

    # Load all origins
    conn = sqlite3.connect(db_path)
    origins = pd.read_sql_query("SELECT * FROM origin", conn)
    conn.close()

    # Convert time to pandas datetime
    origins['origin_time'] = pd.to_datetime(origins['origin_time'])

    # Filter by time range
    mask = (origins['origin_time'] >= pd.Timestamp(start_time)) & (origins['origin_time'] <= pd.Timestamp(end_time))
    origins = origins.loc[mask]

    # Generate week-by-week
    current_start = pd.Timestamp(start_time)
    idx = 0

    while current_start < pd.Timestamp(end_time):
        current_end = current_start + pd.Timedelta(days=7)

        week_data = origins[
            (origins['origin_time'] >= current_start) &
            (origins['origin_time'] < current_end)
        ]

        if not week_data.empty:
            outfile = os.path.join(output_dir, f"heatmap_week_{idx:04d}.png")
            title = f"Heatmap {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}"

            fig = plot_heatmap_montserrat_colored(
                week_data,
                lat_col='latitude',
                lon_col='longitude',
                amp_col='amplitude',
                zoom_level=1,
                inventory=inventory,
                color_scale=0.4,
                cmap='turbo',
                log_scale=True,
                contour=True,
                node_spacing_m=50,
                outfile=outfile,
                region=region,
                title=title
            )

            print(f"[✓] Saved weekly heatmap {outfile}")
        
        else:
            print(f"[i] No events between {current_start} and {current_end}")

        current_start = current_end
        idx += 1


# === CONFIG ===
DB_PATH = '/home/thompsong/Dropbox/ampmap_events.db'
INVENTORY_PATH = "/data/SEISAN_DB/CAL/MV.xml"

EVENT_FRAMES_DIR = "/home/thompsong/Dropbox/ampmap_event_frames"
DAY_FRAMES_DIR = "/home/thompsong/Dropbox/ampmap_day_frames"

EVENT_MOVIE = "/home/thompsong/Dropbox/ampmap_per_event_movie.mp4"
DAY_MOVIE = "/home/thompsong/Dropbox/ampmap_per_day_movie.mp4"
HEATMAP_MOVIE = "/home/thompsong/Dropbox/heatmap_movie.mp4"

FRAMERATE_EVENT = 15  # frames per second
FRAMERATE_DAY = 1
region=[-62.2082, -62.1382, 16.6861, 16.7461]
SCALING_FACTOR = 50

print("[INFO] Loading inventory...")
inventory = read_inventory(INVENTORY_PATH)

# Run it!
if __name__ == "__main__":
    #generate_movies(region, inventory)
    output_dir="/home/thompsong/Dropbox/heatmap_weekly_frames"
    make_weekly_heatmaps(
        db_path="/home/thompsong/Dropbox/ampmap_events.db",
        inventory=inventory,
        output_dir=output_dir,
        start_time=pd.Timestamp("2001-02-09", tz="UTC"),
        end_time=pd.Timestamp("2001-06-14", tz="UTC"),
        region=region  # Optional
    )

    print("[INFO] Building per-day movie...")
    os.system(f"ffmpeg -y -framerate {FRAMERATE_DAY} -pattern_type glob -i '{output_dir}/heatmap_week_*.png' -c:v libx264 -pix_fmt yuv420p10le -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 30 {HEATMAP_MOVIE}")

