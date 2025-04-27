from obspy import read_inventory
import sqlite3
import pandas as pd
from obspy import UTCDateTime
from flovopy.analysis.asl import ASL, montserrat_topo_map  # assuming ASL class is imported
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

def plot_day_of_origins(origins_df, day, inventory, outfile, region):
    global SCALING_FACTOR
    title = day.strftime("%Y-%m-%d")
    fig = montserrat_topo_map(
        show=False,
        inv=inventory,
        stations=[],  # Show all stations
        topo_color=True,
        add_labels=False,
        zoom_level=1,
        title=title,
        region=region
    )

    if origins_df.empty:
        fig.text(x=-62.17, y=16.72, text=f"No events on {day}", font="18p,Helvetica-Bold,black", justify="MC")
    else:
        lons = origins_df['longitude']
        lats = origins_df['latitude']
        amps = origins_df['amplitude'].fillna(0) * SCALING_FACTOR
        sizes = np.sqrt(amps + 1e-10)
        #sizes = (sizes / sizes.max()) * 0.1  # scale

        fig.plot(x=lons, y=lats, style="a"+(sizes.astype(str)+"c"), fill="red", pen="black")

    if region:
        fig.basemap(region=region, frame=True)
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

def make_per_day_movie_frames(db_path, inventory, output_dir):
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

def generate_movies(region):
    print("[INFO] Loading inventory...")
    inventory = read_inventory(INVENTORY_PATH)

    print("[INFO] Making per-event frames...")
    make_per_event_movie_frames(DB_PATH, inventory, EVENT_FRAMES_DIR, region)

    print("[INFO] Making per-day frames...")
    make_per_day_movie_frames(DB_PATH, inventory, DAY_FRAMES_DIR, region)

    print("[INFO] Building per-event movie...")
    os.system(f"ffmpeg -y -framerate {FRAMERATE} -pattern_type glob -i '{EVENT_FRAMES_DIR}/frame_event_*.png' -c:v libx264 -pix_fmt yuv420p -r 30 {EVENT_MOVIE}")

    print("[INFO] Building per-day movie...")
    os.system(f"ffmpeg -y -framerate {FRAMERATE} -pattern_type glob -i '{DAY_FRAMES_DIR}/frame_day_*.png' -c:v libx264 -pix_fmt yuv420p -r 30 {DAY_MOVIE}")

    print("\n[✓] All done! Movies generated.")


# === CONFIG ===
DB_PATH = '/home/thompsong/Dropbox/ampmap_events.db'
INVENTORY_PATH = "/data/SEISAN_DB/CAL/MV.xml"

EVENT_FRAMES_DIR = "/home/thompsong/Dropbox/ampmap_event_frames"
DAY_FRAMES_DIR = "/home/thompsong/Dropbox/ampmap_day_frames"

EVENT_MOVIE = "/home/thompsong/Dropbox/ampmap_per_event_movie.mp4"
DAY_MOVIE = "/home/thompsong/Dropbox/ampmap_per_day_movie.mp4"

FRAMERATE = 1  # frames per second
region=[-62.2082, -62.1382, 16.6861, 16.7461]
SCALING_FACTOR = 50

# Run it!
if __name__ == "__main__":
    generate_movies(region)

'''
# Per-event movie
ffmpeg -framerate 1 -pattern_type glob -i 'event_frames/frame_event_*.png' -c:v libx264 -pix_fmt yuv420p -r 30 per_event_movie.mp4

# Per-day movie
ffmpeg -framerate 1 -pattern_type glob -i 'day_frames/frame_day_*.png' -c:v libx264 -pix_fmt yuv420p -r 30 per_day_movie.mp4
'''