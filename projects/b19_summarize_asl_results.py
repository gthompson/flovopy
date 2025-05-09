import glob
import os
import pandas as pd
from obspy import read_events, UTCDateTime

TOPDIR = '/data/b18_waveform_processing'
lod = []

# --- Walk through all subdirectories ---
for root, dirs, files in os.walk(TOPDIR):
    for d in dirs:
        if d.startswith('MVO'):
            thisdir = os.path.join(root, d)

            mapfile = glob.glob(os.path.join(thisdir, 'map_Q100*.png'))
            if len(mapfile) == 1:
                if os.path.isfile(mapfile[0]):
                    try:
                        dr = pd.read_csv(os.path.join(thisdir, 'database_row.csv'))
                        sm = pd.read_csv(os.path.join(thisdir, 'stream_metrics.csv'))
                        mag = pd.read_csv(os.path.join(thisdir, 'magnitudes.csv'))
                        qmlfile = glob.glob(os.path.join(thisdir, 'event*qml'))

                        if len(qmlfile) == 1:
                            if os.path.isfile(qmlfile[0]):
                                cat = read_events(qmlfile[0])
                                ev = cat.events[0]
                                r = dict()
                                for i, origin in enumerate(ev.origins):
                                    r['time'] = origin['time']
                                    r['latitude'] = origin['latitude']
                                    r['longitude'] = origin['longitude']
                                    r['amplitude'] = ev.amplitudes[i].generic_amplitude
                                    lod.append(r)
                    except Exception as e:
                        print(f"Problem processing {thisdir}: {e}")

# --- Save output ---
df = pd.DataFrame(lod)
outpath = f'/home/thompsong/Dropbox/ASL_results_{int(UTCDateTime().timestamp)}.csv'
df.to_csv(outpath, index=False)
print(f"Saved dataframe to {outpath}")
