import glob
import os
import pandas as pd
from obspy import read_events, UTCDateTime

TOPDIR = '/data/b18_waveform_processing'
all = sorted(glob.glob(os.path.join(TOPDIR, "*MVO*")))
lod = []
for thisdir in all:
    mapfile = glob.glob(os.path.join(thisdir, 'map_Q100*.png'))
    if len(mapfile)==1:
        if os.path.isfile(mapfile[0]):
            dr = pd.read_csv(os.path.join(thisdir, 'database_row.csv'))
            #print(dr)
            sm = pd.read_csv(os.path.join(thisdir, 'stream_metrics.csv'))
            #print(sm)
            mag = pd.read_csv(os.path.join(thisdir, 'magnitudes.csv'))
            #print(mag['ME']) 
            qmlfile = glob.glob(os.path.join(thisdir, 'event*qml'))
            print(qmlfile)
            if len(qmlfile)==1:
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
df = pd.DataFrame(lod)
df.to_csv(f'/home/thompsong/Dropbox/ASL_results_{int(UTCDateTime().timestamp)}.csv')
