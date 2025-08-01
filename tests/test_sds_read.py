from flovopy.sds.sds import SDSobj
import obspy
from flovopy.core.miniseed_io import unmask_gaps
from flovopy.core.trace_utils import streams_equal
import numpy as np



def unmask(st):
    for tr in st:
        unmask_gaps(tr)
def count(tr):
    count = np.count_nonzero(~np.isnan(tr.data) & (tr.data != 0))
    expected_npts = tr.stats.sampling_rate * (etime-stime)
    print(f'{tr.id}: data values: {count} which is {count/expected_npts * 100} complete')    
def compare(st1, st2):
    for tr in st1:
        print()
        subset2 = st2.select(id=tr.id)
        identical = False
        if len(subset2)==1:
            tr2 = subset2[0]
            identical = np.array_equal(tr.data, tr2.data)
        count(tr)
        if not identical:
            if len(subset2)==0:
                print('- got no matching trace')
            else:
                print(subset2)
                for tr2 in subset2:
                    count(tr2)


#sdsin=SDSobj('/data/KSC/beforePASSCAL/CONTINUOUS')
sdsin=SDSobj('/raid/data/SDS_Montserrat')
tstart = obspy.UTCDateTime()
stime = obspy.UTCDateTime(2001,2,23,0,0,1)
etime = obspy.UTCDateTime(2001,2,23,23,59,59)
outdir = '/home/thompsong/work/tmp'
import os
'''
trace_ids = sdsin._get_nonempty_traceids(stime, etime, skip_low_rate_channels=True, speed=1)
print(trace_ids)

avail_df, trace_ids = sdsin.get_percent_availability(stime, etime, skip_low_rate_channels=True,
                                trace_ids=trace_ids, speed=3, verbose=False, progress=True, merge_strategy='obspy')
print(avail_df)

sdsin.plot_availability(avail_df, outfile=os.path.join(outdir,'availability.png'), figsize=(12, 8), fontsize=10, labels=None, cmap='viridis')

#print(trace_ids)
'''
streams = []
for speed in [1,2]:
    file = os.path.join(outdir, f'st{speed}.mseed')
    if os.path.isfile(file):
        streams.append(obspy.read(file,  format='mseed'))
    else:
        sdsin.read(stime, etime, speed=speed)
        unmask(sdsin.stream)
        streams.append(sdsin.stream.copy())
        sdsin.stream.write(file, format='mseed')

#compare(st1, st2)
if len(streams)==2:

    if streams_equal(streams[0], streams[1], verbose=True):
        for tr in streams[0]:
            count(tr)  
    else:
        print('streams are different')
