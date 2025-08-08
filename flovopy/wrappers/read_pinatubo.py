from flovopy.processing.sam import RSAM
import obspy
import os
from glob import glob
from flovopy.core.trace_utils import _fix_legacy_id
starttime = obspy.UTCDateTime(1991,1,1)
endtime = obspy.UTCDateTime(1991,12,31,23,59,59)
RSAM_DIR = '/Users/thompsong/Dropbox/Pinatubo/from_tom/RSAM/RSAM'
for rsamfile in glob(os.path.join(RSAM_DIR, '*.DAT')):
    try:
        rsamobj = RSAM.readRSAMbinary(filepath=rsamfile, stime=starttime, etime=endtime, sampling_interval=600)
        
        station = os.path.basename(rsamfile).split('91')[0]
        trace_id = rsamobj._SAM__get_trace_ids()[0]
        tr = obspy.Trace(header={'sampling_rate':1/600})
        tr.id = f'.{station}..'
        _fix_legacy_id(tr, network='1R')
        rsamobj.dataframes[tr.id] = rsamobj.dataframes.pop(trace_id)
        print(rsamobj)
        rsamobj.write(RSAM_DIR, ext='csv')
        year = starttime.year
        rsamobj.plot(outfile=f'{tr.id}_{year}.png')
    except Exception as e:
        print(f'Failed on {rsamfile} - {e}')
