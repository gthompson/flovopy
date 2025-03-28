import os
import numpy as np
from obspy import read_inventory, Stream
from flovopy.sds.sds import SDSobj
from flovopy.core.inventory import inventory2traceid
from flovopy.core.preprocessing import preprocess_trace

SECONDS_PER_DAY = 86400

def sds_to_disp(sds_dir, disp_dir, invfile, start_time, end_time):
    sds = SDSobj(sds_dir)
    out = SDSobj(disp_dir)
    net = os.path.basename(invfile)[:2]
    inv = read_inventory(invfile)
    ids = inventory2traceid(inv, force_location_code='')

    day = start_time
    while day < end_time:
        if day not in out.find_which_days_missing(start_time, end_time, net):
            day += SECONDS_PER_DAY
            continue

        sds.read(day - 4320, day + SECONDS_PER_DAY + 4320, fixnet=net)
        sds.stream.merge(method=0, fill_value=0)

        for tr in sds.stream:
            trid = '.'.join([tr.stats.network, tr.stats.station, '', tr.stats.channel])
            if tr.id in ids or trid in ids:
                try:
                    tr_ = tr.copy()
                    preprocess_trace(tr_, bool_despike=True, bool_clean=True, inv=inv, filterType="bandpass", freq=[0.01, 30.0], outputType="DISP")
                    Stream(traces=[tr_]).trim(day, day + SECONDS_PER_DAY - 0.01)
                    out.stream = Stream([tr_])
                    out.write(overwrite=False, merge=False)
                except Exception as e:
                    print(e)

        day += SECONDS_PER_DAY