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

import argparse
from obspy import UTCDateTime
from flovopy.wrappers.sds2disp import sds_to_disp

def main():
    parser = argparse.ArgumentParser(description="Compute Displacement metrics from SDS archive")

    parser.add_argument("--start", type=str, required=True,
                        help="Start time in UTC (e.g., 2023-01-01T00:00:00)")
    parser.add_argument("--end", type=str, required=True,
                        help="End time in UTC (e.g., 2023-01-02T00:00:00)")
    parser.add_argument("--sds", type=str, required=True,
                        help="Path to SDS archive top-level directory")
    parser.add_argument("--inventory", type=str, required=True,
                        help="Path to StationXML file")
    parser.add_argument("--trace_ids", type=str, nargs='+', required=True,
                        help="List of N.S.L.C. trace IDs (e.g., XX.STA..BHZ)")
    parser.add_argument("--disp_top", type=str, default="DRS",
                        help="Output directory for displacement/DRS files")
    parser.add_argument("--sampling_interval", type=float, default=60.0,
                        help="Sampling interval in seconds (default: 60)")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing output files")

    args = parser.parse_args()

    sds_to_disp(
        start=UTCDateTime(args.start),
        end=UTCDateTime(args.end),
        sds_top=args.sds,
        trace_ids=args.trace_ids,
        inventory=args.inventory,
        disp_top=args.disp_top,
        sampling_interval=args.sampling_interval,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()
"""
run-sds2disp --start 2023-01-01T00:00:00 --end 2023-01-01T01:00:00 \
             --sds /path/to/sds --inventory metadata/station.xml \
             --trace_ids XX.STA..BHZ --disp_top DRS

"""