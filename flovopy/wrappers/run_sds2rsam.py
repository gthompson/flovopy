from obspy import UTCDateTime
from flovopy.sds.sds import SDSobj
from flovopy.core.sam import RSAM

SECONDS_PER_DAY = 86400

def sds_to_rsam(sds_dir, sam_dir, start_time, end_time, sampling_interval=60, net=None):
    sds = SDSobj(sds_dir)
    day = start_time
    while day < end_time:
        rsam_day = RSAM.read(day, day + SECONDS_PER_DAY, sam_dir, sampling_interval=sampling_interval, ext='pickle')
        if rsam_day.__size__()[1] > 0:
            day += SECONDS_PER_DAY
            continue

        sds.read(day, day + SECONDS_PER_DAY)
        st = sds.stream.select(network=net) if net else sds.stream

        if isinstance(sampling_interval, list):
            for delta in sampling_interval:
                RSAM(stream=st, sampling_interval=delta).write(sam_dir, ext='pickle')
        else:
            RSAM(stream=st, sampling_interval=sampling_interval).write(sam_dir, ext='pickle')

        day += SECONDS_PER_DAY

import argparse
from obspy import UTCDateTime
from flovopy.wrappers.sds2rsam import sds_to_rsam

def main():
    parser = argparse.ArgumentParser(description="Compute RSAM from SDS archive")

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
    parser.add_argument("--rsam_top", type=str, default="RSAM",
                        help="Output directory for RSAM files")
    parser.add_argument("--sampling_interval", type=float, default=60.0,
                        help="Sampling interval in seconds for RSAM (default: 60)")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing RSAM files")

    args = parser.parse_args()

    sds_to_rsam(
        start=UTCDateTime(args.start),
        end=UTCDateTime(args.end),
        sds_top=args.sds,
        trace_ids=args.trace_ids,
        inventory=args.inventory,
        rsam_top=args.rsam_top,
        sampling_interval=args.sampling_interval,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()

"""
run-sds2rsam --start 2023-01-01T00:00:00 --end 2023-01-01T01:00:00 \
             --sds /path/to/sds --inventory metadata/station.xml \
             --trace_ids XX.STA..BHZ --rsam_top RSAM

"""