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