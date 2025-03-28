from flovopy.core.sam import VSAM, VSEM, DSAM
from flovopy.sds.sds import SDSobj
SECONDS_PER_DAY = 86400

def compute_velocity_metrics(sds_dir, sam_dir, start_time, end_time, sampling_interval, net=None, ext='pickle'):
    sds = SDSobj(sds_dir)
    day = start_time
    while day < end_time:
        sds.read(day, day + SECONDS_PER_DAY)
        st = sds.stream.select(network=net) if net else sds.stream
        for tr in st:
            if tr.stats.channel[1] == 'H':
                tr.stats['units'] = 'm/s'
        if sampling_interval:
            for delta in ([sampling_interval] if isinstance(sampling_interval, (int, float)) else sampling_interval):
                if VSAM.read(day, day + SECONDS_PER_DAY, sam_dir, sampling_interval=delta, ext=ext).__size__()[1] == 0:
                    VSAM(stream=st, sampling_interval=delta).write(sam_dir, ext=ext)
                if VSEM.read(day, day + SECONDS_PER_DAY, sam_dir, sampling_interval=delta, ext=ext).__size__()[1] == 0:
                    VSEM(stream=st, sampling_interval=delta).write(sam_dir, ext=ext)
        day += SECONDS_PER_DAY


def compute_displacement_metrics(sds_dir, sam_dir, start_time, end_time, sampling_interval, net=None, ext='pickle'):
    sds = SDSobj(sds_dir)
    day = start_time
    while day < end_time:
        sds.read(day, day + SECONDS_PER_DAY)
        st = sds.stream.select(network=net) if net else sds.stream
        for tr in st:
            if tr.stats.channel[1] == 'H':
                tr.stats['units'] = 'm'
        for delta in ([sampling_interval] if isinstance(sampling_interval, (int, float)) else sampling_interval):
            if DSAM.read(day, day + SECONDS_PER_DAY, sam_dir, sampling_interval=delta, ext=ext).__size__()[1] == 0:
                DSAM(stream=st, sampling_interval=delta).write(sam_dir, ext=ext)
        day += SECONDS_PER_DAY