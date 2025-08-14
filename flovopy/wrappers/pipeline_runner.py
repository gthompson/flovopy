# flovopy/pipeline/pipeline_runner.py

from obspy import UTCDateTime
from flovopy.pipeline.check_data_requirements import check_what_to_do
from flovopy.pipeline.compute_metrics import compute_raw_metrics, compute_SDS_DISP, compute_velocity_metrics, compute_displacement_metrics
from flovopy.pipeline.reduce_metrics import reduce_to_1km
from flovopy.pipeline.seisan_to_sds import seisandb2SDS

def small_sausage(paths, startt, endt, sampling_interval=60, source=None, invfile=None, Q=None, ext='pickle', net=None, do_metric=None):
    """
    Run core data processing pipeline assuming SDS archive already exists.
    Includes RSAM, VSAM, VSEM, DSAM and reduced metrics (VR, ER, DR).
    """
    if not do_metric:
        do_metric = check_what_to_do(paths, net, startt, endt, sampling_interval=sampling_interval, ext=ext, invfile=invfile)

    if do_metric.get('RSAM'):
        compute_raw_metrics(paths, startt, endt, sampling_interval=sampling_interval, do_RSAM=True, net=net)

    if invfile and os.path.isfile(invfile):
        if do_metric.get('SDS_DISP'):
            compute_SDS_DISP(paths, startt, endt, invfile)

        if do_metric.get('VSAM') or do_metric.get('VSEM'):
            compute_velocity_metrics(paths, startt, endt, sampling_interval=sampling_interval, do_VSAM=do_metric.get('VSAM', True), do_VSEM=do_metric.get('VSEM', True), net=net, ext=ext)

        if do_metric.get('DSAM'):
            compute_displacement_metrics(paths, startt, endt, sampling_interval=sampling_interval, do_DSAM=True, net=net, ext=ext)

        if source:
            if not isinstance(sampling_interval, list):
                sampling_interval = [sampling_interval]
            for delta in sampling_interval:
                if do_metric.get('ER') or do_metric.get('DR') or do_metric.get('DRS'):
                    reduce_to_1km(paths, startt.year, do_ER=do_metric.get('ER'), do_DR=do_metric.get('DR'), do_DRS=do_metric.get('DRS'),
                                  sampling_interval=delta, invfile=invfile, source=source, Q=Q, ext=ext)

def big_sausage(seisandbdir, paths, startt, endt, sampling_interval=60, source=None, invfile=None, Q=None, ext='pickle', dbout=None,
                 round_sampling_rate=True, net=None, do_metric=None, MBWHZ_only=False):
    """
    Full pipeline including Seisan to SDS conversion and all derived metrics.
    Suitable for building a complete archive from scratch.
    """
    

    if not do_metric:
        do_metric = check_what_to_do(paths, net, startt, endt, sampling_interval=sampling_interval, ext=ext, invfile=invfile)

    if do_metric.get('SDS_RAW'):
        seisandb2SDS(seisandbdir, paths['SDS_DIR'], startt, endt, net, dbout=dbout, round_sampling_rate=round_sampling_rate, MBWHZ_only=MBWHZ_only)

    small_sausage(paths, startt, endt, sampling_interval=sampling_interval, source=source, invfile=invfile, Q=Q,
                  ext=ext, net=net, do_metric=do_metric)


if __name__ == "__main__":
    import os
    import setup_paths
    from flovopy.core.mvo import dome_location

    paths = setup_paths.paths
    seisandbdir =  '/data/SEISAN_DB/WAV/DSNC_'
    net = 'MV'
    invfile = os.path.join(paths['RESPONSE_DIR'], f"{net}.xml")
    startt = UTCDateTime(2001, 7, 28, 0, 0, 0)
    endt = UTCDateTime(2001, 7, 31, 0, 0, 0)
    sampling_interval = 10  # or [2.56, 10, 60, 600]
    source = dome_location
    dbout = os.path.join(paths['DB_DIR'], f"dbMontserrat{startt.year}")
    Q = None
    ext = 'pickle'

    do_metric = check_what_to_do(paths, net, startt, endt, sampling_interval=sampling_interval, ext=ext)

    big_sausage(seisandbdir, paths, startt, endt, sampling_interval=sampling_interval, source=source,
                invfile=invfile, Q=Q, ext=ext, dbout=dbout, round_sampling_rate=True, net=net, do_metric=do_metric)
