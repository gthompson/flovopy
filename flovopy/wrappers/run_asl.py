import os
import numpy as np
import pandas as pd
from obspy import Stream, Inventory
from flovopy.processing.sam import DSAM, DRS, RSAM
from obsolete.asl import ASL, initial_source, make_grid, dome_location
from flovopy.processing.detection import run_event_detection, detection_snr, plot_detected_stream, run_monte_carlo, detect_network_event
#from flovopy.processing.metrics import signal2noise

def asl_event(st: Stream, raw_st: Stream, inv: Inventory, **kwargs):
    """
    Runs amplitude-based source location (ASL) processing on a preprocessed Stream object.

    Parameters
    ----------
    st : obspy.Stream
        Preprocessed stream.
    raw_st : obspy.Stream
        Raw stream for comparison/archival.
    inv: obspy.Inventory
        Inventory object with station metadata.
    kwargs : dict
        Options controlling ASL processing (Q, peakf, metric, surfaceWaveSpeed_kms, etc.).
    """
    if len(st) == 0 or not isinstance(st, Stream):
        print("[WARNING] Empty or invalid stream passed to asl_event")
        return

    # --- Parse kwargs with defaults ---
    Q = kwargs.get("Q", [23])
    surfaceWaveSpeed_kms = kwargs.get("surfaceWaveSpeed_kms", [2.5])
    peakf = kwargs.get("peakf", [8.0])
    metric = kwargs.get("metric", ['VT'])
    window_seconds = kwargs.get("window_seconds", 5)
    min_stations = kwargs.get("min_stations", 5)
    outdir = os.path.join(kwargs.get("outdir", "."), st[0].stats.starttime.strftime("%Y%m%dT%H%M%S"))
    interactive = kwargs.get("interactive", False)
    freq = [1.0, 15.0]
    compute_DRS_at_fixed_source = kwargs.get("compute_DRS_at_fixed_source", True)
    numtrials = kwargs.get("numtrials", 200)

    if not isinstance(Q, list):
        Q = [Q, 1000]
    if not isinstance(peakf, list):        
        peakf = [peakf]
    if not isinstance(surfaceWaveSpeed_kms, list):
        surfaceWaveSpeed_kms = [surfaceWaveSpeed_kms]
    if not isinstance(metric, list):
        metric = [metric]

    os.makedirs(outdir, exist_ok=True)

    if raw_st:
        raw_st.plot(equal_scale=False, outfile=os.path.join(outdir, "rawstream.png"))

    st.plot(equal_scale=False, outfile=os.path.join(outdir, "stream.png"))
    st.write(os.path.join(outdir, "stream.mseed"), format="MSEED")

    trialscsv = os.path.join(outdir, 'detection_trials.csv')
    if os.path.isfile(trialscsv):
        all_params_df = pd.read_csv(trialscsv)
        best_params = all_params_df.loc[0].to_dict()
    else:
        if interactive:
            st, best_params, all_params_df = run_event_detection(st, n_trials=numtrials)
        else:
            all_params_df, best_params = run_monte_carlo(st, st[0].stats.starttime + 5.0,
                                                     st[0].stats.endtime - 5.0, numtrials)

        all_params_df.sort_values(by="misfit").head(20).to_csv(trialscsv)

    best_trig = detect_network_event(
        st, minchans=None, threshon=best_params['thr_on'], threshoff=best_params['thr_off'],
        sta=best_params['sta'], lta=best_params['lta'], pad=0.0, best_only=True,
        freq=freq, algorithm=best_params['algorithm'], criterion='cft'
    )

    if not best_trig or len(best_trig['trace_ids']) < min_stations:
        print("[INFO] No valid detection or too few stations.")
        return

    detected_st = Stream(tr for tr in st if tr.id in best_trig['trace_ids'])
    plot_detected_stream(detected_st, best_trig, outfile=os.path.join(outdir, "detected_stream.png"))

    snr, fmetrics_dict = detection_snr(detected_st, best_trig, outfile=os.path.join(outdir, "signal2noise.png"))
    print(f'[INFO] Signal-to-noise ratio: {snr}')
    print([tr.stats.metrics.snr_std for tr in detected_st])

    if fmetrics_dict:
        freq = [fmetrics_dict['f_low'], fmetrics_dict['f_high']]
        peakf = [fmetrics_dict['f_peak']]
        print(f'[INFO] Frequency metrics from SNR: {fmetrics_dict}')
    else:
        print('[INFO] No frequency metrics returned from SNR analysis')

    print('before trimming')
    print(detected_st)
    detected_st.trim(best_trig['time'] - 1, best_trig['time'] + best_trig['duration'] + 1)
    print('after trimming')
    print(detected_st)
    for tr in detected_st:
        tr.stats['units'] = 'm'
    dsamObj = DSAM(stream=detected_st, sampling_interval=1.0)
    print(len(dsamObj.dataframes))
    dsamObj.plot(metrics=metric, equal_scale=True, outfile=os.path.join(outdir, "DSAM.png"))
    dsamObj.write(outdir, ext="csv")

    source = initial_source(lat=dome_location['lat'], lon=dome_location['lon'])
    gridobj = make_grid(center_lat=source['lat'], center_lon=source['lon'],
                        node_spacing_m=100, grid_size_lat_m=10000, grid_size_lon_m=8000)

    for this_Q in Q:
        for this_peakf in peakf:
            for this_v in surfaceWaveSpeed_kms:
                for this_metric in metric:
                    if compute_DRS_at_fixed_source:
                        DRSobj = dsamObj.compute_reduced_displacement(inv, source, surfaceWaves=True,
                                                                      Q=this_Q, wavespeed_kms=this_v, peakf=this_peakf)
                        DRSobj.plot(metrics=this_metric, equal_scale=True,
                                    outfile=os.path.join(outdir, f"dome_DRS_Q{this_Q}_f{this_peakf}_v{this_v}_{this_metric}.png"))

                    aslobj = ASL(dsamObj, this_metric, inv, gridobj, window_seconds)
                    aslobj.compute_grid_distances()
                    aslobj.compute_amplitude_corrections(surfaceWaves=True, wavespeed_kms=this_v,
                                                         Q=this_Q, fix_peakf=this_peakf)
                    aslobj.fast_locate()
                    aslobj.print_event()

                    aslobj.save_event(outfile=os.path.join(outdir, f"ASL_Q{this_Q}_f{this_peakf}_v{this_v}_{this_metric}.qml"))
                    aslobj.plot(zoom_level=0, threshold_DR=0.03, scale=0.2, join=True, number=0,
                                equal_size=False, add_labels=True,
                                stations=[tr.stats.station for tr in detected_st],
                                outfile=os.path.join(outdir, f"ASL_Q{this_Q}_f{this_peakf}_v{this_v}_{this_metric}.png"))

#########################################################################################################
#### Below is a fairly generic command line wrapper to run a custom function on a seisan db          ####
#### The only change you might need for a different application are some different command line args ####
#########################################################################################################
import os
import sys
import argparse
from obspy import UTCDateTime, read_inventory
from flovopy.wrappers.seisandb_event_wrappers import apply_custom_function_to_each_event
#from flovopy.core.mvo import load_mvo_master_inventory
#from flovopy.analysis.asl import asl_event

def find_seisan_data_dir():
    if sys.platform == "darwin":
        if os.path.exists("/Volumes/DATA/SEISAN_DB"):
            return "/Volumes/DATA/SEISAN_DB"
        raise RuntimeError("No SEISAN_DB found on macOS.")
    elif sys.platform.startswith("linux"):
        return "/data/SEISAN_DB"
    else:
        raise RuntimeError("Unsupported platform for SEISAN")

def main():
    parser = argparse.ArgumentParser(description="Run ASL event locator on SEISAN DB events")
    parser.add_argument("--start", type=str, required=True, help="Start date (e.g., 2001-01-01)")
    parser.add_argument("--end", type=str, required=True, help="End date (e.g., 2001-01-02)")
    parser.add_argument("--db", type=str, default="MVOE_", help="SEISAN DB name (default: MVOE_)")
    parser.add_argument("--outdir", type=str, default="ASL_DB", help="Output directory")
    parser.add_argument("--Q", type=int, default=23, help="Quality factor for attenuation")
    parser.add_argument("--speed", type=float, default=1.5, help="Surface wave speed (km/s)")
    parser.add_argument("--peakf", type=float, default=8.0, help="Peak frequency (Hz)")
    parser.add_argument("--metric", type=str, default="rms", help="Metric to use (e.g. rms, peak)")
    parser.add_argument("--interactive", action="store_true", help="Use interactive detection tuner")
    parser.add_argument("--min_stations", type=int, default=5, help="Minimum required stations")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    startdate = UTCDateTime(args.start)
    enddate = UTCDateTime(args.end)
    seisan_dir = find_seisan_data_dir()
    xmlfile = os.path.join(seisan_dir, 'CAL', 'MV.xml')
    inv = read_inventory(xmlfile)
    outdir = os.path.join(os.path.dirname(seisan_dir), args.outdir)

    apply_custom_function_to_each_event(
        startdate, enddate,
        SEISAN_DATA=seisan_dir,
        DB=args.db,
        inv=inv,
        post_process_function=asl_event,
        verbose=True,
        bool_clean=True,
        plot=True,
        valid_subclasses="re",
        quality_threshold=1.0,
        outputType="DISP",
        freq=[0.5, 30.0],
        vertical_only=True,
        max_dropout=4.0,
        # arguments for asl_event follow
        outdir=outdir,
        Q=args.Q,
        surfaceWaveSpeed_kms=args.speed,
        peakf=args.peakf,
        metric=args.metric,
        window_seconds=5,
        min_stations=args.min_stations,
        interactive=args.interactive
    )

if __name__ == "__main__":
    main()
"""
run-asl   --start 2001-01-01T00:00:00   --end 2001-01-01T03:00:00   --db MVOE_    --outdir output/asl   --Q 23   --peakf 8.0   --metric rms   --speed 1.5   --interactive --verbose
"""