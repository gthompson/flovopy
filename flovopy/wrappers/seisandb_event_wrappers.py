# Seisan REA database wrappers
import os
import glob
from obspy import UTCDateTime
from flovopy.seisanio.core.sfile import get_sfile_list, Sfile
from flovopy.core.preprocessing import preprocess_stream
from flovopy.core.mvo import read_mvo_waveform_file


def apply_custom_function_to_each_event(
    startdate,
    enddate,
    SEISAN_DATA='/data/SEISAN_DB',
    DB='MVOE_',
    inv=None,
    post_process_function=None,
    verbose=False,
    bool_clean=True,
    plot=False,
    valid_subclasses='',
    quality_threshold=1.0,
    outputType=None,
    freq=[0.5, 30.0],
    seismic_only=False,
    vertical_only=False,
    max_dropout=None,
    **kwargs
):
    """
    Iterates over SEISAN S-files and WAV files, applies preprocessing, then calls a user-defined function.

    Parameters
    ----------
    startdate, enddate : UTCDateTime
        Time range to search SEISAN DB.
    SEISAN_DATA : str
        Path to SEISAN database root.
    DB : str
        Database name (e.g., 'MVOE_').
    inv : obspy.Inventory or None
        Station metadata.
    post_process_function : function or None
        Custom function to call on each event.
    verbose : bool
        Print debug info.
    bool_clean : bool
        Whether to remove bad traces.
    plot : bool
        Show waveform plots if no function is called.
    valid_subclasses : str
        If specified, only process events with these subclasses.
    outputType : str or None
        'VEL' or 'DISP' for corrected output.
    freq : list
        Filter band (Hz).
    seismic_only, vertical_only : bool
        Filter trace types.
    max_dropout : float or None
        Maximum gap % to tolerate.
    kwargs : dict
        Passed to post_process_function.


    Example:
        from obspy import UTCDateTime
        from flovopy.wrappers.seisandb_event_wrappers import apply_custom_function_to_each_event
        from my_analysis_module import my_custom_function  # define this separately

        start = UTCDateTime("2001-01-01")
        end = UTCDateTime("2001-01-02")

        apply_custom_function_to_each_event(
            start,
            end,
            DB="MVOE_",
            post_process_function=my_custom_function,
            inv=my_inventory,
            outdir="/tmp/asl_output",
            Q=23, surfaceWaveSpeed_kms=2.5
        )
    """
    if vertical_only or outputType:
        seismic_only = True

    sfiles = get_sfile_list(SEISAN_DATA, DB, startdate, enddate, verbose=verbose)

    for sfile_path in sfiles:
        try:
            s = Sfile(sfile_path, fast_mode=True)
            d = s.to_dict()

            if valid_subclasses and not (
                d['mainclass'] == 'LV' and d['subclass'] in valid_subclasses
            ):
                continue
        except Exception as e:
            print(f"[WARN] Failed to parse Sfile {sfile_path}: {e}")
            continue

        for item in ['wavfile1', 'wavfile2']:
            wavfile = d.get(item)
            wavfound = False

            if wavfile and DB[:3] in os.path.basename(wavfile):
                if os.path.isfile(wavfile):
                    wavfound = True
                else:
                    altbase = os.path.basename(wavfile).split('.')[0][:-3]
                    candidates = glob.glob(os.path.join(os.path.dirname(wavfile), altbase + '*'))
                    if len(candidates) == 1:
                        wavfile = candidates[0]
                        wavfound = True

            if not wavfound:
                if verbose:
                    print(f"Sfile: {os.path.basename(sfile_path)}; WAVfile: None")
                continue

            try:
                st = read_mvo_waveform_file(
                    wavfile,
                    bool_ASN=False,
                    verbose=verbose,
                    seismic_only=seismic_only,
                    vertical_only=vertical_only
                )
            except Exception as e:
                print(f"[ERROR] Could not load {wavfile}: {e}")
                continue

            raw_st = st.copy()
            preprocess_stream(
                st,
                bool_despike=True,
                bool_clean=bool_clean,
                inv=inv,
                quality_threshold=quality_threshold,
                taperFraction=0.05,
                filterType="bandpass",
                freq=freq,
                corners=2,
                zerophase=False,
                outputType=outputType,
                miniseed_qc=True,
                max_dropout=max_dropout
            )

            if len(st) == 0:
                if verbose:
                    print("[INFO] Stream is empty after cleaning.")
                continue

            if post_process_function:
                print(f"[INFO] Calling {post_process_function.__name__} with kwargs: {kwargs}")
                post_process_function(st, raw_st, **kwargs)
            elif plot:
                print("[INFO] Final stream preview:")
                raw_st.plot(equal_scale=False, title="Raw Data")
                st.plot(equal_scale=False, title="Preprocessed Data")

if __name__ == "__main__":
    pass
