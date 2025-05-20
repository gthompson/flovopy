# a14_preprocess_daily_sds.py

import os
from obspy import UTCDateTime, read_inventory
from flovopy.sds.sds import SDSobj
from flovopy.core.preprocessing import preprocess_stream


def preprocess_sds_daily(
    raw_sds_dir,
    clean_sds_dir,
    inv_file,
    start_date,
    end_date,
    bool_clean=True,
    quality_threshold=0.6,
    freq=(0.1, 30.0),
    outputType="VEL",
    max_dropout=1.0,
    overwrite=False,
    verbose=True
):
    inv = read_inventory(inv_file)
    startt = UTCDateTime(start_date)
    endt = UTCDateTime(end_date)

    current = startt
    while current < endt:
        next_day = current + 86400
        if verbose:
            print(f"[INFO] Processing {current.date}...")

        raw_sds = SDSobj(raw_sds_dir)
        clean_sds = SDSobj(clean_sds_dir)

        raw_sds.read(current, next_day, verbose=verbose)
        st = raw_sds.stream

        if len(st) == 0:
            if verbose:
                print("[INFO] No data for this day.")
            current = next_day
            continue

        preprocess_stream(
            st,
            bool_despike=True,
            bool_clean=bool_clean,
            inv=inv,
            quality_threshold=quality_threshold,
            taperFraction=0.05,
            filterType="bandpass",
            freq=freq,
            corners=6,
            zerophase=False,
            outputType=outputType,
            miniseed_qc=True,
            max_dropout=max_dropout,
            verbose=verbose
        )

        if len(st) == 0:
            if verbose:
                print("[INFO] Stream is empty after preprocessing.")
            current = next_day
            continue

        clean_sds.stream = st
        success = clean_sds.write(overwrite=overwrite)

        if verbose:
            print("[INFO] Write status:", "Success" if success else "Failed")

        current = next_day


if __name__ == "__main__":
    from flovopy.config_projects import get_config
    config = get_config()

    preprocess_sds_daily(
        raw_sds_dir=config['mvo_sds_output_dir'],
        clean_sds_dir=config['mvo_sds_cleaned_dir'],
        inv_file=config['mvo_stationxml'],
        start_date="1996-08-01",
        end_date="1996-08-04",
        bool_clean=True,
        quality_threshold=0.6,
        freq=(0.1, 30.0),
        outputType="VEL",
        max_dropout=1.0,
        overwrite=False,
        verbose=True
    )
