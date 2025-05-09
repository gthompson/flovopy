import os
import glob
from obspy import read_inventory
from flovopy.core.preprocessing import preprocess_stream
from flovopy.core.mvo import read_mvo_waveform_file

def batch_preprocess_mseed(
    root_dir,
    inv_file,
    bool_clean=True,
    quality_threshold=0.5,
    freq=(0.1, 30.0),
    outputType="VEL",
    max_dropout=0.1,
    verbose=True
):
    print(f"[INFO] Loading inventory: {inv_file}")
    inv = read_inventory(inv_file)

    mseed_files = sorted(glob.glob(os.path.join(root_dir, "*", "*", "*.mseed")))
    print(f"[INFO] Found {len(mseed_files)} MiniSEED files to process.")

    for i, filepath in enumerate(mseed_files):
        try:
            cleanedpath = filepath.replace(".mseed", ".cleaned")

            if verbose:
                print(f"[{i+1}/{len(mseed_files)}] Reading: {filepath}")
            if os.path.isfile(cleanedpath):
                continue
            st = read_mvo_waveform_file(filepath)

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
                max_dropout=max_dropout
            )

            if len(st) == 0:
                if verbose:
                    print("[INFO] Stream is empty after cleaning.")
                continue

            # Save cleaned stream or pass it to next stage
            for tr in st:
                if hasattr(tr.stats, "mseed") and "encoding" in tr.stats.mseed:
                    del tr.stats.mseed["encoding"]
            st.write(cleanedpath, format="MSEED")

        except Exception as e:
            print(f"[ERROR] Failed to process {filepath}: {e}")

if __name__ == "__main__":
    batch_preprocess_mseed(
        root_dir="/data/SEISAN_DB/miniseed/MVOE_",
        inv_file="/data/SEISAN_DB/CAL/MV.xml",
        bool_clean=True,
        quality_threshold=0.6,
        freq=(0.1, 30.0),
        outputType="VEL",
        max_dropout=1.0,
        verbose=True
    )
