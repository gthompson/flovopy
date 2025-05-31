#import os
import sys
#import glob
import argparse
from pathlib import Path
from obspy import read_inventory, Stream
from flovopy.core.preprocessing import preprocess_stream
from flovopy.core.mvo import read_mvo_waveform_file
from flovopy.config_projects import get_config


def batch_preprocess_mseed(
    root_dir: Path,
    inv_file: Path,
    bool_clean: bool = True,
    quality_threshold: float = 0.5,
    freq: tuple = (0.1, 30.0),
    output_type: str = "VEL",
    max_dropout: float = 0.1,
    verbose: bool = True,
    dry_run: bool = False
):
    """
    Batch preprocess MiniSEED files using a given inventory and preprocessing configuration.

    Parameters:
    - root_dir: Root directory containing subdirectories with MiniSEED files.
    - inv_file: Path to StationXML inventory file.
    - bool_clean: Whether to apply cleaning during preprocessing.
    - quality_threshold: Threshold for acceptable data quality (0.0 to 1.0).
    - freq: Bandpass frequency range (low, high).
    - output_type: Output waveform type ("VEL", "DISP", etc.).
    - max_dropout: Maximum allowable dropout fraction.
    - verbose: Whether to print progress.
    - dry_run: If True, processes but does not write cleaned files.
    """
    root_dir = Path(root_dir)
    inv_file = Path(inv_file)

    if verbose:
        print(f"[INFO] Loading inventory: {inv_file}")
    inv = read_inventory(str(inv_file))

    mseed_files = sorted(root_dir.glob("*/**/*.mseed"))
    if verbose:
        print(f"[INFO] Found {len(mseed_files)} MiniSEED files to process in {root_dir}.")

    for i, filepath in enumerate(mseed_files):
        cleaned_path = filepath.with_suffix(".cleaned")

        try:
            if cleaned_path.exists():
                if verbose:
                    print(f"[{i+1}/{len(mseed_files)}] Skipping existing: {cleaned_path.name}")
                continue

            if verbose:
                print(f"[{i+1}/{len(mseed_files)}] Processing: {filepath.name}")

            st: Stream = read_mvo_waveform_file(str(filepath))

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
                outputType=output_type,
                miniseed_qc=True,
                max_dropout=max_dropout
            )

            if len(st) == 0:
                if verbose:
                    print("[INFO] Stream is empty after cleaning.")
                continue

            if not dry_run:
                for tr in st:
                    if hasattr(tr.stats, "mseed") and "encoding" in tr.stats.mseed:
                        del tr.stats.mseed["encoding"]
                st.write(str(cleaned_path), format="MSEED")

        except Exception as e:
            print(f"[ERROR] Failed to process {filepath.name}: {e}")


def parse_arguments_or_config():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Batch preprocess MiniSEED files with inventory.")
        parser.add_argument("--root_dir", required=True, help="Root directory of MiniSEED files")
        parser.add_argument("--inv_file", required=True, help="Path to StationXML inventory file")
        parser.add_argument("--bool_clean", action="store_true", help="Apply cleaning steps")
        parser.add_argument("--quality_threshold", type=float, default=0.5, help="Quality threshold (0.0–1.0)")
        parser.add_argument("--freq_low", type=float, default=0.1, help="Low corner frequency (Hz)")
        parser.add_argument("--freq_high", type=float, default=30.0, help="High corner frequency (Hz)")
        parser.add_argument("--output_type", default="VEL", help="Output waveform type")
        parser.add_argument("--max_dropout", type=float, default=0.1, help="Max allowed dropout fraction")
        parser.add_argument("--dry_run", action="store_true", help="Run without writing cleaned files")
        parser.add_argument("--verbose", action="store_true", help="Print progress messages")
        return parser.parse_args()
    else:
        config = get_config()
        class Args:
            root_dir = Path(config['miniseed_top']) / "MVOE_"
            inv_file = Path(config['inventory'])
            bool_clean = True
            quality_threshold = 0.6
            freq_low = 0.1
            freq_high = 30.0
            output_type = "VEL"
            max_dropout = 1.0
            dry_run = False
            verbose = True
        return Args()


if __name__ == "__main__":
    args = parse_arguments_or_config()
    batch_preprocess_mseed(
        root_dir=args.root_dir,
        inv_file=args.inv_file,
        bool_clean=args.bool_clean,
        quality_threshold=args.quality_threshold,
        freq=(args.freq_low, args.freq_high),
        output_type=args.output_type,
        max_dropout=args.max_dropout,
        verbose=args.verbose,
        dry_run=args.dry_run
    )



