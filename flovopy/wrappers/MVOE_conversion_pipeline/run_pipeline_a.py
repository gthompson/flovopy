# Paths to the individual pipeline scripts
from flovopy.wrappers.MVOE_conversion_pipeline.a00_create_mvoe_index_db import main as main00
from flovopy.wrappers.MVOE_conversion_pipeline.a01_index_wav_files import main as main01
from flovopy.wrappers.MVOE_conversion_pipeline.a02_index_aef_files import main as main02
from flovopy.wrappers.MVOE_conversion_pipeline.a03_index_sfiles import main as main03

def run_pipeline(args):
    """
    Run all indexing steps in sequence: database setup, WAV, AEF, SFILE indexing.

    Parameters
    ----------
    database_path : str
        Path to the SQLite index database (e.g., 'mvoe_index.db').
    wav_root : str
        Root directory containing WAV files.
    aef_root : str
        Root directory containing AEF files.
    sfile_root : str
        Root directory containing S-files.
    """

    print("Running Step 00: Create MVOE index database...")
    main00(args)
    
    print("Running Step 01: Index WAV files...")
    main01(args)

    print("Running Step 02: Index AEF files...")
    main02(args)

    print("Running Step 03: Index S-files...")
    main03(args)

    print("All steps completed successfully.")   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index and convert SEISAN WAV files to MiniSEED.")
    parser.add_argument("--wav_dir", required=True, help="Top-level WAV directory")
    parser.add_argument("--aef_dir", required=True, help="Top-level AEF directory")
    parser.add_argument("--sfile_dir", required=True, help="Top-level REA directory")
    parser.add_argument("--archive", choices=["bgs", "mvo"], required=True, help="Which archive to index")    
    parser.add_argument("--mseed_output", required=True, help="Output directory for MiniSEED files")
    parser.add_argument("--json_output", required=True, help="Output directory for JSON files")    
    parser.add_argument("--db", required=True, help="SQLite database path")
    args = parser.parse_args()
    run_pipeline(args)
'''
python flovopy/wrappers/MVOE_conversion_pipeline/run_pipeline_a.py --wav_dir /data/SEISAN_DB/WAV/MVOE_ --aef_dir /data/SEISAN_DB/AEF/MVOE_ --sfile_dir /data/SEISAN_DB/REA/MVOE_ --archive bgs --mseed_output /data/SEISAN_DB/miniseed/MVOE_ --json_output /data/SEISAN_DB/json/MVOE_ --db /data/SEISAN_DB/index_mvoe.sqlite
'''