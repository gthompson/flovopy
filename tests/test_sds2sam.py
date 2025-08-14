# tests/test_sds2sam.py
import os
from flovopy.wrappers import sds2sam

sds_path = os.path.expanduser("~/work/SDS")
sam_path = os.path.expanduser("~/work/SAM_OUT")
stationxml_dir = os.path.join(sds_path, "stationxml")  # fdsn2sds writes here by default

args = [
    "--sds_root", sds_path,
    "--sam_root", sam_path,
    "--start", "2011-03-10",
    "--end",   "2011-03-15",
    "--network", "IU",
    "--station", "DWPF",
    "--location", "*",
    "--channel", "*Z",                  # keep Z (VSAM bands default are vertical/noise bands)
    "--sampling_interval", "60",
    "--minfreq", "0.01",
    "--maxfreq", "18.0",

    # Choose either bands or preset (preset overrides minfreq/maxfreq):
    "--bands-preset", "storm",

    "--remove_response",
    "--output", "VEL",
    "--stationxml", stationxml_dir,
    "--also_rsam",

    "--speed", "2",
    "--max_rate", "250.0",
    "--merge_strategy", "obspy",

    # Use auto workers (half the logical CPUs, rounded up)
    "--nprocs", "auto",

    "--verbose",
]

if __name__ == "__main__":
    sds2sam.main(args)

    # Show SAM tree
    try:
        from flovopy.utils.misc import tree
        print("\n".join(tree(sam_path, prefix="")))
    except Exception:
        pass
    print("SDS → SAM conversion completed.")

"""
Alternatively, you can specify bands directly in the command line:
bands_dict = {
    "PRI": [0.05, 0.10],   # primary microseism
    "SEC": [0.10, 0.35],   # secondary microseism
    "MB":  [0.15, 0.35],   # microbaroms (if you run on infrasound with PA)
    "HI":  [1.0, 5.0],     # local wind/cultural noise (QC)
}
bands_json = json.dumps(bands_dict)
"--bands-preset", bands_json,
"""