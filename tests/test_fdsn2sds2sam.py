# tests/test_fdsn2sam_wrapper.py
"""
import os
from flovopy.wrappers import fdsn2sds2sam

sds_path = os.path.expanduser("~/work/SDS")
sam_path = os.path.expanduser("~/work/SAM_OUT")
stationxml_dir = os.path.join(sds_path, "stationxml")

if __name__ == "__main__":
    args = [
        # Time window (end exclusive)
        "--start", "2011-03-10",
        "--end",   "2011-03-25",

        # Locations
        "--sds_root", sds_path,
        "--sam_root", sam_path,

        # FDSN selection
        "--service", "IRIS",
        "--network", "IU",
        "--station", "DWPF",
        "--location", "*",
        "--channels", "BH?",

        # Downloader settings
        "--threads", "4",
        "--chunk", "86400",
        "--minlen", "0.0",
        "--save_inventory",

        # SAM settings
        "--sampling_interval", "60",
        "--bands_preset", "storm",
        "--remove_response",
        "--output", "VEL",
        "--stationxml", stationxml_dir,
        "--also_rsam",
        "--speed", "2",
        "--max_rate", "250.0",
        "--merge_strategy", "obspy",
        "--nprocs", "auto",

        "--verbose",
    ]

    fdsn2sds2sam.main(args)

    # Optional: print both trees
    try:
        from flovopy.utils.misc import tree
        print("\n[SDS tree]\n" + "\n".join(tree(sds_path, prefix="")))
        print("\n[SAM tree]\n" + "\n".join(tree(sam_path, prefix="")))
    except Exception:
        pass

    print("End-to-end FDSN → SDS → SAM completed (missing days only).")
"""
 # tests/test_fdsn2sds2sam_wrapper.py implements a circular domain around Kennedy Space Center
import os
from flovopy.wrappers import fdsn2sds2sam

sds_path = os.path.expanduser("~/work/SDS")
sam_path = os.path.expanduser("~/work/SAM_OUT")
stationxml_dir = os.path.join(sds_path, "stationxml")

if __name__ == "__main__":
    args = [
        # Time window (end exclusive)
        "--start", "2011-01-01",
        "--end",   "2012-01-01",

        # Locations
        "--sds_root", sds_path,
        "--sam_root", sam_path,

        # ---- FDSN selection + circular domain around KSC ----
        "--service", "IRIS",
        "--network", "IU",
        "--station", "DWPF",           # wildcard stations within the domain
        "--location", "10",
        "--channels", "BHZ",

        "--domain", "circle",
        "--lat", "28.573",          # Kennedy Space Center ~28.573 N
        "--lon", "-80.649",         # ~-80.649 E
        "--radius_km", "100",       # 100 km radius

        # Downloader settings
        "--threads", "4",
        "--chunk", "86400",
        "--minlen", "0.0",
        "--save_inventory",

        # ---- SAM settings ----
        "--sampling_interval", "60",
        "--bands_preset", "storm",  # (note underscore form for the wrapper)
        "--remove_response",
        "--output", "VEL",
        "--stationxml", stationxml_dir,
        "--also_rsam",
        "--speed", "2",
        "--max_rate", "250.0",
        "--merge_strategy", "obspy",
        "--nprocs", "auto",

        "--verbose",
    ]

    fdsn2sds2sam.main(args)

    # Optional: print both trees
    try:
        from flovopy.utils.misc import tree
        print("\n[SDS tree]\n" + "\n".join(tree(sds_path, prefix="")))
        print("\n[SAM tree]\n" + "\n".join(tree(sam_path, prefix="")))
    except Exception:
        pass

    print("End-to-end FDSN → SDS → SAM completed (missing days only).")   