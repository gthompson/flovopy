# tests/test_fdsn2sds.py
import os
from flovopy.wrappers import fdsn2sds

sds_path = os.path.expanduser("~/work/SDS")

if __name__ == "__main__":
    args = [
        "--service", "IRIS",
        "--network", "IU",
        "--station", "DWPF",
        "--location", "*",
        "--channels", "BHZ",
        "--start", "2011-03-10T00:00:00",
        "--end",   "2011-03-15T00:00:00",
        "--sds_root", sds_path,
        "--threads", "4",
        "--save_inventory",
        "--verbose",
    ]
    fdsn2sds.main(args)

    # Show SDS tree
    try:
        from flovopy.utils.misc import tree
        print("\n".join(tree(sds_path, prefix="")))
    except Exception:
        pass
    print("FDSN → SDS download completed.")
