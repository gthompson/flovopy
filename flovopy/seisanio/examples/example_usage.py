"# examples/example_usage.py

import os
from seisanio.core.sfile import Sfile
from seisanio.utils.paths import spath2datetime
from seisanio.utils.station0hyp import parse_STATION0HYP, add_station_locations
from obspy import UTCDateTime

# Set your Seisan database path and parameters
SEISAN_DATA = "/data/SEISAN_DB"
DB = "MVOE_"
example_sfile_path = os.path.join(SEISAN_DATA, "REA", DB, "2002", "07", "27-0027-58L.S200207")

# Instantiate and parse the Sfile
sfile = Sfile(example_sfile_path, use_mvo_parser=True)

# Print a summary of the parsed event
print(sfile)

# Convert to dictionary
event_dict = sfile.to_dict()
print("\nEvent dictionary:\n", event_dict)

# Load waveform and plot (if files are accessible)
if len(sfile.wavfiles) > 0:
    print("\nReading and plotting first WAV file...")
    if sfile.wavfiles[0].read():
        sfile.wavfiles[0].plot(equal_scale=True)

# Print AEF file summary if present
if sfile.aeffiles:
    print("\nAEF File contents:")
    for aef in sfile.aeffiles:
        print(aef)

# Optionally, add station location info to waveforms
station0_file = os.path.join(SEISAN_DATA, "DAT", "STATION0_MVO.HYP")
station_df = parse_STATION0HYP(station0_file)
if station_df is not None and len(sfile.wavfiles) > 0:
    if sfile.wavfiles[0].read():
        add_station_locations(sfile.wavfiles[0].st, station_df)
        print("\nAdded station metadata to traces.")