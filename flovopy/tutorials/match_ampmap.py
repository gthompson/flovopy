#!/usr/bin/env python
# coding: utf-8

# # 1. Set up parameters for ASL

# In[1]:


from pathlib import Path
import numpy as np
import pandas as pd
from obspy import read_inventory
from importlib import reload
from flovopy.asl.wrappers import run_single_event, find_event_files, run_all_events
from flovopy.core.mvo import dome_location, REGION_DEFAULT
from flovopy.processing.sam import VSAM, DSAM 
from flovopy.asl.config import ASLConfig, tweak_config
# -------------------------- Config --------------------------
# directories
HOME = Path.home()
PROJECTDIR      = HOME / "Dropbox" / "BRIEFCASE" / "SSADenver"
LOCALPROJECTDIR = HOME / "work" / "PROJECTS" / "SSADenver_local"
OUTPUT_DIR      = LOCALPROJECTDIR / "AMPMAP_RESULTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR       = '/data/SEISAN_DB/miniseed/MVOE_/2001'
GLOBAL_CACHE    = PROJECTDIR / "asl_global_cache"
METADATA_DIR    = PROJECTDIR / "metadata" 
STATION_CORRECTIONS_DIR = PROJECTDIR / "station_correction_analysis"

# master files
INVENTORY_XML   = METADATA_DIR / "MV_Seismic_and_GPS_stations.xml"
DEM_DEFAULT     = METADATA_DIR / "MONTSERRAT_DEM_WGS84_MASTER.tif"
GRIDFILE_DEFAULT= METADATA_DIR / "MASTER_GRID_MONTSERRAT.pkl"

# parameters for envelopes and cross-correlation
SMOOTH_SECONDS  = 1.0
MAX_LAG_SECONDS = 8.0
MIN_XCORR       = 0.5

# other parameters
DIST_MODE = "3d" # or 2d. will essentially squash Montserrat topography and stations onto a sea-level plane, ignored elevation data, e.g. for computing distances

# Inventory of Montserrat stations
from obspy import read_inventory
INV     = read_inventory(INVENTORY_XML)
print(f"[INV] Networks: {len(INV)}  Stations: {sum(len(n) for n in INV)}  Channels: {sum(len(sta) for net in INV for sta in net)}")

# Montserrat station corrections estimated from regionals
station_corrections_csv = STATION_CORRECTIONS_DIR / "station_gains_intervals.csv"
annual_station_corrections_csv = STATION_CORRECTIONS_DIR / "station_gains_intervals_by_year.csv"
station_corrections_df = pd.read_csv(station_corrections_csv)
annual_station_corrections_df = pd.read_csv(annual_station_corrections_csv)

# Montserrat pre-defined Grid (from 02 tutorial)
from flovopy.asl.grid import Grid
gridobj = Grid.load(GRIDFILE_DEFAULT)
print(gridobj)
landgridobj = Grid.load(GLOBAL_CACHE / "land" / "Grid_9c2fd59b.pkl")

# Montserrat constants
from flovopy.core.mvo import dome_location, REGION_DEFAULT
print("Dome (assumed source) =", dome_location)

# events and wrappers
'''
event_files = list(find_event_files(INPUT_DIR))
eventcsvfile = Path(OUTPUT_DIR) / "mseed_files.csv"
if not eventcsvfile.is_file():
    rows = [{"num": num, "f": str(f)} for num, f in enumerate(event_files)]
    df = pd.DataFrame(rows)
    df.to_csv(eventcsvfile, index=False)
best_file_nums  = [35, 36, 40, 52, 82, 83, 84, 116, 310, 338]
best_event_files = [event_files[i] for i in best_file_nums]
print(f'Best miniseed files are: {best_event_files}')
REFINE_SECTOR = False   # enable triangular dome-to-sea refinement
'''

# Parameters to pass for making pygmt topo maps
topo_kw = {
    "inv": INV,
    "add_labels": True,
    "cmap": "gray",
    "region": REGION_DEFAULT,
    "dem_tif": DEM_DEFAULT,  # basemap shading from your GeoTIFF - but does not actually seem to use this unless topo_color=True and cmap=None
    "frame": True,
    "dome_location": dome_location,
}


# # Build a baseline configuration
# This is inherited by various downstream functions
# This describes the physical parameters, the station metadata, the grid, the misfit algorithm, etc.

# In[ ]:


DEBUG=False
baseline_cfg = ASLConfig(
    inventory=INV,
    output_base=OUTPUT_DIR,
    gridobj=gridobj,
    global_cache=GLOBAL_CACHE,
    station_correction_dataframe=None,#station_corrections_df,
    wave_kind="surface",
    speed=1.5,
    Q=23, 
    peakf=8.0,
    dist_mode="2d", 
    misfit_engine="l2",
    window_seconds=5.0,
    min_stations=5,
    sam_class=VSAM, 
    sam_metric="mean",
    debug=DEBUG,
)
baseline_cfg.build()


# # Subset events for this date range

# In[ ]:


from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from obspy import UTCDateTime

def subset_events_and_build_paths(
    csv_path: str | Path,
    base_dir: str | Path,
    start: UTCDateTime,
    end: UTCDateTime,
    *,
    time_col: str = "time",
    dfile_col: str = "dfile",
    inclusive: str = "left",      # [start, end)
    check_exists: bool = False     # filter to files that exist on disk
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load an events CSV, subset rows by UTC datetime window, and build MiniSEED paths
    like '<base_dir>/YYYY/MM/<dfile>' using the parsed time for year/month.

    Returns
    -------
    dfsubset : pandas.DataFrame
        Filtered DataFrame copy with a new 'mseed_path' column (string paths).
    mseed_files : list[str]
        List of paths as strings (optionally filtered to existing files).
    """
    base_dir = Path(base_dir)

    # Read & ensure UTC for the time column
    df = pd.read_csv(csv_path, parse_dates=[time_col])
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Convert ObsPy UTCDateTime → pandas UTC Timestamps
    start_ts = pd.to_datetime(start.datetime, utc=True)
    end_ts   = pd.to_datetime(end.datetime,   utc=True)

    # Subset by window
    dfsubset = df[df[time_col].between(start_ts, end_ts, inclusive=inclusive)].copy()

    # Vectorized path build: '<base_dir>/YYYY/MM/' + dfile
    dfsubset["mseed_path"] = (
        dfsubset[time_col].dt.strftime(str(base_dir) + "/%Y/%m/")
        + dfsubset[dfile_col].astype(str)
    )

    mseed_files = dfsubset["mseed_path"].tolist()

    if check_exists:
        exists_mask = [Path(p).exists() for p in mseed_files]
        dfsubset = dfsubset.loc[exists_mask].copy()
        mseed_files = [p for p, ok in zip(mseed_files, exists_mask) if ok]

    return dfsubset, mseed_files


from obspy import UTCDateTime

BASE_DIR    = "/data/SEISAN_DB/miniseed/MVOE_"
ROCKFALLCSV = "/home/thompsong/Developer/mvo_data_mastering/asl_inputs/asl_input_events.csv"

startdate = UTCDateTime(2001, 2, 6)
enddate   = UTCDateTime(2001, 4, 2)

dfsubset, mseed_files = subset_events_and_build_paths(
    csv_path=ROCKFALLCSV,
    base_dir=BASE_DIR,
    start=startdate,
    end=enddate,
    inclusive="left",      # [start, end)
    check_exists=False     # set True to drop non-existent files
)

print(f"{len(dfsubset)} events in window; {len(mseed_files)} paths built.")
# peek:
print(dfsubset[["time", "dfile", "mseed_path"]].head())



# # Run events (one event=one miniseed file) with this baseline configuration

# In[ ]:


DEBUG-False
summaries = []
REFINE_SECTOR=False
for ev in mseed_files:
    result = run_single_event(
        mseed_file=str(ev),
        cfg=baseline_cfg,
        refine_sector=REFINE_SECTOR,
        station_gains_df=None,
        switch_event_ctag = True,
        topo_kw=topo_kw,
        mseed_units='m/s', # default units for miniseed files being used - probably "Counts" or "m/s"        
        reduce_time=True,
        debug=DEBUG,
    )
    summaries.append(result)

# Summarize
dfsum = pd.DataFrame(summaries)
display(dfsum)

summary_csv = Path(OUTPUT_DIR) / f"{baseline_cfg.tag()}__summary.csv"
dfsum.to_csv(summary_csv, index=False)
print(f"Summary saved to: {summary_csv}")

if not dfsum.empty:
    n_ok = int((~dfsum.get("error").notna()).sum()) if "error" in dfsum.columns else len(dfsum)
    print(f"Success: {n_ok}/{len(dfsum)}")

