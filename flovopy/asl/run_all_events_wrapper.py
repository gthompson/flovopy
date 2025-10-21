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
OUTPUT_DIR      = LOCALPROJECTDIR / "ASL_RESULTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR       = PROJECTDIR / "ASL_inputs" / "biggest_pdc_events"
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
event_files = list(find_event_files(INPUT_DIR))
eventcsvfile = Path(OUTPUT_DIR) / "mseed_files.csv"
if not eventcsvfile.is_file():
    rows = [{"num": num, "f": str(f)} for num, f in enumerate(event_files)]
    df = pd.DataFrame(rows)
    df.to_csv(eventcsvfile, index=False)
best_file_nums  = [35, 36, 40, 52, 82, 83, 84, 116, 310, 338]
best_event_files = [event_files[i] for i in best_file_nums]
print(f'Best miniseed files are: {best_event_files}')
REFINE_SECTOR = True   # enable triangular dome-to-sea refinement

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


DEBUG=False
baseline_cfg = ASLConfig(
    inventory=INV,
    output_base=OUTPUT_DIR,
    gridobj=gridobj,
    global_cache=GLOBAL_CACHE,
    station_correction_dataframe=station_corrections_df,
    wave_kind="surface",
    speed=1.5,
    Q=23, 
    peakf=2.0,
    dist_mode="3d", 
    misfit_engine="lin",
    window_seconds=5.0,
    min_stations=5,
    sam_class=VSAM, 
    sam_metric="mean",
    debug=DEBUG,
)
baseline_cfg.build()



summary_dir = run_all_events(
    input_dir=INPUT_DIR,   # or a directory of files
    cfg=baseline_cfg,
    topo_kw=topo_kw,
    station_gains_df=None,
    refine_sector=REFINE_SECTOR,
    mseed_units="m/s",            # default units for your MSEED files
    reduce_time=True,
    switch_event_ctag=True,
    use_multiprocessing=False,    # set True if you want parallelism
    mseed_extension='.cleaned',
    debug=DEBUG,
)

# Collect JSONL outputs into a DataFrame
summary_path = Path(summary_dir) / "summary.jsonl"
if summary_path.exists():
    df = pd.read_json(summary_path, lines=True)

    # Optional: also write a CSV copy
    summary_csv = Path(OUTPUT_DIR) / f"{baseline_cfg.tag()}__summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Summary saved to: {summary_csv}")

    if not df.empty:
        n_ok = int((~df.get("error").notna()).sum()) if "error" in df.columns else len(df)
        print(f"Success: {n_ok}/{len(df)}")