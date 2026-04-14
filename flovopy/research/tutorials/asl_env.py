 # --- Imports & paths ---
from pathlib import Path
#from datetime import datetime
from obspy import read_inventory, UTCDateTime
from flovopy.asl.map import topo_map  # your function
#from flovopy.asl.grid import make_grid, Grid, _meters_per_degree,  apply_channel_land_circle_mask

DROPBOXDIR = Path('~').expanduser() / 'Dropbox'
PROJECTDIR = DROPBOXDIR / "BRIEFCASE" / "SSADenver"
LOCALPROJECTDIR = Path("/Users/GlennThompson/work/PROJECTS/SSADenver_local")
METADATADIR = PROJECTDIR / "metadata"
OUTPUT_DIR = LOCALPROJECTDIR / "asl_notebooks"
INVENTORY_XML = METADATADIR / "MV.xml"
#INVENTORY_XML = METADATADIR / "MV_Seismic_and_GPS_stations.xml"
#DEM_TIF_FOR_BMAP = f"{PROJECTDIR}/channel_finder/02_dem_flipped_horizontal.tif"
#DEM_TIF_FOR_BMAP = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/auto_crs_fit_v2/wgs84_s0.4_3_clean.tif")
#DEM_DEFAULT = Path(PROJECTDIR) / "wgs84_s0.4_3_clean_shifted.tif"
DEM_DEFAULT = METADATADIR / "MONTSERRAT_DEM_WGS84_MASTER.tif"

# Montserrat default region (lon_min, lon_max, lat_min, lat_max)
from flovopy.research.mvo.mvo_ids import REGION_DEFAULT, DOME_LOCATION

# Load inventory
INV = read_inventory(INVENTORY_XML)

# I/O
INPUT_DIR = f"{PROJECTDIR}/ASL_inputs/biggest_pdc_events"
DIST_MODE = "3d"        # include elevation
GLOBAL_CACHE = f"{PROJECTDIR}/asl_global_cache"

# Output folder
RUN_TAG = UTCDateTime().strftime("topo_map_test_%Y%m%dT%H%M%S")
OUTDIR = Path(OUTPUT_DIR) / RUN_TAG
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = GLOBAL_CACHE