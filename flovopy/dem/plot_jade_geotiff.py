from flovopy.asl.map import topo_map
from obspy import read_inventory
from pathlib import Path
GRD = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/original/dem_MVO_v3_10m_jadeeyles.grd")
OUT_DIR = GRD.parent
TIF_UTM  = OUT_DIR / "dem_MVO_v3_10m_utm20.tif"     # EPSG:32620
TIF_WGS  = OUT_DIR / "dem_MVO_v3_10m_wgs84.tif"     # EPSG:4326
INV_XML   = "/Users/GlennThompson/Dropbox/BRIEFCASE/SSADenver/MV.xml"

# Optional inventory
try:
    inv = read_inventory(INV_XML)
except Exception:
    inv = None

# A) Simple grayscale topo, light blue sea, plus *uniform* land tint on top
fig1 = topo_map(
    dem_tif=TIF_WGS,
    zoom_level=1,           # try 1, 2, 3… now honored even with dem_tif
    topo_color=False,       # use gray base
    cmap="gray",
    add_shading=True,       # GMT hillshade from your raster
    land_fill="200/200/200",# <- NEW: uniform land overlay (light gray)
    inv=inv,
    add_labels=True,
    outfile = TIF_WGS.with_name(TIF_WGS.stem + "_map_gray_land.png"),
    show=False,
)

# B) Color topo (GMT “geo” CPT), no uniform land overlay
fig2 = topo_map(
    dem_tif=TIF_WGS,
    zoom_level=2,
    topo_color=True,        # colorful topography
    add_shading=True,
    inv=inv,
    add_labels=False,
    outfile = TIF_WGS.with_name(TIF_WGS.stem + "_map_geo.png"),
    show=False,
)