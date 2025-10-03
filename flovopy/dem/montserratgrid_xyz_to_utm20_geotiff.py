from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
import os

# --- Hardcoded paths ---
DEM_IN = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/trickster2025/Mont_10m.grd")
OUT_TIF = DEM_IN.with_name("DEM_10m_1999_xyz_UTM20N.tif")
OUT_TIF_WGS84 = DEM_IN.with_name("DEM_10m_1999_xyz_WGS84.tif")
POINTS_IN = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/trickster2025/PTS_ENH.TXT")
POINTS_OUT = POINTS_IN.with_name("PTS_ENH_WGS84_UTM20.csv")

# --- CRS definitions ---
# Montserrat Grid (TM on Clarke 1880) with Bursa–Wolf to WGS84
mont_crs = CRS.from_proj4(
    "+proj=tmerc +lat_0=0 +lon_0=-62 +k=0.9995 "
    "+x_0=400000 +y_0=0 "
    "+a=6378249.145 +b=6356514.86955 "
    "+towgs84=132.938,-128.285,-383.111,0,0,12.7996,9.9976"
)

# this is fixed by dx_m = -118.697 and dy_m = 835.986 to account for mean difference between df3 and df4 in compare_stationXML.py
# the first are coordinates from station0.hyp.xls and the second are coordinates from Trickster in 2025 that were in original Montserrat grid, 
# along with the grid file used for DEM_IN
mont_crs = CRS.from_proj4(
    "+proj=tmerc +lat_0=0 +lon_0=-62 +k=0.9995 "
    "+x_0=400114.5 +y_0=-835.7 "
    "+a=6378249.145 +b=6356514.86955 "
    "+towgs84=132.938,-128.285,-383.111,0,0,12.7996,0"
)
wgs84 = CRS.from_epsg(4326)
utm20 = CRS.from_epsg(32620)

to_wgs84 = Transformer.from_crs(mont_crs, wgs84, always_xy=True)
to_utm20 = Transformer.from_crs(mont_crs, utm20, always_xy=True)

def write_tif_from_xyz(xyz_path: Path, out_path: Path):
    print(f"[DEM] Reading ASCII XYZ: {xyz_path}")
    xyz = np.loadtxt(xyz_path)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise RuntimeError("XYZ file must have 3 columns (x y z)")

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xs = np.unique(x)
    ys = np.unique(y)

    if xs.size * ys.size != z.size:
        raise RuntimeError("XYZ grid is irregular or ragged; cannot reshape to 2D raster")

    dx = float(np.diff(xs).mean()) if xs.size > 1 else 1.0
    dy = float(np.diff(ys).mean()) if ys.size > 1 else 1.0

    nx, ny = xs.size, ys.size
    # Z should be arranged row-major with Y decreasing from top; reshape accordingly:
    Z = z.reshape((ny, nx))
    transform = from_origin(xs.min(), ys.max(), dx, dy)

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": mont_crs,            # stays in Montserrat CRS here
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(Z.astype("float32"), 1)

    print(f"[DEM] Wrote GeoTIFF (Montserrat CRS): {out_path}")
    with rasterio.open(out_path) as chk:
        arr = chk.read(1, masked=True)
        print(f"[DEM] Size={chk.width}x{chk.height} dtype={arr.dtype} min={arr.min()} max={arr.max()}")

def reproject_grd_to_utm(grd_path: Path, out_path: Path):
    print(f"[DEM] Reading Surfer GRD: {grd_path}")
    with rasterio.open(grd_path) as src:
        # Check we actually have a raster band
        if src.count < 1:
            raise RuntimeError("Input GRD has zero bands; cannot proceed.")

        src_crs = src.crs if src.crs is not None else mont_crs
        if src.crs is None:
            print("[DEM] No CRS in GRD; assigning Montserrat CRS.")

        src_transform = src.transform
        src_profile = src.profile.copy()
        src_nodata = src.nodata if src.nodata is not None else np.nan

        # Read the first band explicitly
        src_arr = src.read(1)  # shape (H, W)
        src_dtype = src_arr.dtype
        print(f"[DEM] src size={src.width}x{src.height} dtype={src_dtype} nodata={src_nodata}")

        # Compute target grid in UTM20
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, utm20, src.width, src.height, *src.bounds
        )

        # Prepare an empty target array
        dst_arr = np.empty((dst_height, dst_width), dtype="float32")

        # Reproject into the array
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=utm20,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=src_nodata if np.isfinite(src_nodata) else None,
        )

        # Build output profile and write
        dst_profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": "float32",
            "crs": utm20,
            "transform": dst_transform,
            "nodata": src_nodata if np.isfinite(src_nodata) else None,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

        with rasterio.open(out_path, "w", **dst_profile) as dst:
            dst.write(dst_arr, 1)

    print(f"[DEM] Wrote GeoTIFF (UTM20N): {out_path}")
    with rasterio.open(out_path) as chk:
        arr = chk.read(1, masked=True)
        print(f"[DEM] dst size={chk.width}x{chk.height} dtype={arr.dtype} min={arr.min()} max={arr.max()}")

# --- DEM branch chooser ---
suffix = DEM_IN.suffix.lower()
if suffix == ".asc":
    write_tif_from_xyz(DEM_IN, OUT_TIF)
elif suffix == ".grd":
    reproject_grd_to_utm(DEM_IN, OUT_TIF)
else:
    raise SystemExit(f"Unsupported DEM format: {DEM_IN.suffix}")

# --- Points: robust parser (E N H Site-with-spaces) + transforms ---
print(f"[POINTS] Reading {POINTS_IN}")

rows = []
num_re = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'  # float/int pattern
line_re = re.compile(rf'^\s*({num_re})\s+({num_re})\s+({num_re})\s+(.*\S)\s*$')

with open(POINTS_IN, "r", encoding="utf-8", errors="ignore") as f:
    for ln, raw in enumerate(f, start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # Skip header row like "E N H Site"
        if ln == 1 and re.search(r'\bE\b', s) and re.search(r'\bN\b', s) and re.search(r'\bH\b', s):
            continue
        m = line_re.match(s)
        if not m:
            print(f"[POINTS:WARN] Skipping unparsable line {ln}: {s}")
            continue
        e, n, h, site = m.groups()
        try:
            rows.append({"E": float(e), "N": float(n), "H": float(h), "Site": site.strip()})
        except ValueError:
            print(f"[POINTS:WARN] Numeric parse failed on line {ln}: {s}")

if rows:
    df = pd.DataFrame(rows)
    # Mont->WGS84 and Mont->UTM20
    lons, lats = to_wgs84.transform(df["E"].values, df["N"].values)
    utm_e, utm_n = to_utm20.transform(df["E"].values, df["N"].values)

    df_out = pd.DataFrame({
        "Site": df["Site"],
        "E_mont": df["E"],
        "N_mont": df["N"],
        "H_m": df["H"],
        "lon": lons,
        "lat": lats,
        "UTM20_E": utm_e,
        "UTM20_N": utm_n,
    })
    df_out.to_csv(POINTS_OUT, index=False)
    print(f"[POINTS] Wrote transformed station list: {POINTS_OUT}")
else:
    print("[POINTS] No valid rows parsed; nothing written.")

# convert UTM GeoTIFF to WGS84
os.system(f"gdalwarp -t_srs EPSG:4326 -r bilinear {OUT_TIF} {OUT_TIF_WGS84}")

# Now plot


import pandas as pd
import pygmt
from flovopy.asl.map import topo_map
from flovopy.core.mvo import REGION_DEFAULT
from pathlib import Path

'''
PROJECTDIR = "/Users/GlennThompson/Dropbox/BRIEFCASE/SSADenver"
DEM_DEFAULT = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/auto_crs_fit_v2/wgs84_s0.4_3_clean.tif")
#DEM_DEFAULT = Path(PROJECTDIR) / "wgs84_s0.4_3_clean_shifted.tif"
#DEM_DEFAULT = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/
DEM_DEFAULT = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/trickster2025/DEM_10m_1999_xyz_WGS84.tif")
DEM_DEFAULT = 
DEM_DEFAULT=None
'''
DEM_LIST = (OUT_TIF_WGS84, None)

# Path to the current script's folder
here = Path(__file__).parent

'''
# check default basemap
if DEM_DEFAULT:
    fig = pygmt.Figure()
    # Plot raster DEM
    fig.grdimage(
        DEM_DEFAULT,
        cmap="gray",      # grayscale colormap
        projection="M6i"  # 6-inch wide Mercator projection
    )

    # Add frame with ticks and annotation

    fig.basemap(frame="af")

    # Show interactively or save
    fig.savefig(here / "trickster_base.png")
    #fig.close()
    del fig
'''

# CSVs in the same folder
csv1 = here / "stations_with_channels_from_MV.xml.csv"
csv2 = here / "stations_with_channels_MontserratDigitalSeismicNetwork.xml.csv"
csv3 = here / "station0.hyp.csv"
csv4 = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/trickster2025/PTS_ENH_WGS84_UTM20.csv")

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)
df4 = pd.read_csv(csv4)

for n, DEM_DEFAULT in enumerate(DEM_LIST):

    # Create base topo map
    fig = topo_map(
        show=False,
        title="Montserrat Topography with Station Overlays",
        add_topography=True,
        cmap="gray",
        region=REGION_DEFAULT,
        dem_tif=DEM_DEFAULT,
    )

    # Plot red dots for the first set
    fig.plot(
        x=df1["longitude"], y=df1["latitude"],
        style="c0.25c", fill="red", pen="black",
        label="StationXML (MV.xml)"
    )

    # Plot yellow dots for the second set
    fig.plot(
        x=df2["longitude"], y=df2["latitude"],
        style="c0.25c", fill="yellow", pen="black",
        label="StationXML (MontserratDigitalSeismicNetwork.xml)"
    )

    # Plot blue dots for the third set
    fig.plot(
        x=df3["longitude"], y=df3["latitude"],
        style="c0.25c", fill="blue", pen="black",
        label="station0.hyp"
    )

    # Plot white dots for the fourth set
    fig.plot(
        x=df4["lon"], y=df4["lat"],
        style="c0.25c", fill="white", pen="black",
        label="Trickster PTS_ENH (transformed)"
    )

    # Add labels (optional, from station codes)
    for _, row in df1.iterrows():
        fig.text(x=row["longitude"], y=row["latitude"], text=row["station"], font="8p,red", offset="0.3c/0.3c")

    for _, row in df2.iterrows():
        fig.text(x=row["longitude"], y=row["latitude"], text=row["station"], font="8p,yellow", offset="0.3c/-0.3c")

    for _, row in df3.iterrows():
        fig.text(x=row["longitude"], y=row["latitude"], text=row["station"], font="8p,blue", offset="0.3c/-0.3c")    

    for _, row in df4.iterrows():
        fig.text(x=row["lon"], y=row["lat"], text=row["Site"], font="4p,white", offset="0.3c/-0.3c") 

    # Add the legend (auto-collect labels from the plotted series)
    fig.legend(position="JTR+jTR+o0.5c/0.5c", box="+gwhite+p1p")

    # Show the map
    fig.savefig(here / f'comparing_station_location_sources_{n}.png')

from pyproj import Geod

# --- Robust MB filters (handle NaN, whitespace, case) ---
mask3 = df3["station"].fillna("").astype(str).str.strip().str.upper().str.startswith("MB")
mask4 = df4["Site"].fillna("").astype(str).str.strip().str.upper().str.startswith("MB")

df3_mb = df3.loc[mask3].copy()
df4_mb = df4.loc[mask4].copy()

# Normalized join key
df3_mb["key"] = df3_mb["station"].astype(str).str.strip().str.upper()
df4_mb["key"] = df4_mb["Site"].astype(str).str.strip().str.upper()

# Drop duplicates per key (keep first)
df3_mb = df3_mb.drop_duplicates(subset="key", keep="first")
df4_mb = df4_mb.drop_duplicates(subset="key", keep="first")

# Merge and compute deltas
merged = pd.merge(
    df3_mb, df4_mb, on="key", suffixes=("_df3", "_df4"), how="inner"
)

merged["dlat"] = merged["latitude"] - merged["lat"]
merged["dlon"] = merged["longitude"] - merged["lon"]

# Optional: great-circle distance in meters (WGS84)
g = Geod(ellps="WGS84")
_, _, merged["dist_m"] = g.inv(
    merged["longitude"].values, merged["latitude"].values,
    merged["lon"].values,        merged["lat"].values
)

print("\n[MB Station coordinate differences df3 vs df4]")
print(merged[["key", "station", "Site", "latitude", "lat", "dlat",
              "longitude", "lon", "dlon", "dist_m"]].sort_values("key"))

from pyproj import Transformer
import numpy as np

# Transformer to UTM20N (EPSG:32620) for x/y distances in meters
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32620", always_xy=True)

# Project both sets of coordinates
x3, y3 = to_utm.transform(merged["longitude"].values, merged["latitude"].values)
x4, y4 = to_utm.transform(merged["lon"].values,        merged["lat"].values)

# Compute differences in meters
merged["dx_m"] = x3 - x4   # east–west
merged["dy_m"] = y3 - y4   # north–south
merged["dist_m_xy"] = np.sqrt(merged["dx_m"]**2 + merged["dy_m"]**2)

# Show results
print("\n[MB Station coordinate differences df3 vs df4]")
print(merged[["key", "station", "Site",
              "latitude", "lat", "dlat",
              "longitude", "lon", "dlon",
              "dx_m", "dy_m", "dist_m", "dist_m_xy"]])

# Compute averages for numeric columns only
numeric_means = merged.select_dtypes(include=[np.number]).mean()

print("\n[Average differences across MB stations df3 vs df4]")
print(numeric_means.round(3))   # round for readability

print(merged[['dx_m', 'dy_m']].describe())

