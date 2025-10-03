import pandas as pd
import pygmt
from flovopy.asl.map import topo_map
from flovopy.core.mvo import REGION_DEFAULT
from pathlib import Path

PROJECTDIR = "/Users/GlennThompson/Dropbox/BRIEFCASE/SSADenver"
DEM_DEFAULT = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/auto_crs_fit_v2/wgs84_s0.4_3_clean.tif")
#DEM_DEFAULT = Path(PROJECTDIR) / "wgs84_s0.4_3_clean_shifted.tif"
#DEM_DEFAULT = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/
DEM_DEFAULT = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/trickster2025/DEM_10m_1999_xyz_WGS84.tif")

#DEM_DEFAULT=None
# Path to the current script's folder
here = Path(__file__).parent


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

# CSVs in the same folder
csv1 = here / "stations_with_channels_from_MV.xml.csv"
csv2 = here / "stations_with_channels_MontserratDigitalSeismicNetwork.xml.csv"
csv3 = here / "station0.hyp.csv"
csv4 = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/trickster2025/PTS_ENH_WGS84_UTM20.csv")

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)
df4 = pd.read_csv(csv4)

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
    x=df1["longitude"],
    y=df1["latitude"],
    style="c0.25c",   # circle, 0.25 cm
    fill="red",
    pen="black",
)

# Plot yellow dots for the second set
fig.plot(
    x=df2["longitude"],
    y=df2["latitude"],
    style="c0.25c",
    fill="yellow",
    pen="black",
)

# Plot blue dots for the third set
fig.plot(
    x=df3["longitude"],
    y=df3["latitude"],
    style="c0.25c",
    fill="blue",
    pen="black",
)

# Plot white dots for the fourth set
fig.plot(
    x=df4["lon"],
    y=df4["lat"],
    style="c0.25c",
    fill="white",
    pen="black",
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


# Show the map
fig.savefig(here / 'comparing_station_location_sources_tricksterbase.png')

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