# plot_two_tifs.py
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
import pygmt
from pyproj import Transformer
from flovopy.asl.map import topo_map
from obspy import read_inventory

GRD = Path("/Users/glennthompson/Dropbox/PROFESSIONAL/DATA/wadgeDEMs/original/dem_MVO_v3_10m_jadeeyles.grd")
OUT_DIR = GRD.parent
TIF_UTM  = OUT_DIR / "dem_MVO_v3_10m_utm20.tif"     # EPSG:32620
TIF_WGS  = OUT_DIR / "dem_MVO_v3_10m_wgs84.tif"     # EPSG:4326
INV_XML   = "/Users/GlennThompson/Dropbox/BRIEFCASE/SSADenver/MV.xml"
try:
    inv = read_inventory(INV_XML)
except Exception:
    inv = None

def xarray_from_geotiff(tif_path: Path) -> xr.DataArray:
    """Return an xarray.DataArray with ascending coords from a GeoTIFF."""
    with rasterio.open(tif_path) as src:
        z = src.read(1, masked=True).filled(np.nan).astype(float)
        tr = src.transform
        ny, nx = z.shape

        # pixel-centered coords
        xs = tr.c + tr.a * (np.arange(nx) + 0.5) + tr.b * 0
        ys = tr.f + tr.e * (np.arange(ny) + 0.5) + tr.d * 0

        # flip so coordinates increase left→right, bottom→top
        if ys[0] > ys[-1]:
            ys = ys[::-1]
            z = z[::-1, :]
        if xs[0] > xs[-1]:
            xs = xs[::-1]
            z = z[:, ::-1]

        # decide coord names by CRS type
        if src.crs and src.crs.is_geographic:
            da = xr.DataArray(z, coords={"lat": ys, "lon": xs}, dims=("lat", "lon"), name="elevation")
        else:
            # projected (e.g., UTM in meters)
            da = xr.DataArray(z, coords={"y": ys, "x": xs}, dims=("y", "x"), name="elevation")
    return da

def plot_wgs84(tif: Path, out_png: Path):
    da = xarray_from_geotiff(tif)

    # region (lon/lat) from coords
    lonmin, lonmax = float(da.lon.min()), float(da.lon.max())
    latmin, latmax = float(da.lat.min()), float(da.lat.max())
    region = [lonmin, lonmax, latmin, latmax]

    # z-range for CPT
    zmin = float(np.nanpercentile(da.values, 1))
    zmax = float(np.nanpercentile(da.values, 99))

    fig = pygmt.Figure()
    # Mercator projection, width 12 cm
    fig.grdimage(
        grid=da,
        region=region,
        projection="M12c",
        cmap=True,  # use current CPT after makecpt
        frame=["WSen", "xaf", "yaf"],
        shading=pygmt.grdgradient(grid=da, radiance=[135, 30], normalize="t1"),
    )
    pygmt.makecpt(cmap="geo", series=[zmin, zmax], continuous=True)
    fig.coast(region=region, projection="M12c", shorelines="1p,black", resolution="f")
    fig.colorbar(frame='+l"Elevation (m)"')
    fig.savefig(str(out_png))

def plot_utm(tif: Path, out_png: Path):
    da = xarray_from_geotiff(tif)

    # region in projected meters
    xmin, xmax = float(da.x.min()), float(da.x.max())
    ymin, ymax = float(da.y.min()), float(da.y.max())
    region = [xmin, xmax, ymin, ymax]

    zmin = float(np.nanpercentile(da.values, 1))
    zmax = float(np.nanpercentile(da.values, 99))

    fig = pygmt.Figure()
    # Linear projection in cm; axes are in meters (easting/northing)
    fig.grdimage(
        grid=da,
        region=region,
        projection="X12c/0",  # linear x; height auto
        cmap=True,
        frame=['WSen', 'xafg+l"Easting (m)"', 'yafg+l"Northing (m)"'],
        shading=pygmt.grdgradient(grid=da, radiance=[135, 30], normalize="t1"),
    )
    pygmt.makecpt(cmap="geo", series=[zmin, zmax], continuous=True)
    fig.colorbar(frame='+l"Elevation (m)"')
    fig.savefig(str(out_png))

def xarray_from_geotiff(tif_path: Path) -> xr.DataArray:
    """Return an xarray.DataArray with ascending coords from a GeoTIFF."""
    with rasterio.open(tif_path) as src:
        z = src.read(1, masked=True).filled(np.nan).astype(float)
        tr = src.transform
        ny, nx = z.shape

        # pixel-centered coords
        xs = tr.c + tr.a * (np.arange(nx) + 0.5) + tr.b * 0
        ys = tr.f + tr.e * (np.arange(ny) + 0.5) + tr.d * 0

        # flip so coordinates increase left→right, bottom→top
        if ys[0] > ys[-1]:
            ys = ys[::-1]; z = z[::-1, :]
        if xs[0] > xs[-1]:
            xs = xs[::-1]; z = z[:, ::-1]

        if src.crs and src.crs.is_geographic:
            da = xr.DataArray(z, coords={"lat": ys, "lon": xs}, dims=("lat", "lon"), name="elevation")
        else:
            da = xr.DataArray(z, coords={"y": ys, "x": xs}, dims=("y", "x"), name="elevation")
    return da

def bounds_lonlat_from_utm(tif_utm: Path, epsg_utm=32620):
    """Transform UTM GeoTIFF bounds to lon/lat region for GMT UTM projection."""
    with rasterio.open(tif_utm) as src:
        b = src.bounds
    # corners in UTM meters
    corners_utm = [(b.left, b.bottom), (b.right, b.top)]
    tf = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)
    (lon_min, lat_min) = tf.transform(*corners_utm[0])
    (lon_max, lat_max) = tf.transform(*corners_utm[1])
    # ensure ordered
    lon0, lon1 = sorted([lon_min, lon_max])
    lat0, lat1 = sorted([lat_min, lat_max])
    return [lon0, lon1, lat0, lat1]

def plot_wgs84(tif: Path, out_png: Path):
    da = xarray_from_geotiff(tif)
    region = [float(da.lon.min()), float(da.lon.max()),
              float(da.lat.min()), float(da.lat.max())]
    z = da.values
    zmin = float(np.nanpercentile(z, 1))
    zmax = float(np.nanpercentile(z, 99))

    fig = pygmt.Figure()
    # build CPT first
    pygmt.makecpt(cmap="geo", series=[zmin, zmax], continuous=True)
    # shaded relief
    shade = pygmt.grdgradient(grid=da, radiance=[135, 30], normalize="t1")

    fig.grdimage(
        grid=da,
        region=region,
        projection="M12c",
        cmap=True,                  # uses the current CPT
        shading=shade,
        frame=["WSen", "xaf", "yaf"],   # draw a full frame with ticks
    )
    # add coastline
    fig.coast(region=region, projection="M12c",
              shorelines="5p,black", resolution="f")
    fig.colorbar(frame='+l"Elevation (m)"')
    fig.savefig(str(out_png))

def plot_utm_with_coast(tif_utm: Path, out_png: Path, utm_epsg=32620, utm_zone="20N"):
    """
    Plot the UTM GeoTIFF using GMT's UTM map projection ('U20N'),
    with a lon/lat region matching the UTM raster, so pscoast can be drawn.
    GMT will handle the projection for both grid and coastline.
    """
    # get lon/lat region that covers the UTM grid
    region_ll = bounds_lonlat_from_utm(tif_utm, epsg_utm=utm_epsg)

    # load raster as xarray (not strictly required; we can pass path directly)
    da = xarray_from_geotiff(tif_utm)
    z = da.values
    zmin = float(np.nanpercentile(z, 1))
    zmax = float(np.nanpercentile(z, 99))

    fig = pygmt.Figure()
    pygmt.makecpt(cmap="geo", series=[zmin, zmax], continuous=True)
    shade = pygmt.grdgradient(grid=str(tif_utm), radiance=[135, 30], normalize="t1")

    # Use UTM map projection so coastlines render correctly
    proj = f"U{utm_zone}/12c"

    fig.grdimage(
        grid=str(tif_utm),         # let GMT read the GeoTIFF directly
        region=region_ll,          # geographic region (lon/lat)
        projection=proj,           # UTM map projection
        cmap=True,
        shading=shade,
        frame=['WSen', 'xafg+l"Longitude"', 'yafg+l"Latitude"'],
    )
    fig.coast(region=region_ll, projection=proj,
              shorelines="5p,black", resolution="f")
    fig.colorbar(frame='+l"Elevation (m)"')
    fig.savefig(str(out_png))

if __name__ == "__main__":
    plot_wgs84(TIF_WGS, TIF_WGS.with_name(TIF_WGS.stem + "_map_wgs84.png"))
    plot_utm(TIF_UTM, TIF_UTM.with_name(TIF_UTM.stem + "_map_utm.png"))

    fig1 = topo_map(
        dem_tif=TIF_UTM,
        zoom_level=0,           # try 1, 2, 3… now honored even with dem_tif
        topo_color=False,       # use gray base
        cmap="gray",
        add_shading=True,       # GMT hillshade from your raster
        #land_fill="200/200/200",# <- NEW: uniform land overlay (light gray)
        inv=inv,
        add_labels=True,
        outfile = TIF_UTM.with_name(TIF_UTM.stem + "_map_gray.png"),
        show=False,
)