from pathlib import Path
import rasterio
from rasterio.transform import Affine

def shift_geotiff(infile: Path, outfile: Path, dx: float = 0.0, dy: float = 0.0):
    """
    Shift a GeoTIFF by dx, dy in the units of the CRS (e.g., meters for UTM).
    
    Parameters
    ----------
    infile : Path
        Path to input GeoTIFF.
    outfile : Path
        Path to write shifted GeoTIFF.
    dx : float
        Shift in x-direction (positive east, negative west).
    dy : float
        Shift in y-direction (positive north, negative south).
    """
    with rasterio.open(infile) as src:
        data = src.read(1)
        profile = src.profile.copy()

        # Original transform
        transform = src.transform
        # Apply shift: adjust translation terms
        new_transform = transform * Affine.translation(dx, dy)

        profile.update(transform=new_transform)

        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(data, 1)