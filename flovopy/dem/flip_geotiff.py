#!/usr/bin/env python3
"""
flip_geotiff.py

Flip a GeoTIFF horizontally (W↔E), vertically (N↔S), or both.

Modes:
  - default (world-preserving): flip pixels AND adjust transform so the world position is unchanged
  - --data-only: flip pixels ONLY (content mirrored in world coords)
  - --transform-only: flip transform ONLY (labels/world axes reversed; pixels unchanged)
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine


def _flip_affine_only(T: Affine, w: int, h: int, mode: str) -> Affine:
    if mode == "horizontal":
        # a' = -a ; c' = c + a*width
        return Affine(-T.a, T.b, T.c + T.a * w, T.d, T.e, T.f)
    elif mode == "vertical":
        # e' = -e ; f' = f + e*height
        return Affine(T.a, T.b, T.c, T.d, -T.e, T.f + T.e * h)
    elif mode == "both":
        return _flip_affine_only(_flip_affine_only(T, w, h, "horizontal"), w, h, "vertical")
    else:
        raise ValueError("mode must be 'horizontal', 'vertical', or 'both'")

def flip_geotiff(infile: Path, outfile: Path, mode: str = "horizontal",
                 data_only: bool = False, transform_only: bool = False) -> None:
    infile = Path(infile); outfile = Path(outfile)

    if data_only and transform_only:
        raise ValueError("Choose either --data-only or --transform-only, not both.")

    with rasterio.open(infile) as src:
        arr = src.read(1)
        prof = src.profile.copy()
        T: Affine = src.transform
        w, h = src.width, src.height

        if transform_only:
            # Flip ONLY the affine (labels/world axes), leave pixels as-is
            T = _flip_affine_only(T, w, h, mode)

        else:
            # Flip pixels
            if mode == "horizontal":
                arr = np.fliplr(arr)
            elif mode == "vertical":
                arr = np.flipud(arr)
            elif mode == "both":
                arr = np.flipud(np.fliplr(arr))
            else:
                raise ValueError("mode must be 'horizontal', 'vertical', or 'both'")

            if not data_only:
                # world-preserving: compensate in affine so world position is unchanged
                T = _flip_affine_only(T, w, h, mode)

        prof.update(transform=T, tiled=False, compress="LZW", predictor=1, count=1)
        with rasterio.open(outfile, "w", **prof) as dst:
            dst.write(arr, 1)

    print(f"✅ Wrote flipped DEM → {outfile}  "
          f"(mode={mode}, data_only={data_only}, transform_only={transform_only})")

def main():
    p = argparse.ArgumentParser(description="Flip a GeoTIFF DEM horizontally/vertically/both.")
    p.add_argument("--infile", required=True, type=Path)
    p.add_argument("--outfile", required=True, type=Path)
    p.add_argument("--flip", required=True, choices=["horizontal", "vertical", "both"])
    p.add_argument("--data-only", action="store_true",
                   help="Flip pixels only; keep original transform (content mirrored in world).")
    p.add_argument("--transform-only", action="store_true",
                   help="Flip transform only (labels/world axes), pixels unchanged.")
    args = p.parse_args()

    flip_geotiff(args.infile, args.outfile, mode=args.flip,
                 data_only=args.data_only, transform_only=args.transform_only)

if __name__ == "__main__":
    main()