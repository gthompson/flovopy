from __future__ import annotations
from typing import Iterable, Tuple
from obspy.geodetics.base import gps2dist_azimuth

def compute_azimuthal_gap(origin_lat: float, origin_lon: float,
                          station_coords: Iterable[Tuple[float, float]]) -> tuple[float, int]:
    """
    Compute the classical azimuthal gap and station count.

    station_coords: iterable of (lat, lon)
    Returns: (max_gap_deg, n_stations)
    """
    azimuths = []
    for stalat, stalon in station_coords:
        _, az, _ = gps2dist_azimuth(origin_lat, origin_lon, stalat, stalon)
        azimuths.append(float(az))

    n = len(azimuths)
    if n < 2:
        return 360.0, n

    azimuths.sort()
    azimuths.append(azimuths[0] + 360.0)
    gaps = [azimuths[i+1] - azimuths[i] for i in range(n)]
    return max(gaps), n