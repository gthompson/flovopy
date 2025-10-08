# flovopy/asl/envelope_locate.py (or a nearby helpers file)

from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np
from obspy import Stream, Inventory
from flovopy.asl.distances import geo_distance_3d_km

DomeLike = Union[int, Tuple[float, float], Dict[str, float]]  # we only need lon/lat (and maybe elev)

def _as_lon_lat_elev(dome_location: DomeLike) -> Tuple[float, float, Optional[float]]:
    """
    Accepts:
      • (lon, lat) or (lon, lat, elev_m)
      • {'lon':..., 'lat':..., 'elev':...}
    Returns (lon, lat, elev_m or None)
    """
    if isinstance(dome_location, dict):
        lon = float(dome_location["lon"])
        lat = float(dome_location["lat"])
        elev_m = float(dome_location.get("elev")) if "elev" in dome_location else None
        return lon, lat, elev_m
    if isinstance(dome_location, (tuple, list)):
        if len(dome_location) == 2:
            lon, lat = float(dome_location[0]), float(dome_location[1])
            return lon, lat, None
        if len(dome_location) >= 3:
            lon, lat = float(dome_location[0]), float(dome_location[1])
            elev_m = float(dome_location[2])
            return lon, lat, elev_m
    raise ValueError("dome_location must be (lon,lat), (lon,lat,elev_m), or {'lon','lat'[,'elev']}")


def compute_travel_times_from_inventory(
    inv: Inventory,
    dome_location: DomeLike,
    *,
    speed_km_s: float,
    stations_subset: Optional[Iterable[str]] = None,
    use_elevation: bool = True,
) -> Dict[str, float]:
    """
    Build {STA -> travel_time_seconds} using station positions from Inventory
    and the great-circle + vertical distance to the dome. Station elevations
    (and optional dome elevation) are included if use_elevation=True.

    Parameters
    ----------
    inv : obspy.Inventory
    dome_location : (lon,lat[,(elev_m)]) or {'lon','lat'[,'elev']}
    speed_km_s : float
        Wave speed to convert distance to time.
    stations_subset : iterable[str] | None
        If provided, limit to these station codes (e.g., those present in the Stream).
    use_elevation : bool
        If False, the vertical term is ignored (elevations set to None).

    Returns
    -------
    dict[str, float]
        Seconds to subtract from each station trace’s starttime.
    """
    if not np.isfinite(speed_km_s) or speed_km_s <= 0:
        raise ValueError(f"speed_km_s must be positive; got {speed_km_s}")

    lon0, lat0, elev0 = _as_lon_lat_elev(dome_location)
    if not use_elevation:
        elev0 = None

    want = set(s.upper() for s in stations_subset) if stations_subset else None
    tt: Dict[str, float] = {}

    for net in inv.networks:
        for sta in net.stations:
            sta_code = sta.code.upper()
            if want and sta_code not in want:
                continue

            lat = float(sta.latitude)
            lon = float(sta.longitude)
            elev_m = float(sta.elevation) if use_elevation else None

            d_km = geo_distance_3d_km(
                lat0, lon0, elev0,
                lat, lon, elev_m
            )
            tt[sta_code] = d_km / float(speed_km_s)  # s

    return tt


def shift_stream_by_travel_time(
    st: Stream,
    inv: Inventory,
    dome_location: DomeLike,
    *,
    speed_km_s: float,
    use_elevation: bool = True,
    inplace: bool = False,
    verbose: bool = False,
    trim: bool = True,
) -> Tuple[Stream, Dict[str, float]]:
    """
    Reduce each trace’s starttime by the dome→station travel time computed
    from Inventory + speed.

    Effect:
      tr.stats.starttime -= distance(dome, station) / speed_km_s

    Returns
    -------
    shifted_stream, travel_times_s_by_station
    """
    sta_in_stream = sorted({tr.stats.station.upper() for tr in st})
    tt = compute_travel_times_from_inventory(
        inv,
        dome_location,
        speed_km_s=speed_km_s,
        stations_subset=sta_in_stream,
        use_elevation=use_elevation,
    )

    out = st if inplace else st.copy()
    for tr in out:
        sta = tr.stats.station.upper()
        dt = tt.get(sta)
        if dt is not None and np.isfinite(dt):
            tr.stats.starttime -= float(dt)

    if trim:
        stime = max([tr.stats.starttime for tr in out])
        etime = min([tr.stats.endtime for tr in out])
        out.trim(starttime=stime, endtime=etime)

    if verbose:
        print(out, tt)

    return out, tt