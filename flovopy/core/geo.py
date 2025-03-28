import numpy as np
from obspy import Stream
from obspy.geodetics import locations2degrees, degrees2kilometers

def attach_distance_to_stream(st, olat, olon):
    """
    Attaches distance (in meters) from a reference point to each trace in a Stream.

    Parameters:
    ----------
    st : obspy.Stream
        Stream of traces, each with coordinate metadata in `tr.stats['coordinates']`.
    olat : float
        Latitude of the origin (e.g., epicenter or station).
    olon : float
        Longitude of the origin.

    Returns:
    -------
    None
        Modifies the Stream in place by setting `tr.stats.distance` (in meters).
    """
    for tr in st:
        try:
            alat = tr.stats['coordinates']['latitude']
            alon = tr.stats['coordinates']['longitude']
            distdeg = locations2degrees(olat, olon, alat, alon)
            distkm = degrees2kilometers(distdeg)
            tr.stats['distance'] = distkm * 1000.0
        except Exception as e:
            print(f'Cannot compute distance for {tr.id}: {e}')


def get_distance_vector(st):
    """
    Extracts a list of distances from a Stream.

    Parameters:
    ----------
    st : obspy.Stream
        Stream with `tr.stats.distance` values.

    Returns:
    -------
    list of float
        Distances (in meters) for each trace.
    """
    return [tr.stats.distance for tr in st]


def order_traces_by_distance(st, r=None, assert_channel_order=False):
    """
    Orders traces in a Stream by distance, optionally enforcing channel order.

    Parameters:
    ----------
    st : obspy.Stream
        Stream of traces with `tr.stats.distance` attributes.
    r : list of float, optional
        Precomputed distance list (in meters). If None, computed from trace metadata.
    assert_channel_order : bool, optional
        If True, breaks ties in distance using location code and channel component.

    Returns:
    -------
    obspy.Stream
        Stream with traces sorted by increasing distance (and channel, if enabled).
    """
    if r is None:
        r = get_distance_vector(st)

    r = list(r)  # Make sure it's mutable

    if assert_channel_order:
        for i, tr in enumerate(st):
            try:
                c1 = int(tr.stats.location) / 1_000_000
            except:
                c1 = 0
            numbers = 'ZNEF0123456789'
            try:
                c2 = numbers.find(tr.stats.channel[2]) / 1_000_000_000
            except:
                c2 = 0
            r[i] += c1 + c2

    indices = np.argsort(r)
    return Stream([st[i].copy() for i in indices])

