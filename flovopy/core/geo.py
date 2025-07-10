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

def get_distance(lat1, lon1, lat2, lon2):
    from obspy.geodetics import gps2dist_azimuth


    # Calculate distance and azimuths
    distance_m, azimuth_deg, back_azimuth_deg = gps2dist_azimuth(lat1, lon1, lat2, lon2)

    # Print results
    print(f"Distance: {distance_m:.2f} m")
    print(f"Bearing (azimuth A ➝ B): {azimuth_deg:.2f}°")
    print(f"Back azimuth (B ➝ A): {back_azimuth_deg:.2f}°")

def plot_arrays():
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load CSV into DataFrame
    df = pd.read_csv("/Users/glennthompson/Dropbox/DATA/station_metadata/ksc_stations_cleaned.csv", encoding='latin1')

    # Clean whitespace (optional if not already cleaned)
    df.columns = df.columns.str.strip()

    # Following line deprecated. Next line added.
    #df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.apply(lambda col: col.map(str.strip) if col.dtypes == "object" else col)

    # Select rows 
 
    subset = df.iloc[4:18]  #0:47

    # Extract lat/lon and labels
    rownums = subset['rownum']
    lats = subset['lat']
    lons = subset['lon']
    locations = subset['location']
    labels = subset['channel']

    origin_lon = -80.572248
    origin_lat = 28.573614
    #print(lons-28.573614, lats-80.572248, labels)

    # Plot
    plt.figure(figsize=(12, 12))
    for rownum, lon, lat, label, location  in zip(rownums, lons, lats, labels, locations):
        lon = float(lon)
        lat = float(lat)
        if rownum in (5, 7, 9, 10, 18, 19, 23, 20,21,22, 24) or rownum in range(35, 47):
            plt.plot(lon-origin_lon, lat-origin_lat, 'g*')
        else:
            plt.plot(lon-origin_lon, lat-origin_lat, 'ro')
        plt.text(lon-origin_lon, lat-origin_lat, f' {rownum}', fontsize=14, verticalalignment='bottom')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Station Map with Channel Labels")
    plt.grid(True)
    #plt.axis("equal")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_arrays()