import os
from obspy import read
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees


######################################################################
##                  Modeling  tools                                 ##
######################################################################

def predict_arrival_times(station, quake):
    """
    Computes predicted seismic phase arrival times for a given station and earthquake using the IASP91 model.

    This function calculates the expected arrival times of seismic phases at a given station
    based on the earthquake's origin time, latitude, longitude, and depth.

    Parameters:
    ----------
    station : dict
        Dictionary containing station metadata:
        ```
        {
            "lat": float,   # Station latitude in degrees
            "lon": float    # Station longitude in degrees
        }
        ```
    quake : dict
        Dictionary containing earthquake metadata:
        ```
        {
            "lat": float,   # Earthquake latitude in degrees
            "lon": float,   # Earthquake longitude in degrees
            "depth": float, # Earthquake depth in km
            "otime": UTCDateTime  # Origin time of the earthquake
        }
        ```

    Returns:
    -------
    dict
        The updated `station` dictionary with a new `phases` key containing predicted arrival times:
        ```
        station["phases"] = {
            "P": "12:45:30",
            "S": "12:47:10",
            "Rayleigh": "12:48:00"
        }
        ```

    Notes:
    ------
    - Uses the **IASP91 travel-time model** via ObsPy's `TauPyModel`.
    - The **Rayleigh wave arrival** is estimated based on the S-wave arrival time.
    - Distances are computed using **gps2dist_azimuth** and converted to degrees.

    Example:
    --------
    ```python
    from obspy import UTCDateTime

    # Define station and earthquake metadata
    station = {"lat": 35.0, "lon": -120.0}
    quake = {"lat": 34.0, "lon": -118.0, "depth": 10.0, "otime": UTCDateTime("2023-03-01T12:45:00")}

    # Compute arrival times
    station = predict_arrival_times(station, quake)

    # Print predicted arrivals
    print(station["phases"])
    ```
    """
    model = TauPyModel(model="iasp91")
    
    [dist_in_m, az1, az2] = gps2dist_azimuth(quake['lat'], quake['lon'], station['lat'], station['lon'])
    station['distance'] = kilometers2degrees(dist_in_m/1000)
    arrivals = model.get_travel_times(source_depth_in_km=quake['depth'],distance_in_degree=station['distance'])
    # https://docs.obspy.org/packages/autogen/obspy.taup.helper_classes.Arrival.html#obspy.taup.helper_classes.Arrival
    
    phases = dict()
    for a in arrivals:
        phasetime = quake['otime'] + a.time
        phases[a.name] = phasetime.strftime('%H:%M:%S')
        if a.name == 'S':
            Rtime = quake['otime'] + a.time/ ((0.8453)**0.5)
            phases['Rayleigh'] = Rtime.strftime('%H:%M:%S')
    station['phases'] = phases
    
    return station

def syngine2stream(station, lat, lon, GCMTeventID, mseedfile):
    """
    Retrieves synthetic seismograms from IRIS Syngine for a specified GCMT event.

    This function generates synthetic seismograms for a given station and GCMT earthquake event.
    If a MiniSEED file already exists, it reads from that file; otherwise, it queries **IRIS Syngine**.

    Parameters:
    ----------
    station : str
        Station name (used for metadata storage).
    lat : float
        Latitude of the station (in degrees).
    lon : float
        Longitude of the station (in degrees).
    GCMTeventID : str
        Global Centroid Moment Tensor (GCMT) event ID.
    mseedfile : str
        Filename to save the downloaded synthetic waveform.

    Returns:
    -------
    obspy.Stream
        A Stream object containing the synthetic seismograms.

    Notes:
    ------
    - The function requests **displacement waveforms** with a sampling interval of 0.02s.
    - If the `mseedfile` already exists, it reads from the file instead of making a new request.
    - After downloading, the synthetic traces are assigned **latitude and longitude metadata**.

    Example:
    --------
    ```python
    # Generate synthetic seismograms for a GCMT event
    st = syngine2stream("ANMO", 34.95, -106.45, "202012312359A", "synthetic.mseed")

    # Plot the synthetic waveforms
    st.plot()
    ```
    """
    if os.path.exists(mseedfile):
        synth_disp = read(mseedfile)
    else:
        synth_disp = read("http://service.iris.edu/irisws/syngine/1/query?"
                  "format=miniseed&units=displacement&dt=0.02&"
                  "receivercenterlat=%f&receivercenterlon=%f&"
                  "eventid=GCMT:%s" % (lat, lon, GCMTeventID))
        for c in range(len(synth_disp)):
            synth_disp[c].stats.centerlat = lat
            synth_disp[c].stats.centerlon = lon
        synth_disp.write(mseedfile)
    return synth_disp