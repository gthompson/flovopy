import numpy as np
from obspy import read, Stream,  UTCDateTime #, Catalog, Event,
#from obspy.core.event import Origin, Magnitude, Comment


def _fix_legacy_id(trace, network=None):
    """
    Fixes legacy trace IDs for old VDAP/analog telemetry networks.

    This function corrects **four-character station codes** where the orientation
    is embedded within the station name.

    Corrections:
    - If `trace.stats.station == 'IRIG'`, sets `trace.stats.channel = 'ACE'`.
    - Converts single-letter channels ('v', 'n', 'e') to SEED channel names ('EHZ', 'EHN', 'EHE').
    - Extracts orientation (4th character of station name) to determine the correct SEED channel.
    - Removes the orientation character from the station name.

    Parameters:
    ----------
    trace : obspy.Trace
        The seismic trace to modify.

    Returns:
    -------
    None
        Modifies `trace.stats.station` and `trace.stats.channel` in place.

    Example:
    --------
    ```python
    from obspy import read

    trace = read("example.mseed")[0]
    _fix_legacy_id(trace)

    print(trace.id)  # Corrected SEED-compliant ID
    ```
    """
    if network:
        trace.stats.network = network
    if trace.stats.station == 'IRIG':
        trace.stats.channel = 'ACE'
    else:
        chan = ""
        if trace.stats.channel:
            chan = trace.stats.channel.upper() 
        elif len(trace.stats.station)==4:
            chan  = trace.stats.station[3].upper()
            trace.stats.station = trace.stats.station[0:3].strip()

        if chan:
            if chan in 'VZT':
                trace.stats.channel='EHZ'
            elif chan=='N':
                trace.stats.channel='EHN'
            elif chan=='E':
                trace.stats.channel='EHE'   
            elif chan=='L':
                trace.stats.channel='ELZ' 
            elif chan=='P':
                trace.stats.channel='EDF'
            else:
                trace.stats.channel='EH' + chan                         
        
        """
        orientation = trace.stats.station[3].strip()  # Position 4        
        if orientation in "ZNE":  # Standard orientations
            channel = f"EH{orientation}"
        elif orientation == "L":  # Special case for "L"
            channel = "ELZ"
        elif orientation == 'P': # assume pressure sensor?
            channel = 'EDF'
        else:
            channel = f'??{orientation}'
            #raise ValueError(f"Unknown orientation '{orientation}' in '{station}'")
        
        trace.stats.channel = channel
        """
        #trace.stats.station = trace.stats.station[0:3].strip()  # Positions 1-3


#######################################################################
##               Read event metadata                                 ##
#######################################################################

def parse_hypo71_line(line):
    """
    Parses a single line of **HYPO71** earthquake location output.

    This function extracts **event origin time, location, depth, magnitude, and residuals**
    from a **fixed-column** formatted HYPO71 output line.

    Parameters:
    ----------
    line : str
        A single line of HYPO71 output.

    Returns:
    -------
    dict or None
        A dictionary containing extracted earthquake information, or `None` if parsing fails:
        ```
        {
            "origin_time": UTCDateTime,
            "latitude": float,
            "longitude": float,
            "depth": float (km),
            "magnitude": float,
            "n_ass": int,          # Number of associated arrivals
            "time_residual": float # RMS time residual in seconds
        }
        ```

    Notes:
    ------
    - The function handles **two-digit years**, assuming `year >= 70` belongs to the 1900s.
    - Converts **latitude and longitude from degrees + minutes** to decimal degrees.
    - Handles special cases where **minute=60** by rolling over to the next hour.

    Example:
    --------
    ```python
    # Example HYPO71 output line
    line = "230301 1205 45.2N 067 12.3W  10.0 3.2 15 0.2"

    # Parse earthquake information
    event_data = parse_hypo71_line(line)

    # Print extracted details
    print(event_data)
    ```
    """
    try:
        # Extract fields using fixed positions
        year = int(line[0:2])
        month = int(line[2:4])
        day = int(line[4:6])
        hour = int(line[7:9]) if line[7:9].strip() else 0
        minute = int(line[9:11]) if line[9:11].strip() else 0
        seconds = float(line[12:17]) if line[12:17].strip() else 0
        
        lat_deg = int(line[17:20].strip())
        lat_min = float(line[21:26].strip())
        lat_hem = line[20].strip().upper()
        
        lon_deg = int(line[27:30].strip())
        lon_min = float(line[31:36].strip())
        lon_hem = line[30].strip().upper()
        
        depth = float(line[37:43].strip())
        magnitude = float(line[44:50].strip())
        n_ass = int(line[51:53].strip())
        time_residual = float(line[62:].strip())
        
        # Handle two-digit years
        year = year + 1900 if year >= 70 else year + 2000

        # handle minute=60
        add_seconds = 0
        if minute==60:
            minute = 0
            add_seconds = 60       
        
        # Convert to UTCDateTime
        origin_time = UTCDateTime(year, month, day, hour, minute, seconds) + add_seconds
        
        # Convert latitude and longitude
        latitude = lat_deg + lat_min / 60.0
        if lat_hem == 'S':
            latitude = -latitude
        
        longitude = lon_deg + lon_min / 60.0
        if lon_hem == 'W':
            longitude = -longitude
        
        return {
            "origin_time": origin_time,
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "magnitude": magnitude,
            "n_ass": n_ass,
            "time_residual": time_residual
        }
    except Exception as e:
        print(f"Failed to parse line: {line.strip()} | Error: {e}")
        return None    

def parse_hypo71_file(file_path):
    """
    Parses an entire **HYPO71 earthquake catalog file** into an ObsPy Catalog object.

    This function reads a **HYPO71 output file**, extracts event metadata from each line,
    and converts the information into an **ObsPy Catalog**.

    Parameters:
    ----------
    file_path : str
        Path to the HYPO71 output file.

    Returns:
    -------
    tuple:
        - **catalog (obspy.Catalog)**: A catalog containing `Event` objects with `Origin` and `Magnitude` attributes.
        - **unparsed_lines (list of str)**: A list of lines that could not be parsed.

    Notes:
    ------
    - Extracted **events include**:
      - **Origin time, latitude, longitude, depth, and magnitude.**
      - **Number of associated arrivals** (stored as an ObsPy `Comment`).
      - **RMS time residual** (stored as an ObsPy `Comment`).
    - If parsing fails for a line, the function prints a warning and stores it in `unparsed_lines`.

    Example:
    --------
    ```python
    # Parse a HYPO71 catalog file
    catalog, errors = parse_hypo71_file("hypo71_catalog.txt")

    # Print parsed events
    print(catalog)

    # Print unparsed lines (if any)
    print(errors)
    ```
    """
    catalog = Catalog()
    parsed = 0
    not_parsed = 0
    unparsed_lines = []
    with open(file_path, "r") as file:
        for line in file:
            #print(line)
            #event_data = parse_hypo71_line(line.strip())
            #if not event_data:
            event_data = parse_hypo71_line(line.strip())
            if event_data:
                parsed +=1
                #print(event_data)
                event = Event()
                origin = Origin(
                    time=event_data["origin_time"],
                    latitude=event_data["latitude"],
                    longitude=event_data["longitude"],
                    depth=event_data["depth"] * 1000  # Convert km to meters
                )
                magnitude = Magnitude(mag=event_data["magnitude"])
                
                # Store number of associated arrivals and time residual as comments
                origin.comments.append(Comment(text=f"n_ass: {event_data['n_ass']}"))
                origin.comments.append(Comment(text=f"time_residual: {event_data['time_residual']} sec"))

                event.origins.append(origin)
                event.magnitudes.append(magnitude)
                #print(event)
                catalog.append(event)
            else:
                print(line)
                not_parsed +=1
                unparsed_lines.append(line)
        
    print(f'parsed={parsed}, not parsed={not_parsed}')

    return catalog, unparsed_lines

#######################################################################
##               Read waveform formats                               ##
#######################################################################

def read_DMX_file(DMXfile, fix=True, defaultnet=''):
    """
    Reads a **DMX** waveform file into an ObsPy Stream, applying optional corrections.

    This function reads a **demultiplexed SUDS (DMX) file**, corrects its metadata, and applies
    data adjustments for compatibility with SAC and MiniSEED formats.

    Parameters:
    ----------
    DMXfile : str
        Path to the DMX file.
    fix : bool, optional
        If `True`, applies metadata and amplitude corrections (default: True).
    defaultnet : str, optional
        Default network code to assign if the original value is missing (default: '').

    Returns:
    -------
    obspy.Stream
        A Stream object containing seismic traces from the DMX file.

    Notes:
    ------
    - The **ObsPy DMX reader** inserts `"unk"` for unknown networks; this function corrects it.
    - Data samples in DMX are **stored as unsigned 16-bit integers**. This function:
      - Converts data to **floating point** for MiniSEED compatibility.
      - Adjusts values by **subtracting 2048** to match SAC file conventions.

    Example:
    --------
    ```python
    # Read and correct a DMX file
    st = read_DMX_file("seismic_data.dmx", fix=True, defaultnet="IU")

    # Print trace details
    print(st)
    ```
    """
    print('Reading %s' % DMXfile)
    st = Stream()
    try:
        st = read(DMXfile)
        print('- read okay')
        if fix:
            for tr in st:
                # ObsPy DMX reader sets network to "unk" if blank. We'd rather keep it blank, or 
                # set with explicitly passing defaultnet named argument.
                if tr.stats.network == 'unk':
                    tr.stats.network = defaultnet
                    
                # ObsPy DMX reader falses adds 2048 to each data sample. Remove that here.
                # Also change data type of tr.data from uint to float so we can write trace to MiniSEED later   
                tr.data = tr.data.astype(float) - 2048.0 
    except:
        print('- ObsPy cannot read this demultiplexed SUDS file')        
    return st