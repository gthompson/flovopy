import os
import glob
import shutil
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core.event import Event, Comment, Catalog
from obspy.core.event.resourceid import ResourceIdentifier


#######################################################################    
########################         WFDISC tools                        ##
#######################################################################
     
def index_waveformfiles(wffiles, ampeng=False, events=True):
    """
    Indexes a list of seismic waveform files and returns a DataFrame similar to a `wfdisc` table.

    This function reads seismic waveform files, extracts relevant metadata, and compiles the 
    information into a **pandas DataFrame** with fields similar to a `wfdisc` database table.
    Additionally, it creates an **ObsPy Catalog** of `Event` objects if `events=True`.

    **Primary Use Cases:**
    - Organizing waveform metadata for seismic data archives (e.g., CALIPSO data).
    - Preparing a catalog of waveform files with event associations.
    - Computing basic waveform statistics such as **amplitude and energy** (optional).

    Parameters:
    ----------
    wffiles : list of str
        List of file paths to seismic waveform data files (MiniSEED, SAC, etc.).
    ampeng : bool, optional
        If `True`, computes **maximum absolute amplitude** and **signal energy** for each trace (default: False).
    events : bool, optional
        If `True`, generates an **ObsPy Catalog** with `Event` objects associated with each file (default: True).

    Returns:
    -------
    tuple:
        - **wfdisc_df (pandas.DataFrame)**:
            A DataFrame containing metadata for each waveform file with columns:
            ```
            file_index    : Index of the file in `wffiles`
            event_id      : Unique Event ID (ObsPy ResourceIdentifier)
            waveform_id   : ResourceIdentifier linking file name and trace index
            traceID       : SEED-formatted trace ID (NET.STA.LOC.CHA)
            starttime     : UTCDateTime of trace start
            endtime       : UTCDateTime of trace end
            npts          : Number of data points
            sampling_rate : Sampling rate (Hz)
            calib         : Calibration factor
            ddir          : Directory containing the waveform file
            dfile         : Waveform file name
            duration      : Trace duration (seconds)
            amplitude     : (Optional) Maximum absolute amplitude of the detrended trace
            energy        : (Optional) Sum of squared amplitudes (energy estimate)
            ```

        - **cat (obspy.Catalog)**:
            If `events=True`, returns an **ObsPy Catalog** containing `Event` objects with metadata 
            extracted from the waveform files. If `events=False`, returns only the `wfdisc_df`.

    Notes:
    ------
    - If a file **cannot be read**, it is skipped, and a warning is printed.
    - If `ampeng=True`, the function attempts to **detrend** the waveform before computing amplitude and energy.
    - The function **sorts** the DataFrame by `starttime` before returning it.

    Example:
    --------
    ```python
    from obspy import read
    import glob

    # List of waveform files
    waveform_files = glob.glob("/data/seismic/*.mseed")

    # Index waveform files and compute amplitude/energy
    wfdisc_df, cat = index_waveformfiles(waveform_files, ampeng=True)

    # Display waveform metadata
    print(wfdisc_df.head())

    # Save the catalog
    cat.write("seismic_catalog.xml", format="QUAKEML")
    ```
    """
    wfdisc_df = pd.DataFrame()
    file_index = []
    event_id = []
    traceids = []
    starttimes = []
    endtimes = []
    sampling_rates = []
    calibs = []
    ddirs = []
    dfiles = []
    npts = []
    duration = []
    waveform_id = []
    if ampeng:
        amp = []
        eng = []

    events = []
    for filenum, wffile in enumerate(sorted(wffiles)):
        dfile = os.path.basename(wffile)
        ddir = os.path.dirname(wffile)
        try:
            this_st = read(wffile)
            print('Read %s\n' % wffile)
        except:
            print('Could not read %s\n' % wffile)
            next
        else:
            ev = Event()
            comments = []
            stime = this_st[0].stats.starttime
            sfilename = stime.strftime("%d-%H%M-%S") + "L.S" + stime.strftime("%Y%m")  
            comments.append(Comment(text=f'wavfile: {wffile}'))
            comments.append(Comment(text=f'sfile: {sfilename}'))                      
            for tracenum, this_tr in enumerate(this_st):
                file_index.append(filenum)
                event_id.append(ev.resource_id)  
                wid = ResourceIdentifier(f'{dfile},[{tracenum}]')
                waveform_id.append(wid)
                r = this_tr.stats
                traceids.append(this_tr.id)
                starttimes.append(r.starttime)
                endtimes.append(r.endtime)
                sampling_rates.append(r.sampling_rate)
                calibs.append(r.calib)
                ddirs.append(ddir)
                dfiles.append(dfile)
                npts.append(r.npts)
                duration.append(r.endtime - r.starttime)
                if ampeng:
                    try:
                        this_tr.detrend('linear')
                    except:
                        this_tr.detrend('constant')
                    y = abs(this_tr.data)
                    amp.append(np.nanmax(y))
                    y = np.nan_to_num(y, nan=0)
                    eng.append(np.sum(np.square(y)))
            ev.comments = comments
        events.append(ev)
      
    if wffiles:
        wfdisc_dict = {'file_index':file_index, 'event_id':event_id, 'waveform_id':waveform_id, 'traceID':traceids, 'starttime':starttimes, 'endtime':endtimes, 'npts':npts, 
                       'sampling_rate':sampling_rates, 'calib':calibs, 'ddir':ddirs, 'dfile':dfiles, 'duration':duration}
        if ampeng:
            wfdisc_dict['amplitude']=amp
            wfdisc_dict['energy']=eng
        wfdisc_df = pd.DataFrame.from_dict(wfdisc_dict)  
        wfdisc_df.sort_values(['starttime'], ascending=[True], inplace=True)
    if events:
        cat = Catalog(events=events) 
        return wfdisc_df, cat
    else:
        return wfdisc_df

def wfdisc_to_BUD(wfdisc_df, TOPDIR, put_away):
    """
    Converts waveform files indexed in a `wfdisc`-like DataFrame into a BUD (Buffer of Uniform Data) archive.

    This function reads waveform data from a **wfdisc-like DataFrame**, extracts individual traces grouped 
    by unique `traceID`, and writes the data into a **BUD (Buffer of Uniform Data) format archive**.

    The function processes waveform files on a **per-day basis**, merging overlapping traces and 
    ensuring that data is correctly organized in the BUD directory structure.

    **Workflow:**
    - Iterates through unique trace IDs in `wfdisc_df`.
    - Determines the earliest and latest timestamps for each trace.
    - Reads and merges waveform files for each day.
    - Writes merged traces into a **BUD-format** archive.
    - Optionally moves processed files to a `.PROCESSED` directory if `put_away=True`.

    Parameters:
    ----------
    wfdisc_df : pandas.DataFrame
        A DataFrame containing waveform metadata (as produced by `index_waveformfiles`).
    TOPDIR : str
        The top-level directory where the BUD archive will be stored.
    put_away : bool
        If `True`, moves successfully processed waveform files to a `.PROCESSED` directory (default: False).

    Returns:
    -------
    None
        The function processes the waveform files and writes data to the BUD archive.

    Notes:
    ------
    - The function processes waveform data **in daily chunks**, merging traces that overlap.
    - Files that cannot be read are skipped with a warning.
    - `Stream_to_BUD(TOPDIR, all_traces)` is used to write the BUD archive.
    - Successfully processed waveform files can be optionally moved using `shutil.move()`.

    Example:
    --------
    ```python
    import pandas as pd

    # Load an example wfdisc DataFrame
    wfdisc_df = pd.read_csv("waveform_index.csv")

    # Convert to BUD archive
    wfdisc_to_BUD(wfdisc_df, "/data/BUD_archive", put_away=True)
    ```
    """

    unique_traceIDs = wfdisc_df['traceID'].unique().tolist()
    print(unique_traceIDs)
    
    successful_wffiles = list()

    for traceID in unique_traceIDs:
        print(traceID)
        
        trace_df = wfdisc_df[wfdisc_df['traceID']==traceID]
        
        # identify earliest start time and latest end time for this channel
        #print(trace_df.iloc[0]['starttime'])
        #print(trace_df.iloc[-1]['endtime'])
        minUTC = trace_df.starttime.min()
        maxUTC = trace_df.endtime.max()
        start_date = minUTC.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = maxUTC.replace(hour=23, minute=59, second=59, microsecond=999999)
        this_date = start_date

        while this_date <= end_date: 
            all_traces = Stream()
        
            # loop from earliest start day to latest end day
            subset_df = trace_df[(trace_df['starttime'] < this_date+86400) & (trace_df['endtime'] >= this_date)]
            #print(subset_df)
            
            if len(subset_df.index)==0:
                next
        
            for index, row in subset_df.iterrows():
                wffile = os.path.join(row['ddir'], row['dfile'])
                start_at = max([this_date, row['starttime']])
                end_at = min([this_date+86400, row['endtime']])
                print('- ',wffile,': START AT:', start_at, ', END AT: ',end_at)
                try:
                    this_st = read(wffile, starttime=start_at, endtime=end_at)

                except:
                    print(' Failed\n')
                    next
                else:
                    print(' Succeeded\n')
                    #raise Exception("Stopping here")
                    if end_at == row['endtime']:
                        successful_wffiles.append(wffile)   
                    for this_tr in this_st:
                        if this_tr.id == traceID:
                            #print(tr.stats)
                            all_traces = all_traces.append(this_tr)
            #print(st.__str__(extended=True)) 
            try:
                all_traces.merge(fill_value=0)
            except:
                print('Failed to merge ', all_traces)
            print(all_traces.__str__(extended=True))
        
            # Check that we really only have a single trace ID before writing the BUD files
            error_flag = False
            for this_tr in all_traces:
                if not this_tr.id == traceID:
                    error_flag = True
            if not error_flag:
                try:
                    Stream_to_BUD(TOPDIR, all_traces)
                except:
                    print('Stream_to_BUD failed for ', all_traces)
            
            this_date += 86400
            
    for wffile in successful_wffiles:
        ddir = os.path.dirname(wffile)
        dbase = "%s.PROCESSED" % os.path.basename(wffile)
        newwffile = os.path.join(ddir, dbase)
        print('move %s %s' % (wffile, newwffile))
        if os.path.exists(wffile) and put_away:
            shutil.move(wffile, newwffile)

def process_wfdirs(wfdirs, filematch, put_away=False):
    """
    Processes directories containing waveform data, indexing the files and converting them to a BUD archive.

    This function scans **one or more directories** (`wfdirs`) for seismic waveform files matching a given 
    pattern (`filematch`), builds a **wfdisc-like DataFrame**, and converts the waveform data into a 
    **BUD (Buffer of Uniform Data) archive**.

    **Workflow:**
    - Scans the provided directories (`wfdirs`) for waveform files.
    - Calls `index_waveformfiles()` to extract waveform metadata into a **pandas DataFrame**.
    - Calls `wfdisc_to_BUD()` to convert and store waveform data in the **BUD archive**.
    - Optionally moves processed files if `put_away=True`.

    Parameters:
    ----------
    wfdirs : list of str
        List of directories containing waveform files.
    filematch : str
        File matching pattern (e.g., `"*.mseed"`, `"*.sac"`) to identify waveform files.
    put_away : bool, optional
        If `True`, moves successfully processed waveform files to a `.PROCESSED` directory (default: False).

    Returns:
    -------
    None
        The function scans the directories, processes waveform data, and writes to the BUD archive.

    Notes:
    ------
    - The function processes all files readable by **ObsPy**.
    - If no waveform files are found in a directory, it is skipped.
    - Calls `wfdisc_to_BUD()` to handle the conversion process.

    Example:
    --------
    ```python
    import glob

    # Define directories and file pattern
    waveform_dirs = ["/data/seismic/day1", "/data/seismic/day2"]
    file_pattern = "*.mseed"

    # Process and convert to BUD archive
    process_wfdirs(waveform_dirs, file_pattern, put_away=True)
    ```
    """

    for wfdir in wfdirs:
        print('Processing %s' % wfdir)
        wffiles = glob.glob(os.path.join(wfdir, filematch))
        if wffiles:
            #print(wffiles)
            wfdisc_df = index_waveformfiles(wffiles)
            #print(wfdisc_df)
            if not wfdisc_df.empty:
                wfdisc_to_BUD(wfdisc_df, TOPDIR, put_away)  
    print('Done.')

#######################################################################
##                BUD tools                                          ##
#######################################################################


def Stream_to_BUD(TOPDIR, all_traces):
    """
    Converts a Stream object into the IRIS/PASSCAL BUD (Buffer of Uniform Data) format.

    This function takes a **Stream** of seismic waveform data, splits it into **24-hour-long segments** per 
    channel, and writes it into a **BUD directory structure** based on station, network, and date.

    **BUD Directory Structure Example:**
    ```
    DAYS/
    ├── BHP2
    │   ├── 1R.BHP2..EH1.2020.346
    │   ├── 1R.BHP2..EH2.2020.346
    │   └── 1R.BHP2..EHZ.2020.346
    ├── FIREP
    │   ├── 1R.FIREP..EH1.2020.346
    │   ├── 1R.FIREP..EH2.2020.346
    │   └── 1R.FIREP..EHZ.2020.346
    ```
    where:
    - `BHP2`, `FIREP`, etc., are station names.
    - `1R` is the network name.
    - Channels are `EH[Z12]`.
    - The year is `2020`, and the Julian day is `346`.

    Parameters:
    ----------
    TOPDIR : str
        The top-level directory where the BUD archive will be stored.
    all_traces : obspy.Stream
        The input Stream object containing multiple seismic traces.

    Returns:
    -------
    None
        The function writes files in **MiniSEED format** to the BUD archive.

    Notes:
    ------
    - The function ensures that traces are merged and padded to **24-hour segments**.
    - If a trace already exists in the BUD archive, it merges new data using `Trace_merge_with_BUDfile()`.
    - Created for **ROCKETSEIS** and **CALIPSO** data archives.
    
    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data
    st = read("example.mseed")

    # Convert to BUD format
    Stream_to_BUD("/data/BUD_archive", st)
    ```
    """
    
    all_traces = Stream_to_24H(all_traces)
    
    daysDir = os.path.join(TOPDIR, 'DAYS')

    for this_tr in all_traces:
        YYYY = this_tr.stats.starttime.year
        JJJ = this_tr.stats.starttime.julday
        stationDaysDir = os.path.join(daysDir, this_tr.stats.station)
        if not os.path.exists(stationDaysDir):
            os.makedirs(stationDaysDir)
            #print(stationDaysDir)
        mseedDayBasename = "%s.%04d.%03d" % (this_tr.id, YYYY, JJJ  )
        mseedDayFile = os.path.join(stationDaysDir, mseedDayBasename)
        #print(mseedDayFile)
        if os.path.exists(mseedDayFile):
            this_tr = Trace_merge_with_BUDfile(this_tr, mseedDayFile)

        this_tr.write(mseedDayFile, format='MSEED') 


    
def BUD_load_day(BUDDIR, year, jday):
    """
    Loads all seismic waveform files for a given **year** and **Julian day** from a BUD archive.

    This function searches a BUD directory structure for waveform files matching the specified 
    **year** and **Julian day**, reads them, and returns a merged **Stream**.

    Parameters:
    ----------
    BUDDIR : str
        The root directory containing the BUD archive.
    year : int
        The year of the requested data (e.g., `2020`).
    jday : int
        The Julian day of the requested data (e.g., `346` for Dec 11 in a leap year).

    Returns:
    -------
    obspy.Stream
        A Stream object containing all traces for the requested day.

    Notes:
    ------
    - The function scans all station subdirectories within `BUDDIR` for matching files.
    - If a file cannot be read, it is skipped with a warning.
    
    Example:
    --------
    ```python
    # Load waveform data for year 2020, Julian day 346
    st = BUD_load_day("/data/BUD_archive", 2020, 346)

    # Print available traces
    print(st)
    ```
    """

    all_stations = glob.glob(os.path.join(BUDDIR, '*'))
    all_traces = Stream()
    for station_dir in all_stations:
        all_files = glob.glob(os.path.join(station_dir, '*.%04d.%03d' % (year, jday)))
        for this_file in all_files:
            try:
                these_traces = read(this_file)
            except:
                print('Cannot read %s' % this_file)
            else:
                for this_tr in these_traces:
                    all_traces.append(this_tr)
    return all_traces


def Stream_to_24H(all_traces):
    """
    Pads and merges a Stream object so that each trace spans a full **24-hour period**.

    This function:
    - **Merges traces** with the same ID.
    - **Pads missing data** with zeros to create **continuous 24-hour segments**.
    - Returns a new Stream with traces that **start at 00:00:00 UTC** and end at **23:59:59 UTC**.

    Parameters:
    ----------
    all_traces : obspy.Stream
        The input Stream object containing seismic traces.

    Returns:
    -------
    obspy.Stream
        A Stream object with each trace spanning exactly **24 hours**.

    Notes:
    ------
    - Uses `Stream_min_starttime()` to determine the earliest/largest time window.
    - Pads traces using `.trim(starttime, endtime, pad=True, fill_value=0)`.
    - Used for **ROCKETSEIS** and **CALIPSO** data archives.

    Example:
    --------
    ```python
    from obspy import read

    # Load waveform data
    st = read("example.mseed")

    # Convert to 24-hour segments
    st_24h = Stream_to_24H(st)

    # Print trace info
    print(st_24h)
    ```
    """

    all_traces.merge(fill_value=0)
    min_stime, max_stime, min_etime, max_etime = Stream_min_starttime(all_traces)
    
    desired_stime = UTCDateTime(min_stime.year, min_stime.month, min_stime.day, 0, 0, 0.0)
    desired_etime = desired_stime + 86400
    
    days = Stream()
    while True:
        
        this_st = all_traces.copy()
        this_st.trim(starttime=desired_stime, endtime=desired_etime, pad=True, fill_value=0)
        for this_tr in this_st:
            days.append(this_tr)
        desired_stime += 86400
        desired_etime += 86400
        if desired_etime > max_etime + 86400:
            break
    return days



def Trace_merge_with_BUDfile(this_tr, budfile):
    """
    Merges an existing trace with a corresponding BUD file, preserving non-zero data values.

    This function:
    - Reads the existing **BUD file**.
    - Checks if `this_tr` has the **same trace ID, sampling rate, and time window**.
    - Merges the traces, prioritizing **non-zero data values**.

    Parameters:
    ----------
    this_tr : obspy.Trace
        The trace to be merged into the BUD file.
    budfile : str
        The existing BUD file path.

    Returns:
    -------
    obspy.Trace
        The merged trace, preserving **non-zero data values**.

    Notes:
    ------
    - If trace metadata (ID, sampling rate, or time range) **does not match**, the trace with 
      **more non-zero values** is returned.
    - This function prevents overwriting valuable data with zeroes.

    Example:
    --------
    ```python
    from obspy import read

    # Load an existing trace
    tr = read("new_trace.mseed")[0]

    # Merge it with an existing BUD file
    merged_tr = Trace_merge_with_BUDfile(tr, "BUD/1R.BHP2..EHZ.2020.346")
    
    # Save the merged result
    merged_tr.write("merged_trace.mseed", format="MSEED")
    ```
    """

    other_st = read(budfile)
    error_flag = False
    
    if len(other_st)>1:
        print('More than 1 trace in %s. Cannot merge.' % budfile)
        error_flag = True
        
    other_tr = other_st[0]
    if not (this_tr.id == other_tr.id):
        print('Different trace IDs. Cannot merge.')
        error_flag = True
        
    if not (this_tr.stats.sampling_rate == other_tr.stats.sampling_rate):
        print('Different sampling rates. Cannot merge.')
        error_flag = True
        
    if (abs(this_tr.stats.starttime - other_tr.stats.starttime) > this_tr.stats.delta/4):
        print('Different start times. Cannot merge.')  
        error_flag = True

    if (abs(this_tr.stats.endtime - other_tr.stats.endtime) > this_tr.stats.delta/4):
        print('Different end times. Cannot merge.')  
        error_flag = True
        
    if error_flag: # traces incompatible, so return the trace with the most non-zero values
        this_good = np.count_nonzero(this_tr.data)
        #print(this_tr.stats)
        other_good = np.count_nonzero(other_tr.data)
        #print(other_tr.stats)
        if other_good > this_good:
            return other_tr
        else:
            return this_tr
    
    else: # things are good
        indices = np.where(other_tr.data == 0)
        other_tr.data[indices] = this_tr.data[indices]
        return other_tr
