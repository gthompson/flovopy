import os
import glob
from obspy import UTCDateTime, read, read_inventory, Stream
from flovopy.sds.sds import SDSObj
from flovopy.core.mvo import fix_trace_mvo
from flovopy.stationmetadata.utils import inventory2traceid
from flovopy.core.preprocessing import preprocess_trace
from flovopy.core.sam import RSAM, VSAM, VSEM, DSAM
from obsolete.asl import dome_location
secondsPerDay = 60 * 60 * 24

# SEISAN to SDS or Antelope

def sds2db(dboutday, SDS_DIR, jday):
    allfiles = glob.glob(os.path.join(SDS_DIR, '*', '*', '*', '*.D', f'*{jday}'))
    allfilesstr = " ".join(allfiles)
    os.system(f"miniseed2db {allfilesstr} {dboutday}")

def seisandb2SDS(seisandbdir, sdsdir, startt0, endt0, net, dbout=None, round_sampling_rate=True, Montserrat=True, MBWHZ_only=False):
    print('> seisandb2SDS')
    sdsobj = SDSobj(sdsdir)
    startt = UTCDateTime(startt0.year, startt0.month, startt0.day)
    endt = UTCDateTime(endt0.year, endt0.month, endt0.day)    
    if endt>startt: # multiple days, so probably normal mode
        missing_days = sdsobj.find_which_days_missing(startt, endt, net)
        print(f'missing_days in {sdsdir} from {startt} to {endt}: {missing_days}')
    else:
        print('Checking if can read data from existing SDS')
        if sdsobj.read(startt0, endt0, skip_low_rate_channels=True, trace_ids=None, speed=1, verbose=True ):
            print(sdsobj.stream)
            missing_days = [startt]
        else:
            print(sdsobj.stream)
            missing_days = []            
    mseeddir = 'seisan2mseed'
    if not os.path.isdir(mseeddir):
        os.makedirs(mseeddir)

    #os.system(f"rm -rf {sdsdir}/* {dbout}.*")
    if not os.path.isdir(sdsdir):
        os.makedirs(sdsdir)
    laststarttime=0
    dayt = startt

    while (dayt <= endt):
        if not dayt in missing_days:
            dayt += secondsPerDay
            continue

        print(dayt, end="\n")
        ymd = dayt.strftime("%Y%m%d")
        #chuckfile = f"chuckmseedblocks{ymd}.msd"
        #dayepoch = int(start_dt_utc.timestamp())
        dayepoch = int(startt.timestamp)
        endepoch = dayepoch + secondsPerDay
        jday = dayt.strftime("%j")
        
        yyyy = dayt.strftime("%Y")
        mm = dayt.strftime("%m")
        dd = dayt.strftime("%d")
        currentseisandbdir = f"{seisandbdir}/{yyyy}/{mm}"
       
        pdayt = dayt - 86400
        pyyyy = pdayt.strftime("%Y")
        pmm = pdayt.strftime("%m")
        pdd = pdayt.strftime("%d")
        lastseisandbdir = f"{seisandbdir}/{pyyyy}/{pmm}"
    
        allfiles0 = glob.glob(os.path.join(lastseisandbdir, f'{pyyyy}-{pmm}-{pdd}-23[45]*S.MVO___*'))
        allfiles1 = glob.glob(os.path.join(currentseisandbdir, f'{yyyy}-{mm}-{dd}*S.MVO___*'))
        allfiles = sorted(allfiles0 + allfiles1)
        allfiles = sorted(list(set(allfiles)))
        
        print('- Found %d files' % len(allfiles))
        if len(allfiles)==0:
            print(glob.glob(os.path.join(currentseisandbdir, f'{yyyy[2:4]}*{mm}-{dd}*S.MVO___*')))
        firstfile = True
        print(allfiles)
        for file in allfiles:
            print(f'- Processing {file}')
            try:
                st = read(file, format='seisan')
                
            except:
                continue
   
            if Montserrat:
                thisstarttime = st[0].stats.starttime
                if thisstarttime - laststarttime == 1201.0: # should be 20 * 60 s = 1200 s
                    thisstarttime -= 1.0
                elif thisstarttime - laststarttime == 1199.0: # should be 20 * 60 s = 1200 s
                    thisstarttime += 1.0
                for tr in st:
                    tr.stats.starttime = thisstarttime
                    fix_trace_mvo(tr, legacy=False, netcode=net)

                if MBWHZ_only:
                    st = st.select(station='MBWH', component='Z')

            st.trim(starttime=dayt,endtime=dayt+86400)
                
            sdsobj.stream = st
            sdsobj.write(overwrite=firstfile) # just try this to see if it gets rid of merging non-contiguous data. just want to overwrite with first file each day, then append others
            firstfile = False
        if dbout:
            dboutday = f"{dbout}{ymd}"
            sds2db(dboutday, sdsdir, jday)
        dayt += secondsPerDay

def compute_raw_metrics(paths, startTime, endTime, sampling_interval=60, do_RSAM=True, net=None):
    # read SDS, write RSAM data
    #rawSDSclient = sdsclient(paths['SDS_DIR'])
    rawSDSclient = SDSobj(paths['SDS_DIR'], sds_type='D', format='MSEED')
    numDays = (endTime-startTime)/secondsPerDay
    daytime = startTime
    if do_RSAM:
        while daytime < endTime:
            # try to read RSAM data for this day
            rsam24h = RSAM.read(daytime, daytime+secondsPerDay, paths['SAM_DIR'], sampling_interval=sampling_interval, ext='pickle')
            (l, w) = rsam24h.__size__()
            if w>0:
                daytime += secondsPerDay
                continue

            print(f'Loading Stream data for {daytime}')
            rawSDSclient.read(daytime, daytime+secondsPerDay)
            st = rawSDSclient.stream
            #st = rawSDSclient.get_waveforms("MV", "*", "*", "[SBEHCD]*", daytime, daytime+secondsPerDay)
            print(f'- got {len(st)} Trace ids') 
            print(f'Computing RSAM metrics for {daytime}, and saving to pickle files')
            if net:
            	st = st.select(network=net)            	
            if isinstance(sampling_interval, list):
            	for delta in sampling_interval:
            	    rsam24h = RSAM(stream=st, sampling_interval=delta)
            	    rsam24h.write(paths['SAM_DIR'], ext='pickle')   
            else:
            	rsam24h = RSAM(stream=st, sampling_interval=sampling_interval)
            	rsam24h.write(paths['SAM_DIR'], ext='pickle')                	
            daytime += secondsPerDay
    del rawSDSclient


def compute_SDS_DISP(paths, startTime, endTime, invfile, kind='DISP'):
    # Not much point running this until a raw SDS archive has been completely built from events and continuous data
    net = os.path.basename(invfile)[0:2]
    print(f"compute_SDS_corrected: invfile = {invfile}")

    #rawSDSclient = sdsclient(paths['SDS_DIR'])
    rawSDSclient = SDSobj(paths['SDS_DIR'], sds_type='D', format='MSEED')
    outSDSclient = SDSobj(paths['SDS_%s_DIR' % kind], sds_type='D', format='MSEED')
    missing_days = outSDSclient.find_which_days_missing(startTime, endTime, net)
    inv = read_inventory(invfile, format='stationxml')  
    #print(inv) 
    inv_ids = inventory2traceid(inv, force_location_code='')
    #print(f"inv_ids = {inv_ids}")
    smalltime = 0.01
    daytime = startTime
    taperFraction = 0.05
    taperSecs = secondsPerDay * taperFraction
    while daytime < endTime:     
        if not daytime in missing_days:
            daytime += secondsPerDay
            continue
        print(f'Loading Stream data for {daytime}')
        #st = mySDSreadClient.get_waveforms("MV", "*", "*", "[SBEHCD]*", daytime-taperSecs, daytime+secondsPerDay+taperSecs)
        rawSDSclient.read(daytime - taperSecs, daytime+secondsPerDay+taperSecs, fixnet=net)
        st = rawSDSclient.stream
        st.merge(method=0, fill_value=0) # SCAFFOLD: different merge here?
        print(st)    
        #vel_st = obspy.core.Stream()
        print(f"seed IDs in {invfile} are {inv_ids}")
        for tr in st:
            print('Processing ',tr)
            non_loc_id = '.'.join((tr.stats.network, tr.stats.station, '', tr.stats.channel))
            if tr.id in inv_ids or non_loc_id in inv_ids:
                print(tr.id,' is valid')
                try: 
                    this_tr = tr.copy()
                    preprocess_trace(this_tr, bool_despike=True, bool_clean=True, inv=inv, quality_threshold=-np.Inf, taperFraction=taperFraction, \
                        filterType="bandpass", freq=[0.01, 30.0], corners=6, zerophase=False, outputType=kind, \
                        miniseed_qc=True, verbose=False, max_dropout=1.0, units='Counts', bool_detrend=True, min_sampling_rate=20.0)
                    this_st = Stream(traces=this_tr)
                    this_st.trim(starttime=daytime, endtime=daytime+secondsPerDay-smalltime)
                    outSDSclient.stream = this_st
                    print('Saving to SDS: ', outSDSclient.stream)
                    outSDSclient.write(overwrite=False, merge=False)
                    outSDSclient.stream = Stream() # wipe it for next time, just in case
                except Exception as e:
                    print(e)
        daytime += secondsPerDay 
    del rawSDSclient, outSDSclient


def check_what_to_do(paths, net, startt, endt, sampling_interval=60, ext='pickle', invfile=None, verbose=True):
    # check that derived data created by small or big sausage exist

    do_metric = {}

    # Check inventory exists in StationXML
    print('Checking for stationXML')
    if not invfile:
        invfile = os.path.join(paths['RESPONSE_DIR'],f"{net}.xml")
    do_metric['inventory'] = True
    if os.path.isfile(invfile):
        do_metric['inventory'] = False
        
    # Check raw SDS data exist
    print('Checking for raw SDS data')
    rawSDSclient = SDSobj(paths['SDS_DIR'], sds_type='D', format='MSEED') 
    missing_days_sds_raw = rawSDSclient.find_which_days_missing(startt, endt, net)
    do_metric['SDS_RAW']=True
    if len(missing_days_sds_raw)==0:
            do_metric['SDS_RAW']=False

def small_sausage(paths, startt, endt, sampling_interval=60, source=None, invfile=None, Q=None, ext='pickle', net=None, do_metric=None):
    if not do_metric:
        do_metric = check_what_to_do(paths, net, startt, endt, sampling_interval=sampling_interval, ext=ext, invfile=invfile)
    if 'RSAM' in do_metric and do_metric['RSAM']:
        compute_raw_metrics(paths, startt, endt, sampling_interval=sampling_interval, do_RSAM=True, net=net)

    print(f"invfile={invfile}")
    if invfile and os.path.isfile(invfile):

        if 'SDS_DISP' in do_metric and do_metric['SDS_DISP']:
            print('Calling compute_SDS_DISP')
            compute_SDS_DISP(paths, startt, endt, invfile)

       
        if 'VSAM' in do_metric and 'VSEM' in do_metric:
            print('Calling compute_velocity_metrics')
            compute_velocity_metrics(paths, startt, endt, sampling_interval=sampling_interval, do_VSAM=do_metric['VSAM'], \
                                 do_VSEM=do_metric['VSEM'], net=net, ext=ext) 

        if 'DSAM' in do_metric and do_metric['DSAM']:
            print('Calling compute_displacement_metrics')  
            compute_displacement_metrics(paths, startt, endt, sampling_interval=sampling_interval, do_DSAM=do_metric['DSAM'], net=net, ext=ext) 
    
        if source:   
            if not isinstance(sampling_interval, list):
                sampling_interval = [sampling_interval]
            for delta in sampling_interval:
                if 'ER' in do_metric and 'DR' in do_metric and 'DRS' in do_metric:
                    reduce_to_1km(paths, startt.year, do_ER=do_metric['ER'], do_DR=do_metric['DR'], do_DRS=do_metric['DRS'], sampling_interval=delta, invfile=invfile, source=source, Q=Q, ext=ext)

def check_SAM_metric_exists(samtype, stime, etime, SAM_DIR, sampling_interval, ext):
    evalstr = f"{samtype}.read(stime, etime, SAM_DIR, sampling_interval=sampling_interval, ext=ext)"
    sam24h = eval(evalstr)
    (l, w) = sam24h.__size__()
    print(samtype, stime, l, w)
    return w


def compute_velocity_metrics(paths, startt, endt, sampling_interval=60, do_VSAM=True, do_VSEM=True, net=None, ext='pickle'):      
    # read SDS_VEL, write VSAM, VSEM data
    #velSDSclient = sdsclient(paths['SDS_DIR'])
    velSDSclient = SDSobj(paths['SDS_VEL_DIR'], sds_type='D', format='MSEED')
    numDays = (endt-startt)/secondsPerDay
    daytime = startt
    if do_VSAM or do_VSEM:
        while daytime < endt:
            
            if do_VSAM:
                w = check_SAM_metric_exists('VSAM', daytime, daytime+secondsPerDay, paths['SAM_DIR'], sampling_interval, ext)
            if do_VSEM:
                w = check_SAM_metric_exists('VSEM', daytime, daytime+secondsPerDay, paths['SAM_DIR'], sampling_interval, ext)  
            if w>0:
                daytime += secondsPerDay
                continue
            
            print(f'Loading Stream data for {daytime}')
            velSDSclient.read(daytime, daytime+secondsPerDay, fixnet=net)
            st = velSDSclient.stream
            for tr in st:
                if tr.stats.channel[1]=='H':
                    tr.stats['units'] = 'm/s'
            if net:
            	st = st.select(network=net)
            if len(st)>0:
                if do_VSAM:
            	    print(f'Computing VSAM metrics for {daytime}, and saving to pickle files')
            	    if isinstance(sampling_interval, list):
            	        for delta in sampling_interval:
            	            vsam24h = VSAM(stream=st, sampling_interval=delta)
            	            vsam24h.write(paths['SAM_DIR'], ext='pickle')   
            	    else:
            	        vsam24h = VSAM(stream=st, sampling_interval=sampling_interval)
            	        vsam24h.write(paths['SAM_DIR'], ext='pickle')       
                if do_VSEM:
            	    print(f'Computing VSEM metrics for {daytime}, and saving to pickle files') 
            	    if isinstance(sampling_interval, list):
            	        for delta in sampling_interval:
            	            vsem24h = VSEM(stream=st, sampling_interval=delta)
            	            vsem24h.write(paths['SAM_DIR'], ext='pickle')   
            	    else:
            	        vsem24h = VSAM(stream=st, sampling_interval=sampling_interval)
            	        vsem24h.write(paths['SAM_DIR'], ext='pickle')
            daytime += secondsPerDay
    del velSDSclient
    
def compute_displacement_metrics(paths, startt, endt, sampling_interval=60, do_DSAM=True, net=None, ext='pickle'):       
    # read SDS_DISP, write DSAM
    #dispSDSclient = sdsclient(paths['SDS_DIR'])
    dispSDSclient = SDSobj(paths['SDS_DISP_DIR'], sds_type='D', format='MSEED')
    numDays = (endt-startt)/secondsPerDay
    daytime = startt
    if do_DSAM:
        while daytime < endt:
            
            w = check_SAM_metric_exists('DSAM', daytime, daytime+secondsPerDay, paths['SAM_DIR'], sampling_interval, ext) 
            if w>0:
                daytime += secondsPerDay
                continue            
            
            print(f'Loading Stream data for {daytime}')
            dispSDSclient.read(daytime, daytime+secondsPerDay, fixnet=net)
            st = dispSDSclient.stream
            for tr in st:
                if tr.stats.channel[1]=='H':
                    tr.stats['units'] = 'm'
            if net:
            	st = st.select(network=net)
            if len(st)>0 and do_DSAM:            
            	print(f'Computing DSAM metrics for {daytime}, and saving to pickle files')
            	if isinstance(sampling_interval, list):
            	    for delta in sampling_interval:
            	        dsam24h = DSAM(stream=st, sampling_interval=delta)
            	        dsam24h.write(paths['SAM_DIR'], ext='pickle')   
            	else:
            	    dsam24h = DSAM(stream=st, sampling_interval=sampling_interval)
            	    dsam24h.write(paths['SAM_DIR'], ext='pickle')               	            	
            daytime += secondsPerDay
    del dispSDSclient  


def reduce_to_1km(paths, year, do_VR=False, do_VRS=False, do_ER=True, do_DR=True, do_DRS=True, sampling_interval=60, invfile=None, source=None, Q=None, ext='pickle'):
    startTime = UTCDateTime(year,1,1)
    endTime = UTCDateTime(year,12,31,23,59,59.9)
    inv = None
    if invfile:
        if os.path.isfile(invfile):
            inv = read_inventory(invfile)
    if not inv:
        return
    if not source:
        return
    

    if do_VR or do_VRS:
        vsamObj = VSAM.read(startTime, endTime, SAM_DIR=paths['SAM_DIR'], sampling_interval=sampling_interval, ext=ext)
        if do_VR:
            VRobj = vsamObj.compute_reduced_velocity(inv, source, surfaceWaves=False, Q=Q)
            VRobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)
        if do_VRS:
            VRSobj = vsamObj.compute_reduced_velocity(inv, source, surfaceWaves=True, Q=Q)
            VRSobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)        

    if do_ER:
        vsemObj = VSEM.read(startTime, endTime, SAM_DIR=paths['SAM_DIR'], sampling_interval=sampling_interval, ext=ext)
        if do_ER:
            ERobj = vsemObj.compute_reduced_energy(inv, source, Q=Q)
            ERobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)
            
    if do_DR or do_DRS:
        dsamObj = DSAM.read(startTime, endTime, SAM_DIR=paths['SAM_DIR'], sampling_interval=sampling_interval, ext=ext)
        if do_DR:
            DRobj = dsamObj.compute_reduced_displacement(inv, source, surfaceWaves=False, Q=None)
            DRobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True)
        if do_DRS:
            DRSobj = dsamObj.compute_reduced_displacement(inv, source, surfaceWaves=True, Q=None)
            DRSobj.write(SAM_DIR=paths['SAM_DIR'], overwrite=True) 

def big_sausage(seisandbdir, paths, startt, endt, sampling_interval=60, source=None, invfile=None, Q=None, \
                ext='pickle', dbout=None, round_sampling_rate=True, net=None, do_metric=None, MBWHZ_only=False):
    # includes everything in small sausage, but with a Seisan to SDS conversion first. For Montserrat only

    if not do_metric:
        do_metric = check_what_to_do(paths, net, startt, endt, sampling_interval=sampling_interval, ext=ext, invfile=invfile)

    print(do_metric)

    if 'SDS_RAW' in do_metric and do_metric['SDS_RAW']:
        seisandb2SDS(seisandbdir, paths['SDS_DIR'], startt, endt, net, dbout=dbout, round_sampling_rate=round_sampling_rate, MBWHZ_only=MBWHZ_only)
    
    # small sauage stuff
    small_sausage(paths, startt, endt, sampling_interval=sampling_interval, source=source, invfile=invfile, Q=Q,\
                ext=ext, net=net, do_metric=do_metric)




if __name__ == "__main__":
    import setup_paths
    paths = setup_paths.paths
    #sys.path.append('../../src/lib')
    seisandbdir =  '/data/SEISAN_DB/WAV/DSNC_'
    net = 'MV'
    invfile = os.path.join(paths['RESPONSE_DIR'],f"{net}.xml")
    startt = UTCDateTime(2001, 7, 28, 0, 0, 0)
    endt = UTCDateTime(2001, 7, 31, 0, 0, 0)
    #sampling_interval = [2.56, 60, 600]
    sampling_interval= 10 # seconds # can also be a list of different sampling rates, e.g. [2.56, 60, 600] to mimic original RSAM system. 
    # For VT band (4-18 Hz), 2.56s is fine (10 cycles+)
    # For LP band (0.5 - 4 Hz), 2.56s might work (1.25 cycles+), but 10s might be better (5 cycles+), and 60s would be amazing (30 cycles+)
    # For VLP band (0.02 - ?), 60-s might work (1.2 cycles+), but 600s would be better (12 cycles+)
    # So best compromise might be [2.56, 10, 60, 600] then we have everything we need for RSAM bar graph simulator, detecting VTs (2.56s), LPs (2.56s or 10s), and VLPs (60s or 600s), and 10s for ASL.
    source = dome_location
    dbout = os.path.join(paths['DB_DIR'],f"dbMontserrat{startt.year}")
    Q = None
    ext = 'pickle'

    do_metric = check_what_to_do(paths, net, startt, endt, sampling_interval=sampling_interval, ext=ext)

    big_sausage(seisandbdir, paths, startt, endt, sampling_interval=sampling_interval, source=source, \
                invfile=invfile, Q=Q, ext=ext, dbout=dbout, round_sampling_rate=True, net=net, do_metric=do_metric)
    # looks like MBLY supposed to be 100 Hz, so by forcing it to 75 Hz, I am messing it up. So replacing 75.0 with np.round(tr.stats.sampling_rate, 0) should fix this.


    
