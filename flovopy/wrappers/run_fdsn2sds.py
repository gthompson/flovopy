from flovopy.sds.sds import SDSobj
from flovopy.stationmetadata.utils import inventory2traceid
from flovopy.core.fdsn import get_inventory, get_stream

def FDSN_to_SDS_daily_wrapper(startt, endt, SDS_TOP, centerlat=None, centerlon=None, searchRadiusDeg=None, trace_ids=None, \
        fdsnURL="http://service.iris.edu", overwrite=True, inv=None):
    '''
    Download Stream from FDSN server and save to SDS format. Default is to overwrite each time.
    
    NSLC combinations to download either come from (1) trace_ids name-value pair, (2) inv name-value pair, (3) circular search parameters, in that order.

        Parameters:
            startt (UTCDateTime): An ObsPy UTCDateTime marking the start date/time of the data request.
            endt (UTCDateTime)  : An ObsPy UTCDateTime marking the end date/time of the data request.

            SDS_TOP (str)       : The path to the SDS directory structure.

        Optional Name-Value Parameters:
            trace_ids (List)    : A list of N.S.L.C strings. Default None. If given, this overrides other options.
            inv (Inventory)     : An ObsPy Inventory object. Default None. If given, trace_ids will be extracted from it, unless explicity given.
            centerlat (float)   : Decimal degrees latitude for circular station search. Default None.
            centerlon (float)   : Decimal degrees longitude for circular station search. Default None.
            searchRadiusDeg (float) : Decimal degrees radius for circular station search. Default None.
            fdsnURL (str) : URL corresponding to FDSN server. Default is "http://service.iris.edu".
            overwrite (bool) : If True, overwrite existing data in SDS archive.

        Returns: None. Instead an SDS volume is created/expanded.

    '''

    secsPerDay = 86400  
    while startt<endt:
        print(startt)
        endOfRsamTimeWindow = startt+secsPerDay 
        # read from SDS - if no data download from FDSN

        thisSDSobj = SDSobj(SDS_TOP) 
        
        if thisSDSobj.read(startt, endOfRsamTimeWindow, speed=2) or overwrite: # non-zero return value means no data in SDS so we will use FDSN
            # read from FDSN
            if not trace_ids:
                if inv: 
                    trace_ids = inventory2traceid(inv)
                else:
                    inv = get_inventory(fdsnURL, startt, endOfRsamTimeWindow, centerlat, centerlon, \
                                                        searchRadiusDeg, overwrite=overwrite ) # could add N S L C requirements too
                    if inv:
                        trace_ids = inventory2traceid(inv)
            if trace_ids:
                st = get_stream(fdsnURL, trace_ids, startt, endOfRsamTimeWindow, overwrite=overwrite)
                thisSDSobj.stream = st
                thisSDSobj.write(overwrite=overwrite) # save raw data to SDS
            else:
                print('SDS archive not written to.')

    

        startt+=secsPerDay # add 1 day 
        
def main():
    import argparse
    from obspy import UTCDateTime

    parser = argparse.ArgumentParser(description="Download FDSN data into SDS structure.")
    parser.add_argument("--network", required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--outdir", default="SDS")
    args = parser.parse_args()

    FDSN_to_SDS_daily_wrapper(
        startt=UTCDateTime(args.start),
        endt=UTCDateTime(args.end),
        outdir=args.outdir,
        network=args.network,
        station=args.station
    )
