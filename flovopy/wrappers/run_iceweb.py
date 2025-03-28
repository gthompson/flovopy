import os
import gc
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
import sqlite3
from obspy import UTCDateTime, Stream
from flovopy.core.sam import RSAM, DRS
from flovopy.core.spectrograms import icewebSpectrogram
from flovopy.sds.sds import SDSobj
from flovopy.core.inventory import attach_station_coordinates_from_inventory, attach_distance_to_stream

logger = logging.getLogger(__name__)


class IceWebDatabase:
    def __init__(self, dbpath: str):
        self.dbpath = dbpath
        self.conn = sqlite3.connect(self.dbpath)
        self._create_table()

    def _create_table(self):
        sql = '''CREATE TABLE IF NOT EXISTS products (
            subnet TEXT NOT NULL,
            startTime TEXT NOT NULL,
            endTime TEXT NOT NULL,
            datasource TEXT,
            rsamDone INTEGER DEFAULT 0,
            drsDone INTEGER DEFAULT 0,
            sgramDone INTEGER DEFAULT 0,
            specParamsDone INTEGER DEFAULT 0,
            locked INTEGER DEFAULT 0,
            PRIMARY KEY (subnet, startTime, endTime)
        );'''
        self.conn.execute(sql)
        self.conn.commit()

    def select_row(self, subnet, start, end):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM products WHERE subnet=? AND startTime=? AND endTime=?",
                    (subnet, start, end))
        return cur.fetchone()

    def insert_row(self, subnet, start, end):
        try:
            self.conn.execute("INSERT INTO products(subnet, startTime, endTime) VALUES (?, ?, ?)",
                              (subnet, start, end))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(e)
            return False

    def update_row(self, subnet, start, end, field, value):
        try:
            self.conn.execute(f"UPDATE products SET {field}=? WHERE subnet=? AND startTime=? AND endTime=?",
                              (value, subnet, start, end))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(e)
            return False

    def lock_row(self, subnet, start, end, create=False):
        row = self.select_row(subnet, start, end)
        if row and row[-1]:
            logger.warning(f"{subnet} {start} {end} already locked")
            return False
        elif row:
            return self.update_row(subnet, start, end, 'locked', 1)
        elif create:
            inserted = self.insert_row(subnet, start, end)
            return inserted and self.update_row(subnet, start, end, 'locked', 1)
        return False

    def unlock_row(self, subnet, start, end):
        return self.update_row(subnet, start, end, 'locked', 0)

    def close(self):
        self.conn.close()


def StreamToIcewebProducts(st, seismogramType, db: IceWebDatabase, subnet, startStr, endStr,
                            verbose=False, rsamSamplingIntervalSeconds=60,
                            RSAM_TOP='.', SGRAM_TOP='.', dbscale=True,
                            equal_scale=True, clim=[1e-8, 1e-5], fmin=0.5, fmax=18.0,
                            overwrite=False):
    print(f"\n[{UTCDateTime.now().isoformat()}] Processing {seismogramType} products for {subnet} from {startStr} to {endStr}")

    row = db.select_row(subnet, startStr, endStr)
    if not row:
        raise Exception("No matching row found in the products table.")

    _, _, _, _, rsamDone, drsDone, sgramDone, _, _ = row

    if seismogramType == 'VEL' and not rsamDone:
        st_abs = Stream(tr for tr in st if not tr.stats.get('config', {}).get('keepRaw', False))
        st_noabs = Stream(tr for tr in st if tr.stats.get('config', {}).get('keepRaw', False))

        if st_abs:
            rsam = RSAM(st_abs, sampling_interval=rsamSamplingIntervalSeconds, absolute=True, filter=True)
            rsam.compute()
            rsam.write(RSAM_TOP)

        if st_noabs:
            rsam = RSAM(st_noabs, sampling_interval=rsamSamplingIntervalSeconds, absolute=False, filter=False)
            rsam.compute()
            rsam.write(RSAM_TOP)

        db.update_row(subnet, startStr, endStr, field='rsamDone', value=True)

    if seismogramType == 'VEL' and not sgramDone:
        st_sgram = Stream(tr for tr in st if tr.stats.get('config', {}).get('sgram', False))
        if st_sgram:
            sgramdir = os.path.join(SGRAM_TOP, st_sgram[0].stats.network, startStr[:4], startStr[5:8])
            os.makedirs(sgramdir, exist_ok=True)
            sgramfile = os.path.join(sgramdir, f"{subnet}_{startStr[:10]}-{startStr[11:13]}{startStr[14:16]}.png")
            if not os.path.exists(sgramfile) or overwrite:
                sp = icewebSpectrogram(st_sgram)
                sp.plot(outfile=sgramfile, dbscale=dbscale, title=sgramfile,
                        equal_scale=equal_scale, clim=clim, fmin=fmin, fmax=fmax)
                db.update_row(subnet, startStr, endStr, field='sgramDone', value=True)
            plt.close('all')

    elif seismogramType == 'DISP' and not drsDone:
        drs = DRS(st, sampling_interval=rsamSamplingIntervalSeconds, absolute=True, filter=True)
        drs.compute()
        drs.write(RSAM_TOP)
        db.update_row(subnet, startStr, endStr, field='drsDone', value=True)

    gc.collect()


def process_timewindows(startt, endt, dsobj, db: IceWebDatabase, freqmin=0.5, freqmax=None,
                        zerophase=False, corners=2, sampling_interval=60.0,
                        sourcelat=None, sourcelon=None, inv=None, trace_ids=None,
                        overwrite=True, verbose=False, timeWindowMinutes=10,
                        timeWindowOverlapMinutes=5, subnet='unknown',
                        SGRAM_TOP='.', RSAM_TOP='.'):
    taperSecs = timeWindowOverlapMinutes * 60
    current = startt

    while current < endt:
        next_time = current + timeWindowMinutes * 60
        startStr, endStr = current.isoformat(), next_time.isoformat()

        if db.select_row(subnet, startStr, endStr):
            current = next_time
            continue

        print(f"\n[{UTCDateTime.now().isoformat()}] Processing {startStr} to {endStr}")

        try:
            st = dsobj.get_waveforms(current - taperSecs, next_time + taperSecs,
                                     trace_ids=trace_ids, inv=inv)
        except Exception as e:
            print(f"Waveform retrieval failed: {e}")
            current = next_time
            continue

        if isinstance(inv, obspy.Inventory):
            attach_station_coordinates_from_inventory(inv, st)
            attach_distance_to_stream(st, sourcelat, sourcelon)
            r = [tr.stats.distance for tr in st]
            st = order_traces_by_distance(st, r, assert_channel_order=True)
            pre_filt = [freqmin/1.2, freqmin, freqmax, freqmax*1.2]

            for seis_type in ['VEL', 'DISP']:
                corrected = st.copy().select(channel="*H*").remove_response(
                    output=seis_type, inventory=inv, pre_filt=pre_filt, water_level=60)
                corrected.trim(starttime=current, endtime=next_time)
                if db.lock_row(subnet, startStr, endStr, create=True):
                    StreamToIcewebProducts(corrected, seis_type, db, subnet, startStr, endStr,
                                           SGRAM_TOP=SGRAM_TOP, RSAM_TOP=RSAM_TOP, overwrite=overwrite)
                    db.unlock_row(subnet, startStr, endStr)

        elif isinstance(inv, pd.DataFrame):
            for i, row in inv.iterrows():
                for tr in st:
                    if row['trace_id'] == tr.id:
                        tr.data = tr.data / row['calib']
                        tr.stats['units'] = 'm/s' if tr.stats.channel[1] == 'H' else 'Pa'
                        tr.stats['config'] = {
                            'maxPower': row['maxPower'],
                            'keepRaw': row['keepRaw'],
                            'sgram': row['sgram'],
                            'calib': row['calib']
                        }

            if db.lock_row(subnet, startStr, endStr, create=True):
                StreamToIcewebProducts(st, 'VEL', db, subnet, startStr, endStr,
                                       SGRAM_TOP=SGRAM_TOP, RSAM_TOP=RSAM_TOP, overwrite=overwrite)
                db.unlock_row(subnet, startStr, endStr)

        current = next_time
        gc.collect()

def order_traces_by_distance(st, r=None, assert_channel_order=False):
    if r is None:
        r = [tr.stats.distance for tr in st]
    if assert_channel_order:
        numbers = 'ZNEF0123456789'
        for i, tr in enumerate(st):
            loc = int(tr.stats.location or 0)/1e6
            chan = numbers.find(tr.stats.channel[2])/1e9
            r[i] += loc + chan
    return Stream([st[i].copy() for i in np.argsort(r)])

def read_config(configdir='config', leader='iceweb', PRODUCTS_TOP=None):
    """
    Load IceWeb configuration files and substitute variables in general config.

    Parameters
    ----------
    configdir : str
        Directory containing the CSV config files.
    leader : str
        Prefix for the config file names (e.g., 'iceweb_general.config.csv').
    PRODUCTS_TOP : str or None
        Optional override to set the top-level output directory.

    Returns
    -------
    config : dict
        Dictionary with keys 'general', 'jobs', 'traceids', 'places'.
        config['general'] is a flat dictionary of resolved variable paths.
    """
    config = {}
    for section in ['general', 'jobs', 'traceids', 'places']:
        fname = os.path.join(configdir, f"{leader}_{section}.config.csv")
        config[section] = pd.read_csv(fname)

    # Allow runtime override of PRODUCTS_TOP
    if PRODUCTS_TOP:
        config['general'].loc[config['general']['Variable'] == 'PRODUCTS_TOP', 'Value'] = PRODUCTS_TOP

    # Flatten general config into a dict of Variable → Value
    resolved = {row['Variable']: row['Value'] for _, row in config['general'].iterrows()}

    # Handle simple $VAR substitutions
    for key, val in resolved.items():
        if '$' in val:
            parts = val.split('/')
            base_key = parts[0][1:]
            resolved[key] = resolved.get(base_key, '') + '/' + '/'.join(parts[1:]) if len(parts) > 1 else resolved.get(base_key, '')

    config['general'] = resolved
    return config

class IceWebJob:
    def __init__(self, subnet, config):
        self.subnet = subnet
        self.config = config
        self.db = IceWebDatabase(self.config['DBPATH'])
        self.ds = Datasource(config['general']['DATASOURCE'],
                             url=config['general'].get('FDSN_URL'),
                             SDS_TOP=config['general'].get('SDS_TOP'))

    def run(self, startt, endt, inv, trace_ids):
        twmin = int(self.config['general']['timeWindowMinutes'])
        twov = int(self.config['general']['timeWindowOverlapMinutes'])
        rsam_interval = float(self.config['general']['samplingInterval'])
        freqmin = float(self.config['general']['freqmin'])
        freqmax = float(self.config['general']['freqmax'])

        current = startt
        while current < endt:
            nextt = current + twmin * 60
            sstr, estr = current.isoformat(), nextt.isoformat()

            if self.db.select_row(self.subnet, sstr, estr):
                current = nextt
                continue

            try:
                st = self.ds.get_waveforms(current - twov * 60, nextt + twov * 60,
                                           trace_ids=trace_ids, inv=inv)
            except Exception as e:
                logger.error(f"Waveform retrieval failed: {e}")
                current = nextt
                continue

            pre_filt = [freqmin / 1.2, freqmin, freqmax, freqmax * 1.2]
            for seismogram_type in ['VEL', 'DISP']:
                try:
                    cst = st.copy().select(channel="*H*").remove_response(
                        output=seismogram_type, inventory=inv, pre_filt=pre_filt, water_level=60)
                    cst.trim(current, nextt)
                    if self.db.lock_row(self.subnet, sstr, estr, create=True):
                        self.process_stream(cst, seismogram_type, sstr, estr, inv)
                        self.db.unlock_row(self.subnet, sstr, estr)

                except Exception as e:
                    logger.error(f"Error processing {seismogram_type}: {e}")
            gc.collect()
            current = nextt

        self.db.close()
        self.ds.close()
        self.ds = None


def process_stream(self, st, seismogram_type, sstr, estr, inv):
    rsam_interval = float(self.config['general']['samplingInterval'])
    try:
        StreamToIcewebProducts(
            st, seismogram_type, self.db.conn, self.subnet,
            sstr, estr,
            RSAM_TOP=self.config['general']['RSAM_TOP'],
            SGRAM_TOP=self.config['general']['SGRAM_TOP'],
            rsamSamplingIntervalSeconds=rsam_interval,
            overwrite=True
        )
        self.db.update_row(self.subnet, sstr, estr, f"{seismogram_type.lower()}Done", 1)
    except Exception as e:
        logger.error(f"process_stream failed for {seismogram_type}: {e}")


class Datasource:
    def __init__(self, dstype, url=None, SDS_TOP=None):
        """
        Generic interface to SDS or FDSN waveform and metadata sources.
        """
        self.dstype = dstype.lower()
        self.url = url
        self.SDS_TOP = SDS_TOP
        self.connector = None

        if self.dstype == 'sds':
            self.connector = SDSobj(SDS_TOP, sds_type='D', format='MSEED')
        elif self.dstype == 'fdsn':
            self.connector = obspy.clients.fdsn.Client(base_url=url)
        else:
            raise ValueError(f"Unsupported datasource type: {dstype}")

    def get_waveforms(self, startt, endt, trace_ids=None, speed=2, verbose=False, inv=None):
        st = Stream()
        if self.dstype == 'sds':
            self.connector.read(startt, endt, trace_ids=trace_ids, speed=speed, verbose=verbose)
            st = self.connector.stream
            if inv:
                st.attach_response(inv)

        elif self.dstype == 'fdsn':
            for trace_id in (trace_ids or []):
                net, sta, loc, chan = trace_id.split('.')
                try:
                    this_st = self.connector.get_waveforms(
                        net, sta, loc, chan,
                        starttime=startt,
                        endtime=endt,
                        attach_response=True
                    )
                    this_st.merge(method=1, fill_value=0)
                    st += this_st
                except Exception as e:
                    print(f"FDSN waveform error for {trace_id}: {e}")

        return st

    def get_inventory(self, startt, endt, centerlat, centerlon, searchRadiusDeg,
                      network='*', station='*', channel='*'):
        """
        Returns an ObsPy Inventory object, and optionally saves it to SDS metadata dir.
        """
        inv = None
        if self.dstype == 'sds':
            filename = os.path.join(self.SDS_TOP, 'metadata',
                                    f"{centerlat}_{centerlon}_{startt.strftime('%Y%m%d')}_{endt.strftime('%Y%m%d')}_{searchRadiusDeg}.sml")
            if os.path.isfile(filename):
                inv = obspy.read_inventory(filename)
            else:
                print(f"[WARNING] Inventory file not found: {filename}")

        elif self.dstype == 'fdsn':
            try:
                inv = self.connector.get_stations(
                    network=network,
                    station=station,
                    channel=channel,
                    latitude=centerlat,
                    longitude=centerlon,
                    maxradius=searchRadiusDeg,
                    starttime=startt,
                    endtime=endt,
                    level='response'
                )
                if self.SDS_TOP:
                    meta_dir = os.path.join(self.SDS_TOP, 'metadata')
                    os.makedirs(meta_dir, exist_ok=True)
                    outfile = os.path.join(meta_dir,
                                           f"{centerlat}_{centerlon}_{startt.strftime('%Y%m%d')}_{endt.strftime('%Y%m%d')}_{searchRadiusDeg}.sml")
                    inv.write(outfile, format="STATIONXML")
            except Exception as e:
                print(f"[ERROR] Failed to fetch FDSN inventory: {e}")

        return inv

    def close(self):
        if self.dstype == 'fdsn' and hasattr(self.connector, 'close'):
            self.connector.close()
        self.connector = None
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Run IceWeb processing pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config directory")
    parser.add_argument("--subnet", type=str, required=True, help="Subnet label")
    parser.add_argument("--start", type=str, required=True, help="Start time in UTC (e.g., 2023-01-01T00:00:00)")
    parser.add_argument("--end", type=str, required=True, help="End time in UTC (e.g., 2023-01-02T00:00:00)")
    parser.add_argument("--trace_ids", type=str, nargs='+', help="List of trace IDs (e.g., XX.STA..BHZ)")
    parser.add_argument("--inventory", type=str, help="Path to StationXML inventory file")
    args = parser.parse_args()

    from flovopy.wrappers.run_iceweb import read_config

    config = read_config(configdir=args.config)

    job = IceWebJob(subnet=args.subnet, config=config)

    startt = UTCDateTime(args.start)
    endt = UTCDateTime(args.end)

    inv = None
    if args.inventory:
        inv = obspy.read_inventory(args.inventory)

    trace_ids = args.trace_ids if args.trace_ids else None

    job.run(startt, endt, inv, trace_ids)

if __name__ == "__main__":
    main()
"""
Try:

python -m flovopy.wrappers.run_iceweb \
  --config config/ \
  --subnet SHV \
  --start 2023-01-01T00:00:00 \
  --end 2023-01-02T00:00:00 \
  --trace_ids XX.STA..BHZ XX.STB..BHZ \
  --inventory my_station_metadata.xml
"""