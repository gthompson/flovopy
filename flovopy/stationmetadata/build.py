import os
import pandas as pd
from obspy import UTCDateTime
from obspy.core.inventory.response import Response, InstrumentSensitivity
from obspy.clients.nrl import NRL
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from flovopy.stationmetadata.utils import (
    responses2inventory,
    expand_channel_code,
    build_dataframe_from_table
)
from flovopy.stationmetadata.raspberryshake import (
    _rs_kind,
    build_inv_from_template
)
from flovopy.stationmetadata.infrabsu import (
    get_infrabsu_sensor_template,
    get_infrabsu_centaur,
)
from collections import defaultdict


def NRL2inventory(
    nrl_path, net, sta, loc, chans,
    datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0,
    lat=0.0, lon=0.0, elev=0.0, depth=0.0, sitename='',
    ondate=UTCDateTime(1970, 1, 1), offdate=UTCDateTime(2025, 12, 31),
    sensitivity=None, units=None
) -> Inventory:
    """
    Constructs an ObsPy Inventory using datalogger/sensor response from the IRIS NRL or local NRL path.

    If sensor/datalogger keys are not found, provides fallback PAZ responses for some supported sensors.

    Parameters:
    -----------
    nrl_path : str
        Path to local NRL directory or None to use remote IRIS NRL.
    net : str
        Network code (e.g. 'XX').
    sta : str
        Station code (e.g. 'ABCD').
    loc : str
        Location code (e.g. '00').
    chans : list of str
        List of channel codes (e.g. ['HDF', 'HDF']).
    datalogger : str
        Datalogger type ('Centaur', 'RT130', etc.).
    sensor : str
        Sensor model name or keyword (e.g. 'TCP', 'L-22', 'infrabsu').
    Vpp : float
        Peak-to-peak voltage (1 or 40 for Centaur).
    fsamp : float
        Sampling rate in Hz.
    lat, lon, elev, depth : float
        Station coordinates.
    sitename : str
        Optional site name (currently unused).
    ondate, offdate : UTCDateTime
        Start and end date for metadata validity.
    sensitivity : float, optional
        Fallback sensitivity value for custom sensor.
    units : str, optional
        Input units for fallback response (e.g. 'm/s', 'Pa').

    Returns:
    --------
    inventory : obspy.core.inventory.Inventory
        Inventory containing one station with the specified response.
    """
    print(f"[INFO] NRL2inventory: Building inventory for {net}.{sta}.{loc} with channels: {chans}")
    if nrl_path and os.path.isdir(nrl_path):
        try:
            nrl = NRL(nrl_path)
        except Exception as e:
            print(f"[ERROR] Failed to load local NRL from {nrl_path}: {e}")
            nrl = NRL("http://ds.iris.edu/NRL/")
    else:
        nrl = NRL("http://ds.iris.edu/NRL/")

    # Datalogger NRL keys
    if datalogger == 'Centaur':
        if Vpp == 40:
            datalogger_keys = ['Nanometrics', 'Centaur', '40 Vpp (1)', 'Off', 'Linear phase', f"{int(fsamp)}"]
        elif Vpp == 1:
            datalogger_keys = ['Nanometrics', 'Centaur', '1 Vpp (40)', 'Off', 'Linear phase', f"{int(fsamp)}"]
        else:
            raise ValueError(f"Unsupported Vpp: {Vpp}")
    elif datalogger == 'RT130':
        datalogger_keys = ['REF TEK', 'RT 130 & 130-SMA', '1', f"{int(fsamp)}"]
    else:
        raise ValueError(f"Unsupported datalogger: {datalogger}")

    # Sensor NRL keys
    if sensor == 'TCP':
        sensor_keys = ['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '754 V/m/s']
    elif sensor == 'L-22':
        sensor_keys = ['Sercel/Mark Products', 'L-22D', '2200 Ohms', '10854 Ohms']
    elif sensor.lower().startswith('chap'):
        sensor_keys = ['Chaparral Physics', '25', 'Low: 0.4 V/Pa']
    elif sensor.lower() == 'infrabsu':
        sensor_keys = ['JeffreyBJohnson', 'infraBSU', '0.000046 V/Pa']
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")

    try:
        thisresponse = nrl.get_response(sensor_keys=sensor_keys, datalogger_keys=datalogger_keys)
    except Exception as e:
        print(f"[WARNING] NRL lookup failed for {net}.{sta}.{loc} ({sensor}): {e}")

        # Fallback PAZ response for selected sensors
        if sensor.lower() == 'infrabsu':
            poles = [-0.301593 + 0j]
            zeros = [0j]
            sensitivity = sensitivity or 18400.0
        elif sensor.lower().startswith('chap'):
            poles = [-1.0 + 0j, -3.03 + 0j, -3.03 + 0j]
            zeros = [0j, 0j, 0j]
            sensitivity = sensitivity or 160000.0
        else:
            raise ValueError(f"No fallback response available for sensor: {sensor}")

        input_units = units or 'm/s'
        output_units = 'counts'

        try:
            thisresponse = Response.from_paz(
                zeros=zeros, poles=poles, stage_gain=sensitivity,
                input_units=input_units, output_units=output_units
            )            
        except Exception as err:
            print(f"[ERROR] Failed to build fallback PAZ response: {err}")
            thisresponse = Response(instrument_sensitivity=InstrumentSensitivity(
                value=sensitivity, frequency=1.0,
                input_units=input_units, output_units=output_units
            ))

    responses = {chan: thisresponse for chan in chans}
    inventory = responses2inventory(
        net, sta, loc, fsamp, responses,
        lat=lat, lon=lon, elev=elev, depth=depth,
        start_date=ondate, end_date=offdate
    )
    return inventory



# ---- Global StationXML unit normalizer -------------------------------------
try:
    # ObsPy ≥1.5: Units object
    from obspy.core.inventory.util import Units as _ObsPyUnits
    def _mk_units(name: str):
        return _ObsPyUnits(name=name)
except Exception:
    # ObsPy 1.4: plain strings are fine
    def _mk_units(name: str):
        return name

# Map any legacy/odd spellings to controlled tokens
_UNIT_MAP = {
    # counts
    "count": "count",
    "counts": "count",
    "COUNT": "count",
    "COUNTS": "count",
    "Counts": "count",

    # pascal
    "pa": "Pa",
    "PA": "Pa",
    "Pa": "Pa",

    # velocity
    "m/s": "m/s",
    "M/S": "m/s",

    # (optional, commonly seen)
    "v": "V",
    "V": "V",
}

def _norm_unit_token(x) -> str | None:
    """
    Accepts an ObsPy Units object, a string, or None. Returns normalized token or None.
    """
    if x is None:
        return None
    # ObsPy Units or plain string
    name = getattr(x, "name", x if isinstance(x, str) else None)
    if not name:
        return None
    key = str(name).strip()
    return _UNIT_MAP.get(key, key)  # keep unknowns unchanged

def _set_units_attr(obj, attr: str, token: str | None):
    if token is None:
        return
    try:
        setattr(obj, attr, _mk_units(token))
    except Exception:
        setattr(obj, attr, token)  # fallback to plain string, ObsPy 1.4-safe

def sanitize_inventory_units(inv) -> None:
    """
    Replace units across the whole Inventory:
      - 'COUNTS'/'Counts' -> 'count'
      - 'M/S' -> 'm/s'
      - 'PA' -> 'Pa'
    Applies to per-stage input/output units and channel-level InstrumentSensitivity.
    """
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                resp = getattr(cha, "response", None)
                if resp is None:
                    continue

                # Stage-level input/output units
                for stg in getattr(resp, "response_stages", []) or []:
                    in_tok = _norm_unit_token(getattr(stg, "input_units", None))
                    out_tok = _norm_unit_token(getattr(stg, "output_units", None))
                    _set_units_attr(stg, "input_units", in_tok)
                    _set_units_attr(stg, "output_units", out_tok)

                # Channel-level InstrumentSensitivity units
                ins = getattr(resp, "instrument_sensitivity", None)
                if ins is not None:
                    inu = _norm_unit_token(getattr(ins, "input_units", None))
                    onu = _norm_unit_token(getattr(ins, "output_units", None))
                    _set_units_attr(ins, "input_units", inu)
                    _set_units_attr(ins, "output_units", onu)


def sensor_type_dispatch(
    row, 
    nrl_path=None, 
    infrabsu_xml=None,
    verbose=False
) -> Inventory:
    """
    Handle special sensor types (Raspberry Shake, infraBSU, Chaparral) and return an Inventory.
    """

    # Extract row values safely
    net = str(row["network"]).strip()
    sta = str(row["station"]).strip()
    loc = str(row["location"]).strip()
    chans = expand_channel_code(row["channel"])
    lat = float(row.get("lat", 0.0))
    lon = float(row.get("lon", 0.0))
    elev = float(row.get("elev", 0.0))
    depth = float(row.get("depth", 0.0))
    fsamp = float(row.get("fsamp", 100.0))
    vpp = float(row.get("vpp", 40.0))
    datalogger = str(row.get('datalogger', '')).strip()
    sensor = str(row.get('sensor', '')).strip()
    ondate = row.get("ondate", UTCDateTime("2000-01-01"))
    offdate = row.get("offdate", UTCDateTime("2100-01-01"))

    inv = Inventory(networks=[], source="sensor_type_dispatch")

    # --- Special handling ---
    # inside sensor_type_dispatch(...)
    if datalogger.upper() in ['RSB', 'RBOOM', 'RS1D', 'RS3D']:
        kind = _rs_kind(datalogger, fsamp)
        try:
            inv = build_inv_from_template(
                kind=kind,
                net=net, sta=sta, loc=loc,
                #keep_channels=chans,                   # keep whatever is passed
                keep_channels=None,                   # keep whatever’s in the template
                coords=(lat, lon, elev, depth),
                validity=(ondate, offdate),
                sr_overrides=None,                    # usually not needed
                propagate_response=True,
            )
            return inv
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print("Run `download_rshake_seiscompxml_convert_stationxml_wrapper.sh` on newton "
                "to create missing template StationXML files for Raspberry Shakes.")
            raise e

    elif sensor.lower().startswith("infrabsu"):
        # ensure we have a cached template
        if not infrabsu_xml:
            infrabsu_xml = str(get_infrabsu_sensor_template())

        for ch in chans:
            inv += get_infrabsu_centaur(
                template_path=infrabsu_xml,
                fsamp=fsamp,
                vpp=int(vpp),
                network=net,
                station=sta,
                location=loc,
                channel=ch.strip(),
                latitude=lat,
                longitude=lon,
                elevation=elev,
                depth=depth,
                start_date=ondate,
                end_date=offdate,
                # nrl_path=nrl_path,  # optional: pass a local NRL if you have it
                verbose=verbose,
            )
        return inv    



    elif "chaparral" in sensor.lower():
        inv = NRL2inventory(
            nrl_path,
            net, sta, loc, chans,
            fsamp=fsamp, Vpp=vpp,
            datalogger=datalogger,
            sensor=sensor,
            lat=lat, lon=lon, elev=elev, depth=depth,
            ondate=ondate, offdate=offdate
        )

        # --- Chaparral M-25 overall sensitivity (counts/Pa) ---
        # Sensor: 0.05 V/Pa
        SENSOR_V_PER_PA = 0.05
        # Centaur: counts/volt = 0.4e6 * 40 / inputVoltageRange
        # p2p/range source: Excel column 'vpp' (already parsed to vpp above)
        input_range = float(vpp) if vpp in (1, 40, 1.0, 40.0) else 40.0
        counts_per_volt = 0.4e6 * 40.0 / input_range
        desired_counts_per_pa = SENSOR_V_PER_PA * counts_per_volt  # -> 20_000 (40 Vpp) or 800_000 (1 Vpp)

        # Normalize units, set instrument_sensitivity, and fix stage-gain product to match
        _set_overall_counts_per_pa(inv, desired_counts_per_pa, freq_hz=1.0, tweak_stage=True, verbose=verbose)

        return inv

    else:
        # Default: try NRL
        inv = NRL2inventory(
            nrl_path,
            net,
            sta,
            loc,
            chans,
            fsamp=fsamp,
            Vpp=vpp,
            datalogger=datalogger,
            sensor=sensor,
            lat=lat,
            lon=lon,
            elev=elev,
            depth=depth,
            ondate=ondate,
            offdate=offdate
        )
        return inv
    
import math

def _set_channel_orientation(inv, net, sta, loc, chan, azimuth, dip):
    """
    Find Channel(net, sta, loc, chan) in `inv` and set azimuth/dip if provided.
    Accepts NaN/None as 'skip'.
    """
    def _isnum(x):
        try:
            return (x is not None) and (not (isinstance(x, float) and math.isnan(x)))
        except Exception:
            return False

    if not _isnum(azimuth) and not _isnum(dip):
        return  # nothing to set

    for n in inv.networks:
        if n.code != str(net):
            continue
        for s in n.stations:
            if s.code != str(sta):
                continue
            for c in s.channels:
                if c.location_code == str(loc) and c.code == str(chan):
                    if _isnum(azimuth):
                        # wrap [0, 360)
                        c.azimuth = float(azimuth) % 360.0
                    if _isnum(dip):
                        c.dip = float(dip)
                    return
                
# --- Units / response utilities ---------------------------------------------

def _units_name(u):
    # Works for ObsPy Units object or plain str
    if u is None:
        return None
    return getattr(u, "name", u if isinstance(u, str) else None)

def _make_units(name: str, desc: str | None = None):
    try:
        from obspy.core.inventory.util import Units
        u = Units(name=name)
        try:
            if desc:
                u.description = desc
        except Exception:
            pass
        return u
    except Exception:
        # Older ObsPy accepts plain strings
        return name

def _normalize_units_on_inventory(inv) -> None:
    """
    Replace 'COUNTS'/'Counts'->'count', 'M/S'->'m/s', 'PA'->'Pa' across all channels/stages.
    """
    def _norm(s):
        if not s:
            return s
        t = str(s)
        if t.upper() in ("COUNT", "COUNTS"):
            return "count"
        if t.upper() == "M/S":
            return "m/s"
        if t.upper() == "PA":
            return "Pa"
        return t

    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                resp = cha.response
                if not resp:
                    continue
                # Stages
                for stg in getattr(resp, "response_stages", []) or []:
                    for attr in ("input_units", "output_units"):
                        old = getattr(stg, attr, None)
                        nm = _norm(_units_name(old))
                        if nm:
                            setattr(stg, attr, _make_units(nm))
                # Top-level instrument_sensitivity (if present now or later)
                ins = getattr(resp, "instrument_sensitivity", None)
                if ins is not None:
                    inu = _norm(_units_name(ins.input_units)) or "Pa"
                    onu = _norm(_units_name(ins.output_units)) or "count"
                    ins.input_units  = _make_units(inu, "Pascals" if inu == "Pa" else None)
                    ins.output_units = _make_units(onu, "Digital counts" if onu == "count" else None)

def _stage_gain_value(stg):
    g = getattr(stg, "stage_gain", None)
    if g is None:
        return None
    # ObsPy 1.5+ often stores float; 1.4 used StageGain(value=..)
    v = getattr(g, "value", None)
    try:
        return float(v if v is not None else g)
    except Exception:
        return None

def _set_stage_gain(stg, value, freq=1.0):
    try:
        from obspy.core.inventory.response import StageGain  # old style
    except Exception:
        stg.stage_gain = float(value)
        stg.stage_gain_frequency = float(freq)
    else:
        stg.stage_gain = StageGain(value=float(value), frequency=float(freq))

def _product_of_stage_gains(resp) -> float | None:
    prod = 1.0
    saw = False
    for stg in getattr(resp, "response_stages", []) or []:
        v = _stage_gain_value(stg)
        if v is None:
            continue
        saw = True
        prod *= float(v)
    return prod if saw else None

def _ensure_decimation_defaults(resp, channel_rate: float):
    # Keep existing values; if missing, make a factor=1 decimation (validator-friendly)
    try:
        from obspy.core.inventory.response import Decimation
    except Exception:
        Decimation = None

    for stg in getattr(resp, "response_stages", []) or []:
        dec = getattr(stg, "decimation", None)
        factor = getattr(dec, "factor", None) if dec is not None else None
        ok = False
        try:
            ok = (factor is not None) and (int(factor) >= 1)
        except Exception:
            ok = False

        if not ok:
            if Decimation is not None:
                stg.decimation = Decimation(
                    input_sample_rate=float(channel_rate),
                    factor=1,
                    offset=0.0, delay=0.0, correction=0.0
                )
                stg.decimation.output_sample_rate = float(channel_rate)
            else:
                class _D: pass
                d = _D()
                d.input_sample_rate = float(channel_rate)
                d.factor = 1
                d.output_sample_rate = float(channel_rate)
                d.offset = d.delay = d.correction = 0.0
                stg.decimation = d

def _set_overall_counts_per_pa(inv, desired_overall_counts_per_pa: float, freq_hz: float = 1.0, tweak_stage=True, verbose=False):
    """
    Set instrument_sensitivity (counts/Pa) and, optionally, scale the final stage gain
    so the product of stage gains equals the total (to satisfy validator 412).
    """
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                resp = cha.response
                if not resp:
                    continue

                # Ensure units are normalized before setting sensitivity
                _normalize_units_on_inventory(inv)

                # Set instrument_sensitivity
                from obspy.core.inventory.response import InstrumentSensitivity
                resp.instrument_sensitivity = InstrumentSensitivity(
                    value=float(desired_overall_counts_per_pa),
                    frequency=float(freq_hz),
                    input_units=_make_units("Pa", "Pascals"),
                    output_units=_make_units("count", "Digital counts"),
                )

                # Make stage-gain product match total (if requested)
                if tweak_stage:
                    prod = _product_of_stage_gains(resp)
                    if prod is None or prod == 0.0:
                        # If there are no stage gains yet, make a final synthetic one
                        if verbose:
                            print(f"[chaparral] No stage gains found; adding synthetic stage gain -> {desired_overall_counts_per_pa}")
                        class _Stage: pass
                        stg = _Stage()
                        _set_stage_gain(stg, desired_overall_counts_per_pa, freq_hz)
                        resp.response_stages = (resp.response_stages or []) + [stg]
                    else:
                        scale = float(desired_overall_counts_per_pa) / prod
                        if not np.isclose(scale, 1.0, rtol=1e-6, atol=1e-6):
                            last = (resp.response_stages or [])[-1]
                            last_gain = _stage_gain_value(last) or 1.0
                            _set_stage_gain(last, last_gain * scale, freq_hz)

                # Decimation defaults (in case some stages lack it)
                _ensure_decimation_defaults(resp, cha.sample_rate or 100.0)
    
def build_inventory_from_dataframe(df, nrl_path=None, infrabsu_xml=None, verbose=True):
    """
    Converts a cleaned metadata DataFrame into an ObsPy Inventory.

    This function dispatches each row to an appropriate sensor handling function
    based on sensor type (e.g., Raspberry Shake, infraBSU, Chaparral, or NRL-based).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned metadata table with one row per NSLC entry. Must include columns:
        'network', 'station', 'location', 'channel', 'sensor', 'datalogger',
        'lat', 'lon', 'elev', 'depth', 'fsamp', 'vpp', 'ondate', and 'offdate'.
    nrl_path : str, optional
        Path to local NRL directory. If None, remote IRIS NRL is used.
    infrabsu_xml : str, optional
        Path to cached base StationXML file for infraBSU response reuse.
    verbose : bool, default=True
        If True, logs progress and warnings.

    Returns
    -------
    obspy.core.inventory.Inventory
        Combined ObsPy Inventory object for all valid rows.
    """
    import traceback
    inventory = Inventory(networks=[], source="build_inventory_from_dataframe")

    for _, row in df.iterrows():
        if verbose:
            print('\n' + f'[INFO] build_inventory_from_dataframe: Processing {row.get("network", "?")}.{row.get("station", "?")}.{row.get("location", "")}.{row.get("channel", "")}')

        try:
            inv_piece = sensor_type_dispatch(
                row,
                nrl_path=nrl_path,
                infrabsu_xml=infrabsu_xml,
                verbose=verbose
            )

            # --- NEW: push channel_azimuth / channel_dip into the Channel ---
            az = row.get("channel_azimuth", None)
            dp = row.get("channel_dip", None)
            _set_channel_orientation(
                inv_piece,
                net=row.get("network", ""),
                sta=row.get("station", ""),
                loc=row.get("location", ""),
                chan=row.get("channel", ""),
                azimuth=az,
                dip=dp
            )
            # ---------------------------------------------------------------
            sanitize_inventory_units(inv_piece)
            inventory += inv_piece
        except Exception as e:
            if verbose:
                net = row.get("network", "?")
                sta = row.get("station", "?")
                loc = row.get("location", "")
                chan = row.get("channel", "")
                print(f"[WARN] Failed to parse {net}.{sta}.{loc}.{chan}: {e}")
                traceback.print_exc()

    print('\nregular merge')
    return merge_inventories(inventory)

def print_station_epochs(inv: Inventory, *, stations=None, networks=None, limit=None):
    """
    Print (loc, chan, start, end) per station; optionally filter by station/network.
    """
    count = 0
    for net in inv.networks:
        if networks and net.code not in set(networks): 
            continue
        for sta in net.stations:
            if stations and sta.code not in set(stations): 
                continue
            print(f"\n== {net.code}.{sta.code} [{sta.start_date} → {sta.end_date}] ==")
            buckets = defaultdict(list)
            for ch in sta.channels:
                if ch.code=='DD1':
                    buckets[(ch.location_code, ch.code)].append(ch)
            for (loc, chan), chans in sorted(buckets.items()):
                chans_sorted = sorted(chans, key=lambda c: (c.start_date, c.end_date or c.start_date))
                for c in chans_sorted:
                    print(f"  {loc}.{chan}  {c.start_date} → {c.end_date}")
                    count += 1
                    if limit and count >= limit:
                        return


def build_inventory_from_table(path, sheet_name='ksc_stations_master', nrl_path=None, infrabsu_xml=None, verbose=True):
    """
    Builds an ObsPy Inventory from an Excel or CSV metadata table.

    This function reads a metadata file (CSV or Excel), processes it into a clean
    DataFrame, and converts the DataFrame into an ObsPy Inventory. It supports
    response reuse for infraBSU sensors if a prebuilt StationXML is provided.

    Parameters
    ----------
    path : str
        Path to the Excel (.xlsx) or CSV (.csv) metadata file.
    sheet_name : str, optional
        Sheet name to read (used only for Excel files). Default is 'ksc_stations_master'.
    nrl_path : str, optional
        Path to a local copy of the NRL. If None, remote NRL is used.
    infrabsu_xml : str, optional
        Path to a prebuilt StationXML file for infraBSU response reuse.
    verbose : bool, default=True
        If True, print progress and warnings.

    Returns
    -------
    obspy.core.inventory.Inventory
        ObsPy Inventory object representing all stations and responses.
    """
    df = build_dataframe_from_table(path, sheet_name=sheet_name)
    return build_inventory_from_dataframe(df, nrl_path=nrl_path, infrabsu_xml=infrabsu_xml, verbose=verbose)


def create_trace_inventory(tr, netname='', sitename='', net_ondate=None, \
                           sta_ondate=None, lat=0.0, lon=0.0, elev=0.0, depth=0.0, azimuth=0.0, dip=-90.0, stationXml=None):
    """
    Create a minimal ObsPy Inventory for a single trace, using NRL responses.

    This function constructs a minimal station inventory for one trace using metadata from
    the trace header and assumed/default metadata values. It fetches the instrument response
    from the online NRL (Nominal Response Library) and optionally writes a StationXML file.

    Parameters:
    tr (Trace): ObsPy Trace object.
    netname (str): Description of the network.
    sitename (str): Description of the station site.
    net_ondate (UTCDateTime or None): Start date for the network. Defaults to trace starttime.
    sta_ondate (UTCDateTime or None): Start date for the station. Defaults to trace starttime.
    lat (float): Station latitude.
    lon (float): Station longitude.
    elev (float): Station elevation in meters.
    depth (float): Channel sensor depth in meters.
    azimuth (float): Sensor azimuth in degrees.
    dip (float): Sensor dip in degrees (e.g. -90 for vertical).
    stationXml (str or None): Optional output path for StationXML file.

    Returns:
    Inventory: ObsPy Inventory containing the trace's metadata and instrument response.
    """
 
    inv = Inventory(networks=[], source='Glenn Thompson')
    if not sta_ondate:
        net_ondate = tr.stats.starttime
    if not sta_ondate:
        sta_ondate = tr.stats.starttime        
    net = Network(
        # This is the network code according to the SEED standard.
        code=tr.stats.network,
        # A list of stations. We'll add one later.
        stations=[],
        description=netname,
        # Start-and end dates are optional.
        start_date=net_ondate)

    sta = Station(
        # This is the station code according to the SEED standard.
        code=tr.stats.station,
        latitude=lat,
        longitude=lon,
        elevation=elev,
        creation_date=sta_ondate, 
        site=Site(name=sitename))

    cha = Channel(
        # This is the channel code according to the SEED standard.
        code=tr.stats.channel,
        # This is the location code according to the SEED standard.
        location_code=tr.stats.location,
        # Note that these coordinates can differ from the station coordinates.
        latitude=lat,
        longitude=lon,
        elevation=elev,
        depth=depth,
        azimuth=azimuth,
        dip=dip,
        sample_rate=tr.stats.sampling_rate)

    # By default this accesses the NRL online. Offline copies of the NRL can
    # also be used instead
    nrl = NRL()
    # The contents of the NRL can be explored interactively in a Python prompt,
    # see API documentation of NRL submodule:
    # http://docs.obspy.org/packages/obspy.clients.nrl.html
    # Here we assume that the end point of data logger and sensor are already
    # known:
    response = nrl.get_response( # doctest: +SKIP
        sensor_keys=['Streckeisen', 'STS-1', '360 seconds'],
        datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])


    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)
    
    # And finally write it to a StationXML file. We also force a validation against
    # the StationXML schema to ensure it produces a valid StationXML file.
    #
    # Note that it is also possible to serialize to any of the other inventory
    # output formats ObsPy supports.
    if stationXml:
        print('Writing inventory to %s' % stationXml)
        inv.write(stationXml, format="stationxml", validate=True)    
    return inv



def merge_inventories(*inventories):
    """
    Merge and patch one or more ObsPy Inventory objects into a single, deduplicated Inventory.

    This function:
    - Combines all provided inventories into a new Inventory.
    - Deduplicates networks and stations.
    - Merges stations with the same code, even across overlapping or adjacent time windows.
    - Merges channels by (code, location_code) and allows multiple versions if:
        - Time windows do not overlap, or
        - Metadata differs (with a warning).
    - Fills missing site name and description where possible.

    Parameters:
    -----------
    *inventories : obspy.core.inventory.Inventory
        One or more Inventory objects to be merged and cleaned.

    Returns:
    --------
    Inventory
        A new, merged, deduplicated, and patched Inventory object.
    """
    def channels_overlap(c1, c2):
        """Check if two channels overlap in time."""
        s1 = c1.start_date or UTCDateTime(0)
        e1 = c1.end_date or UTCDateTime(2100, 1, 1)
        s2 = c2.start_date or UTCDateTime(0)
        e2 = c2.end_date or UTCDateTime(2100, 1, 1)
        return s1 <= e2 and s2 <= e1

    def channel_metadata_equivalent(c1, c2):
        """Compare essential metadata fields between two channels."""
        keys = ["code", "location_code", "sample_rate", "azimuth", "dip"]
        for key in keys:
            if getattr(c1, key, None) != getattr(c2, key, None):
                return False
        # Compare responses (hash or text representation)
        try:
            return c1.response.__str__() == c2.response.__str__()
        except Exception:
            return False

    merged_inv = Inventory(networks=[], source="Merged and patched by merge_inventories()")

    all_networks = defaultdict(list)
    for inv in inventories:
        for net in inv:
            all_networks[net.code].append(net)

    for net_code, nets in all_networks.items():
        merged_net = Network(code=net_code, description=nets[0].description or "")
        stations_by_code = defaultdict(list)

        for net in nets:
            for sta in net:
                stations_by_code[sta.code].append(sta)

        for sta_code, sta_list in stations_by_code.items():
            sta_list.sort(key=lambda s: s.start_date or UTCDateTime(0))
            merged_stations = []

            while sta_list:
                current = sta_list.pop(0).copy()
                current_start = current.start_date or UTCDateTime(0)
                current_end = current.end_date or UTCDateTime(2100, 1, 1)

                i = 0
                while i < len(sta_list):
                    other = sta_list[i]
                    other_start = other.start_date or UTCDateTime(0)
                    other_end = other.end_date or UTCDateTime(2100, 1, 1)

                    if other_start <= current_end and other_end >= current_start:
                        sta_list.pop(i)

                        # Patch missing site info
                        if not current.site.name and other.site.name:
                            current.site.name = other.site.name
                        if not current.site.description and other.site.description:
                            current.site.description = other.site.description

                        # Merge channels with smart logic
                        for new_chan in other.channels:
                            match_found = False
                            for existing_chan in current.channels:
                                if (new_chan.code == existing_chan.code and
                                    new_chan.location_code == existing_chan.location_code):
                                    if channels_overlap(new_chan, existing_chan):
                                        if channel_metadata_equivalent(new_chan, existing_chan):
                                            match_found = True  # exact duplicate
                                        else:
                                            print(f"[WARN] Channel metadata mismatch for {sta_code}.{new_chan.location_code}.{new_chan.code} during overlapping period")
                            if not match_found:
                                current.channels.append(new_chan)

                        # Extend time window
                        current_start = min(current_start, other_start)
                        current_end = max(current_end, other_end)
                        current.start_date = current_start
                        current.end_date = current_end
                    else:
                        i += 1

                """
                # >>> ADD: coalesce channel epochs by (loc, code, response) and stitch touching windows
                _condense_station_channels(current, touch_tol=1e-6) # in the other merge_inv

                # >>> ADD: ensure station epoch covers union of channel epochs
                if current.channels:
                    c_starts = [ch.start_date for ch in current.channels if ch.start_date]
                    c_ends   = [ch.end_date   for ch in current.channels if ch.end_date]
                    if c_starts:
                        smin = min(c_starts)
                        current.start_date = min(current.start_date or smin, smin)
                    if any(ch.end_date is None for ch in current.channels):
                        current.end_date = None
                    elif c_ends:
                        smax = max(c_ends)
                        current.end_date = max(current.end_date or smax, smax)
                # <<< END ADD
                """

                if not current.site.name:
                    current.site.name = f"AUTO_NAME_{sta_code}"
                if not current.site.description:
                    current.site.description = f"Autogenerated site description for {sta_code}"

                merged_stations.append(current)

            merged_net.stations.extend(merged_stations)

        merged_inv.networks.append(merged_net)

    return merged_inv


if __name__ == "__main__":
    import traceback
    import pandas as pd

    print("[TEST] build/sensor_type_dispatch glue – quick integration test")

    # 1) Make a few representative “rows” like what your DataFrame would contain
    # NOTE: Times are strings to mimic CSV/Excel; build_dataframe_from_table normally parses, so we do it manually here
    rows = [
        # Raspberry Boom (will load local template if present, else print hint)
        {
            "network": "AM", "station": "RBTEST", "location": "00", "channel": "HDF",
            "sensor": "RBOOM", "datalogger": "RBOOM",
            "lat": 28.6, "lon": -80.65, "elev": 5.0, "depth": 0.0,
            "fsamp": 100.0, "vpp": 40,
            "ondate": UTCDateTime("2024-01-01"), "offdate": UTCDateTime("2100-01-01"),
        },
        # Raspberry Shake 1D v6 @100 Hz (HHZ->EHZ patch happens inside build_inv_from_template)
        {
            "network": "AM", "station": "RS1D6T", "location": "00", "channel": "EHZ",
            "sensor": "RS1D", "datalogger": "RS1D",
            "lat": 28.6, "lon": -80.65, "elev": 5.0, "depth": 0.0,
            "fsamp": 100.0, "vpp": 40,
            "ondate": UTCDateTime("2024-01-01"), "offdate": UTCDateTime("2100-01-01"),
        },
        # infraBSU + Centaur path (auto-downloads template if missing)
        {
            "network": "1R", "station": "INFRA1", "location": "10", "channel": "HDF",
            "sensor": "infraBSU", "datalogger": "Centaur",
            "lat": 28.5721, "lon": -80.6480, "elev": 3.0, "depth": 0.0,
            "fsamp": 100.0, "vpp": 40,
            "ondate": UTCDateTime("2024-01-01"), "offdate": UTCDateTime("2100-01-01"),
        },
        # Chaparral via NRL (falls back to PAZ if lookup fails)
        {
            "network": "XX", "station": "CHAP1", "location": "00", "channel": "HDF",
            "sensor": "CHAP-25", "datalogger": "Centaur",
            "lat": 0.0, "lon": 0.0, "elev": 0.0, "depth": 0.0,
            "fsamp": 100.0, "vpp": 40,
            "ondate": UTCDateTime("2020-01-01"), "offdate": UTCDateTime("2025-12-31"),
        },
        # Default NRL (TCP + Centaur)
        {
            "network": "XX", "station": "TCPT1", "location": "00", "channel": "HHZ",
            "sensor": "TCP", "datalogger": "Centaur",
            "lat": 0.0, "lon": 0.0, "elev": 0.0, "depth": 0.0,
            "fsamp": 200.0, "vpp": 40,
            "ondate": UTCDateTime("2019-01-01"), "offdate": UTCDateTime("2021-01-01"),
        },
    ]

    # 2) Wrap each row through sensor_type_dispatch directly
    for r in rows:
        tag = f"{r['network']}.{r['station']}.{r['location']}.{r['channel']} [{r['datalogger']}/{r['sensor']}]"
        print("\n--- Testing row:", tag)
        try:
            inv_piece = sensor_type_dispatch(r, nrl_path=None, infrabsu_xml=None, verbose=True)
            print(inv_piece)
        except FileNotFoundError as e:
            # This is expected when a Raspberry Shake template is missing
            print(f"[MISSING] {e}")
            print("[HINT] run download_rshake_seiscompxml_convert_stationxml_wrapper.sh on newton "
                  "to create missing template StationXML files for Raspberry Shakes.")
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()

    # 3) Also test the full DataFrame -> Inventory pipeline in one go
    print("\n[TEST] DataFrame → build_inventory_from_dataframe")
    df = pd.DataFrame(rows)
    try:
        inv_all = build_inventory_from_dataframe(df, nrl_path=None, infrabsu_xml=None, verbose=True)
        print(inv_all)
    except Exception as e:
        print(f"[ERROR] building inventory from dataframe: {e}")
        traceback.print_exc()