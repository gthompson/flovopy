"""
infraBSU + Centaur StationXML utilities (backwards compatible with older ObsPy).

- download_infrabsu_centaur_stationxml(): fetch & cache infraBSU sensor-only StationXML
- get_infrabsu_centaur(): load sensor template, append Centaur datalogger chain, stamp NSLC/coords/times
"""

from __future__ import annotations

import copy
from typing import Optional, Iterable

import numpy as np
import requests

from obspy import read_inventory, UTCDateTime
from obspy.core.inventory import Inventory, Site
from obspy.core.inventory.response import InstrumentSensitivity
from obspy.clients.nrl import NRL
from pathlib import Path
import requests
from flovopy.stationmetadata.utils import get_templates_dir



# --- Units compatibility shim -----------------------------------------------
try:
    # Newer ObsPy exposes a Units class
    from obspy.core.inventory.util import Units as _ObsPyUnits
    def _make_units(name: str, description: str | None = None):
        u = _ObsPyUnits(name=name)
        # Not all versions let us set description; guard it
        try:
            if description is not None:
                u.description = description
        except Exception:
            pass
        return u
    _UNITS_IS_OBJ = True
except Exception:
    # Older ObsPy: InstrumentSensitivity accepts plain strings
    def _make_units(name: str, description: str | None = None):
        return name
    _UNITS_IS_OBJ = False

DEFAULT_INFRABSU_URL = (
    "https://service.iris.edu/irisws/nrl/1/combine"
    "?instconfig=sensor_JeffreyBJohnson_infraBSU_LP21_SG0.000046_STairPressure"
    "&format=stationxml"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_stage_gains(stages: Iterable) -> Iterable[float]:
    """
    Yield numeric stage gains across ObsPy versions:
    - newer: stage.stage_gain.value
    - older: stage.stage_gain (float)
    """
    for stg in stages or []:
        sg = getattr(stg, "stage_gain", None)
        if sg is None:
            continue
        val = getattr(sg, "value", None)
        if val is None:
            try:
                val = float(sg)
            except Exception:
                val = None
        if val not in (None, 0):
            yield float(val)

def _normalize_decimation_fields(resp, start_rate: float) -> None:
    """
    Make StationXML-writer-safe decimation metadata:
    - Ensure nested `stage.decimation` with coherent input/factor/output
    - Ensure legacy flat attributes exist and are None (writer checks them)
    """
    current = float(start_rate)
    for stg in getattr(resp, "response_stages", []) or []:
        factor = None
        nested = getattr(stg, "decimation", None)
        if nested is not None:
            f = getattr(nested, "factor", None)
            try:
                factor = float(f) if f not in (None, 0) else None
            except Exception:
                factor = None
        if factor in (None, 0) and hasattr(stg, "decimation_factor"):
            try:
                f = getattr(stg, "decimation_factor")
                factor = float(f) if f not in (None, 0) else None
            except Exception:
                factor = None

        # Ensure nested decimation object exists
        if nested is None:
            class _Decim:  # simple holder with attributes
                pass
            nested = _Decim()
            stg.decimation = nested

        # Fill nested fields
        nested.input_sample_rate = current
        if factor and factor != 0.0:
            nested.factor = factor
            nested.output_sample_rate = current / factor
            current = nested.output_sample_rate
        else:
            nested.factor = 1.0
            nested.output_sample_rate = current

        # Defaults that writers tolerate
        if not hasattr(nested, "offset"):     nested.offset = 0.0
        if not hasattr(nested, "delay"):      nested.delay = 0.0
        if not hasattr(nested, "correction"): nested.correction = 0.0

        # Legacy flat fields must exist but be None so writer doesn't treat them as typed objects
        stg.decimation_input_sample_rate = None
        stg.decimation_output_sample_rate = None
        stg.decimation_factor = None

def _reset_overall_instrument_sensitivity(resp):
    """
    Set a correct top-level InstrumentSensitivity from stage gains.
    InputUnits: from first stage input (fallback 'Pa')
    OutputUnits: from last stage output (fallback 'COUNTS')
    Frequency: 1.0 Hz
    """
    stages = getattr(resp, "response_stages", []) or []
    if not stages:
        return

    in_units_obj = getattr(stages[0], "input_units", None)
    out_units_obj = getattr(stages[-1], "output_units", None)

    in_units_name = (getattr(in_units_obj, "name", None)
                     if in_units_obj is not None else None)
    if isinstance(in_units_obj, str):
        in_units_name = in_units_obj

    out_units_name = (getattr(out_units_obj, "name", None)
                      if out_units_obj is not None else None)
    if isinstance(out_units_obj, str):
        out_units_name = out_units_obj

    in_units_name = in_units_name or "Pa"
    out_units_name = out_units_name or "COUNTS"

    overall = 1.0
    for s in stages:
        g = getattr(s, "stage_gain", None)
        try:
            if g is not None:
                overall *= float(g)
        except Exception:
            pass

    resp.instrument_sensitivity = InstrumentSensitivity(
        value=overall,
        frequency=1.0,
        input_units=_make_units(in_units_name, "Pascals" if in_units_name == "Pa" else None),
        output_units=_make_units(out_units_name, "Digital Counts" if out_units_name.upper() == "COUNTS" else None),
    )

# ---------------------------------------------------------------------------
# 1) Download & stash infraBSU sensor StationXML template
# ---------------------------------------------------------------------------

def get_infrabsu_sensor_template(local_filename="infraBSU_sensor.xml", url=DEFAULT_INFRABSU_URL, timeout=60) -> Path:
    """
    Ensure the infraBSU sensor StationXML template exists under stationxml_templates/.
    Only downloads if the local file is missing.
    """
    templates_dir = get_templates_dir()
    

    local_path = templates_dir / local_filename
    if local_path.exists():
        print(f"[INFO] Template exists: {local_path}")
        return local_path

    print(f"[INFO] Downloading infraBSU sensor template from {url}")
    templates_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    local_path.write_bytes(r.content)
    print(f"[OK] Downloaded and saved to {local_path}")
    return local_path

# ---------------------------------------------------------------------------
# 2) Load template & build combined infraBSU + Centaur Inventory
# ---------------------------------------------------------------------------

def get_infrabsu_centaur(
    template_path: str | Path,
    *,
    fsamp: float = 100.0,
    vpp: int = 40,
    network: str = "XX",
    station: str = "DUMM",
    location: str = "10",
    channel: str = "HDF",
    latitude: float = 0.0,
    longitude: float = 0.0,
    elevation: float = 0.0,
    depth: float = 0.0,
    start_date: UTCDateTime = UTCDateTime(1970, 1, 1),
    end_date: UTCDateTime = UTCDateTime(2100, 1, 1),
    sitename: Optional[str] = None,
    nrl_path: Optional[str] = None,   # local NRL path if you have it; None => remote
    verbose: bool = True,
) -> Inventory:
    """
    Build a full infraBSU + Centaur response Inventory from a *sensor-only* template.
    """
    template_path = Path(template_path).expanduser().resolve()
    if verbose:
        print(f"[INFO] Reading infraBSU sensor template: {template_path}")
    inv = read_inventory(str(template_path))

    # Expect 1 net / 1 sta / 1 chan
    try:
        net = inv.networks[0]
        sta = net.stations[0]
        chan = sta.channels[0]
    except Exception as e:
        raise ValueError(f"Template must contain exactly one network/station/channel: {e}")

    if chan.response is None:
        raise ValueError("Template channel has no sensor response; cannot combine with Centaur.")

    sensor_resp = copy.deepcopy(chan.response)

    # NRL selection
    if vpp == 40:
        dl_keys = ['Nanometrics', 'Centaur', '40 Vpp (1)', 'Off', 'Linear phase', f"{int(fsamp)}"]
    elif vpp == 1:
        dl_keys = ['Nanometrics', 'Centaur', '1 Vpp (40)', 'Off', 'Linear phase', f"{int(fsamp)}"]
    else:
        raise ValueError(f"Unsupported Vpp: {vpp}. Expected 40 or 1.")

    dummy_sensor_keys = ['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '754 V/m/s']

    if verbose:
        print(f"[INFO] Querying NRL ({nrl_path or 'remote'}) for Centaur fsamp={fsamp}, Vpp={vpp}")
    nrl = NRL(nrl_path) if nrl_path else NRL()
    centaur_resp = nrl.get_response(sensor_keys=dummy_sensor_keys, datalogger_keys=dl_keys)

    stages = getattr(centaur_resp, "response_stages", None)
    if not stages:
        raise ValueError("NRL returned a response without stages; cannot combine.")

    # Combine responses: sensor first, then Centaur (skip dummy sensor at 0)
    combined_resp = copy.deepcopy(sensor_resp)
    add_stages = stages[1:] if len(stages) >= 2 else []
    combined_resp.response_stages.extend(copy.deepcopy(add_stages))

    # Normalize stage decimation metadata and recompute overall sensitivity
    _normalize_decimation_fields(combined_resp, start_rate=float(fsamp))
    _reset_overall_instrument_sensitivity(combined_resp)

    # Stamp metadata
    net.code = network
    sta.code = station
    sta.site = Site(
        name=sitename or f"{station}_SITE",
        description=f"InfraBSU component {channel} at {station}"
    )

    sta.latitude = latitude
    sta.longitude = longitude
    sta.elevation = elevation
    sta.start_date = start_date
    sta.end_date = end_date

    # Ensure we only keep the single channel from the template
    sta.channels = [chan]
    chan.code = channel
    chan.location_code = location
    chan.latitude = latitude
    chan.longitude = longitude
    chan.elevation = elevation
    chan.depth = depth
    chan.sample_rate = float(fsamp)
    chan.start_date = start_date
    chan.end_date = end_date
    chan.response = combined_resp

    if verbose:
        print(f"[OK] Built infraBSU+Centaur for {network}.{station}.{location}.{channel} @ {fsamp} Hz")

    return inv


if __name__ == "__main__":
    # Simple manual test
    tpl = get_infrabsu_sensor_template()
    inv = get_infrabsu_centaur(
        template_path=tpl,
        fsamp=100.0, vpp=40,
        network="1R", station="TEST", location="10", channel="HDF",
        latitude=28.5721, longitude=-80.6480, elevation=3.0, depth=0.0,
        start_date=UTCDateTime("2024-01-01"), end_date=UTCDateTime("2100-01-01"),
        sitename="Test Site",
        nrl_path=None,
        verbose=True,
    )
    print(inv)
    inv.write("TEST_INFRABSU_CENTAUR.xml", format="stationxml", validate=True)