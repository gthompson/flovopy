# flovopy/stationmetadata/raspberryshake.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Dict, Sequence, Tuple

from obspy import read_inventory, UTCDateTime
from obspy.core.inventory import Inventory, Station

from flovopy.stationmetadata.utils import get_templates_dir


# ---------------------------------------------------------------------------
# Template registry (filenames that live in stationxml_templates/)
# ---------------------------------------------------------------------------

_TEMPLATE_FILENAMES = {
    "RBOOM_v3_100Hz_20s": "RBOOM_v3_100Hz_20s.xml",
    "RSnB_v3_100Hz_20s":  "RSnB_v3_100Hz_20s.xml",
    "RS1D_v4_50Hz":       "RS1D_v4_50Hz.xml",
    "RS1D_v6_100Hz":      "RS1D_v6_100Hz.xml",
    "RS3D_v5_100Hz":      "RS3D_v5_100Hz.xml",
}


def _template_path(kind: str) -> Path:
    """
    Compute the on-repo path for a given template 'kind'.
    """
    try:
        fname = _TEMPLATE_FILENAMES[kind]
    except KeyError as e:
        raise KeyError(f"Unknown Raspberry Shake template kind: {kind}") from e
    return get_templates_dir() / fname


def _load_template(kind: str) -> Inventory:
    """
    Load a Raspberry Shake StationXML template from stationxml_templates/.
    If missing, instruct the user to build it (no auto-download here).
    """
    tpl = _template_path(kind)
    if not tpl.exists():
        print(f"[MISSING] Template not found: {tpl}")
        print(
            "[ACTION] Run this on 'newton' to create missing StationXML templates "
            "for Raspberry Shakes:\n"
            "  download_rshake_seiscompxml_convert_stationxml_wrapper.sh"
        )
        raise FileNotFoundError(f"Missing Raspberry Shake StationXML template: {tpl}")
    return read_inventory(str(tpl))


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def _set_coords_and_times(
    sta: Station,
    lat: Optional[float],
    lon: Optional[float],
    elev: Optional[float],
    depth: Optional[float],
    start: Optional[UTCDateTime],
    end: Optional[UTCDateTime],
) -> None:
    # Station-level
    if lat is not None:
        sta.latitude = float(lat)
    if lon is not None:
        sta.longitude = float(lon)
    if elev is not None:
        sta.elevation = float(elev)
    if start is not None:
        sta.start_date = start
    if end is not None:
        sta.end_date = end

    # Mirror to channels (common practice)
    for ch in sta.channels:
        if lat is not None:
            ch.latitude = float(lat)
        if lon is not None:
            ch.longitude = float(lon)
        if elev is not None:
            ch.elevation = float(elev)
        if depth is not None:
            ch.depth = float(depth)
        if start is not None:
            ch.start_date = start
        if end is not None:
            ch.end_date = end


def _set_ids(inv: Inventory, net_code: str, sta_code: str, loc_code: str) -> None:
    # single-net, single-station assumption for templates
    net = inv.networks[0]
    sta = net.stations[0]
    net.code = net_code
    sta.code = sta_code
    for ch in sta.channels:
        ch.location_code = loc_code


def _filter_channels(sta: Station, keep_codes: Optional[Iterable[str]]) -> None:
    if not keep_codes:
        return
    keep = {c.upper() for c in keep_codes}
    sta.channels = [ch for ch in sta.channels if ch.code.upper() in keep]


def _apply_sample_rate_overrides(
    sta: Station,
    sr_overrides: Optional[Dict[str, float]] = None,
    propagate_response: bool = True,
) -> None:
    """
    sr_overrides: map channel code -> sample rate (Hz). e.g., {"HDF": 100.0}
    If propagate_response=True, try to keep response stage decimation I/O rates aligned.
    """
    if not sr_overrides:
        return
    for ch in sta.channels:
        if ch.code in sr_overrides:
            new_sr = float(sr_overrides[ch.code])
            ch.sample_rate = new_sr
            if propagate_response and ch.response and getattr(ch.response, "response_stages", None):
                # naive forward propagation through stages (if present)
                current = new_sr
                for stage in ch.response.response_stages:
                    if hasattr(stage, "decimation_input_sample_rate"):
                        stage.decimation_input_sample_rate = current
                    if hasattr(stage, "decimation_factor") and stage.decimation_factor not in (None, 0):
                        try:
                            fac = float(stage.decimation_factor)
                        except Exception:
                            fac = None
                        if fac and fac != 0:
                            out_sr = current / fac
                            if hasattr(stage, "decimation_output_sample_rate"):
                                stage.decimation_output_sample_rate = out_sr
                            current = out_sr


# ---------------------------------------------------------------------------
# Generic builder from a named template
# ---------------------------------------------------------------------------
def _patch_channel_codes(sta, mapping: dict[str, str]) -> None:
    for ch in sta.channels:
        new = mapping.get(ch.code.upper())
        if new:
            ch.code = new

def build_inv_from_template(
    kind: str,
    net: str,
    sta: str,
    loc: str,
    keep_channels: Optional[Sequence[str]] = None,
    coords: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = (None, None, None, None),
    validity: Tuple[Optional[UTCDateTime], Optional[UTCDateTime]] = (None, None),
    sr_overrides: Optional[Dict[str, float]] = None,
    propagate_response: bool = True,
) -> Inventory:
    """
    Load a StationXML template and rewrite it with your metadata.
    """
    inv = _load_template(kind)
    net_obj = inv.networks[0]
    sta_obj = net_obj.stations[0]

    # Fix RS1D v6 mislabeled channel
    if kind == "RS1D_v6_100Hz":
        _patch_channel_codes(sta_obj, {"HHZ": "EHZ"})

    _set_ids(inv, net, sta, loc)
    #_filter_channels(sta_obj, keep_channels)
    _set_coords_and_times(sta_obj, coords[0], coords[1], coords[2], coords[3], validity[0], validity[1])
    _apply_sample_rate_overrides(sta_obj, sr_overrides, propagate_response=propagate_response)

    # Special-case: RS1D v6 templates use HHZ, rename to EHZ
    if kind.lower().startswith("rs1d_v6"):
        for ch in sta_obj.channels:
            if ch.code == "HHZ":
                print(f"[INFO] Renaming channel HHZ -> EHZ for {kind}")
                ch.code = "EHZ"

    # Light sanity: no channels left?
    if not sta_obj.channels:
        raise ValueError(f"Template {kind} left no channels after filtering {keep_channels}")

    return inv

def _rs_kind(datalogger: str, fsamp: float) -> str:
    dl = datalogger.strip().upper()
    if dl == "RBOOM": return "RBOOM_v3_100Hz_20s"
    if dl == "RSB":   return "RSnB_v3_100Hz_20s"
    if dl == "RS3D":  return "RS3D_v5_100Hz"
    if dl == "RS1D":
        if int(fsamp) == 50:  return "RS1D_v4_50Hz"
        if int(fsamp) == 100: return "RS1D_v6_100Hz"
        raise ValueError(f"RS1D needs fsamp 50 or 100, got {fsamp}")
    raise ValueError(f"Unsupported Raspberry Shake type: {datalogger}")

# ---------------------------------------------------------------------------
# Convenience wrappers for common RS models
#   - These just pick a template 'kind' and (optionally) a default keep list
# ---------------------------------------------------------------------------
'''
def get_rboom(
    sta: str,
    loc: str,
    *,
    net: str = "AM",
    fsamp: Optional[float] = 100.0,
    lat: float = 0.0,
    lon: float = 0.0,
    elev: float = 0.0,
    depth: float = 0.0,
    start_date: UTCDateTime = UTCDateTime(1900, 1, 1),
    end_date: UTCDateTime = UTCDateTime(2100, 1, 1),
) -> Inventory:
    """
    Raspberry Boom (HDF infrasound). Uses: RBOOM_v3_100Hz_20s template.
    """
    sr_overrides = {"HDF": fsamp} if fsamp else None
    return build_inv_from_template(
        kind="RBOOM_v3_100Hz_20s",
        net=net,
        sta=sta,
        loc=loc,
        keep_channels=["HDF"],
        coords=(lat, lon, elev, depth),
        validity=(start_date, end_date),
        sr_overrides=sr_overrides,
        propagate_response=True,
    )


def get_rsb(
    sta: str,
    loc: str,
    *,
    net: str = "AM",
    fsamp_hdf: Optional[float] = 100.0,
    fsamp_ehz: Optional[float] = 100.0,
    lat: float = 0.0,
    lon: float = 0.0,
    elev: float = 0.0,
    depth: float = 0.0,
    start_date: UTCDateTime = UTCDateTime(1900, 1, 1),
    end_date: UTCDateTime = UTCDateTime(2100, 1, 1),
) -> Inventory:
    """
    Raspberry Shake & Boom (EHZ + HDF). Uses: RSnB_v3_100Hz_20s template.
    """
    sr_overrides = {}
    if fsamp_hdf:
        sr_overrides["HDF"] = fsamp_hdf
    if fsamp_ehz:
        sr_overrides["EHZ"] = fsamp_ehz
    if not sr_overrides:
        sr_overrides = None

    return build_inv_from_template(
        kind="RSnB_v3_100Hz_20s",
        net=net,
        sta=sta,
        loc=loc,
        keep_channels=["EHZ", "HDF"],
        coords=(lat, lon, elev, depth),
        validity=(start_date, end_date),
        sr_overrides=sr_overrides,
        propagate_response=True,
    )


def get_rs1d_v4(
    sta: str,
    loc: str,
    *,
    net: str = "AM",
    fsamp: Optional[float] = 50.0,  # many early 1D units @ 50 Hz
    lat: float = 0.0,
    lon: float = 0.0,
    elev: float = 0.0,
    depth: float = 0.0,
    start_date: UTCDateTime = UTCDateTime(1900, 1, 1),
    end_date: UTCDateTime = UTCDateTime(2100, 1, 1),
) -> Inventory:
    """
    Raspberry Shake 1D (v4 response). Uses: RS1D_v4_50Hz template.
    """
    sr_overrides = {"EHZ": fsamp} if fsamp else None
    return build_inv_from_template(
        kind="RS1D_v4_50Hz",
        net=net,
        sta=sta,
        loc=loc,
        keep_channels=["EHZ"],
        coords=(lat, lon, elev, depth),
        validity=(start_date, end_date),
        sr_overrides=sr_overrides,
        propagate_response=True,
    )


def get_rs1d_v6(
    sta: str,
    loc: str,
    *,
    net: str = "AM",
    fsamp: Optional[float] = 100.0,
    lat: float = 0.0,
    lon: float = 0.0,
    elev: float = 0.0,
    depth: float = 0.0,
    start_date: UTCDateTime = UTCDateTime(1900, 1, 1),
    end_date: UTCDateTime = UTCDateTime(2100, 1, 1),
) -> Inventory:
    """
    Raspberry Shake 1D (v6 response). Uses: RS1D_v6_100Hz template.
    """
    sr_overrides = {"EHZ": fsamp} if fsamp else None
    return build_inv_from_template(
        kind="RS1D_v6_100Hz",
        net=net,
        sta=sta,
        loc=loc,
        keep_channels=["EHZ"],
        coords=(lat, lon, elev, depth),
        validity=(start_date, end_date),
        sr_overrides=sr_overrides,
        propagate_response=True,
    )


def get_rs3d_v5(
    sta: str,
    loc: str,
    *,
    net: str = "AM",
    fsamp_ehz: Optional[float] = 100.0,
    lat: float = 0.0,
    lon: float = 0.0,
    elev: float = 0.0,
    depth: float = 0.0,
    start_date: UTCDateTime = UTCDateTime(1900, 1, 1),
    end_date: UTCDateTime = UTCDateTime(2100, 1, 1),
) -> Inventory:
    """
    Raspberry Shake 3D (v5 response). Uses: RS3D_v5_100Hz template.
    """
    sr_overrides = {"EHZ": fsamp_ehz} if fsamp_ehz else None
    return build_inv_from_template(
        kind="RS3D_v5_100Hz",
        net=net,
        sta=sta,
        loc=loc,
        keep_channels=["EHZ", "EHN", "EHE"],  # keep all 3 components by default
        coords=(lat, lon, elev, depth),
        validity=(start_date, end_date),
        sr_overrides=sr_overrides,
        propagate_response=True,
    )
'''

if __name__ == "__main__":
    print("[TEST] Raspberry Shake template loading test\n")

    # name, kind_slug, kwargs to pass to build_inv_from_template
    tests = [
        ("RBOOM",   "RBOOM_v3_100Hz_20s", {"net": "AM", "sta": "RBTEST",  "loc": "00"}),
        ("RS&B",    "RSnB_v3_100Hz_20s",  {"net": "AM", "sta": "RSBTEST", "loc": "00"}),
        ("RS1D v4", "RS1D_v4_50Hz",       {"net": "AM", "sta": "RS1D4T",  "loc": "00"}),
        # Keep EHZ just to illustrate keep_channels; HHZ->EHZ patch happens inside build_inv_from_template for v6
        ("RS1D v6", "RS1D_v6_100Hz",      {"net": "AM", "sta": "RS1D6T",  "loc": "00", "keep_channels": ["EHZ"]}),
        ("RS3D v5", "RS3D_v5_100Hz",      {"net": "AM", "sta": "RS3D5T",  "loc": "00"}),
    ]

    for name, kind, kwargs in tests:
        print(f"\n--- Testing {name} ({kind}) ---")
        try:
            inv = build_inv_from_template(kind, **kwargs)
            print(inv)
        except FileNotFoundError as e:
            print(f"[MISSING] {e}")
            print("[HINT] run download_rshake_seiscompxml_convert_stationxml_wrapper.sh on newton "
                  "to create missing template StationXML files for Raspberry Shakes.")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
