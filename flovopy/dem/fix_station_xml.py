#!/usr/bin/env python3
# stationxml_dual_to_dataframe.py

from pathlib import Path
import pandas as pd
from obspy import read_inventory
from obspy.core.inventory import Inventory, Channel
from typing import List, Dict, Any, Optional

# --- Hardcoded StationXML paths ---
FILES = [
    Path("/Users/glennthompson/Dropbox/BRIEFCASE/SSADenver/MV.xml"),
    Path("/Users/glennthompson/Dropbox/MontserratDigitalSeismicNetwork.xml"),
]

OUT_CSV = Path("/Users/glennthompson/Dropbox/BRIEFCASE/SSADenver/stations_channels_table.csv")

def overall_sensitivity(chan: Channel) -> Optional[float]:
    """Return overall instrument sensitivity if available."""
    try:
        if chan.response and chan.response.instrument_sensitivity:
            return float(chan.response.instrument_sensitivity.value)
    except Exception:
        pass
    return None

def sampling_rate_hz(chan: Channel) -> Optional[float]:
    """Return channel sample rate in Hz if available."""
    try:
        return float(chan.sample_rate) if chan.sample_rate is not None else None
    except Exception:
        return None

def inventory_to_dataframe(inv: Inventory, source_file: str) -> pd.DataFrame:
    """Convert ObsPy Inventory to per-channel-epoch DataFrame."""
    rows: List[Dict[str, Any]] = []
    for net in inv:
        for sta in net:
            for cha in sta:
                rows.append({
                    "network": net.code,
                    "station": sta.code,
                    "location": cha.location_code or "",
                    "channel": cha.code,
                    "starttime": cha.start_date.isoformat() if cha.start_date else None,
                    "endtime": cha.end_date.isoformat() if cha.end_date else None,
                    "latitude": float(cha.latitude) if cha.latitude is not None else float(sta.latitude),
                    "longitude": float(cha.longitude) if cha.longitude is not None else float(sta.longitude),
                    "elevation_m": float(cha.elevation) if cha.elevation is not None else float(sta.elevation),
                    "sampling_rate_hz": sampling_rate_hz(cha),
                    "sensitivity": overall_sensitivity(cha),
                    "source_file": source_file,
                })
    return pd.DataFrame(rows)

import pandas as pd
from pathlib import Path

KEY_COLS = [
    "network", "station", "location", "channel",
    "starttime", "endtime",
    "elevation_m", "sampling_rate_hz", "sensitivity",
]

def _prep_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and round floats for robust equality."""
    out = df.copy()

    # Ensure presence of all key columns
    for c in KEY_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    # Coerce numerics (safe for missing)
    for numcol in ["elevation_m", "sampling_rate_hz", "sensitivity"]:
        out[numcol] = pd.to_numeric(out[numcol], errors="coerce")

    # Round to stabilize tiny differences (adjust if needed)
    out["elevation_m"] = out["elevation_m"].round(3)
    out["sampling_rate_hz"] = out["sampling_rate_hz"].round(6)
    out["sensitivity"] = out["sensitivity"].round(6)

    # Fill NA with a sentinel for exact matching (strings only)
    for c in ["location", "starttime", "endtime"]:
        out[c] = out[c].fillna("")

    return out[KEY_COLS].drop_duplicates()

def compare_stationxml_rows(csv_path: str | Path):
    """
    From a combined CSV (with 'source_file' column), report rows
    that are present in one StationXML but not the other, matching on KEY_COLS.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "source_file" not in df.columns:
        raise ValueError("CSV must contain a 'source_file' column.")

    # Identify the two sources
    src_a = "MV.xml"
    src_b = "MontserratDigitalSeismicNetwork.xml"

    if not ((df["source_file"] == src_a).any() and (df["source_file"] == src_b).any()):
        raise ValueError(f"Expected both '{src_a}' and '{src_b}' in source_file column.")

    a_raw = df[df["source_file"] == src_a].copy()
    b_raw = df[df["source_file"] == src_b].copy()

    A = _prep_keys(a_raw)
    B = _prep_keys(b_raw)

    # Anti-joins (outer merge with indicator)
    a_not_in_b = (
        A.merge(B, on=KEY_COLS, how="left", indicator=True)
         .query("_merge == 'left_only'")
         .drop(columns=["_merge"])
    )

    b_not_in_a = (
        B.merge(A, on=KEY_COLS, how="left", indicator=True)
         .query("_merge == 'left_only'")
         .drop(columns=["_merge"])
    )

    # Pretty print summaries
    print(f"\n[DIFF] Rows in {src_a} but NOT in {src_b}: {len(a_not_in_b)}")
    if not a_not_in_b.empty:
        # Show a few examples
        print(a_not_in_b.head(20).to_string(index=False))

    print(f"\n[DIFF] Rows in {src_b} but NOT in {src_a}: {len(b_not_in_a)}")
    if not b_not_in_a.empty:
        print(b_not_in_a.head(20).to_string(index=False))

    # Return the dataframes if you want to save/export them
    return a_not_in_b, b_not_in_a

from pathlib import Path
from typing import Optional, Tuple, Dict
import math
import pandas as pd
from obspy import read_inventory
from obspy.core.inventory import Inventory, Network, Station
from flovopy.stationmetadata.build import merge_inventories, sanitize_inventory_units
from flovopy.stationmetadata.utils import validate_inventory

# --- Inputs / outputs (edit as needed) ---
IN_XML  = Path("/Users/thompsong/Dropbox/MontserratDigitalSeismicNetwork.xml")
IN_CSV  = Path("/mnt/data/merged_pts_hyp_trimmed.csv")  # <- your attached CSV path
OUT_XML = IN_XML.with_name(IN_XML.stem + "_fixed_coords.xml")

# Aliases (normalize legacy names to station codes used in StationXML)
ALIASES = {
    "MWZH": "MWZ",
    "MGHZ": "MGH",
    # Add more if needed...
}

def _norm_code(s: str) -> str:
    """Uppercase, strip, and apply simple alias mapping."""
    if s is None:
        return ""
    s2 = s.strip().upper()
    return ALIASES.get(s2, s2)

def _get_float(x) -> Optional[float]:
    """Convert to float or return None if missing/NaN/blank."""
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def load_coord_table(csv_path: Path) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    """
    Build a dict: station_code -> (lat, lon, elev_m) from CSV.
    Accepts either columns [lat, lon] or [latitude, longitude].
    Elevation column expected as 'elevation_m' (if present).
    Uses 'Site' or 'station' column as the station identifier; prefers 'Site'.
    """
    df = pd.read_csv(csv_path)

    # Choose station label column
    station_col = None
    for cand in ("Site", "station", "Station"):
        if cand in df.columns:
            station_col = cand
            break
    if station_col is None:
        raise ValueError("CSV must have a 'Site' or 'station' column.")

    # Find latitude/longitude columns (support both naming conventions)
    lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
    if lat_col is None or lon_col is None:
        raise ValueError("CSV must contain lat/lon or latitude/longitude columns.")

    elev_col = "elevation_m" if "elevation_m" in df.columns else None

    lut: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for _, row in df.iterrows():
        code_raw = row.get(station_col, "")
        code = _norm_code(str(code_raw))
        if not code:
            continue

        lat = _get_float(row.get(lat_col))
        lon = _get_float(row.get(lon_col))
        elev = _get_float(row.get(elev_col)) if elev_col else None

        # Skip entirely empty coordinate rows
        if lat is None and lon is None and elev is None:
            continue

        # Keep the last non-null values if duplicates arise
        prev = lut.get(code, (None, None, None))
        lat = lat if lat is not None else prev[0]
        lon = lon if lon is not None else prev[1]
        elev = elev if elev is not None else prev[2]

        lut[code] = (lat, lon, elev)

    return lut

def update_inventory_coords(inv: Inventory, coord_lut: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]) -> Inventory:
    """
    For each station in the inventory, if there's a CSV match:
      - Update station latitude/longitude/elevation (when provided).
      - Update all channel latitude/longitude/elevation (when provided).
    Then for any CSV station not present in the inventory, add a new Station
    (no channels) under the first network (default "MV" if none are present).
    """
    # Build a set of existing station codes (by network)
    present = set()
    for net in inv:
        for sta in net:
            present.add(_norm_code(sta.code))

    # 1) Update existing
    for net in inv:
        for sta in net:
            code = _norm_code(sta.code)
            lat, lon, elev = coord_lut.get(code, (None, None, None))

            # Update Station coords if given
            if lat is not None:
                sta.latitude = float(lat)
            if lon is not None:
                sta.longitude = float(lon)
            if elev is not None:
                sta.elevation = float(elev)

            # Update all channels at that station (keep them consistent)
            for cha in sta.channels:
                if lat is not None:
                    cha.latitude = float(lat)
                if lon is not None:
                    cha.longitude = float(lon)
                if elev is not None:
                    cha.elevation = float(elev)

    # 2) Add CSV sites not present
    # Decide which network to add to: use existing 'MV' if present, else first, else create one.
    mv_net: Optional[Network] = None
    if len(inv.networks) > 0:
        for net in inv.networks:
            if net.code.upper() == "MV":
                mv_net = net
                break
        if mv_net is None:
            mv_net = inv.networks[0]
    else:
        mv_net = Network(code="MV")
        inv.networks.append(mv_net)

    added = 0
    for code, (lat, lon, elev) in coord_lut.items():
        if code in present:
            continue
        # Add a bare Station (no channels). This is valid in ObsPy.
        # If you need at least one channel later, you can append one.
        sta = Station(
            code=code,
            latitude=float(lat) if lat is not None else 0.0,
            longitude=float(lon) if lon is not None else 0.0,
            elevation=float(elev) if elev is not None else 0.0,
        )
        mv_net.stations.append(sta)
        added += 1

    print(f"[INFO] Updated coordinates on existing stations.")
    print(f"[INFO] Added {added} new station(s) from CSV that weren’t present in StationXML.")
    return inv

def main():
    frames = []
    for f in FILES:
        print(f"[INFO] Reading {f}")
        inv = read_inventory(str(f))
        frames.append(inventory_to_dataframe(inv, f.name))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(
        by=["network", "station", "location", "channel", "starttime"]
    ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {len(df)} rows to {OUT_CSV}")


    a_not_in_b, b_not_in_a = compare_stationxml_rows(
        "/Users/glennthompson/Dropbox/BRIEFCASE/SSADenver/stations_channels_table.csv"
    )

    # Optionally save the diffs:
    #outdir = Path("/Users/glennthompson/Dropbox/BRIEFCASE/SSADenver")
    #a_not_in_b.to_csv(outdir / "diff_MV_not_in_MDSN.csv", index=False)
    #b_not_in_a.to_csv(outdir / "diff_MDSN_not_in_MV.csv", index=False) 
    # 
   # --- Inputs / outputs (edit as needed) ---
    IN_XML  = Path("/Users/glennthompson/Dropbox/MontserratDigitalSeismicNetwork.xml")
    IN_CSV  = Path("/Users/glennthompson/Developer/flovopy/flovopy/dem/merged_pts_hyp_trimmed.csv")  # <- your attached CSV path
    OUT_XML = IN_XML.with_name(IN_XML.stem + "_fixed_coords.xml")

    # Aliases (normalize legacy names to station codes used in StationXML)
    ALIASES = {
        "MWZH": "MWZ",
        "MGHZ": "MGH",
        # Add more if needed...
    }

    print(f"[READ] StationXML: {IN_XML}")
    inv = read_inventory(str(IN_XML))

    print(f"[READ] CSV: {IN_CSV}")
    coord_lut = load_coord_table(IN_CSV)
    print(f"[CSV] Parsed {len(coord_lut)} station coordinate rows.")

    inv2 = update_inventory_coords(inv, coord_lut)
    inv2 = merge_inventories(inv2)
    sanitize_inventory_units(inv2)
    validate_inventory(inv2, verbose=True)

    OUT_XML.parent.mkdir(parents=True, exist_ok=True)
    inv2.write(str(OUT_XML), format="STATIONXML")
    print(f"[WRITE] Wrote fixed StationXML: {OUT_XML}")   

if __name__ == "__main__":
    main()