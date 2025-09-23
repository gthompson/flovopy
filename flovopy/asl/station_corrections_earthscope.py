#!/usr/bin/env python3
"""
Station Corrections Tool (FLOVOpy-compatible)

Purpose
-------
Given an ObsPy Inventory, this tool will:
  A) Query IRIS/EarthScope FDSN for regional earthquakes 100–300 km away from the network centroid.
  B) For each event, fetch **short event‑based waveform windows directly from IRIS/EarthScope** (remote mode) *or* segment from a **local SDS** archive (local mode).
  C) Preprocess the data using FLOVOpy's `preprocess_stream()` (detrend, remove instrument response, etc.).
  D) Compute per-station amplitude residuals vs the network median for each event and aggregate across events to estimate station corrections.

Alternative Mode (Remote FDSN)
-------------------------------
If your dataset is hosted at IRIS/EarthScope (i.e., you do not have a local SDS archive), you can:
  1) Define a region (lat/lon bounding box or center + radius) and time range.
  2) Download a fresh Inventory for that region/time range via FDSN.
  3) **Request short windows per event directly from FDSN** using theoretical arrivals (TauP) to minimize bandwidth.

Output
------
A CSV of station corrections (by SEED id) and a JSON sidecar with per-event residuals.
------
A CSV of station corrections (by SEED id) and a JSON sidecar with per-event residuals.

Notes
-----
* This script assumes FLOVOpy is installed and importable, providing:
    - from flovopy.sds.sds import SDSobj
    - from flovopy.core.preprocess import preprocess_stream
* IRIS/EarthScope FDSN services are used via ObsPy's Client('IRIS') or Client('EARTHSCOPE') if available.
* Amplitude measure is robust: log10 of max absolute, computed after preprocessing and bandpass filtering.
  You can adapt to peak displacement/velocity/acceleration as needed.

CLI Examples
------------
# Local SDS mode (recommended if you have a continuous archive):
python station_corrections_tool.py \
  --inventory inv.xml \
  --sds-root /data/SDS \
  --start 2024-01-01 --end 2024-12-31 \
  --min-km 100 --max-km 300 \
  --channels 'BH?,HH?' \
  --output station_corrections.csv

# Remote FDSN mode (no local SDS): fetch short windows per event
python station_corrections_tool.py \
  --fdsn-region "16.75,-62.18,250" \
  --start 2024-01-01 --end 2024-12-31 \
  --min-km 100 --max-km 300 \
  --channels 'BH?,HH?' \
  --remote-waveforms \
  --arrival-window "preP=20,postS=80" \
  --output station_corrections.csv

"""
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read_inventory, Stream
from obspy.clients.fdsn import Client
from obspy.core.inventory import Inventory, Network, Station
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
from obspy.taup import TauPyModel
from obspy.signal.filter import bandpass

# FLOVOpy expected imports (adjust if your module paths differ)
try:
    from flovopy.sds.sds import SDSobj
    from flovopy.core.preprocess import preprocess_stream
except Exception as e:
    # Allow running some parts without FLOVOpy for testing, but warn loudly.
    SDSobj = None
    preprocess_stream = None
    print("[WARN] Could not import FLOVOpy modules: ", e, file=sys.stderr)


# -----------------------
# Data classes & helpers
# -----------------------

@dataclass
class Region:
    lat: float
    lon: float
    radius_deg: float  # degrees


def parse_region(arg: str) -> Region:
    """Parse --fdsn-region 'lat, lon, radius_km' -> Region in degrees."""
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise ValueError("--fdsn-region expects 'lat, lon, radius_km'")
    lat, lon, r_km = float(parts[0]), float(parts[1]), float(parts[2])
    return Region(lat=lat, lon=lon, radius_deg=r_km / 111.19)


def centroid_of_inventory(inv: Inventory) -> Tuple[float, float]:
    lats, lons = [], []
    for net in inv.networks:
        for sta in net.stations:
            lats.append(sta.latitude)
            lons.append(sta.longitude)
    if not lats:
        raise ValueError("Inventory has no stations")
    return float(np.mean(lats)), float(np.mean(lons))


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    return d / 1000.0


def pick_events_in_range(
    client: Client,
    center_lat: float,
    center_lon: float,
    t0: UTCDateTime,
    t1: UTCDateTime,
    min_km: float,
    max_km: float,
    min_mag: Optional[float] = None,
    max_mag: Optional[float] = None,
) -> List[dict]:
    """Query FDSN for events whose epicentral distance from (center_lat, center_lon)
    lies between [min_km, max_km]. Returns a list of dicts with basic fields.
    """
    minr = min_km / 111.19
    maxr = max_km / 111.19
    cat = client.get_events(
        starttime=t0, endtime=t1,
        latitude=center_lat, longitude=center_lon,
        minradius=minr, maxradius=maxr,
        minmagnitude=min_mag, maxmagnitude=max_mag,
        orderby="time-asc",
    )
    events = []
    for ev in cat:
        ot = ev.preferred_origin() or ev.origins[0]
        mag = (ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None))
        events.append(
            {
                "time": ot.time,
                "lat": ot.latitude,
                "lon": ot.longitude,
                "depth_km": (ot.depth or 0) / 1000.0,
                "mag": mag.mag if mag else None,
                "event_id": str(ev.resource_id),
            }
        )
    return events


def seeds_from_inventory(inv: Inventory, channel_glob: str = "BH?,HH?") -> List[str]:
    """Return a list of SEED ids NET.STA.LOC.CHA matching the channel_glob patterns.
    channel_glob may be a comma-separated list of globs.
    """
    import fnmatch

    patterns = [p.strip() for p in channel_glob.split(",") if p.strip()]
    out = []
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                code = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                if any(fnmatch.fnmatch(cha.code, pat) for pat in patterns):
                    out.append(code)
    return sorted(set(out))


def network_for_seed(seed: str) -> Tuple[str, str, str, str]:
    net, sta, loc, cha = seed.split(".")
    return net, sta, loc, cha


# -----------------------
# Waveform retrieval
# -----------------------

def read_event_waveforms_remote_bulk(
    client: Client,
    seeds: List[str],
    event: dict,
    inv: Inventory,
    model: TauPyModel,
    preP: float = 20.0,
    postS: float = 80.0,
) -> Dict[str, Stream]:
    """Fetch short windows around theoretical arrivals for each SEED using FDSN bulk.
    Returns a dict seed -> Stream (may be empty if unavailable)."""
    # Map station coordinates for distance/arrival estimates
    sta_coords: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for net in inv.networks:
        for sta in net.stations:
            sta_coords[(net.code, sta.code)] = (sta.latitude, sta.longitude, getattr(sta, 'elevation', 0.0))

    ot: UTCDateTime = event["time"]
    ev_lat, ev_lon, ev_depth_km = event["lat"], event["lon"], event.get("depth_km", 10.0)

    bulk: List[Tuple[str, str, str, str, UTCDateTime, UTCDateTime]] = []
    per_seed_windows: Dict[str, Tuple[UTCDateTime, UTCDateTime]] = {}

    for seed in seeds:
        net, sta, loc, cha = seed.split('.')
        sc = sta_coords.get((net, sta))
        if sc is None:
            continue
        slat, slon, _ = sc
        d_km = distance_km(ev_lat, ev_lon, slat, slon)
        # Skip stations far outside typical regional range just in case
        if d_km < 50 or d_km > 800:
            continue
        # Theoretical P and S
        try:
            arrivals = model.get_travel_times(source_depth_in_km=ev_depth_km, distance_in_km=d_km, phase_list=['p','P','s','S'])
            tP = None
            tS = None
            for arr in arrivals:
                if tP is None and arr.phase.name.upper().startswith('P'):
                    tP = arr.time
                if tS is None and arr.phase.name.upper().startswith('S'):
                    tS = arr.time
            if tP is None:
                tP = d_km / 6.0  # crude fallback ~6 km/s
            if tS is None:
                tS = d_km / 3.5  # crude fallback ~3.5 km/s
        except Exception:
            tP = d_km / 6.0
            tS = d_km / 3.5
        start = ot + (tP - preP)
        end = ot + (tS + postS)
        per_seed_windows[seed] = (start, end)
        bulk.append((net, sta, loc, cha, start, end))

    st_by_seed: Dict[str, Stream] = {s: Stream() for s in per_seed_windows.keys()}
    if not bulk:
        return st_by_seed

    # Use get_waveforms_bulk with retries
    requests = []
    for (net, sta, loc, cha, start, end) in bulk:
        requests.append((net, sta, loc, cha, start, end))

    try:
        st = client.get_waveforms_bulk(requests, attach_response=False)
    except Exception as e:
        logging.debug(f"bulk fetch failed; falling back per-trace: {e}")
        st = Stream()
        for (net, sta, loc, cha, start, end) in requests:
            try:
                st += client.get_waveforms(net, sta, loc, cha, start, end, attach_response=False)
            except Exception:
                continue

    # Split back by seed
    for tr in st:
        seed = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"
        if seed in st_by_seed:
            st_by_seed[seed] += tr

    # Attach response if we have it
    try:
        st.attach_response(inv)
    except Exception:
        pass

    return st_by_seed

# -----------------------
# Preprocess and amplitude
# -----------------------

def preprocess_with_flovopy(st: Stream, inv: Inventory) -> Optional[Stream]:
    if preprocess_stream is None:
        raise RuntimeError("FLOVOpy preprocess_stream not available")
    try:
        # Example kwargs—tune to your preprocess defaults
        stp = preprocess_stream(
            st.copy(),
            inventory=inv,
            detrend=True,
            taper=True,
            taper_fraction=0.05,
            remove_response=True,
            output="VEL",  # choose DIS/VEL/ACC depending on your preference
            pre_filt=(0.5, 0.8, 20.0, 25.0),  # example prefilter for response removal
            water_level=60,
            demean=True,
        )
        # Optional extra bandpass to harmonize spectral content across stations
        stp.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)
        return stp
    except Exception as e:
        logging.debug(f"preprocess failed: {e}")
        return None


def robust_log_amplitude(st: Stream) -> Optional[float]:
    """Return log10(max abs) across all traces after preprocessing."""
    try:
        if not st:
            return None
        data_max = 0.0
        for tr in st:
            if tr.data is None or len(tr.data) == 0:
                continue
            data_max = max(data_max, float(np.nanmax(np.abs(tr.data))))
        if data_max <= 0:
            return None
        return float(np.log10(data_max))
    except Exception:
        return None


# -----------------------
# Corrections core
# -----------------------

def event_window(ot: UTCDateTime, pad_before: float = 20.0, pad_after: float = 120.0) -> Tuple[UTCDateTime, UTCDateTime]:
    return ot - pad_before, ot + pad_after


def compute_station_corrections(
    inv: Inventory,
    events: List[dict],
    read_waveforms_fn,
    preprocess_inv: Inventory,
    channel_glob: str = "BH?,HH?",
) -> Tuple[pd.DataFrame, dict]:
    """
    For each event and each station/channel, compute log10 amplitude residuals vs network median,
    then aggregate (median) residual per station as correction.

    Returns
    -------
    corrections_df: columns = [network, station, location, channel, correction_log10, n_events]
    residuals_by_event: dict[event_id] -> { seed: residual }
    """
    seeds = seeds_from_inventory(inv, channel_glob=channel_glob)
    residuals_by_event: Dict[str, Dict[str, float]] = {}
    per_station_residuals: Dict[str, List[float]] = {seed: [] for seed in seeds}

    for ev in events:
        ev_id = ev["event_id"]
        ot: UTCDateTime = ev["time"]
        t0, t1 = event_window(ot)
        amp_by_seed: Dict[str, float] = {}

        for seed in seeds:
            st = read_waveforms_fn(seed, t0, t1)
            if st is None or len(st) == 0:
                continue
            stp = preprocess_with_flovopy(st, preprocess_inv)
            if stp is None or len(stp) == 0:
                continue
            a = robust_log_amplitude(stp)
            if a is not None:
                amp_by_seed[seed] = a

        if len(amp_by_seed) < 3:
            logging.info(f"Event {ev_id}: insufficient data across stations ({len(amp_by_seed)})")
            continue

        # Network median and residuals
        amps = np.array(list(amp_by_seed.values()))
        net_med = float(np.median(amps))
        res = {seed: (amp - net_med) for seed, amp in amp_by_seed.items()}
        residuals_by_event[ev_id] = res
        for seed, r in res.items():
            per_station_residuals[seed].append(r)

    rows = []
    for seed, vals in per_station_residuals.items():
        if not vals:
            continue
        net, sta, loc, cha = network_for_seed(seed)
        med_res = float(np.median(vals))
        rows.append({
            "network": net,
            "station": sta,
            "location": loc,
            "channel": cha,
            "seed": seed,
            "correction_log10": med_res,
            "n_events": len(vals),
        })
    df = pd.DataFrame(rows).sort_values(["network", "station", "channel"]).reset_index(drop=True)
    return df, residuals_by_event


# -----------------------
# Inventory acquisition for remote mode
# -----------------------

def download_inventory_region(
    client: Client,
    region: Region,
    t0: UTCDateTime,
    t1: UTCDateTime,
    channel_glob: str,
) -> Inventory:
    # Map globs like 'BH?,HH?' into ObsPy channel list requests by making multiple calls
    inv_total = None
    for pat in [p.strip() for p in channel_glob.split(',') if p.strip()]:
        inv = client.get_stations(
            latitude=region.lat, longitude=region.lon, maxradius=region.radius_deg,
            starttime=t0, endtime=t1,
            level="response", channel=pat
        )
        inv_total = inv if inv_total is None else inv_total + inv
    return inv_total


# -----------------------
# Main
# -----------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute station corrections using regional events (100–300 km)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--inventory", help="Path to local Inventory XML/StationXML file")
    src.add_argument("--fdsn-region", help="lat,lon,radius_km to download Inventory via FDSN (remote mode)")

    ap.add_argument("--sds-root", help="Root of local SDS archive (for local SDS mode)")
    ap.add_argument("--remote-waveforms", action="store_true", help="Fetch short event windows via FDSN instead of local SDS")

    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--min-km", type=float, default=100.0, help="Min epicentral distance (km)")
    ap.add_argument("--max-km", type=float, default=300.0, help="Max epicentral distance (km)")
    ap.add_argument("--min-mag", type=float, default=None, help="Optional min magnitude filter")
    ap.add_argument("--max-mag", type=float, default=None, help="Optional max magnitude filter")
    ap.add_argument("--channels", default="BH?,HH?", help="Channel glob(s), comma-separated")
    ap.add_argument("--arrival-window", default="preP=20,postS=80", help="Remote mode: window around P..S (e.g. 'preP=10,postS=60')")
    ap.add_argument("--phase-model", default="ak135", help="TauP model for arrivals (ak135, iasp91)")
    ap.add_argument("--output", default="station_corrections.csv", help="Output CSV path")
    ap.add_argument("--residuals-json", default="station_residuals_by_event.json", help="Per-event residuals JSON")
    ap.add_argument("--log", default="INFO", help="Logging level")

    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(levelname)s %(message)s")

    t0 = UTCDateTime(f"{args.start}T00:00:00")
    t1 = UTCDateTime(f"{args.end}T23:59:59")

    # Choose FDSN client (prefer EarthScope, then IRIS); fall back to USGS for events if needed.
    try:
        fdsn = Client("EARTHSCOPE")
    except Exception:
        try:
            fdsn = Client("IRIS")
        except Exception:
            fdsn = Client("USGS")

    try:
        fdsn = Client("IRIS")
    except Exception:
        fdsn = Client("USGS")

    if args.inventory:
        inv = read_inventory(args.inventory)
    else:
        region = parse_region(args.fdsn_region)
        inv = download_inventory_region(fdsn, region, t0, t1, args.channels)

    # Event selection around the inventory centroid
    cen_lat, cen_lon = centroid_of_inventory(inv)
    events = pick_events_in_range(
        client=fdsn,
        center_lat=cen_lat,
        center_lon=cen_lon,
        t0=t0,
        t1=t1,
        min_km=args.min_km,
        max_km=args.max_km,
        min_mag=args.min_mag,
        max_mag=args.max_mag,
    )
    if not events:
        logging.error("No events found in specified distance/time range.")
        return 2
    logging.info(f"Selected {len(events)} regional events between {args.min_km}-{args.max_km} km.")

    # Choose waveform reader
    seeds = seeds_from_inventory(inv, channel_glob=args.channels)

    # Parse arrival window
    preP, postS = 20.0, 80.0
    try:
        parts = dict(kv.split('=') for kv in args.arrival_window.split(',') if '=' in kv)
        preP = float(parts.get('preP', preP))
        postS = float(parts.get('postS', postS))
    except Exception:
        pass
    model = TauPyModel(args.phase_model)

    if args.remote_waveforms:
        def reader_map(ev: dict) -> Dict[str, Stream]:
            return read_event_waveforms_remote_bulk(
                client=fdsn,
                seeds=seeds,
                event=ev,
                inv=inv,
                model=model,
                preP=preP,
                postS=postS,
            )
        def reader(seed: str, t0_, t1_):
            # Not used in remote bulk mode; placeholder to satisfy signature
            return Stream()
    else:
        if not args.sds_root:
            ap.error("--sds-root is required when not using --remote-waveforms")
        sds_root = Path(args.sds_root)
        if not sds_root.exists():
            ap.error(f"SDS root does not exist: {sds_root}")
        def reader(seed: str, t0_, t1_):
            return read_waveforms_local_sds(sds_root, seed, t0_, t1_)

    # Compute corrections
    if args.remote_waveforms:
        # Remote mode: fetch per-event window maps, then compute
        events = pick_events_in_range(
            client=fdsn,
            center_lat=cen_lat,
            center_lon=cen_lon,
            t0=t0,
            t1=t1,
            min_km=args.min_km,
            max_km=args.max_km,
            min_mag=args.min_mag,
            max_mag=args.max_mag,
        )
        if not events:
            logging.error("No events found in specified distance/time range.")
            return 2
        logging.info(f"Selected {len(events)} regional events between {args.min_km}-{args.max_km} km.")

        residuals_by_event: Dict[str, Dict[str, float]] = {}
        per_station_residuals: Dict[str, List[float]] = {seed: [] for seed in seeds}

        for ev in events:
            ev_id = ev["event_id"]
            st_map = read_event_waveforms_remote_bulk(fdsn, seeds, ev, inv, model, preP, postS)
            amp_by_seed: Dict[str, float] = {}
            for seed, st in st_map.items():
                if st is None or len(st) == 0:
                    continue
                stp = preprocess_with_flovopy(st, inv)
                if stp is None or len(stp) == 0:
                    continue
                a = robust_log_amplitude(stp)
                if a is not None:
                    amp_by_seed[seed] = a
            if len(amp_by_seed) < 3:
                logging.info(f"Event {ev_id}: insufficient data across stations ({len(amp_by_seed)})")
                continue
            amps = np.array(list(amp_by_seed.values()))
            net_med = float(np.median(amps))
            res = {seed: (amp - net_med) for seed, amp in amp_by_seed.items()}
            residuals_by_event[ev_id] = res
            for seed, r in res.items():
                per_station_residuals[seed].append(r)

        rows = []
        for seed, vals in per_station_residuals.items():
            if not vals:
                continue
            net, sta, loc, cha = network_for_seed(seed)
            med_res = float(np.median(vals))
            rows.append({
                "network": net,
                "station": sta,
                "location": loc,
                "channel": cha,
                "seed": seed,
                "correction_log10": med_res,
                "n_events": len(vals),
            })
        corrections_df = pd.DataFrame(rows).sort_values(["network", "station", "channel"]).reset_index(drop=True)
        residuals = residuals_by_event
    else:
        corrections_df, residuals = compute_station_corrections(
            inv=inv,
            events=events,
            read_waveforms_fn=reader,
            preprocess_inv=inv,
            channel_glob=args.channels,
        )

    if corrections_df.empty:
        logging.error("No corrections could be computed (no usable waveforms preprocessed).")
        return 3

    # Save outputs
    out_csv = Path(args.output)
    corrections_df.to_csv(out_csv, index=False)
    out_json = Path(args.residuals_json)
    with open(out_json, "w") as f:
        json.dump(residuals, f, default=str, indent=2)

    logging.info(f"Wrote {out_csv} and {out_json}")
    print(corrections_df.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
