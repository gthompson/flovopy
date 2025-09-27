#!/usr/bin/env python3
"""
isc_waveform_harvester.py

Given an inventory (StationXML) or a station search spec, a date range, and a
distance band, download:
  1) ISC catalog events within the window & distance band (around a center),
  2) waveforms for each inventory channel per event, saving to YYYY/MM folders.

Examples
--------
# Use an existing StationXML, distance band 100-1000 km around inventory centroid
python isc_waveform_harvester.py \
  --start 1996-09-01 --end 2008-10-01 \
  --inv MV.xml \
  --min-km 100 --max-km 1000 \
  --out-quakeml ISC_raw_quakeml.xml \
  --wave-root /data/waveforms

# Discover stations in a box if no StationXML is provided
python isc_waveform_harvester.py \
  --start 2000-01-01 --end 2001-01-01 \
  --bbox 14 -65 19 -60 \
  --net MV --chan '[BESHCD]H?' \
  --min-km 100 --max-km 800 \
  --out-inventory stations.xml \
  --out-quakeml ISC_2000.xml \
  --wave-root ./wf_2000

Notes
-----
* ISC provides *catalog*; for *waveforms*, this script queries a list of FDSN
  waveform providers (default: IRIS/EarthScope) per channel. You can add more.
* Travel-time windows are crude but robust:
    tP = dist_km / Vp,  tS = dist_km / Vs
    window = [tP - prepad, tS + postpad] relative to origin time
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, List, Sequence, Iterable, Dict

import numpy as np
import pandas as pd

from obspy import UTCDateTime, read_inventory
from obspy.core.event import Catalog, read_events
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees, degrees2kilometers

# ------------------------
# Defaults / configuration
# ------------------------

DEFAULT_ISC_MINMAG = 2.0
DEFAULT_CHUNK_DAYS = 366
DEFAULT_MIN_SPAN_D = 14
DEFAULT_RETRIES    = 3
DEFAULT_BACKOFF_S  = 3.0

# Waveform providers (ordered); feel free to add more
DEFAULT_WAVE_PROVIDERS = ["IRIS"]

# Travel time window
DEFAULT_VP = 8.0  # km/s
DEFAULT_VS = 5.0  # km/s
DEFAULT_TP_PRE  = 30.0  # seconds before P
DEFAULT_TS_POST = 60.0  # seconds after S

# ------------------------
# Helpers
# ------------------------

def _centroid_from_inventory(inv) -> Tuple[float, float]:
    lats, lons = [], []
    for net in inv.networks:
        for sta in net.stations:
            if sta.latitude is not None and sta.longitude is not None:
                lats.append(sta.latitude)
                lons.append(sta.longitude)
    if not lats:
        raise ValueError("Inventory has no station lat/lon")
    return float(np.mean(lats)), float(np.mean(lons))

def _distance_km(lat1, lon1, lat2, lon2) -> float:
    d_m, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    return d_m / 1000.0

def _fetch_isc_chunked(t0: UTCDateTime, t1: UTCDateTime, *,
                       lat: float, lon: float, radius_km: float, minmag: float,
                       max_span_days: int = DEFAULT_CHUNK_DAYS,
                       min_span_days: int = DEFAULT_MIN_SPAN_D,
                       max_retries: int = DEFAULT_RETRIES,
                       backoff_sec: float = DEFAULT_BACKOFF_S,
                       verbose: bool = True) -> Catalog:
    """
    Robust ISC fetch: yearly chunks, retries on errors, and recursive bisection
    on persistent failures. Returns an ObsPy Catalog.
    """
    client = FDSNClient("ISC")
    maxradius = kilometer2degrees(radius_km)

    def span_days(a, b): return (b - a) / 86400.0

    def get_span(a: UTCDateTime, b: UTCDateTime) -> Catalog:
        span = span_days(a, b)
        if span < 0.5:
            if verbose: print(f"[ISC] skip tiny span {a}→{b}")
            return Catalog()
        for attempt in range(1, max_retries + 1):
            try:
                if verbose:
                    print(f"[ISC] get_events {a.date} → {b.date} (maxradius={maxradius:.3f}°) try {attempt}/{max_retries}")
                cat = client.get_events(
                    starttime=a, endtime=b,
                    latitude=lat, longitude=lon,
                    maxradius=maxradius,
                    minmagnitude=minmag,
                    includearrivals=False,
                    orderby="time-asc",
                )
                return cat
            except FDSNNoDataException:
                return Catalog()
            except (FDSNException, Exception) as e:
                if attempt == max_retries:
                    if verbose:
                        print(f"[WARN] ISC failed {a}→{b}: {e}")
                    break
                time.sleep(backoff_sec * attempt)
        # bisect on failure
        if span > min_span_days:
            mid = a + (b - a) / 2.0
            left = get_span(a, mid)
            right = get_span(mid, b)
            left.extend(right)
            return left
        return Catalog()

    out = Catalog()
    step = t0
    while step < t1:
        step2 = min(step + max_span_days * 86400, t1)
        out.extend(get_span(step, step2))
        step = step2

    # de-duplicate by resource_id
    seen = set()
    dedup = Catalog()
    for ev in out:
        rid = getattr(getattr(ev, "resource_id", None), "id", None)
        if rid and rid in seen:
            continue
        if rid:
            seen.add(rid)
        dedup.events.append(ev)

    if verbose:
        print(f"[ISC] fetched {len(dedup)} events across chunk(s)")
    return dedup

def _catalog_to_df(cat: Catalog) -> pd.DataFrame:
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        if o is None or o.time is None or o.latitude is None or o.longitude is None:
            continue
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows.append({
            "isc_id": getattr(getattr(ev, "resource_id", None), "id", None),
            "time":   pd.to_datetime(o.time.datetime, utc=True),
            "lat":    float(o.latitude),
            "lon":    float(o.longitude),
            "depth_m": float(o.depth) if o.depth is not None else np.nan,
            "mag":    (float(m.mag) if (m and m.mag is not None) else np.nan),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def _select_events_by_distance(cat_df: pd.DataFrame, center_lat: float, center_lon: float,
                               min_km: float, max_km: float) -> pd.DataFrame:
    if cat_df.empty:
        return cat_df
    dist = np.array([
        _distance_km(r.lat, r.lon, center_lat, center_lon) for _, r in cat_df.iterrows()
    ], dtype=float)
    out = cat_df.copy()
    out["dist_km_center"] = dist
    mask = (dist >= float(min_km)) & (dist <= float(max_km))
    return out.loc[mask].reset_index(drop=True)

def _save_quakeml(cat: Catalog, outfile: str):
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    cat.write(outfile, format="QUAKEML")
    print(f"[OUT] wrote {outfile} ({len(cat)} events)")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _iter_inventory_seed_ids(inv, chan_selector: str) -> List[Tuple[str, str, str, str]]:
    """
    Yield (net, sta, loc, cha) for channels that match `chan_selector` (ObsPy-style, e.g. '[BES]H?').
    """
    out = []
    for net in inv:
        for sta in net:
            for cha in sta.channels:
                code = cha.code or ""
                if chan_selector and not FDSNClient._match(code, chan_selector):  # internal matcher exists; fallback:
                    # very simple fallback: if selector is blank we accept; else require exact match
                    # (custom pattern support could be added if needed)
                    pass
                # We'll just do a soft filter: if a selector is given and doesn't look like a glob, require equality
                if chan_selector and ("*" not in chan_selector and "?" not in chan_selector and "[" not in chan_selector):
                    if code != chan_selector:
                        continue
                out.append((net.code, sta.code, cha.location_code or "", cha.code))
    # de-dup
    return sorted(set(out))

def _get_inv_or_fetch(args) -> 'Inventory':
    if args.inv:
        inv = read_inventory(args.inv)
        print(f"[INV] loaded {args.inv}: {sum(len(net.stations) for net in inv)} stations")
        return inv

    # fetch via FDSN station service
    provider = args.station_provider
    cli = FDSNClient(provider)
    kw = dict(starttime=UTCDateTime(args.start), endtime=UTCDateTime(args.end),
              level="channel", includerestricted=False)
    geo = None
    if args.bbox and len(args.bbox) == 4:
        minlat, minlon, maxlat, maxlon = map(float, args.bbox)
        kw.update(dict(minlatitude=minlat, minlongitude=minlon,
                       maxlatitude=maxlat, maxlongitude=maxlon))
        geo = f"bbox=({minlat},{minlon},{maxlat},{maxlon})"
    elif args.circle and len(args.circle) == 3:
        clat, clon, r_km = float(args.circle[0]), float(args.circle[1]), float(args.circle[2])
        kw.update(dict(latitude=clat, longitude=clon, maxradius=kilometer2degrees(r_km)))
        geo = f"circle=({clat},{clon}, {r_km} km)"
    if args.net:
        kw["network"] = args.net
    if args.sta:
        kw["station"] = args.sta
    if args.chan:
        kw["channel"] = args.chan

    print(f"[INV] fetching from {provider} ({geo or 'global'})…")
    inv = cli.get_stations(**kw)
    print(f"[INV] fetched: {sum(len(net.stations) for net in inv)} stations, "
          f"{sum(len(ch.channels) for net in inv for ch in net.stations)} channels")
    if args.out_inventory:
        Path(args.out_inventory).parent.mkdir(parents=True, exist_ok=True)
        inv.write(args.out_inventory, format="STATIONXML")
        print(f"[OUT] wrote {args.out_inventory}")
    return inv

def _wave_window(ot: UTCDateTime, dist_km: float, vp: float, vs: float,
                 preP: float, postS: float) -> Tuple[UTCDateTime, UTCDateTime]:
    tP = dist_km / max(vp, 0.1)
    tS = dist_km / max(vs, 0.1)
    start = ot + (tP - preP)
    end   = ot + (tS + postS)
    if end <= start:
        end = start + 60.0
    return start, end

def _download_one_trace(providers: Sequence[str], net: str, sta: str, loc: str, cha: str,
                        t0: UTCDateTime, t1: UTCDateTime):
    last_err = None
    for prov in providers:
        try:
            cli = FDSNClient(prov)
            st = cli.get_waveforms(network=net, station=sta, location=loc or "",
                                   channel=cha, starttime=t0, endtime=t1,
                                   attach_response=False)
            return st
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("waveform fetch failed")

def main():
    ap = argparse.ArgumentParser(description="ISC catalog + waveform harvester")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")

    # Inventory source
    ap.add_argument("--inv", help="Path to StationXML; if omitted, fetch via FDSN")
    ap.add_argument("--out-inventory", help="If fetching, write StationXML here")
    ap.add_argument("--station-provider", default="IRIS", help="FDSN station service (default IRIS)")
    ap.add_argument("--net", help="FDSN network code (optional)")
    ap.add_argument("--sta", help="FDSN station code (optional)")
    ap.add_argument("--chan", default="[BESHCD]H?", help="Channel selector (ObsPy/regex-lite)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLAT","MINLON","MAXLAT","MAXLON"),
                    help="If no inv, fetch stations in bounding box")
    ap.add_argument("--circle", nargs=3, type=float, metavar=("LAT","LON","R_KM"),
                    help="If no inv, fetch stations in circle (km radius)")

    # Event selection
    ap.add_argument("--min-km", type=float, default=100.0, help="Min distance from center (km)")
    ap.add_argument("--max-km", type=float, default=1000.0, help="Max distance from center (km)")
    ap.add_argument("--center-lat", type=float, help="Override center latitude")
    ap.add_argument("--center-lon", type=float, help="Override center longitude")
    ap.add_argument("--isc-minmag", type=float, default=DEFAULT_ISC_MINMAG, help="ISC minimum magnitude")
    ap.add_argument("--raw-quakeml", required=True, help="Raw ISC catalog path (cache file)")
    ap.add_argument("--out-quakeml", required=True, help="Filtered ISC catalog path")

    # Waveform
    ap.add_argument("--wave-root", required=True, help="Root directory to save waveforms")
    ap.add_argument("--providers", nargs="*", default=DEFAULT_WAVE_PROVIDERS, help="Waveform providers in order")
    ap.add_argument("--vp", type=float, default=DEFAULT_VP, help="P velocity km/s")
    ap.add_argument("--vs", type=float, default=DEFAULT_VS, help="S velocity km/s")
    ap.add_argument("--preP", type=float, default=DEFAULT_TP_PRE, help="Seconds before P")
    ap.add_argument("--postS", type=float, default=DEFAULT_TS_POST, help="Seconds after S")
    ap.add_argument("--max-per-event", type=int, help="Optional cap on channels per event")

    # Fetch robustness
    ap.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    ap.add_argument("--min-span-days", type=int, default=DEFAULT_MIN_SPAN_D)
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    ap.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF_S)

    # Misc
    ap.add_argument("--log-csv", help="Write a CSV log of downloaded traces/events")
    ap.add_argument("--dry-run", action="store_true", help="Do not download waveforms, just list")

    args = ap.parse_args()

    t0 = UTCDateTime(args.start)
    t1 = UTCDateTime(args.end)
    if t1 <= t0:
        raise SystemExit("end must be after start")

    # Inventory
    inv = _get_inv_or_fetch(args)
    # Center
    if args.center_lat is not None and args.center_lon is not None:
        clat, clon = float(args.center_lat), float(args.center_lon)
        print(f"[CENTER] using user center: ({clat:.4f}, {clon:.4f})")
    else:
        clat, clon = _centroid_from_inventory(inv)
        print(f"[CENTER] inventory centroid: ({clat:.4f}, {clon:.4f})")

    # Seed ids to request
    seeds = _iter_inventory_seed_ids(inv, args.chan)
    if not seeds:
        print("[WARN] No channels found in inventory matching selector; exiting.")
        return

    # 1) Raw ISC catalog (cache once)
    raw_path = Path(args.raw_quakeml)
    if raw_path.exists():
        cat = read_events(str(raw_path))
        print(f"[ISC] loaded cached {raw_path} ({len(cat)} events)")
    else:
        cat = _fetch_isc_chunked(
            t0, t1, lat=clat, lon=clon, radius_km=args.max_km, minmag=args.isc_minmag,
            max_span_days=args.chunk_days, min_span_days=args.min_span_days,
            max_retries=args.retries, backoff_sec=args.backoff, verbose=True
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _save_quakeml(cat, str(raw_path))

    # 2) Filter by distance ring
    cat_df = _catalog_to_df(cat)
    if cat_df.empty:
        print("[WARN] ISC catalog empty after parse; nothing to do.")
        return
    cat_sel = _select_events_by_distance(cat_df, clat, clon, args.min_km, args.max_km)
    print(f"[CAT] selected {len(cat_sel)} / {len(cat_df)} events in {args.min_km}-{args.max_km} km ring")

    # 3) Save filtered catalog
    keep_ids = set(cat_sel["isc_id"].dropna().astype(str))
    pruned = Catalog([ev for ev in cat if getattr(getattr(ev, "resource_id", None), "id", None) in keep_ids])
    _save_quakeml(pruned, args.out_quakeml)

    # 4) Waveform harvesting
    wave_root = Path(args.wave_root)
    _ensure_dir(wave_root)

    logs: List[dict] = []
    for idx, row in cat_sel.iterrows():
        ev_id = str(row["isc_id"])
        ot = UTCDateTime(pd.Timestamp(row["time"]).to_pydatetime())
        ev_year = ot.year
        ev_month = ot.month

        # per station/channel window via station distance to event
        downloaded = 0
        for (net, sta, loc, cha) in seeds:
            # station coordinates from inventory
            try:
                sta_obj = inv.select(network=net, station=sta)[0][0]
                slat, slon = float(sta_obj.latitude), float(sta_obj.longitude)
            except Exception:
                continue
            dist_km = _distance_km(row["lat"], row["lon"], slat, slon)
            t0_req, t1_req = _wave_window(ot, dist_km, args.vp, args.vs, args.preP, args.postS)

            if args.dry_run:
                fpath = wave_root / f"{ev_year:04d}" / f"{ev_month:02d}" / f"{ev_id}_{net}.{sta}.{loc}.{cha}.mseed"
                print(f"[DRY] would fetch {net}.{sta}.{loc}.{cha} {t0_req}→{t1_req} → {fpath}")
                continue

            try:
                st = _download_one_trace(args.providers, net, sta, loc, cha, t0_req, t1_req)
            except Exception as e:
                logs.append(dict(event_id=ev_id, time=str(row["time"]),
                                 net=net, sta=sta, loc=loc, cha=cha,
                                 ok=False, error=str(e)))
                continue

            outdir = wave_root / f"{ev_year:04d}" / f"{ev_month:02d}"
            _ensure_dir(outdir)
            outpath = outdir / f"{ev_id}_{net}.{sta}.{loc}.{cha}.mseed"
            try:
                st.write(str(outpath), format="MSEED")
                logs.append(dict(event_id=ev_id, time=str(row["time"]),
                                 net=net, sta=sta, loc=loc, cha=cha,
                                 ok=True, error=""))
                downloaded += 1
            except Exception as e:
                logs.append(dict(event_id=ev_id, time=str(row["time"]),
                                 net=net, sta=sta, loc=loc, cha=cha,
                                 ok=False, error=f"write:{e}"))

            if args.max_per_event and downloaded >= args.max_per_event:
                break

    if args.log_csv:
        pd.DataFrame(logs).to_csv(args.log_csv, index=False)
        print(f"[OUT] wrote log {args.log_csv}")

    print("[✓] Done.")

if __name__ == "__main__":
    main()