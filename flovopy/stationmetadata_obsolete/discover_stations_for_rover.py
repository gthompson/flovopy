#!/usr/bin/env python3
"""
Summarize FDSN channel availability in a circular domain and emit a Rover request.

- Queries one or more FDSN providers (e.g., IRIS, Raspberry Shake).
- Returns a DataFrame of channels within <radius_km> of (lat, lon) that overlap the
  requested time window.
- Builds a Rover request file with per-(NET, STA, LOC, channel family) lines whose
  START/END come from the discovered availability windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Optional, Tuple, Dict

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# ----------------------- utilities -----------------------

def _make_client(prov: str, timeout: int = 60) -> Client:
    """
    Return an ObsPy FDSN Client for a provider name or base URL.
    Raspberry Shake now lives at https://data.raspberryshake.org/fdsnws/.
    We normalize anything 'raspberryshake.org' to that base and provide explicit mappings.
    """
    prov_str = str(prov).strip()

    # Convenience aliases → Raspberry Shake
    if prov_str.upper() in {"RASPISHAKE", "RASPBERRY_SHAKE", "AM"}:
        base = "https://data.raspberryshake.org/fdsnws"
        return Client(
            base_url=base,
            service_mappings={
                "station":    f"{base}/station/1/query",
                "dataselect": f"{base}/dataselect/1/query",
            },
            timeout=timeout,
        )

    # Any URL containing raspberryshake.org → normalize to /fdsnws with explicit mappings
    if "raspberryshake.org" in prov_str:
        base = "https://data.raspberryshake.org/fdsnws"
        return Client(
            base_url=base,
            service_mappings={
                "station":    f"{base}/station/1/query",
                "dataselect": f"{base}/dataselect/1/query",
            },
            timeout=timeout,
        )

    # Otherwise standard handling
    if prov_str.startswith(("http://", "https://")):
        return Client(base_url=prov_str, timeout=timeout)
    return Client(prov_str, timeout=timeout)

def _close_client(cli: Client) -> None:
    """
    Best-effort cleanup to avoid lingering threads at interpreter shutdown.
    Works with plain requests.Session and requests_futures FuturesSession.
    Safe to call even if attributes don’t exist.
    """
    sess = getattr(cli, "_session", None) or getattr(cli, "session", None)
    if sess is not None:
        # If it's a FuturesSession, shut down its executor
        executor = getattr(sess, "executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
        try:
            sess.close()
        except Exception:
            pass

def _km_to_deg(km: Optional[float]) -> Optional[float]:
    """Approx. convert kilometers to great-circle degrees."""
    if km is None:
        return None
    return km / 111.19  # mean Earth radius conversion

def _to_pandas_ts(x) -> pd.Timestamp:
    """
    Robustly convert ObsPy UTCDateTime / datetime / str / None to pandas Timestamp.
    Returns pandas.NaT on failure/None.
    """
    if x is None:
        return pd.NaT
    try:
        if isinstance(x, UTCDateTime):
            return pd.Timestamp(x.datetime)
    except Exception:
        pass
    return pd.to_datetime(x, errors="coerce")

def _loc_for_rover(loc: str) -> str:
    """Rover uses '--' for blank location codes."""
    if not loc or loc.strip() in {"", "--"}:
        return "--"
    return loc

def _loc_for_availability(loc: str) -> str:
    """For availability/dataselect, prefer '*' over '--' when blank."""
    if not loc or loc.strip() in {"", "--"}:
        return "*"
    return loc

def _chan_family(cha: str) -> str:
    """
    Collapse a specific 3+ char channel (e.g., BHZ/BHN/BHE/BH1/BH2) to a family like 'BH?'.
    If shorter/odd, return '*' as a safe wildcard.
    """
    cha = (cha or "").strip().upper()
    return f"{cha[:2]}?" if len(cha) >= 2 else "*"

def _merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge overlapping/adjacent intervals [(S,E),...] -> merged list."""
    xs = [(s, e) for (s, e) in intervals if (pd.notna(s) and pd.notna(e) and s < e)]
    if not xs:
        return []
    xs.sort()
    out = [xs[0]]
    for s, e in xs[1:]:
        ps, pe = out[-1]
        if s <= pe:  # overlap/adjacent
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out

def _interval_diff(a: list[tuple[pd.Timestamp, pd.Timestamp]],
                   b: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return A \ B where A and B are lists of (S,E).
    Assumes S<E. Returns list of non-overlapping (S,E) in A not covered by B.
    """
    A = _merge_intervals(a)
    B = _merge_intervals(b)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    j = 0
    for s, e in A:
        cur_s = s
        while j < len(B) and B[j][1] <= cur_s:
            j += 1
        k = j
        seg_s = cur_s
        while k < len(B) and B[k][0] < e:
            bs, be = B[k]
            if bs > seg_s:
                out.append((seg_s, min(e, bs)))
            if be >= e:
                seg_s = e
                break
            seg_s = max(seg_s, be)
            k += 1
        if seg_s < e:
            out.append((seg_s, e))
    return [(s, e) for s, e in out if s < e]

def _choose_winner(net_windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
                   net_priority: list[str] | None) -> str:
    """
    Pick the winner network in a co-located overlap group.
    Strategy: priority list first; then total coverage length; then lexical.
    net_windows: {NET: [(S,E), ...]} (already merged per NET)
    """
    rank = {n: i for i, n in enumerate(net_priority)} if net_priority else {}
    cov: dict[str, pd.Timedelta] = {}
    for net, ivals in net_windows.items():
        total = pd.Timedelta(0)
        for s, e in ivals:
            if pd.isna(s) or pd.isna(e):
                continue
            total += (e - s)
        cov[net] = total
    def keyf(n: str):
        return (rank.get(n, 10_000), -cov[n].total_seconds(), n)
    return sorted(net_windows.keys(), key=keyf)[0]

# ----------------------- optional pruning helper -----------------------

def filter_colocated_by_containment(
    summary: pd.DataFrame,
    *,
    coord_decimals: int = 4,
    default_start=None,
    default_end=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop stations fully contained in time by another station at (nearly) the same coordinates,
    for the same location code and channel family.

    Expects columns: ['seed_id','win_start','win_end','latitude','longitude',...]
    """
    if summary.empty:
        return summary.copy()

    df = summary.copy()
    df[["NET","STA","LOC","CHA"]] = df["seed_id"].str.split(".", expand=True)
    df["FAM"] = df["CHA"].map(_chan_family)

    df["S"] = pd.to_datetime(df["win_start"], errors="coerce", utc=False)
    df["E"] = pd.to_datetime(df["win_end"],   errors="coerce", utc=False)
    if default_start is not None:
        df["S"] = df["S"].fillna(pd.to_datetime(default_start, utc=False))
    if default_end is not None:
        df["E"] = df["E"].fillna(pd.to_datetime(default_end, utc=False))
    df = df.dropna(subset=["S", "E"]).copy()
    if df.empty:  # nothing left
        return df

    df["lat_k"] = df["latitude"].round(coord_decimals)
    df["lon_k"] = df["longitude"].round(coord_decimals)
    df["NS"] = df["NET"] + "." + df["STA"]

    collapsed = (df.groupby(["lat_k","lon_k","LOC","FAM","NS"], as_index=False)
                   .agg(S=("S","min"), E=("E","max"),
                        latitude=("latitude","first"),
                        longitude=("longitude","first")))

    keep_mask = pd.Series(True, index=collapsed.index)
    dropped = []

    for (lat_k, lon_k, loc, fam), g in collapsed.groupby(["lat_k","lon_k","LOC","FAM"]):
        g_order = g.assign(dur=(g["E"] - g["S"])).sort_values("dur", ascending=False)
        for i, row in g_order.iterrows():
            if not keep_mask.loc[i]:
                continue
            S_i, E_i = row["S"], row["E"]
            covers = (g_order.index != i) & (g_order["S"] <= S_i) & (g_order["E"] >= E_i)
            if covers.any():
                keep_mask.loc[i] = False
                dropped.append((row["NS"], loc, fam, float(lat_k), float(lon_k), S_i, E_i))

    filtered = collapsed.loc[keep_mask].copy()
    if verbose and dropped:
        print("[co-located prune] Dropped contained station windows:")
        for ns, loc, fam, lat_k, lon_k, S_i, E_i in dropped:
            print(f"  - {ns} LOC={loc} FAM={fam} @({lat_k:.4f},{lon_k:.4f}) "
                  f"{UTCDateTime(S_i.to_pydatetime()).isoformat()} → {UTCDateTime(E_i.to_pydatetime()).isoformat()}")

    filtered["seed_like"] = filtered["NS"] + "." + filtered["LOC"] + "." + filtered["FAM"]
    out = filtered.rename(columns={"S":"win_start","E":"win_end"})
    return out[["seed_like","NS","LOC","FAM","win_start","win_end","latitude","longitude"]].reset_index(drop=True)

# ----------------------- core queries -----------------------

def stations_in_circle_as_df(
    providers: Sequence[str] = ("IRIS", "https://fdsnws.raspberryshakedata.com"),
    start: str | UTCDateTime = "1950-01-01",
    end: str | UTCDateTime = "2026-01-01",
    lat: float = 16.72,
    lon: float = -62.18,
    radius_km: float = 100.0,
    network: str = "*",
    station: str = "*",
    location: str = "*",
    channel: str = "*",
    include_restricted: bool = True,
    timeout: int = 60,
    ignore_filters: bool = False,
) -> pd.DataFrame:
    """
    Query one or more FDSN providers for channels within a circular domain
    that overlap [start, end), returning a combined DataFrame.

    Columns: provider, network, station, location, channel, seed_id,
             win_start, win_end, latitude, longitude, elevation
    """
    start = UTCDateTime(start)
    end = UTCDateTime(end)
    maxradius = _km_to_deg(radius_km)
    rows: List[Dict] = []

    for prov in providers:
        cli = None
        
        if (isinstance(prov, str) and "raspberryshake" in prov.lower()) or str(prov).upper() in {"RASPISHAKE","RASPBERRY_SHAKE","AM"}:
            if pd.Timestamp(end.datetime) < pd.Timestamp("2016-01-01"):
                print("[info] Skipping Raspberry Shake for pre-2016 window.")
                continue
        try:
            cli = _make_client(prov, timeout=timeout)
            inv = cli.get_stations(
                network=network, station=station, location=location, channel=channel,
                startbefore=end, endafter=start,
                latitude=lat, longitude=lon, minradius=0.0, maxradius=maxradius,
                level="channel", includerestricted=include_restricted,
            )
        except Exception as e:
            print(f"[warn] provider {prov}: {e}")
            continue
        finally:
            # Close ASAP to avoid lingering threads/locks
            if cli is not None:
                _close_client(cli)



        instrument_filter = set("HLDVSAP")
        bandcode_filter = set("HEBSL")
        for net in inv:
            for sta in net:
                for cha in sta.channels:
                    if not ignore_filters and channel == "*" and (cha.code[1] not in instrument_filter or cha.code[0] not in bandcode_filter):
                        continue
                   
                    rows.append({
                        "provider": str(prov),
                        "network": net.code,
                        "station": sta.code,
                        "location": cha.location_code or "--",
                        "channel": cha.code,
                        "seed_id": f"{net.code}.{sta.code}.{(cha.location_code or '--')}.{cha.code}",
                        "win_start": _to_pandas_ts(cha.start_date),
                        "win_end":   _to_pandas_ts(cha.end_date),
                        "latitude":  cha.latitude,
                        "longitude": cha.longitude,
                        "elevation": cha.elevation,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=["provider", "seed_id", "win_start", "win_end"]
    ).reset_index(drop=True)

    return df

# ----------------------- rover file helpers -----------------------

def build_nonoverlapping_rover_lines_from_raw(
    raw_df: pd.DataFrame,
    *,
    default_start=None,
    default_end=None,
    coord_decimals: int = 4,
    keep_location: bool = True,
    net_priority: list[str] | None = None,   # e.g., ["TR","MC","WI","IU","XT"]
) -> list[str]:
    """
    From RAW rows (provider, network, station, location, channel, seed_id, win_start, win_end, lat, lon, elev),
    build Rover lines that are non-overlapping across networks for co-located, same STA/LOC/FAM.

    Enforces a hard clamp to [default_start, default_end] when provided.
    """
    if raw_df.empty:
        return []

    # Parse inputs -> timestamps
    df = raw_df.copy()
    df[["NET","STA","LOC","CHA"]] = df["seed_id"].str.split(".", expand=True)
    df["FAM"] = df["channel"].map(_chan_family)
    df["S"] = pd.to_datetime(df["win_start"], errors="coerce", utc=False)
    df["E"] = pd.to_datetime(df["win_end"],   errors="coerce", utc=False)

    # Fill missing with defaults
    dstart_ts = pd.to_datetime(default_start, utc=False) if default_start is not None else None
    dend_ts   = pd.to_datetime(default_end,   utc=False) if default_end   is not None else None
    if dstart_ts is not None:
        df["S"] = df["S"].fillna(dstart_ts)
    if dend_ts is not None:
        df["E"] = df["E"].fillna(dend_ts)

    # Hard CLAMP to [default_start, default_end] if provided
    if dstart_ts is not None:
        df.loc[df["S"] < dstart_ts, "S"] = dstart_ts
    if dend_ts is not None:
        df.loc[df["E"] > dend_ts, "E"] = dend_ts

    # Drop unusable rows
    df = df.dropna(subset=["S","E"])
    df = df.loc[df["S"] < df["E"]]
    if df.empty:
        return []

    # Normalize LOC and co-location rounding
    df["LOC_OUT"] = df["LOC"].map(_loc_for_rover)
    if not keep_location:
        df["LOC_OUT"] = "*"
    df["LAT_R"] = df["latitude"].round(coord_decimals)
    df["LON_R"] = df["longitude"].round(coord_decimals)

    # Collapse to one (Smin,Emax) per NET.STA.LOC.FAM at a coordinate — already clamped
    fam_win = (df.groupby(["LAT_R","LON_R","NET","STA","LOC_OUT","FAM"], as_index=False)
                 .agg(Smin=("S","min"), Emax=("E","max")))

    lines: list[str] = []

    # Build co-located groups (same STA/LOC/FAM at same rounded coordinate)
    for (lat_r, lon_r, sta, loc, fam), sub in fam_win.groupby(["LAT_R","LON_R","STA","LOC_OUT","FAM"]):
        # Per-network intervals (merged)
        net_windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for _, r in sub.iterrows():
            net_windows.setdefault(r.NET, []).append((r.Smin, r.Emax))
        for n in list(net_windows):
            net_windows[n] = _merge_intervals(net_windows[n])

        winner = _choose_winner(net_windows, net_priority)
        winner_out = net_windows[winner][:]

        # Emit helper with final safety clamp
        def _emit(net: str, s: pd.Timestamp, e: pd.Timestamp):
            if dstart_ts is not None and s < dstart_ts:
                s = dstart_ts
            if dend_ts is not None and e > dend_ts:
                e = dend_ts
            if s >= e:
                return
            loc_out = _loc_for_availability(loc) if keep_location else "*"
            lines.append(f"{net} {sta} {loc_out} {fam} "
                        f"{UTCDateTime(s.to_pydatetime()).isoformat()} "
                        f"{UTCDateTime(e.to_pydatetime()).isoformat()}")

        # Winner gets its (already clamped) coverage
        for s, e in winner_out:
            _emit(winner, s, e)

        # Non-winners get only their non-overlapping parts
        mask_others = _merge_intervals([iv for n, vs in net_windows.items() if n == winner for iv in vs])
        for net, ivals in net_windows.items():
            if net == winner:
                continue
            # Mask = winner plus all other nets except this one
            mask = _merge_intervals(mask_others + [iv for n2, vs in net_windows.items() if n2 not in (net, winner) for iv in vs])
            for s, e in _interval_diff(ivals, mask):
                _emit(net, s, e)

    return sorted(set(lines))

import requests

def filter_lines_by_iris_availability(
    lines: list[str],
    *,
    avail_base: str = "https://service.iris.edu/fdsnws/availability/1/query",
    timeout: int = 60,
    batch_size: int = 200,
    verbose: bool = True,
) -> list[str]:
    """
    Keep only rover lines that IRIS reports as having any availability.
    - Request lines may use LOC='*' and CHA families like 'BH?'.
    - Response lines have concrete LOC (e.g., '00' or '') and concrete CHA (e.g., 'BHZ').
    We normalize response to families and allow LOC wildcard matching.
    """
    if not lines:
        return []

    def post_batch(batch: list[str]) -> set[tuple[str, str, str, str]]:
        """Return a set of normalized availability keys from the response."""
        payload = "merge=samplerate,quality\n" + "\n".join(batch) + "\n"
        params = {"format": "text"}
        try:
            r = requests.post(
                avail_base, params=params, data=payload,
                headers={"Content-Type": "text/plain"}, timeout=timeout
            )
            if r.status_code == 404:
                if verbose:
                    print(f"[avail] 0 / {len(batch)} lines matched (404).")
                return set()
            r.raise_for_status()
        except Exception as e:
            print(f"[warn] availability probe failed for batch of {len(batch)}: {e}")
            return set()

        found = set()
        for ln in r.text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("merge="):
                continue
            parts = ln.split()
            if len(parts) < 6:
                continue
            net, sta, loc_resp, cha_resp = parts[:4]
            fam_resp = _chan_family(cha_resp)  # 'BHZ' -> 'BH?'
            # Normalize blank locations from response to '' (we handle wildcard on request side)
            loc_resp = loc_resp or ""
            found.add((net, sta, loc_resp, fam_resp))
        return found

    kept: list[str] = []
    # Collect response availability across all batches
    all_found: set[tuple[str, str, str, str]] = set()

    # Batch request lines to avoid huge posts
    buf: list[str] = []
    for ln in lines:
        buf.append(ln)
        if len(buf) >= batch_size:
            all_found |= post_batch(buf)
            buf = []
    if buf:
        all_found |= post_batch(buf)

    if verbose:
        print(f"[avail] response contained {len(all_found)} availability rows (normalized).")

    if not all_found:
        return []

    # Build a quick index by (NET, STA, FAM) → set of LOCs that had data
    by_nsf: dict[tuple[str, str, str], set[str]] = {}
    for net, sta, loc_resp, fam_resp in all_found:
        by_nsf.setdefault((net, sta, fam_resp), set()).add(loc_resp)

    # Now decide, per original request line, whether to keep it
    for ln in lines:
        parts = ln.split()
        if len(parts) < 6:
            continue
        net, sta, loc_req, fam_req = parts[:4]
        fam_req = fam_req if fam_req.endswith("?") else _chan_family(fam_req)
        loc_req = "" if (not loc_req or loc_req == "--") else loc_req  # normalize blank to ''
        # Wildcard match: if request loc is '*', accept any LOC for this (NET, STA, FAM)
        locs = by_nsf.get((net, sta, fam_req))
        if not locs:
            continue
        if loc_req == "*" or loc_req in locs:
            kept.append(ln)

    if verbose:
        print(f"[avail] kept {len(kept)} / {len(lines)} lines after normalization.")
    return sorted(set(kept))

def rover_lines_from_summary(
    summary_df: pd.DataFrame,
    *,
    use_family: bool = True,     # Collapse BHZ/BHN/BHE -> BH?
    keep_location: bool = True,  # True keeps per-LOC; False uses '*' location wildcard
    deduplicate: bool = True,
    default_start: str | pd.Timestamp | UTCDateTime | None = None,
    default_end: str | pd.Timestamp | UTCDateTime | None = None,
) -> List[str]:
    """
    Build Rover request lines (NET STA LOC CHA START END) from a summary frame with:
      columns = ['seed_id','win_start','win_end', ...]

    - If use_family=True, rows are merged per (NET, STA, LOC, CH_FAM):
        START = min(win_start), END = max(win_end) across that family.
    - If keep_location=False, LOC is set to '*'.
    - Missing start/end are filled with default_start/default_end if provided.
    - NEW: If computed START < default_start or END > default_end, clamp to those defaults.
    """
    if summary_df.empty:
        return []

    tmp = summary_df.copy()
    tmp[["NET", "STA", "LOC", "CHA"]] = tmp["seed_id"].str.split(".", expand=True)

    tmp["LOC"] = tmp["LOC"].map(_loc_for_rover)
    if not keep_location:
        tmp["LOC"] = "*"

    tmp["S"] = pd.to_datetime(tmp["win_start"], errors="coerce", utc=False)
    tmp["E"] = pd.to_datetime(tmp["win_end"],   errors="coerce", utc=False)

    dstart_ts = pd.to_datetime(default_start, utc=False) if default_start is not None else None
    dend_ts   = pd.to_datetime(default_end,   utc=False) if default_end   is not None else None

    if dstart_ts is not None:
        tmp["S"] = tmp["S"].fillna(dstart_ts)
    if dend_ts is not None:
        tmp["E"] = tmp["E"].fillna(dend_ts)

    tmp = tmp.dropna(subset=["S", "E"])
    if tmp.empty:
        return []

    rows: List[str] = []

    def _emit_line(net: str, sta: str, loc: str, cha_or_fam: str, S: pd.Timestamp, E: pd.Timestamp):
        # Clamp to [default_start, default_end] if provided
        if dstart_ts is not None and S < dstart_ts:
            S = dstart_ts
        if dend_ts is not None and E > dend_ts:
            E = dend_ts
        if S >= E:  # skip empty/invalid after clamping
            return
        s_iso = UTCDateTime(S.to_pydatetime()).isoformat()
        e_iso = UTCDateTime(E.to_pydatetime()).isoformat()
        loc_out = _loc_for_availability(loc) if keep_location else "*"
        rows.append(f"{net} {sta} {loc_out} {cha_or_fam} {s_iso} {e_iso}")

    if use_family:
        tmp["FAM"] = tmp["CHA"].map(_chan_family)
        g = (tmp.groupby(["NET", "STA", "LOC", "FAM"], as_index=False)
                .agg(S=("S", "min"), E=("E", "max")))
        for _, r in g.iterrows():
            _emit_line(r.NET, r.STA, r.LOC, r.FAM, r.S, r.E)
    else:
        for _, r in tmp.iterrows():
            _emit_line(r.NET, r.STA, r.LOC, r.CHA, r.S, r.E)

    return sorted(set(rows)) if deduplicate else sorted(rows)

def write_rover_file(lines: Iterable[str], path: str) -> None:
    path = str(Path(path))
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def generate_rover_request(
    startdate: str = "2022-07-27",
    enddate: str = "2022-07-30",
    lat: float = 16.72,
    lon: float = -62.18,
    radius: float = 65.0,                     # km
    rover_base: str = "/data/rover_data_repo",
    region: str = "Montserrat",
    *,
    providers: Sequence[str] = ("IRIS", "RASPISHAKE"),
    network: str = "*",
    channel: str = "*",
    coord_decimals: int = 4,
    precheck_availability: bool = True,
    ignore_filters: bool = False,
    verbose: bool = True,
):
    """
    Query FDSN providers in a circular domain and emit a Rover request file.

    Returns a dict with:
      {
        "out_path": <str or None>,
        "num_channels": <int>,
        "num_lines": <int>,
        "lines": <list[str]>
      }
    """
    import os

    # Optional: skip Raspberry Shake for pre-2016 windows
    try:
        if any(
            (isinstance(p, str) and ("raspberryshake" in p.lower() or p.upper() in {"RASPISHAKE","RASPBERRY_SHAKE","AM"}))
            for p in providers
        ):
            if pd.Timestamp(UTCDateTime(enddate).datetime) < pd.Timestamp("2016-01-01"):
                if verbose:
                    print("[info] Skipping Raspberry Shake for pre-2016 window.")
                providers = tuple(p for p in providers if not (
                    isinstance(p, str) and ("raspberryshake" in p.lower() or p.upper() in {"RASPISHAKE","RASPBERRY_SHAKE","AM"})
                ))
    except Exception:
        pass

    # 1) Discover candidate channels from metadata
    raw_df = stations_in_circle_as_df(
        providers=providers,
        start=startdate,
        end=enddate,
        lat=lat, lon=lon, radius_km=radius,
        network=network,
        channel=channel,
        ignore_filters=ignore_filters,
    )
    print(raw_df)

    if verbose:
        print(f"Found {len(raw_df)} channels in the specified circular domain.")

    if raw_df.empty:
        return {"out_path": None, "num_channels": 0, "num_lines": 0, "lines": []}

    # 2) Build non-overlapping rover lines, clamped to [startdate, enddate]
    lines = build_nonoverlapping_rover_lines_from_raw(
        raw_df,
        default_start=startdate,
        default_end=enddate,
        coord_decimals=coord_decimals,
        keep_location=True,
    )

    # 3) Optional: prefilter against IRIS availability so rover won’t 404
    if precheck_availability:
        lines = filter_lines_by_iris_availability(lines)
        if verbose:
            print(f"[post-filter] {len(lines)} lines remain after IRIS availability check.")

    if not lines:
        print("No lines with data at IRIS for the requested window after availability check.")
        return {"out_path": None, "num_channels": len(raw_df), "num_lines": 0, "lines": []}

    # 4) Write rover file
    os.makedirs(rover_base, exist_ok=True)
    out_path = os.path.join(
        rover_base,
        f"rover_request_circle_{region}_radius{radius:.0f}km_{startdate.replace('-','')}_{enddate.replace('-','')}.txt"
    )
    write_rover_file(lines, out_path)
    print(f"Wrote {len(lines)} lines to {out_path}")

    # 5) Convenience: echo the file and show next steps
    try:
        os.system(f"cat {out_path}")
    except Exception:
        pass

    print("Next steps:")
    print(f"cd {rover_base}")
    print(f"rover retrieve {out_path}")
    sd = UTCDateTime(startdate)
    print("tree")
    print(f"python plot_rover_day.py --root {rover_base} -y {sd.year} -d {sd.julday}")

    return {"out_path": out_path, "num_channels": len(raw_df), "num_lines": len(lines), "lines": lines}
# ----------------------- script entry -----------------------

if __name__ == "__main__":
    generate_rover_request()  # uses the defaults you specified