#!/usr/bin/env python3
"""
Summarize FDSN channel availability in a circular domain and emit a Rover request.

- Queries one or more FDSN providers (e.g., IRIS, Raspberry Shake).
- Returns a DataFrame of channels within <radius_km> of (lat, lon) that overlap the
  requested time window.
- Builds a Rover request file with per-(NET, STA, LOC, channel family) lines whose
  START/END come from the discovered availability windows.

Example:
    python summarize_fdsn_data_available.py
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
    Special-cases Raspberry Shake with explicit service_mappings.
    """
    prov_str = str(prov).strip()

    # Named convenience
    if prov_str.upper() in {"RASPISHAKE", "RASPBERRY_SHAKE", "AM"}:
        return Client(
            base_url="https://fdsnws.raspberryshakedata.com",
            service_mappings={
                "station":    "https://fdsnws.raspberryshakedata.com/station/1/query",
                "dataselect": "https://fdsnws.raspberryshakedata.com/dataselect/1/query",
            },
            timeout=timeout,
        )

    # If a URL and it's the Raspberry Shake hostname, also force mappings
    if "raspberryshakedata.com" in prov_str:
        return Client(
            base_url="https://fdsnws.raspberryshakedata.com",
            service_mappings={
                "station":    "https://fdsnws.raspberryshakedata.com/station/1/query",
                "dataselect": "https://fdsnws.raspberryshakedata.com/dataselect/1/query",
            },
            timeout=timeout,
        )

    # Otherwise use normal discovery (IRIS, USGS, etc.)
    if prov_str.startswith(("http://", "https://")):
        return Client(base_url=prov_str, timeout=timeout)
    return Client(prov_str, timeout=timeout)

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
    # ObsPy UTCDateTime -> naive UTC datetime
    try:
        if isinstance(x, UTCDateTime):
            return pd.Timestamp(x.datetime)
    except Exception:
        pass
    # Let pandas try the rest
    return pd.to_datetime(x, errors="coerce")

def _loc_for_rover(loc: str) -> str:
    """Rover uses '--' for blank location codes."""
    if not loc or loc.strip() in {"", "--"}:
        return "--"
    return loc

def _chan_family(cha: str) -> str:
    """
    Collapse a specific 3+ char channel (e.g., BHZ/BHN/BHE/BH1/BH2) to a family like 'BH?'.
    If shorter/odd, return '*' as a safe wildcard.
    """
    cha = (cha or "").strip().upper()
    return f"{cha[:2]}?" if len(cha) >= 2 else "*"

def filter_colocated_by_containment(
    summary: pd.DataFrame,
    *,
    coord_decimals: int = 4,       # ~11 m at equator; adjust if needed
    default_start=None,            # fill NaT starts (optional)
    default_end=None,              # fill NaT ends (e.g., your query end)
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop stations fully contained in time by another station at (nearly) the same coordinates,
    for the same location code and channel family.

    Expects columns: ['seed_id','win_start','win_end','latitude','longitude',...]

    Returns a filtered copy of `summary`.
    """
    if summary.empty:
        return summary.copy()

    df = summary.copy()

    # Split seed_id -> NET, STA, LOC, CHA; compute channel family (e.g., HH? / BH?)
    df[["NET","STA","LOC","CHA"]] = df["seed_id"].str.split(".", expand=True)
    df["FAM"] = df["CHA"].map(_chan_family)

    # Normalize times to pandas Timestamps
    df["S"] = pd.to_datetime(df["win_start"], errors="coerce", utc=False)
    df["E"] = pd.to_datetime(df["win_end"],   errors="coerce", utc=False)

    if default_start is not None:
        df["S"] = df["S"].fillna(pd.to_datetime(default_start, utc=False))
    if default_end is not None:
        df["E"] = df["E"].fillna(pd.to_datetime(default_end, utc=False))

    # Drop rows still missing S/E
    df = df.dropna(subset=["S", "E"]).copy()
    if df.empty:
        return df

    # Bucket coordinates (so "same site" within a small tolerance)
    df["lat_k"] = df["latitude"].round(coord_decimals)
    df["lon_k"] = df["longitude"].round(coord_decimals)

    # Collapse to one window per (NET.STA, site, LOC, FAM): take min start, max end
    df["NS"] = df["NET"] + "." + df["STA"]
    collapsed = (df.groupby(["lat_k","lon_k","LOC","FAM","NS"], as_index=False)
                   .agg(S=("S","min"), E=("E","max"),
                        latitude=("latitude","first"),
                        longitude=("longitude","first")))

    # Within each (site, LOC, FAM), drop any NS fully contained by another NS
    keep_mask = pd.Series(True, index=collapsed.index)
    dropped = []

    for (lat_k, lon_k, loc, fam), g in collapsed.groupby(["lat_k","lon_k","LOC","FAM"]):
        # Sort by longest coverage first; containment tests become stable/predictable
        g_order = g.assign(dur=(g["E"] - g["S"])).sort_values("dur", ascending=False)

        # For each candidate, see if some *other* row covers it fully
        for i, row in g_order.iterrows():
            if not keep_mask.loc[i]:
                continue  # already marked to drop

            S_i, E_i = row["S"], row["E"]
            # Any j with S_j <= S_i and E_j >= E_i, j != i ?
            # Because we sorted by dur desc, the first longer/equal often wins.
            covers = (g_order.index != i) & \
                     (g_order["S"] <= S_i) & (g_order["E"] >= E_i)
            if covers.any():
                keep_mask.loc[i] = False
                dropped.append((row["NS"], loc, fam, float(lat_k), float(lon_k), S_i, E_i))

    filtered = collapsed.loc[keep_mask].copy()

    if verbose and dropped:
        print("[co-located prune] Dropped contained station windows:")
        for ns, loc, fam, lat_k, lon_k, S_i, E_i in dropped:
            print(f"  - {ns} LOC={loc} FAM={fam} @({lat_k:.4f},{lon_k:.4f}) "
                  f"{UTCDateTime(S_i.to_pydatetime()).isoformat()} → {UTCDateTime(E_i.to_pydatetime()).isoformat()}")

    # Map back to a summary-like frame keyed by NET.STA (you’ll still write Rover per LOC/FAM)
    # Keep seed-like identity as NET.STA.LOC.FAM
    filtered["seed_like"] = filtered["NS"] + "." + filtered["LOC"] + "." + filtered["FAM"]
    out = filtered.rename(columns={"S":"win_start","E":"win_end"})
    # Put columns in a handy order
    return out[["seed_like","NS","LOC","FAM","win_start","win_end","latitude","longitude"]].reset_index(drop=True)

import pandas as pd
from obspy import UTCDateTime

def _fam(ch: str) -> str:
    ch = (ch or "").upper()
    return f"{ch[:2]}?" if len(ch) >= 2 else "*"

def _loc_for_rover(loc: str) -> str:
    return "--" if (not loc or loc.strip() in {"", "--"}) else loc

def _to_ts(x):  # robust Timestamp
    if x is None:
        return pd.NaT
    if isinstance(x, UTCDateTime):
        return pd.Timestamp(x.datetime)
    return pd.to_datetime(x, errors="coerce", utc=False)

def _choose_winner(net_windows, net_priority):
    """
    Pick the winner network in a co-located overlap group.
    Strategy: priority list first; then total coverage length; then lexical.
    net_windows: {NET: [(S,E), ...]}  (already merged per NET)
    """
    # Priority ranking (lower is better)
    rank = {n: i for i, n in enumerate(net_priority)} if net_priority else {}
    # total coverage per NET
    cov = {}
    for net, ivals in net_windows.items():
        total = pd.Timedelta(0)
        for s, e in ivals:
            if pd.isna(s) or pd.isna(e):
                continue
            total += (e - s)
        cov[net] = total
    # sort by (rank, -coverage, net)
    def keyf(n):
        return (rank.get(n, 10_000), -cov[n].total_seconds(), n)
    return sorted(net_windows.keys(), key=keyf)[0]

def _merge_intervals(intervals):
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

def _interval_diff(a, b):
    """
    Return A backslasg B where A and B are half-open-like lists of (S,E).
    Assumes S<E. Returns list of non-overlapping (S,E) in A not covered by B.
    """
    # Merge for robustness
    A = _merge_intervals(a)
    B = _merge_intervals(b)
    out = []
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

    Returns lines like:
      NET STA LOC FAM START END
    """
    if raw_df.empty:
        return []

    df = raw_df.copy()
    df[["NET","STA","LOC","CHA"]] = df["seed_id"].str.split(".", expand=True)
    df["FAM"] = df["channel"].map(_fam)
    df["S"] = pd.to_datetime(df["win_start"], errors="coerce", utc=False)
    df["E"] = pd.to_datetime(df["win_end"],   errors="coerce", utc=False)

    if default_start is not None:
        df["S"] = df["S"].fillna(pd.to_datetime(default_start, utc=False))
    if default_end is not None:
        df["E"] = df["E"].fillna(pd.to_datetime(default_end, utc=False))

    # Drop rows missing time bounds after fills
    df = df.dropna(subset=["S","E"])
    if df.empty:
        return []

    # Normalize LOC + co-location rounding
    df["LOC_OUT"] = df["LOC"].map(_loc_for_rover)
    if not keep_location:
        df["LOC_OUT"] = "*"
    df["LAT_R"] = df["latitude"].round(coord_decimals)
    df["LON_R"] = df["longitude"].round(coord_decimals)

    # Collapse to station-family windows per NET.STA.LOC.FAM at a coordinate
    fam_win = (df.groupby(["LAT_R","LON_R","NET","STA","LOC_OUT","FAM"], as_index=False)
                 .agg(Smin=("S","min"), Emax=("E","max")))

    lines = []

    # For each co-located cluster with same STA, LOC, FAM:
    for (lat_r, lon_r, sta, loc, fam), sub in fam_win.groupby(["LAT_R","LON_R","STA","LOC_OUT","FAM"]):
        # Build per-network interval lists (here each net has a single merged interval)
        net_windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for _, r in sub.iterrows():
            net_windows.setdefault(r.NET, []).append((r.Smin, r.Emax))
        # Merge per net (just in case)
        for n in list(net_windows):
            net_windows[n] = _merge_intervals(net_windows[n])

        # Winner network for the overlapping portions
        winner = _choose_winner(net_windows, net_priority)

        # Compute the union of all intervals
        all_ivals = _merge_intervals([iv for vals in net_windows.values() for iv in vals])

        # Assign union pieces to a single network:
        #  - winner gets overlap parts,
        #  - non-winners get only their non-overlapping parts (A \ others).
        # Build “others” merged
        others = _merge_intervals([iv for n, vals in net_windows.items() if n != winner for iv in vals])

        # Winner intervals = union ∩ (winner ∪ (winner overlapped))  → just take winner coverage + any union overlap
        # Simplify: winner gets its own coverage plus any overlap; others trimmed.
        # We do: winner_out = merge(winner coverage ∪ (union ∩ others)) == merge(union ∩ (winner ∪ others)) == union
        # but to avoid assigning periods the winner does not actually have, keep winner to its own coverage:
        winner_out = net_windows[winner][:]  # only what winner actually covers

        # Non-winners: A \ (all others). Here “others” for each non-winner = union of (winner + other non-winners except itself).
        for net, ivals in net_windows.items():
            if net == winner:
                for s, e in winner_out:
                    lines.append(f"{net} {sta} {loc} {fam} {UTCDateTime(s.to_pydatetime()).isoformat()} {UTCDateTime(e.to_pydatetime()).isoformat()}")
                continue
            # Build mask = winner coverage ∪ all other nets except this one
            mask = _merge_intervals(winner_out + [iv for n2, vs in net_windows.items() if n2 not in (net, winner) for iv in vs])
            trimmed = _interval_diff(ivals, mask)
            for s, e in trimmed:
                lines.append(f"{net} {sta} {loc} {fam} {UTCDateTime(s.to_pydatetime()).isoformat()} {UTCDateTime(e.to_pydatetime()).isoformat()}")

    # Deduplicate + sort
    lines = sorted(set(lines))
    return lines
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

        instrument_filter = set("HLDVSAP")
        for net in inv:
            for sta in net:
                for cha in sta.channels:
                    if channel == "*" and cha.code[1] not in instrument_filter:
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

    # Drop exact duplicates (can happen if providers overlap in coverage).
    df = df.drop_duplicates(
        subset=["provider", "seed_id", "win_start", "win_end"]
    ).reset_index(drop=True)

    return df



# ----------------------- rover file helpers -----------------------

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
    """
    if summary_df.empty:
        return []

    tmp = summary_df.copy()
    tmp[["NET", "STA", "LOC", "CHA"]] = tmp["seed_id"].str.split(".", expand=True)

    tmp["LOC"] = tmp["LOC"].map(_loc_for_rover)
    if not keep_location:
        tmp["LOC"] = "*"

    # Parse to timestamps
    tmp["S"] = pd.to_datetime(tmp["win_start"], errors="coerce", utc=False)
    tmp["E"] = pd.to_datetime(tmp["win_end"],   errors="coerce", utc=False)

    # Fill missing with provided defaults (e.g., your query bounds)
    if default_start is not None:
        tmp["S"] = tmp["S"].fillna(pd.to_datetime(default_start, utc=False))
    if default_end is not None:
        tmp["E"] = tmp["E"].fillna(pd.to_datetime(default_end, utc=False))

    # Drop rows still missing times
    tmp = tmp.dropna(subset=["S", "E"])
    if tmp.empty:
        return []

    rows: List[str] = []

    if use_family:
        tmp["FAM"] = tmp["CHA"].map(_chan_family)
        g = (tmp.groupby(["NET", "STA", "LOC", "FAM"], as_index=False)
                .agg(S=("S", "min"), E=("E", "max")))
        for _, r in g.iterrows():
            s_iso = UTCDateTime(r.S.to_pydatetime()).isoformat()
            e_iso = UTCDateTime(r.E.to_pydatetime()).isoformat()
            rows.append(f"{r.NET} {r.STA} {r.LOC} {r.FAM} {s_iso} {e_iso}")
    else:
        for _, r in tmp.iterrows():
            s_iso = UTCDateTime(r.S.to_pydatetime()).isoformat()
            e_iso = UTCDateTime(r.E.to_pydatetime()).isoformat()
            rows.append(f"{r.NET} {r.STA} {r.LOC} {r.CHA} {s_iso} {e_iso}")

    return sorted(set(rows)) if deduplicate else sorted(rows)

def write_rover_file(lines: Iterable[str], path: str) -> None:
    path = str(Path(path))
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

# ----------------------- script entry -----------------------

if __name__ == "__main__":
    # Query both IRIS and Raspberry Shake in a 100 km circle around Montserrat
    startdate = "1989-01-01"
    enddate   = UTCDateTime.now().strftime("%Y-%m-%d")
    radius=50.0  #  km radius
    raw_df = stations_in_circle_as_df(
        providers=("IRIS", "https://data.raspberryshake.org"),
        start=startdate,
        end=enddate,
        lat=16.72, lon=-62.18, radius_km=radius,
        network="*",                         # AM will be included
        #channel="BH?,EH?,HH?,SH?,BD?,ED?,HD?,SD?",
        channel="*",
    )

    print(f"Found {len(raw_df)} channels in the specified circular domain.")

    if not raw_df.empty:
        # STEP 1 — prune co-located stations fully contained by others
        '''
        pruned_df = filter_colocated_by_containment(
            raw_df,
            coord_decimals=4,
            default_start=startdate,
            default_end=enddate,
            verbose=True,
        )
        '''

        lines = build_nonoverlapping_rover_lines_from_raw(
            raw_df,
            default_start=startdate,
            default_end=enddate,
            coord_decimals=4,
            keep_location=True,
        )

        '''
        # STEP 2 — now aggregate to one row per seed_id for Rover
        summary = (
            pruned_df.groupby("seed_id", as_index=False)
                     .agg(win_start=("win_start", "min"),
                          win_end=("win_end", "max"),
                          latitude=("latitude", "first"),
                          longitude=("longitude", "first"),
                          elevation=("elevation", "first"))
        )

        # STEP 3 — build Rover lines from the pruned summary
        lines = rover_lines_from_summary(
            summary,
            use_family=True,
            keep_location=True,
            default_start=startdate,
            default_end=enddate,
        )

        # Preview and write
        print("\nSample Rover lines:")
        print("\n".join(lines[:10]))
        '''
        out_path = f"montserrat_circle_{radius:.0f}km.rover"
        write_rover_file(lines, out_path)
        print(f"Wrote {len(lines)} lines to {out_path}")