# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable, Union, List, Tuple, Dict, Any
from datetime import datetime, timezone
import re
import os

import numpy as np
import pandas as pd
from obspy import UTCDateTime
import matplotlib.pyplot as plt

# project
from flovopy.asl.map import plot_heatmap_colored
from flovopy.enhanced.eventrate import EventRate, EventRateConfig
from flovopy.enhanced.event import EnhancedEvent
from flovopy.enhanced.catalog import EnhancedCatalog


# ----------------------------
# Heatmap
# ----------------------------
def plot_heatmap_colored(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    amp_col: str = "amplitude",
    inventory=None,
    cmap: str = "turbo",
    log_scale: bool = True,
    node_spacing_m: int = 50,
    outfile: Optional[Union[str, Path]] = None,
    region: Optional[List[float]] = None,
    title: Optional[str] = None,
    dem_tif: Optional[Union[str, Path]] = None,
    topo_kw: Optional[Dict[str, Any]] = None,
) -> "pygmt.Figure":
    """
    Render a colored heatmap of energy (sum amplitude^2) on a topo basemap.
    Draw basemap first, then create CPT, then plot points (fill=z with cmap=True), then colorbar.
    """
    if df is None or df.empty:
        raise ValueError("plot_heatmap_colored: input DataFrame is empty")

    df = df.copy()
    df["energy"] = df[amp_col].astype(float) ** 2
    grouped = df.groupby([lat_col, lon_col], as_index=False)["energy"].sum()

    x = grouped[lon_col].to_numpy(float)
    y = grouped[lat_col].to_numpy(float)
    z = grouped["energy"].to_numpy(float)
    if log_scale:
        z = np.log10(z + 1e-12)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        raise ValueError("plot_heatmap_colored: no finite (lon,lat,energy) to plot")

    # 1) Basemap first
    if topo_kw is None:
        topo_kw = {
            "inv": inventory,
            "add_labels": False,
            "cmap": "gray",
            "region": region,
            "dem_tif": dem_tif,
            "frame": True,
            "topo_color": False,
            "add_colorbar": False,
        }
    fig = topo_map(**topo_kw, title=title, add_colorbar=False)#, add_topography=False)

    # 2) Make our CPT (so it is the current CPT)
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if zmax == zmin:
        zmin -= 0.01
        zmax += 0.01
    step = (zmax - zmin) / 100.0 if zmax > zmin else 0.01
    pygmt.makecpt(cmap=cmap, series=[zmin, zmax, step], continuous=True)

    # 3) Plot points — IMPORTANT: fill=z *and* cmap=True
    symbol_size_cm = max(0.06, node_spacing_m * 0.077 / 50.0 * scale)
    fig.plot(
        x=x,
        y=y,
        style=f"s{symbol_size_cm}c",
        fill=z,       # numeric array to map through CPT
        cmap=True,    # use current CPT to color by 'fill' values
        pen=None
    )

    # 4) Colorbar from current CPT
    fig.colorbar(frame='+l"Log10 Total Energy"' if log_scale else '+l"Total Energy"')

    # 5) Optional title (use position OR x/y, not both)
    if title:
        fig.text(text=title, position="TL", offset="0.5c/0.5c", no_clip=True)

    if outfile:
        outpath = Path(outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(outpath))

    return fig

# ======================================================================
# --------------------------- SHARED HELPERS ----------------------------
# ======================================================================

def _to_utc_datetime(
    dt_like: Union[str, datetime, "UTCDateTime", pd.Timestamp]
) -> datetime:
    """Normalize input into a timezone-aware UTC datetime."""
    if isinstance(dt_like, pd.Timestamp):
        return (
            dt_like.tz_convert("UTC").to_pydatetime()
            if dt_like.tzinfo
            else dt_like.tz_localize("UTC").to_pydatetime()
        )
    if UTCDateTime and isinstance(dt_like, UTCDateTime):  # type: ignore
        return dt_like.datetime.replace(tzinfo=timezone.utc)
    if isinstance(dt_like, datetime):
        return dt_like.astimezone(timezone.utc) if dt_like.tzinfo else dt_like.replace(tzinfo=timezone.utc)
    if isinstance(dt_like, str):
        try:
            d = datetime.fromisoformat(dt_like)
        except ValueError:
            # fallback: 'YYYY-mm-dd-HHMM-SS'
            d = datetime.strptime(dt_like, "%Y-%m-%d-%H%M-%S")
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    raise TypeError(f"Unsupported datetime-like type: {type(dt_like)}")


# Event folders often start with 'YYYY-mm-dd-HHMM-SS...' (e.g. '2001-03-09-1248-42S.MVO___019')
_DIR_TS_RE = re.compile(r"^(?P<stamp>\d{4}-\d{2}-\d{2}-\d{4}-\d{2,3}).*$")
_TAGFN_RE  = re.compile(r"^source_(?P<tag>.+)\.csv$", re.IGNORECASE)

def _parse_event_time_from_dirname(dirname: str) -> Optional[datetime]:
    """Extract UTC datetime from leading 'YYYY-mm-dd-HHMM-SS...' pattern."""
    m = _DIR_TS_RE.match(dirname)
    if not m:
        return None
    stamp = m.group("stamp")
    parts = stamp.split("-")
    # Trim SSS -> SS if present
    if len(parts[-1]) == 3:
        stamp = "-".join(parts[:-1] + [parts[-1][:2]])
    try:
        dt = datetime.strptime(stamp, "%Y-%m-%d-%H%M-%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None

def _infer_tag_from_filename(path: Union[str, Path]) -> Optional[str]:
    """Get '<tag>' from a 'source_<tag>.csv' path or filename."""
    name = Path(path).name
    m = _TAGFN_RE.match(name)
    return m.group("tag") if m else None

def _discover_tags(event_dir: Path) -> List[str]:
    """Return tags found under event_dir/*/source_<tag>.csv."""
    tags: list[str] = []
    for sub in event_dir.iterdir():
        if not sub.is_dir():
            continue
        t = sub.name
        if (sub / f"source_{t}.csv").exists():
            tags.append(t)
    return sorted(tags)

def _ensure_iterable_tags(tag: Optional[Union[str, Iterable[str]]]) -> Optional[List[str]]:
    if tag is None:
        return None
    if isinstance(tag, str):
        return [tag]
    return list(tag)


# ======================================================================
# -------------------- HEAD: CTAG SOURCE COLLECTION --------------------
# ======================================================================

def _load_source_csv(path: str) -> pd.DataFrame | None:
    """
    Load a single 'source_<ctag>[ _refined].csv', requiring lat/lon and DR
    (case-insensitive for DR). Output columns are exactly: lat, lon, DR
    """
    try:
        df = pd.read_csv(path)
        lower = {c.lower(): c for c in df.columns}
        if "lat" not in lower or "lon" not in lower:
            return None
        dr_key = lower.get("dr", None)
        if dr_key is None:
            return None

        sub = df[[lower["lat"], lower["lon"], dr_key]].copy()
        sub.columns = ["lat", "lon", "DR"]
        # numeric & finite
        for c in ["lat", "lon", "DR"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub.dropna(subset=["lat", "lon", "DR"], inplace=True)
        return sub if not sub.empty else None
    except Exception:
        return None


def collect_sources_for_ctag(output_root: str | Path, ctag: str, refined: bool = False) -> pd.DataFrame:
    """
    Collect per-event source CSVs for a given ctag across:
        OUTPUT_ROOT/{event}/{ctag}/source_{ctag}[ _refined].csv
    Returns a single DataFrame with columns: lat, lon, DR
    """
    output_root = Path(output_root)
    dfs: list[pd.DataFrame] = []

    for ev_dir in sorted(d for d in output_root.iterdir() if d.is_dir()):
        run_dir = ev_dir / ctag
        if not run_dir.is_dir():
            continue
        f = run_dir / (f"source_{ctag}_refined.csv" if refined else f"source_{ctag}.csv")
        if f.is_file():
            sub = _load_source_csv(str(f))
            if sub is not None:
                dfs.append(sub)

    if not dfs:
        label = "refined" if refined else "primary"
        print(f"[HEATMAP] No {label} source CSVs found for ctag={ctag}")
        return pd.DataFrame(columns=["lat", "lon", "DR"])

    return pd.concat(dfs, ignore_index=True)


def write_global_heatmap_for_ctag(
    output_root: str | Path,
    ctag: str,
    *,
    node_spacing_m: float = 10.0,
    topo_kw: dict | None = None,
    title: str | None = None,
    include_refined: bool = True,
    log_scale: bool = False,
    cmap: str = "viridis",
    scale: float = 1.0,
    verbose: bool = False,
) -> Dict[str, Optional[str]]:
    """
    Aggregate source_<ctag>.csv (and optionally *_refined.csv) across
    OUTPUT_ROOT/{event}/{ctag} and write heatmaps into OUTPUT_ROOT/.
    Returns {"primary": <png or None>, "refined": <png or None>}
    """
    results: Dict[str, Optional[str]] = {"primary": None, "refined": None}

    for refined_flag, label in [(False, "primary"), (True, "refined")]:
        if refined_flag and not include_refined:
            continue

        df = collect_sources_for_ctag(output_root, ctag, refined=refined_flag)
        if df.empty:
            continue

        out_png = Path(output_root) / f"heatmap_{ctag}_{label}.png"
        plot_heatmap_colored(
            df,
            lat_col="lat",
            lon_col="lon",
            amp_col="DR",
            log_scale=log_scale,
            node_spacing_m=node_spacing_m,
            outfile=str(out_png),
            title=title or f"Energy Heatmap ({label}) — {ctag}",
            topo_kw=topo_kw,
            cmap=cmap,
            scale=scale,
            verbose=verbose,
        )
        print(f"[HEATMAP] Wrote {label}: {out_png}")
        results[label] = str(out_png)

    return results


# ======================================================================
# ------------------------ DATE-RANGE HEATMAPS -------------------------
# (single tag → one figure; multi-tag → one per tag)
# ======================================================================

def make_asl_heatmap_from_events(
    startdate: Union[str, datetime, "UTCDateTime", pd.Timestamp],
    enddate:   Union[str, datetime, "UTCDateTime", pd.Timestamp],
    *,
    localprojectdir: Union[str, Path],
    tag: Optional[str] = None,
    default_tag_from: Optional[Union[str, Path]] = None,  # e.g., '/path/to/source_<tag>.csv'
    # CSV schema: t, lat, lon, DR, misfit, azgap, nsta, node_index, connectedness
    lat_col: str = "lat",
    lon_col: str = "lon",
    amp_col: str = "DR",
    # Optional filters (applied only if column exists)
    misfit_max: Optional[float] = None,
    nsta_min: Optional[int] = None,
    connectedness_min: Optional[float] = None,
    azgap_max: Optional[float] = None,
    dr_min: Optional[float] = None,
    dr_max: Optional[float] = None,
    # Plot options (forwarded to plot_heatmap_colored)
    inventory=None,
    cmap: str = "turbo",
    log_scale: bool = True,
    node_spacing_m: int = 50,
    region: Optional[List[float]] = None,
    dem_tif: Optional[Union[str, Path]] = None,
    title_fmt: str = "ASL Heatmap: {tag} ({start}–{end} UTC)",
    outfile: Optional[Union[str, Path]] = None,   # e.g., "heatmaps/asl_{start}_{end}.png"
    topo_kw: Optional[Dict[str, Any]] = None,
    return_df: bool = False,
    verbose: bool = False,
) -> Union["pygmt.Figure", Tuple["pygmt.Figure", pd.DataFrame]]:
    """
    Build ONE heatmap by concatenating all per-event 'source_<tag>.csv' files
    under LOCALPROJECTDIR where the event folder timestamp is inside the window.
    The tag can be provided, inferred from default_tag_from, or discovered if unique.
    """
    lp = Path(localprojectdir)
    if not lp.exists():
        raise FileNotFoundError(f"localprojectdir not found: {lp}")

    start_dt = _to_utc_datetime(startdate)
    end_dt   = _to_utc_datetime(enddate)
    if end_dt < start_dt:
        raise ValueError("enddate is earlier than startdate")

    # Resolve tag
    if tag is None and default_tag_from:
        inferred = _infer_tag_from_filename(default_tag_from)
        if inferred:
            tag = inferred

    if tag is None:
        # discover from first in-range event
        probe_dir = None
        for event_dir in sorted(lp.iterdir()):
            if not event_dir.is_dir():
                continue
            evt_time = _parse_event_time_from_dirname(event_dir.name)
            if evt_time is None or not (start_dt <= evt_time <= end_dt):
                continue
            probe_dir = event_dir
            break
        if probe_dir:
            found = _discover_tags(probe_dir)
            if len(found) == 1:
                tag = found[0]
            elif len(found) > 1:
                raise ValueError(
                    f"Multiple tags present under {probe_dir}. Please pass tag=... (choices: {found})"
                )

    if not tag:
        raise ValueError("No tag provided and could not infer a tag.")

    rows: List[pd.DataFrame] = []
    missing = 0

    for event_dir in sorted(lp.iterdir()):
        if not event_dir.is_dir():
            continue
        evt_time = _parse_event_time_from_dirname(event_dir.name)
        if evt_time is None or not (start_dt <= evt_time <= end_dt):
            continue

        csv_path = event_dir / tag / f"source_{tag}.csv"
        if not csv_path.exists():
            missing += 1
            if verbose:
                print(f"[miss] {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            if verbose:
                print(f"[warn] Could not read {csv_path}: {e}")
            continue

        needed = {lat_col, lon_col, amp_col}
        if not needed.issubset(df.columns):
            if verbose:
                print(f"[skip] Missing {needed - set(df.columns)} in {csv_path}")
            continue

        # Optional filters
        m = pd.Series(True, index=df.index)
        if misfit_max is not None and "misfit" in df.columns:
            m &= df["misfit"] <= misfit_max
        if nsta_min is not None and "nsta" in df.columns:
            m &= df["nsta"] >= nsta_min
        if connectedness_min is not None and "connectedness" in df.columns:
            m &= df["connectedness"] >= connectedness_min
        if azgap_max is not None and "azgap" in df.columns:
            m &= df["azgap"] <= azgap_max
        if dr_min is not None:
            m &= df[amp_col] >= dr_min
        if dr_max is not None:
            m &= df[amp_col] <= dr_max

        df = df.loc[m, [lat_col, lon_col, amp_col]].copy()
        if df.empty:
            continue

        df[[lat_col, lon_col, amp_col]] = df[[lat_col, lon_col, amp_col]].astype(float)
        finite = np.isfinite(df[lat_col]) & np.isfinite(df[lon_col]) & np.isfinite(df[amp_col])
        df = df.loc[finite]
        if df.empty:
            continue

        df["event_time"] = evt_time
        rows.append(df)

    if not rows:
        raise ValueError(
            f"No matching data for tag='{tag}' after filtering "
            f"({pd.Timestamp(start_dt)}–{pd.Timestamp(end_dt)} UTC). Missing files count={missing}."
        )

    df_all = pd.concat(rows, ignore_index=True)

    # Titles / filenames
    start_str  = pd.Timestamp(start_dt).strftime("%Y-%m-%d %H:%M:%S")
    end_str    = pd.Timestamp(end_dt).strftime("%Y-%m-%d %H:%M:%S")
    start_safe = pd.Timestamp(start_dt).strftime("%Y%m%dT%H%M%S")
    end_safe   = pd.Timestamp(end_dt).strftime("%Y%m%dT%H%M%S")
    title = title_fmt.format(tag=tag, start=start_str, end=end_str)

    if outfile:
        outfile = str(outfile).format(tag=tag, start=start_safe, end=end_safe)
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    fig = plot_heatmap_colored(
        df_all,
        lat_col=lat_col,
        lon_col=lon_col,
        amp_col=amp_col,
        inventory=inventory,
        cmap=cmap,
        log_scale=log_scale,
        node_spacing_m=node_spacing_m,
        outfile=outfile,
        region=region,
        title=title,
        dem_tif=dem_tif,
        topo_kw=topo_kw,
    )

    return (fig, df_all) if return_df else fig


def make_asl_heatmaps_per_tag(
    startdate: Union[str, datetime, "UTCDateTime"],
    enddate:   Union[str, datetime, "UTCDateTime"],
    *,
    localprojectdir: Union[str, Path],
    tag: Optional[Union[str, Iterable[str]]] = None,
    default_tag_from: Optional[Union[str, Path]] = None,  # e.g., '/path/to/source_<tag>.csv'
    # CSV schema: t, lat, lon, DR, misfit, azgap, nsta, node_index, connectedness
    lat_col: str = "lat",
    lon_col: str = "lon",
    amp_col: str = "DR",
    # Row filters (only applied if column exists)
    misfit_max: Optional[float] = None,
    nsta_min: Optional[int] = None,
    connectedness_min: Optional[float] = None,
    azgap_max: Optional[float] = None,
    dr_min: Optional[float] = None,
    dr_max: Optional[float] = None,
    # Plot options
    inventory=None,
    cmap: str = "turbo",
    log_scale: bool = True,
    node_spacing_m: int = 50,
    region: Optional[List[float]] = None,
    dem_tif: Optional[str] = None,
    title_fmt: str = "ASL Heatmap: {tag} ({start}–{end} UTC)",
    outfile_pattern: Optional[str] = None,  # e.g., "heatmaps/{tag}_{start}_{end}.png"
    return_df: bool = True,
) -> Dict[str, Tuple["pygmt.Figure", Optional[pd.DataFrame]]]:
    """
    Build one heatmap per tag from per-event 'source_<tag>.csv' files
    found under LOCALPROJECTDIR between startdate and enddate (inclusive).
    """
    lp = Path(localprojectdir)
    if not lp.exists():
        raise FileNotFoundError(f"localprojectdir not found: {lp}")

    start_dt = _to_utc_datetime(startdate)
    end_dt   = _to_utc_datetime(enddate)
    if end_dt < start_dt:
        raise ValueError("enddate is earlier than startdate")

    tags = _ensure_iterable_tags(tag)
    if tags is None and default_tag_from:
        inferred = _infer_tag_from_filename(default_tag_from)
        if inferred:
            tags = [inferred]
    if tags is None:
        # Discover tags from the first in-range event
        probe_dir = None
        for event_dir in sorted(lp.iterdir()):
            if not event_dir.is_dir():
                continue
            evt_time = _parse_event_time_from_dirname(event_dir.name)
            if evt_time is None or not (start_dt <= evt_time <= end_dt):
                continue
            probe_dir = event_dir
            break
        if probe_dir:
            found = _discover_tags(probe_dir)
            if len(found) == 1:
                tags = [found[0]]
            elif len(found) > 1:
                raise ValueError(
                    f"Multiple tags present under {probe_dir}. "
                    f"Please pass tag=... (choices: {found})"
                )

    if not tags:
        raise ValueError("No tag provided and could not infer any tag.")

    start_str  = pd.Timestamp(start_dt).strftime("%Y-%m-%d %H:%M:%S")
    end_str    = pd.Timestamp(end_dt).strftime("%Y-%m-%d %H:%M:%S")
    start_safe = pd.Timestamp(start_dt).strftime("%Y%m%dT%H%M%S")
    end_safe   = pd.Timestamp(end_dt).strftime("%Y%m%dT%H%M%S")

    results: Dict[str, Tuple["pygmt.Figure", Optional[pd.DataFrame]]] = {}

    for tname in tags:
        rows: List[pd.DataFrame] = []
        missing_files = 0

        for event_dir in sorted(lp.iterdir()):
            if not event_dir.is_dir():
                continue
            evt_time = _parse_event_time_from_dirname(event_dir.name)
            if evt_time is None or not (start_dt <= evt_time <= end_dt):
                continue

            csv_path = event_dir / tname / f"source_{tname}.csv"
            if not csv_path.exists():
                missing_files += 1
                continue

            try:
                df_evt = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[warn] Could not read {csv_path}: {e}")
                continue

            needed = {"t", lat_col, lon_col, amp_col}
            if not needed.issubset(df_evt.columns):
                continue

            # Optional filters
            m = pd.Series(True, index=df_evt.index)
            if misfit_max is not None and "misfit" in df_evt.columns:
                m &= df_evt["misfit"] <= misfit_max
            if nsta_min is not None and "nsta" in df_evt.columns:
                m &= df_evt["nsta"] >= nsta_min
            if connectedness_min is not None and "connectedness" in df_evt.columns:
                m &= df_evt["connectedness"] >= connectedness_min
            if azgap_max is not None and "azgap" in df_evt.columns:
                m &= df_evt["azgap"] <= azgap_max
            if dr_min is not None:
                m &= df_evt[amp_col] >= dr_min
            if dr_max is not None:
                m &= df_evt[amp_col] <= dr_max

            df_evt = df_evt.loc[m, [lat_col, lon_col, amp_col]].copy()
            if df_evt.empty:
                continue

            df_evt["event_time"] = evt_time
            rows.append(df_evt)

        if not rows:
            raise ValueError(
                f"No matching data for tag='{tname}' after filtering "
                f"({start_str}–{end_str} UTC). Missing files (sample) count={missing_files}."
            )

        df_tag = pd.concat(rows, ignore_index=True)
        title = title_fmt.format(tag=tname, start=start_str, end=end_str)
        if outfile_pattern:
            outpath = str(outfile_pattern).format(tag=tname, start=start_safe, end=end_safe)
        else:
            outpath = None

        fig = plot_heatmap_colored(
            df_tag,
            lat_col=lat_col,
            lon_col=lon_col,
            amp_col=amp_col,
            inventory=inventory,
            cmap=cmap,
            log_scale=log_scale,
            node_spacing_m=node_spacing_m,
            outfile=outpath,
            region=region,
            title=title,
            dem_tif=dem_tif,
            topo_kw=None,
        )

        results[tname] = (fig, df_tag if return_df else None)

    return results


# ======================================================================
# ------------------------- AMPMAP DIRECTORY ---------------------------
# ======================================================================

def make_asl_heatmap_for_ampmap(
    startdate: Union[str, "UTCDateTime", pd.Timestamp, datetime],
    enddate:   Union[str, "UTCDateTime", pd.Timestamp, datetime],
    *,
    localprojectdir: Path,
    lat_col: str = "lat",
    lon_col: str = "lon",
    amp_col: str = "DR",
    # Optional row filters (only applied if column exists)
    misfit_max: Optional[float] = None,
    nsta_min: Optional[int] = None,
    connectedness_min: Optional[float] = None,
    azgap_max: Optional[float] = None,
    dr_min: Optional[float] = None,
    dr_max: Optional[float] = None,
    # Plot options
    inventory=None,
    cmap: str = "turbo",
    log_scale: bool = True,
    node_spacing_m: int = 50,
    region: Optional[List[float]] = None,
    dem_tif: Optional[Union[str, Path]] = None,
    title_fmt: str = "ASL Heatmap (AMPMAP): {start}–{end} UTC",
    outfile: Optional[Union[str, Path]] = None,  # e.g. "heatmaps/ampmap_{start}_{end}.png"
    # File discovery
    glob_pattern: str = "*.csv",
    recursive: bool = False,
    topo_kw: Optional[Dict[str, Any]] = None,
    return_df: bool = True,
) -> Tuple["pygmt.Figure", Optional[pd.DataFrame]]:
    """
    Load generic AMPMAP-like CSVs from a directory (not per-event layout), filter by
    time and optional quality fields, concatenate rows, and plot a heatmap.
    """
    lp = Path(localprojectdir)
    if not lp.exists():
        raise FileNotFoundError(f"localprojectdir not found: {lp}")

    start_dt = _to_utc_datetime(startdate)
    end_dt   = _to_utc_datetime(enddate)
    if end_dt < start_dt:
        raise ValueError("enddate is earlier than startdate")

    rows: List[pd.DataFrame] = []
    csv_iter = lp.rglob(glob_pattern) if recursive else lp.glob(glob_pattern)

    for csv_path in csv_iter:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[warn] Could not read {csv_path}: {e}")
            continue

        needed = {"t", lat_col, lon_col, amp_col}
        if not needed.issubset(df.columns):
            continue

        t_parsed = pd.to_datetime(df["t"], utc=True, errors="coerce")
        mask_time = (t_parsed >= pd.Timestamp(start_dt)) & (t_parsed <= pd.Timestamp(end_dt))
        df = df.loc[mask_time].copy()
        if df.empty:
            continue

        if misfit_max is not None and "misfit" in df.columns:
            df = df[df["misfit"] <= misfit_max]
        if nsta_min is not None and "nsta" in df.columns:
            df = df[df["nsta"] >= nsta_min]
        if connectedness_min is not None and "connectedness" in df.columns:
            df = df[df["connectedness"] >= connectedness_min]
        if azgap_max is not None and "azgap" in df.columns:
            df = df[df["azgap"] <= azgap_max]
        if dr_min is not None:
            df = df[df[amp_col] >= dr_min]
        if dr_max is not None:
            df = df[df[amp_col] <= dr_max]
        if df.empty:
            continue

        df = df[[lat_col, lon_col, amp_col]].astype(float)
        finite = np.isfinite(df[lat_col]) & np.isfinite(df[lon_col]) & np.isfinite(df[amp_col])
        df = df.loc[finite]
        if not df.empty:
            rows.append(df)

    if not rows:
        raise ValueError("No AMPMAP CSV rows matched the date range and filters in the specified directory.")

    df_all = pd.concat(rows, ignore_index=True)

    start_str = pd.Timestamp(start_dt).strftime("%Y-%m-%d %H:%M:%S")
    end_str   = pd.Timestamp(end_dt).strftime("%Y-%m-%d %H:%M:%S")
    title = title_fmt.format(start=start_str, end=end_str)

    if outfile:
        start_safe = pd.Timestamp(start_dt).strftime("%Y%m%dT%H%M%S")
        end_safe   = pd.Timestamp(end_dt).strftime("%Y%m%dT%H%M%S")
        outfile = str(outfile).format(start=start_safe, end=end_safe)

    fig = plot_heatmap_colored(
        df_all,
        lat_col=lat_col,
        lon_col=lon_col,
        amp_col=amp_col,
        inventory=inventory,
        cmap=cmap,
        log_scale=log_scale,
        node_spacing_m=node_spacing_m,
        outfile=outfile,
        region=region,
        title=title,
        dem_tif=dem_tif,
        topo_kw=topo_kw,
    )

    return (fig, df_all) if return_df else (fig, None)


# ======================================================================
# ------------------------ ENHANCED CATALOGS ---------------------------
# (unchanged from HEAD; plus event-rate writers)
# ======================================================================

def enhanced_catalogs_from_outputs(
    outputs_list: List[Dict[str, Any]],
    *,
    outdir: str,
    write_files: bool = True,
    load_waveforms: bool = False,
    primary_name: str = "catalog_primary",
    refined_name: str = "catalog_refined",
) -> Dict[str, Any]:

    os.makedirs(outdir, exist_ok=True)

    prim_recs, ref_recs = [], []

    def _append_rec(block: Dict[str, Any], bucket: list):
        if not block:
            return
        qml = block.get("qml")
        jjs = block.get("json")
        if not qml or not jjs:
            return
        if not (os.path.exists(qml) and os.path.exists(jjs)):
            return
        try:
            enh = EnhancedEvent.load(os.path.splitext(qml)[0])
            if not load_waveforms:
                enh.stream = None
            bucket.append(enh)
        except Exception:
            pass

    for out in outputs_list:
        _append_rec((out or {}).get("primary") or {}, prim_recs)
        _append_rec((out or {}).get("refined") or {}, ref_recs)

    prim_cat = EnhancedCatalog(
        events=[r.event for r in prim_recs],
        records=prim_recs,
        description="Primary ASL locations",
    )
    ref_cat = EnhancedCatalog(
        events=[r.event for r in ref_recs],
        records=ref_recs,
        description="Refined ASL locations",
    )

    primary_qml = refined_qml = primary_csv = refined_csv = None
    if write_files:
        if len(prim_cat):
            primary_qml = os.path.join(outdir, f"{primary_name}.qml")
            prim_cat.write(primary_qml, format="QUAKEML")
            primary_csv = os.path.join(outdir, f"{primary_name}.csv")
            prim_cat.export_csv(primary_csv)
        if len(ref_cat):
            refined_qml = os.path.join(outdir, f"{refined_name}.qml")
            ref_cat.write(refined_qml, format="QUAKEML")
            refined_csv = os.path.join(outdir, f"{refined_name}.csv")
            ref_cat.export_csv(refined_csv)

    return {
        "primary": prim_cat,
        "refined": ref_cat,
        "primary_qml": primary_qml,
        "refined_qml": refined_qml,
        "primary_csv": primary_csv,
        "refined_csv": refined_csv,
    }


# ======================================================================
# -------------------------- ASSIMILATION ------------------------------
# (scan OUTPUT_DIR/{event}/{ctag}; write EventRate CSV+plots)
# ======================================================================

def _iter_event_run_dirs(output_root: str | Path, ctag: str) -> List[Path]:
    root = Path(output_root)
    out: List[Path] = []
    for ev_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if ev_dir.name == ctag:  # skip accidental top-level tag folder
            continue
        run_dir = ev_dir / ctag
        if run_dir.is_dir():
            out.append(run_dir)
    return out


def gather_post_metrics(output_root: str | Path, ctag: str) -> str | None:
    """
    Concatenate all post_metrics.csv under OUTPUT_DIR/{event}/{ctag}/
    into OUTPUT_DIR/{ctag}/post_metrics.csv. Returns the output path or None.
    """
    root = Path(output_root)
    outdir = root / ctag
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "post_metrics.csv"

    dfs: List[pd.DataFrame] = []
    for run_dir in _iter_event_run_dirs(root, ctag):
        f = run_dir / "post_metrics.csv"
        if f.is_file():
            try:
                df = pd.read_csv(f)
                df["__event_dir"] = run_dir.parent.name  # traceability
                dfs.append(df)
            except Exception:
                pass

    if not dfs:
        print(f"[ASSIM] No post_metrics.csv found for ctag={ctag}")
        return None

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"[ASSIM] Wrote: {out_csv}  ({len(out)} rows)")
    return str(out_csv)


def gather_network_magnitudes(output_root: str | Path, ctag: str) -> str | None:
    """
    Concatenate all network_magnitudes.csv under OUTPUT_DIR/{event}/{ctag}/
    into OUTPUT_DIR/{ctag}/network_magnitudes.csv. Returns the output path or None.
    """
    root = Path(output_root)
    outdir = root / ctag
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "network_magnitudes.csv"

    dfs: List[pd.DataFrame] = []
    for run_dir in _iter_event_run_dirs(root, ctag):
        f = run_dir / "network_magnitudes.csv"
        if f.is_file():
            try:
                df = pd.read_csv(f)
                df["__event_dir"] = run_dir.parent.name
                dfs.append(df)
            except Exception:
                pass

    if not dfs:
        print(f"[ASSIM] No network_magnitudes.csv found for ctag={ctag}")
        return None

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"[ASSIM] Wrote: {out_csv}  ({len(out)} rows)")
    return str(out_csv)


def _discover_qml_json_pairs(run_dir: Path, base_name: str) -> tuple[str | None, str | None]:
    """
    Return (qml, json) under run_dir for base_name, or (None, None).
    """
    qml = run_dir / f"{base_name}.qml"
    jsn = run_dir / f"{base_name}.json"
    if qml.is_file() and jsn.is_file():
        return str(qml), str(jsn)
    return None, None


def _load_catalog_for_tag_kind(output_root: str | Path, ctag: str, *, refined: bool):
    """
    Walk OUTPUT_DIR/{event}/{ctag}/ and collect EnhancedEvents from
    {ctag}.qml/json or {ctag}_refined.qml/json depending on `refined`.
    """


    base = f"{ctag}_refined" if refined else f"{ctag}"
    records = []
    for run_dir in _iter_event_run_dirs(output_root, ctag):
        qml, jsn = _discover_qml_json_pairs(run_dir, base)
        if qml and jsn:
            try:
                enh = EnhancedEvent.load(os.path.splitext(qml)[0])
                enh.stream = None  # no waveforms during assimilation
                records.append(enh)
            except Exception as e:
                print(f"[ASSIM:WARN] Skipping {run_dir}: {e}")

    return EnhancedCatalog(events=[r for r in records], records=records,
                           description=("Refined" if refined else "Primary") + f" ASL locations ({ctag})")


def _write_event_rate_outputs(cat, outdir: Path, prefix: str,
                              *, er_cfg: EventRateConfig | None) -> Dict[str, Any]:
    """
    Build EventRate, write CSV and three plots (count, cumulative magnitude, dual).
    Returns dict with file paths.
    """
    out: Dict[str, Any] = {}
    if len(cat) == 0:
        return out

    cfg = er_cfg or EventRateConfig(interval="D", rolling=7, ema_alpha=None)
    er = cat.to_event_rate(config=cfg)

    # CSV
    er_csv = outdir / f"eventrate_{prefix}.csv"
    er.to_csv(str(er_csv))
    out["csv"] = str(er_csv)

    # Plots
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    er.plot_event_count(ax=ax1)
    fig1.tight_layout()
    f1 = outdir / f"eventrate_{prefix}_count.png"
    fig1.savefig(f1, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    er.plot_cumulative_magnitude(ax=ax2)
    fig2.tight_layout()
    f2 = outdir / f"eventrate_{prefix}_cumMag.png"
    fig2.savefig(f2, dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    er.plot_dual(ax=ax3)
    fig3.tight_layout()
    f3 = outdir / f"eventrate_{prefix}_dual.png"
    fig3.savefig(f3, dpi=150)
    plt.close(fig3)

    out["plots"] = [str(f1), str(f2), str(f3)]
    print(f"[ASSIM] EventRate ({prefix}) -> {er_csv}, {f1.name}, {f2.name}, {f3.name}")
    return out


def build_enhanced_catalogs_for_tag(
    output_root: str | Path,
    ctag: str,
    *,
    write_event_rates: bool = True,
    er_config: EventRateConfig | None = None,
) -> Dict[str, Any]:
    """
    Build catalogs by scanning OUTPUT_DIR/{event}/{ctag}/ for:
      - Primary:  {ctag}.qml + {ctag}.json
      - Refined:  {ctag}_refined.qml + {ctag}_refined.json

    Writes event-rate CSV + plots into OUTPUT_DIR/{ctag}/ if requested.
    """

    root = Path(output_root)
    outdir = root / ctag
    outdir.mkdir(parents=True, exist_ok=True)

    primary = _load_catalog_for_tag_kind(root, ctag, refined=False)
    refined = _load_catalog_for_tag_kind(root, ctag, refined=True)

    res: Dict[str, Any] = {
        "primary": primary,
        "refined": refined,
        "primary_eventrate": {},
        "refined_eventrate": {},
    }

    if write_event_rates:
        if len(primary):
            res["primary_eventrate"] = _write_event_rate_outputs(primary, outdir, "primary", er_cfg=er_config)
        if len(refined):
            res["refined_eventrate"] = _write_event_rate_outputs(refined, outdir, "refined", er_cfg=er_config)

    return res