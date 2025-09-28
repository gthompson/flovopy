# flovopy/asl/pipeline.py
from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Dict, Any, Optional,  List, Tuple, Iterable
import itertools
from dataclasses import dataclass, asdict
import time, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import Stream, read, read_inventory

from flovopy.processing.sam import VSAM
from flovopy.asl.ampcorr import AmpCorr, AmpCorrParams
from flovopy.asl.asl import ASL
from flovopy.asl.station_corrections import apply_interval_station_gains
from flovopy.asl.utils import _grid_mask_indices
from flovopy.core.mvo import dome_location  # <-- for sector apex
from flovopy.asl.distances import distances_signature, compute_or_load_distances
from flovopy.asl.grid import make_grid, nodegrid_from_channel_csvs
from flovopy.asl.station_corrections import load_station_gains_df
from flovopy.asl.map import plot_heatmap_colored
# Misfit backends (import directly)
from flovopy.asl.misfit import (
    StdOverMeanMisfit,
    R2DistanceMisfit,
    LinearizedDecayMisfit,
)
# from flovopy.asl.misfit import HuberMisfit  # if implemented

# Montserrat defaults (immutable tuple to avoid mutable-default gotchas)
DEFAULT_REGION = (-62.255, -62.135, 16.66, 16.84)  # (lon_min, lon_max, lat_min, lat_max)
# Later we can create configs like this and modify functions below to use them
@dataclass(frozen=True)
class VolcanoConfig:
    name: str
    region: tuple[float, float, float, float]
MONTSERRAT = VolcanoConfig("Montserrat", DEFAULT_REGION)

def asl_sausage(
    stream: Stream,
    event_dir: str,
    asl_config: dict,
    output_dir: str,  # kept for parity (not used here)
    dry_run: bool = False,
    peakf_override: Optional[float] = None,
    station_gains_df: Optional["pd.DataFrame"] = None,
    allow_station_fallback: bool = True,
    *,
    # NEW: enable a triangular (apex→sea) sector refinement pass
    refine_sector: bool = False,
):
    """
    Run ASL on a single (already preprocessed) event stream.

    Required keys in `asl_config`:
      - gridobj : Grid or NodeGrid
      - node_distances_km : dict[seed_id -> np.ndarray]
      - station_coords : dict[seed_id -> {latitude, longitude, elevation}]
      - ampcorr : AmpCorr
      - vsam_metric : str
      - window_seconds : float
    """
    print(f"[ASL] Preparing VSAM for event folder: {event_dir}")
    os.makedirs(event_dir, exist_ok=True)

    # 1) Apply station gains (optional)
    if station_gains_df is not None and len(station_gains_df):
        info = apply_interval_station_gains(
            stream,
            station_gains_df,
            allow_station_fallback=allow_station_fallback,
            verbose=True,
        )
        s = info.get("interval_start"); e = info.get("interval_end")
        used = info.get("used", []); miss = info.get("missing", [])
        print(f"[GAINS] Interval used: {s} → {e} | corrected {len(used)} traces; missing {len(miss)}")
    else:
        print("[GAINS] No station gains DataFrame provided; skipping.")

    # Ensure velocity units for downstream plots (don’t overwrite if already set)
    for tr in stream:
        if tr.stats.get("units") in (None, ""):
            tr.stats["units"] = "m/s"

    # 2) Build VSAM
    vsamObj = VSAM(stream=stream, sampling_interval=1.0)
    if len(vsamObj.dataframes) == 0:
        raise IOError("[ASL:ERR] No dataframes in VSAM object")

    if not dry_run:
        print("[ASL:PLOT] Writing VSAM preview (VSAM.png)")
        vsamObj.plot(equal_scale=False, outfile=os.path.join(event_dir, "VSAM.png"))
        plt.close('all')

    # 3) Decide event peakf
    if peakf_override is None:
        freqs = [df.attrs.get("peakf") for df in vsamObj.dataframes.values() if df.attrs.get("peakf") is not None]
        if freqs:
            peakf_event = int(round(sum(freqs) / len(freqs)))
            print(f"[ASL] Event peak frequency inferred from VSAM: {peakf_event} Hz")
        else:
            peakf_event = int(round(asl_config["ampcorr"].params.peakf))
            print(f"[ASL] Using default peak frequency from ampcorr: {peakf_event} Hz")
    else:
        peakf_event = int(round(peakf_override))
        print(f"[ASL] Using peak frequency override: {peakf_event} Hz")

    # 4) Amplitude corrections cache (swap if peakf differs)
    ampcorr: AmpCorr = asl_config["ampcorr"]
    if abs(float(ampcorr.params.peakf) - float(peakf_event)) > 1e-6:
        print(f"[ASL] Switching amplitude corrections to peakf={peakf_event} Hz (from {ampcorr.params.peakf})")

        grid_sig = asl_config["gridobj"].signature()
        dist_sig = distances_signature(asl_config["node_distances_km"])
        inv_sig  = tuple(sorted(asl_config["node_distances_km"].keys()))

        params = ampcorr.params
        new_params = AmpCorrParams(
            surface_waves=params.surface_waves,
            wave_speed_kms=params.wave_speed_kms,
            Q=params.Q,
            peakf=float(peakf_event),
            grid_sig=grid_sig,
            inv_sig=inv_sig,
            dist_sig=dist_sig,
            mask_sig=None,
            code_version=params.code_version,
        )

        ampcorr = AmpCorr(new_params, cache_dir=ampcorr.cache_dir)
        ampcorr.compute_or_load(asl_config["node_distances_km"])
        asl_config[f"ampcorr_peakf_{peakf_event}"] = ampcorr
    else:
        print(f"[ASL] Using existing amplitude corrections (peakf={ampcorr.params.peakf} Hz)")

    # 5) Build ASL object and inject geometry/corrections
    print("[ASL] Building ASL object…")
    aslobj = ASL(
        vsamObj,
        asl_config["vsam_metric"],
        asl_config["gridobj"],
        asl_config["window_seconds"],
    )

    idx = _grid_mask_indices(asl_config["gridobj"])
    if idx is not None and idx.size:
        aslobj._node_mask = idx

    aslobj.node_distances_km     = asl_config["node_distances_km"]
    aslobj.station_coordinates   = asl_config["station_coords"]
    aslobj.amplitude_corrections = ampcorr.corrections

    # keep parameters on the ASL for provenance/filenames
    aslobj.Q             = ampcorr.params.Q
    aslobj.peakf         = ampcorr.params.peakf
    aslobj.wavespeed_kms = ampcorr.params.wave_speed_kms
    aslobj.surfaceWaves  = ampcorr.params.surface_waves

    # 6) Locate
    meta = asl_config.get("dist_meta", {})
    station_coords = asl_config["station_coords"]

    print(f"[DIST] used_3d={meta.get('used_3d')}  "
          f"has_node_elevations={meta.get('has_node_elevations')}  "
          f"n_nodes={meta.get('n_nodes')}  n_stations={meta.get('n_stations')}")
    z_grid = getattr(asl_config["gridobj"], "node_elev_m", None)
    if z_grid is not None:
        print(f"[GRID] Node elevations: min={np.nanmin(z_grid):.1f} m  max={np.nanmax(z_grid):.1f} m")
    ze = [c.get("elevation", 0.0) for c in station_coords.values()]
    print(f"[DIST] Station elevations: min={min(ze):.1f} m  max={max(ze):.1f} m")

    print("[ASL] Locating source with fast_locate()…")
    # Pull optional settings from config if present
    min_sta = int(asl_config.get("min_stations", 3))
    misfit_backend = _resolve_misfit_backend(
        asl_config.get("misfit_engine", None),
        peakf_hz=float(asl_config.get("ampcorr", ampcorr).params.peakf),
        speed_kms=float(asl_config.get("ampcorr", ampcorr).params.wave_speed_kms),
    )
    aslobj.fast_locate(
        min_stations=min_sta,
        misfit_backend=misfit_backend,
    )
    print("[ASL] Location complete.")

    aslobj.print_event()

    # 7) Outputs (baseline)
    if not dry_run:
        qml_out = os.path.join(event_dir, f"event_Q{int(aslobj.Q)}_F{int(peakf_event)}.qml")
        print(f"[ASL:OUT] Writing QuakeML: {qml_out}")
        aslobj.save_event(outfile=qml_out)

        dem_tif_for_bmap = asl_config.get("dem_tif") or asl_config.get("dem_tif_for_bmap")

        print("[ASL:PLOT] Writing map and diagnostic plots…")
        aslobj.plot(
            zoom_level=0,
            threshold_DR=0.0,
            scale=0.2,
            join=True,
            number=0,
            add_labels=True,
            stations=[tr.stats.station for tr in stream],
            outfile=os.path.join(event_dir, f"map_Q{int(aslobj.Q)}_F{int(peakf_event)}.png"),
            dem_tif=dem_tif_for_bmap,
            simple_basemap=True,
            region=asl_config.get("region", DEFAULT_REGION),  # ensure consistent extent
        )
        plt.close('all')

        print("[ASL:SOURCE_TO_CSV] Writing source to a CSV…")
        aslobj.source_to_csv(os.path.join(event_dir, f"source_Q{int(aslobj.Q)}_F{int(peakf_event)}.csv"))

        print("[ASL:PLOT_REDUCED_DISPLACEMENT]")
        aslobj.plot_reduced_displacement(
            outfile=os.path.join(event_dir, f"reduced_disp_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
        )
        plt.close()

        print("[ASL:PLOT_MISFIT]")
        aslobj.plot_misfit(
            outfile=os.path.join(event_dir, f"misfit_Q{int(aslobj.Q)}_F{int(peakf_event)}.png")
        )
        plt.close()

        print("[ASL:PLOT_MISFIT_HEATMAP]")
        aslobj.plot_misfit_heatmap(
            outfile=os.path.join(event_dir, f"misfit_heatmap_Q{int(aslobj.Q)}_F{int(peakf_event)}.png"),
            region=asl_config.get("region", DEFAULT_REGION),  # ensure consistent extent
        )
        plt.close('all')

    # 8) OPTIONAL: sector refinement pass (triangular wedge from dome toward sea)
    if refine_sector:
        print("[ASL] Refinement pass: sector wedge from dome apex…")
        try:
            # Use dome apex as the sector origin
            apex_lat = float(dome_location["lat"])
            apex_lon = float(dome_location["lon"])

            # call refine_and_relocate() with sector mask + Viterbi smoothing defaults
            aslobj.refine_and_relocate(
                mask_method="sector",
                apex_lat=apex_lat, apex_lon=apex_lon,
                length_km=8.0, inner_km=0.0, half_angle_deg=25.0,
                prefer_misfit=True,
                temporal_smooth_mode="viterbi",
                temporal_smooth_win=7,
                viterbi_lambda_km=8.0,
                viterbi_max_step_km=30.0,
                misfit_backend=misfit_backend,            # ← new
                min_stations=min_sta,                     # ← new
                verbose=True,
            )
            print("[ASL] Sector refinement complete.")
        except Exception:
            print("[ASL:ERR] Sector refinement failed.")
            raise

        if not dry_run:
            # extra outputs with a `_refined` suffix
            qml_ref = os.path.join(event_dir, f"event_Q{int(aslobj.Q)}_F{int(peakf_event)}_refined.qml")
            print(f"[ASL:OUT] Writing refined QuakeML: {qml_ref}")
            aslobj.save_event(outfile=qml_ref)

            dem_tif_for_bmap = asl_config.get("dem_tif") or asl_config.get("dem_tif_for_bmap")

            print("[ASL:PLOT] Writing refined map and diagnostics…")
            aslobj.plot(
                zoom_level=0,
                threshold_DR=0.0,
                scale=0.2,
                join=True,
                number=0,
                add_labels=True,
                stations=[tr.stats.station for tr in stream],
                outfile=os.path.join(event_dir, f"map_Q{int(aslobj.Q)}_F{int(peakf_event)}_refined.png"),
                dem_tif=dem_tif_for_bmap,
                simple_basemap=True,
                region=asl_config.get("region", DEFAULT_REGION),  # ensure consistent extent
            )
            plt.close('all')

            aslobj.source_to_csv(os.path.join(event_dir, f"source_Q{int(aslobj.Q)}_F{int(peakf_event)}_refined.csv"))

            aslobj.plot_reduced_displacement(
                outfile=os.path.join(event_dir, f"reduced_disp_Q{int(aslobj.Q)}_F{int(peakf_event)}_refined.png")
            )
            plt.close()

            aslobj.plot_misfit(
                outfile=os.path.join(event_dir, f"misfit_Q{int(aslobj.Q)}_F{int(peakf_event)}_refined.png")
            )
            plt.close()

            aslobj.plot_misfit_heatmap(
                outfile=os.path.join(event_dir, f"misfit_heatmap_Q{int(aslobj.Q)}_F{int(peakf_event)}_refined.png"),
                region=asl_config.get("region", DEFAULT_REGION),  # ensure consistent extent
            )
            plt.close('all')

    if not dry_run and asl_config.get("interactive", False):
        input("[ASL] Press Enter to continue to next event…")

def prepare_asl_context(
    *,
    sweep_or_cfg,
    inventory_xml: str,
    output_base: str,
    node_spacing_m: int,
    peakf: float,
    regular_grid_dem: Optional[str],
    channels_dir: Optional[str],
    channels_step_m: float,
    channels_dem_tif: Optional[str],
    dem_cache_dir: Optional[str],
    dem_tif_for_bmap: Optional[str],
    region: tuple[float, float, float, float] = DEFAULT_REGION,   # ← NEW default
) -> tuple[dict, "Inventory", str]:
    """
    Build grid, compute/load distances + ampcorr once, return (asl_config, inventory, outdir).
    """


    tag = sweep_or_cfg.tag()
    outdir = Path(output_base) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    inv = read_inventory(inventory_xml)

    # Grid or NodeGrid
    if sweep_or_cfg.grid_kind == "streams":
        if not channels_dir:
            raise ValueError("grid_kind='streams' requires channels_dir")
        gridobj = nodegrid_from_channel_csvs(
            channels_dir=channels_dir,
            step_m=channels_step_m,
            dem_tif=channels_dem_tif,
            approx_spacing_m=channels_step_m,
            max_points=None,
        )
    else:
        dem_spec = None
        if regular_grid_dem:
            if regular_grid_dem.startswith("pygmt:"):
                res = regular_grid_dem.split(":", 1)[1] or "01s"
                dem_spec = ("pygmt", {"resolution": res, "cache_dir": dem_cache_dir, "tag": res})
            elif regular_grid_dem.startswith("geotiff:"):
                path = regular_grid_dem.split(":", 1)[1]
                dem_spec = ("geotiff", {"path": path, "tag": Path(path).name})
        gridobj = make_grid(
            center_lat=dome_location["lat"],
            center_lon=dome_location["lon"],
            node_spacing_m=node_spacing_m,
            grid_size_lat_m=10_000,
            grid_size_lon_m=14_000,
            dem=dem_spec,
        )
        try:
            gridobj.apply_land_mask_from_dem(sea_level=1.0)
        except Exception:
            pass

    # Distances (shared cache per sweep tag + 2D/3D)
    cache_root = Path(output_base) / "_nbcache" / tag
    dist_cache_dir = cache_root / "distances" / ("3d" if sweep_or_cfg.dist_mode.lower() == "3d" else "2d")
    dist_cache_dir.mkdir(parents=True, exist_ok=True)

    node_distances_km, station_coords, dist_meta = compute_or_load_distances(
        gridobj,
        inventory=inv,
        stream=None,
        cache_dir=str(dist_cache_dir),
        force_recompute=False,
        use_elevation=(sweep_or_cfg.dist_mode.lower() == "3d"),
    )

    # AmpCorr (shared cache)
    ampcorr_cache = cache_root / "ampcorr"
    ampcorr_cache.mkdir(parents=True, exist_ok=True)
    params = AmpCorrParams(
        surface_waves=(sweep_or_cfg.wave_kind == "surface"),
        wave_speed_kms=sweep_or_cfg.speed,
        Q=sweep_or_cfg.Q,
        peakf=float(peakf),
        grid_sig=gridobj.signature(),
        inv_sig=tuple(sorted(node_distances_km.keys())),
        dist_sig=distances_signature(node_distances_km),
        mask_sig=None,
        code_version="v1",
    )
    ampcorr = AmpCorr(params, cache_dir=str(ampcorr_cache))
    ampcorr.compute_or_load(node_distances_km, inventory=inv)
    ampcorr.validate_against_nodes(gridobj.gridlat.size)

    asl_config = {
        "window_seconds": 5,  # caller will override if needed
        "min_stations": 5,    # caller will override if needed
        "Q": sweep_or_cfg.Q,
        "surfaceWaveSpeed_kms": sweep_or_cfg.speed,
        "vsam_metric": "VT",  # caller will override if needed
        "gridobj": gridobj,
        "node_distances_km": node_distances_km,
        "station_coords": station_coords,
        "ampcorr": ampcorr,
        "inventory": inv,
        "interactive": False,
        "numtrials": 200,
        "dist_meta": dist_meta,
        "misfit_engine": sweep_or_cfg.misfit_engine,
        "dem_tif_for_bmap": dem_tif_for_bmap,
    }

    # Optional grid preview (was hard-coded)
    try:
        preview_png = outdir / f"grid_preview_{getattr(gridobj, 'id', 'grid')[:8]}.png"
        topo_kw = {
            "inv": inv,
            "add_labels": False,
            "topo_color": True,
            "region": region,                 # ← uses parameter
            "DEM_DIR": dem_cache_dir,
        }
        if dem_tif_for_bmap:
            topo_kw["dem_tif"] = dem_tif_for_bmap
        gridobj.plot(show=False, topo_map_kwargs=topo_kw, outfile=str(preview_png))
    except Exception:
        pass

    # stash region for downstream plotting if you want
    asl_config["region"] = region
    return asl_config, inv, str(outdir)


def run_single_event(
    cfg: "MiniConfig",
    *,
    mseed_file: str,
    inventory_xml: str,
    output_base: str,
    node_spacing_m: int = 50,
    metric: str = "VT",
    window_seconds: int = 5,
    peakf: float = 8.0,
    channels_dir: Optional[str] = None,
    channels_step_m: float = 100.0,
    channels_dem_tif: Optional[str] = None,
    regular_grid_dem: Optional[str] = "pygmt:01s",
    dem_tif_for_bmap: Optional[str] = None,
    simple_basemap: bool = True,
    refine_sector: bool = False,
    region: tuple[float, float, float, float] = DEFAULT_REGION,   # ← REPLACES ISLAND_REGION
    MIN_STATIONS: int = 5,
    GLOBAL_CACHE: str = None,
) -> Dict[str, Any]:
    """Minimal, notebook-friendly runner (delegates prep to prepare_asl_context)."""


    t0 = time.time()
    outdir = Path(output_base) / cfg.tag()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # One-shot prep (grid + distances + ampcorr + optional grid preview)
        asl_config, inv, outdir_str = prepare_asl_context(
            sweep_or_cfg=cfg,
            inventory_xml=inventory_xml,
            output_base=output_base,
            node_spacing_m=node_spacing_m,
            peakf=peakf,
            regular_grid_dem=regular_grid_dem,
            channels_dir=channels_dir,
            channels_step_m=channels_step_m,
            channels_dem_tif=channels_dem_tif,
            dem_cache_dir=(GLOBAL_CACHE and os.path.join(GLOBAL_CACHE, "dem")) or None,
            dem_tif_for_bmap=dem_tif_for_bmap,
            region=region,   # ← pass through
        )

        # Per-run overrides
        asl_config["window_seconds"] = window_seconds
        asl_config["min_stations"]   = MIN_STATIONS
        asl_config["vsam_metric"]    = metric

        # Read stream and run once
        st = read(mseed_file).select(component="Z")
        if len(st) < MIN_STATIONS:
            raise RuntimeError(f"Not enough stations: {len(st)} < {MIN_STATIONS}")

        event_dir = Path(outdir_str) / Path(mseed_file).stem
        event_dir.mkdir(exist_ok=True)

        print(f"[ASL] Running single event: {mseed_file}")
        asl_sausage(
            stream=st,
            event_dir=str(event_dir),
            asl_config=asl_config,
            output_dir=outdir_str,
            dry_run=False,
            peakf_override=None,
            station_gains_df=None,
            allow_station_fallback=True,
            refine_sector=refine_sector,
        )

        # Summary
        src_csv = next((p for p in event_dir.glob("source_*.csv")), None)
        qml     = next((p for p in event_dir.glob("event_*.qml")), None)
        heat    = next((p for p in event_dir.glob("map_Q*_F*.png")), None)
        misf    = next((p for p in event_dir.glob("*misfit*heatmap*.png")), None)

        summ = {
            "tag": cfg.tag(),
            "outdir": str(outdir_str),
            "event_dir": str(event_dir),
            "source_csv": str(src_csv) if src_csv else None,
            "event_qml": str(qml) if qml else None,
            "map_png_exists": bool(heat),
            "misfit_png_exists": bool(misf),
            "elapsed_s": round(time.time() - t0, 2),
        }
        print(f"[ASL] Single-event summary: {summ}")
        return summ

    except Exception as e:
        traceback.print_exc()
        return {
            "tag": cfg.tag(),
            "error": f"{type(e).__name__}: {e}",
            "outdir": str(outdir),
        }
    

def run_all_events(
    *,
    sweep: SweepPoint,
    input_dir: str,
    base_output_dir: str,
    inventory_path: str,
    min_stations: int = 5,
    metric: str = "VT",
    window_seconds: int = 5,
    peakf: float = 8.0,
    node_spacing_m: int = 50,
    max_events: Optional[int] = None,
    channels_dir: Optional[str] = None,
    channels_step_m: float = 100.0,
    channels_dem_tif: Optional[str] = None,
    regular_grid_dem: Optional[str] = None,
    station_gains_csv: Optional[str] = None,
    allow_station_fallback: bool = True,
    dem_tif_for_bmap: Optional[str] = None,
    dry_run: bool = False,
    dem_cache: Optional[str] = None,
    region: tuple[float, float, float, float] = DEFAULT_REGION,    # ← NEW default
) -> str:
    """
    Execute one configuration and return the run directory.

    Mirrors all stdout/stderr to <run_dir>/run.log with timestamps.
    """
    import sys, io
    from contextlib import redirect_stdout, redirect_stderr

    class _Tee(io.TextIOBase):
        """Write to two text streams; line-buffered timestamp prefix."""
        def __init__(self, a, b):
            self.a = a; self.b = b; self._buf = ""
        def write(self, s):
            if not isinstance(s, str): s = str(s)
            self._buf += s
            lines = self._buf.splitlines(keepends=True)
            self._buf = "" if (not lines or lines[-1].endswith("\n")) else lines.pop()
            for ln in lines:
                ts = time.strftime("[%Y-%m-%d %H:%M:%S] ")
                self.a.write(ts + ln); self.b.write(ts + ln)
            return len(s)
        def flush(self):
            self.a.flush(); self.b.flush()

    # Shared cache root (still used by prepare_asl_context for DEM tiles)
    shared_cache = dem_cache or os.environ.get("FLOVOPY_CACHE") \
                   or os.path.join(os.path.expanduser("~"), ".flovopy_cache")
    os.makedirs(shared_cache, exist_ok=True)

    run_dir = ensure_dir(os.path.join(base_output_dir, sweep.tag()))
    log_path = os.path.join(run_dir, "run.log")

    with open(log_path, "a", buffering=1, encoding="utf-8") as _log, \
         redirect_stdout(_Tee(sys.stdout, _log)), \
         redirect_stderr(_Tee(sys.stderr, _log)):

        print("\n" + "=" * 100)
        print("[RUN] ", sweep.tag())
        print("      ", asdict(sweep))
        print("=" * 100)
        print(f"[LOG] Capturing output to: {log_path}")

        # ---- One-shot run prep (grid, distances, ampcorr, preview, etc.)
        asl_config, inv, run_dir = prepare_asl_context(
            sweep_or_cfg=sweep,
            inventory_xml=inventory_path,
            output_base=base_output_dir,
            node_spacing_m=node_spacing_m,
            peakf=peakf,
            regular_grid_dem=regular_grid_dem,
            channels_dir=channels_dir,
            channels_step_m=channels_step_m,
            channels_dem_tif=channels_dem_tif,
            dem_cache_dir=os.path.join(shared_cache, "dem"),
            dem_tif_for_bmap=dem_tif_for_bmap,
            region=region,   # ← pass through
        )

        # Per-run overrides
        asl_config["window_seconds"] = window_seconds
        asl_config["min_stations"]   = min_stations
        asl_config["vsam_metric"]    = metric

        # Optional station gains table
        gains_df = load_station_gains_df(station_gains_csv) \
            if (getattr(sweep, "station_corr", False) and station_gains_csv) else None
        if gains_df is not None:
            print(f"[GAINS] Loaded station gains: {len(gains_df)} rows")

        # ---- Event loop
        file_count = 0
        for file_num, mseed_path in enumerate(sorted(find_event_files(input_dir))):
            try:
                st = read(mseed_path).select(component="Z")
                if len(st) < min_stations:
                    print(f"[SKIP] Not enough stations ({len(st)}): {mseed_path}")
                    continue

                stime = min(tr.stats.starttime for tr in st)
                event_dir = ensure_dir(os.path.join(run_dir, stime.strftime("%Y%m%dT%H%M%S")))

                if not dry_run:
                    try:
                        st.plot(equal_scale=False, outfile=os.path.join(event_dir, "raw.png"))
                    except Exception:
                        pass

                asl_sausage(
                    stream=st,
                    event_dir=event_dir,
                    asl_config=asl_config,
                    output_dir=run_dir,
                    dry_run=dry_run,
                    peakf_override=None,
                    station_gains_df=gains_df,
                    allow_station_fallback=allow_station_fallback,
                    refine_sector=getattr(sweep, "refine_sector", False),  # pass-through
                )
                print(f"[✓] Processed: {mseed_path}")
                file_count += 1

            except Exception as e:
                print(f"[ERR] Failed on {mseed_path}: {e}")

            if max_events and (file_num + 1) >= max_events:
                print(f"[STOP] Reached max-events limit: {max_events}")
                break

        print(f"[DONE] Run complete. Events processed: {file_count}")

        # ---- Heatmap aggregation & plot
        try:
            df = collect_node_results(run_dir)
            if not df.empty:
                heat_png = os.path.join(run_dir, "heatmap_energy.png")
                print("[HEATMAP] Generating overall heatmap…")
                plot_heatmap_colored(
                    df,
                    lat_col="latitude",
                    lon_col="longitude",
                    amp_col="amplitude",
                    zoom_level=0,
                    inventory=inv,
                    log_scale=True,
                    node_spacing_m=node_spacing_m,
                    outfile=heat_png,
                    dem_tif=dem_tif_for_bmap,
                    title=f"Energy Heatmap — {sweep.tag()}",
                    region=region,  # ← ensure same extent
                )
                print("[HEATMAP] Wrote:", heat_png)
            else:
                print("[HEATMAP] No data found to plot.")
        except Exception as e:
            print("[HEATMAP:WARN] Failed to generate heatmap:", e)

        return run_dir
    


# From 13_run_sweep.py

RESULT_PATTERNS = (
    # add/modify to match what asl_sausage writes
    "**/*node_results.csv",
    "**/*node_metrics.csv",
    "**/*vsam_nodes.csv",
)

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def collect_node_results(run_outdir: str) -> pd.DataFrame:
    """
    Crawl a run's output directory and collect node-level results with
    columns at least: latitude, longitude, amplitude.
    """
    matches: List[str] = []
    for pat in RESULT_PATTERNS:
        matches.extend(glob(os.path.join(run_outdir, pat), recursive=True))

    dfs: List[pd.DataFrame] = []
    for f in sorted(set(matches)):
        try:
            if f.lower().endswith(".parquet"):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        have = all(k in cols for k in ["latitude", "longitude", "amplitude"])
        if not have:
            continue
        dfs.append(df.rename(columns=cols))  # normalize to canonical names

    if not dfs:
        print(f"[HEATMAP] No node CSVs matched under: {run_outdir}")
        return pd.DataFrame(columns=["latitude", "longitude", "amplitude"])

    cat = pd.concat(dfs, ignore_index=True)
    # keep only required columns, drop NaNs
    keep = cat[["latitude", "longitude", "amplitude"]].dropna()
    return keep


# ----------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SweepPoint:
    grid_kind: str           # 'streams' or 'regular'
    wave_kind: str           # 'surface' or 'body'
    station_corr: bool       # True/False
    speed: float             # km/s
    Q: int
    dist_mode: str           # '2d' or '3d'
    misfit_engine: str       # e.g., 'l2', 'huber' (passed through)

    def tag(self) -> str:
        """Short, filesystem-safe label for this combination."""
        parts = [
            f"G_{self.grid_kind}",
            f"W_{self.wave_kind}",
            f"SC_{'on' if self.station_corr else 'off'}",
            f"V_{self.speed:g}",
            f"Q_{self.Q}",
            f"D_{self.dist_mode}",
            f"M_{self.misfit_engine}",
        ]
        return "__".join(parts)

def generate_sweep(
    *,
    speeds: Tuple[float, float] = (1.5, 3.2),
    Qs: Tuple[int, int] = (50, 200),
    grid_kinds: Tuple[str, str] = ("streams", "regular"),
    wave_kinds: Tuple[str, str] = ("surface", "body"),
    station_corr_opts: Tuple[bool, bool] = (True, False),
    dist_modes: Tuple[str, str] = ("2d", "3d"),
    misfit_engines: Tuple[str, str] = ("l2", "huber"),
) -> List[SweepPoint]:
    pts = []
    for g, w, sc, v, q, d, m in itertools.product(
        grid_kinds, wave_kinds, station_corr_opts, speeds, Qs, dist_modes, misfit_engines
    ):
        pts.append(SweepPoint(g, w, sc, float(v), int(q), d, m))
    return pts

def find_event_files(root_dir: str, extensions=(".cleaned", ".mseed")) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extensions):
                files.append(os.path.join(dirpath, filename))
    return sorted(files)


def _resolve_misfit_backend(name_or_obj, *, peakf_hz: float | None = None, speed_kms: float | None = None):
    """
    Accepts a backend instance or a string name and returns a backend instance.
    Known names:
      'l2', 'std', 'som', 'std_over_mean' -> StdOverMeanMisfit()
      'r2', 'r2distance'                  -> R2DistanceMisfit()
      'lin', 'linearized', 'linearized_decay' -> LinearizedDecayMisfit(f_hz=peakf_hz or 8.0, v_kms=speed_kms or 1.5)
      'huber' (if available)              -> HuberMisfit()
    """
    if name_or_obj is None:
        return None
    if not isinstance(name_or_obj, str):
        return name_or_obj

    key = name_or_obj.strip().lower()
    if key in ("l2", "std", "som", "std_over_mean"):
        return StdOverMeanMisfit()
    if key in ("r2", "r2distance"):
        return R2DistanceMisfit()
    if key in ("lin", "linearized", "linearized_decay"):
        return LinearizedDecayMisfit(
            f_hz=(peakf_hz if peakf_hz is not None else 8.0),
            v_kms=(speed_kms if speed_kms is not None else 1.5),
            alpha=1.0,  # pure linearized cost; tune if you want blending
        )
    # if you ship a Huber version:
    # if key in ("huber",):
    #     return HuberMisfit()
    print(f"[ASL:MISFIT] Unknown backend '{name_or_obj}', using default StdOverMean.")
    return StdOverMeanMisfit()