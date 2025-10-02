# flovopy/asl/pipeline.py
from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Dict, Any, Optional,  List, Tuple
import itertools
from dataclasses import dataclass, asdict
import time
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from obspy import Stream, read, read_inventory, Inventory

from flovopy.processing.sam import SAM, RSAM, VSAM, DSAM
from flovopy.asl.ampcorr import AmpCorr, AmpCorrParams, summarize_ampcorr_ranges
from flovopy.asl.asl import ASL
from flovopy.asl.station_corrections import apply_interval_station_gains
from flovopy.asl.utils import _grid_mask_indices
from flovopy.asl.distances import distances_signature, compute_or_load_distances
from flovopy.asl.grid import Grid, summarize_station_node_distances
from flovopy.asl.station_corrections import load_station_gains_df
from flovopy.asl.map import plot_heatmap_colored
# Misfit backends (import directly)
from flovopy.asl.misfit import (
    StdOverMeanMisfit,
    R2DistanceMisfit,
    LinearizedDecayMisfit,
)
# from flovopy.asl.misfit import HuberMisfit  # if implemented
from flovopy.enhanced.event import EnhancedEvent, EnhancedEventMeta
from flovopy.enhanced.catalog import EnhancedCatalog



def build_asl_config(
    *,
    sweep_or_cfg,
    inventory_xml: Inventory | str | Path,
    output_base: str | Path,
    peakf: float,
    gridobj,                         # Grid
    sam_class=None,                  # e.g. VSAM (a class, not an instance)
    sam_metric: str = "mean",
    window_seconds: float = 5.0,
    min_stations: int = 5,
    global_cache: str | Path | None = None,
    debug: bool = False,
) -> Tuple[dict, Inventory, str]:
    """
    Compute/load distances + ampcorr once, return (asl_config, inventory, outdir_str).

    Notes
    -----
    - Distances use the mask-aware, vectorized pipeline; dense output (masked nodes = NaN).
    - AmpCorr corrections match full grid length; masked nodes remain NaN.
    - Input Grid should be unmasked at creation; runtime masks are honored by callers.
    """
    # --- tag and output dir ---
    tag = sweep_or_cfg.tag()
    output_base = Path(output_base)
    outdir = output_base / tag
    outdir.mkdir(parents=True, exist_ok=True)

    # --- inventory load/normalize ---
    if isinstance(inventory_xml, Inventory):
        inv = inventory_xml
    else:
        inv = read_inventory(str(inventory_xml))
    if debug:
        print('[build_asl_config] Inventory:', inv)

    # --- cache root ---
    cache_root = (Path(global_cache) if global_cache else output_base) / tag
    cache_root.mkdir(parents=True, exist_ok=True)

    # --- distances (mask-aware & vectorized; 3D if requested) ---
    dist_cache_dir = cache_root / "distances" / ("3d" if sweep_or_cfg.dist_mode.lower() == "3d" else "2d")
    dist_cache_dir.mkdir(parents=True, exist_ok=True)

    # Uses your new compute_or_load_distances() that calls distances_mask_aware(...)
    print('[build_asl_config] Computing distances to each node from each station')
    node_distances_km, station_coords, dist_meta = compute_or_load_distances(
        gridobj,
        inventory=inv,
        stream=None,
        cache_dir=str(dist_cache_dir),
        force_recompute=False,
        use_elevation=(sweep_or_cfg.dist_mode.lower() == "3d"),
    )
    if debug:
        print('[build_asl_config]',summarize_station_node_distances(node_distances_km, reduce_to_station=True))

    # --- ampcorr (vectorized fused correction; mask-aware via NaNs) ---
    ampcorr_cache = cache_root / "ampcorr"
    ampcorr_cache.mkdir(parents=True, exist_ok=True)

    # mask signature (valid_count, total) or None
    mask_idx = getattr(gridobj, "_node_mask_idx", None)
    total_nodes = int(gridobj.gridlat.size)
    mask_sig = (int(mask_idx.size), total_nodes) if mask_idx is not None else None

    params = AmpCorrParams(
        assume_surface_waves=(sweep_or_cfg.wave_kind == "surface"),
        wave_speed_kms=float(sweep_or_cfg.speed),
        Q=sweep_or_cfg.Q,
        peakf=float(peakf),
        grid_sig=gridobj.signature(),
        inv_sig=tuple(sorted(node_distances_km.keys())),
        dist_sig=distances_signature(node_distances_km),
        mask_sig=mask_sig,
        code_version="v1",
    )

    ampcorr = AmpCorr(params, cache_dir=str(ampcorr_cache))
    print('[build_asl_config] Computing amplitude corrections for each node from each station')
    ampcorr.compute_or_load(node_distances_km, inventory=inv)

    # Validate length against full grid (we store dense vectors with NaNs on masked nodes)
    ampcorr.validate_against_nodes(total_nodes)

    # After ampcorr.compute_or_load(...)
    if debug:
        print('[build_asl_config]',summarize_ampcorr_ranges(
            ampcorr.corrections,
            total_nodes=gridobj.gridlat.size,   # dense grid length (masked nodes are NaN)
            reduce_to_station=True,             # collapse NET.STA across channels
            include_percentiles=True,
            sort_by="min_corr",
        ))
   

    return {
        "tag": tag,
        "window_seconds": float(window_seconds),
        "min_stations": int(min_stations),
        "gridobj": gridobj,
        "node_distances_km": node_distances_km,
        "station_coords": station_coords,
        "ampcorr": ampcorr,
        "inventory": inv,
        "dist_meta": dist_meta,
        "misfit_engine": sweep_or_cfg.misfit_engine,
        "outdir": str(outdir),
        "sam_class": (sam_class if sam_class is not None else VSAM),
        "sam_metric": sam_metric,
    }


# helper method for outputting source data after fast_locate() or refine_and_relocate()
def _asl_output_source_results(
    aslobj,
    *,
    stream,                 # ObsPy Stream used for station labels on plots
    event_dir: str,
    asl_config: dict,
    peakf_event: int,
    suffix: str = "",       # "" or "_refined"
    topo_kw: dict = None,
    show: bool = True,
    debug: bool = False,
) -> dict:
    """
    Save QuakeML+JSON (EnhancedEvent), CSV, and all diagnostic plots for the current ASL source.
    Returns a dict of the produced file paths (some may be None if failures occur).
    """
    os.makedirs(event_dir, exist_ok=True)

    if debug:
        print('source sanity check:\n', aslobj.source_to_dataframe().describe())

    # Title / filenames
    base_title = f"ASL event (Q={int(aslobj.Q)}, F={int(aslobj.peakf)} Hz, v={aslobj.wave_speed_kms:g} km/s)"
    if suffix:
        base_title += " refined"
    base_name = f"event_Q{int(aslobj.Q)}_F{int(peakf_event)}{suffix}"

    # QuakeML + JSON sidecar (EnhancedEvent)
    ee_meta = EnhancedEventMeta()
    aslobj.event = EnhancedEvent.from_asl(
        aslobj,
        meta=ee_meta,
        stream=stream,                                  # optional
        inventory=asl_config.get("inventory"),          # optional
        title_comment=base_title,
    )
    qml_out, json_out = aslobj.event.save(
        event_dir, base_name,
        write_quakeml=True,
        write_obspy_json=asl_config.get("write_event_obspy_json", False),
        include_trajectory_in_sidecar=asl_config.get("include_trajectory", True),
    )    
    print(f"[ASL:OUT] Saved QuakeML: {qml_out}")
    print(f"[ASL:OUT] Saved JSON sidecar: {json_out}")

    # Map
    map_png = os.path.join(event_dir, f"map_Q{int(aslobj.Q)}_F{int(peakf_event)}{suffix}.png")
    print("[ASL:PLOT] Writing map and diagnostic plots…")
    aslobj.plot(
        topo_kw=topo_kw,
        threshold_DR=0.0,
        scale=0.2,
        join=False,
        number=0,
        stations = [tr.stats.station for tr in (stream or [])],
        outfile=map_png,
        show=show,
    )
    plt.close('all')

    # CSV
    csv_out = os.path.join(event_dir, f"source_Q{int(aslobj.Q)}_F{int(peakf_event)}{suffix}.csv")
    print("[ASL:SOURCE_TO_CSV] Writing source to a CSV…")
    aslobj.source_to_csv(csv_out)

    # Reduced displacement
    rd_png = os.path.join(event_dir, f"reduced_disp_Q{int(aslobj.Q)}_F{int(peakf_event)}{suffix}.png")
    print("[ASL:PLOT_REDUCED_DISPLACEMENT]")
    aslobj.plot_reduced_displacement(outfile=rd_png, show=show)
    plt.close()

    # Misfit (line)
    mis_png = os.path.join(event_dir, f"misfit_Q{int(aslobj.Q)}_F{int(peakf_event)}{suffix}.png")
    print("[ASL:PLOT_MISFIT]")
    aslobj.plot_misfit(outfile=mis_png, show=show)
    plt.close()

    # Misfit heatmap
    mh_png = os.path.join(event_dir, f"misfit_heatmap_Q{int(aslobj.Q)}_F{int(peakf_event)}{suffix}.png")
    print("[ASL:PLOT_MISFIT_HEATMAP]")
    aslobj.plot_misfit_heatmap(outfile=mh_png, topo_kw=topo_kw, show=show)
    plt.close('all')

    return {
        "qml": qml_out,
        "json": json_out,
        "map_png": map_png,
        "source_csv": csv_out,
        "reduced_disp_png": rd_png,
        "misfit_png": mis_png,
        "misfit_heatmap_png": mh_png,
    }


def asl_sausage(
    stream: Stream,
    event_dir: str,
    asl_config: dict,
    dry_run: bool = False,
    peakf_override: Optional[float] = None,
    station_gains_df: Optional["pd.DataFrame"] = None,
    allow_station_fallback: bool = True,
    *,
    # NEW: enable a triangular (apex→sea) sector refinement pass
    refine_sector: bool = False,
    vertical_only: bool = True,
    topo_kw: dict = None,
    show: bool=True,
    debug: bool=False,
):
    """
    Run ASL on a single (already preprocessed) event stream.

    Required keys in `asl_config`:
      - gridobj : Grid or NodeGrid
      - node_distances_km : dict[seed_id -> np.ndarray]
      - station_coords : dict[seed_id -> {latitude, longitude, elevation}]
      - ampcorr : AmpCorr
      - sam_metric : str
      - window_seconds : float
    """

    # Information about results from fast_locate() and refine_and_relocate()
    primary_out = None
    refined_out = None

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
    if vertical_only:
        stream = stream.select(component='Z')

    units_by_class  = {DSAM: "m",   VSAM: "m/s"}
    output_by_class = {DSAM: "DISP", VSAM: "VEL"}
    output = output_by_class.get(asl_config["sam_class"])

    for tr in stream:
        # Default units
        units = tr.stats.get("units") or "Counts"
        tr.stats["units"] = units

        # Only reason about response if still in counts
        if units == "Counts":
            median_abs = float(np.nanmedian(np.abs(tr.data))) if tr.data is not None else np.inf

            if median_abs < 1.0:
                # Looks already corrected but units not set; fix units based on SAM class
                new_units = units_by_class.get(asl_config["sam_class"])
                if new_units:
                    tr.stats["units"] = new_units
            else:
                # Still in counts at a plausible scale -> we should correct later
                print('Removing instrument response from {tr.id}')
                tr.remove_response(inventory=asl_config["inventory"], output=output)
    
    if debug:
        stream.plot(equal_scale=False, outfile=os.path.join(event_dir, 'stream.png'))


    # 2) Build SAM object
    samObj = asl_config['sam_class'](stream=stream, sampling_interval=1.0)
    if len(samObj.dataframes) == 0:
        raise IOError("[ASL:ERR] No dataframes in VSAM object")

    if not dry_run:
        print("[ASL:PLOT] Writing VSAM preview")
        samObj.plot(metrics=asl_config['sam_metric'], equal_scale=False, outfile=os.path.join(event_dir, "SAM.png"))
        plt.close('all')

    # 3) Decide event peakf
    if peakf_override is None:
        freqs = [df.attrs.get("peakf") for df in samObj.dataframes.values() if df.attrs.get("peakf") is not None]
        if freqs:
            peakf_event = int(round(sum(freqs) / len(freqs)))
            print(f"[ASL] Event peak frequency inferred from VSAM: {peakf_event} Hz")
        else:
            peakf_event = int(round(asl_config["ampcorr"].params.peakf))
            print(f"[ASL] Using default peak frequency from ampcorr: {peakf_event} Hz")
    else:
        peakf_event = int(round(peakf_override))
        print(f"[ASL] Using peak frequency override: {peakf_event} Hz")

    ################### IMPLEMENT THIS FOR CHECKING ################
    ampcorr_params = asl_config["ampcorr"].params

    if debug:
        print("ASL_CONFIG (keys):", sorted(asl_config.keys()))
        print("ASL_CONFIG AmpCorr object:")
        pprint(list(vars(asl_config["ampcorr"]).keys()))
        print(f"[ASL:CHECK] Params → surface_waves={ampcorr_params.assume_surface_waves}  "
            f"v={ampcorr_params.wave_speed_kms} km/s  Q={ampcorr_params.Q}  peakf_event={peakf_event}")

        # Sanity check distances (km)
        nd = asl_config["node_distances_km"]
        all_max = [float(np.nanmax(v)) for v in nd.values() if np.size(v)]
        if all_max:
            dmin = min(float(np.nanmin(v)) for v in nd.values())
            dmax = max(all_max)
            dp95 = np.percentile(np.concatenate([np.ravel(v) for v in nd.values()]), 95)
            print(f"[ASL:DISTS] node→station distances (km): min={dmin:.3f}  p95={dp95:.3f}  max={dmax:.3f}")
            if dmax > 100:  # Montserrat scale guardrail
                print("[ASL:WARN] Distances look like meters! (max > 100 km)")

        # compute reduced velocity at the summit
        if topo_kw['dome_location']:
            if asl_config['sam_class']==VSAM:
                print('[ASL_SAUSAGE]: Sanity check: VSAM data reduced to VR/VRS')
                VR = samObj.compute_reduced_velocity(
                    asl_config["inventory"],
                    topo_kw['dome_location'],
                    surfaceWaves=ampcorr_params.assume_surface_waves,
                    Q=ampcorr_params.Q,
                    wavespeed_kms=ampcorr_params.wave_speed_kms,
                    peakf=peakf_event,
                )
                print(VR)
                VR.plot(outfile=os.path.join(event_dir, "VR_at_dome.png"))

            elif asl_config['sam_class']==DSAM:
                print('[ASL_SAUSAGE]: Sanity check: DSAM data reduced to DR/DRS')
                DR = samObj.compute_reduced_displacement(
                    asl_config["inventory"],
                    topo_kw['dome_location'],
                    surfaceWaves=ampcorr_params.assume_surface_waves,
                    Q=ampcorr_params.Q,
                    wavespeed_kms=ampcorr_params.wave_speed_kms,
                    peakf=peakf_event,
                )
                print(DR)
                DR.plot(outfile=os.path.join(event_dir, "DR_at_dome.png"))
            
    # 4) Amplitude corrections cache (swap if peakf differs)
    ampcorr: AmpCorr = asl_config["ampcorr"]
    if abs(float(ampcorr.params.peakf) - float(peakf_event)) > 1e-6:
        print(f"[ASL] Switching amplitude corrections to peakf={peakf_event} Hz (from {ampcorr.params.peakf})")

        grid_sig = asl_config["gridobj"].signature()
        dist_sig = distances_signature(asl_config["node_distances_km"])
        inv_sig  = tuple(sorted(asl_config["node_distances_km"].keys()))

        params = ampcorr.params
        new_params = AmpCorrParams(
            assume_surface_waves=params.assume_surface_waves,
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
        asl_config['ampcorr'] = ampcorr

    else:
        print(f"[ASL] Using existing amplitude corrections (peakf={ampcorr.params.peakf} Hz)")

    # 5) Build ASL object and inject geometry/corrections
    print("[ASL] Building ASL object…")
    aslobj = ASL(
        samObj,
        config=asl_config,
    )

    idx = _grid_mask_indices(asl_config["gridobj"])
    if idx is not None and idx.size:
        aslobj._node_mask = idx

    # 6) Locate
    meta = asl_config.get("dist_meta", {})
    if debug:
        print(f"[DIST] used_3d={meta.get('used_3d')}  "
            f"has_node_elevations={meta.get('has_node_elevations')}  "
            f"n_nodes={meta.get('n_nodes')}  n_stations={meta.get('n_stations')}")
        z_grid = getattr(asl_config["gridobj"], "node_elev_m", None)
        if z_grid is not None:
            print(f"[GRID] Node elevations: min={np.nanmin(z_grid):.1f} m  max={np.nanmax(z_grid):.1f} m")
        ze = [c.get("elevation", 0.0) for c in aslobj.station_coordinates.values()]
        print(f"[DIST] Station elevations: min={min(ze):.1f} m  max={max(ze):.1f} m")

    print("[ASL] Locating source with fast_locate()…")
    # Pull optional settings from config if present
    min_sta = int(asl_config.get("min_stations", 3))
    misfit_backend = _resolve_misfit_backend(
        asl_config.get("misfit_engine", None),
        peakf_hz=float(ampcorr.params.peakf),
        speed_kms=float(ampcorr.params.wave_speed_kms),
    )
    aslobj.fast_locate(
        min_stations=min_sta,
        misfit_backend=misfit_backend,
    )
    print("[ASL] Location complete.")


    # 7) Outputs (baseline)
    if not dry_run:
        primary_out = _asl_output_source_results(
            aslobj,
            stream=stream,
            event_dir=event_dir,
            asl_config=asl_config,
            peakf_event=peakf_event,
            topo_kw=topo_kw,
            suffix="",
            show=show,
            debug=debug,
        )

    ############## SCAFFOLD
    src = aslobj.source
    for k in ("lat","lon","DR","misfit","t","nsta"):
        v = np.asarray(src.get(k, []))
        if v.size == 0:
            finite_pct = 0.0
        elif np.issubdtype(v.dtype, np.number):
            finite_pct = 100 * np.isfinite(v).mean()
        else:
            finite_pct = float("nan")  # not applicable
        print(f"[CHECK:SRC] {k}: shape={v.shape} dtype={v.dtype} finite%={finite_pct:.1f}") 

    ################### 

    # 8) OPTIONAL: sector refinement pass (triangular wedge from dome toward sea)
    if refine_sector and topo_kw['dome_location']:
        print("[ASL] Refinement pass: sector wedge from dome apex…")
        try:
            # Use dome apex as the sector origin
            apex_lat = float(topo_kw['dome_location']["lat"])
            apex_lon = float(topo_kw['dome_location']["lon"])

            # call refine_and_relocate() with sector mask + Viterbi smoothing defaults
            aslobj.refine_and_relocate(
                mask_method="sector",
                apex_lat=apex_lat, apex_lon=apex_lon,
                length_km=8.0, inner_km=0.0, half_angle_deg=25.0,
                prefer_misfit=True,
                temporal_smooth_mode="median", #"viterbi",
                temporal_smooth_win=7,
                #viterbi_lambda_km=8.0,
                #viterbi_max_step_km=30.0,
                misfit_backend=misfit_backend,            # ← new
                min_stations=min_sta,                     # ← new
                verbose=True,
            )
            print("[ASL] Sector refinement complete.")
        except Exception:
            print("[ASL:ERR] Sector refinement failed.")
            raise

        if not dry_run:
            refined_out = _asl_output_source_results(
                aslobj,
                stream=stream,
                event_dir=event_dir,
                asl_config=asl_config,
                peakf_event=peakf_event,
                topo_kw=topo_kw,
                suffix="_refined",
                show=False,
                debug=debug,                
            )

    # Return a structured summary for callers
    return {"primary": primary_out, "refined": refined_out}


def run_single_event(
    mseed_file: str,
    asl_config: dict = None,
    refine_sector: bool = False,
    station_gains_df: Optional[pd.DataFrame] = None,
    topo_kw: dict = None,
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Minimal, notebook-friendly runner (delegates prep to build_asl_config).

    NOTE: Expects asl_sausage() to return:
      {"primary": {...}, "refined": {... or None}}
    Each {...} must include (at least) "qml" and "json" paths.
    """

    t0 = time.time()
    Path(asl_config['outdir']).mkdir(parents=True, exist_ok=True)

    try:

        # Read stream and run once
        st = read(mseed_file).select(component="Z")
        if len(st) < asl_config['min_stations']:
            raise RuntimeError(f"Not enough stations: {len(st)} < {asl_config['min_stations']}")

        event_dir = Path(asl_config['outdir']) / Path(mseed_file).stem
        event_dir.mkdir(exist_ok=True)


        # show the grid
        if debug:
            print('You are using this Grid:')
            print(asl_config['gridobj'])
            gridpng = os.path.join(asl_config['outdir'], f"grid.png")
            if not os.path.isfile(gridpng):
                asl_config['gridobj'].plot(show=True, topo_map_kwargs=topo_kw, force_all_nodes=True, outfile=gridpng)

        print(f"[ASL] Running single event: {mseed_file}")
        outputs = asl_sausage(
            stream=st,
            event_dir=str(event_dir),
            asl_config=asl_config,
            dry_run=False,
            peakf_override=None,
            station_gains_df=station_gains_df,
            allow_station_fallback=True,
            refine_sector=refine_sector,
            vertical_only=True,
            topo_kw=topo_kw,   
            debug=debug,         
        )   

        # Validate returned structure; fail fast if missing
        if not isinstance(outputs, dict) or "primary" not in outputs:
            raise RuntimeError("asl_sausage() did not return the expected outputs dict.")

        primary = outputs.get("primary")
        if not isinstance(primary, dict) or not primary.get("qml") or not primary.get("json"):
            raise RuntimeError("asl_sausage() returned outputs without required 'qml'/'json' paths for the primary solution.")

        # Build summary strictly from returned outputs
        summ = {
            "tag": asl_config['tag'],
            "outdir": asl_config['outdir'],
            "event_dir": str(event_dir),
            "outputs": outputs,                  # {"primary": {...}, "refined": {... or None}}
            "elapsed_s": round(time.time() - t0, 2),
        }
        print(f"[ASL] Single-event summary: {summ}")
        return summ

    except Exception as e:
        traceback.print_exc()
        # Do not hide failures — return the error so callers can surface it
        return {
            "tag": asl_config['tag'],
            "error": f"{type(e).__name__}: {e}",
            "outdir": str(asl_config['outdir']),
        }
    

import os
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd  # only if you pass a DataFrame for gains


def run_all_events(
    input_dir: str,
    *,
    asl_config: dict,
    topo_kw: Optional[dict] = None,
    station_gains_df: Optional[pd.DataFrame] = None,
    refine_sector: bool = False,
    max_events: Optional[int] = None,
    use_multiprocessing: bool = True,
    workers: Optional[int] = None,
    debug: bool = True,
) -> str:
    """
    Process all miniSEED files under input_dir with the same configuration.
    - Parallel by default using N-2 workers (never less than 1).
    - Writes a JSONL summary of per-event results.
    - After the loop, generates heatmap and builds EnhancedCatalogs (if any events succeeded).
    Returns the run directory (asl_config['outdir']).
    """

    run_dir = Path(asl_config["outdir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(set(map(str, find_event_files(input_dir))))
    if max_events:
        files = files[: int(max_events)]
    if not files:
        print("[RUN] No event files found.")
        return str(run_dir)

    # worker count
    if use_multiprocessing:
        if workers is None:
            cpu = os.cpu_count() or 2
            workers = max(1, cpu - 2)
        print(f"[RUN] multiprocessing ON  workers={workers}")
    else:
        workers = 1
        print("[RUN] multiprocessing OFF")

    t0 = time.time()
    processed = 0
    all_outputs: List[Dict[str, Any]] = []
    summary_path = run_dir / "summary.jsonl"

    # Helper to serialize each result to summary file
    def _append_summary(rec: Dict[str, Any]):
        with open(summary_path, "a", encoding="utf-8") as sfo:
            sfo.write(json.dumps(rec) + "\n")

    # Serial path (easier for debugging)
    if workers == 1:
        for f in files:
            try:
                res = run_single_event(
                    mseed_file=f,
                    asl_config=asl_config,
                    refine_sector=refine_sector,
                    station_gains_df=station_gains_df,
                    topo_kw=topo_kw,
                    debug=debug,
                )
            except Exception as e:
                res = {
                    "tag": asl_config.get("tag", "<unknown>"),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "mseed_file": f,
                }
            _append_summary(res)
            all_outputs.append(res)
            processed += 1
            print("[OK]" if "error" not in res else "[ERR]", f)

    # Parallel path
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut2file = {
                ex.submit(
                    run_single_event,
                    mseed_file=f,
                    asl_config=asl_config,
                    refine_sector=refine_sector,
                    station_gains_df=station_gains_df,
                    topo_kw=topo_kw,
                    debug=debug,
                ): f
                for f in files
            }
            for fut in as_completed(fut2file):
                f = fut2file[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "tag": asl_config.get("tag", "<unknown>"),
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                        "mseed_file": f,
                    }
                _append_summary(res)
                all_outputs.append(res)
                processed += 1
                print("[OK]" if "error" not in res else "[ERR]", f)

    dt = time.time() - t0
    print(f"[DONE] Processed {processed}/{len(files)} events in {dt:.1f}s")

    # ---- Heatmap aggregation & plot
    try:
        df = collect_node_results(str(run_dir))
        if not df.empty:
            heat_png = os.path.join(run_dir, "heatmap_energy.png")
            print("[HEATMAP] Generating overall heatmap…")
            plot_heatmap_colored(
                df,
                lat_col="latitude",
                lon_col="longitude",
                amp_col="amplitude",
                log_scale=True,
                node_spacing_m=asl_config["gridobj"].node_spacing_m,
                outfile=heat_png,
                title=f"Energy Heatmap — {asl_config.get('tag','')}",
                topo_kw=topo_kw,
            )
            print("[HEATMAP] Wrote:", heat_png)
        else:
            print("[HEATMAP] No data found to plot.")
    except Exception as e:
        print("[HEATMAP:WARN] Failed to generate heatmap:", e)

    # ---- Optional: assemble EnhancedCatalogs from per-event outputs
    try:
        # Keep only successful runs that carry the expected structure
        good = []
        for rec in all_outputs:
            if isinstance(rec, dict) and "outputs" in rec and isinstance(rec["outputs"], dict):
                good.append(rec["outputs"])
        if good:
            cat_info = enhanced_catalogs_from_outputs(
                good,
                outdir=str(run_dir),
                write_files=True,
                load_waveforms=False,  # set True if you want streams in memory
                primary_name="catalog_primary",
                refined_name="catalog_refined",
            )
            if cat_info.get("primary_qml"):
                print("[CATALOG] Primary:", cat_info["primary_qml"])
            if cat_info.get("refined_qml"):
                print("[CATALOG] Refined:", cat_info["refined_qml"])
            if cat_info.get("primary_csv"):
                print("[CATALOG] Primary CSV:", cat_info["primary_csv"])
            if cat_info.get("refined_csv"):
                print("[CATALOG] Refined CSV:", cat_info["refined_csv"])
        else:
            print("[CATALOG] No successful event outputs to assemble.")
    except Exception as e:
        print(f"[CATALOG:WARN] Could not build EnhancedCatalogs: {e}")

    return str(run_dir)
    




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

def enhanced_catalogs_from_outputs(
    outputs_list: list[dict],
    *,
    outdir: str,
    write_files: bool = True,
    load_waveforms: bool = False,
    primary_name: str = "catalog_primary",
    refined_name: str = "catalog_refined",
) -> dict:
    """
    Build EnhancedCatalogs (primary, refined) from the per-event outputs dicts
    returned by `asl_sausage()`.

    Each element in outputs_list is expected to be:
      {"primary": {"qml": "...", "json": "...", ...},
       "refined": {"qml": "...", "json": "...", ...} or None,
       "_meta": {...}}

    Returns:
      {
        "primary": EnhancedCatalog,
        "refined": EnhancedCatalog,
        "primary_qml": <path or None>,
        "refined_qml": <path or None>,
        "primary_csv": <path or None>,
        "refined_csv": <path or None>,
      }
    """

    prim_recs = []
    ref_recs  = []

    def _append_rec(block: dict, bucket: list):
        if not block:
            return
        qml = block.get("qml")
        jjs = block.get("json")
        if not qml or not jjs:
            return
        base = os.path.splitext(qml)[0]  # strip .qml
        # Ensure the JSON next to it matches
        if not os.path.exists(qml) or not os.path.exists(jjs):
            return
        try:
            enh = EnhancedEvent.load(base)
            if not load_waveforms:
                # Clear in-memory waveform to keep object light
                enh.stream = None
            bucket.append(enh)
        except Exception:
            # Keep going; one bad event shouldn't kill the catalogs
            pass

    for out in outputs_list:
        _append_rec((out or {}).get("primary") or {}, prim_recs)
        _append_rec((out or {}).get("refined") or {}, ref_recs)

    # Build EnhancedCatalogs
    prim_cat = EnhancedCatalog(
        events=[r.event for r in prim_recs],
        records=prim_recs,
        description="Primary ASL locations"
    )
    ref_cat = EnhancedCatalog(
        events=[r.event for r in ref_recs],
        records=ref_recs,
        description="Refined ASL locations"
    )

    # Optionally write combined QuakeML + CSV summaries
    primary_qml = None
    refined_qml = None
    primary_csv = None
    refined_csv = None

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
    matches: List[str] = []
    for pat in RESULT_PATTERNS:
        matches.extend(glob(os.path.join(run_outdir, pat), recursive=True))

    dfs: List[pd.DataFrame] = []
    for f in sorted(set(matches)):
        try:
            df = pd.read_parquet(f) if f.lower().endswith(".parquet") else pd.read_csv(f)
        except Exception:
            continue

        # Normalize to lowercase column names for robust downstream usage
        lower_map = {orig: orig.lower() for orig in df.columns}
        df = df.rename(columns=lower_map)
        have = all(k in df.columns for k in ["latitude", "longitude", "amplitude"])
        if not have:
            continue

        dfs.append(df)

    if not dfs:
        print(f"[HEATMAP] No node CSVs matched under: {run_outdir}")
        return pd.DataFrame(columns=["latitude", "longitude", "amplitude"])

    cat = pd.concat(dfs, ignore_index=True)
    return cat[["latitude", "longitude", "amplitude"]].dropna()