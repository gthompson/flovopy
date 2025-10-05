# flovopy/asl/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime, Stream
from obspy.core.inventory import Inventory

from flovopy.asl.distances import distances_signature
from flovopy.asl.ampcorr import AmpCorrParams, AmpCorr
#from flovopy.processing.sam import VSAM  # default
from flovopy.asl.config import ASLConfig
from flovopy.asl.station_corrections import apply_interval_station_gains
from flovopy.asl.misfit import (
    StdOverMeanMisfit,
    R2DistanceMisfit,
    LinearizedDecayMisfit,
)
# from flovopy.asl.misfit import HuberMisfit  # if implemented
from flovopy.enhanced.event import EnhancedEvent, EnhancedEventMeta
from flovopy.enhanced.catalog import EnhancedCatalog
from flovopy.enhanced.stream import EnhancedStream
from flovopy.asl.map import plot_heatmap_colored
from flovopy.asl.asl import ASL
from flovopy.processing.spectrograms import icewebSpectrogram

# Needed for run_all_events()
import traceback
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

# Needed for monte carlo simulations
from dataclasses import replace
from typing import List, Dict, Any, Optional

# --- at module top (once) ---
import sys, io,  contextlib


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            try: st.write(s)
            except Exception: pass
        return len(s)
    def flush(self):
        for st in self._streams:
            try: st.flush()
            except Exception: pass

@contextlib.contextmanager
def tee_stdouterr(log_path: str, also_console: bool = True):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        ts = UTCDateTime().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n===== START {ts} pid={os.getpid()} =====\n")
        f.flush()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = _Tee(*( (old_out,) if also_console else () ), f)
            sys.stderr = _Tee(*( (old_err,) if also_console else () ), f)
            yield
        finally:
            try:
                sys.stdout.flush(); sys.stderr.flush()
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                ts2 = UTCDateTime().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n===== END {ts2} pid={os.getpid()} =====\n")
                f.flush()

def _asl_output_source_results(
    aslobj,
    *,
    stream,                 # ObsPy Stream or None (used for station labels on plots)
    event_dir: str | Path,
    cfg: ASLConfig,
    peakf_event: int,
    suffix: str = "",       # "" or "_refined"
    networkmag_df: Optional[pd.DataFrame] = None,
    topo_kw: dict | None = None,
    show: bool = True,
    debug: bool = False,
) -> dict:
    """
    Save QuakeML+JSON (EnhancedEvent), CSV, and diagnostic plots for the current ASL source.
    Returns a dict of produced file paths (some may be None if a step fails).
    """


    event_dir = Path(event_dir)
    event_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we always return all keys
    outs = {
        "qml": None,
        "json": None,
        "map_png": None,
        "source_csv": None,
        "reduced_disp_png": None,
        "misfit_png": None,
        "misfit_heatmap_png": None,
    }

    if debug:
        try:
            df_desc = aslobj.source_to_dataframe().describe()
            print("source sanity check:\n", df_desc)
        except Exception as e:
            print(f"[ASL:WARN] Could not describe source dataframe: {e}")

    # ---- Title / filenames
    cfg_tag = getattr(cfg, "tag_str", None) or cfg.tag()
    # Build a readable multi-line title in Python
    base_title = (
        f"ASL event {aslobj.starttime.strftime('%Y/%m/%d %H:%M:%S')}, {aslobj.duration:.0f} s\n"
        f"\n{cfg_tag}"
    )

    # Optionally append magnitudes on a third line
    if networkmag_df is not None and not networkmag_df.empty:
        ml_mean = networkmag_df.get("ML_mean", pd.Series(dtype=float)).iloc[-1]
        me_mean = networkmag_df.get("ME_mean", pd.Series(dtype=float)).iloc[-1]
        mag_parts = []
        if (ml_mean is not None) and np.isfinite(float(ml_mean)):
            mag_parts.append(f"ML={float(ml_mean):.2f}")
        if (me_mean is not None) and np.isfinite(float(me_mean)):
            mag_parts.append(f"ME={float(me_mean):.2f}")
        if mag_parts:
            base_title += "\n" + ", ".join(mag_parts)


    print(f"[ASL] Title:\n{base_title}")
    if suffix:
        base_title += " refined"
    # Include cfg tag to disambiguate products across parameter sweeps
    base_name = f"{cfg_tag}{suffix}"

    # ---- QuakeML + JSON sidecar (EnhancedEvent)
    try:
        ee_meta = EnhancedEventMeta()
        aslobj.event = EnhancedEvent.from_asl(
            aslobj,
            meta=ee_meta,
            stream=stream,                   # optional
            title_comment=base_title,
        )
        qml_out, json_out = aslobj.event.save(
            str(event_dir), base_name,
            write_quakeml=True,
            write_obspy_json=getattr(cfg, "write_event_obspy_json", False),
            include_trajectory_in_sidecar=getattr(cfg, "include_trajectory", True),
        )
        outs["qml"] = qml_out
        outs["json"] = json_out
        print(f"[ASL:OUT] Saved QuakeML: {qml_out}")
        print(f"[ASL:OUT] Saved JSON sidecar: {json_out}")
    except Exception as e:
        print(f"[ASL:WARN] Failed to write EnhancedEvent: {e}")


    def _safe_plot(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            plt.close('all')
            return True
        except Exception as e:
            print(f"[ASL:WARN] Plot failed in {getattr(fn, '__name__', str(fn))}: {e}", file=sys.stderr)
            traceback.print_exc()  # <-- full traceback to STDERR
            plt.close('all')
            return False


    # ---- Map (with station labels if stream provided)
    outs["map_png"] = str(event_dir / f"map_{base_name}.png")
    print("[ASL:PLOT] Writing map and diagnostic plots…")
    try:
        station_labels = [tr.stats.station for tr in (stream or [])]
        ok = _safe_plot(
            aslobj.plot,
            topo_kw=topo_kw,
            threshold_DR=0.0,
            scale=0.2,
            join=False,
            number=0,
            stations=station_labels,
            outfile=outs["map_png"],
            title=base_title,
            show=show,
        )
        if not ok:
            outs["map_png"] = None
    except Exception as e:
        print(f"[ASL:WARN] Map plot failed: {e}")
        outs["map_png"] = None

    # ---- CSV export of the source
    outs["source_csv"] = str(event_dir / f"source_{base_name}.csv")
    print("[ASL:SOURCE_TO_CSV] Writing source to CSV…")
    try:
        aslobj.source_to_csv(outs["source_csv"])
    except Exception as e:
        print(f"[ASL:WARN] Failed to write source CSV: {e}")
        outs["source_csv"] = None

    # ---- Reduced displacement/velocity plot
    outs["reduced_disp_png"] = str(event_dir / f"reduced_disp_{base_name}.png")
    print("[ASL:PLOT_REDUCED_DISPLACEMENT]")
    ok = _safe_plot(aslobj.plot_reduced_displacement, outfile=outs["reduced_disp_png"], show=show)
    if not ok:
        outs["reduced_disp_png"] = None

    # ---- Misfit line plot
    outs["misfit_png"] = str(event_dir / f"misfit_{base_name}.png")
    print("[ASL:PLOT_MISFIT]")
    ok = _safe_plot(aslobj.plot_misfit, outfile=outs["misfit_png"], show=show)
    if not ok:
        outs["misfit_png"] = None

    # ---- Misfit heatmap
    outs["misfit_heatmap_png"] = str(event_dir / f"misfit_heatmap_{base_name}.png")
    print("[ASL:PLOT_MISFIT_HEATMAP]")
    ok = _safe_plot(aslobj.plot_misfit_heatmap, outfile=outs["misfit_heatmap_png"], topo_kw=topo_kw, show=show)
    if not ok:
        outs["misfit_heatmap_png"] = None

    return outs


def asl_sausage(
    stream: Stream,
    event_dir: str,
    cfg: ASLConfig,                     # new object-based config
    dry_run: bool = False,
    peakf_override: Optional[float] = None,
    *,
    refine_sector: bool = False,
    networkmag_df: Optional[pd.DataFrame] = None,
    topo_kw: dict = None,
    show: bool = True,
    debug: bool = False,
):
    """
    Run ASL on a single (already preprocessed) event stream.

    Notes
    -----
    - `cfg` should already be built (cfg.build()). If not, this function will
      call `cfg.build()` once.
    - If `station_gains_df` is None, we will use `cfg.station_corr_df_built`
      (loaded/stashed by ASLConfig.build) when present.
    """

    # Ensure configuration is fully built (distances, ampcorr, etc.)
    if cfg.node_distances_km is None or cfg.ampcorr is None or getattr(cfg, "inventory", None) is None:
        if debug:
            print("[ASL] cfg not built yet; building now...")
        cfg = cfg.build()  # returns same instance (frozen dataclass fields set via __setattr__)

    primary_out = None
    refined_out = None

    os.makedirs(event_dir, exist_ok=True)
    print(f"[ASL] Preparing {cfg.sam_class.__name__} for event folder: {event_dir}")

    # 3) Build SAM object
    samObj = cfg.sam_class(stream=stream, sampling_interval=1.0)
    if len(samObj.dataframes) == 0:
        raise IOError("[ASL:ERR] No dataframes in SAM object")

    if not dry_run:
        print("[ASL:PLOT] Writing SAM preview…")
        samObj.plot(metrics=cfg.sam_metric, equal_scale=False, outfile=os.path.join(event_dir, "SAM.png"))
        plt.close("all")

    # 4) Decide event peakf
    if peakf_override is None:
        freqs = [df.attrs.get("peakf") for df in samObj.dataframes.values() if df.attrs.get("peakf") is not None]
        if freqs:
            peakf_event = int(round(sum(freqs) / len(freqs)))
            print(f"[ASL] Event peak frequency inferred from SAM: {peakf_event} Hz")
        else:
            peakf_event = int(round(cfg.ampcorr.params.peakf))
            print(f"[ASL] Using default peak frequency from ampcorr: {peakf_event} Hz")
    else:
        peakf_event = int(round(peakf_override))
        print(f"[ASL] Using peak frequency override: {peakf_event} Hz")

    # Debug info
    ampcorr_params = cfg.ampcorr.params
    if debug:
        print(f"[ASL:CHECK] Params → surface_waves={ampcorr_params.assume_surface_waves}  "
              f"v={ampcorr_params.wave_speed_kms} km/s  Q={ampcorr_params.Q}  peakf_event={peakf_event}")

        nd = cfg.node_distances_km or {}
        all_max = [float(np.nanmax(v)) for v in nd.values() if np.size(v)]
        if all_max:
            dmin = min(float(np.nanmin(v)) for v in nd.values())
            dmax = max(all_max)
            dp95 = np.percentile(np.concatenate([np.ravel(v) for v in nd.values()]), 95)
            print(f"[ASL:DISTS] node→station distances (km): min={dmin:.3f}  p95={dp95:.3f}  max={dmax:.3f}")
            if dmax > 100:
                print("[ASL:WARN] Distances look like meters! (max > 100 km)")

        # Summit sanity check (reduction)
        if topo_kw and topo_kw.get("dome_location"):
            if cfg.sam_class.__name__ == "VSAM":
                print("[ASL_SAUSAGE]: Sanity check: VSAM → VR/VRS at dome")
                VR = samObj.compute_reduced_velocity(
                    cfg.inventory,
                    topo_kw["dome_location"],
                    surfaceWaves=ampcorr_params.assume_surface_waves,
                    Q=ampcorr_params.Q,
                    wavespeed_kms=ampcorr_params.wave_speed_kms,
                    peakf=peakf_event,
                )
                print(VR)
                VR.plot(outfile=os.path.join(event_dir, "VR_at_dome.png"))
            elif cfg.sam_class.__name__ == "DSAM":
                print("[ASL_SAUSAGE]: Sanity check: DSAM → DR/DRS at dome")
                DR = samObj.compute_reduced_displacement(
                    cfg.inventory,
                    topo_kw["dome_location"],
                    surfaceWaves=ampcorr_params.assume_surface_waves,
                    Q=ampcorr_params.Q,
                    wavespeed_kms=ampcorr_params.wave_speed_kms,
                    peakf=peakf_event,
                )
                print(DR)
                DR.plot(outfile=os.path.join(event_dir, "DR_at_dome.png"))

    # 5) Amplitude corrections cache (swap if peakf differs)
    ampcorr: AmpCorr = cfg.ampcorr
    if abs(float(ampcorr.params.peakf) - float(peakf_event)) > 1e-6:
        print(f"[ASL] Switching amplitude corrections to peakf={peakf_event} Hz (from {ampcorr.params.peakf})")

        grid_sig = cfg.gridobj.signature()
        dist_sig = distances_signature(cfg.node_distances_km)
        inv_sig  = tuple(sorted(cfg.node_distances_km.keys()))

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

        new_ampcorr = AmpCorr(new_params, cache_dir=ampcorr.cache_dir)
        new_ampcorr.compute_or_load(cfg.node_distances_km, inventory=cfg.inventory)
        # cfg is frozen; update via object.__setattr__
        object.__setattr__(cfg, "ampcorr", new_ampcorr)
        ampcorr = new_ampcorr
    else:
        print(f"[ASL] Using existing amplitude corrections (peakf={ampcorr.params.peakf} Hz)")

    # 6) Build ASL object and inject geometry/corrections
    print("[ASL] Building ASL object…")
    aslobj = ASL(
        samObj,
        cfg,     # pass the ASLConfig object
    )

    idx = cfg.gridobj.get_mask_indices()
    if idx is not None and getattr(idx, "size", 0):
        aslobj._node_mask = idx

    # 7) Locate
    meta = cfg.dist_meta or {}
    if debug:
        print(f"[DIST] used_3d={meta.get('used_3d')}  "
              f"has_node_elevations={meta.get('has_node_elevations')}  "
              f"n_nodes={meta.get('n_nodes')}  n_stations={meta.get('n_stations')}")
        z_grid = getattr(cfg.gridobj, "node_elev_m", None)
        if z_grid is not None:
            print(f"[GRID] Node elevations: min={np.nanmin(z_grid):.1f} m  max={np.nanmax(z_grid):.1f} m")
        ze = [c.get("elevation", 0.0) for c in aslobj.station_coordinates.values()]
        print(f"[DIST] Station elevations: min={min(ze):.1f} m  max={max(ze):.1f} m")

    print("[ASL] Locating source with fast_locate()…")
    min_sta = int(cfg.min_stations)
    misfit_backend = _resolve_misfit_backend(
        cfg.misfit_engine,
        peakf_hz=float(cfg.ampcorr.params.peakf),
        speed_kms=float(cfg.ampcorr.params.wave_speed_kms),
    )
    aslobj.fast_locate(
        min_stations=min_sta,
        misfit_backend=misfit_backend,
    )
    print("[ASL] Location complete.")

    # 8) Outputs (baseline)
    if not dry_run:
        primary_out = _asl_output_source_results(
            aslobj,
            stream=stream,
            event_dir=event_dir,
            cfg=cfg,
            peakf_event=peakf_event,
            networkmag_df=networkmag_df,
            topo_kw=topo_kw,
            suffix="",
            show=show,
            debug=debug,
        )

    # Scaffold checks
    src = aslobj.source
    for k in ("lat", "lon", "DR", "misfit", "t", "nsta"):
        v = np.asarray(src.get(k, []))
        finite_pct = 0.0 if v.size == 0 else (100 * np.isfinite(v).mean() if np.issubdtype(v.dtype, np.number) else float("nan"))
        print(f"[CHECK:SRC] {k}: shape={v.shape} dtype={v.dtype} finite%={finite_pct:.1f}")

    # 9) Optional sector refinement
    if refine_sector and topo_kw and topo_kw.get("dome_location"):
        print("[ASL] Refinement pass: sector wedge from dome apex…")
        try:
            apex_lat = float(topo_kw["dome_location"]["lat"])
            apex_lon = float(topo_kw["dome_location"]["lon"])
            aslobj.refine_and_relocate(
                mask_method="sector",
                apex_lat=apex_lat, apex_lon=apex_lon,
                length_km=8.0, inner_km=0.0, half_angle_deg=25.0,
                prefer_misfit=True,
                temporal_smooth_mode="median",
                temporal_smooth_win=7,
                misfit_backend=misfit_backend,
                min_stations=min_sta,
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
                cfg=cfg,
                peakf_event=peakf_event,
                networkmag_df=networkmag_df,
                topo_kw=topo_kw,
                suffix="_refined",
                show=False,
                debug=debug,
            )

    return {"primary": primary_out, "refined": refined_out}

# ---------------------------------------------------------------------
# Single event
# ---------------------------------------------------------------------

def run_single_event(
    mseed_file: str,
    cfg: ASLConfig,
    *,
    refine_sector: bool = False,
    station_gains_df: Optional[pd.DataFrame] = None,
    topo_kw: Optional[dict] = None,
    switch_event_ctag: bool = True,
    vertical_only: bool = True,
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Minimal, notebook-friendly runner (delegates to asl_sausage).

    Returns:
      {
        "tag": <cfg tag string>,
        "outdir": <cfg outdir>,
        "event_dir": "<per-event dir>",
        "outputs": {"primary": {...}, "refined": {... or None}},
        "elapsed_s": <float>
      }
      or, on error:
      {
        "tag": <cfg tag string>,
        "error": "...",
        "outdir": <cfg outdir>
      }
    """

    t0 = UTCDateTime()

    try:
        if getattr(cfg, "inventory", None) is None or not getattr(cfg, "outdir", ""):
            if debug:
                print("[ASL] Building configuration (distances/ampcorr caches)…")
            cfg.build()
    except Exception as e:
        traceback.print_exc()
        tag_str = getattr(cfg, "tag_str", None) or cfg.tag()
        return {"tag": tag_str, "error": f"{type(e).__name__}: {e}", "outdir": str(getattr(cfg, "outdir", ""))}

    try:
        # could replace this with my more robust read_mseed() from flovopy.core.miniseed_io
        if vertical_only:
            st = read(mseed_file, format='MSEED').select(component="Z")
        else:
            st = read(mseed_file, format='MSEED')

        ##################################################################################
        ### DO ANY STREAM PREPROCESSING IN THIS BLOCK, OR BEFORE CALLING THIS FUNCTION ###
        ##################################################################################

        # 1) Station gains (optional)

        # Resolve station-gains / corrections
        gains_df = station_gains_df
        if gains_df is None:
            gains_df = getattr(cfg, "station_corr_df_built", None)

        if gains_df is not None and len(gains_df):
            info = apply_interval_station_gains(
                st,
                gains_df,
                allow_station_fallback=True,
                verbose=True,
            )
            s = info.get("interval_start"); e = info.get("interval_end")
            used = info.get("used", []); miss = info.get("missing", [])
            print(f"[GAINS] Interval used: {s} → {e} | corrected {len(used)} traces; missing {len(miss)}")
        else:
            print("[GAINS] No station gains DataFrame provided; skipping.")

        st.merge(fill_value='interpolate')
        st.detrend('linear')
        st.taper(max_percentage=0.02, type='cosine')
        st.filter('bandpass', freqmin=0.1, freqmax=18.0, corners=2, zerophase=True)

        # 2) Remove response if needed / set units based on SAM class
        try:
            from flovopy.processing.sam import DSAM, VSAM as _VSAM  # for class checks if available
        except Exception:
            DSAM = type("DSAM_PLACEHOLDER", (), {})  # sentinel if DSAM not importable
            _VSAM = cfg.sam_class

        units_by_class  = {DSAM: "m",    _VSAM: "m/s"}
        output_by_class = {DSAM: "DISP", _VSAM: "VEL"}
        output = output_by_class.get(cfg.sam_class)

        for tr in st:
            units = tr.stats.get("units") or "Counts"
            tr.stats["units"] = units
            if units == "Counts":
                median_abs = float(np.nanmedian(np.abs(tr.data))) if tr.data is not None else np.inf
                if median_abs < 1.0:
                    # probably already physical units
                    new_units = units_by_class.get(cfg.sam_class)
                    if new_units:
                        tr.stats["units"] = new_units
                else:
                    print(f"[RESP] Removing instrument response from {tr.id}")
                    try: # could replace this with my more robust remove_response() from flovopy.core.preprocessing
                        tr.remove_response(inventory=cfg.inventory, output=output)
                    except:
                        st.remove(tr)

        if len(st) < int(cfg.min_stations):
            raise RuntimeError(f"Not enough stations: {len(st)} < {cfg.min_stations}")


        ##################################################################################
        ### END OF STREAM PREPROCESSING BLOCK                                          ###
        ##################################################################################

        # Directory setup
        if switch_event_ctag:
            event_dir = Path(cfg.outdir).parent / Path(mseed_file).stem
            #cfg.outdir = event_dir / cfg.tag() # immutable
            products_dir = event_dir / Path(cfg.outdir).name
        else:
            products_dir = Path(cfg.outdir) / Path(mseed_file).stem
            event_dir = products_dir

        Path(products_dir).mkdir(parents=True, exist_ok=True)
        tag_str = getattr(cfg, "tag_str", None) or cfg.tag()

        # Per-event log file
        log_file = str(products_dir / "event.log")


        # Optionally plot grid once (unchanged)
        if debug and getattr(cfg, "gridobj", None) is not None and not switch_event_ctag:
            print("[ASL] Using this Grid:")
            print(cfg.gridobj)
            gridpng = os.path.join(cfg.outdir, "grid.png")
            if not os.path.isfile(gridpng):
                try:
                    cfg.gridobj.plot(show=True, topo_map_kwargs=topo_kw, force_all_nodes=True, outfile=gridpng)
                except Exception as e:
                    print(f"[ASL:WARN] Grid plot failed: {e}")

        if debug:
            stream_png = os.path.join(event_dir, "stream.png")
            if not os.path.isfile(stream_png):
                st.plot(equal_scale=False, outfile=stream_png)
            sgram_png = os.path.join(event_dir, "spectrogram.png")
            if not os.path.isfile(sgram_png):
                icewebSpectrogram(st).plot(fmin=0.1, fmax=10.0, log=True, cmap='plasma', dbscale=False, outfile=sgram_png)

        # --- Everything below gets teed to event.log ---
        with tee_stdouterr(log_file, also_console=debug):
            print(f"[ASL] Running single event: {mseed_file}")

            outputs = asl_sausage(
                stream=st,
                event_dir=str(products_dir),
                cfg=cfg,
                dry_run=False,
                peakf_override=None,
                refine_sector=refine_sector,
                topo_kw=topo_kw,
                debug=debug,
            )

            if not isinstance(outputs, dict) or "primary" not in outputs:
                raise RuntimeError("asl_sausage() did not return the expected outputs dict.")
            primary = outputs.get("primary")
            if not isinstance(primary, dict) or not primary.get("qml") or not primary.get("json"):
                raise RuntimeError("asl_sausage() returned outputs without required 'qml'/'json' paths for the primary solution.")

            # Inject the log-file path into the sidecar JSON (safe/no-op on error)
            try:
                sidecar = primary.get("json")
                if sidecar and os.path.exists(sidecar):
                    with open(sidecar, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    payload.setdefault("metrics", {})
                    payload["metrics"]["log_file"] = log_file
                    with open(sidecar, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, default=str)
            except Exception as e:
                print(f"[ASL:WARN] Could not tag sidecar with log path: {e}")

            summ = {
                "tag": tag_str,
                "outdir": cfg.outdir,
                "event_dir": str(event_dir),
                "log_file": log_file,
                "outputs": outputs,
                "elapsed_s": round(UTCDateTime() - t0, 2),
            }
            if debug:
                print(f"[ASL] Single-event summary: {summ}")
            return summ

    except Exception as e:
        traceback.print_exc()
        return {"tag": tag_str, "error": f"{type(e).__name__}: {e}", "outdir": cfg.outdir}
    
# ---------------------------------------------------------------------
# All events (with optional multiprocessing)
# ---------------------------------------------------------------------


def run_all_events(
    input_dir: str,
    *,
    cfg: ASLConfig,
    topo_kw: Optional[dict] = None,
    station_gains_df: Optional[pd.DataFrame] = None,
    refine_sector: bool = False,
    max_events: Optional[int] = None,
    use_multiprocessing: bool = True,
    workers: Optional[int] = None,
    debug: bool = True,
) -> str:
    """
    Process all miniSEED files under input_dir with the same ASLConfig.
    - Parallel by default using N-2 workers (>=1).
    - Writes JSONL summaries.
    - Generates a global heatmap and builds EnhancedCatalogs when possible.

    Returns cfg.outdir.
    """
    # Ensure cfg is fully built (outdir, inventory, caches)
    if getattr(cfg, "inventory", None) is None or not getattr(cfg, "outdir", ""):
        if debug:
            print("[RUN] Building configuration before batch run…")
        cfg.build()

    run_dir = Path(cfg.outdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(set(map(str, find_event_files(input_dir))))
    if max_events:
        files = files[: int(max_events)]
    if not files:
        print("[RUN] No event files found.")
        return str(run_dir)

    if use_multiprocessing:
        if workers is None:
            cpu = os.cpu_count() or 2
            workers = max(1, cpu - 2)
        print(f"[RUN] multiprocessing ON  workers={workers}")
    else:
        workers = 1
        print("[RUN] multiprocessing OFF")

    t0 = UTCDateTime()
    processed = 0
    all_outputs: List[Dict[str, Any]] = []
    summary_path = run_dir / "summary.jsonl"

    def _append_summary(rec: Dict[str, Any]):
        with open(summary_path, "a", encoding="utf-8") as sfo:
            sfo.write(json.dumps(rec) + "\n")

    # Serial path (helpful for debugging)
    if workers == 1:
        for f in files:
            try:
                res = run_single_event(
                    mseed_file=f,
                    cfg=cfg,
                    refine_sector=refine_sector,
                    station_gains_df=station_gains_df,
                    topo_kw=topo_kw,
                    debug=debug,
                )
            except Exception as e:
                res = {
                    "tag": getattr(cfg, "tag_str", None) or cfg.tag(),
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
        # NOTE: cfg must be picklable; ASLConfig dataclass is OK if its fields are picklable.
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut2file = {
                ex.submit(
                    run_single_event,
                    mseed_file=f,
                    cfg=cfg,
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
                        "tag": getattr(cfg, "tag_str", None) or cfg.tag(),
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                        "mseed_file": f,
                    }
                _append_summary(res)
                all_outputs.append(res)
                processed += 1
                print("[OK]" if "error" not in res else "[ERR]", f)

    dt = UTCDateTime() - t0  # seconds as float
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
                node_spacing_m=cfg.gridobj.node_spacing_m,
                outfile=heat_png,
                title=f"Energy Heatmap — {getattr(cfg, 'tag_str', None) or cfg.tag()}",
                topo_kw=topo_kw,
            )
            print("[HEATMAP] Wrote:", heat_png)
        else:
            print("[HEATMAP] No data found to plot.")
    except Exception as e:
        print("[HEATMAP:WARN] Failed to generate heatmap:", e)

    # ---- Optional: assemble EnhancedCatalogs from per-event outputs
    try:
        good = []
        for rec in all_outputs:
            if isinstance(rec, dict) and "outputs" in rec and isinstance(rec["outputs"], dict):
                good.append(rec["outputs"])
        if good:
            cat_info = enhanced_catalogs_from_outputs(
                good,
                outdir=str(run_dir),
                write_files=True,
                load_waveforms=False,
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




def _resolve_misfit_backend(name_or_obj, *, peakf_hz: float | None = None, speed_kms: float | None = None):
    """
    Accepts a backend instance or a string name and returns a backend instance.
    Known names:
      'l2','std','som','std_over_mean' -> StdOverMeanMisfit()
      'r2','r2distance'                -> R2DistanceMisfit()
      'lin','linearized','linearized_decay' -> LinearizedDecayMisfit(f_hz=..., v_kms=..., alpha=1.0)
      'huber' (if shipped)            -> HuberMisfit()
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
            f_hz=peakf_hz if peakf_hz is not None else 8.0,
            v_kms=speed_kms if speed_kms is not None else 1.5,
            alpha=1.0,
        )
    # if you ship Huber:
    # if key in ("huber",):
    #     return HuberMisfit()

    print(f"[ASL:MISFIT] Unknown backend '{name_or_obj}', using StdOverMean.")
    return StdOverMeanMisfit()


def find_event_files(root_dir: str, extensions=(".cleaned", ".mseed")) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extensions):
                files.append(os.path.join(dirpath, filename))
    return sorted(files)


RESULT_PATTERNS = (
    "**/*node_results.csv",
    "**/*node_results.parquet",
    "**/*node_metrics.csv",
    "**/*vsam_nodes.csv",
)

_COL_ALIASES = {
    "latitude":  ("latitude", "lat", "y"),
    "longitude": ("longitude", "lon", "x"),
    "amplitude": ("amplitude", "amp", "energy", "value"),
}

def _pick(df: pd.DataFrame, canonical: str) -> str | None:
    for cand in _COL_ALIASES[canonical]:
        if cand in df.columns:
            return cand
    return None

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

        # normalize to lowercase col names
        df.columns = [c.strip().lower() for c in df.columns]

        lat_col = _pick(df, "latitude")
        lon_col = _pick(df, "longitude")
        amp_col = _pick(df, "amplitude")
        if not (lat_col and lon_col and amp_col):
            continue

        sub = df[[lat_col, lon_col, amp_col]].rename(
            columns={lat_col: "latitude", lon_col: "longitude", amp_col: "amplitude"}
        )
        sub = sub.dropna(subset=["latitude", "longitude", "amplitude"])
        if not sub.empty:
            dfs.append(sub)

    if not dfs:
        print(f"[HEATMAP] No node CSVs matched under: {run_outdir}")
        return pd.DataFrame(columns=["latitude", "longitude", "amplitude"])

    return pd.concat(dfs, ignore_index=True)


def enhanced_catalogs_from_outputs(
    outputs_list: List[Dict[str, Any]],
    *,
    outdir: str,
    write_files: bool = True,
    load_waveforms: bool = False,
    primary_name: str = "catalog_primary",
    refined_name: str = "catalog_refined",
) -> Dict[str, Any]:

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


#########################################



# --- helper -------------------------------------------------------------


# wrappers2.py
import os
from pathlib import Path
from contextlib import contextmanager
import sys

@contextmanager
def tee_stdouterr(filepath: str, also_console: bool = True):
    """
    Tee both stdout and stderr to `filepath`.
    Works in notebooks and scripts.
    """
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    f = open(filepath, "a", buffering=1)
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, x):
            for s in self.streams: 
                try: s.write(x)
                except Exception: pass
        def flush(self):
            for s in self.streams:
                try: s.flush()
                except Exception: pass
    try:
        sys.stdout  = Tee(old_out, f) if also_console else Tee(f)
        sys.stderr  = Tee(old_err, f) if also_console else Tee(f)
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.flush(); f.close()

# --- public API ---------------------------------------------------------

def run_event_monte_carlo(
    mseed_file: str,
    configs: List[ASLConfig],
    *,
    inventory: Inventory | str | Path | None,
    output_base: str | Path,
    gridobj: Any,
    topo_kw: Optional[dict] = None,
    station_gains_df=None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    global_cache: str | Path | None = None,
    debug: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a **single event** across many ASL parameter configurations.

    Each item in `configs` is an `ASLConfig` carrying the *physical knobs*
    (wave kind, speed, Q, misfit, etc.). This function injects the shared
    run context (`inventory`, `output_base`, `gridobj`, optional `global_cache`)
    into each config (without mutating originals), then delegates to
    `run_single_event` for execution.

    Notes on performance / parallelism
    ----------------------------------
    - If `parallel=True`, processes run in separate workers via
      `ProcessPoolExecutor`.
    - Passing a large ObsPy `Inventory` object between processes may be
      expensive. For best performance in parallel mode, prefer passing a
      **path** to StationXML (string/Path) as `inventory`; each worker will
      read it locally.
    - `run_single_event` calls `cfg.ensure_built()`, so configs do **not**
      need to be built ahead of time.

    Parameters
    ----------
    mseed_file : str
        Path to the event’s miniSEED (already preprocessed).
    configs : list of ASLConfig
        Parameter draws to evaluate.
    inventory : Inventory | str | Path | None
        Shared station inventory (or path). If None, each `cfg` must already
        have a usable `inventory`.
    output_base : str | Path
        Base output directory; each config writes to a unique tag-scoped subdir.
    gridobj : Any
        Shared node grid (must implement `.signature()` at minimum).
    topo_kw : dict, optional
        Extra map/plotting kwargs passed down to plotting helpers.
    station_gains_df : pandas.DataFrame, optional
        Optional per-station gain corrections for this event.
    parallel : bool, default True
        Run configurations in parallel.
    max_workers : int, optional
        Worker count; defaults to CPU-2 (min 1).
    global_cache : str | Path, optional
        Shared cache root for distances/ampcorr; if None, configs’ own values apply.
    debug : bool, default True
        Verbose logging.

    Returns
    -------
    list of dict
        One result per configuration. Each element is the dict returned by
        `run_single_event`, or an error record of the form:
        `{"tag": "<cfg tag or unknown>", "error": "Type: message"}`
    """

    def _with_context(
        cfg: ASLConfig,
        *,
        inventory: Inventory | str | Path | None,
        output_base: str | Path | None,
        gridobj: Any | None,
        global_cache: str | Path | None = None,
        debug: Optional[bool] = None,
    ) -> ASLConfig:
        """
        Return a new ASLConfig with context fields set (without mutating the original).
        """
        updates = {}
        if inventory is not None:
            updates["inventory"] = inventory
        if output_base is not None:
            updates["output_base"] = output_base
        if gridobj is not None:
            updates["gridobj"] = gridobj
        if global_cache is not None:
            updates["global_cache"] = global_cache
        if debug is not None:
            updates["debug"] = debug
        return replace(cfg, **updates)


    def _run_one(cfg0):
        try:
            # Materialize a concrete cfg with the shared context (your existing helper)
            cfg = _with_context(
                cfg0,
                inventory=inventory,
                output_base=output_base,
                gridobj=gridobj,
                global_cache=global_cache,
                debug=debug,
            )

            # Ensure cfg is built so tag/outdir exist
            if getattr(cfg, "outdir", "") == "" or getattr(cfg, "ampcorr", None) is None:
                if debug:
                    print("[MC] Building cfg (distances/ampcorr)…")
                cfg.build()

            # Event stem for foldering/log names
            event_stem = Path(mseed_file).stem
            tag = getattr(cfg, "tag_str", None) or cfg.tag()

            # Per-trial log path: <outdir>/<event_stem>/monte_carlo/<tag>.log
            trial_dir = Path(cfg.outdir) / event_stem / "monte_carlo"
            trial_dir.mkdir(parents=True, exist_ok=True)
            log_path = trial_dir / f"{tag}.log"

            with tee_stdouterr(str(log_path), also_console=debug):
                print(f"[MC] Starting trial tag={tag} event={mseed_file}")

                result = run_single_event(
                    mseed_file=mseed_file,
                    cfg=cfg,
                    refine_sector=False,
                    station_gains_df=station_gains_df,
                    topo_kw=topo_kw,
                    debug=debug,
                )

                status = "OK" if "error" not in result else f"ERROR: {result['error']}"
                print(f"[MC] Finished trial tag={tag} status={status}")
                return result

        except Exception as e:
            # Make sure the exception surfaces both in console and logs
            try:
                # Best-effort to still write into a per-trial log
                event_stem = Path(mseed_file).stem
                tag = getattr(cfg0, "tag_str", None) or getattr(cfg0, "tag", lambda: "<unknown>")()
                trial_dir = Path(getattr(cfg0, "outdir", output_base)) / event_stem / "monte_carlo"
                trial_dir.mkdir(parents=True, exist_ok=True)
                log_path = trial_dir / f"{tag}.log"
                with open(log_path, "a") as fh:
                    fh.write(f"[MC:ERROR] Trial tag={tag} raised {type(e).__name__}: {e}\n")
                    fh.write(traceback.format_exc())
            except Exception:
                pass  # don’t mask original error if logging setup fails

            print(f"[MC:ERROR] Trial raised {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            return {"tag": getattr(cfg0, "tag_str", None) or getattr(cfg0, "tag", lambda: "<unknown>")(),
                    "error": f"{type(e).__name__}: {e}"}

    if not parallel:
        return [_run_one(cfg) for cfg in configs]

    # Parallel path (per-process logs, one file per config tag)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one, cfg): cfg for cfg in configs}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"tag": "<unknown>", "error": f"{type(e).__name__}: {e}"})
    return results

'''
def monte_carlo_wrapper_example() -> None:
    """
    Example: build a small grid of configs and run a single event across them.

    Illustrates:
    - Creating parameter draws with `ASLConfig.generate_config_list`
      (tweak the sequences to taste).
    - Supplying shared run context (inventory path, output_base, gridobj).
    - Invoking `run_event_monte_carlo` in parallel.
    """

    # Simple 6-draw sweep (replace with your own priors/sequences)
    configs = ASLConfig.generate_config_list(
        wave_kinds=("surface",),
        station_corr_opts=(False, True),
        speeds=(1.4, 2.0),
        Qs=(100,),
        dist_modes=("3d",),
        misfit_engines=("l2",),
        peakfs=(6.0, 8.0),
        # context can be set later; set here if you like:
        inventory_xml=None,
        output_base=None,
        gridobj=None,
        global_cache=None,
        debug=False,
    )

    # Shared run context
    mseed_file   = "/path/to/event.mseed"
    inventory_xml= "/path/to/stations.xml"     # NOTE: path recommended for parallel performance
    output_base  = "/tmp/asl_runs"
    final_grid   = build_or_load_your_nodegrid_somehow()  # user-defined

    # Optional map bits
    topo_kw = dict(
        dome_location={"lat": 16.72, "lon": -62.18},
        # ... any other keys your plotting utils expect
    )

    results = run_event_monte_carlo(
        mseed_file=mseed_file,
        configs=configs,
        inventory=inventory_xml,
        output_base=output_base,
        gridobj=final_grid,
        topo_kw=topo_kw,
        station_gains_df=None,
        parallel=True,
        max_workers=None,
        global_cache=None,
        debug=True,
    )

    # Inspect or summarize `results` as needed
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"[MC] Completed {n_ok}/{len(results)} runs OK")
'''

from pprint import pprint
def run_single_event(
    mseed_file: str,
    cfg: ASLConfig,
    *,
    refine_sector: bool = False,
    station_gains_df: Optional[pd.DataFrame] = None,
    topo_kw: Optional[dict] = None,
    switch_event_ctag: bool = True,
    vertical_only: bool = True,
    enhance: bool = True,           # <— NEW
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Minimal, notebook-friendly runner (delegates to asl_sausage).

    When `enhance=True`:
      - compute trace-level metrics (ampengfft) once up-front on an EnhancedStream,
      - compute provisional magnitudes using topo_kw['dome_location'] if provided,
      - after ASL location, recompute magnitudes only with the new source,
      - write pre/post metrics CSVs and tag their paths into the sidecar JSON.

    Returns
    -------
    dict
      On success:
        {
          "tag": <cfg tag string>,
          "outdir": <cfg outdir>,
          "event_dir": "<per-event dir>",
          "log_file": "<event.log>",
          "outputs": {"primary": {...}, "refined": {... or None}},
          "elapsed_s": <float>
        }
      On error:
        {
          "tag": <cfg tag string>,
          "error": "...",
          "outdir": <cfg outdir>
        }
    """


    # ---- nested helpers -------------------------------------------------
    def _write_metrics_and_mags_csv(es, basepath: str) -> tuple[str, str]:
        """
        Write per-trace metrics and station summary CSVs for an EnhancedStream `es`,
        *without* writing waveforms. Returns (trace_csv, station_csv).
        """
        import pandas as pd
        if basepath.endswith(".mseed"):
            basepath = basepath[:-6]
        outdir = os.path.dirname(basepath)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        # --- per-trace metrics (flatten spectrum + metrics + coords) ---
        trace_rows = []
        for tr in es:
            s = tr.stats
            row = {
                "id": tr.id,
                "starttime": s.starttime,
                "Fs": s.sampling_rate,
                "calib": getattr(s, "calib", None),
                "units": getattr(s, "units", None),
                "quality": getattr(s, "quality_factor", None),
            }
            if hasattr(s, "spectrum"):
                for item in ["medianF", "peakF", "peakA", "bw_min", "bw_max"]:
                    row[item] = s.spectrum.get(item, None)
            if hasattr(s, "metrics"):
                for k, v in s.metrics.items():
                    if isinstance(v, dict):
                        for subk, subv in v.items():
                            row[f"{k}_{subk}"] = subv
                    else:
                        row[k] = v
            if hasattr(s, "coordinates"):
                row["latitude"]  = s.coordinates.latitude
                row["longitude"] = s.coordinates.longitude
                row["elevation"] = s.coordinates.elevation
            trace_rows.append(row)

        df_traces = pd.DataFrame(trace_rows)
        csv_tr = basepath + ".csv"
        df_traces.to_csv(csv_tr, index=False)

        # --- station-level summary ---
        try:
            sdf = getattr(es, "station_metrics", None)
            if sdf is None or sdf.empty:
                sdf = es._station_level_metrics()
            csv_sta = basepath + "_station.csv"
            if not sdf.empty:
                sdf.to_csv(csv_sta, index=False)
            else:
                # still create an empty file with headers for consistency
                sdf.to_csv(csv_sta, index=False)
        except Exception as e:
            print(f"[WARN] station-level CSV not written: {e}")
            csv_sta = basepath + "_station.csv"

        return csv_tr, csv_sta

    def _best_source_from_sidecar(path: str) -> dict | None:
        """Extract a simple {'latitude','longitude','depth?'} source from sidecar JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            org = (payload.get("preferred_origin")
                   or (payload.get("origins") or [{}])[0])
            lat = org.get("latitude"); lon = org.get("longitude")
            if lat is None or lon is None:
                return None
            out = {"latitude": float(lat), "longitude": float(lon)}
            if org.get("depth") is not None:
                out["depth"] = float(org["depth"])
            return out
        except Exception:
            return None

    def upsert_network_magnitudes_csv(
        csv_path,
        *,
        stage: str,
        ml_mean, ml_std, n_ml,
        me_mean, me_std, n_me,
        metadata: dict | None = None,
    ):
        import numpy as np
        import pandas as pd
        import os

        def _num(x):
            try:
                return np.nan if x is None else float(x)
            except Exception:
                return np.nan

        row = {
            "stage": stage,
            "ML_mean": _num(ml_mean), "ML_std": _num(ml_std), "ML_n": int(n_ml or 0),
            "ME_mean": _num(me_mean), "ME_std": _num(me_std), "ME_n": int(n_me or 0),
        }
        if metadata:
            row.update(metadata)

        df_new = pd.DataFrame([row])

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "stage" in df:
                df = df[df["stage"] != stage]
            # Ensure any new columns exist
            for c in df_new.columns:
                if c not in df.columns:
                    df[c] = pd.NA
            df = pd.concat([df, df_new[df.columns]], ignore_index=True)
        else:
            df = df_new

        # Optional: enforce numeric dtypes on these columns
        for c in ["ML_mean","ML_std","ML_n","ME_mean","ME_std","ME_n"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df.to_csv(csv_path, index=False)
        return df
    # --------------------------------------------------------------------
    t0 = UTCDateTime()
    print('\n' + '='*80)
    print(f"[ASL] Running event: {mseed_file} with config tag={getattr(cfg, 'tag_str', None) or cfg.tag()}")
    print(f"[ASL] Config outdir: {cfg.outdir}")
    # ----------------------------------------------------------------
    # Directory setup
    # ----------------------------------------------------------------
    if switch_event_ctag:
        event_dir = Path(cfg.outdir).parent / Path(mseed_file).stem
        products_dir = event_dir / Path(cfg.outdir).name
    else:
        products_dir = Path(cfg.outdir) / Path(mseed_file).stem
        event_dir = products_dir

    
    tag_str = getattr(cfg, "tag_str", None) or cfg.tag()

    # Per-event log file
    log_file = str(products_dir / "event.log")

    netmagcsv = event_dir / "network_magnitudes.csv"
    networkmag_df = pd.DataFrame()

    es_pickle_file = products_dir / "enhanced_stream.pkl"

    # Ensure cfg is built (inventory/outdir etc.)
    try:
        if getattr(cfg, "inventory", None) is None or not getattr(cfg, "outdir", ""):
            if debug:
                print("[ASL] Building configuration (distances/ampcorr caches)…")
            cfg.build()
    except Exception as e:
        traceback.print_exc()
        tag_str = getattr(cfg, "tag_str", None) or cfg.tag()
        return {"tag": tag_str, "error": f"{type(e).__name__}: {e}", "outdir": str(getattr(cfg, "outdir", ""))}


    try:
        # Read waveforms
        if vertical_only:
            st = read(mseed_file, format="MSEED").select(component="Z")
        else:
            st = read(mseed_file, format="MSEED")

        # ----------------------------------------------------------------
        # STREAM PREPROCESSING
        # ----------------------------------------------------------------

        # 1) Station gains (optional)
        from flovopy.asl.station_corrections import apply_interval_station_gains

        gains_df = station_gains_df or getattr(cfg, "station_corr_df_built", None)
        if gains_df is not None and len(gains_df):
            info = apply_interval_station_gains(
                st, gains_df, allow_station_fallback=True, verbose=True,
            )
            s = info.get("interval_start"); e = info.get("interval_end")
            used = info.get("used", []); miss = info.get("missing", [])
            print(f"[GAINS] Interval used: {s} → {e} | corrected {len(used)} traces; missing {len(miss)}")
        else:
            print("[GAINS] No station gains DataFrame provided; skipping.")

        st.merge(fill_value="interpolate")
        st.detrend("linear")
        st.taper(max_percentage=0.02, type="cosine")
        st.filter("bandpass", freqmin=0.1, freqmax=18.0, corners=2, zerophase=True)

        # 2) Remove response if needed / set units based on SAM class
        try:
            from flovopy.processing.sam import DSAM, VSAM as _VSAM  # for class checks if available
        except Exception:
            DSAM = type("DSAM_PLACEHOLDER", (), {})
            _VSAM = cfg.sam_class

        units_by_class  = {DSAM: "m",    _VSAM: "m/s"}
        output_by_class = {DSAM: "DISP", _VSAM: "VEL"}
        output = output_by_class.get(cfg.sam_class)

        for tr in list(st):
            units = tr.stats.get("units") or "Counts"
            tr.stats["units"] = units
            if units == "Counts":
                median_abs = float(np.nanmedian(np.abs(tr.data))) if tr.data is not None else np.inf
                if median_abs < 1.0:
                    # probably already physical units
                    new_units = units_by_class.get(cfg.sam_class)
                    if new_units:
                        tr.stats["units"] = new_units
                else:
                    print(f"[RESP] Removing instrument response from {tr.id}")
                    try:
                        tr.remove_response(inventory=cfg.inventory, output=output)
                    except Exception:
                        # drop trace if response removal fails
                        try:
                            st.remove(tr)
                        except Exception:
                            pass

        if len(st) < int(cfg.min_stations):
            raise RuntimeError(f"Not enough stations: {len(st)} < {cfg.min_stations}")

        # If we got this far, it is time to create output directories for products
        Path(products_dir).mkdir(parents=True, exist_ok=True)

        # Debug plots (outside tee to avoid capturing large binary dumps)
        if debug:
            stream_png = os.path.join(event_dir, "stream.png")
            if not os.path.isfile(stream_png):
                try:
                    st.plot(equal_scale=False, outfile=stream_png)
                except Exception as e:
                    print(f"[ASL:WARN] stream plot failed: {e}")
            sgram_png = os.path.join(event_dir, "spectrogram.png")
            if not os.path.isfile(sgram_png):
                try:
                    icewebSpectrogram(st).plot(fmin=0.1, fmax=10.0, log=True, cmap="plasma",
                                               dbscale=False, outfile=sgram_png)
                except Exception as e:
                    print(f"[ASL:WARN] spectrogram plot failed: {e}")

        # --- Everything below gets teed to event.log ---


        with tee_stdouterr(log_file, also_console=debug):
            print(f"[ASL] Running single event: {mseed_file}")

            # ----------------------------------------------------------------
            # ENHANCED METRICS + PROVISIONAL MAGNITUDES (optional)
            # ----------------------------------------------------------------
            pre_tr_csv = pre_sta_csv = None
            post_tr_csv = post_sta_csv = None
            es = None


            if enhance:
                try:
                    from flovopy.enhanced.stream import EnhancedStream
                    es = EnhancedStream(stream=st.copy())

                    # compute per-trace metrics ONCE
                    es.ampengfft(
                        differentiate=True,            # tweak to your default
                        compute_spectral=True,
                        compute_ssam=False,
                        compute_bandratios=False,
                        compute_sam=True,
                        compute_sem=False,
                    )

                    # Provisional magnitudes using dome_location if present
                    dome = (topo_kw or {}).get("dome_location")
                    if dome and isinstance(dome, dict) and all(k in dome for k in ("lat", "lon")):
                        src_pre = {"latitude": float(dome["lat"]), "longitude": float(dome["lon"])}
                        if "depth" in dome and dome["depth"] is not None:
                            src_pre["depth"] = float(dome["depth"])

                        c_earth_ms = float(getattr(cfg, "speed", 2.5)) * 1000.0    # km/s → m/s
                        Q = float(getattr(cfg, "Q", 50.0))

                        print("[MAG] Provisional magnitudes (dome_location)…")
                        es.compute_station_magnitudes(
                            inventory=cfg.inventory,
                            source_coords=src_pre,
                            model="body", 
                            Q=cfg.Q, 
                            c_earth=c_earth_ms, 
                            use_boatwright=True,
                            attach_coords=True, 
                            compute_distances=True,
                        )
                        

                        print(es)
                        for tr in es:
                            print()
                            print(tr.id,":")
                            print("  coords:", tr.stats.get("coordinates", None))
                            print("  distance_m:", tr.stats.get("distance", None))
                            print('  metrics:\n',tr.stats.metrics._pretty_str())
                            print('  spectrum:\n',tr.stats.spectrum._pretty_str())

                        # save enhancedStream as a pickle file
                        # SCAFFOLD - I have to figure out how to read a pickle file into an EnhancedStream first
                        #es.write(es_pickle_file, format="PICKLE")
                        

                        ml_mean, ml_std, n_ml = es.estimate_network_magnitude("local_magnitude")
                        me_mean, me_std, n_me = es.estimate_network_magnitude("energy_magnitude")
                        networkmag_df = upsert_network_magnitudes_csv(
                            netmagcsv,
                            stage="provisional",
                            ml_mean=ml_mean, ml_std=ml_std, n_ml=n_ml,
                            me_mean=me_mean, me_std=me_std, n_me=n_me,
                            metadata={
                                "tag": getattr(cfg, "tag_str", None) or cfg.tag(),
                                "Q": float(getattr(cfg.ampcorr.params, "Q", float("nan"))),
                                "c_earth_kms": float(getattr(cfg.ampcorr.params, "wave_speed_kms", float("nan"))),
                                "used_source": "dome_location",
                            },
                        )
                        print(networkmag_df)

                    # Persist pre-metrics (with/without provisional mags)
                    pre_base = str(products_dir / "pre_metrics")
                    pre_tr_csv, pre_sta_csv = _write_metrics_and_mags_csv(es, pre_base)
                except Exception as e:
                    print(f"[ENHANCE:WARN] Pre-metrics/magnitude step failed: {e}")
                    es = st  # continue safely
            else:
                es = st
            
    

            # ----------------------------------------------------------------
            # Run ASL sausage
            # ----------------------------------------------------------------
            outputs = asl_sausage(
                stream=es,
                event_dir=str(products_dir),
                cfg=cfg,
                dry_run=False,
                peakf_override=None,
                refine_sector=refine_sector,
                topo_kw=topo_kw,
                networkmag_df=networkmag_df,
                debug=debug,
            )

            if not isinstance(outputs, dict) or "primary" not in outputs:
                raise RuntimeError("asl_sausage() did not return the expected outputs dict.")
            primary = outputs.get("primary")
            if not isinstance(primary, dict) or not primary.get("qml") or not primary.get("json"):
                raise RuntimeError("asl_sausage() returned outputs without required 'qml'/'json' paths for the primary solution.")

            # ----------------------------------------------------------------
            # ENHANCED: Update magnitudes with ASL source and persist post CSVs
            # ----------------------------------------------------------------
            if enhance and es is not None:
                try:
                    sidecar = primary.get("json")
                    src_post = _best_source_from_sidecar(sidecar) if sidecar else None
                    if src_post:
                        print("[MAG] Updated magnitudes (ASL source)…")
                        c_earth_ms = float(getattr(cfg, "speed", 2.5)) * 1000.0
                        Q = float(getattr(cfg, "Q", 50.0))
                        es.compute_station_magnitudes(
                            inventory=cfg.inventory,
                            source_coords=src_post,
                            model="body", Q=Q, c_earth=c_earth_ms, correction=3.7,
                            a=1.6, b=-0.15, g=0.0,
                            use_boatwright=True,
                            rho_earth=2000.0, S=1.0, A=1.0,
                            rho_atmos=1.2, c_atmos=340.0, z=100000.0,
                            attach_coords=True, compute_distances=True,
                        )
                        ml_mean, ml_std, n_ml = es.estimate_network_magnitude("local_magnitude")
                        me_mean, me_std, n_me = es.estimate_network_magnitude("energy_magnitude")
                        networkmag_df = upsert_network_magnitudes_csv(
                            netmagcsv,
                            stage="revised",
                            ml_mean=ml_mean, ml_std=ml_std, n_ml=n_ml,
                            me_mean=me_mean, me_std=me_std, n_me=n_me,
                            metadata={
                                "used_source": "ASL_location",
                                "best_lat": best_lat, "best_lon": best_lon,  # if you have them
                            },
                        )
                        print(networkmag_df)

                    post_base = str(products_dir / "post_metrics")
                    post_tr_csv, post_sta_csv = _write_metrics_and_mags_csv(es, post_base)

                    # Tag sidecar with log + CSV pointers
                    try:
                        if sidecar and os.path.exists(sidecar):
                            with open(sidecar, "r", encoding="utf-8") as f:
                                payload = json.load(f)
                            payload.setdefault("metrics", {})
                            payload["metrics"]["log_file"] = log_file
                            if pre_tr_csv:
                                payload["metrics"]["pre_trace_csv"] = pre_tr_csv
                            if pre_sta_csv:
                                payload["metrics"]["pre_station_csv"] = pre_sta_csv
                            if post_tr_csv:
                                payload["metrics"]["post_trace_csv"] = post_tr_csv
                            if post_sta_csv:
                                payload["metrics"]["post_station_csv"] = post_sta_csv
                            with open(sidecar, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2, default=str)
                    except Exception as e:
                        print(f"[ASL:WARN] Could not tag sidecar with magnitude CSVs: {e}")

                except Exception as e:
                    print(f"[ENHANCE:WARN] Post-magnitude update failed: {e}")

            # Always tag sidecar with log_file, even if enhance=False
            try:
                sidecar = primary.get("json")
                if sidecar and os.path.exists(sidecar):
                    with open(sidecar, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    payload.setdefault("metrics", {})
                    payload["metrics"]["log_file"] = log_file
                    with open(sidecar, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, default=str)
            except Exception as e:
                print(f"[ASL:WARN] Could not tag sidecar with log path: {e}")

            summ = {
                "tag": tag_str,
                "outdir": cfg.outdir,
                "event_dir": str(event_dir),
                "log_file": log_file,
                "outputs": outputs,
                "elapsed_s": round(UTCDateTime() - t0, 2),
            }
            if debug:
                print(f"[ASL] Single-event summary: {summ}")
            return summ

    except Exception as e:
        traceback.print_exc()
        tag_str = getattr(cfg, "tag_str", None) or cfg.tag()
        return {"tag": tag_str, "error": f"{type(e).__name__}: {e}", "outdir": cfg.outdir}