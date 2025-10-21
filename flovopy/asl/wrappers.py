from __future__ import annotations

# stdlib
import contextlib
import io
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

# third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime, Stream

# project
from flovopy.asl.ampcorr import AmpCorrParams, AmpCorr
from flovopy.asl.asl import ASL
from flovopy.asl.distances import distances_signature
from flovopy.asl.map import plot_heatmap_colored
from flovopy.asl.misfit import (
    StdOverMeanMisfit,
    R2DistanceMisfit,
    LinearizedDecayMisfit,
    # HuberMisfit,  # if/when implemented
)
from flovopy.asl.reduced_time import shift_stream_by_travel_time
from flovopy.asl.station_corrections import apply_interval_station_gains
from flovopy.enhanced.catalog import EnhancedCatalog
from flovopy.enhanced.event import EnhancedEvent, EnhancedEventMeta
from flovopy.enhanced.stream import EnhancedStream
from flovopy.processing.sam import VSAM, DSAM
from flovopy.processing.spectrograms import plot_strongest_trace
from flovopy.asl.config import tweak_config, ASLConfig
from flovopy.core.trace_utils import stream_add_units, add_processing_step
from flovopy.core.preprocess import preprocess_trace
from flovopy.core.remove_response import safe_pad_taper_filter, safe_pad_taper_filter_stream

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

#####################################
# -------- HELPER FUNCTIONS ---------
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


def _prepare_stream_for_sam(
    st_in: Stream,
    *,
    cfg,
    mseed_units: str | None,
    band: tuple[float, float] = (0.2, 18.0),
    taper_fraction: float = 0.05,
    debug: bool = False,
) -> Stream:
    """
    Return a new Stream in the units required by the chosen SAM:
      - VSAM → 'm/s'
      - DSAM → 'm'
    Uses flovopy.core.preprocess/remove_response for pad/taper-safe filtering & response removal.
    """
    want_units      = "m/s" if cfg.sam_class is VSAM else "m"
    response_output = "VEL" if want_units == "m/s" else "DISP"

    st = st_in.copy()
    if mseed_units:
        stream_add_units(st, default_units=mseed_units)

    out = Stream()
    for tr in st:
        units = tr.stats.get("units") or "Counts"
        tr2 = tr.copy()

        # A) Counts → remove response directly to the target output (VEL/DISP),
        #    with pad/taper-safe filtering built in.
        if units == "Counts":
            ok = preprocess_trace(
                tr2,
                do_clean=True,
                taper_fraction=taper_fraction,
                filter_type="bandpass",
                freq=band,
                corners=2,
                zerophase=True,
                inv=cfg.inventory,
                output_type=response_output,
                verbose=debug,
            )
            if not ok:
                if debug: print(f"[RESP] drop {tr2.id} (Counts→{response_output}) failed")
                continue
            tr2.stats["units"] = want_units
            if debug: add_processing_step(tr2, f"units:{want_units}")

        # B) Already physical → clean without response; convert if needed
        else:
            # Clean with pad/taper-safe filter
            ok = preprocess_trace(
                tr2,
                do_clean=True,
                taper_fraction=taper_fraction,
                filter_type="bandpass",
                freq=band,
                corners=2,
                zerophase=True,
                inv=None,                 # no response removal on physical data
                output_type="VEL",        # ignored when inv=None
                verbose=debug,
            )
            if not ok:
                if debug: print(f"[CLEAN] drop {tr2.id} (bandpass) failed")
                continue

            # Convert if domain mismatches desired SAM units
            cur = tr2.stats.get("units") or units
            if want_units == "m/s" and cur == "m":
                if debug: print(f"[CONVERT] {tr2.id} m→m/s (differentiate)")
                try:
                    tr2.detrend("linear")
                    tr2.differentiate()
                    tr2.stats["units"] = "m/s"
                except Exception as e:
                    if debug: print(f"[CONVERT:WARN] {tr2.id}: diff failed: {e}")
                    continue

            elif want_units == "m" and cur == "m/s":
                if debug: print(f"[CONVERT] {tr2.id} m/s→m (integrate + HP)")
                try:
                    tr2.detrend("linear")
                    tr2.integrate()
                    # stabilize LF drift with a *padded* highpass:
                    ok_hp = safe_pad_taper_filter(
                        tr2,
                        taper_fraction=taper_fraction,
                        filter_type="highpass",
                        freq=0.2,
                        corners=2,
                        zerophase=True,
                        inv=None,
                        output_type="DISP",
                        verbose=debug,
                    )
                    if not ok_hp:
                        if debug: print(f"[CONVERT:WARN] {tr2.id}: HP after integrate failed")
                        continue
                    tr2.stats["units"] = "m"
                except Exception as e:
                    if debug: print(f"[CONVERT:WARN] {tr2.id}: integ failed: {e}")
                    continue

        out += tr2

    return out




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
    if cfg.sam_class == DSAM:
        displacement = True
    else:
        displacement = False
        outs['reduced_disp_png'] =  outs["reduced_disp_png"].replace('disp', 'vel')
    print("[ASL:PLOT_REDUCED_DISPLACEMENT/VELOCITY]")
    ok = _safe_plot(aslobj.plot_reduced_displacement, displacement=displacement, outfile=outs["reduced_disp_png"], show=show)
    if not ok:
        outs["reduced_disp_png"] = None

    # ---- Misfit line plot
    outs["misfit_png"] = str(event_dir / f"misfit_{base_name}.png")
    print("[ASL:PLOT_MISFIT]")
    ok = _safe_plot(aslobj.plot_misfit, outfile=outs["misfit_png"], show=show)
    if not ok:
        outs["misfit_png"] = None

    # ---- Misfit heatmap
    # this is faiing after refine_and_relocate still
    try:
        outs["misfit_heatmap_png"] = str(event_dir / f"misfit_heatmap_{base_name}.png")
        print("[ASL:PLOT_MISFIT_HEATMAP]")
        ok = _safe_plot(aslobj.plot_misfit_heatmap, outfile=outs["misfit_heatmap_png"], topo_kw=topo_kw, show=show)
        if not ok:
            outs["misfit_heatmap_png"] = None
    except Exception as e:
        print(e)

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





#########################################




def run_single_event(
    mseed_file: str,
    cfg: ASLConfig,
    *,
    mseed_units: str = None,
    refine_sector: bool = False,
    station_gains_df: Optional[pd.DataFrame] = None,
    topo_kw: Optional[dict] = None,
    switch_event_ctag: bool = True,
    vertical_only: bool = True,
    enhance: bool = True,           # <— NEW
    debug: bool = True,
    reduce_time: bool = True,
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

    netmagcsv = products_dir / "network_magnitudes.csv"
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
        if reduce_time:
            raw = st.copy()

        # ----------------------------------------------------------------
        # STREAM PREPROCESSING
        # ----------------------------------------------------------------

        # 1) Station gains (optional)
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
        
        #st.merge(fill_value="interpolate")

        # 2) Prepare stream in the SAM-required domain using pad/taper-safe core utils
        st = _prepare_stream_for_sam(
            st,
            cfg=cfg,
            mseed_units=mseed_units,
            band=(0.2, 18.0),
            taper_fraction=0.05,
            debug=debug,
        )

        if len(st) < int(cfg.min_stations):
            raise RuntimeError(f"Not enough stations: {len(st)} < {cfg.min_stations}")

        # 3) Travel-time reduction AFTER units & band are correct
        shift_stream_by_travel_time(
            st,
            cfg.inventory,
            topo_kw['dome_location'],
            speed_km_s=cfg.speed,
            use_elevation=True,
            inplace=True,
            trim=True,
            verbose=True,
        )

        # 4) Build both convenience views for plotting, independent of input file units
        if cfg.sam_class is VSAM:
            vel = st
            disp = st.copy().integrate().detrend('linear')
            safe_pad_taper_filter_stream(disp, taper_fraction=0.05, filter_type="highpass", freq=0.2, corners=2, zerophase=True, inv=None, verbose=False)
            for tr in disp:
                tr.stats["units"] = "m"
        else:
            disp = st
            vel = st.copy().differentiate().detrend('linear')
            # (Velocity after differentiation is usually OK; pad-HP not strictly needed here.)
            vel.stats["units"] = "m/s"
            for tr in vel:
                tr["units"] = "m/s"

        # --------------------------------------------------------------------------
        # END OF STREAM PREPROCESSING - should now have a vel and disp Stream object
        # Make figures now
        # --------------------------------------------------------------------------
        

        # If we got this far, it is time to create output directories for products
        Path(products_dir).mkdir(parents=True, exist_ok=True)

        # Debug plots (outside tee to avoid capturing large binary dumps)
        if debug:
            stream_VEL_png = os.path.join(event_dir, "stream_VEL.png")
            if not os.path.isfile(stream_VEL_png):
                try:
                    vel.plot(equal_scale=False, outfile=stream_VEL_png)
                except Exception as e:
                    print(f"[ASL:WARN] VEL stream plot failed: {e}")
            stream_DISP_png = os.path.join(event_dir, "stream_DISP.png")
            if not os.path.isfile(stream_DISP_png):
                try:
                    disp.plot(equal_scale=False, outfile=stream_DISP_png)
                except Exception as e:
                    print(f"[ASL:WARN] DISP stream plot failed: {e}")                    

            if reduce_time:
                stream_RAW_png = os.path.join(event_dir, "stream_RAW.png")
                if not os.path.isfile(stream_RAW_png):
                    try:
                        raw.plot(equal_scale=False, outfile=stream_RAW_png)
                    except Exception as e:
                        print(f"[ASL:WARN] RAW stream plot failed: {e}")  
                del raw            

            sgram_png = os.path.join(event_dir, "spectrogram_VEL.png")
            if not os.path.isfile(sgram_png):
                plot_strongest_trace(vel, log=False, dbscale=True, cmap='inferno', fmin=0.01, fmax=20.0, secsPerFFT=5.0, outfile=sgram_png)


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
                    
                    es = EnhancedStream(stream=st.copy())

                    # compute per-trace metrics ONCE
                    es.ampengfft(
                        compute_spectral=True,
                        compute_ssam=False,
                        compute_bandratios=False,
                        compute_sam=True,
                        compute_sem=False,
                    )

                    # Provisional magnitudes using dome_location if present
                    dome = (topo_kw or {}).get("dome_location")
                    if 'lat' in dome:

                        c_earth_ms = float(getattr(cfg, "speed", 2.5)) * 1000.0    # km/s → m/s
                        Q = float(getattr(cfg, "Q", 50.0))

                        print("[MAG] Provisional magnitudes (dome_location)…")
                        es.compute_station_magnitudes(
                            inventory=cfg.inventory,
                            source_coords={'latitude':dome['lat'], 'longitude':dome['lon'], 'elevation':dome['elev']},
                            model=cfg.wave_kind, 
                            Q=cfg.Q, 
                            c_earth=c_earth_ms, 
                            use_boatwright=True,
                            attach_coords=True, 
                            compute_distances=True,
                            correction=2.4, 
                        )
                        
                        if debug:
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
                        if debug:
                            print(networkmag_df)

                            # ---- Record section
                        try:
                            record_section_png = products_dir / "record_section.png"
                            if debug:
                                print("[ASL:RECORD_SECTION]")
                            st.plot(type='section', 
                                    vred=cfg.speed*1000, 
                                    norm_method='stream', 
                                    ev_coords=[topo_kw['dome_location']['lat'], topo_kw['dome_location']['lon']], 
                                    orientation='horizontal', 
                                    fillcolors=(None ,None), 
                                    scale=1.5,
                                    outfile=record_section_png);
                        except:
                            print('Record section failed')

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
                            use_boatwright=True,
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
                                #"best_lat": best_lat, "best_lon": best_lon,  # if you have them
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
    

# -------------------------------------------------------------------
# run_all_events(): call the new global heatmap writer
# -------------------------------------------------------------------

def run_all_events(
    input_dir: str | Path,
    *,
    cfg,  # ASLConfig
    topo_kw: Optional[dict] = None,
    station_gains_df: Optional[pd.DataFrame] = None,
    refine_sector: bool = False,
    max_events: Optional[int] = None,
    use_multiprocessing: bool = True,
    workers: Optional[int] = None,
    mseed_units: Optional[str] = None,
    reduce_time: bool = True,
    switch_event_ctag: bool = True,
    mseed_extension: str = ".cleaned",
    debug: bool = True,
) -> str:
    """
    Process all event files under input_dir with the same ASLConfig.
    Writes JSONL summaries and a single global heatmap at OUTPUT_DIR/heatmap_{ctag}.png.
    """
    from flovopy.asl.wrappers import run_single_event, find_event_files  # ensure available

    if getattr(cfg, "inventory", None) is None or not getattr(cfg, "outdir", ""):
        if debug:
            print("[RUN] Building configuration before batch run…")
        cfg.build()

    run_dir = Path(cfg.outdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(set(map(str, find_event_files(input_dir, extensions=(mseed_extension,)))))
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

    if workers == 1:
        for f in files:
            try:
                res = run_single_event(
                    mseed_file=f,
                    cfg=cfg,
                    refine_sector=refine_sector,
                    station_gains_df=station_gains_df,
                    topo_kw=topo_kw,
                    switch_event_ctag=switch_event_ctag,
                    mseed_units=mseed_units,
                    reduce_time=reduce_time,
                    debug=debug,
                )
            except Exception as e:
                res = {
                    "tag": getattr(cfg, "tag_str", None) or cfg.tag(),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": __import__("traceback").format_exc(),
                    "mseed_file": f,
                }
            _append_summary(res)
            all_outputs.append(res)
            processed += 1
            print("[OK]" if "error" not in res else "[ERR]", f)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut2file = {
                ex.submit(
                    run_single_event,
                    mseed_file=f,
                    cfg=cfg,
                    refine_sector=refine_sector,
                    station_gains_df=station_gains_df,
                    topo_kw=topo_kw,
                    switch_event_ctag=switch_event_ctag,
                    mseed_units=mseed_units,
                    reduce_time=reduce_time,
                    debug=debug,
                ): f
                for f in files
            }
            for fut in as_completed(fut2file):
                f = fut2file[fut]
                try:
                    res = fut.result()
                except Exception:
                    import traceback
                    res = {
                        "tag": getattr(cfg, "tag_str", None) or cfg.tag(),
                        "error": "FutureError",
                        "traceback": traceback.format_exc(),
                        "mseed_file": f,
                    }
                _append_summary(res)
                all_outputs.append(res)
                processed += 1
                print("[OK]" if "error" not in res else "[ERR]", f)

    dt = UTCDateTime() - t0
    print(f"[DONE] Processed {processed}/{len(files)} events in {dt:.1f}s")

    # ---- Single global heatmap across events for this ctag ----
    try:
        ctag = getattr(cfg, "tag_str", None) or cfg.tag()
        write_global_heatmap_for_ctag(
            output_root=run_dir.parent if switch_event_ctag else run_dir,
            ctag=ctag,
            node_spacing_m=cfg.gridobj.node_spacing_m,
            topo_kw=topo_kw,
            title=f"Energy Heatmap — {ctag}",
            include_refined=True,
        )
    except Exception as e:
        print(f"[HEATMAP:WARN] Failed to generate global heatmap: {e}")

    # ---- Optional: assemble EnhancedCatalogs from per-event outputs (unchanged)
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




# ---------------------------------------------------------------------
# Simple sweep (“Monte Carlo”) runner built on tweak_config / .copy()
# ---------------------------------------------------------------------

def run_event_sweep(
    mseed_file: str | Path,
    *,
    baseline_cfg: ASLConfig,
    axes: Optional[Dict[str, List]] = None,
    changes: Optional[List[Dict]] = None,
    variants: Optional[Dict[str, ASLConfig]] = None,  # skip building if you already have them
    station_gains_df: Optional[pd.DataFrame] = None,
    topo_kw: Optional[dict] = None,
    switch_event_ctag: bool = True,
    mseed_units: Optional[str] = None,
    reduce_time: bool = True,
    parallel: bool = True,
    workers: Optional[int] = None,
    debug: bool = True,
) -> List[Dict]:
    """
    Run a single event across many configs. Returns list of per-run dicts from run_single_event().
    - Provide either `variants` or (`axes` and/or `changes`). If none given, runs baseline only.
    - Minimal JSONL log is written under each config’s tag folder: <cfg.outdir>/<event>/sweep.jsonl
    """
    

    mseed_file = str(mseed_file)
    event_stem = Path(mseed_file).stem

    # Build variants if not supplied
    if variants is None:
        variants = tweak_config(
            baseline_cfg,
            axes=axes,
            changes=changes,
            include_baseline=True,   # include baseline by default for easy comparison
        )

    # Ensure all configs are “built” so tag/outdir exist (copy() already builds when needed;
    # for safety, call build() if someone passed an unbuilt cfg).
    ready = []
    for cfg in variants.values():
        if not getattr(cfg, "outdir", "") or getattr(cfg, "ampcorr", None) is None:
            cfg = cfg.build()
        ready.append(cfg)

    def _run_one(cfg: ASLConfig) -> Dict:
        tag = getattr(cfg, "tag_str", None) or cfg.tag()
        outdir = Path(cfg.outdir) / event_stem
        outdir.mkdir(parents=True, exist_ok=True)
        log_jsonl = outdir / "sweep.jsonl"
        try:
            res = run_single_event(
                mseed_file=mseed_file,
                cfg=cfg,
                refine_sector=False,
                station_gains_df=station_gains_df,
                topo_kw=topo_kw,
                switch_event_ctag=switch_event_ctag,
                mseed_units=mseed_units,
                reduce_time=reduce_time,
                debug=debug,
            )
        except Exception as e:
            res = {
                "tag": tag,
                "mseed_file": mseed_file,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        # append a minimal JSONL record
        try:
            with open(log_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(res) + "\n")
        except Exception:
            pass
        if debug:
            print(("[OK] " if "error" not in res else "[ERR] ") + tag)
        return res

    # Serial or parallel
    if not parallel:
        return [_run_one(cfg) for cfg in ready]

    if workers is None:
        cpu = os.cpu_count() or 2
        workers = max(1, cpu - 2)
    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, cfg): cfg for cfg in ready}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"tag": "<unknown>", "mseed_file": mseed_file,
                                "error": f"{type(e).__name__}: {e}"})
    return results


'''
EXAMPLES:

1. Grid search example:
-----------------------
# build a sweep that tries speed × Q, and also the whole grid with 2D distances
# Cartesian sweep over speed × Q

from flovopy.asl.config import tweak_config

variants = tweak_config(
    baseline_cfg,
    axes={
        "speed": [1.5, 3.0],
        "Q": [23, 100],
    },
    changes=[{"dist_mode": "2d"}],  # also try all of the above in 2D
    include_baseline=True,          # optional: include baseline itself
)

# run a SINGLE event across the sweep
_ = run_event_sweep(
    mseed_file=best_event_files[0],
    baseline_cfg=baseline_cfg,
    variants=variants,                 # pass prebuilt variants (or omit to build inline)
    station_gains_df=station_gains_df, # optional
    topo_kw=topo_kw,                   # optional
    mseed_units="m/s",
    reduce_time=True,
    parallel=True,
    debug=True,
)

We can also use:


2. Monte Carlo:
---------------

import random
def sample_axes(ranges: Dict[str, List], n: int) -> Dict[str, List]:
    return {k: [random.choice(v) for _ in range(n)] for k, v in ranges.items()}

rand_axes = sample_axes({"speed":[1.0,1.5,2.0,3.0], "Q":[20,40,60,80,100]}, n=20)
variants = build_variants(baseline_cfg, axes=rand_axes, include_baseline=True)

'''

# -------------------------------------------------------------------
# EVERYTHING BELOW IS FOR HEATMAPS
# -------------------------------------------------------------------

def _load_source_csv(path: str) -> pd.DataFrame | None:
    """
    Load a single source_<ctag>.csv, requiring lat/lon and DR (case-insensitive for DR).
    Output columns are exactly: lat, lon, DR
    """
    try:
        df = pd.read_csv(path)
        # normalize column names for selection, but we will restore 'DR' as uppercase
        lower = {c.lower(): c for c in df.columns}
        # require lat/lon
        if "lat" not in lower or "lon" not in lower:
            return None
        # DR can appear as 'DR' or 'dr'
        dr_key = lower.get("dr", None)
        if dr_key is None:
            return None

        sub = df[[lower["lat"], lower["lon"], dr_key]].copy()
        sub.columns = ["lat", "lon", "DR"]  # enforce exact output names
        # coerce and drop bad rows
        sub["lat"] = pd.to_numeric(sub["lat"], errors="coerce")
        sub["lon"] = pd.to_numeric(sub["lon"], errors="coerce")
        sub["DR"]  = pd.to_numeric(sub["DR"],  errors="coerce")
        sub.dropna(subset=["lat", "lon", "DR"], inplace=True)

        return sub if not sub.empty else None
    except Exception:
        return None


def collect_sources_for_ctag(output_root: str | Path, ctag: str, refined: bool = False) -> pd.DataFrame:
    """
    Collect source CSVs for a given ctag.
    If refined=True, loads only *_refined.csv.
    If refined=False, loads only the unrefined CSVs.
    """
    output_root = Path(output_root)
    dfs: list[pd.DataFrame] = []

    for ev_dir in sorted(d for d in output_root.iterdir() if d.is_dir()):
        run_dir = ev_dir / ctag
        if not run_dir.is_dir():
            continue

        if refined:
            f = run_dir / f"source_{ctag}_refined.csv"
        else:
            f = run_dir / f"source_{ctag}.csv"

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
    cmap = 'viridis',
    scale: float = 1.0,
    verbose: bool = False,
) -> str | None:
    """
    Aggregate source_<ctag>.csv (+ optional _refined) across OUTPUT_DIR/{event}/{ctag}
    and write one heatmap: OUTPUT_DIR/heatmap_{ctag}.png
    """
    from flovopy.asl.map import plot_heatmap_colored  # local import to avoid cycles

    results = {"primary": None, "refined": None}

    for refined_flag, label in [(False, "primary"), (True, "refined")]:
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



'''
Example:

# Basic: aggregate all events found under OUTPUT_DIR/<event>/<ctag>
res = rebuild_heatmap_for_ctag(
    OUTPUT_DIR,
    ctag=cfg.tag(),
    topo_kw=topo_kw,
    cfg=cfg,                   # provides node_spacing_m and a nice title
    write_catalogs=False       # set True if you also want per-event catalogs refreshed
)
print(res["heatmap_png"])

# With a date window filter (example: event folders like '2024-08-02_13-41-00')
from obspy import UTCDateTime
date_from = UTCDateTime("2024-07-01")
date_to   = UTCDateTime("2024-07-31")

def in_july(name: str) -> bool:
    # parse your event folder naming here; this is just an example
    # expect 'YYYY-MM-DD_HH-MM-SS'
    try:
        ts = UTCDateTime(name.replace("_", " ").replace("-", ":", 2))
        return date_from <= ts <= date_to
    except Exception:
        return False

res = rebuild_heatmap_for_ctag(
    OUTPUT_DIR,
    ctag=cfg.tag(),
    topo_kw=topo_kw,
    cfg=cfg,
    event_filter=in_july,
    outfile_suffix="2024-07",
)

'''


# -------------------------------------------------------------------
# Enhanced catalogs (unchanged)
# -------------------------------------------------------------------

def enhanced_catalogs_from_outputs(
    outputs_list: List[Dict[str, Any]],
    *,
    outdir: str,
    write_files: bool = True,
    load_waveforms: bool = False,
    primary_name: str = "catalog_primary",
    refined_name: str = "catalog_refined",
) -> Dict[str, Any]:
    import os
    from flovopy.enhanced.event import EnhancedEvent
    from flovopy.enhanced.catalog import EnhancedCatalog

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
# Assimilation helpers across OUTPUT_DIR/{event}/{tag}
# - CSV gatherers
# - Catalog builders (primary/refined)
# - EventRate CSV + plots (count, cumulative magnitude, dual)
# ======================================================================

from flovopy.enhanced.eventrate import EventRate, EventRateConfig

def _iter_event_run_dirs(output_root: str | Path, ctag: str) -> List[Path]:
    root = Path(output_root)
    out: List[Path] = []
    for ev_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        # skip a top-level folder accidentally named exactly like the ctag
        if ev_dir.name == ctag:
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
    Given a run_dir and base_name (e.g., 'VSAM_...'), return (qml, json) if both exist, else (None, None).
    """
    qml = run_dir / f"{base_name}.qml"
    jsn = run_dir / f"{base_name}.json"
    if qml.is_file() and jsn.is_file():
        return str(qml), str(jsn)
    return None, None


def _load_catalog_for_tag_kind(output_root: str | Path, ctag: str, *, refined: bool) -> EnhancedCatalog:
    """
    Walk OUTPUT_DIR/{event}/{ctag}/ and collect EnhancedEvents from {ctag}.qml/json
    or {ctag}_refined.qml/json depending on `refined`.
    """
    base = f"{ctag}_refined" if refined else f"{ctag}"
    records = []
    for run_dir in _iter_event_run_dirs(output_root, ctag):
        qml, jsn = _discover_qml_json_pairs(run_dir, base)
        if qml and jsn:
            try:
                enh = EnhancedEvent.load(os.path.splitext(qml)[0])
                # Do not attach waveforms during assimilation
                enh.stream = None
                records.append(enh)
            except Exception as e:
                print(f"[ASSIM:WARN] Skipping {run_dir}: {e}")

    return EnhancedCatalog(events=[r for r in records], records=records,
                           description=("Refined" if refined else "Primary") + f" ASL locations ({ctag})")


def _write_event_rate_outputs(cat: EnhancedCatalog, outdir: Path, prefix: str,
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
    import matplotlib.pyplot as plt

    # 1) Event count
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    er.plot_event_count(ax=ax1)
    fig1.tight_layout()
    f1 = outdir / f"eventrate_{prefix}_count.png"
    fig1.savefig(f1, dpi=150)
    plt.close(fig1)

    # 2) Cumulative magnitude
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    er.plot_cumulative_magnitude(ax=ax2)
    fig2.tight_layout()
    f2 = outdir / f"eventrate_{prefix}_cumMag.png"
    fig2.savefig(f2, dpi=150)
    plt.close(fig2)

    # 3) Dual
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

    Returns:
      {
        "primary": EnhancedCatalog,
        "refined": EnhancedCatalog,
        "primary_eventrate": {"csv": ..., "plots": [...] } or {},
        "refined_eventrate": {"csv": ..., "plots": [...] } or {},
      }
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