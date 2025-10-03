# flovopy/asl/config.py
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterable, Callable, Sequence

import numpy as np
from obspy.core.inventory import Inventory, read_inventory

from flovopy.asl.distances import compute_or_load_distances, distances_signature, summarize_station_node_distances
from flovopy.asl.ampcorr import AmpCorrParams, AmpCorr, summarize_ampcorr_ranges
from flovopy.processing.sam import VSAM  # default

from __future__ import annotations
import os, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

Number = float | int

@dataclass(frozen=True)
class ASLConfig:
    # ---- knobs (formerly SweepPoint + essentials) ----
    wave_kind: str                 # 'surface'|'body'
    station_corr: bool             # True/False
    speed: float                   # km/s
    Q: int
    dist_mode: str                 # '2d'|'3d'
    misfit_engine: str             # 'l2', 'huber', ...
    peakf: float                   # Hz

    window_seconds: float = 5.0
    min_stations: int = 5
    sam_class: type = VSAM         # class, not instance
    sam_metric: str = "mean"

    # ---- inputs for build() ----
    inventory_xml: Inventory | str | Path | None = None
    output_base: str | Path | None = None
    gridobj: Any | None = None
    global_cache: str | Path | None = None
    debug: bool = False

    # ---- built artifacts (populated by build()) ----
    tag_str: str = field(init=False, default="")
    outdir: str = field(init=False, default="")
    inventory: Inventory | None = field(init=False, default=None)
    node_distances_km: Dict[str, np.ndarray] | None = field(init=False, default=None)
    station_coords: Dict[str, dict] | None = field(init=False, default=None)
    dist_meta: Dict[str, Any] | None = field(init=False, default=None)
    ampcorr: AmpCorr | None = field(init=False, default=None)

    # ------------- labels / caching -------------
    def tag(self) -> str:
        parts = [
            self.sam_class.__name__,
            self.sam_metric,
            f"{int(self.window_seconds)}s",
            self.wave_kind,
            f"v{self.speed:g}",
            f"Q{self.Q}",
            f"F{self.peakf:g}",
            self.dist_mode,
            self.misfit_engine,
        ]
        if self.station_corr:
            parts.append("SC")
        return "_".join(parts)

    # ------------- builder (was build_asl_config) -------------
    def build(self) -> "ASLConfig":
        assert self.gridobj is not None, "ASLConfig.build(): gridobj must be set"
        assert self.output_base is not None, "ASLConfig.build(): output_base must be set"
        assert self.inventory_xml is not None, "ASLConfig.build(): inventory_xml must be set"

        self.tag_str = self.tag()
        output_base = Path(self.output_base)
        outdir = output_base / self.tag_str
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = str(outdir)

        # inventory
        inv = self.inventory_xml if isinstance(self.inventory_xml, Inventory) else read_inventory(str(self.inventory_xml))
        self.inventory = inv
        if self.debug:
            print('[build_asl_config] Inventory:', inv)

        # cache roots
        cache_root = (Path(self.global_cache) if self.global_cache else output_base) / self.tag_str
        cache_root.mkdir(parents=True, exist_ok=True)

        # distances
        use_elev = (self.dist_mode.lower() == "3d")
        dist_cache_dir = cache_root / "distances" / ("3d" if use_elev else "2d")
        dist_cache_dir.mkdir(parents=True, exist_ok=True)

        print('[build_asl_config] Computing distances to each node from each station')
        node_distances_km, station_coords, dist_meta = compute_or_load_distances(
            self.gridobj,
            inventory=inv,
            stream=None,
            cache_dir=str(dist_cache_dir),
            force_recompute=False,
            use_elevation=use_elev,
        )
        self.node_distances_km = node_distances_km
        self.station_coords = station_coords
        self.dist_meta = dist_meta

        if self.debug:
            print('[build_asl_config]', summarize_station_node_distances(node_distances_km, reduce_to_station=True))

        # amp corrections
        ampcorr_cache = cache_root / "ampcorr"
        ampcorr_cache.mkdir(parents=True, exist_ok=True)

        mask_idx = getattr(self.gridobj, "_node_mask_idx", None)
        total_nodes = int(self.gridobj.gridlat.size)
        mask_sig = (int(mask_idx.size), total_nodes) if mask_idx is not None else None

        params = AmpCorrParams(
            assume_surface_waves=(self.wave_kind == "surface"),
            wave_speed_kms=float(self.speed),
            Q=self.Q,
            peakf=float(self.peakf),
            grid_sig=self.gridobj.signature(),
            inv_sig=tuple(sorted(node_distances_km.keys())),
            dist_sig=distances_signature(node_distances_km),
            mask_sig=mask_sig,
            code_version="v1",
        )
        ampcorr = AmpCorr(params, cache_dir=str(ampcorr_cache))
        print('[build_asl_config] Computing amplitude corrections for each node from each station')
        ampcorr.compute_or_load(node_distances_km, inventory=inv)
        ampcorr.validate_against_nodes(total_nodes)
        self.ampcorr = ampcorr

        if self.debug:
            print('[build_asl_config]', summarize_ampcorr_ranges(
                ampcorr.corrections,
                total_nodes=self.gridobj.gridlat.size,
                reduce_to_station=True,
                include_percentiles=True,
                sort_by="min_corr",
            ))

        return self

    # ------------- sweep helpers -------------
    @staticmethod
    def generate_grid(
        *,
        wave_kinds: Sequence[str] = ("surface", "body"),
        station_corr_opts: Sequence[bool] = (True, False),
        speeds: Sequence[float] = (1.5, 3.2),
        Qs: Sequence[int] = (50, 200),
        dist_modes: Sequence[str] = ("2d", "3d"),
        misfit_engines: Sequence[str] = ("l2", "huber"),
        peakfs: Sequence[float] = (4.0, 8.0),
        window_seconds: float = 5.0,
        min_stations: int = 5,
        sam_class: type = VSAM,
        sam_metric: str = "mean",
        # context (these can be set per-config after creation if you prefer)
        inventory_xml: Inventory | str | Path | None = None,
        output_base: str | Path | None = None,
        gridobj: Any | None = None,
        global_cache: str | Path | None = None,
        debug: bool = False,
    ) -> List["ASLConfig"]:
        out: List[ASLConfig] = []
        for w, sc, v, q, d, m, f in itertools.product(
            wave_kinds, station_corr_opts, speeds, Qs, dist_modes, misfit_engines, peakfs
        ):
            out.append(ASLConfig(
                wave_kind=w, station_corr=sc, speed=float(v), Q=int(q),
                dist_mode=d, misfit_engine=m, peakf=float(f),
                window_seconds=float(window_seconds), min_stations=int(min_stations),
                sam_class=sam_class, sam_metric=sam_metric,
                inventory_xml=inventory_xml, output_base=output_base, gridobj=gridobj,
                global_cache=global_cache, debug=debug,
            ))
        return out
    

###############

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
) -> ASLConfig:
    """
    Back-compat shim: construct and .build() an ASLConfig and return it.
    Old code returned a dict; new code returns an ASLConfig instance.
    """
    if isinstance(sweep_or_cfg, ASLConfig):
        cfg = sweep_or_cfg
        # allow overriding a few build-time fields:
        if inventory_xml is not None: cfg.inventory_xml = inventory_xml
        if output_base  is not None: cfg.output_base  = output_base
        if gridobj      is not None: cfg.gridobj      = gridobj
        if global_cache is not None: cfg.global_cache = global_cache
        if debug               is not None: cfg.debug = debug
        if sam_class is not None: cfg.sam_class = sam_class
        if sam_metric is not None: cfg.sam_metric = sam_metric
        if window_seconds is not None: cfg.window_seconds = float(window_seconds)
        if min_stations   is not None: cfg.min_stations   = int(min_stations)
        if peakf          is not None: cfg.peakf          = float(peakf)
        return cfg.build()

    # Otherwise, sweep_or_cfg is a “point” carrying the physical knobs.
    # Expect attributes: wave_kind, station_corr, speed, Q, dist_mode, misfit_engine
    cfg = ASLConfig(
        wave_kind=sweep_or_cfg.wave_kind,
        station_corr=bool(sweep_or_cfg.station_corr),
        speed=float(sweep_or_cfg.speed),
        Q=int(sweep_or_cfg.Q),
        dist_mode=str(sweep_or_cfg.dist_mode),
        misfit_engine=str(sweep_or_cfg.misfit_engine),
        peakf=float(peakf),

        window_seconds=float(window_seconds),
        min_stations=int(min_stations),
        sam_class=(sam_class if sam_class is not None else VSAM),
        sam_metric=sam_metric,

        inventory_xml=inventory_xml,
        output_base=output_base,
        gridobj=gridobj,
        global_cache=global_cache,
        debug=debug,
    )
    return cfg.build()


def _asl_output_source_results(
    aslobj,
    *,
    stream,                 # ObsPy Stream used for station labels on plots
    event_dir: str,
    cfg: ASLConfig,
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
        stream=stream,                               # optional
        inventory=cfg.inventory,                     # optional
        title_comment=base_title,
    )
    qml_out, json_out = aslobj.event.save(
        event_dir, base_name,
        write_quakeml=True,
        write_obspy_json=getattr(cfg, "write_event_obspy_json", False),
        include_trajectory_in_sidecar=getattr(cfg, "include_trajectory", True),
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
        stations=[tr.stats.station for tr in (stream or [])],
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
    cfg: ASLConfig,                     # << object, not dict
    dry_run: bool = False,
    peakf_override: Optional[float] = None,
    station_gains_df: Optional["pd.DataFrame"] = None,
    allow_station_fallback: bool = True,
    *,
    refine_sector: bool = False,
    vertical_only: bool = True,
    topo_kw: dict = None,
    show: bool=True,
    debug: bool=False,
):
    """
    Run ASL on a single (already preprocessed) event stream.

    Requires cfg to be an ASLConfig that has been .build()'d or will self-build.
    """

    # Information about results from fast_locate() and refine_and_relocate()
    primary_out = None
    refined_out = None

    print(f"[ASL] Preparing VSAM for event folder: {event_dir}")
    os.makedirs(event_dir, exist_ok=True)

    # 1) Station gains (optional)
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

    if vertical_only:
        stream = stream.select(component='Z')

    units_by_class  = {DSAM: "m",    VSAM: "m/s"}
    output_by_class = {DSAM: "DISP", VSAM: "VEL"}
    output = output_by_class.get(cfg.sam_class)

    for tr in stream:
        units = tr.stats.get("units") or "Counts"
        tr.stats["units"] = units
        if units == "Counts":
            median_abs = float(np.nanmedian(np.abs(tr.data))) if tr.data is not None else np.inf
            if median_abs < 1.0:
                new_units = units_by_class.get(cfg.sam_class)
                if new_units:
                    tr.stats["units"] = new_units
            else:
                print(f'Removing instrument response from {tr.id}')
                tr.remove_response(inventory=cfg.inventory, output=output)

    if debug:
        stream.plot(equal_scale=False, outfile=os.path.join(event_dir, 'stream.png'))

    # 2) Build SAM object
    samObj = cfg.sam_class(stream=stream, sampling_interval=1.0)
    if len(samObj.dataframes) == 0:
        raise IOError("[ASL:ERR] No dataframes in SAM object")

    if not dry_run:
        print("[ASL:PLOT] Writing SAM preview")
        samObj.plot(metrics=cfg.sam_metric, equal_scale=False, outfile=os.path.join(event_dir, "SAM.png"))
        plt.close('all')

    # 3) Decide event peakf
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
        if topo_kw and topo_kw.get('dome_location'):
            if cfg.sam_class == VSAM:
                print('[ASL_SAUSAGE]: Sanity check: VSAM → VR/VRS at dome')
                VR = samObj.compute_reduced_velocity(
                    cfg.inventory,
                    topo_kw['dome_location'],
                    surfaceWaves=ampcorr_params.assume_surface_waves,
                    Q=ampcorr_params.Q,
                    wavespeed_kms=ampcorr_params.wave_speed_kms,
                    peakf=peakf_event,
                )
                print(VR)
                VR.plot(outfile=os.path.join(event_dir, "VR_at_dome.png"))
            elif cfg.sam_class == DSAM:
                print('[ASL_SAUSAGE]: Sanity check: DSAM → DR/DRS at dome')
                DR = samObj.compute_reduced_displacement(
                    cfg.inventory,
                    topo_kw['dome_location'],
                    surfaceWaves=ampcorr_params.assume_surface_waves,
                    Q=ampcorr_params.Q,
                    wavespeed_kms=ampcorr_params.wave_speed_kms,
                    peakf=peakf_event,
                )
                print(DR)
                DR.plot(outfile=os.path.join(event_dir, "DR_at_dome.png"))

    # 4) Amplitude corrections cache (swap if peakf differs)
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

        ampcorr = AmpCorr(new_params, cache_dir=ampcorr.cache_dir)
        ampcorr.compute_or_load(cfg.node_distances_km, inventory=cfg.inventory)
        cfg.ampcorr = ampcorr
    else:
        print(f"[ASL] Using existing amplitude corrections (peakf={ampcorr.params.peakf} Hz)")

    # 5) Build ASL object and inject geometry/corrections
    print("[ASL] Building ASL object…")
    aslobj = ASL(
        samObj,
        config=cfg,     # pass the object
    )

    idx = _grid_mask_indices(cfg.gridobj)
    if idx is not None and idx.size:
        aslobj._node_mask = idx

    # 6) Locate
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

    # 7) Outputs (baseline)
    if not dry_run:
        primary_out = _asl_output_source_results(
            aslobj,
            stream=stream,
            event_dir=event_dir,
            cfg=cfg,
            peakf_event=peakf_event,
            topo_kw=topo_kw,
            suffix="",
            show=show,
            debug=debug,
        )

    # Scaffold checks
    src = aslobj.source
    for k in ("lat","lon","DR","misfit","t","nsta"):
        v = np.asarray(src.get(k, []))
        finite_pct = 0.0 if v.size == 0 else (100 * np.isfinite(v).mean() if np.issubdtype(v.dtype, np.number) else float("nan"))
        print(f"[CHECK:SRC] {k}: shape={v.shape} dtype={v.dtype} finite%={finite_pct:.1f}")

    # 8) Optional sector refinement
    if refine_sector and topo_kw and topo_kw.get('dome_location'):
        print("[ASL] Refinement pass: sector wedge from dome apex…")
        try:
            apex_lat = float(topo_kw['dome_location']["lat"])
            apex_lon = float(topo_kw['dome_location']["lon"])
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
                topo_kw=topo_kw,
                suffix="_refined",
                show=False,
                debug=debug,
            )

    return {"primary": primary_out, "refined": refined_out}


##############

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import os, traceback
import numpy as np

# One parameter draw
def _one_param_run(
    mseed_file: str,
    cfg: ASLConfig,
    *,
    inventory_xml,
    output_base,
    gridobj,
    topo_kw: dict | None,
    station_gains_df=None,
    auto_correct: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    try:
        # Build tag-scoped caches (distances, ampcorr) and get a concrete config
        built_cfg = cfg.build(
            inventory_xml=inventory_xml,
            output_base=output_base,
            gridobj=gridobj,
            global_cache=None,
            debug=debug,
        )

        # Run the event with these params
        return run_single_event(
            mseed_file=mseed_file,
            cfg=built_cfg,
            refine_sector=False,
            station_gains_df=station_gains_df,
            topo_kw=topo_kw,
            auto_correct=auto_correct,   # keep only if run_single_event accepts it
            debug=debug,
        )

    except Exception as e:
        traceback.print_exc()
        return {"tag": getattr(cfg, "tag", "<unknown>"), "error": f"{type(e).__name__}: {e}"}

def run_event_monte_carlo(
    mseed_file: str,
    configs: List[ASLConfig],
    *,
    inventory_xml,
    output_base,
    gridobj,
    topo_kw: dict | None = None,
    station_gains_df=None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    auto_correct: bool = True,
    debug: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run ONE event across many parameter draws (configs).
    Returns a list of per-config outputs (like run_single_event).
    """
    if not parallel:
        results: List[Dict[str, Any]] = []
        for cfg in configs:
            results.append(_one_param_run(
                mseed_file, cfg,
                inventory_xml=inventory_xml,
                output_base=output_base,
                gridobj=gridobj,
                topo_kw=topo_kw,
                station_gains_df=station_gains_df,
                auto_correct=auto_correct,
                debug=debug,
            ))
        return results

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = max(1, cpu - 2)

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                _one_param_run,
                mseed_file, cfg,
                inventory_xml=inventory_xml,
                output_base=output_base,
                gridobj=gridobj,
                topo_kw=topo_kw,
                station_gains_df=station_gains_df,
                auto_correct=auto_correct,
                debug=debug
            )
            for cfg in configs
        ]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"tag": "unknown", "error": f"{type(e).__name__}: {e}"})
    return results

def monte_carlo_wrapper_example():
    # Make ~20 random configs
    configs = ASLConfig.monte_carlo(
        n=20,
        wave_kind_prior=lambda: np.random.choice(['surface','body']),
        station_corr_prior=lambda: bool(np.random.randint(0,2)),
        speed_prior=lambda: float(np.random.uniform(1.2, 3.5)),
        Q_prior=lambda: int(np.random.choice([50,100,200,300])),
        dist_mode_prior=lambda: np.random.choice(['2d','3d']),
        misfit_engine_prior=lambda: 'l2',
        peakf_prior=lambda: float(np.random.choice([4.0, 6.0, 8.0])),
        window_seconds=5.0,
        min_stations=5,
    )

    results = run_event_monte_carlo(
        mseed_file="/path/to/event.mseed",
        configs=configs,
        inventory_xml="/path/to/stationxml.xml",
        output_base="/tmp/asl_runs",
        gridobj=final_grid,
        topo_kw=topo_kw,
        parallel=True,
    )


################

# ---------------------------------------------------------------------
# Single event
# ---------------------------------------------------------------------
from pathlib import Path
from typing import Optional, Dict, Any
import os, time, traceback
from obspy import read
import pandas as pd

from flovopy.asl.config import ASLConfig
from flovopy.asl.pipeline import asl_sausage
from flovopy.asl.catalog_tools import enhanced_catalogs_from_outputs  # if you had it elsewhere
from flovopy.asl.collect import collect_node_results                   # ditto
from flovopy.asl.find import find_event_files                          # ditto
from flovopy.asl.map import plot_heatmap_colored                       # already imported in your file


def run_single_event(
    mseed_file: str,
    cfg: ASLConfig,
    *,
    refine_sector: bool = False,
    station_gains_df: Optional[pd.DataFrame] = None,
    topo_kw: Optional[dict] = None,
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Minimal, notebook-friendly runner (delegates to asl_sausage).

    Returns:
      {
        "tag": cfg.tag,
        "outdir": cfg.outdir,
        "event_dir": "<per-event dir>",
        "outputs": {"primary": {...}, "refined": {... or None}},
        "elapsed_s": <float>
      }
      or, on error:
      {
        "tag": cfg.tag,
        "error": "...",
        "outdir": cfg.outdir
      }
    """
    t0 = time.time()
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)

    try:
        # Read stream and quick station-count guard
        st = read(mseed_file).select(component="Z")
        if len(st) < cfg.min_stations:
            raise RuntimeError(f"Not enough stations: {len(st)} < {cfg.min_stations}")

        event_dir = Path(cfg.outdir) / Path(mseed_file).stem
        event_dir.mkdir(exist_ok=True)

        # Show grid (once per run dir, unless already saved)
        if debug:
            print('You are using this Grid:')
            print(cfg.gridobj)
            gridpng = os.path.join(cfg.outdir, "grid.png")
            if not os.path.isfile(gridpng):
                cfg.gridobj.plot(show=True, topo_map_kwargs=topo_kw, force_all_nodes=True, outfile=gridpng)

        print(f"[ASL] Running single event: {mseed_file}")
        outputs = asl_sausage(
            stream=st,
            event_dir=str(event_dir),
            cfg=cfg,
            dry_run=False,
            peakf_override=None,
            station_gains_df=station_gains_df,
            allow_station_fallback=True,
            refine_sector=refine_sector,
            vertical_only=True,
            topo_kw=topo_kw,
            debug=debug,
        )

        # Validate shape of outputs
        if not isinstance(outputs, dict) or "primary" not in outputs:
            raise RuntimeError("asl_sausage() did not return the expected outputs dict.")
        primary = outputs.get("primary")
        if not isinstance(primary, dict) or not primary.get("qml") or not primary.get("json"):
            raise RuntimeError("asl_sausage() returned outputs without required 'qml'/'json' paths for the primary solution.")

        summ = {
            "tag": cfg.tag,
            "outdir": cfg.outdir,
            "event_dir": str(event_dir),
            "outputs": outputs,
            "elapsed_s": round(time.time() - t0, 2),
        }
        print(f"[ASL] Single-event summary: {summ}")
        return summ

    except Exception as e:
        traceback.print_exc()
        return {
            "tag": cfg.tag,
            "error": f"{type(e).__name__}: {e}",
            "outdir": cfg.outdir,
        }
    
# ---------------------------------------------------------------------
# All events (with optional multiprocessing)
# ---------------------------------------------------------------------
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

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

    t0 = time.time()
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
                    "tag": cfg.tag,
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
        # NOTE: cfg must be picklable; dataclass ASLConfig is fine as long as fields are picklable.
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
                        "tag": cfg.tag,
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
                node_spacing_m=cfg.gridobj.node_spacing_m,
                outfile=heat_png,
                title=f"Energy Heatmap — {cfg.tag}",
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