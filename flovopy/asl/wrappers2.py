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
# flovopy/asl/config.py
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, List, Dict, Any

import numpy as np
from obspy.core.inventory import Inventory, read_inventory

from .distances import compute_or_load_distances, distances_signature, summarize_station_node_distances
from .ampcorr import AmpCorrParams, AmpCorr, summarize_ampcorr_ranges
from ..processing.sam import VSAM  # default SAM class

@dataclass
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
        # build tag-scoped caches (distances, ampcorr)
        asl_config = cfg.build(
            inventory_xml=inventory_xml,
            output_base=output_base,
            gridobj=gridobj,
            global_cache=None,
            debug=debug,
        )
        # run the event with these params
        return run_single_event(
            mseed_file,
            asl_config=asl_config,
            refine_sector=False,
            station_gains_df=station_gains_df,
            topo_kw=topo_kw,
            auto_correct=auto_correct,
            debug=debug,
        )
    except Exception as e:
        traceback.print_exc()
        return {"tag": cfg.tag(), "error": f"{type(e).__name__}: {e}"}

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
    Returns list of per-config outputs (like run_single_event).
    """
    if not parallel:
        results = []
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

