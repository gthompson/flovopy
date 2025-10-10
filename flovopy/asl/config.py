from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
from obspy.core.inventory import Inventory, read_inventory
from flovopy.asl.distances import compute_or_load_distances, distances_signature
from flovopy.asl.grid import summarize_station_node_distances, Grid
from flovopy.asl.ampcorr import AmpCorrParams, AmpCorr, summarize_ampcorr_ranges
from flovopy.processing.sam import VSAM  # default

def _is_pathlike(x):
    return isinstance(x, (str, Path))

def _value_equal(a, b) -> bool:
    # Same object shortcut
    if a is b:
        return True

    # Handle None vs non-None
    if (a is None) ^ (b is None):
        return False

    # Pathlike: normalize to absolute string
    if _is_pathlike(a) and _is_pathlike(b):
        try:
            return Path(a).resolve() == Path(b).resolve()
        except Exception:
            return str(a) == str(b)

    # Pandas DataFrame/Series: structural equality
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        try:
            return a.equals(b)
        except Exception:
            return False
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        try:
            return a.equals(b)
        except Exception:
            return False

    # Numpy arrays
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b, equal_nan=True)
        except Exception:
            return False

    # Fallback: regular equality (guard exceptions)
    try:
        return a == b
    except Exception:
        return False
@dataclass(frozen=True)
class ASLConfig:
    """
    Configuration object for Amplitude Source Location (ASL) runs.

    Encapsulates all parameters, inputs, and cached artifacts needed to
    configure and execute a single ASL run. Replaces the legacy ``asl_config``
    dict with a strongly-typed, self-contained object.

    Parameters
    ----------
    inventory : Inventory | str | Path
        ObsPy Inventory *or* path to a StationXML file.
    output_base : str | Path
        Root directory where run-specific subfolders are created.
    gridobj : Any
        Grid object representing ASL nodes (must implement ``signature()``,
        and expose grid size via ``gridlat.size``; optional mask via
        ``_node_mask_idx``).
    global_cache : str | Path
        Root directory for caches (distances, amplitude corrections).
    wave_kind : str, default "surface"
        Wave type ("surface" or "body").
    station_correction_dataframe : pd.DataFrame | str | Path | None, default None
        Per-station gain table to apply. If provided (either as a DataFrame
        or path), station corrections are considered enabled.
    speed : float, default 1.5
        Assumed wave speed (km/s).
    Q : int, default 100
        Attenuation Q.
    dist_mode : str, default "3d"
        Distance mode ("2d" or "3d").
    misfit_engine : str, default "l2"
        Misfit backend key (e.g., "l2", "huber").
    peakf : float, default 8.0
        Peak frequency of interest (Hz).
    window_seconds : float, default 5.0
        ASL analysis window length (s).
    min_stations : int, default 5
        Minimum number of stations required.
    sam_class : type, default VSAM
        SAM class (e.g., VSAM, DSAM). Pass the class, not an instance.
    sam_metric : str, default "mean"
        Aggregation metric for SAM values ("mean", "median", etc.).
    debug : bool, default False
        Verbose debug logging.

    Built Attributes (populated by :meth:`build`)
    ---------------------------------------------
    tag_str : str
        Unique tag for this configuration (used in directory names).
    outdir : str
        Path to the run-specific output directory.
    node_distances_km : dict[str, np.ndarray] | None
        Distances station→node (km) for each station.
    station_coords : dict[str, dict] | None
        Station coordinate metadata used for distances.
    dist_meta : dict | None
        Metadata describing the distance computation.
    ampcorr : AmpCorr | None
        Amplitude corrections for each station/node.

    Notes
    -----
    * Call :meth:`build` before running ASL with this config.
    * Caches are stored under ``output_base / tag_str``.
    """

    # --- required inputs for building ---
    inventory: Inventory | str | Path
    output_base: str | Path
    gridobj: Grid
    global_cache: str | Path

    # --- physical/config knobs ---
    wave_kind: str = "surface"  # 'surface' | 'body'
    station_correction_dataframe: pd.DataFrame | str | Path | None = None
    speed: float = 1.5  # km/s
    Q: int = 100
    dist_mode: str = "3d"  # '2d' | '3d'
    misfit_engine: str = "l2"
    peakf: float = 8.0  # Hz
    window_seconds: float = 5.0
    min_stations: int = 5
    sam_class: type = VSAM  # class, not instance
    sam_metric: str = "mean"
    debug: bool = False

    # --- built artifacts (filled by build()) ---
    tag_str: str = field(init=False, default="")
    outdir: str = field(init=False, default="")
    node_distances_km: Dict[str, np.ndarray] | None = field(init=False, default=None)
    station_coords: Dict[str, dict] | None = field(init=False, default=None)
    dist_meta: Dict[str, Any] | None = field(init=False, default=None)
    ampcorr: AmpCorr | None = field(init=False, default=None)
    # not a field: station_corr_df_built : Optional[pd.DataFrame] (set via object.__setattr__)

    # ---------------- helpers ----------------

    def has_station_corr(self) -> bool:
        """True if a station corrections table was provided (DataFrame or path)."""
        return self.station_correction_dataframe is not None

    def _resolve_station_corr_df(self) -> Optional[pd.DataFrame]:
        """Load/return the corrections table regardless of whether it's a DF or a path."""
        sc = self.station_correction_dataframe
        if sc is None:
            return None
        if isinstance(sc, pd.DataFrame):
            return sc
        p = Path(sc)
        suf = p.suffix.lower()
        if suf in {".csv"}:
            return pd.read_csv(p)
        if suf in {".tsv", ".tab"}:
            return pd.read_csv(p, sep="\t")
        if suf in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        # Fallback: let pandas guess
        return pd.read_csv(p)

    # ---------------- labeling ----------------

    def tag(self) -> str:
        """
        Construct a unique string identifier for this ASL configuration.

        The tag encodes key parameters (wave kind, velocity, Q, frequency,
        distance mode, misfit engine, etc.) into a compact string suitable
        for directory names and filenames.

        Returns
        -------
        str
            A string like ``"VSAM_mean_5s_surface_v2.5_Q200_F6_2d_l2_SC"``.
            The suffix "SC" is added if station corrections are enabled.
        """
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
        if self.has_station_corr():
            parts.append("SC")
        return "_".join(parts)

    # ---------------- builder ----------------

    def build(self) -> "ASLConfig":
        """
        Populate caches and derived artifacts required for an ASL run.

        Steps:
          1) Create tagged output directory.
          2) Ensure Inventory is loaded.
          3) Compute/load station→node distances (2D/3D).
          4) Compute/load amplitude corrections (AmpCorr).
          5) Resolve and stash station-corrections DataFrame (if any).

        Returns
        -------
        ASLConfig
            This instance with built attributes populated.

        Raises
        ------
        AssertionError
            If `gridobj`, `output_base`, or `inventory_xml` are not set.
        RuntimeError
            If distance or amplitude correction calculations fail.
        """
        # 1) tag + outdir
        tag = self.tag()
        object.__setattr__(self, "tag_str", tag)
        outdir = Path(self.output_base) / tag
        #outdir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "outdir", str(outdir))

        # 2) inventory load
        inv: Inventory
        if isinstance(self.inventory, Inventory):
            inv = self.inventory
        else:
            inv = read_inventory(str(self.inventory))
        if self.debug:
            print("[ASLConfig.build] Inventory loaded:", inv)
        object.__setattr__(self, "inventory", inv)  # pin the concrete Inventory

        # cache roots
        cache_root = Path(self.global_cache) / tag
        cache_root.mkdir(parents=True, exist_ok=True)

        # 3) distances
        use_elev = (self.dist_mode.lower() == "3d")
        dist_cache_dir = cache_root / "distances" / ("3d" if use_elev else "2d")
        dist_cache_dir.mkdir(parents=True, exist_ok=True)

        print("[ASLConfig.build] Computing/loading station→node distances …")
        node_distances_km, station_coords, dist_meta = compute_or_load_distances(
            self.gridobj,
            inventory=inv,
            stream=None,
            cache_dir=str(dist_cache_dir),
            force_recompute=False,
            use_elevation=use_elev,
        )
        object.__setattr__(self, "node_distances_km", node_distances_km)
        object.__setattr__(self, "station_coords", station_coords)
        object.__setattr__(self, "dist_meta", dist_meta)

        src = dist_meta.get("source")
        if self.debug:
            print(f"[ASLConfig.build] Distances: {src or 'unknown'} (mask={dist_meta.get('mask_signature')})")
            print(
                "[ASLConfig.build]",
                summarize_station_node_distances(node_distances_km, reduce_to_station=True),
            )

        # 4) amplitude corrections
        ampcorr_cache = cache_root / "ampcorr"
        ampcorr_cache.mkdir(parents=True, exist_ok=True)

        mask_idx = getattr(self.gridobj, "_node_mask_idx", None)
        total_nodes = int(self.gridobj.gridlat.size)
        mask_sig = (int(mask_idx.size), total_nodes) if mask_idx is not None else None

        params = AmpCorrParams(
            assume_surface_waves=(self.wave_kind == "surface"),
            wave_speed_kms=float(self.speed),
            Q=int(self.Q),
            peakf=float(self.peakf),
            grid_sig=self.gridobj.signature(),
            inv_sig=tuple(sorted(node_distances_km.keys())),
            dist_sig=distances_signature(node_distances_km),
            mask_sig=mask_sig,
            code_version="v1",
        )
        ampcorr = AmpCorr(params, cache_dir=str(ampcorr_cache))
        print("[ASLConfig.build] Computing/loading amplitude corrections …")
        ampcorr.compute_or_load(node_distances_km, inventory=inv)
        ampcorr.validate_against_nodes(total_nodes)
        object.__setattr__(self, "ampcorr", ampcorr)

        if self.debug:
            print(
                "[ASLConfig.build]",
                summarize_ampcorr_ranges(
                    ampcorr.corrections,
                    total_nodes=self.gridobj.gridlat.size,
                    reduce_to_station=True,
                    include_percentiles=True,
                    sort_by="min_corr",
                ),
            )

        # 5) resolve/stash station corrections DF (optional)
        try:
            corr_df = self._resolve_station_corr_df()
        except Exception as e:
            raise RuntimeError(f"Failed to load station corrections table: {e}") from e
        object.__setattr__(self, "station_corr_df_built", corr_df)

        return self

    @property
    def built(self) -> bool:
        """True iff this config has been built (tag_str populated)."""
        return bool(getattr(self, "tag_str", ""))

    def _changes_dict(self, overrides: dict) -> dict:
        """Return only the keys that actually change value, using robust equality."""
        out = {}
        for k, v in (overrides or {}).items():
            if not _value_equal(getattr(self, k), v):
                out[k] = v
        return out

    def needs_rebuild(self, *, changes: dict | None = None) -> dict:
        """
        Decide what to recompute given a set of changes.

        Returns a dict:
        {
            "requires_distances": bool,   # dist_mode change (2d ↔ 3d) or structural changes
            "requires_ampcorr":   bool,   # wave_kind/speed/Q/peakf change (or above)
            "requires_any":       bool,   # either of the above OR not built yet OR station corr changed
        }
        """
        changes = self._changes_dict(changes or {})

        structural = {"inventory", "gridobj", "global_cache"}
        structural_changed = any(k in changes for k in structural)

        requires_distances = structural_changed or ("dist_mode" in changes)

        physics = {"wave_kind", "speed", "Q", "peakf"}
        physics_changed = any(k in changes for k in physics)
        requires_ampcorr = structural_changed or physics_changed or requires_distances

        # If the station corrections table/path changed, we want to run build() to
        # resolve & stash the new DataFrame even though it doesn't affect distances/ampcorr.
        stacorr_changed = "station_correction_dataframe" in changes

        requires_any = (not self.built) or requires_distances or requires_ampcorr or stacorr_changed

        return {
            "requires_distances": requires_distances or (not self.built),
            "requires_ampcorr":   requires_ampcorr   or (not self.built),
            "requires_any":       requires_any,
        }

    # --- copying APIs ---------------------------------------------------
    def copy(self, **overrides) -> "ASLConfig":
        """
        Make a new config with one or more parameters replaced. If the new
        config needs derived artifacts (distances/ampcorr), this will call
        .build() automatically; otherwise it returns a cheap, unbuilt copy.

        Example:
            cfg2 = cfg.copy(Q=200, speed=3.0)        # auto-builds (AmpCorr depends)
            cfg3 = cfg.copy(dist_mode="2d")          # auto-builds (distances depend)
            cfg4 = cfg.copy(sam_metric="median")     # returns as-is (no rebuild needed)
        """
        from dataclasses import replace

        overrides = dict(overrides or {})
        new_cfg = replace(self, **overrides)

        need = self.needs_rebuild(changes=overrides)
        return new_cfg.build() if need["requires_any"] else new_cfg

    def copy_unbuilt(self, **overrides) -> "ASLConfig":
        """
        Make a new config with overrides but **do not build** it.
        Useful if you want to stage many variants and build later.
        """
        from dataclasses import replace
        return replace(self, **(overrides or {}))

    # ---------------- sweep helper ----------------

    @staticmethod
    def generate_config_list(
        *,
        inventory: Inventory | str | Path,
        output_base: str | Path,
        gridobj: Grid,
        global_cache: str | Path,
        wave_kinds: Sequence[str] = ("surface", "body"),
        station_corr_tables: Sequence[pd.DataFrame | str | Path | None] = (None,),
        speeds: Sequence[float] = (1.5, 3.2),
        Qs: Sequence[int] = (50, 200),
        dist_modes: Sequence[str] = ("2d", "3d"),
        misfit_engines: Sequence[str] = ("l2", "r2", "lin"),
        peakfs: Sequence[float] = (4.0, 8.0),
        window_seconds: float = 5.0,
        min_stations: int = 5,
        sam_class: type = VSAM,
        sam_metric: str = "mean",
        debug: bool = False,
    ) -> List["ASLConfig"]:
        """
        Generate a list of different ASLConfig objects with different parameter settings

        Produces one ASLConfig instance for every combination of the
        provided parameter sequences (Cartesian product). Useful for
        Monte Carlo or grid search studies.

        Parameters
        ----------
        wave_kinds : sequence of str, optional
            Wave kinds to include ("surface", "body").
        station_corr_opts : sequence of bool, optional
            Whether to apply station corrections.
        speeds : sequence of float, optional
            Wave speeds in km/s.
        Qs : sequence of int, optional
            Attenuation Q values.
        dist_modes : sequence of str, optional
            Distance calculation modes ("2d", "3d").
        misfit_engines : sequence of str, optional
            Misfit engines to test ("l2", "huber", etc.).
        peakfs : sequence of float, optional
            Peak frequencies to test (Hz).
        window_seconds : float, optional
            Time window length in seconds.
        min_stations : int, optional
            Minimum number of stations required.
        sam_class : type, optional
            SAM class to use (default: VSAM).
        sam_metric : str, optional
            Metric for SAM aggregation (default: "mean").
        inventory_xml : Inventory or str or Path or None, optional
            StationXML inventory or path.
        output_base : str or Path or None, optional
            Output base directory.
        gridobj : Any or None, optional
            Node grid object.
        global_cache : str or Path or None, optional
            Global cache root for distances/ampcorr.
        debug : bool, optional
            If True, print verbose diagnostic output.

        Returns
        -------
        list of ASLConfig
            One configuration per combination of parameter values.
        """
        # Create a baseline “template” (unbuilt) and then use axes
        base = ASLConfig(
            inventory=inventory, output_base=output_base,
            gridobj=gridobj, global_cache=global_cache,
            wave_kind=wave_kinds[0],  # seed; will be overridden
            station_correction_dataframe=station_corr_tables[0],
            speed=speeds[0], Q=Qs[0], dist_mode=dist_modes[0],
            misfit_engine=misfit_engines[0], peakf=peakfs[0],
            window_seconds=window_seconds, min_stations=min_stations,
            sam_class=sam_class, sam_metric=sam_metric, debug=debug,
        )
        axes = {
            "wave_kind": wave_kinds,
            "station_correction_dataframe": station_corr_tables,
            "speed": speeds,
            "Q": Qs,
            "dist_mode": dist_modes,
            "misfit_engine": misfit_engines,
            "peakf": peakfs,
        }
        # reuse your variants helper:
        from flovopy.asl.compare_runs import tweak_config  # or cfg_variants_from
        return list(tweak_config(base, axes=axes, include_baseline=False).values())