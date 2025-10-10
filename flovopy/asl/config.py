from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional, Iterable
import numpy as np
import pandas as pd

from obspy.core.inventory import Inventory, read_inventory

from flovopy.asl.distances import compute_or_load_distances, distances_signature
from flovopy.asl.grid import summarize_station_node_distances, Grid
from flovopy.asl.ampcorr import AmpCorrParams, AmpCorr, summarize_ampcorr_ranges
from flovopy.processing.sam import VSAM  # default


__all__ = [
    "ASLConfig",
    "tweak_config",
]


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))

def _value_equal(a: Any, b: Any) -> bool:
    """Robust equality for config fields (paths, arrays, pandas, etc.)."""
    if a is b:
        return True
    if (a is None) ^ (b is None):
        return False

    if _is_pathlike(a) and _is_pathlike(b):
        try:
            return Path(a).resolve() == Path(b).resolve()
        except Exception:
            return str(a) == str(b)

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

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b, equal_nan=True)
        except Exception:
            return False

    try:
        return a == b
    except Exception:
        return False


# ---------------------------------------------------------------------
# ASLConfig
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ASLConfig:
    """
    Strongly-typed configuration for Amplitude Source Location (ASL) runs.
    Call .build() before use to materialize caches (distances, ampcorr) and outdir.
    """

    # Required
    inventory: Inventory | str | Path
    output_base: str | Path
    gridobj: Grid
    global_cache: str | Path

    # Physical/config knobs
    wave_kind: str = "surface"  # "surface" | "body"
    station_correction_dataframe: pd.DataFrame | str | Path | None = None
    speed: float = 1.5  # km/s
    Q: int = 100
    dist_mode: str = "3d"       # "2d" | "3d"
    misfit_engine: str = "l2"
    peakf: float = 8.0          # Hz
    window_seconds: float = 5.0
    min_stations: int = 5
    sam_class: type = VSAM       # class, not instance
    sam_metric: str = "mean"
    debug: bool = False

    # Built artifacts (populated by .build())
    tag_str: str = field(init=False, default="")
    outdir: str = field(init=False, default="")
    node_distances_km: Dict[str, np.ndarray] | None = field(init=False, default=None)
    station_coords: Dict[str, dict] | None = field(init=False, default=None)
    dist_meta: Dict[str, Any] | None = field(init=False, default=None)
    ampcorr: AmpCorr | None = field(init=False, default=None)
    # station_corr_df_built is set via object.__setattr__ in build()

    # ---------------- helpers ----------------

    def has_station_corr(self) -> bool:
        return self.station_correction_dataframe is not None

    def _resolve_station_corr_df(self) -> Optional[pd.DataFrame]:
        sc = self.station_correction_dataframe
        if sc is None:
            return None
        if isinstance(sc, pd.DataFrame):
            return sc
        p = Path(sc)
        suf = p.suffix.lower()
        if suf == ".csv":
            return pd.read_csv(p)
        if suf in {".tsv", ".tab"}:
            return pd.read_csv(p, sep="\t")
        if suf in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        # Fallback: let pandas guess
        return pd.read_csv(p)

    # ---------------- labeling ----------------

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
        if self.has_station_corr():
            parts.append("SC")
        return "_".join(parts)

    # ---------------- builder ----------------

    def build(self) -> "ASLConfig":
        tag = self.tag()
        object.__setattr__(self, "tag_str", tag)
        outdir = Path(self.output_base) / tag
        object.__setattr__(self, "outdir", str(outdir))

        # Inventory (allow path or Inventory)
        if isinstance(self.inventory, Inventory):
            inv = self.inventory
        else:
            inv = read_inventory(str(self.inventory))
        if self.debug:
            print("[ASLConfig.build] Inventory loaded:", inv)
        object.__setattr__(self, "inventory", inv)

        # Cache root
        cache_root = Path(self.global_cache) / tag
        cache_root.mkdir(parents=True, exist_ok=True)

        # Distances
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

        if self.debug:
            src = dist_meta.get("source")
            print(f"[ASLConfig.build] Distances: {src or 'unknown'} "
                  f"(mask={dist_meta.get('mask_signature')})")
            print("[ASLConfig.build]",
                  summarize_station_node_distances(node_distances_km, reduce_to_station=True))

        # AmpCorr
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
            print("[ASLConfig.build]",
                  summarize_ampcorr_ranges(
                      ampcorr.corrections,
                      total_nodes=self.gridobj.gridlat.size,
                      reduce_to_station=True,
                      include_percentiles=True,
                      sort_by="min_corr",
                  ))

        # Station corrections DF (optional)
        try:
            corr_df = self._resolve_station_corr_df()
        except Exception as e:
            raise RuntimeError(f"Failed to load station corrections table: {e}") from e
        object.__setattr__(self, "station_corr_df_built", corr_df)

        return self

    # ---------------- status / change detection ----------------

    @property
    def built(self) -> bool:
        return bool(getattr(self, "tag_str", ""))

    def _changes_dict(self, overrides: dict) -> dict:
        """Return only keys whose value would actually change."""
        out: dict = {}
        for k, v in (overrides or {}).items():
            if not _value_equal(getattr(self, k), v):
                out[k] = v
        return out

    def needs_rebuild(self, *, changes: dict | None = None) -> dict:
        """
        Decide which derived artifacts must be recomputed for the given changes.

        Returns
        -------
        dict with booleans:
            requires_distances, requires_ampcorr, requires_any
        """
        changes = self._changes_dict(changes or {})

        structural = {"inventory", "gridobj", "global_cache"}
        structural_changed = any(k in changes for k in structural)

        requires_distances = structural_changed or ("dist_mode" in changes)

        physics = {"wave_kind", "speed", "Q", "peakf"}
        physics_changed = any(k in changes for k in physics)
        requires_ampcorr = structural_changed or physics_changed or requires_distances

        stacorr_changed = "station_correction_dataframe" in changes

        requires_any = (not self.built) or requires_distances or requires_ampcorr or stacorr_changed

        return {
            "requires_distances": requires_distances or (not self.built),
            "requires_ampcorr":   requires_ampcorr   or (not self.built),
            "requires_any":       requires_any,
        }

    # ---------------- copying APIs ----------------

    def copy(self, **overrides) -> "ASLConfig":
        """
        Make a new config with overrides. Auto-build if derived artifacts are needed.
        """
        new_cfg = replace(self, **(overrides or {}))
        need = self.needs_rebuild(changes=overrides or {})
        return new_cfg.build() if need["requires_any"] else new_cfg

    def copy_unbuilt(self, **overrides) -> "ASLConfig":
        """Make a new config with overrides but DO NOT build it."""
        return replace(self, **(overrides or {}))


# ---------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------

def tweak_config(
    baseline: ASLConfig,
    *,
    changes: Optional[Iterable[dict]] = None,
    axes: Optional[Dict[str, Iterable]] = None,
    landgridobj=None,                           # backward-compat convenience
    annual_station_corrections_df=None,         # backward-compat convenience
    include_baseline: bool = False,
    dedupe: bool = True,
) -> Dict[str, ASLConfig]:
    """
    Build a dict of ASLConfig variants derived from `baseline`.

    - `changes`: iterable of override dicts, e.g. [{'Q':100}, {'speed':3.0}]
    - `axes`: Cartesian sweep dict, e.g. {'speed':[1.0,3.0], 'Q':[10,100]}
    - Keys of the returned dict are each config's `tag()` string.
    """
    from itertools import product as _product

    specs: List[dict] = []
    if include_baseline:
        specs.append({})

    if changes:
        specs.extend(dict(c) for c in changes)

    if axes:
        keys = list(axes.keys())
        vals = [list(axes[k]) for k in keys]
        for tup in _product(*vals):
            specs.append({k: v for k, v in zip(keys, tup)})

    # Back-compat conveniences
    if landgridobj is not None:
        specs.append({"gridobj": landgridobj})
    if annual_station_corrections_df is not None:
        specs.append({"station_correction_dataframe": annual_station_corrections_df})

    if not specs:
        return {}

    out: Dict[str, ASLConfig] = {}
    for spec in specs:
        cfg = baseline.copy(**spec)      # copy() auto-builds if needed
        key = cfg.tag()                  # canonical label
        if key in out and dedupe:
            # silently keep the last one
            pass
        out[key] = cfg

    return out
