# flovopy/asl/ampcorr.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Callable, Optional, Tuple, Iterable, Sequence

import os
import pickle
import numpy as np
from obspy.core.inventory import Inventory
from flovopy.core.physics import total_amp_correction
from flovopy.utils.make_hash import make_hash
#from flovopy.asl.distances import distances_signature


# ----------------------------------------------------------------------
# Parameterization & cache key
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AmpCorrParams:
    assume_surface_waves: bool
    wave_speed_kms: float
    Q: Optional[float]
    peakf: float
    grid_sig: Tuple                    # (nlat, nlon, minlat, maxlat, minlon, maxlon, spacing_m)
    inv_sig: Tuple                     # sorted seed_ids tuple
    dist_sig: Tuple                    # sorted (seed_id, n_nodes) tuples
    mask_sig: Optional[Tuple[int, int]] = None  # (valid_count, total) or None
    code_version: str = "v1"           # bump if formulas change

    def key(self) -> str:
        return make_hash(
            self.code_version,
            self.assume_surface_waves,
            self.wave_speed_kms,
            self.Q,
            self.peakf,
            self.grid_sig,
            self.inv_sig,
            self.dist_sig,
            self.mask_sig,
        )


# ----------------------------------------------------------------------
# Amplitude corrections container
# ----------------------------------------------------------------------
@dataclass
class AmpCorr:
    params: AmpCorrParams
    corrections: Dict[str, np.ndarray] = field(default_factory=dict)
    cache_dir: str = "asl_cache"

    # Optional backends (dependency injection)
    #geom_spread_fn: Optional[Callable[[np.ndarray, str, bool, float, float], np.ndarray]] = None
    #inelastic_att_fn: Optional[Callable[[np.ndarray, float, float, Optional[float]], np.ndarray]] = None

    # ------------- Cache -------------
    def cache_path(self) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"ampcorr_{self.params.key()}.pkl")

    def load(self) -> bool:
        path = self.cache_path()
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        # bundle may be just a dict (older) or a structured bundle (newer)
        if isinstance(bundle, dict) and "corrections" in bundle:
            self.corrections = bundle["corrections"]
        else:
            self.corrections = bundle  # backwards compatibility
        return True

    def save(self) -> None:
        path = self.cache_path()
        bundle = {
            "corrections": self.corrections,
            "meta": self.meta(),
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    # ------------- Compute -------------
    def compute(
        self,
        node_distances_km: Dict[str, np.ndarray],
        *,
        inventory: Optional[Inventory] = None,
        seed_ids: Optional[Sequence[str]] = None,
        require_all: bool = False,
        dtype: str = "float32",
    ) -> None:
        """
        Build amplitude corrections per channel.

        Parameters
        ----------
        node_distances_km : dict[seed_id -> np.ndarray]
            Distances (km) from each node to the channel. All vectors must have
            the same length (n_nodes).
        inventory : Inventory, optional
            Used to enumerate channels if seed_ids is not provided. If neither
            is given, we iterate keys(node_distances_km).
        seed_ids : list/tuple of seed IDs, optional
            Exact channels to compute. Overrides inventory.
        require_all : bool
            If True, raise if any seed_id has no distances.
        dtype : {"float32","float64"}
            dtype for saved corrections (float32 reduces cache size).
        """

        # Decide channel iteration list
        if seed_ids is not None:
            ids = list(seed_ids)
        elif inventory is not None:
            ids = [
                f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                for net in inventory for sta in net for cha in sta
            ]
        else:
            ids = list(node_distances_km.keys())

        # Infer n_nodes from any available distances
        _first_vec = next((np.asarray(v) for v in node_distances_km.values()), None)
        if _first_vec is None:
            self.corrections = {}
            return
        n_nodes = int(np.asarray(_first_vec).size)

        # Precompute stable params and dtype
        assume_sw = bool(self.params.assume_surface_waves)
        v_kms     = float(self.params.wave_speed_kms)
        Q         = None if self.params.Q is None else float(self.params.Q)
        peakf     = float(self.params.peakf)
        out_dtype = np.float32 if dtype == "float32" else np.float64

        out: Dict[str, np.ndarray] = {}
        missing: list[str] = []

        for sid in ids:
            d = node_distances_km.get(sid)
            if d is None:
                missing.append(sid)
                if require_all:
                    raise KeyError(f"Missing distances for {sid}")
                continue

            d = np.asarray(d, dtype=out_dtype, order="C")
            if d.size != n_nodes:
                raise ValueError(f"Distance vector length mismatch for {sid}: {d.size} != {n_nodes}")

            # Fast skip: all masked
            if not np.isfinite(d).any():
                out[sid] = np.full_like(d, np.nan, dtype=out_dtype)
                continue

            chan3 = sid.split(".")[-1][-3:]

            # Fused (vectorized) correction: single pass over d
            # If you defined it as a VSAM method:
            corr = total_amp_correction(
                d,
                chan=chan3,
                surface_waves=assume_sw,
                wavespeed_kms=v_kms,
                peakf_hz=peakf,
                Q=Q,
                out_dtype=("float32" if out_dtype == np.float32 else "float64"),
            )

            # Preserve NaNs (masked nodes remain NaN), but kill +/-inf
            if not np.all(np.isfinite(corr) | np.isnan(corr)):
                corr = np.where(np.isinf(corr), 0.0, corr)

            # Ensure final dtype
            corr = np.asarray(corr, dtype=out_dtype, order="C")

            # Preserve NaNs (masked nodes remain NaN), but kill +/-inf
            if not np.all(np.isfinite(corr) | np.isnan(corr)):
                corr = np.where(np.isinf(corr), 0.0, corr)

            out[sid] = corr

        self.corrections = out

    def compute_or_load(
        self,
        node_distances_km: Dict[str, np.ndarray],
        *,
        inventory: Optional[Inventory] = None,
        seed_ids: Optional[Sequence[str]] = None,
        force_recompute: bool = False,
        require_all: bool = False,
        dtype: str = "float32",
    ) -> "AmpCorr":
        if not force_recompute and self.load():
            return self
        self.compute(
            node_distances_km,
            inventory=inventory,
            seed_ids=seed_ids,
            require_all=require_all,
            dtype=dtype,
        )
        self.save()
        return self

    # ------------- Utilities -------------
    def apply_to_vector(self, seed_id: str, y: float, node_index: int) -> float:
        """Return corrected amplitude for a single scalar y at node index."""
        return float(y * self.corrections[seed_id][node_index])

    def apply_to_station_array(self, seed_ids: Sequence[str], y_vec: np.ndarray, node_index: int) -> np.ndarray:
        """Return array of corrected amplitudes for all stations at a node index."""
        y_vec = np.asarray(y_vec, dtype=float)
        # gather corrections into an array in one shot
        corr = np.array([self.corrections[sid][node_index] for sid in seed_ids], dtype=float)
        return y_vec * corr

    def validate_against_nodes(self, expected_nodes: int) -> None:
        """Ensure each correction vector matches the expected node count."""
        if not self.corrections:
            return
        for sid, vec in self.corrections.items():
            L = int(np.asarray(vec).size)
            if L != expected_nodes:
                raise ValueError(f"{sid}: corrections length {L} != expected {expected_nodes}")

    def meta(self) -> dict:
        return asdict(self.params) | {
            "cache_path": self.cache_path(),
            "n_channels": len(self.corrections),
        }


# ----------------------------------------------------------------------
# Signature helpers (keep colocated for convenience)
# ----------------------------------------------------------------------
'''
def grid_signature(grid) -> tuple:
    return (
        getattr(grid, "nlat", grid.gridlat.shape[0]),
        getattr(grid, "nlon", grid.gridlat.shape[1]),
        float(np.nanmin(grid.gridlat)),
        float(np.nanmax(grid.gridlat)),
        float(np.nanmin(grid.gridlon)),
        float(np.nanmax(grid.gridlon)),
        float(getattr(grid, "node_spacing_m", 0.0)),
    )
'''
def inv_signature(inventory: Inventory) -> tuple:
    return tuple(sorted(
        f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
        for net in inventory for sta in net for cha in sta
    ))

import numpy as np
import pandas as pd
from typing import Dict, Optional

def summarize_ampcorr_ranges(
    corrections: Dict[str, np.ndarray],   # e.g., ampcorr.corrections
    *,
    total_nodes: Optional[int] = None,    # len of dense grid (for pct_masked); if None, infer per-row
    reduce_to_station: bool = False,      # True → aggregate NET.STA across channels
    include_percentiles: bool = True,     # add p10/p50/p90
    sort_by: str = "min_corr",            # "min_corr" | "max_corr" | "station"
    to_csv: Optional[str] = None,         # optional path to save CSV
) -> pd.DataFrame:
    """
    Build a summary table of per-station/channel amplitude-correction ranges.

    Notes
    -----
    - Ignores NaNs (masked nodes).
    - 'pct_masked' is computed vs total_nodes if provided; otherwise from each vector length.
    - If reduce_to_station=True, rows are collapsed by NET.STA with robust aggregations.
    """

    def _seed_to_station(seed: str) -> str:
        parts = seed.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else seed

    rows = []
    for sid, c in corrections.items():
        c = np.asarray(c, dtype=float).ravel()
        n_vec = c.size
        n_tot = int(total_nodes) if total_nodes is not None else n_vec

        finite = np.isfinite(c)
        n_finite = int(finite.sum())

        if n_finite == 0:
            minc = maxc = medc = p10 = p90 = np.nan
        else:
            vals = c[finite]
            minc = float(np.min(vals))
            maxc = float(np.max(vals))
            medc = float(np.median(vals))
            if include_percentiles:
                p10 = float(np.percentile(vals, 10))
                p90 = float(np.percentile(vals, 90))
            else:
                p10 = p90 = np.nan

        rows.append({
            "station": sid,
            "n_total": n_tot,
            "n_finite": n_finite,
            "pct_masked": 0.0 if n_tot == 0 else float(100.0 * (n_tot - n_finite) / n_tot),
            "min_corr": minc,
            **({"p10_corr": p10, "median_corr": medc, "p90_corr": p90} if include_percentiles else {"median_corr": medc}),
            "max_corr": maxc,
        })

    df = pd.DataFrame(rows)

    if reduce_to_station:
        key = df["station"].map(_seed_to_station)
        agg = {
            "n_total": "max",
            "n_finite": "sum",
            "pct_masked": "mean",       # indicative only if channels differ
            "min_corr": "min",
            "max_corr": "max",
            "median_corr": "median",
        }
        if include_percentiles:
            agg["p10_corr"] = "median"  # robust across channels
            agg["p90_corr"] = "median"
        df["station"] = key
        df = df.groupby("station", as_index=False).agg(agg)

    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)
    else:
        df = df.sort_values("station").reset_index(drop=True)

    if to_csv:
        df.to_csv(to_csv, index=False)

    return df