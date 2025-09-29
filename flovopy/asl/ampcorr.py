# flovopy/asl/ampcorr.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Callable, Optional, Tuple, Iterable, Sequence

import os
import pickle
import numpy as np
from obspy.core.inventory import Inventory
from flovopy.processing.sam import VSAM
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
    geom_spread_fn: Optional[Callable[[np.ndarray, str, bool, float, float], np.ndarray]] = None
    inelastic_att_fn: Optional[Callable[[np.ndarray, float, float, Optional[float]], np.ndarray]] = None

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
        # Late import backends if not injected
        if self.geom_spread_fn is None or self.inelastic_att_fn is None:
            self.geom_spread_fn = self.geom_spread_fn or VSAM.compute_geometrical_spreading_correction
            self.inelastic_att_fn = self.inelastic_att_fn or VSAM.compute_inelastic_attenuation_correction

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

        out: Dict[str, np.ndarray] = {}
        missing: list[str] = []

        for sid in ids:
            d = node_distances_km.get(sid)
            if d is None:
                missing.append(sid)
                if require_all:
                    raise KeyError(f"Missing distances for {sid}")
                continue

            d = np.asarray(d, dtype=float)
            if d.size != n_nodes:
                raise ValueError(f"Distance vector length mismatch for {sid}: {d.size} != {n_nodes}")

            # Channel code suffix (e.g., SHZ/BHZ)
            chan3 = sid.split(".")[-1][-3:]

            # Compute multiplicative factors
            g = self.geom_spread_fn(
                d, chan3, self.params.assume_surface_waves, self.params.wave_speed_kms, self.params.peakf
            )
            a = self.inelastic_att_fn(
                d, self.params.peakf, self.params.wave_speed_kms, self.params.Q
            )

            # Numerical guards & dtype
            g = np.asarray(g, dtype=float)
            a = np.asarray(a, dtype=float)
            corr = (g * a).astype(dtype, copy=False)

            # Replace non-finite with zeros to avoid NaNs propagating into misfit ratios
            if not np.all(np.isfinite(corr)):
                corr = np.where(np.isfinite(corr), corr, 0.0).astype(dtype, copy=False)

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
        return np.array(
            [y_vec[k] * self.corrections[seed_ids[k]][node_index] for k in range(len(seed_ids))],
            dtype=float,
        )

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

