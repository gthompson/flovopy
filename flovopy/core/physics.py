# flovopy/core/physics.py
from __future__ import annotations

import numpy as np
from math import pi
from typing import Iterable, Tuple, Union, Optional

from obspy.core.trace import Trace
from obspy.geodetics.base import gps2dist_azimuth

Number = Union[float, int, np.floating, np.integer]

# ---------------------------------------------------------------------------
# Basic ratios / conversions
# ---------------------------------------------------------------------------

def VASR(Eacoustic: Number, Eseismic: Number) -> Optional[float]:
    """
    Volcanic Acoustic-Seismic Ratio after Johnson & Aster (2005).
    Returns None if Eseismic is zero/invalid.
    """
    try:
        Ea = float(Eacoustic)
        Es = float(Eseismic)
        return Ea / Es if np.isfinite(Ea) and np.isfinite(Es) and Es != 0 else None
    except Exception:
        return None


def Eseismic2magnitude(Eseismic: Union[Number, Iterable[Number]], correction: float = 3.7):
    """
    Energy (J) -> magnitude using Hanks & Kanamori (1979) with joule correction (3.7).
    Accepts scalar or iterable. Returns scalar or list.
    """
    def _one(E):
        E = float(E)
        return np.log10(E) / 1.5 - correction if (E > 0 and np.isfinite(E)) else np.nan
    if isinstance(Eseismic, (list, tuple, np.ndarray)):
        return [_one(E) for E in Eseismic]
    return _one(Eseismic)


def magnitude2Eseismic(mag: Union[Number, Iterable[Number]], correction: float = 3.7):
    """
    Magnitude -> energy (J) using Hanks & Kanamori (1979) with joule correction (3.7).
    Accepts scalar or iterable. Returns scalar or list.
    """
    def _one(M):
        M = float(M)
        return float(np.power(10.0, 1.5 * M + correction)) if np.isfinite(M) else np.nan
    if isinstance(mag, (list, tuple, np.ndarray)):
        return [_one(M) for M in mag]
    return _one(mag)


def Mlrichter(peakA: Number, R_km: Number, a: float = 1.6, b: float = -0.15, g: float = 0) -> Optional[float]:
    """
    Local magnitude ML = log10(peakA) + a*log10(R_km) + b + g.
    Returns None if inputs are non-finite or R_km <= 0 or peakA <= 0.
    """
    try:
        A = float(peakA)
        Rk = float(R_km)
        if not (np.isfinite(A) and np.isfinite(Rk)) or A <= 0 or Rk <= 0:
            return None
        return np.log10(A) + a * np.log10(Rk) + b + g
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Distance & attenuation helpers
# ---------------------------------------------------------------------------

def estimate_distance(trace: Trace, source_coords: dict) -> Optional[float]:
    """
    Hypocentral distance R (m) from trace.stats.coordinates to source (lat, lon, depth[m]).
    Requires trace.stats.coordinates = {'latitude','longitude','elevation'}.
    Returns None if coordinates missing.
    """
    try:
        sta = getattr(trace.stats, "coordinates", None)
        if not sta:
            return None
        lat1 = float(sta.get("latitude"))
        lon1 = float(sta.get("longitude"))
        elev = float(sta.get("elevation", 0.0))

        lat2 = float(source_coords["latitude"])
        lon2 = float(source_coords["longitude"])
        depth = float(source_coords.get("depth", 0.0))

        epic_dist, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
        dz = depth + elev  # depth positive downward, add site elevation
        R = np.sqrt(epic_dist ** 2 + dz ** 2)
        return float(R)
    except Exception:
        return None


def pick_f_peak(trace: Trace) -> Optional[float]:
    """
    Try to obtain a representative peak frequency for attenuation:
      - stats.spectrum['peakF'] if present
      - else argmax of stats.spectral['amplitudes'] over ['freqs']
    """
    try:
        s = getattr(trace.stats, "spectrum", None)
        if s and "peakF" in s and np.isfinite(s["peakF"]):
            return float(s["peakF"])
    except Exception:
        pass

    try:
        sp = getattr(trace.stats, "spectral", None)
        if sp and "freqs" in sp and "amplitudes" in sp:
            f = np.asarray(sp["freqs"], dtype=float)
            A = np.asarray(sp["amplitudes"], dtype=float)
            if f.size and A.size and np.any(np.isfinite(A)):
                i = int(np.nanargmax(A))
                return float(f[i])
    except Exception:
        pass

    return None


def attenuation(trace: Trace, R: Number, Q: float = 50.0, c_earth: float = 2500.0,
                f_peak: Optional[float] = None) -> Optional[float]:
    """
    Path attenuation factor A_att = exp(-pi * f_peak * R / (Q * c_earth)).
    If f_peak not supplied, attempts to infer via pick_f_peak(trace).
    Returns None if inputs invalid.
    """
    try:
        R = float(R)
        if not np.isfinite(R) or R < 0:
            return None
        if f_peak is None:
            f_peak = pick_f_peak(trace)
        if f_peak is None or f_peak <= 0 or not np.isfinite(f_peak):
            return None
        return float(np.exp(-np.pi * f_peak * R / (Q * c_earth)))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Source energy estimators
# ---------------------------------------------------------------------------

def _geom_spreading(model: str, R: Number, f_peak: float, c_earth: float) -> Optional[float]:
    """
    Geometric spreading undo factor for energy back-projection.
    - 'body': proportional to R^2 (hemispherical)
    - 'surface': proportional to R * wavelength (2D cylindrical), wavelength = c/f
    Returns None if invalid.
    """
    try:
        R = float(R)
        if not np.isfinite(R) or R <= 0:
            return None
        if model == "body":
            return R ** 2
        if model == "surface":
            wavelength = c_earth / f_peak if f_peak > 0 else None
            return R * wavelength if wavelength and np.isfinite(wavelength) else None
        return None
    except Exception:
        return None


def estimate_source_energy(trace: Trace, R: Number, *, model: str = "body",
                           Q: float = 50.0, c_earth: float = 2500.0) -> Optional[float]:
    """
    Estimate source energy E0 (J) from observed station energy:
        E0 = E_obs * geom_spreading / attenuation
    Uses time-domain energy in trace.stats.metrics['energy'] and a representative f_peak.
    Returns None on missing data.
    """
    try:
        E_obs = getattr(trace.stats, "metrics", {}).get("energy")
        if E_obs is None or not np.isfinite(E_obs):
            return None

        fpk = pick_f_peak(trace)
        if fpk is None:
            return None

        A_att = np.exp(-np.pi * fpk * float(R) / (Q * c_earth))
        geom = _geom_spreading(model, R, fpk, c_earth)
        if geom is None or A_att <= 0:
            return None

        return float(E_obs * geom / A_att)
    except Exception:
        return None


# --- Boatwright-style back-projection (Johnson & Aster, 2005) ----------------

def Eseismic_Boatwright(val: Union[Trace, Number], R: Number,
                        rho_earth: float = 2000.0, c_earth: float = 2500.0,
                        S: float = 1.0, A: float = 1.0) -> Optional[float]:
    """
    Boatwright seismic energy back-projection:
        E0 = 2 * pi * R^2 * rho * c * S^2 * E_station / A
    - val: Trace (reads .stats.metrics.energy) or station energy (float, J)
    - R (m), rho (kg/m^3), c (m/s)
    Returns None if inputs invalid.
    """
    try:
        if isinstance(val, Trace):
            E_station = getattr(val.stats, "metrics", {}).get("energy")
        else:
            E_station = float(val)
        R = float(R)
        if not (np.isfinite(E_station) and np.isfinite(R)) or R <= 0:
            return None
        return float(2.0 * pi * (R ** 2) * rho_earth * c_earth * (S ** 2) * E_station / A)
    except Exception:
        return None


def Eacoustic_Boatwright(val: Union[Trace, Number], R: Number,
                         rho_atmos: float = 1.2, c_atmos: float = 340.0,
                         z: float = 100000.0) -> Optional[float]:
    """
    Boatwright acoustic energy back-projection (with far-field heuristic):
      - Near field (R <= 100 km):  E0 = 2 * pi * R^2 / (rho * c) * E_station
      - Very far field (R > 100 km): scale by z to limit divergence
    Returns None if inputs invalid.
    """
    try:
        if isinstance(val, Trace):
            E_station = getattr(val.stats, "metrics", {}).get("energy")
        else:
            E_station = float(val)
        R = float(R)
        if not (np.isfinite(E_station) and np.isfinite(R)) or R <= 0:
            return None

        if R > 100_000.0:
            E_if_at_z = 2.0 * pi * (z ** 2) / (rho_atmos * c_atmos) * E_station
            return float(E_if_at_z * (R / 1e5))
        else:
            return float(2.0 * pi * (R ** 2) / (rho_atmos * c_atmos) * E_station)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Magnitude wrappers
# ---------------------------------------------------------------------------

def estimate_local_magnitude(trace: Trace, R_km: Number,
                             a: float = 1.6, b: float = -0.15, g: float = 0) -> Optional[float]:
    """
    Convenience wrapper: reads `trace.stats.metrics['peakamp']` and computes ML.
    Stores result in `trace.stats.metrics['local_magnitude']` if successful.
    """
    try:
        peakamp = getattr(trace.stats, "metrics", {}).get("peakamp")
        ml = Mlrichter(peakamp, R_km, a=a, b=b, g=g) if peakamp is not None else None
        if ml is not None:
            trace.stats.metrics["local_magnitude"] = ml
        return ml
    except Exception:
        return None