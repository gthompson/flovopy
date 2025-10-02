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


def Eseismic2magnitude(Eseismic: Union[Number, Iterable[Number]], correction: float = 3.2):
    """
    Energy (J) -> magnitude using Hanks & Kanamori (1979) with joule correction (3.7).
    Accepts scalar or iterable. Returns scalar or list.

    
    Convert a (vector of) magnitude(s) into a (vector of) equivalent energy(/ies).
    
    Conversion is based on the the following formula from Hanks and Kanamori (1979):
 
       mag = 2/3 * log10(energy) - 4.7
 
    That is, energy (Joules) is roughly proportional to the peak amplitude to the power of 1.5.
    This obviously is based on earthquake waveforms following a characteristic shape.
    For a waveform of constant amplitude, energy would be proportional to peak amplitude
    to the power of 2.
 
    For Montserrat data, when calibrating against events in the SRU catalog, a factor of
    3.7 was preferred to 4.7.

    based on https://github.com/geoscience-community-codes/GISMO/blob/master/core/%2Bmagnitude/eng2mag.m
    
    *** Edited 2024/04/17

    Now based on 2024 SSA Presentation:

    ME = 2/3 log10(E) - 3.7

    ME = b * log10(E) + a

    So:

    E = 10^(1.5 * (ME + 3.7))

    E = 10^(1/b * (ME - a))

    *** Edited 2024/04/18

    Now based on Choy & Boatright [1995] who find best value for a=-3.2, so altering default

    """
    def _one(E):
        E = float(E)
        return np.log10(E) / 1.5 - correction if (E > 0 and np.isfinite(E)) else np.nan
    if isinstance(Eseismic, (list, tuple, np.ndarray)):
        return [_one(E) for E in Eseismic]
    return _one(Eseismic)


def magnitude2Eseismic(mag: Union[Number, Iterable[Number]], correction: float = 3.2):
    """
    Magnitude -> energy (J) using Hanks & Kanamori (1979) with joule correction (3.7).
    Accepts scalar or iterable. Returns scalar or list.

    
    Convert (a vector of) magnitude into (a vector of) equivalent energy(ies).
   
    Conversion is based on the equation 7 Hanks and Kanamori (1979):
 
       mag = 2/3 * log10(energy) - 4.7
 
    That is, energy (Joules) is roughly proportional to the peak amplitude to the power of 1.5.
    This obviously is based on earthquake waveforms following a characteristic shape.
    For a waveform of constant amplitude, energy would be proportional to peak amplitude
    to the power of 2.
 
    For Montserrat data, when calibrating against events in the SRU catalog, a factor of
    a=-3.7 was preferred to a=-4.7.
    
    based on https://github.com/geoscience-community-codes/GISMO/blob/master/core/%2Bmagnitude/mag2eng.m

    *** Edited 2024/04/17

    Now based on 2024 SSA Presentation:

    ME = 2/3 log10(E) - 3.7

    ME = b * log10(E) + a

    So:

    E = 10^(1.5 * (ME + 3.7))

    E = 10^(1/b * (ME - a))
    
    E = 3 x 10^5 x 10^(1.5 ME)
    
    *** Edited 2024/04/18

    Now based on Choy & Boatright [1995] who find best value for a=-3.2, so altering default 


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
    
#--------------------------------------------------------------------------------

# THESE PHYSICS COME FROM flovopy.processing.sam


def _as_dtype(x, out_dtype: str) -> np.ndarray:
    return np.asarray(x, dtype=np.float32 if out_dtype == "float32" else np.float64)

def _preserve_nan(mask_from: np.ndarray, arr: np.ndarray) -> np.ndarray:
    # preserve NaNs from mask_from into arr
    if np.isnan(mask_from).any():
        arr = np.where(np.isnan(mask_from), np.nan, arr)
    return arr

# --------------------------
# Geometrical spreading (amplitude)
# --------------------------
def geom_spread_amp(
    dist_km: np.ndarray,
    *,
    chan: str | None,
    surface_waves: bool,
    wavespeed_kms: float,
    peakf_hz: float,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Amplitude geometric-spreading factor g_amp:
      - Surface waves (on seismic channels '*H*'): g_amp = sqrt( d_km * (v/f) )
      - Otherwise:                                g_amp = d_km
    Preserves NaNs (masked nodes).
    """
    d = _as_dtype(dist_km, out_dtype)
    if surface_waves and chan and len(chan) >= 2 and chan[1] == "H" and wavespeed_kms > 0 and peakf_hz > 0:
        lam_km = wavespeed_kms / float(peakf_hz)
        out = np.sqrt(np.clip(d, 0.0, None) * lam_km, dtype=d.dtype)
    else:
        out = d.copy()
    return _preserve_nan(d, out)

# --------------------------
# Geometrical spreading (energy)
# --------------------------
def geom_spread_energy_from_amp(g_amp: np.ndarray) -> np.ndarray:
    """
    Energy geometric-spreading factor g_energy = (g_amp)^2.
    """
    g = np.asarray(g_amp)
    out = g * g
    return _preserve_nan(g, out)

def geom_spread_energy(
    dist_km: np.ndarray,
    *,
    chan: str | None,
    surface_waves: bool,
    wavespeed_kms: float,
    peakf_hz: float,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Direct energy form, derived from amplitude rule:
      - Surface: g_energy = d_km * (v/f)
      - Body:    g_energy = d_km^2  (NOTE: if you previously used g_energy = d_km,
                                     switch to square(d_km) for strict amplitude^2 physics.
                                     If you want legacy behavior, use geom_spread_energy_legacy below.)
    """
    d = _as_dtype(dist_km, out_dtype)
    if surface_waves and chan and len(chan) >= 2 and chan[1] == "H" and wavespeed_kms > 0 and peakf_hz > 0:
        lam_km = wavespeed_kms / float(peakf_hz)
        out = np.clip(d, 0.0, None) * lam_km               # (sqrt(d*lam))^2
    else:
        out = np.clip(d, 0.0, None) * np.clip(d, 0.0, None)  # (d_km)^2
    return _preserve_nan(d, out)

def geom_spread_energy_legacy(
    dist_km: np.ndarray,
    *,
    surface_waves: bool,
    wavespeed_kms: float,
    peakf_hz: float,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Legacy energy rule to mimic older behavior where energy factor ~ d_km (not d_km^2) for body waves.
    """
    d = _as_dtype(dist_km, out_dtype)
    if surface_waves and wavespeed_kms > 0 and peakf_hz > 0:
        lam_km = wavespeed_kms / float(peakf_hz)
        out = np.clip(d, 0.0, None) * lam_km
    else:
        out = np.clip(d, 0.0, None)  # legacy
    return _preserve_nan(d, out)

# --------------------------
# Inelastic attenuation
# --------------------------
def inelastic_amp(
    dist_km: np.ndarray,
    *,
    peakf_hz: float,
    wavespeed_kms: float,
    Q: float | None,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Amplitude attenuation factor (vector): exp( (pi*f/(Q*v)) * d_km )
    - Q<=0 or v<=0 or f<=0 → ones
    - NaNs preserved
    - Overflow-safe (clip exponent)
    """
    d = _as_dtype(dist_km, out_dtype)
    if Q is None or Q <= 0 or wavespeed_kms <= 0 or peakf_hz <= 0:
        out = np.ones_like(d, dtype=d.dtype)
        return _preserve_nan(d, out)

    const = (np.pi * d.dtype.type(peakf_hz)) / (d.dtype.type(Q) * d.dtype.type(wavespeed_kms))
    exponent = const * d
    max_exp = d.dtype.type(88.0) if d.dtype == np.float32 else d.dtype.type(700.0)
    exponent = np.clip(exponent, -max_exp, max_exp)
    out = np.exp(exponent, dtype=d.dtype)
    return _preserve_nan(d, out)

def inelastic_energy(
    dist_km: np.ndarray,
    *,
    peakf_hz: float,
    wavespeed_kms: float,
    Q: float | None,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    Energy attenuation factor = (inelastic_amp)^2.
    """
    a = inelastic_amp(dist_km, peakf_hz=peakf_hz, wavespeed_kms=wavespeed_kms, Q=Q, out_dtype=out_dtype)
    return a * a

# --------------------------
# Fused helpers
# --------------------------
def total_amp_correction(
    dist_km: np.ndarray,
    *,
    chan: str | None,
    surface_waves: bool,
    wavespeed_kms: float,
    peakf_hz: float,
    Q: float | None,
    out_dtype: str = "float32",
) -> np.ndarray:
    """
    corr_amp = geom_amp * inelastic_amp
    """
    g = geom_spread_amp(dist_km, chan=chan, surface_waves=surface_waves,
                        wavespeed_kms=wavespeed_kms, peakf_hz=peakf_hz, out_dtype=out_dtype)
    a = inelastic_amp(dist_km, peakf_hz=peakf_hz, wavespeed_kms=wavespeed_kms, Q=Q, out_dtype=out_dtype)
    out = g * a
    return _preserve_nan(np.asarray(dist_km), out)

def total_energy_correction(
    dist_km: np.ndarray,
    *,
    chan: str | None,
    surface_waves: bool,
    wavespeed_kms: float,
    peakf_hz: float,
    Q: float | None,
    out_dtype: str = "float32",
    legacy_body: bool = False,
) -> np.ndarray:
    """
    corr_energy = geom_energy * inelastic_energy
      - If legacy_body=True: use legacy geom rule (body: ~d_km).
    """
    if legacy_body:
        gE = geom_spread_energy_legacy(dist_km, surface_waves=surface_waves,
                                       wavespeed_kms=wavespeed_kms, peakf_hz=peakf_hz, out_dtype=out_dtype)
    else:
        gE = geom_spread_energy(dist_km, chan=chan, surface_waves=surface_waves,
                                wavespeed_kms=wavespeed_kms, peakf_hz=peakf_hz, out_dtype=out_dtype)
    aE = inelastic_energy(dist_km, peakf_hz=peakf_hz, wavespeed_kms=wavespeed_kms, Q=Q, out_dtype=out_dtype)
    out = gE * aE
    return _preserve_nan(np.asarray(dist_km), out)
