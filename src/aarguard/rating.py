"""M-rating: IEEE Std 738 steady-state ampacity R_c(T) for fixed conservative inputs.

Own vectorized implementation of the steady-state heat balance  q_c + q_r = q_s + I^2 R(T_c)  in SI units with the
IEEE 738 (2012 numbering) convection, radiation and clear-atmosphere solar formulas; cross-checked against the
`linerate` package. The ampacity is evaluated at the maximum operating temperature T_max for the
ambient temperature T; the only time-varying input is T (an ambient-adjusted rating), plus the day / night solar flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Conductor:
    name: str
    diameter_m: float            # outside diameter D
    r25_ohm_per_m: float         # AC resistance at 25 degC
    r75_ohm_per_m: float         # AC resistance at 75 degC


@dataclass(frozen=True)
class FixedInputs:
    wind_speed: float = 0.61               # m/s (2 ft/s)
    wind_angle_deg: float = 90.0           # angle between wind and conductor axis (90 = perpendicular)
    emissivity: float = 0.5
    absorptivity: float = 0.5
    elevation_m: float = 0.0
    latitude_deg: float = 30.0


def air_props(t_film, elevation_m):
    mu = 1.458e-6 * (t_film + 273.0) ** 1.5 / (t_film + 383.4)
    rho = (1.293 - 1.525e-4 * elevation_m + 6.379e-9 * elevation_m ** 2) / (1.0 + 0.00367 * t_film)
    k = 2.424e-2 + 7.477e-5 * t_film - 4.407e-9 * t_film ** 2
    return mu, rho, k


def k_angle(phi_deg):
    phi = np.radians(phi_deg)
    return 1.194 - np.cos(phi) + 0.194 * np.cos(2 * phi) + 0.368 * np.sin(2 * phi)


def convective_cooling(t_s, t_a, d, inp: FixedInputs):
    t_film = (t_s + t_a) / 2.0
    mu, rho, k = air_props(t_film, inp.elevation_m)
    dt = np.maximum(t_s - t_a, 0.0)
    n_re = d * rho * inp.wind_speed / mu
    ka = k_angle(inp.wind_angle_deg)
    qc1 = ka * (1.01 + 1.35 * n_re ** 0.52) * k * dt
    qc2 = ka * 0.754 * n_re ** 0.6 * k * dt
    qcn = 3.645 * rho ** 0.5 * d ** 0.75 * dt ** 1.25
    return np.maximum(np.maximum(qc1, qc2), qcn)


def radiative_cooling(t_s, t_a, d, emissivity):
    return 17.8 * d * emissivity * (((t_s + 273.0) / 100.0) ** 4 - ((t_a + 273.0) / 100.0) ** 4)


_QS = (-42.2391, 63.8044, -1.9220, 3.46921e-2, -3.61118e-4, 1.94318e-6, -4.07608e-9)


def clear_sky_flux(h_c_deg, elevation_m):
    """Total heat flux density from a clear atmosphere (W/m^2) at solar altitude H_c, elevation-corrected."""
    h = np.asarray(h_c_deg, dtype=float)
    q = sum(c * h ** i for i, c in enumerate(_QS))
    k_solar = 1.0 + 1.148e-4 * elevation_m - 1.108e-8 * elevation_m ** 2
    return np.where(h > 0, np.maximum(q, 0.0) * k_solar, 0.0)


def solstice_noon_altitude(latitude_deg):
    return 90.0 - abs(latitude_deg - 23.44)


def daytime_solar_heating(d, inp: FixedInputs):
    """Fixed conservative daytime solar gain: clear sky, solar noon of the summer solstice, sun perpendicular to the
    conductor (sin(theta) = 1). Night: 0."""
    return inp.absorptivity * clear_sky_flux(solstice_noon_altitude(inp.latitude_deg), inp.elevation_m) * d


def resistance(t_c, c: Conductor):
    return c.r25_ohm_per_m + (c.r75_ohm_per_m - c.r25_ohm_per_m) / 50.0 * (t_c - 25.0)


def ampacity(t_ambient, c: Conductor, t_max: float, inp: FixedInputs, daytime):
    """Steady-state ampacity (A) at conductor temperature t_max for ambient temperature(s) t_ambient (degC).

    Returns 0 where the heat balance has no positive solution (t_ambient >= t_max or solar gain exceeds cooling).
    """
    t_a = np.asarray(t_ambient, dtype=float)
    qc = convective_cooling(t_max, t_a, c.diameter_m, inp)
    qr = radiative_cooling(t_max, t_a, c.diameter_m, inp.emissivity)
    qs = np.where(np.asarray(daytime, dtype=bool), daytime_solar_heating(c.diameter_m, inp), 0.0)
    num = qc + qr - qs
    amp = np.sqrt(np.maximum(num, 0.0) / resistance(t_max, c))
    return np.where(t_a < t_max, amp, 0.0)


# --- Fixed rating inputs -------------------------------------------------------------
KFT_PER_M = 1.0 / 304.8
DRAKE = Conductor("Drake", 1.108 * 0.0254, 0.0219 * KFT_PER_M, 0.0262 * KFT_PER_M)
LINNET = Conductor("Linnet", 0.720 * 0.0254, 0.0514 * KFT_PER_M, 0.0615 * KFT_PER_M)
CONDUCTORS = {"Drake": DRAKE, "Linnet": LINNET}
T_MAX_CASES = (75.0, 100.0)
T_LOW = -30.0


def station_inputs(latitude_deg: float, elevation_m: float) -> FixedInputs:
    return FixedInputs(wind_speed=0.61, wind_angle_deg=90.0, emissivity=0.5, absorptivity=0.5,
                       elevation_m=elevation_m, latitude_deg=latitude_deg)


def sun_elevation_deg(times_utc, latitude_deg: float, longitude_deg: float):
    """Solar elevation (deg) at UTC instants, NOAA general solar position approximation (fractional-year form)."""
    import pandas as pd
    t = pd.DatetimeIndex(times_utc)
    doy = t.dayofyear.values
    hour = t.hour.values + t.minute.values / 60.0
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    tst = hour * 60.0 + eqtime + 4.0 * longitude_deg
    ha = np.radians(tst / 4.0 - 180.0)
    lat = np.radians(latitude_deg)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return 90.0 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))
