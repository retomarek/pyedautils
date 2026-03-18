"""Thermodynamic functions for the Mollier h,x-diagram.

Ported from d3-mollierhx/src/mollierFunctions.js and drawComfort.js.
Reference: Glück, "Zustands- und Stoffwerte — Wasser — Dampf — Luft", ch. 2.1–2.2.

Coordinate system:
    x — absolute humidity [kg/kg]
    y — defined as (h - r_0*x) / c_pL  [°C]
        At x=0 this equals the temperature, so the y-axis can be labeled in °C.
"""

import math

import numpy as np

# Physical constants
C_PL = 1.01       # kJ/(kg·K) — specific heat capacity of dry air
C_PW = 1.86       # kJ/(kg·K) — specific heat capacity of water vapour
R_0 = 2501.0      # kJ/kg     — latent heat of vaporisation at 0 °C
K = 0.6222         # kg/kg     — molar-mass ratio water / dry air (18.02/28.96)
R = 8.3144         # kJ/(kmol·K) — universal gas constant
R_W = R / 18.02    # kJ/(kg·K)  — specific gas constant of water vapour
K_0C = 273.15      # K          — zero Celsius in Kelvin

# Coefficients for the saturation-pressure polynomial (piecewise, threshold 0.01 °C)
_C = [
    -4.909965e-4, +8.183197e-2, -5.552967e-4, -2.228376e-5, -6.211808e-7,
    -1.91275e-4,  +7.258e-2,    -2.939e-4,    +9.841e-7,    -1.92e-9,
]


# ---------------------------------------------------------------------------
# Scalar helper functions
# ---------------------------------------------------------------------------

def _p_sat_scalar(t):
    """Saturation vapour pressure [Pa] for temperature *t* [°C] (scalar)."""
    if t < 0.01:
        return 611.0 * math.exp(
            _C[0] + _C[1] * t + _C[2] * t**2 + _C[3] * t**3 + _C[4] * t**4
        )
    return 611.0 * math.exp(
        _C[5] + _C[6] * t + _C[7] * t**2 + _C[8] * t**3 + _C[9] * t**4
    )


def _log_p_sat_scalar(t):
    """log(saturation pressure) — used by Newton solver."""
    if t < 0.01:
        return math.log(611.0) + (
            _C[0] + _C[1] * t + _C[2] * t**2 + _C[3] * t**3 + _C[4] * t**4
        )
    return math.log(611.0) + (
        _C[5] + _C[6] * t + _C[7] * t**2 + _C[8] * t**3 + _C[9] * t**4
    )


# ---------------------------------------------------------------------------
# Vectorised saturation-pressure (numpy)
# ---------------------------------------------------------------------------

def p_sat(t):
    """Saturation vapour pressure [Pa] for temperature *t* [°C].

    Works with scalars and numpy arrays.
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)

    lo = t < 0.01
    hi = ~lo

    result = np.empty_like(t)
    result[lo] = 611.0 * np.exp(
        _C[0] + _C[1] * t[lo] + _C[2] * t[lo]**2
        + _C[3] * t[lo]**3 + _C[4] * t[lo]**4
    )
    result[hi] = 611.0 * np.exp(
        _C[5] + _C[6] * t[hi] + _C[7] * t[hi]**2
        + _C[8] * t[hi]**3 + _C[9] * t[hi]**4
    )

    return float(result[0]) if scalar else result


def temperature_p_sat(p_s):
    """Inverse of *p_sat*: temperature [°C] from saturation pressure [Pa].

    Uses Newton's method on log(p_sat) for better convergence.
    """
    if p_s >= math.exp(14.2):
        raise ValueError(f"Saturation pressure too high: {p_s}")

    log_p_s = math.log(p_s)
    t = 0.0
    eps = 1e-4

    for _ in range(200):
        residual = _log_p_sat_scalar(t) - log_p_s
        if abs(residual) <= 1e-3:
            return t
        deriv = (_log_p_sat_scalar(t + eps) - _log_p_sat_scalar(t - eps)) / (2 * eps)
        t -= residual / deriv

    raise RuntimeError("temperature_p_sat did not converge")  # pragma: no cover


# ---------------------------------------------------------------------------
# Coordinate functions (scalar)
# ---------------------------------------------------------------------------

def enthalpy(x, y):
    """Enthalpy [kJ/kg] from diagram coordinates (*x*, *y*)."""
    return R_0 * x + C_PL * y


def temperature(x, y):
    """Temperature [°C] from diagram coordinates (*x*, *y*)."""
    return (y * C_PL * (1 + x) + R_0 * x**2) / (C_PL + x * C_PW)


def rel_humidity(x, y, p):
    """Relative humidity [0–1] from diagram coordinates and pressure [Pa]."""
    return x / (K + x) * p / _p_sat_scalar(temperature(x, y))


def density(x, y, p):
    """Air density [kg/m³] from diagram coordinates and pressure [Pa]."""
    t = temperature(x, y)
    return p / (R_W * (K_0C + t)) * (1 + x) / (K + x) / 1000


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def _t_to_y(t, x):
    """Convert temperature + absolute humidity to y-coordinate."""
    return (t * (C_PL + x * C_PW) - R_0 * x**2) / (C_PL * (1 + x))


def get_x_y(t, phi, p):
    """(x, y) from temperature [°C], relative humidity [0–1], pressure [Pa].

    Works with scalars and numpy arrays.
    """
    t = np.asarray(t, dtype=float)
    phi = np.asarray(phi, dtype=float)
    scalar = t.ndim == 0 and phi.ndim == 0

    ps = p_sat(t)
    x_val = phi * K / (p / ps - phi)
    y_val = (t * (C_PL + x_val * C_PW) - R_0 * x_val**2) / (C_PL * (1 + x_val))

    if scalar:
        return float(x_val), float(y_val)
    return x_val, y_val


def get_x_y_tx(t, x, p):
    """(x, y) from temperature [°C] and absolute humidity [kg/kg]."""
    y = _t_to_y(t, x)
    return x, y


def y_phix(phi, x, p):
    """y-coordinate from relative humidity, absolute humidity, pressure."""
    t_s = temperature_p_sat(x * p / (phi * (K + x)))
    return (C_PL + x * C_PW) / (C_PL * (1 + x)) * (
        t_s - R_0 * x**2 / (C_PL + x * C_PW)
    )


def x_phiy(phi, y, p):
    """Absolute humidity from relative humidity and y-coordinate (Newton)."""
    def _phi_of_x(xv):
        return xv / (K + xv) * p / _p_sat_scalar(temperature(xv, y))

    x = 0.0
    eps = 1e-6
    for _ in range(200):
        res = _phi_of_x(x) - phi
        if abs(res) <= 1e-5:
            return x
        deriv = (_phi_of_x(x + eps) - _phi_of_x(x - eps)) / (2 * eps)
        x -= res / deriv

    raise RuntimeError("x_phiy did not converge")  # pragma: no cover


def x_hy(h, y):
    """Absolute humidity from enthalpy [kJ/kg] and y-coordinate."""
    return (h - C_PL * y) / R_0


def y_hx(h, x):
    """y-coordinate from enthalpy [kJ/kg] and absolute humidity."""
    return (h - R_0 * x) / C_PL


def y_rhox(rho, x, p):
    """y-coordinate from density [kg/m³], absolute humidity, pressure [Pa]."""
    return (C_PL + x * C_PW) / (C_PL * (1 + x)) * (
        p / (R_W * rho) * (1 + x) / (K + x) * 0.001
        - K_0C
        - R_0 * x**2 / (C_PL + x * C_PW)
    )


# ---------------------------------------------------------------------------
# Comfort zone polygon
# ---------------------------------------------------------------------------

def _sort_range(r):
    return (min(r), max(r))


def _isin(x, rng):
    return (rng[0] < x < rng[1]) or (rng[1] < x < rng[0])


class _ComfortBuilder:
    """Stateful builder that traces the comfort-zone boundary."""

    def __init__(self, range_x, p):
        self.range_x = range_x
        self.p = p
        self.output = []
        self.old_x = 0.0
        self.old_y = 0.0
        self.inrange = False

    def init_point(self, x0, y0):
        self.old_x, self.old_y = x0, y0
        self.inrange = self.range_x[0] < x0 < self.range_x[1]

    def handle_step(self, punkt_x, punkt_y, variable, func):
        for boundary in (self.range_x[0], self.range_x[1]):
            if _isin(boundary, (self.old_x, punkt_x)) or boundary == punkt_x:
                ix, iy = func(variable, boundary, self.p)
                self.output.append((ix, iy))
                self.inrange = not self.inrange
        if self.inrange:
            self.output.append((punkt_x, punkt_y))
        self.old_x, self.old_y = punkt_x, punkt_y


def _phi_interp(phi, xv, p):
    """Interpolation function for constant-phi boundary crossings."""
    return get_x_y_tx(temperature(xv, y_phix(phi, xv, p)), xv, p)


def _sweep_side(builder, values, phi_or_t, get_point, interp_func):
    """Sweep one side of the comfort rectangle."""
    for val in values:
        px, py = get_point(val)
        builder.handle_step(px, py, phi_or_t, interp_func)


def create_comfort(range_t, range_phi, range_x, p):  # noqa: C901
    """Create the comfort-zone polygon as a list of (x, y) tuples.

    Args:
        range_t: (min, max) temperature [°C].
        range_phi: (min, max) relative humidity [0–1].
        range_x: (min, max) absolute humidity [kg/kg].
        p: Pressure [Pa].

    Returns:
        List of (x, y) coordinate pairs forming a closed polygon.
    """
    range_t = _sort_range(range_t)
    range_phi = _sort_range(range_phi)
    range_x = _sort_range(range_x)

    if range_phi[1] == 0:
        return [(0, range_t[0]), (0, range_t[1]), (0, range_t[0])]

    dT = 0.1
    dPhi = 0.01

    builder = _ComfortBuilder(range_x, p)
    T = range_t[0]
    Phi = range_phi[0]
    x0, y0 = get_x_y(T, Phi, p)
    builder.init_point(x0, y0)

    # Side 1: T increases at Phi = rangePhi[0]
    if Phi != 0:
        t_vals = _arange_inclusive(T + dT, range_t[1], dT)
        _sweep_side(builder, t_vals, Phi,
                    lambda t: get_x_y(t, Phi, p), _phi_interp)
        T = range_t[1]
    else:
        T = range_t[1]
        px, py = get_x_y(T, Phi, p)
        builder.inrange = (range_x[0] == 0)
        if builder.inrange:
            builder.output.append((px, py))
            builder.old_x, builder.old_y = px, py

    # Side 2: Phi increases at T = rangeT[1]
    phi_vals = _arange_inclusive(Phi + dPhi, range_phi[1], dPhi)
    _sweep_side(builder, phi_vals, T,
                lambda phi: get_x_y(T, phi, p), get_x_y_tx)
    Phi = range_phi[1]

    # Side 3: T decreases at Phi = rangePhi[1]
    t_vals = _arange_inclusive(T - dT, range_t[0], -dT)
    _sweep_side(builder, t_vals, Phi,
                lambda t: get_x_y(t, Phi, p), _phi_interp)
    T = range_t[0]

    # Side 4: Phi decreases at T = rangeT[0]
    phi_vals = _arange_inclusive(Phi - dPhi, range_phi[0], -dPhi)
    _sweep_side(builder, phi_vals, T,
                lambda phi: get_x_y(T, phi, p), get_x_y_tx)

    if builder.output:
        builder.output.append(builder.output[0])

    return builder.output


def _arange_inclusive(start, stop, step):
    """Generate values from *start* toward *stop* (inclusive) with *step*."""
    vals = []
    if step > 0:
        v = start
        while v < stop:
            vals.append(v)
            v += step
        vals.append(stop)
    else:
        v = start
        while v > stop:
            vals.append(v)
            v += step
        vals.append(stop)
    return vals
