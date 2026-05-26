"""Thermodynamic functions for the Mollier h,x-diagram.

Ported from d3-mollierhx/src/mollierFunctions.js and drawComfort.js.
Reference: Glück, "Zustands- und Stoffwerte — Wasser — Dampf — Luft", ch. 2.1–2.2.

Coordinate system:
    x — absolute humidity [kg/kg]
    y — defined as (h - r_0*x) / c_pL  [°C]
        At x=0 this equals the temperature, so the y-axis can be labeled in °C.

Conventions:
    'classical'  — Enthalpy h normalised per kg of dry air (Mollier 1923,
                   Recknagel/Sprenger). Isotherms tilt slightly *upward* with x.
                   Default since pyedautils 0.x (was implicitly 'glueck' before).
    'glueck'     — Enthalpy h normalised per kg of moist air (Glück book).
                   Isotherms tilt slightly *downward* with x; T=0 isotherm is
                   approximately horizontal with a quadratic deviation in x.

Both conventions are mathematically self-consistent — pick the one that
matches your reference plots. The convention only affects how the (x, y)
plane is parametrised; physical quantities (T, φ, ρ, h) come out the same
when you round-trip through a single convention.
"""

import math

import numpy as np

# Physical constants
C_PL = 1.01       # kJ/(kg·K) — specific heat capacity of dry air
C_PW = 1.86       # kJ/(kg·K) — specific heat capacity of water vapour
C_W_FL = 4.19     # kJ/(kg·K) — specific heat capacity of liquid water
R_0 = 2501.0      # kJ/kg     — latent heat of vaporisation at 0 °C
K = 0.6222         # kg/kg     — molar-mass ratio water / dry air (18.02/28.96)
R = 8.3144         # kJ/(kmol·K) — universal gas constant
R_W = R / 18.02    # kJ/(kg·K)  — specific gas constant of water vapour
K_0C = 273.15      # K          — zero Celsius in Kelvin
P_STD = 101325.0   # Pa         — standard atmospheric pressure at sea level

# Coefficients for the saturation-pressure polynomial (piecewise, threshold 0.01 °C)
_C = [
    -4.909965e-4, +8.183197e-2, -5.552967e-4, -2.228376e-5, -6.211808e-7,
    -1.91275e-4,  +7.258e-2,    -2.939e-4,    +9.841e-7,    -1.92e-9,
]

DEFAULT_CONVENTION = 'classical'
_VALID_CONVENTIONS = ('classical', 'glueck')


def _check_convention(convention):
    if convention is None:
        return DEFAULT_CONVENTION
    if convention not in _VALID_CONVENTIONS:
        raise ValueError(
            f"Unknown Mollier convention {convention!r}; "
            f"expected one of {_VALID_CONVENTIONS}"
        )
    return convention


# ---------------------------------------------------------------------------
# Scalar helper functions
# ---------------------------------------------------------------------------

def _p_sat_scalar(t) -> float:
    """Saturation vapour pressure [Pa] for temperature *t* [°C] (scalar)."""
    if t < 0.01:
        return 611.0 * math.exp(
            _C[0] + _C[1] * t + _C[2] * t**2 + _C[3] * t**3 + _C[4] * t**4
        )
    return 611.0 * math.exp(
        _C[5] + _C[6] * t + _C[7] * t**2 + _C[8] * t**3 + _C[9] * t**4
    )


def _log_p_sat_scalar(t) -> float:
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

    Works with scalars and numpy arrays. Convention-independent.
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


def temperature_p_sat(p_s) -> float:
    """Inverse of *p_sat*: temperature [°C] from saturation pressure [Pa].

    Convention-independent. Uses Newton's method on log(p_sat) for better
    convergence.
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
# Convention-independent derived properties
# ---------------------------------------------------------------------------
# All functions below describe physical quantities of the moist-air state.
# They do not depend on how the (x, y) plane is parametrised — i.e. no
# `convention` argument — and round-trip cleanly with the diagram coordinates.

def pressure_from_altitude(altitude_m) -> float:
    """Atmospheric pressure [Pa] at given altitude [m above sea level].

    International Standard Atmosphere (ISA) barometric formula, valid in the
    troposphere (0–11 km). Reference state: p₀ = 101325 Pa, T₀ = 288.15 K,
    L = 0.0065 K/m.
    """
    return P_STD * (1 - 2.25577e-5 * altitude_m) ** 5.25588


def vapor_pressure(x, p) -> float:
    """Water-vapour partial pressure [Pa] from absolute humidity and pressure.

    Args:
        x: Absolute humidity [kg/kg dry air].
        p: Total pressure [Pa].
    """
    return x * p / (K + x)


def dew_point(x, p) -> float:
    """Dew-point temperature [°C] from absolute humidity and pressure.

    The dew point is the temperature at which the current water-vapour
    partial pressure equals the saturation pressure (φ = 1). Inverts
    ``temperature_p_sat`` for the vapour-pressure of the given (x, p).
    """
    return temperature_p_sat(vapor_pressure(x, p))


def specific_volume(t, x, p) -> float:
    """Specific volume [m³/kg dry air] from temperature, abs. humidity, pressure.

    Args:
        t: Dry-bulb temperature [°C].
        x: Absolute humidity [kg/kg dry air].
        p: Total pressure [Pa].
    """
    # Density of moist air [kg/m³], same formula as ``density()`` but expressed
    # in (T, x) directly so we don't need to round-trip through the diagram.
    t_k = K_0C + t
    rho_moist = p / (R_W * t_k) * (1 + x) / (K + x) / 1000.0
    return (1 + x) / rho_moist


def wet_bulb(t, x, p) -> float:
    """Wet-bulb temperature [°C] from dry-bulb T, abs. humidity x, pressure p.

    Solves the adiabatic-saturation energy balance:

        h(T_db, x) = h(T_wb, x_sat(T_wb)) − (x_sat(T_wb) − x)·c_W_fl·T_wb

    iteratively (Newton with numerical derivative). Returns T_wb such that
    air at T_db, x is in equilibrium with water adiabatically evaporated
    to saturation at T_wb.

    Notes:
        - At saturation (x = x_sat(T_db)), T_wb = T_db.
        - Convention-independent (T_wb is a physical thermodynamic quantity).
    """
    def x_sat_at(t_w):
        ps = _p_sat_scalar(t_w)
        # Guard against p_sat ≥ p which would yield negative or infinite x_sat.
        if ps >= p:
            raise ValueError(f"Saturation pressure exceeds total pressure at {t_w} °C")
        return K * ps / (p - ps)

    def residual(t_w):
        h_db = C_PL * t + x * (R_0 + C_PW * t)
        xs = x_sat_at(t_w)
        h_sat = C_PL * t_w + xs * (R_0 + C_PW * t_w)
        h_liq_correction = (xs - x) * C_W_FL * t_w
        return h_db - h_sat + h_liq_correction

    t_w = t - 1.0  # T_wb ≤ T_db for unsaturated air; start just below
    eps = 1e-4
    for _ in range(200):
        r = residual(t_w)
        if abs(r) <= 1e-3:
            return t_w
        deriv = (residual(t_w + eps) - residual(t_w - eps)) / (2 * eps)
        if deriv == 0:
            break  # pragma: no cover
        t_w -= r / deriv

    raise RuntimeError("wet_bulb did not converge")  # pragma: no cover


# ---------------------------------------------------------------------------
# Convention-dependent low-level transforms
# ---------------------------------------------------------------------------

def _y_to_t(x, y, convention):
    """Compute temperature [°C] from diagram coordinates (x, y)."""
    if convention == 'classical':
        return y * C_PL / (C_PL + x * C_PW)
    # 'glueck'
    return (y * C_PL * (1 + x) + R_0 * x**2) / (C_PL + x * C_PW)


def _t_to_y(t, x, convention):
    """Compute diagram y-coordinate from temperature [°C] and abs. humidity [kg/kg]."""
    if convention == 'classical':
        return t * (C_PL + x * C_PW) / C_PL
    # 'glueck'
    return (t * (C_PL + x * C_PW) - R_0 * x**2) / (C_PL * (1 + x))


# ---------------------------------------------------------------------------
# Convention-independent transforms
# ---------------------------------------------------------------------------

def enthalpy(x, y) -> float:
    """Enthalpy [kJ/kg] from diagram coordinates (*x*, *y*).

    Convention-independent: h = R_0*x + C_PL*y holds by the y-axis definition
    in both conventions.
    """
    return R_0 * x + C_PL * y


def x_hy(h, y) -> float:
    """Absolute humidity from enthalpy [kJ/kg] and y-coordinate.

    Convention-independent.
    """
    return (h - C_PL * y) / R_0


def y_hx(h, x) -> float:
    """y-coordinate from enthalpy [kJ/kg] and absolute humidity.

    Convention-independent.
    """
    return (h - R_0 * x) / C_PL


# ---------------------------------------------------------------------------
# Convention-dependent coordinate functions (scalar)
# ---------------------------------------------------------------------------

def temperature(x, y, convention=None) -> float:
    """Temperature [°C] from diagram coordinates (*x*, *y*)."""
    return _y_to_t(x, y, _check_convention(convention))


def rel_humidity(x, y, p, convention=None) -> float:
    """Relative humidity [0–1] from diagram coordinates and pressure [Pa]."""
    convention = _check_convention(convention)
    return x / (K + x) * p / _p_sat_scalar(_y_to_t(x, y, convention))


def density(x, y, p, convention=None) -> float:
    """Air density [kg/m³] from diagram coordinates and pressure [Pa]."""
    convention = _check_convention(convention)
    t = _y_to_t(x, y, convention)
    return p / (R_W * (K_0C + t)) * (1 + x) / (K + x) / 1000


# ---------------------------------------------------------------------------
# Convention-dependent coordinate conversions
# ---------------------------------------------------------------------------

def get_x_y(t, phi, p, convention=None):
    """(x, y) from temperature [°C], relative humidity [0–1], pressure [Pa].

    Works with scalars and numpy arrays.
    """
    convention = _check_convention(convention)
    t = np.asarray(t, dtype=float)
    phi = np.asarray(phi, dtype=float)
    scalar = t.ndim == 0 and phi.ndim == 0

    ps = p_sat(t)
    x_val = phi * K / (p / ps - phi)
    y_val = _t_to_y(t, x_val, convention)

    if scalar:
        return float(x_val), float(y_val)
    return x_val, y_val


def get_x_y_tx(t, x, p, convention=None):
    """(x, y) from temperature [°C] and absolute humidity [kg/kg]."""
    convention = _check_convention(convention)
    return x, _t_to_y(t, x, convention)


def y_phix(phi, x, p, convention=None) -> float:
    """y-coordinate from relative humidity, absolute humidity, pressure."""
    convention = _check_convention(convention)
    t_s = temperature_p_sat(x * p / (phi * (K + x)))
    return _t_to_y(t_s, x, convention)


def x_phiy(phi, y, p, convention=None) -> float:
    """Absolute humidity from relative humidity and y-coordinate (Newton)."""
    convention = _check_convention(convention)

    def _phi_of_x(xv):
        return xv / (K + xv) * p / _p_sat_scalar(_y_to_t(xv, y, convention))

    x = 0.0
    eps = 1e-6
    for _ in range(200):
        res = _phi_of_x(x) - phi
        if abs(res) <= 1e-5:
            return x
        deriv = (_phi_of_x(x + eps) - _phi_of_x(x - eps)) / (2 * eps)
        x -= res / deriv

    raise RuntimeError("x_phiy did not converge")  # pragma: no cover


def y_rhox(rho, x, p, convention=None) -> float:
    """y-coordinate from density [kg/m³], absolute humidity, pressure [Pa]."""
    convention = _check_convention(convention)
    t = p / (R_W * rho) * (1 + x) / (K + x) * 0.001 - K_0C
    return _t_to_y(t, x, convention)


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


def _phi_interp(phi, xv, p, convention):
    """Interpolation function for constant-phi boundary crossings."""
    return get_x_y_tx(
        _y_to_t(xv, y_phix(phi, xv, p, convention), convention),
        xv, p, convention,
    )


def _sweep_side(builder, values, phi_or_t, get_point, interp_func):
    """Sweep one side of the comfort rectangle."""
    for val in values:
        px, py = get_point(val)
        builder.handle_step(px, py, phi_or_t, interp_func)


def create_comfort(range_t, range_phi, range_x, p, convention=None):  # noqa: C901
    """Create the comfort-zone polygon as a list of (x, y) tuples.

    Args:
        range_t: (min, max) temperature [°C].
        range_phi: (min, max) relative humidity [0–1].
        range_x: (min, max) absolute humidity [kg/kg].
        p: Pressure [Pa].
        convention: Mollier convention ('classical' default, or 'glueck').

    Returns:
        List of (x, y) coordinate pairs forming a closed polygon.
    """
    convention = _check_convention(convention)
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
    x0, y0 = get_x_y(T, Phi, p, convention)
    builder.init_point(x0, y0)

    interp = lambda v, xv, pp: _phi_interp(v, xv, pp, convention)  # noqa: E731

    # Side 1: T increases at Phi = rangePhi[0]
    if Phi != 0:
        t_vals = _arange_inclusive(T + dT, range_t[1], dT)
        _sweep_side(builder, t_vals, Phi,
                    lambda t: get_x_y(t, Phi, p, convention), interp)
        T = range_t[1]
    else:
        T = range_t[1]
        px, py = get_x_y(T, Phi, p, convention)
        builder.inrange = (range_x[0] == 0)
        if builder.inrange:
            builder.output.append((px, py))
            builder.old_x, builder.old_y = px, py

    # Side 2: Phi increases at T = rangeT[1]
    phi_vals = _arange_inclusive(Phi + dPhi, range_phi[1], dPhi)
    _sweep_side(builder, phi_vals, T,
                lambda phi: get_x_y(T, phi, p, convention),
                lambda v, xv, pp: get_x_y_tx(v, xv, pp, convention))
    Phi = range_phi[1]

    # Side 3: T decreases at Phi = rangePhi[1]
    t_vals = _arange_inclusive(T - dT, range_t[0], -dT)
    _sweep_side(builder, t_vals, Phi,
                lambda t: get_x_y(t, Phi, p, convention), interp)
    T = range_t[0]

    # Side 4: Phi decreases at T = rangeT[0]
    phi_vals = _arange_inclusive(Phi - dPhi, range_phi[0], -dPhi)
    _sweep_side(builder, phi_vals, T,
                lambda phi: get_x_y(T, phi, p, convention),
                lambda v, xv, pp: get_x_y_tx(v, xv, pp, convention))

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
