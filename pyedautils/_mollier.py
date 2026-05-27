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
from dataclasses import dataclass
from typing import Optional

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


# ---------------------------------------------------------------------------
# State point — MoistAirState dataclass + universal factory
# ---------------------------------------------------------------------------
# An immutable record of a single moist-air point, populated with every
# derived property in one shot. The companion factory ``state(...)`` accepts
# any two of {t, t_wb, t_dp, phi, x, h} plus pressure (or altitude) and
# optional mass/volume flow.


@dataclass(frozen=True)
class MoistAirState:
    """Immutable moist-air state with all derived psychrometric properties.

    Construct via :func:`state`, never call this constructor directly.

    Intensive properties (always populated):
        t:      dry-bulb temperature [°C]
        phi:    relative humidity [0..1]
        x:      absolute humidity [kg/kg dry air]
        h:      specific enthalpy [kJ/kg dry air]
        t_wb:   wet-bulb temperature [°C]
        t_dp:   dew-point temperature [°C]
        p_v:    water-vapour partial pressure [Pa]
        rho:    moist-air density [kg/m³]
        v:      specific volume [m³/kg dry air]
        y:      diagram y-coordinate (depends on ``convention``)

    Extensive properties (optional, ``None`` when not specified):
        m_dot_dry:   dry-air mass flow [kg/s]
        volume_flow: volumetric flow rate [m³/h], evaluated at ``v``

    Context:
        p:          total pressure [Pa]
        convention: Mollier coordinate convention ('classical' | 'glueck')
    """
    t: float
    phi: float
    x: float
    h: float
    t_wb: float
    t_dp: float
    p_v: float
    rho: float
    v: float
    y: float
    m_dot_dry: Optional[float] = None
    volume_flow: Optional[float] = None
    p: float = P_STD
    convention: str = DEFAULT_CONVENTION


def state(*, t=None, t_wb=None, t_dp=None, phi=None, x=None, h=None,
          p=None, altitude=None, m_dot_dry=None, volume_flow=None,
          convention=None) -> MoistAirState:
    """Construct a :class:`MoistAirState` from any two psychrometric properties.

    Exactly two of ``{t, t_wb, t_dp, phi, x, h}`` must be given. All other
    properties are computed and returned.

    Args:
        t, t_wb, t_dp:   dry-bulb, wet-bulb, dew-point temperatures [°C]
        phi:             relative humidity [0..1]
        x:               absolute humidity [kg/kg dry air]
        h:               enthalpy [kJ/kg dry air]
        p:               total pressure [Pa] (defaults to sea-level standard
                         when neither ``p`` nor ``altitude`` is given)
        altitude:        altitude in metres a.s.l.; mutually exclusive with ``p``
        m_dot_dry:       dry-air mass flow [kg/s] (mutually exclusive with
                         ``volume_flow``)
        volume_flow:     volumetric flow [m³/h] at the state's specific volume;
                         converted internally to ``m_dot_dry``
        convention:      'classical' (default) or 'glueck'

    Raises:
        ValueError: on invalid argument combinations (e.g. wrong number of
            psychrometric properties, both ``p`` and ``altitude``, degenerate
            pair like ``(x, t_dp)``).
    """
    convention = _check_convention(convention)

    if altitude is not None and p is not None:
        raise ValueError("Specify either p or altitude, not both")
    if altitude is not None:
        p = pressure_from_altitude(altitude)
    if p is None:
        p = P_STD

    if m_dot_dry is not None and volume_flow is not None:
        raise ValueError("Specify either m_dot_dry or volume_flow, not both")

    provided = {k: v for k, v in {
        't': t, 't_wb': t_wb, 't_dp': t_dp,
        'phi': phi, 'x': x, 'h': h,
    }.items() if v is not None}
    if len(provided) != 2:
        raise ValueError(
            f"state() requires exactly two of {{t, t_wb, t_dp, phi, x, h}}; "
            f"got {len(provided)}: {sorted(provided.keys())}"
        )

    t_val, x_val = _resolve_t_x(provided, p)

    ps_t = _p_sat_scalar(t_val)
    phi_val = x_val / (K + x_val) * p / ps_t
    h_val = C_PL * t_val + x_val * (R_0 + C_PW * t_val)
    t_wb_val = wet_bulb(t_val, x_val, p)
    t_dp_val = dew_point(x_val, p)
    p_v_val = vapor_pressure(x_val, p)
    rho_val = p / (R_W * (K_0C + t_val)) * (1 + x_val) / (K + x_val) / 1000.0
    v_val = (1 + x_val) / rho_val
    y_val = _t_to_y(t_val, x_val, convention)

    if volume_flow is not None:
        m_dot_dry = volume_flow / 3600.0 / v_val

    return MoistAirState(
        t=t_val, phi=phi_val, x=x_val, h=h_val,
        t_wb=t_wb_val, t_dp=t_dp_val, p_v=p_v_val,
        rho=rho_val, v=v_val, y=y_val,
        m_dot_dry=m_dot_dry, volume_flow=volume_flow,
        p=p, convention=convention,
    )


# ---------------------------------------------------------------------------
# Private resolvers — find (t, x) from any pair of psychrometric properties
# ---------------------------------------------------------------------------

def _resolve_t_x(provided, p):  # noqa: C901
    keys = frozenset(provided.keys())

    if keys == frozenset({'x', 't_dp'}):
        raise ValueError(
            "state(): (x, t_dp) is degenerate — both fix x without "
            "constraining T. Provide t, phi, or h instead."
        )

    if 't_dp' in keys:
        tdp = provided['t_dp']
        ps_tdp = _p_sat_scalar(tdp)
        if ps_tdp >= p:
            raise ValueError(f"t_dp={tdp} °C: saturation pressure exceeds total pressure")
        x_from_tdp = K * ps_tdp / (p - ps_tdp)
        rest = {k: v for k, v in provided.items() if k != 't_dp'}
        rest['x'] = x_from_tdp
        return _resolve_t_x(rest, p)

    if 't' in keys:
        t_val = provided['t']
        if 'x' in keys:
            return t_val, provided['x']
        if 'phi' in keys:
            ps = _p_sat_scalar(t_val)
            return t_val, K * provided['phi'] * ps / (p - provided['phi'] * ps)
        if 'h' in keys:
            return t_val, (provided['h'] - C_PL * t_val) / (R_0 + C_PW * t_val)
        if 't_wb' in keys:
            return t_val, _x_from_t_twb(t_val, provided['t_wb'], p)

    if 'x' in keys:
        x_val = provided['x']
        if 'phi' in keys:
            return _t_from_x_phi(x_val, provided['phi'], p), x_val
        if 'h' in keys:
            return (provided['h'] - R_0 * x_val) / (C_PL + C_PW * x_val), x_val
        if 't_wb' in keys:
            return _t_from_x_twb(x_val, provided['t_wb'], p), x_val

    if keys == frozenset({'phi', 'h'}):
        return _t_x_from_phi_h(provided['phi'], provided['h'], p)
    if keys == frozenset({'phi', 't_wb'}):
        return _t_x_from_phi_twb(provided['phi'], provided['t_wb'], p)
    if keys == frozenset({'h', 't_wb'}):
        return _t_x_from_h_twb(provided['h'], provided['t_wb'], p)

    raise ValueError(f"state(): unsupported property pair {sorted(keys)}")  # pragma: no cover


def _x_from_t_twb(t, t_wb, p):
    """Closed-form solution: x from (T_db, T_wb, p)."""
    ps_wb = _p_sat_scalar(t_wb)
    if ps_wb >= p:
        raise ValueError(f"t_wb={t_wb} °C: saturation pressure exceeds total pressure")
    x_sat = K * ps_wb / (p - ps_wb)
    num = C_PL * (t_wb - t) + x_sat * (R_0 + (C_PW - C_W_FL) * t_wb)
    den = R_0 + C_PW * t - C_W_FL * t_wb
    return num / den


def _t_from_x_twb(x, t_wb, p):
    """Closed-form solution: t from (x, T_wb, p)."""
    ps_wb = _p_sat_scalar(t_wb)
    if ps_wb >= p:
        raise ValueError(f"t_wb={t_wb} °C: saturation pressure exceeds total pressure")
    x_sat = K * ps_wb / (p - ps_wb)
    num = (C_PL * t_wb
           + x_sat * (R_0 + (C_PW - C_W_FL) * t_wb)
           + x * (C_W_FL * t_wb - R_0))
    den = C_PL + x * C_PW
    return num / den


def _t_from_x_phi(x, phi, p):
    """Direct: t from (x, phi, p) via the phi definition."""
    return temperature_p_sat(x * p / (phi * (K + x)))


def _newton_1d(residual, t0, tol=1e-3, max_iter=200, eps=1e-4, label="state"):
    """Generic 1-D Newton with numerical derivative."""
    t_val = t0
    for _ in range(max_iter):
        r = residual(t_val)
        if abs(r) <= tol:
            return t_val
        deriv = (residual(t_val + eps) - residual(t_val - eps)) / (2 * eps)
        if deriv == 0:
            break  # pragma: no cover
        t_val -= r / deriv
    raise RuntimeError(f"_newton_1d ({label}) did not converge")  # pragma: no cover


def _t_x_from_phi_h(phi, h, p):
    """Newton on t: enthalpy with x bound to (t, phi)."""
    def x_at_t(t_test):
        ps = _p_sat_scalar(t_test)
        return K * phi * ps / (p - phi * ps)

    def residual(t_test):
        xv = x_at_t(t_test)
        return C_PL * t_test + xv * (R_0 + C_PW * t_test) - h

    t_val = _newton_1d(residual, h / C_PL, label="phi,h")
    return t_val, x_at_t(t_val)


def _t_x_from_phi_twb(phi, t_wb, p):
    """Newton on t: x from (t, phi) equals x from (t, t_wb)."""
    def x_from_phi(t_test):
        ps = _p_sat_scalar(t_test)
        return K * phi * ps / (p - phi * ps)

    def residual(t_test):
        return x_from_phi(t_test) - _x_from_t_twb(t_test, t_wb, p)

    t_val = _newton_1d(residual, t_wb + 5.0, tol=1e-7, label="phi,t_wb")
    return t_val, x_from_phi(t_val)


def _t_x_from_h_twb(h, t_wb, p):
    """Newton on t: enthalpy with x bound to (t, t_wb)."""
    def residual(t_test):
        xv = _x_from_t_twb(t_test, t_wb, p)
        return C_PL * t_test + xv * (R_0 + C_PW * t_test) - h

    t_val = _newton_1d(residual, t_wb + 5.0, label="h,t_wb")
    return t_val, _x_from_t_twb(t_val, t_wb, p)


# ---------------------------------------------------------------------------
# Processes — psychrometric transformations between states
# ---------------------------------------------------------------------------
# Each process function takes one (or two, for mix/heat_recovery) input
# MoistAirState plus the process spec, and returns ``(new_state, balance)``.
# Mass flow propagates through single-stream processes (heat, cool, humidify
# preserve ``m_dot_dry``); ``mix`` sums dry-air mass flows.


@dataclass(frozen=True)
class ProcessBalance:
    """Energy and water balance for one psychrometric process step.

    Attributes:
        name:             process identifier ('heat', 'cool', …)
        dh:               specific enthalpy change [kJ/kg dry air]
                          (positive = heating, negative = cooling).
                          0 by convention for ``mix``.
        dx:               humidity-ratio change [kg/kg dry air]
                          (positive = humidification, negative = drying).
                          0 by convention for ``mix``.
        power_kw:         m_dot_dry × dh [kW]; None when m_dot_dry unset
        condensate_kgh:   m_dot_dry × |dx| × 3600 [kg/h] for cooling that
                          crosses the dew point; None otherwise
        water_kgh:        m_dot_dry × dx × 3600 [kg/h] for humidification;
                          None otherwise
        epsilon:          efficiency for heat_recovery; None otherwise
    """
    name: str
    dh: float
    dx: float
    power_kw: Optional[float] = None
    condensate_kgh: Optional[float] = None
    water_kgh: Optional[float] = None
    epsilon: Optional[float] = None


def _exactly_one(name, **kwargs):
    """Helper: ensure exactly one keyword arg is non-None; return its value."""
    given = {k: v for k, v in kwargs.items() if v is not None}
    if len(given) != 1:
        raise ValueError(
            f"{name}: specify exactly one of {sorted(kwargs.keys())}; "
            f"got {sorted(given.keys())}"
        )
    return next(iter(given.items()))


def _state_tx(t, x, ref):
    """Build a new state at (t, x) preserving pressure/convention/m_dot from
    a reference state."""
    return state(t=t, x=x, p=ref.p, convention=ref.convention,
                 m_dot_dry=ref.m_dot_dry)


def _power_kw(m_dot_dry, dh):
    """Compute kW = m_dot_dry × dh. Returns None if mass flow unknown."""
    return None if m_dot_dry is None else m_dot_dry * dh


def heat(s, t_out=None, dt=None, power_kw=None):
    """Sensible heating at constant absolute humidity x.

    Args:
        s:        input MoistAirState
        t_out:    target dry-bulb temperature [°C]
        dt:       temperature change [K]; positive = heating
        power_kw: heating power [kW]; requires ``s.m_dot_dry`` to be set

    Returns:
        (new_state, ProcessBalance)
    """
    mode, val = _exactly_one('heat', t_out=t_out, dt=dt, power_kw=power_kw)
    if mode == 'power_kw':
        if s.m_dot_dry is None:
            raise ValueError("heat(power_kw=...) requires m_dot_dry on the input state")
        dh = val / s.m_dot_dry
        h_new = s.h + dh
        t_new = (h_new - R_0 * s.x) / (C_PL + C_PW * s.x)
    elif mode == 'dt':
        t_new = s.t + val
    else:  # t_out
        t_new = val

    new_s = _state_tx(t_new, s.x, s)
    dh = new_s.h - s.h
    return new_s, ProcessBalance(
        name='heat',
        dh=dh, dx=0.0,
        power_kw=_power_kw(s.m_dot_dry, dh),
    )


def cool(s, t_out=None, dt=None):
    """Cooling. Sensible while T_out ≥ dew point; latent below (output state
    saturated at the coil temperature, BF = 0).

    Args:
        s:      input MoistAirState
        t_out:  target dry-bulb temperature [°C] (= coil temperature in the
                latent regime)
        dt:     temperature change [K]; should be negative

    Returns:
        (new_state, ProcessBalance)
    """
    mode, val = _exactly_one('cool', t_out=t_out, dt=dt)
    t_target = val if mode == 't_out' else s.t + val
    if t_target >= s.t:
        raise ValueError(
            f"cool(): target {t_target} °C is not below inlet {s.t} °C"
        )

    if t_target >= s.t_dp:
        # Sensible regime — x preserved.
        new_s = _state_tx(t_target, s.x, s)
        dh = new_s.h - s.h
        return new_s, ProcessBalance(
            name='cool',
            dh=dh, dx=0.0,
            power_kw=_power_kw(s.m_dot_dry, dh),
        )

    # Latent regime — outlet saturated at the coil temperature.
    ps = _p_sat_scalar(t_target)
    x_new = K * ps / (s.p - ps)
    new_s = _state_tx(t_target, x_new, s)
    dh = new_s.h - s.h
    dx = new_s.x - s.x  # negative — water condensed out
    condensate = None if s.m_dot_dry is None else -dx * s.m_dot_dry * 3600.0
    return new_s, ProcessBalance(
        name='cool',
        dh=dh, dx=dx,
        power_kw=_power_kw(s.m_dot_dry, dh),
        condensate_kgh=condensate,
    )


def humidify_isothermal(s, x_out=None, phi_out=None):
    """Steam humidifier: T stays constant, water vapour added.

    Args:
        s:        input MoistAirState
        x_out:    target humidity ratio [kg/kg]
        phi_out:  target relative humidity [0..1]

    Returns:
        (new_state, ProcessBalance)
    """
    mode, val = _exactly_one('humidify_isothermal', x_out=x_out, phi_out=phi_out)
    if mode == 'phi_out':
        ps = _p_sat_scalar(s.t)
        if val * ps >= s.p:
            raise ValueError(f"humidify_isothermal: phi_out={val} unreachable at t={s.t}")
        x_new = K * val * ps / (s.p - val * ps)
    else:
        x_new = val

    if x_new <= s.x:
        raise ValueError(
            f"humidify_isothermal: target x={x_new:.5g} ≤ inlet x={s.x:.5g}"
        )

    new_s = _state_tx(s.t, x_new, s)
    dh = new_s.h - s.h
    dx = new_s.x - s.x
    water_kgh = None if s.m_dot_dry is None else dx * s.m_dot_dry * 3600.0
    return new_s, ProcessBalance(
        name='humidify_isothermal',
        dh=dh, dx=dx,
        power_kw=_power_kw(s.m_dot_dry, dh),
        water_kgh=water_kgh,
    )


def humidify_adiabatic(s, x_out=None, phi_out=None):
    """Adiabatic humidifier (evaporative): h stays constant, water added,
    T drops.

    Args:
        s:        input MoistAirState
        x_out:    target humidity ratio [kg/kg]
        phi_out:  target relative humidity [0..1]

    Returns:
        (new_state, ProcessBalance)
    """
    mode, val = _exactly_one('humidify_adiabatic', x_out=x_out, phi_out=phi_out)
    if mode == 'x_out':
        x_new = val
        t_new = (s.h - R_0 * x_new) / (C_PL + C_PW * x_new)
    else:
        # Solve (h_const, phi) → (t_new, x_new) using the existing helper.
        t_new, x_new = _t_x_from_phi_h(val, s.h, s.p)

    if x_new <= s.x:
        raise ValueError(
            f"humidify_adiabatic: target x={x_new:.5g} ≤ inlet x={s.x:.5g}"
        )

    new_s = _state_tx(t_new, x_new, s)
    dh = new_s.h - s.h  # ≈ 0 by construction
    dx = new_s.x - s.x
    water_kgh = None if s.m_dot_dry is None else dx * s.m_dot_dry * 3600.0
    return new_s, ProcessBalance(
        name='humidify_adiabatic',
        dh=dh, dx=dx,
        power_kw=_power_kw(s.m_dot_dry, dh),
        water_kgh=water_kgh,
    )


def mix(s_a, s_b):
    """Adiabatic mixing of two moist-air streams.

    Both inputs must have ``m_dot_dry`` set and matching pressure. The output
    has ``m_dot_dry = s_a.m_dot_dry + s_b.m_dot_dry`` and the mass-weighted
    x and h.

    Returns:
        (mixed_state, ProcessBalance with dh=dx=0)
    """
    if s_a.m_dot_dry is None or s_b.m_dot_dry is None:
        raise ValueError("mix() requires m_dot_dry on both input states")
    if abs(s_a.p - s_b.p) > 1.0:
        raise ValueError(f"mix(): pressures differ ({s_a.p} vs {s_b.p})")
    if s_a.convention != s_b.convention:
        raise ValueError(
            f"mix(): convention mismatch ({s_a.convention!r} vs {s_b.convention!r})"
        )

    m_a = s_a.m_dot_dry
    m_b = s_b.m_dot_dry
    m_total = m_a + m_b
    x_mix = (m_a * s_a.x + m_b * s_b.x) / m_total
    h_mix = (m_a * s_a.h + m_b * s_b.h) / m_total
    t_mix = (h_mix - R_0 * x_mix) / (C_PL + C_PW * x_mix)

    new_s = state(t=t_mix, x=x_mix, p=s_a.p, convention=s_a.convention,
                  m_dot_dry=m_total)

    return new_s, ProcessBalance(name='mix', dh=0.0, dx=0.0)


def heat_recovery(supply, extract, eps_sensible, eps_latent=0.0):
    """Two-stream heat (and optionally moisture) recovery exchanger.

    Returns the new state of the SUPPLY side after recovery. The extract side
    is treated as an input boundary condition; compute it separately if you
    need the exhaust state.

    Args:
        supply:       inlet state of the supply stream
        extract:      inlet state of the extract (return) stream
        eps_sensible: temperature effectiveness [0..1]
        eps_latent:   humidity effectiveness [0..1]; default 0 (sensible-only HX)

    Returns:
        (new_supply_state, ProcessBalance)
    """
    if not 0.0 <= eps_sensible <= 1.0:
        raise ValueError(f"eps_sensible must be in [0, 1]; got {eps_sensible}")
    if not 0.0 <= eps_latent <= 1.0:
        raise ValueError(f"eps_latent must be in [0, 1]; got {eps_latent}")
    if abs(supply.p - extract.p) > 1.0:
        raise ValueError(f"heat_recovery(): pressures differ ({supply.p} vs {extract.p})")

    t_new = supply.t + eps_sensible * (extract.t - supply.t)
    x_new = supply.x + eps_latent * (extract.x - supply.x)

    new_s = _state_tx(t_new, x_new, supply)
    dh = new_s.h - supply.h
    dx = new_s.x - supply.x
    return new_s, ProcessBalance(
        name='heat_recovery',
        dh=dh, dx=dx,
        power_kw=_power_kw(supply.m_dot_dry, dh),
        epsilon=eps_sensible,
    )


def chain_summary(steps):
    """Tabulate a sequence of (label, ProcessBalance) tuples as a DataFrame.

    Args:
        steps: iterable of ``(label, ProcessBalance)`` pairs.

    Returns:
        pandas.DataFrame indexed by ``label`` with columns
        ``[name, dh, dx, power_kw, condensate_kgh, water_kgh]``.
    """
    import pandas as pd  # local import keeps the rest of the module pandas-free

    rows = []
    labels = []
    for label, b in steps:
        labels.append(label)
        rows.append({
            'name': b.name,
            'dh': b.dh,
            'dx': b.dx,
            'power_kw': b.power_kw,
            'condensate_kgh': b.condensate_kgh,
            'water_kgh': b.water_kgh,
        })
    return pd.DataFrame(rows, index=labels)
