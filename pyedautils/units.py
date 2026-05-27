"""SI ↔ IP unit conversion helpers for psychrometric applications.

Pure conversion functions, no state, no dependencies beyond stdlib. Conversion
factors come from NIST SP 811 (2008). Each pair (X_to_Y / Y_to_X) round-trips
to better than 1 part in 10⁶.

Conventions:
    SI  — °C, kg/kg, kJ/kg dry air, Pa, kg/s, m³/h, m, kg/m³, m³/kg
    IP  — °F, gr/lb (grains per pound), BTU/lb dry air, psi/inHg, lb/h,
          CFM, ft, lb/ft³, ft³/lb

Use :func:`state_to_ip` to convert an entire ``MoistAirState`` to a dict
of IP-unit values in one shot.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from pyedautils._mollier import MoistAirState


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def c_to_f(c: float) -> float:
    """Celsius → Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    """Fahrenheit → Celsius."""
    return (f - 32.0) * 5.0 / 9.0


def c_to_k(c: float) -> float:
    """Celsius → Kelvin."""
    return c + 273.15


def k_to_c(k: float) -> float:
    """Kelvin → Celsius."""
    return k - 273.15


# ---------------------------------------------------------------------------
# Humidity ratio (absolute humidity)
# ---------------------------------------------------------------------------

def kgkg_to_grlb(x: float) -> float:
    """kg water / kg dry air → grains water / lb dry air (mass ratio).

    1 lb = 7000 grains; the mass ratio is dimensionless so the factor is 7000.
    """
    return x * 7000.0


def grlb_to_kgkg(x: float) -> float:
    """grains water / lb dry air → kg water / kg dry air."""
    return x / 7000.0


def kgkg_to_gkg(x: float) -> float:
    """kg / kg → g / kg (convenience)."""
    return x * 1000.0


def gkg_to_kgkg(x: float) -> float:
    """g / kg → kg / kg (convenience)."""
    return x / 1000.0


# ---------------------------------------------------------------------------
# Mass flow
# ---------------------------------------------------------------------------

_KG_PER_LB = 0.45359237  # exact (NIST)


def kgs_to_lbh(m: float) -> float:
    """kg/s → lb/h."""
    return m * 3600.0 / _KG_PER_LB


def lbh_to_kgs(m: float) -> float:
    """lb/h → kg/s."""
    return m / 3600.0 * _KG_PER_LB


def kgs_to_kgh(m: float) -> float:
    """kg/s → kg/h."""
    return m * 3600.0


def kgh_to_kgs(m: float) -> float:
    """kg/h → kg/s."""
    return m / 3600.0


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

_KW_PER_BTUH = 0.000293071  # 1 BTU/h = 0.293071 W


def kw_to_btuh(p: float) -> float:
    """kW → BTU/h."""
    return p / _KW_PER_BTUH


def btuh_to_kw(p: float) -> float:
    """BTU/h → kW."""
    return p * _KW_PER_BTUH


# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------

_PA_PER_PSI = 6894.757293168    # NIST
_PA_PER_INHG = 3386.389         # NIST (at 60 °F)


def pa_to_kpa(p: float) -> float:
    """Pa → kPa."""
    return p * 0.001


def kpa_to_pa(p: float) -> float:
    """kPa → Pa."""
    return p * 1000.0


def pa_to_psi(p: float) -> float:
    """Pa → psi."""
    return p / _PA_PER_PSI


def psi_to_pa(p: float) -> float:
    """psi → Pa."""
    return p * _PA_PER_PSI


def pa_to_inhg(p: float) -> float:
    """Pa → inches of mercury (at 60 °F)."""
    return p / _PA_PER_INHG


def inhg_to_pa(p: float) -> float:
    """inches of mercury (at 60 °F) → Pa."""
    return p * _PA_PER_INHG


# ---------------------------------------------------------------------------
# Volume flow
# ---------------------------------------------------------------------------

_M3H_PER_CFM = 1.69901082432    # 1 CFM = 1.69901 m³/h


def m3h_to_cfm(v: float) -> float:
    """m³/h → CFM (ft³/min)."""
    return v / _M3H_PER_CFM


def cfm_to_m3h(v: float) -> float:
    """CFM (ft³/min) → m³/h."""
    return v * _M3H_PER_CFM


# ---------------------------------------------------------------------------
# Length (for altitude conversion)
# ---------------------------------------------------------------------------

_M_PER_FT = 0.3048  # exact (NIST)


def m_to_ft(m: float) -> float:
    """metre → foot."""
    return m / _M_PER_FT


def ft_to_m(ft: float) -> float:
    """foot → metre."""
    return ft * _M_PER_FT


# ---------------------------------------------------------------------------
# Specific enthalpy
# ---------------------------------------------------------------------------

_KJKG_PER_BTULB = 2.326   # 1 BTU/lb = 2.326 kJ/kg (NIST)


def kjkg_to_btulb(h: float) -> float:
    """kJ/kg → BTU/lb."""
    return h / _KJKG_PER_BTULB


def btulb_to_kjkg(h: float) -> float:
    """BTU/lb → kJ/kg."""
    return h * _KJKG_PER_BTULB


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

_KGM3_PER_LBFT3 = 16.01846337  # 1 lb/ft³ = 16.01846 kg/m³


def kgm3_to_lbft3(rho: float) -> float:
    """kg/m³ → lb/ft³."""
    return rho / _KGM3_PER_LBFT3


def lbft3_to_kgm3(rho: float) -> float:
    """lb/ft³ → kg/m³."""
    return rho * _KGM3_PER_LBFT3


# ---------------------------------------------------------------------------
# Specific volume
# ---------------------------------------------------------------------------

def m3kg_to_ft3lb(v: float) -> float:
    """m³/kg → ft³/lb (reciprocal of density conversion)."""
    return v * _KGM3_PER_LBFT3


def ft3lb_to_m3kg(v: float) -> float:
    """ft³/lb → m³/kg."""
    return v / _KGM3_PER_LBFT3


# ---------------------------------------------------------------------------
# Convenience: full MoistAirState to IP-unit dict
# ---------------------------------------------------------------------------

def state_to_ip(s: "MoistAirState") -> dict:
    """Convert a :class:`MoistAirState` (SI) to a dict of IP-unit values.

    The returned dict has explicit unit suffixes in the keys so it can be
    handed directly to a UI/template without losing context.
    """
    def _opt(value, conv):
        return None if value is None else conv(value)

    return {
        't_F':            c_to_f(s.t),
        't_wb_F':         c_to_f(s.t_wb),
        't_dp_F':         c_to_f(s.t_dp),
        'phi_pct':        s.phi * 100.0,
        'x_grlb':         kgkg_to_grlb(s.x),
        'h_btulb':        kjkg_to_btulb(s.h),
        'p_v_inhg':       pa_to_inhg(s.p_v),
        'rho_lbft3':      kgm3_to_lbft3(s.rho),
        'v_ft3lb':        m3kg_to_ft3lb(s.v),
        'p_psi':          pa_to_psi(s.p),
        'm_dot_dry_lbh':  _opt(s.m_dot_dry, kgs_to_lbh),
        'volume_flow_cfm': _opt(s.volume_flow, m3h_to_cfm),
        'convention':     s.convention,
    }
