import unittest
import math
import numpy as np
import pandas as pd
from pyedautils._mollier import (
    C_PL,
    DEFAULT_CONVENTION,
    MoistAirState,
    P_STD,
    ProcessBalance,
    R_0,
    chain_summary,
    cool,
    create_comfort,
    density,
    dew_point,
    enthalpy,
    get_x_y,
    get_x_y_tx,
    heat,
    heat_recovery,
    humidify_adiabatic,
    humidify_isothermal,
    mix,
    p_sat,
    pressure_from_altitude,
    rel_humidity,
    specific_volume,
    state,
    temperature,
    temperature_p_sat,
    vapor_pressure,
    wet_bulb,
    x_hy,
    x_phiy,
    y_hx,
    y_phix,
    y_rhox,
)
from pyedautils.plots import plot_mollier_hx


class TestSaturationPressure(unittest.TestCase):
    """Tests for p_sat and temperature_p_sat."""

    def test_p_sat_at_0C(self):
        self.assertAlmostEqual(p_sat(0.0), 611.0, delta=1.0)

    def test_p_sat_at_100C(self):
        self.assertAlmostEqual(p_sat(100.0), 101325.0, delta=2000.0)

    def test_p_sat_negative_temperature(self):
        result = p_sat(-10.0)
        self.assertGreater(result, 0)
        self.assertLess(result, 611.0)

    def test_p_sat_vectorised(self):
        temps = np.array([-10.0, 0.0, 20.0, 50.0])
        result = p_sat(temps)
        self.assertEqual(len(result), 4)
        self.assertTrue(np.all(np.diff(result) > 0))

    def test_roundtrip_temperature_p_sat(self):
        for t in [-10.0, 0.0, 15.0, 30.0, 50.0]:
            ps = p_sat(t)
            t_back = temperature_p_sat(ps)
            self.assertAlmostEqual(t, t_back, places=2)

    def test_temperature_p_sat_too_high(self):
        with self.assertRaises(ValueError):
            temperature_p_sat(math.exp(14.2) + 1)


class TestCoordinateFunctions(unittest.TestCase):
    """Tests for enthalpy, temperature, rel_humidity, density."""

    def test_enthalpy(self):
        self.assertAlmostEqual(enthalpy(0, 20), C_PL * 20, places=5)
        self.assertAlmostEqual(enthalpy(0.01, 0), R_0 * 0.01, places=5)

    def test_temperature_at_x0(self):
        for y in [-10, 0, 20, 40]:
            self.assertAlmostEqual(temperature(0, y), y, places=5)

    def test_roundtrip_get_x_y_temperature(self):
        p = 101325.0
        for t, phi in [(20, 0.5), (0, 0.3), (35, 0.8), (-5, 0.9)]:
            xv, yv = get_x_y(t, phi, p)
            t_back = temperature(xv, yv)
            self.assertAlmostEqual(t, t_back, places=2)

    def test_roundtrip_get_x_y_rel_humidity(self):
        p = 101325.0
        for t, phi in [(20, 0.5), (10, 0.3), (30, 0.7)]:
            xv, yv = get_x_y(t, phi, p)
            phi_back = rel_humidity(xv, yv, p)
            self.assertAlmostEqual(phi, phi_back, places=3)

    def test_get_x_y_vectorised(self):
        t = np.array([10.0, 20.0, 30.0])
        phi = np.array([0.3, 0.5, 0.7])
        xs, ys = get_x_y(t, phi, 101325.0)
        self.assertEqual(len(xs), 3)
        self.assertEqual(len(ys), 3)

    def test_density_positive(self):
        p = 101325.0
        rho = density(0.005, 20, p)
        self.assertGreater(rho, 1.0)
        self.assertLess(rho, 1.4)

    def test_get_x_y_tx(self):
        t, x_abs = 25.0, 0.01
        xv, yv = get_x_y_tx(t, x_abs, 101325.0)
        self.assertEqual(xv, x_abs)
        t_back = temperature(xv, yv)
        self.assertAlmostEqual(t, t_back, places=3)


class TestEnthalpyDensityConversions(unittest.TestCase):
    """Tests for x_hy, y_hx, y_rhox, y_phix, x_phiy."""

    def test_x_hy_y_hx_roundtrip(self):
        h, y = 50.0, 20.0
        x = x_hy(h, y)
        y_back = y_hx(h, x)
        self.assertAlmostEqual(y, y_back, places=5)

    def test_y_rhox(self):
        p = 101325.0
        rho = density(0.005, 20, p)
        yv = y_rhox(rho, 0.005, p)
        self.assertAlmostEqual(yv, 20, places=1)

    def test_y_phix(self):
        p = 101325.0
        xv, yv = get_x_y(20, 0.5, p)
        y_back = y_phix(0.5, xv, p)
        self.assertAlmostEqual(yv, y_back, places=2)

    def test_x_phiy(self):
        p = 101325.0
        xv, yv = get_x_y(20, 0.5, p)
        x_back = x_phiy(0.5, yv, p)
        self.assertAlmostEqual(xv, x_back, places=5)


class TestComfortZone(unittest.TestCase):
    """Tests for create_comfort."""

    def test_comfort_returns_closed_polygon(self):
        polygon = create_comfort((20, 26), (0.30, 0.65), (0, 0.0115), 101325.0)
        self.assertGreater(len(polygon), 3)
        self.assertAlmostEqual(polygon[0][0], polygon[-1][0], places=8)
        self.assertAlmostEqual(polygon[0][1], polygon[-1][1], places=8)

    def test_comfort_zero_phi(self):
        polygon = create_comfort((20, 26), (0, 0), (0, 0.01), 101325.0)
        self.assertEqual(len(polygon), 3)

    def test_comfort_points_in_range(self):
        polygon = create_comfort((20, 26), (0.30, 0.65), (0, 0.0115), 101325.0)
        for x, y in polygon:
            self.assertGreaterEqual(x, -0.001)
            self.assertLessEqual(x, 0.02)


class TestComfortZonePhiZeroStart(unittest.TestCase):
    """Test create_comfort with phi_min=0 (exercises the Phi==0 branch)."""

    def test_comfort_phi_starts_at_zero(self):
        polygon = create_comfort((20, 26), (0, 0.65), (0, 0.0115), 101325.0)
        self.assertGreater(len(polygon), 3)
        self.assertAlmostEqual(polygon[0][0], polygon[-1][0], places=8)


class TestScalarInternals(unittest.TestCase):
    """Test scalar internal functions to reach full coverage."""

    def test_rel_humidity_calls_scalar_p_sat(self):
        # rel_humidity calls _p_sat_scalar internally (scalar path)
        phi = rel_humidity(0.005, 20, 101325.0)
        self.assertGreater(phi, 0)
        self.assertLess(phi, 1)

    def test_rel_humidity_below_zero(self):
        # Exercises _p_sat_scalar with t < 0.01
        phi = rel_humidity(0.001, -10, 101325.0)
        self.assertGreater(phi, 0)


class TestConvention(unittest.TestCase):
    """Tests for the configurable Mollier convention ('classical' vs 'glueck')."""

    def test_default_is_classical(self):
        self.assertEqual(DEFAULT_CONVENTION, 'classical')

    def test_invalid_convention_raises(self):
        with self.assertRaises(ValueError):
            temperature(0.005, 20, convention='nonsense')

    def test_classical_isotherm_slopes_up_with_x(self):
        # At T=25°C, x=0: y=25 (exact). At T=25°C, x=20 g/kg: y > 25.
        _, y0 = get_x_y_tx(25, 0.0, 101325, convention='classical')
        _, y1 = get_x_y_tx(25, 0.020, 101325, convention='classical')
        self.assertAlmostEqual(y0, 25.0, places=4)
        self.assertGreater(y1, y0)

    def test_glueck_isotherm_slopes_down_with_x(self):
        # Same input, Glück convention: y at x>0 should be < y at x=0.
        _, y0 = get_x_y_tx(25, 0.0, 101325, convention='glueck')
        _, y1 = get_x_y_tx(25, 0.020, 101325, convention='glueck')
        self.assertAlmostEqual(y0, 25.0, places=4)
        self.assertLess(y1, y0)

    def test_conventions_differ_at_nonzero_x(self):
        _, y_c = get_x_y_tx(25, 0.010, 101325, convention='classical')
        _, y_g = get_x_y_tx(25, 0.010, 101325, convention='glueck')
        self.assertNotAlmostEqual(y_c, y_g, places=2)

    def test_conventions_agree_at_x_zero(self):
        # By construction, at x=0 both conventions give y = T.
        _, y_c = get_x_y_tx(25, 0.0, 101325, convention='classical')
        _, y_g = get_x_y_tx(25, 0.0, 101325, convention='glueck')
        self.assertAlmostEqual(y_c, y_g, places=8)

    def test_roundtrip_both_conventions(self):
        p = 101325.0
        for conv in ('classical', 'glueck'):
            for t, phi in [(20, 0.5), (5, 0.3), (35, 0.8)]:
                xv, yv = get_x_y(t, phi, p, convention=conv)
                t_back = temperature(xv, yv, convention=conv)
                phi_back = rel_humidity(xv, yv, p, convention=conv)
                self.assertAlmostEqual(t, t_back, places=2,
                                       msg=f"{conv}: t round-trip")
                self.assertAlmostEqual(phi, phi_back, places=3,
                                       msg=f"{conv}: phi round-trip")

    def test_roundtrip_y_rhox_both_conventions(self):
        p = 101325.0
        for conv in ('classical', 'glueck'):
            rho = density(0.005, 20, p, convention=conv)
            yv = y_rhox(rho, 0.005, p, convention=conv)
            self.assertAlmostEqual(yv, 20.0, places=1, msg=conv)

    def test_roundtrip_x_phiy_both_conventions(self):
        p = 101325.0
        for conv in ('classical', 'glueck'):
            xv, yv = get_x_y(20, 0.5, p, convention=conv)
            x_back = x_phiy(0.5, yv, p, convention=conv)
            self.assertAlmostEqual(xv, x_back, places=5, msg=conv)

    def test_comfort_zone_both_conventions(self):
        for conv in ('classical', 'glueck'):
            polygon = create_comfort(
                (20, 26), (0.30, 0.65), (0, 0.0115), 101325.0,
                convention=conv,
            )
            self.assertGreater(len(polygon), 3, msg=conv)


class TestPropertyHelpers(unittest.TestCase):
    """Tests for the Phase-1 convention-independent derived properties:
    pressure_from_altitude, vapor_pressure, dew_point, specific_volume, wet_bulb.
    """

    def test_pressure_at_sea_level(self):
        self.assertAlmostEqual(pressure_from_altitude(0), P_STD, places=1)

    def test_pressure_at_450m(self):
        # Matches the default in mollier-hx-card.
        self.assertAlmostEqual(pressure_from_altitude(450), 96035.0, delta=5.0)

    def test_pressure_at_1500m(self):
        # Common Swiss-Alps reference altitude.
        self.assertAlmostEqual(pressure_from_altitude(1500), 84559.0, delta=20.0)

    def test_pressure_monotonic(self):
        # Higher altitude → lower pressure.
        prev = pressure_from_altitude(0)
        for h in [100, 500, 1000, 2000, 5000]:
            p = pressure_from_altitude(h)
            self.assertLess(p, prev, f"non-monotonic at {h} m")
            prev = p

    def test_vapor_pressure_dry_air(self):
        self.assertEqual(vapor_pressure(0, P_STD), 0.0)

    def test_vapor_pressure_known(self):
        # x = 0.01 kg/kg, p = 101325 Pa → p_v ≈ 1602 Pa
        self.assertAlmostEqual(vapor_pressure(0.01, P_STD), 1602.4, delta=1.0)

    def test_dew_point_at_saturation(self):
        # Saturated air at 20 °C has T_dp = 20 °C.
        xv, _ = get_x_y(20.0, 1.0, P_STD)
        self.assertAlmostEqual(dew_point(xv, P_STD), 20.0, places=1)

    def test_dew_point_below_db(self):
        # For unsaturated air, T_dp < T_db.
        xv, _ = get_x_y(25.0, 0.5, P_STD)
        self.assertLess(dew_point(xv, P_STD), 25.0)

    def test_dew_point_known(self):
        # 20 °C, 50 % RH at sea level → T_dp ≈ 9.3 °C
        xv, _ = get_x_y(20.0, 0.5, P_STD)
        self.assertAlmostEqual(dew_point(xv, P_STD), 9.3, delta=0.3)

    def test_specific_volume_dry_air_20C(self):
        # Dry air at 20 °C, sea level → v ≈ 0.831 m³/kg
        self.assertAlmostEqual(specific_volume(20.0, 0.0, P_STD), 0.831, delta=0.005)

    def test_specific_volume_increases_with_temperature(self):
        v1 = specific_volume(0.0, 0.005, P_STD)
        v2 = specific_volume(30.0, 0.005, P_STD)
        self.assertGreater(v2, v1)

    def test_specific_volume_matches_inverse_density(self):
        # v = (1 + x) / rho_moist — cross-check against existing density() at
        # the same state. Pick (x, y) that round-trips to a known (t, x).
        t, phi = 22.0, 0.45
        xv, yv = get_x_y(t, phi, P_STD)
        rho = density(xv, yv, P_STD)
        v_expected = (1 + xv) / rho
        self.assertAlmostEqual(specific_volume(t, xv, P_STD), v_expected, places=5)

    def test_wet_bulb_at_saturation(self):
        # φ = 1.0 → T_wb = T_db.
        xv, _ = get_x_y(15.0, 1.0, P_STD)
        self.assertAlmostEqual(wet_bulb(15.0, xv, P_STD), 15.0, places=1)

    def test_wet_bulb_25C_50RH(self):
        # ASHRAE-table reference: 25 °C / 50 % RH / sea level → T_wb ≈ 17.9 °C
        xv, _ = get_x_y(25.0, 0.5, P_STD)
        self.assertAlmostEqual(wet_bulb(25.0, xv, P_STD), 17.9, delta=0.3)

    def test_wet_bulb_dry_air_large_depression(self):
        # Very dry air → big T_wb depression.
        xv, _ = get_x_y(30.0, 0.1, P_STD)
        twb = wet_bulb(30.0, xv, P_STD)
        self.assertLess(twb, 20.0)  # at least 10 K depression
        self.assertGreater(twb, 5.0)  # but not absurd

    def test_wet_bulb_below_db(self):
        # For any unsaturated state, T_wb < T_db.
        for t, phi in [(15, 0.3), (20, 0.7), (35, 0.4)]:
            xv, _ = get_x_y(t, phi, P_STD)
            self.assertLess(wet_bulb(t, xv, P_STD), t)


class TestStateFactory(unittest.TestCase):
    """Phase-2 tests: the MoistAirState dataclass and the universal
    ``state(...)`` factory.
    """

    # Canonical reference: 25 °C / 50 % RH at sea level. See
    # TestPropertyHelpers.test_wet_bulb_25C_50RH for the T_wb=17.9 °C basis.
    REF = dict(t=25.0, phi=0.5, p=P_STD)
    REF_X = 0.009876
    REF_H = 50.41
    REF_T_WB = 17.9
    REF_T_DP = 13.86

    # ---------------------------------------------------------------------
    # Single-property pairs that involve t — direct (no iteration)
    # ---------------------------------------------------------------------

    def test_from_t_phi(self):
        s = state(t=25.0, phi=0.5, p=P_STD)
        self.assertAlmostEqual(s.x, self.REF_X, places=5)
        self.assertAlmostEqual(s.h, self.REF_H, places=1)
        self.assertAlmostEqual(s.t_wb, self.REF_T_WB, delta=0.3)
        self.assertAlmostEqual(s.t_dp, self.REF_T_DP, delta=0.3)
        self.assertEqual(s.p, P_STD)
        self.assertEqual(s.convention, 'classical')
        self.assertIsNone(s.m_dot_dry)
        self.assertIsNone(s.volume_flow)

    def test_from_t_x(self):
        s = state(t=25.0, x=self.REF_X, p=P_STD)
        self.assertAlmostEqual(s.phi, 0.5, delta=1e-4)
        self.assertAlmostEqual(s.h, self.REF_H, places=1)

    def test_from_t_h(self):
        s = state(t=25.0, h=self.REF_H, p=P_STD)
        self.assertAlmostEqual(s.x, self.REF_X, places=4)
        self.assertAlmostEqual(s.phi, 0.5, delta=1e-3)

    def test_from_t_t_dp(self):
        s = state(t=25.0, t_dp=self.REF_T_DP, p=P_STD)
        self.assertAlmostEqual(s.phi, 0.5, delta=0.02)
        self.assertAlmostEqual(s.x, self.REF_X, delta=2e-4)

    def test_from_t_t_wb(self):
        s = state(t=25.0, t_wb=self.REF_T_WB, p=P_STD)
        self.assertAlmostEqual(s.phi, 0.5, delta=0.02)
        self.assertAlmostEqual(s.x, self.REF_X, delta=2e-4)

    # ---------------------------------------------------------------------
    # Pairs without t — solve for t from the second property
    # ---------------------------------------------------------------------

    def test_from_x_phi(self):
        s = state(x=self.REF_X, phi=0.5, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, places=1)

    def test_from_x_h(self):
        s = state(x=self.REF_X, h=self.REF_H, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, places=1)

    def test_from_x_t_wb(self):
        s = state(x=self.REF_X, t_wb=self.REF_T_WB, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, delta=0.3)

    def test_from_t_dp_phi(self):
        s = state(t_dp=self.REF_T_DP, phi=0.5, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, delta=0.5)

    def test_from_t_dp_h(self):
        s = state(t_dp=self.REF_T_DP, h=self.REF_H, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, delta=0.3)

    def test_from_t_dp_t_wb(self):
        s = state(t_dp=self.REF_T_DP, t_wb=self.REF_T_WB, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, delta=0.5)

    # ---------------------------------------------------------------------
    # Fully iterative pairs (no t, no x)
    # ---------------------------------------------------------------------

    def test_from_phi_h(self):
        s = state(phi=0.5, h=self.REF_H, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, delta=0.3)
        self.assertAlmostEqual(s.x, self.REF_X, delta=2e-4)

    def test_from_phi_t_wb(self):
        s = state(phi=0.5, t_wb=self.REF_T_WB, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, delta=0.5)

    def test_from_h_t_wb(self):
        s = state(h=self.REF_H, t_wb=self.REF_T_WB, p=P_STD)
        self.assertAlmostEqual(s.t, 25.0, delta=0.5)

    # ---------------------------------------------------------------------
    # Pressure and altitude
    # ---------------------------------------------------------------------

    def test_pressure_defaults_to_sea_level(self):
        s = state(t=20.0, phi=0.5)
        self.assertEqual(s.p, P_STD)

    def test_altitude_overrides_default(self):
        s = state(t=20.0, phi=0.5, altitude=450.0)
        self.assertAlmostEqual(s.p, 96035.0, delta=5.0)

    def test_p_and_altitude_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            state(t=20.0, phi=0.5, p=P_STD, altitude=450.0)

    # ---------------------------------------------------------------------
    # Mass and volume flow
    # ---------------------------------------------------------------------

    def test_volume_flow_resolves_m_dot_dry(self):
        s = state(t=25.0, phi=0.5, p=P_STD, volume_flow=1500.0)
        expected_m_dot = 1500.0 / 3600.0 / s.v
        self.assertAlmostEqual(s.m_dot_dry, expected_m_dot, places=5)
        self.assertEqual(s.volume_flow, 1500.0)

    def test_m_dot_dry_passes_through(self):
        s = state(t=25.0, phi=0.5, p=P_STD, m_dot_dry=0.5)
        self.assertEqual(s.m_dot_dry, 0.5)
        self.assertIsNone(s.volume_flow)

    def test_m_dot_and_volume_flow_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            state(t=25.0, phi=0.5, p=P_STD, m_dot_dry=0.5, volume_flow=1500.0)

    # ---------------------------------------------------------------------
    # Convention
    # ---------------------------------------------------------------------

    def test_convention_changes_only_y(self):
        # Physical properties identical, only the diagram y-coord differs.
        sc = state(t=25.0, phi=0.5, p=P_STD, convention='classical')
        sg = state(t=25.0, phi=0.5, p=P_STD, convention='glueck')
        for attr in ('t', 'phi', 'x', 'h', 't_wb', 't_dp', 'p_v', 'rho', 'v'):
            self.assertAlmostEqual(getattr(sc, attr), getattr(sg, attr), places=8,
                                   msg=f"{attr} differs between conventions")
        self.assertNotAlmostEqual(sc.y, sg.y, places=2)

    def test_invalid_convention_raises(self):
        with self.assertRaises(ValueError):
            state(t=20.0, phi=0.5, convention='nonsense')

    # ---------------------------------------------------------------------
    # Error handling
    # ---------------------------------------------------------------------

    def test_requires_exactly_two_properties(self):
        with self.assertRaises(ValueError):
            state(t=25.0, p=P_STD)
        with self.assertRaises(ValueError):
            state(t=25.0, phi=0.5, x=0.01, p=P_STD)
        with self.assertRaises(ValueError):
            state(p=P_STD)

    def test_x_t_dp_degenerate(self):
        with self.assertRaises(ValueError):
            state(x=0.01, t_dp=12.0, p=P_STD)

    # ---------------------------------------------------------------------
    # Immutability
    # ---------------------------------------------------------------------

    def test_state_is_frozen(self):
        s = state(t=20.0, phi=0.5, p=P_STD)
        with self.assertRaises(Exception):  # FrozenInstanceError
            s.t = 30.0  # type: ignore[misc]

    def test_state_is_dataclass(self):
        s = state(t=20.0, phi=0.5, p=P_STD)
        self.assertIsInstance(s, MoistAirState)


class TestProcesses(unittest.TestCase):
    """Phase-3 tests: heat, cool, humidify (iso/adiabatic), mix, heat_recovery,
    chain_summary.
    """

    P = P_STD

    # ---------------------------------------------------------------------
    # heat
    # ---------------------------------------------------------------------

    def test_heat_preserves_x(self):
        s0 = state(t=15.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        s1, b = heat(s0, dt=10.0)
        self.assertIsInstance(b, ProcessBalance)
        self.assertEqual(b.name, 'heat')
        self.assertAlmostEqual(s1.t, 25.0, places=2)
        self.assertAlmostEqual(s1.x, s0.x, places=8)  # x preserved
        self.assertEqual(s1.m_dot_dry, 0.5)
        self.assertGreater(b.dh, 0)
        self.assertEqual(b.dx, 0.0)

    def test_heat_t_out_mode(self):
        s0 = state(t=15.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        s1, _ = heat(s0, t_out=22.0)
        self.assertAlmostEqual(s1.t, 22.0, places=2)

    def test_heat_power_kw_mode(self):
        s0 = state(t=15.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        # 5 kW into 0.5 kg/s = 10 kJ/kg added
        s1, b = heat(s0, power_kw=5.0)
        self.assertAlmostEqual(b.power_kw, 5.0, places=2)
        self.assertAlmostEqual(b.dh, 10.0, delta=0.1)

    def test_heat_power_kw_requires_mass_flow(self):
        s0 = state(t=15.0, phi=0.5, p=self.P)
        with self.assertRaises(ValueError):
            heat(s0, power_kw=5.0)

    def test_heat_requires_exactly_one_spec(self):
        s0 = state(t=15.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        with self.assertRaises(ValueError):
            heat(s0)
        with self.assertRaises(ValueError):
            heat(s0, dt=5, t_out=22)

    # ---------------------------------------------------------------------
    # cool
    # ---------------------------------------------------------------------

    def test_cool_sensible_above_dew_point(self):
        # Inlet 30 °C / 40 % → t_dp ≈ 14 °C. Cool to 20 °C: sensible only.
        s0 = state(t=30.0, phi=0.4, p=self.P, m_dot_dry=0.5)
        s1, b = cool(s0, t_out=20.0)
        self.assertAlmostEqual(s1.x, s0.x, places=8)
        self.assertEqual(b.dx, 0.0)
        self.assertIsNone(b.condensate_kgh)
        self.assertLess(b.power_kw, 0)  # cooling consumes negative power

    def test_cool_latent_below_dew_point(self):
        # Inlet 32 °C / 65 % → t_dp ≈ 24 °C. Cool to 14 °C: condenses.
        s0 = state(t=32.0, phi=0.65, p=self.P, m_dot_dry=0.5)
        s1, b = cool(s0, t_out=14.0)
        self.assertAlmostEqual(s1.t, 14.0, places=2)
        self.assertAlmostEqual(s1.phi, 1.0, places=2)  # saturated outlet
        self.assertLess(s1.x, s0.x)  # water removed
        self.assertGreater(b.condensate_kgh, 0)

    def test_cool_target_above_inlet_raises(self):
        s0 = state(t=20.0, phi=0.5, p=self.P)
        with self.assertRaises(ValueError):
            cool(s0, t_out=22.0)

    def test_cool_dt_mode(self):
        s0 = state(t=25.0, phi=0.4, p=self.P)
        s1, _ = cool(s0, dt=-5.0)
        self.assertAlmostEqual(s1.t, 20.0, places=2)

    # ---------------------------------------------------------------------
    # humidify_isothermal
    # ---------------------------------------------------------------------

    def test_humidify_iso_keeps_temperature(self):
        s0 = state(t=21.0, phi=0.20, p=self.P, m_dot_dry=0.5)
        s1, b = humidify_isothermal(s0, phi_out=0.45)
        self.assertAlmostEqual(s1.t, 21.0, places=2)
        self.assertAlmostEqual(s1.phi, 0.45, delta=1e-3)
        self.assertGreater(b.water_kgh, 0)
        self.assertGreater(b.power_kw, 0)  # steam adds energy

    def test_humidify_iso_x_out_mode(self):
        s0 = state(t=21.0, phi=0.20, p=self.P)
        x_target = 0.008
        s1, _ = humidify_isothermal(s0, x_out=x_target)
        self.assertAlmostEqual(s1.x, x_target, places=6)
        self.assertAlmostEqual(s1.t, 21.0, places=2)

    def test_humidify_iso_drying_raises(self):
        s0 = state(t=21.0, phi=0.80, p=self.P)
        with self.assertRaises(ValueError):
            humidify_isothermal(s0, phi_out=0.20)

    # ---------------------------------------------------------------------
    # humidify_adiabatic
    # ---------------------------------------------------------------------

    def test_humidify_adiabatic_preserves_enthalpy(self):
        s0 = state(t=21.0, phi=0.20, p=self.P, m_dot_dry=0.5)
        s1, b = humidify_adiabatic(s0, phi_out=0.45)
        self.assertAlmostEqual(s1.h, s0.h, delta=0.05)  # adiabatic
        self.assertLess(s1.t, s0.t)  # evaporative cooling
        self.assertAlmostEqual(s1.phi, 0.45, delta=1e-3)
        self.assertGreater(b.water_kgh, 0)
        self.assertAlmostEqual(b.power_kw, 0.0, delta=0.05)

    def test_humidify_adiabatic_x_out_mode(self):
        s0 = state(t=25.0, phi=0.30, p=self.P)
        x_target = s0.x * 1.5
        s1, _ = humidify_adiabatic(s0, x_out=x_target)
        self.assertAlmostEqual(s1.x, x_target, places=6)
        self.assertAlmostEqual(s1.h, s0.h, delta=0.05)

    # ---------------------------------------------------------------------
    # mix
    # ---------------------------------------------------------------------

    def test_mix_equal_streams_averages(self):
        s_a = state(t=10.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        s_b = state(t=30.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        s_mix, b = mix(s_a, s_b)
        # Mass-flow-weighted average — equal masses → arithmetic mean
        self.assertAlmostEqual(s_mix.x, 0.5 * (s_a.x + s_b.x), places=8)
        self.assertAlmostEqual(s_mix.h, 0.5 * (s_a.h + s_b.h), places=4)
        self.assertAlmostEqual(s_mix.m_dot_dry, 1.0, places=8)
        self.assertEqual(b.name, 'mix')

    def test_mix_weighted_by_mass_flow(self):
        # 1:3 ratio
        s_a = state(t=10.0, phi=0.5, p=self.P, m_dot_dry=0.25)
        s_b = state(t=30.0, phi=0.5, p=self.P, m_dot_dry=0.75)
        s_mix, _ = mix(s_a, s_b)
        h_expected = 0.25 * s_a.h + 0.75 * s_b.h
        self.assertAlmostEqual(s_mix.h, h_expected, places=4)

    def test_mix_requires_mass_flow(self):
        s_a = state(t=10.0, phi=0.5, p=self.P)  # no m_dot
        s_b = state(t=30.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        with self.assertRaises(ValueError):
            mix(s_a, s_b)

    def test_mix_pressure_mismatch_raises(self):
        s_a = state(t=10.0, phi=0.5, p=self.P, m_dot_dry=0.5)
        s_b = state(t=30.0, phi=0.5, p=self.P - 5000, m_dot_dry=0.5)
        with self.assertRaises(ValueError):
            mix(s_a, s_b)

    # ---------------------------------------------------------------------
    # heat_recovery
    # ---------------------------------------------------------------------

    def test_heat_recovery_sensible_only(self):
        supply = state(t=0.0, phi=0.8, p=self.P, m_dot_dry=0.5)
        extract = state(t=22.0, phi=0.4, p=self.P, m_dot_dry=0.5)
        s1, b = heat_recovery(supply, extract, eps_sensible=0.5)
        self.assertAlmostEqual(s1.t, 11.0, places=2)  # 0 + 0.5*(22-0)
        self.assertAlmostEqual(s1.x, supply.x, places=8)  # no latent
        self.assertEqual(b.epsilon, 0.5)
        self.assertGreater(b.power_kw, 0)

    def test_heat_recovery_with_latent(self):
        supply = state(t=0.0, phi=0.8, p=self.P, m_dot_dry=0.5)
        extract = state(t=22.0, phi=0.4, p=self.P, m_dot_dry=0.5)
        s1, _ = heat_recovery(supply, extract, eps_sensible=0.6, eps_latent=0.6)
        # x_supply increases toward x_extract
        self.assertGreater(s1.x, supply.x)
        self.assertLess(s1.x, extract.x)

    def test_heat_recovery_eps_out_of_range_raises(self):
        s = state(t=20.0, phi=0.5, p=self.P)
        with self.assertRaises(ValueError):
            heat_recovery(s, s, eps_sensible=1.5)
        with self.assertRaises(ValueError):
            heat_recovery(s, s, eps_sensible=0.5, eps_latent=-0.1)

    def test_heat_recovery_pressure_mismatch_raises(self):
        s_a = state(t=20.0, phi=0.5, p=self.P)
        s_b = state(t=22.0, phi=0.4, p=self.P - 5000)
        with self.assertRaises(ValueError):
            heat_recovery(s_a, s_b, eps_sensible=0.7)

    def test_mix_convention_mismatch_raises(self):
        s_a = state(t=20.0, phi=0.5, p=self.P, m_dot_dry=0.5,
                    convention='classical')
        s_b = state(t=22.0, phi=0.4, p=self.P, m_dot_dry=0.5,
                    convention='glueck')
        with self.assertRaises(ValueError):
            mix(s_a, s_b)

    def test_humidify_isothermal_phi_unreachable_raises(self):
        # At 101 °C, p_sat exceeds atmospheric → high phi unreachable.
        s = state(t=101.0, x=0.05, p=self.P)
        with self.assertRaises(ValueError):
            humidify_isothermal(s, phi_out=0.99)

    def test_humidify_adiabatic_drying_phi_raises(self):
        # Already humid air, asking for lower phi via adiabatic process.
        s = state(t=15.0, phi=0.95, p=self.P)
        with self.assertRaises(ValueError):
            humidify_adiabatic(s, phi_out=0.30)

    # ---------------------------------------------------------------------
    # chain_summary
    # ---------------------------------------------------------------------

    def test_chain_summary_returns_dataframe(self):
        import pandas as pd
        s0 = state(t=-5.0, phi=0.8, p=self.P, m_dot_dry=0.5)
        s_extract = state(t=22.0, phi=0.4, p=self.P, m_dot_dry=0.5)
        s1, b1 = heat_recovery(s0, s_extract, eps_sensible=0.7)
        s2, b2 = heat(s1, t_out=21.0)
        s3, b3 = humidify_adiabatic(s2, phi_out=0.45)

        df = chain_summary([('WRG', b1), ('Heizung', b2), ('Befeuchter', b3)])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.index), ['WRG', 'Heizung', 'Befeuchter'])
        self.assertIn('power_kw', df.columns)
        self.assertIn('water_kgh', df.columns)
        self.assertIn('condensate_kgh', df.columns)
        # Befeuchter row has water but no condensate
        self.assertGreater(df.loc['Befeuchter', 'water_kgh'], 0)

    # ---------------------------------------------------------------------
    # Mass-flow propagation through single-stream processes
    # ---------------------------------------------------------------------

    def test_m_dot_propagates_through_single_stream_processes(self):
        s = state(t=20.0, phi=0.5, p=self.P, m_dot_dry=0.7)
        for new_s, _ in (
            heat(s, dt=5.0),
            humidify_isothermal(s, phi_out=0.7),
            humidify_adiabatic(s, x_out=s.x * 1.5),
        ):
            self.assertEqual(new_s.m_dot_dry, 0.7)


class TestExtremeGuards(unittest.TestCase):
    """Defensive error branches that protect against physically impossible
    inputs (very high T where p_sat exceeds total pressure).
    """

    def test_wet_bulb_above_boiling_at_atm_raises(self):
        # T_db so high that the first iteration step lands on T where
        # p_sat(T) >= 101325 Pa.
        xv, _ = get_x_y(100.0, 0.5, P_STD)
        with self.assertRaises(ValueError):
            wet_bulb(200.0, xv, P_STD)

    def test_state_t_wb_above_boiling_raises(self):
        # (t, t_wb) where p_sat(t_wb) >= p.
        with self.assertRaises(ValueError):
            state(t=120.0, t_wb=110.0, p=P_STD)

    def test_state_x_and_t_wb_above_boiling_raises(self):
        with self.assertRaises(ValueError):
            state(x=0.01, t_wb=110.0, p=P_STD)

    def test_state_t_dp_above_boiling_raises(self):
        # T_dp where p_sat(T_dp) >= p.
        with self.assertRaises(ValueError):
            state(t=120.0, t_dp=110.0, p=P_STD)


class TestPlotMollierHx(unittest.TestCase):
    """Tests for plot_mollier_hx (D3 HTML output)."""

    def test_basic_no_data(self):
        html = plot_mollier_hx()
        self.assertIsInstance(html, str)
        self.assertIn("d3.v5.min.js", html)
        self.assertIn("drawHXCoordinates", html)
        self.assertIn("createComfort", html)

    def test_returns_div_fragment(self):
        html = plot_mollier_hx()
        self.assertIn("<div id=", html)
        self.assertNotIn("<html>", html)

    def test_custom_pressure(self):
        html = plot_mollier_hx(pressure=95000.0)
        self.assertIn("95000", html)

    def test_custom_comfort_zone(self):
        html = plot_mollier_hx(comfort_zone={
            "temperature": (18, 24),
            "rel_humidity": (0.20, 0.70),
            "abs_humidity": (0, 0.012),
        })
        self.assertIn("[18, 24]", html)

    def test_with_synthetic_data_all_seasons(self):
        # Full year to cover all 4 seasons in _get_season_fast
        timestamps = pd.date_range("2023-01-01", periods=365 * 24, freq="h")
        np.random.seed(42)
        n = len(timestamps)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "humidity": np.random.uniform(30, 70, n),
            "temperature": np.random.uniform(15, 30, n),
        })
        html = plot_mollier_hx(data=df)
        self.assertIn("dataRecords", html)
        for season in ["Winter", "Frühling", "Sommer", "Herbst"]:
            self.assertIn(season, html)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "humidity", "temperature"])
        html = plot_mollier_hx(data=df)
        self.assertIn("dataRecords = null", html)

    def test_data_with_nan(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=10, freq="h"),
            "humidity": [50, np.nan, 60, 55, 50, np.nan, 45, 50, 55, 60],
            "temperature": [22, 23, np.nan, 24, 25, 22, 23, 24, 25, 26],
        })
        html = plot_mollier_hx(data=df)
        self.assertIsInstance(html, str)

    def test_no_comfort_zone(self):
        html = plot_mollier_hx(comfort_zone=False)
        self.assertIn("[0,0]", html)

    def test_custom_height(self):
        html = plot_mollier_hx(height=500)
        self.assertIn("500", html)

    def test_convention_classical_default(self):
        html = plot_mollier_hx()
        self.assertIn('"classical"', html)

    def test_convention_glueck(self):
        html = plot_mollier_hx(convention='glueck')
        self.assertIn('"glueck"', html)

    def test_convention_invalid_raises(self):
        with self.assertRaises(ValueError):
            plot_mollier_hx(convention='nonsense')

    # ---------------------------------------------------------------------
    # Phase 4: process-chain visualization (states= kwarg)
    # ---------------------------------------------------------------------

    def test_states_renders_process_chain(self):
        from pyedautils._mollier import heat, humidify_adiabatic
        s0 = state(t=-5, phi=0.80, p=P_STD, volume_flow=1500)
        s1, _ = heat(s0, t_out=21)
        s2, _ = humidify_adiabatic(s1, phi_out=0.45)
        html = plot_mollier_hx(states=[s0, s1, s2])
        self.assertIn('process-chain', html)
        self.assertIn('arrow-', html)
        # statePoints JSON must contain three records
        self.assertIn('statePoints', html)
        self.assertNotIn('statePoints = null', html)

    def test_states_with_labels(self):
        from pyedautils._mollier import heat
        s0 = state(t=10, phi=0.5, p=P_STD)
        s1, _ = heat(s0, dt=10)
        html = plot_mollier_hx(states=[s0, s1], labels=['Inlet', 'After heater'])
        self.assertIn('Inlet', html)
        self.assertIn('After heater', html)

    def test_states_default_labels_start_at_1(self):
        from pyedautils._mollier import heat
        s0 = state(t=10, phi=0.5, p=P_STD)
        s1, _ = heat(s0, dt=10)
        html = plot_mollier_hx(states=[s0, s1])
        # Default labels match the chart circle numbers: "1" and "2".
        self.assertIn('"number": "1"', html)
        self.assertIn('"number": "2"', html)
        self.assertNotIn('"number": "0"', html)

    def test_states_circle_uses_number_not_label(self):
        # The chart-circle text uses d.number; the legend uses d.label.
        from pyedautils._mollier import heat
        s0 = state(t=10, phi=0.5, p=P_STD)
        s1, _ = heat(s0, dt=10)
        html = plot_mollier_hx(states=[s0, s1], labels=['Inlet', 'Outlet'])
        # State legend is enabled only when explicit labels are provided.
        self.assertIn('showStateLegend = true', html)

    def test_states_default_labels_skip_legend(self):
        # Without explicit labels, the process legend is suppressed (labels
        # would just duplicate the circle numbers).
        from pyedautils._mollier import heat
        s0 = state(t=10, phi=0.5, p=P_STD)
        s1, _ = heat(s0, dt=10)
        html = plot_mollier_hx(states=[s0, s1])
        self.assertIn('showStateLegend = false', html)

    def test_states_labels_length_mismatch_raises(self):
        s0 = state(t=10, phi=0.5, p=P_STD)
        with self.assertRaises(ValueError):
            plot_mollier_hx(states=[s0], labels=['a', 'b'])

    def test_states_none_means_no_chain(self):
        html = plot_mollier_hx()
        self.assertIn('statePoints = null', html)

    def test_states_empty_list_means_no_chain(self):
        html = plot_mollier_hx(states=[])
        self.assertIn('statePoints = null', html)

    def test_states_coexist_with_data(self):
        import pandas as pd
        from pyedautils._mollier import heat
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=3, freq="h"),
            "humidity": [50, 60, 55],
            "temperature": [22, 24, 23],
        })
        s0 = state(t=22, phi=0.5, p=P_STD)
        s1, _ = heat(s0, dt=5)
        html = plot_mollier_hx(data=df, states=[s0, s1])
        # Both visualizations active
        self.assertIn('process-chain', html)
        self.assertIn('dataRecords', html)
        self.assertNotIn('dataRecords = null', html)

    def test_highlight_latest_default_with_data(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=5, freq="h"),
            "humidity": [50, 55, 60, 45, 50],
            "temperature": [22, 23, 24, 22, 25],
        })
        html = plot_mollier_hx(data=df)
        self.assertIn("current-point", html)
        # currentRecord JSON should not be null when data is given
        self.assertNotIn("currentRecord = null", html)

    def test_highlight_latest_disabled(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=3, freq="h"),
            "humidity": [50, 60, 55],
            "temperature": [22, 24, 23],
        })
        html = plot_mollier_hx(data=df, highlight_latest=False)
        self.assertIn("currentRecord = null", html)

    def test_highlight_latest_no_data(self):
        # Without data, no current point regardless of the flag.
        html = plot_mollier_hx()
        self.assertIn("currentRecord = null", html)

    def test_highlight_color_default_black(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=3, freq="h"),
            "humidity": [50, 60, 55],
            "temperature": [22, 24, 23],
        })
        html = plot_mollier_hx(data=df)
        self.assertIn('highlightColor = "black"', html)

    def test_highlight_color_custom(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=3, freq="h"),
            "humidity": [50, 60, 55],
            "temperature": [22, 24, 23],
        })
        html = plot_mollier_hx(data=df, highlight_color='red')
        self.assertIn('highlightColor = "red"', html)

    def test_highlight_color_none_uses_season(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=3, freq="h"),
            "humidity": [50, 60, 55],
            "temperature": [22, 24, 23],
        })
        html = plot_mollier_hx(data=df, highlight_color=None)
        self.assertIn('highlightColor = null', html)

    def test_highlight_latest_picks_max_timestamp(self):
        # Out-of-order rows: the row with the newest timestamp should be
        # selected, not df.iloc[-1].
        df = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2023-06-05 10:00"),  # newest
                pd.Timestamp("2023-06-01 08:00"),
                pd.Timestamp("2023-06-03 12:00"),
            ],
            "humidity": [42.0, 99.0, 99.0],
            "temperature": [11.0, 99.0, 99.0],
        })
        html = plot_mollier_hx(data=df)
        # The newest row had humidity=42, temperature=11 — its phi (rounded
        # percent) must appear in the currentRecord JSON.
        # The phi% value at T=11°C, RH=42% is roughly 42, so look for the
        # serialised "temp": 11.0 marker which is unique to that row.
        self.assertRegex(html, r'currentRecord = \{[^}]*"temp": 11')


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
