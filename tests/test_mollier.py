import unittest
import math
import numpy as np
import pandas as pd
from pyedautils._mollier import (
    C_PL,
    DEFAULT_CONVENTION,
    R_0,
    create_comfort,
    density,
    enthalpy,
    get_x_y,
    get_x_y_tx,
    p_sat,
    rel_humidity,
    temperature,
    temperature_p_sat,
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
