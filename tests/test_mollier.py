import unittest
import math
import numpy as np
import pandas as pd
from pyedautils._mollier import (
    C_PL,
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


class TestPlotMollierHx(unittest.TestCase):
    """Tests for plot_mollier_hx (D3 HTML output)."""

    def test_basic_no_data(self):
        html = plot_mollier_hx()
        self.assertIsInstance(html, str)
        self.assertIn("d3.v5.min.js", html)
        self.assertIn("drawHXCoordinates", html)
        self.assertIn("createComfort", html)

    def test_title_not_in_d3(self):
        # D3 version doesn't have a title parameter, just check it returns HTML
        html = plot_mollier_hx()
        self.assertIn("<html>", html)

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

    def test_with_synthetic_data(self):
        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "humidity": np.random.uniform(30, 70, n),
            "temperature": np.random.uniform(15, 30, n),
        })
        html = plot_mollier_hx(data=df)
        self.assertIn("dataRecords", html)
        # Should contain season labels in German
        self.assertIn("Winter", html)

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

    def test_custom_height(self):
        html = plot_mollier_hx(height=500)
        self.assertIn("500", html)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
