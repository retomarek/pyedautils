import unittest
import math
import numpy as np
import pandas as pd
from unittest.mock import patch

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


def _fast_get_season(date, **kwargs):
    """Fast mock for get_season that skips ephem calculations."""
    if isinstance(date, pd.Series):
        return date.apply(lambda x: _fast_get_season(x))
    month = date.month
    if month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    elif month in (9, 10, 11):
        return "Fall"
    else:
        return "Winter"


class TestSaturationPressure(unittest.TestCase):
    """Tests for p_sat and temperature_p_sat."""

    def test_p_sat_at_0C(self):
        # At 0°C, saturation pressure ≈ 611 Pa
        self.assertAlmostEqual(p_sat(0.0), 611.0, delta=1.0)

    def test_p_sat_at_100C(self):
        # At 100°C, saturation pressure ≈ 101325 Pa
        self.assertAlmostEqual(p_sat(100.0), 101325.0, delta=2000.0)

    def test_p_sat_negative_temperature(self):
        # Should work below 0°C (ice region)
        result = p_sat(-10.0)
        self.assertGreater(result, 0)
        self.assertLess(result, 611.0)

    def test_p_sat_vectorised(self):
        temps = np.array([-10.0, 0.0, 20.0, 50.0])
        result = p_sat(temps)
        self.assertEqual(len(result), 4)
        # Monotonically increasing
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
        # At x=0, temperature(0, y) = y
        for y in [-10, 0, 20, 40]:
            self.assertAlmostEqual(temperature(0, y), y, places=5)

    def test_roundtrip_get_x_y_temperature(self):
        """get_x_y followed by temperature should return the original T."""
        p = 101325.0
        for t, phi in [(20, 0.5), (0, 0.3), (35, 0.8), (-5, 0.9)]:
            xv, yv = get_x_y(t, phi, p)
            t_back = temperature(xv, yv)
            self.assertAlmostEqual(t, t_back, places=2)

    def test_roundtrip_get_x_y_rel_humidity(self):
        """get_x_y followed by rel_humidity should return the original phi."""
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
        # First and last point should be the same (closed)
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
    """Tests for plot_mollier_hx."""

    def test_basic_no_data(self):
        import plotly.graph_objects as go
        fig = plot_mollier_hx()
        self.assertIsInstance(fig, go.Figure)
        # Should have traces (iso-lines + comfort zone)
        self.assertGreater(len(fig.data), 5)

    def test_title(self):
        fig = plot_mollier_hx(title="Custom Title")
        self.assertIn("Custom Title", fig.layout.title.text)

    def test_custom_pressure(self):
        fig = plot_mollier_hx(pressure=95000.0)
        self.assertGreater(len(fig.data), 5)

    def test_custom_comfort_zone(self):
        fig = plot_mollier_hx(comfort_zone={
            "temperature": (18, 24),
            "rel_humidity": (0.20, 0.70),
            "abs_humidity": (0, 0.012),
        })
        self.assertGreater(len(fig.data), 5)

    def test_custom_colors(self):
        fig = plot_mollier_hx(colors={"temperature": "red"})
        self.assertGreater(len(fig.data), 5)

    def test_show_isolines_disabled(self):
        fig_all = plot_mollier_hx()
        fig_no_t = plot_mollier_hx(show_isolines={"temperature": False})
        self.assertLess(len(fig_no_t.data), len(fig_all.data))

    @patch('pyedautils.season.get_season', side_effect=_fast_get_season)
    def test_with_synthetic_data(self, _mock):
        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "humidity": np.random.uniform(30, 70, n),
            "temperature": np.random.uniform(15, 30, n),
        })
        fig = plot_mollier_hx(data=df)
        # Should have iso-line traces plus marker traces
        has_markers = any(
            t.mode == "markers" for t in fig.data if hasattr(t, "mode")
        )
        self.assertTrue(has_markers)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "humidity", "temperature"])
        fig = plot_mollier_hx(data=df)
        self.assertGreater(len(fig.data), 5)

    @patch('pyedautils.season.get_season', side_effect=_fast_get_season)
    def test_data_with_nan(self, _mock):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-06-01", periods=10, freq="h"),
            "humidity": [50, np.nan, 60, 55, 50, np.nan, 45, 50, 55, 60],
            "temperature": [22, 23, np.nan, 24, 25, 22, 23, 24, 25, 26],
        })
        fig = plot_mollier_hx(data=df)
        self.assertGreater(len(fig.data), 5)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
