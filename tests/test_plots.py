import unittest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch
from pyedautils.plots import plot_daily_profiles_overview, plot_daily_profiles, plot_daily_profiles_decomposed, plot_heatmap_median_weeks, plot_heatmap_calendar


def _load_local_data():
    """Load the sample CSV from pyedautils/data and compute diff values."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'pyedautils', 'data', 'ele_meter.csv')
    df = pd.read_csv(csv_path, sep=';')
    return df


def _make_synthetic_data(n_days=30, freq='h', start='2023-01-01'):
    """Create a small synthetic DataFrame with timestamp and value columns."""
    timestamps = pd.date_range(start=start, periods=n_days * 24, freq=freq)
    np.random.seed(42)
    values = np.random.rand(len(timestamps)) * 100
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


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


class TestPlots(unittest.TestCase):
    """Core regression test using real data (no mock — validates exact values)."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_local_data()
        cls.fig = plot_daily_profiles_overview(cls.df)

    def test_plot_daily_profiles_overview(self):
        self.assertTrue(self.fig._data[111]["y"][23] == 171.0)
        self.assertTrue(self.fig._data[111]["x"][23] == 23)


@patch('pyedautils.season.get_season', side_effect=_fast_get_season)
class TestPlotDailyProfilesOverviewCustomParams(unittest.TestCase):
    """Tests for plot_daily_profiles_overview with custom parameters.
    Uses fast mocked season detection."""

    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='h')
        np.random.seed(42)
        values = np.random.rand(len(timestamps)) * 100
        cls.df = pd.DataFrame({'timestamp': timestamps, 'value': values})

    def test_custom_title(self, _mock):
        fig = plot_daily_profiles_overview(self.df, title="My Custom Title")
        self.assertIn("My Custom Title", fig.layout.title.text)

    def test_custom_ylab(self, _mock):
        fig = plot_daily_profiles_overview(self.df, ylab="kWh")
        self.assertGreater(len(fig.data), 0)

    def test_custom_confidence(self, _mock):
        fig = plot_daily_profiles_overview(self.df, confidence=50.0)
        self.assertGreater(len(fig.data), 0)

    def test_custom_colors(self, _mock):
        custom_colors = {"median": "red", "bounds": "blue", "fill": "yellow"}
        fig = plot_daily_profiles_overview(self.df, colors=custom_colors)
        median_trace = fig.data[2]
        self.assertEqual(median_trace.line.color, "red")

    def test_custom_seasons(self, _mock):
        custom_seasons = ["Frühling", "Sommer", "Herbst", "Winter"]
        fig = plot_daily_profiles_overview(self.df, seasons=custom_seasons)
        self.assertGreater(len(fig.data), 0)

    def test_custom_weekdays(self, _mock):
        custom_weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = plot_daily_profiles_overview(self.df, weekdays=custom_weekdays)
        annotations = [a.text for a in fig.layout.annotations]
        for day in custom_weekdays:
            self.assertIn(day, annotations)


@patch('pyedautils.season.get_season', side_effect=_fast_get_season)
class TestPlotDailyProfilesOverviewEdgeCases(unittest.TestCase):
    """Edge-case tests using small synthetic data."""

    def test_data_with_nan_values(self, _mock):
        df = _make_synthetic_data(n_days=30)
        np.random.seed(99)
        nan_indices = np.random.choice(len(df), size=100, replace=False)
        df.loc[nan_indices, 'value'] = np.nan
        fig = plot_daily_profiles_overview(df)
        self.assertGreater(len(fig.data), 0)

    def test_empty_subsets(self, _mock):
        df = _make_synthetic_data(n_days=30, start='2023-01-01')
        fig = plot_daily_profiles_overview(df)
        self.assertGreater(len(fig.data), 0)


class TestPlotDailyProfilesDecomposed(unittest.TestCase):
    """Tests for plot_daily_profiles_decomposed."""

    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='h')
        np.random.seed(42)
        values = np.random.rand(len(timestamps)) * 100
        cls.df = pd.DataFrame({'timestamp': timestamps, 'value': values})
        cls.fig = plot_daily_profiles_decomposed(cls.df)

    def test_basic_decomposed(self):
        self.assertEqual(len(self.fig.data), 7)

    def test_decomposed_trace_names(self):
        expected_days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
        trace_names = [t.name for t in self.fig.data]
        self.assertEqual(trace_names, expected_days)

    def test_custom_title(self):
        fig = plot_daily_profiles_decomposed(self.df, title="Custom Decomposed")
        self.assertIn("Custom Decomposed", fig.layout.title.text)

    def test_custom_ylab(self):
        fig = plot_daily_profiles_decomposed(self.df, ylab="Power [W]")
        self.assertEqual(fig.layout.yaxis.title.text, "Power [W]")

    def test_custom_timezone(self):
        fig = plot_daily_profiles_decomposed(self.df, loc_time_zone="Europe/Zurich")
        self.assertEqual(len(fig.data), 7)

    def test_custom_k_and_digits(self):
        fig = plot_daily_profiles_decomposed(self.df, k=336, digits=2)
        self.assertEqual(len(fig.data), 7)

    def test_x_values_are_hours(self):
        for trace in self.fig.data:
            x_vals = trace.x
            self.assertGreaterEqual(min(x_vals), 0)
            self.assertLessEqual(max(x_vals), 23.75)

    def test_decomposed_with_synthetic_data(self):
        df = _make_synthetic_data(n_days=30)
        fig = plot_daily_profiles_decomposed(df)
        self.assertEqual(len(fig.data), 7)


class TestPlotDailyProfiles(unittest.TestCase):
    """Tests for plot_daily_profiles with method parameter."""

    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='h')
        np.random.seed(42)
        values = np.random.rand(len(timestamps)) * 100
        cls.df = pd.DataFrame({'timestamp': timestamps, 'value': values})

    def test_mean_method(self):
        fig = plot_daily_profiles(self.df, method="mean")
        self.assertEqual(len(fig.data), 7)

    def test_decomposed_method(self):
        fig = plot_daily_profiles(self.df, method="decomposed")
        self.assertEqual(len(fig.data), 7)

    def test_mean_default_title(self):
        fig = plot_daily_profiles(self.df, method="mean")
        self.assertIn("Mean", fig.layout.title.text)

    def test_decomposed_default_title(self):
        fig = plot_daily_profiles(self.df, method="decomposed")
        self.assertIn("Decomposed", fig.layout.title.text)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            plot_daily_profiles(self.df, method="invalid")

    def test_mean_trace_names(self):
        fig = plot_daily_profiles(self.df, method="mean")
        expected_days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
        trace_names = [t.name for t in fig.data]
        self.assertEqual(trace_names, expected_days)

    def test_mean_custom_title_and_ylab(self):
        fig = plot_daily_profiles(self.df, method="mean", title="Custom", ylab="kWh")
        self.assertIn("Custom", fig.layout.title.text)

    def test_overlayed_returns_html(self):
        html = plot_daily_profiles(self.df, method="overlayed")
        self.assertIsInstance(html, str)
        self.assertIn("plotly_hover", html)

    def test_overlayed_default_title(self):
        html = plot_daily_profiles(self.df, method="overlayed")
        self.assertIn("Overlayed", html)

    def test_overlayed_contains_weekdays(self):
        html = plot_daily_profiles(self.df, method="overlayed")
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]:
            self.assertIn(day, html)


@patch('pyedautils.season.get_season', side_effect=_fast_get_season)
class TestPlotHeatmapMedianWeeks(unittest.TestCase):
    """Tests for plot_heatmap_median_weeks."""

    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='h')
        np.random.seed(42)
        values = np.random.rand(len(timestamps)) * 100
        cls.df = pd.DataFrame({'timestamp': timestamps, 'value': values})

    def test_basic_heatmap(self, _mock):
        fig = plot_heatmap_median_weeks(self.df)
        # 4 seasons = 4 heatmap traces
        self.assertEqual(len(fig.data), 4)

    def test_traces_are_heatmaps(self, _mock):
        fig = plot_heatmap_median_weeks(self.df)
        for trace in fig.data:
            self.assertEqual(trace.type, "heatmap")

    def test_custom_title(self, _mock):
        fig = plot_heatmap_median_weeks(self.df, title="My Heatmap")
        self.assertIn("My Heatmap", fig.layout.title.text)

    def test_custom_seasons(self, _mock):
        custom = ["Frühling", "Sommer", "Herbst", "Winter"]
        fig = plot_heatmap_median_weeks(self.df, seasons=custom)
        self.assertEqual(len(fig.data), 4)

    def test_heatmap_z_shape(self, _mock):
        fig = plot_heatmap_median_weeks(self.df)
        # Each heatmap: 7 weekdays x 24 hours
        for trace in fig.data:
            self.assertEqual(len(trace.z), 7)
            self.assertEqual(len(trace.z[0]), 24)

    def test_colorscale_param(self, _mock):
        fig = plot_heatmap_median_weeks(self.df, colorscale="Viridis")
        # Plotly expands named colorscales to tuples; check first entry
        self.assertIsNotNone(fig.data[0].colorscale)


class TestPlotHeatmapCalendar(unittest.TestCase):
    """Tests for plot_heatmap_calendar."""

    @classmethod
    def setUpClass(cls):
        timestamps = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='h')
        np.random.seed(42)
        values = np.random.rand(len(timestamps)) * 100
        cls.df = pd.DataFrame({'timestamp': timestamps, 'value': values})

    def test_basic_calendar(self):
        fig = plot_heatmap_calendar(self.df)
        self.assertEqual(len(fig.data), 1)  # 1 year = 1 heatmap

    def test_trace_is_heatmap(self):
        fig = plot_heatmap_calendar(self.df)
        self.assertEqual(fig.data[0].type, "heatmap")

    def test_custom_title(self):
        fig = plot_heatmap_calendar(self.df, title="My Calendar")
        self.assertIn("My Calendar", fig.layout.title.text)

    def test_z_shape(self):
        fig = plot_heatmap_calendar(self.df)
        # 7 weekdays x 53 possible weeks
        self.assertEqual(len(fig.data[0].z), 7)
        self.assertEqual(len(fig.data[0].z[0]), 54)

    def test_multi_year(self):
        timestamps = pd.date_range(start='2022-01-01', end='2023-12-31 23:00', freq='h')
        np.random.seed(42)
        values = np.random.rand(len(timestamps)) * 100
        df = pd.DataFrame({'timestamp': timestamps, 'value': values})
        fig = plot_heatmap_calendar(df)
        self.assertEqual(len(fig.data), 2)  # 2 years = 2 heatmaps


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
