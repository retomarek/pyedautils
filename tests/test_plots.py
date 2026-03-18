import unittest
import os
import numpy as np
import pandas as pd
from pyedautils.plots import plot_daily_profiles_overview, plot_daily_profiles_decomposed


def _load_local_data():
    """Load the sample CSV from pyedautils/data and compute diff values."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'pyedautils', 'data', 'ele_meter.csv')
    df = pd.read_csv(csv_path, sep=';')
    df['value'] = df['value'].diff()
    df = df.dropna()
    return df


def _make_synthetic_data(n_days=90, freq='15min', start='2023-01-01'):
    """Create a small synthetic DataFrame with timestamp and value columns."""
    timestamps = pd.date_range(start=start, periods=n_days * 24 * 4, freq=freq)
    np.random.seed(42)
    values = np.random.rand(len(timestamps)) * 100
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


class TestPlots(unittest.TestCase):

    def test_plot_daily_profiles_overview(self):
        url = "https://raw.githubusercontent.com/retomarek/pyedautils/main/pyedautils/data/ele_meter.csv"

        # Call the function under test
        df = pd.read_csv(url, engine="python", sep=None)
        df['value'] = df['value'].diff()
        df = df.dropna()

        fig = plot_daily_profiles_overview(df)

        # Assertions
        self.assertTrue(fig._data[111]["y"][23] == 42.75)
        self.assertTrue(fig._data[111]["x"][23] == 23)


class TestPlotDailyProfilesOverviewCustomParams(unittest.TestCase):
    """Tests for plot_daily_profiles_overview with custom parameters."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_local_data()

    def test_custom_title(self):
        fig = plot_daily_profiles_overview(self.df, title="My Custom Title")
        self.assertIn("My Custom Title", fig.layout.title.text)

    def test_custom_ylab(self):
        fig = plot_daily_profiles_overview(self.df, ylab="kWh")
        # ylab is applied to the first column y-axes (row labels are seasons)
        # Check that the figure was created successfully
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_custom_confidence(self):
        fig_90 = plot_daily_profiles_overview(self.df, confidence=90.0)
        fig_50 = plot_daily_profiles_overview(self.df, confidence=50.0)
        # Both should produce valid figures with data
        self.assertGreater(len(fig_90.data), 0)
        self.assertGreater(len(fig_50.data), 0)

    def test_custom_colors(self):
        custom_colors = {"median": "red", "bounds": "blue", "fill": "yellow"}
        fig = plot_daily_profiles_overview(self.df, colors=custom_colors)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)
        # Verify the median line color is applied (every 4th trace starting at index 2)
        # Trace order per subplot: fill, upper bound, median, lower bound
        median_trace = fig.data[2]
        self.assertEqual(median_trace.line.color, "red")

    def test_custom_seasons(self):
        custom_seasons = ["Frühling", "Sommer", "Herbst", "Winter"]
        fig = plot_daily_profiles_overview(self.df, seasons=custom_seasons)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_custom_weekdays(self):
        custom_weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = plot_daily_profiles_overview(self.df, weekdays=custom_weekdays)
        self.assertIsNotNone(fig)
        # Check subplot titles contain custom weekday names
        annotations = [a.text for a in fig.layout.annotations]
        for day in custom_weekdays:
            self.assertIn(day, annotations)


class TestPlotDailyProfilesOverviewEdgeCases(unittest.TestCase):
    """Edge-case tests for plot_daily_profiles_overview."""

    def test_data_with_nan_values(self):
        """NaN values in the value column should not crash the function."""
        df = _make_synthetic_data(n_days=90)
        # Inject NaN values at random positions
        np.random.seed(99)
        nan_indices = np.random.choice(len(df), size=200, replace=False)
        df.loc[nan_indices, 'value'] = np.nan
        fig = plot_daily_profiles_overview(df)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_empty_subsets(self):
        """Data covering only one season should still produce a figure without errors."""
        # Only January-February data => only Winter season will have data
        df = _make_synthetic_data(n_days=60, start='2023-01-01')
        fig = plot_daily_profiles_overview(df)
        self.assertIsNotNone(fig)
        # Figure should still have 4x7 subplot structure even if some are empty
        self.assertGreater(len(fig.data), 0)


class TestPlotDailyProfilesDecomposed(unittest.TestCase):
    """Tests for plot_daily_profiles_decomposed."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_local_data()

    def test_basic_decomposed(self):
        """Test that the function returns a valid figure with 7 traces (one per weekday)."""
        fig = plot_daily_profiles_decomposed(self.df)
        self.assertIsNotNone(fig)
        # Should have 7 traces (one per weekday)
        self.assertEqual(len(fig.data), 7)

    def test_decomposed_trace_names(self):
        """Each trace should be named after a weekday."""
        fig = plot_daily_profiles_decomposed(self.df)
        expected_days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
        trace_names = [t.name for t in fig.data]
        self.assertEqual(trace_names, expected_days)

    def test_custom_title(self):
        fig = plot_daily_profiles_decomposed(self.df, title="Custom Decomposed")
        self.assertIn("Custom Decomposed", fig.layout.title.text)

    def test_custom_ylab(self):
        fig = plot_daily_profiles_decomposed(self.df, ylab="Power [W]")
        self.assertEqual(fig.layout.yaxis.title.text, "Power [W]")

    def test_custom_timezone(self):
        fig = plot_daily_profiles_decomposed(self.df, loc_time_zone="Europe/Zurich")
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 7)

    def test_custom_k_and_digits(self):
        fig = plot_daily_profiles_decomposed(self.df, k=336, digits=2)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 7)

    def test_x_values_are_hours(self):
        """X values should represent hours of the day (0-23 range)."""
        fig = plot_daily_profiles_decomposed(self.df)
        for trace in fig.data:
            x_vals = trace.x
            self.assertGreaterEqual(min(x_vals), 0)
            self.assertLessEqual(max(x_vals), 23.75)

    def test_decomposed_with_synthetic_data(self):
        """Test decomposed plot with small synthetic data to avoid network dependency."""
        df = _make_synthetic_data(n_days=60)
        fig = plot_daily_profiles_decomposed(df)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 7)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
