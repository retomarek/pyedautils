import unittest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch
from pyedautils.plots import (
    plot_daily_profiles_overview, plot_daily_profiles,
    plot_heatmap_median_weeks, plot_heatmap_calendar,
    plot_energy_signature, plot_energy_signature_pes,
    plot_decomposition, plot_density_seasons,
    plot_seasonal_overlapping, plot_seasonal_miniplots,
    plot_seasonal_before_after, plot_seasonal_polar,
    plot_sum_frequency,
    plot_comfort_sia180, plot_comfort_temp_humidity,
    plot_timeseries, plot_distribution, plot_boxplot,
    plot_outliers, plot_correlation, plot_scatter,
    plot_autocorrelation,
)


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


@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
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


@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
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


@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
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


def _load_simple_data():
    """Load the bundled simple energy signature CSV."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'pyedautils', 'data',
        'bldg_engy_sig_simple.csv',
    )
    return pd.read_csv(csv_path, sep=';')


def _load_pes_data():
    """Load the bundled PES energy signature CSV."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'pyedautils', 'data',
        'bldg_engy_sig_pes.csv',
    )
    return pd.read_csv(csv_path)


@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
class TestPlotEnergySignature(unittest.TestCase):
    """Tests for plot_energy_signature."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_simple_data()

    def test_returns_figure(self, _mock):
        fig = plot_energy_signature(self.df)
        self.assertIsNotNone(fig)

    def test_has_season_traces(self, _mock):
        fig = plot_energy_signature(self.df)
        trace_names = {t.name for t in fig.data}
        # Should have at least some seasons
        self.assertTrue(
            trace_names.intersection({"Winter", "Spring", "Summer", "Fall"})
        )

    def test_custom_title(self, _mock):
        fig = plot_energy_signature(self.df, title="My Title")
        self.assertIn("My Title", fig.layout.title.text)

    def test_custom_axis_labels(self, _mock):
        fig = plot_energy_signature(
            self.df, xlab="Temp [°C]", ylab="Energy [kWh]",
        )
        self.assertIn("Temp", fig.layout.xaxis.title.text)
        self.assertIn("Energy", fig.layout.yaxis.title.text)

    def test_custom_colors(self, _mock):
        colors = {"Winter": "red", "Summer": "blue"}
        fig = plot_energy_signature(self.df, colors=colors)
        winter = [t for t in fig.data if t.name == "Winter"]
        if winter:
            self.assertEqual(winter[0].marker.color, "red")

    def test_scatter_mode(self, _mock):
        fig = plot_energy_signature(self.df)
        for trace in fig.data:
            self.assertEqual(trace.mode, "markers")


@patch('pyedautils.energy_signature.get_season', side_effect=_fast_get_season)
@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
class TestPlotEnergySignaturePES(unittest.TestCase):
    """Tests for plot_energy_signature_pes."""

    @classmethod
    def setUpClass(cls):
        cls.df = _load_pes_data()

    def test_returns_figure(self, _mock_season, _mock_es):
        fig = plot_energy_signature_pes(self.df)
        self.assertIsNotNone(fig)

    def test_has_regression_lines(self, _mock_season, _mock_es):
        fig = plot_energy_signature_pes(self.df)
        trace_names = [t.name for t in fig.data]
        self.assertTrue(any("Heating" in n for n in trace_names))
        self.assertTrue(any("P_dhw" in n for n in trace_names))
        self.assertTrue(any("P_dhwc" in n for n in trace_names))

    def test_custom_title(self, _mock_season, _mock_es):
        fig = plot_energy_signature_pes(self.df, title="PES Custom")
        self.assertIn("PES Custom", fig.layout.title.text)

    def test_has_annotations(self, _mock_season, _mock_es):
        fig = plot_energy_signature_pes(self.df)
        self.assertGreaterEqual(len(fig.layout.annotations), 2)

    def test_with_p_ihg(self, _mock_season, _mock_es):
        fig = plot_energy_signature_pes(self.df, p_ihg=4.8)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)


@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
class TestPlotComfort(unittest.TestCase):
    """Tests for comfort plot functions."""

    @classmethod
    def setUpClass(cls):
        base = os.path.join(os.path.dirname(__file__), '..', 'pyedautils', 'data')
        cls.df_oa = pd.read_csv(os.path.join(base, 'outside_temp.csv'), sep=';')
        cls.df_r = pd.read_csv(os.path.join(base, 'flat_temp.csv'), sep=';')
        cls.df_th = pd.read_csv(os.path.join(base, 'flat_temp_hum.csv'), sep=';')

    def test_sia180_returns_figure(self, _mock):
        fig = plot_comfort_sia180(self.df_oa, self.df_r)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_sia180_has_boundary_lines(self, _mock):
        fig = plot_comfort_sia180(self.df_oa, self.df_r)
        names = [t.name for t in fig.data]
        self.assertTrue(any("Lower" in n for n in names))
        self.assertTrue(any("active" in n for n in names))

    def test_temp_hum_returns_figure(self, _mock):
        fig = plot_comfort_temp_humidity(self.df_th)
        self.assertIsNotNone(fig)

    def test_temp_hum_has_comfort_zones(self, _mock):
        fig = plot_comfort_temp_humidity(self.df_th)
        names = [t.name for t in fig.data]
        self.assertIn("Comfortable", names)
        self.assertIn("Still comfortable", names)

    def test_temp_hum_custom_title(self, _mock):
        fig = plot_comfort_temp_humidity(self.df_th, title="My Comfort")
        self.assertIn("My Comfort", fig.layout.title.text)


class TestPlotSumFrequency(unittest.TestCase):
    """Tests for plot_sum_frequency."""

    @classmethod
    def setUpClass(cls):
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', 'pyedautils', 'data',
            'outside_temp.csv',
        )
        cls.df = pd.read_csv(csv_path, sep=';')

    def test_daily_returns_figure(self):
        fig = plot_sum_frequency(self.df, resolution='daily', year=2019)
        self.assertIsNotNone(fig)

    def test_hourly_returns_figure(self):
        fig = plot_sum_frequency(self.df, resolution='hourly', year=2019)
        self.assertIsNotNone(fig)

    def test_reverse_flag(self):
        fig_asc = plot_sum_frequency(self.df, resolution='daily', year=2019)
        fig_desc = plot_sum_frequency(self.df, resolution='daily',
                                      year=2019, reverse=True)
        # First y-value should be lowest for ascending, highest for descending
        self.assertLess(fig_asc.data[0].y[0], fig_desc.data[0].y[0])

    def test_custom_title(self):
        fig = plot_sum_frequency(self.df, title="Custom SF")
        self.assertIn("Custom SF", fig.layout.title.text)


@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
class TestPlotDensitySeasons(unittest.TestCase):
    """Tests for plot_density_seasons."""

    @classmethod
    def setUpClass(cls):
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', 'pyedautils', 'data',
            'flat_temp.csv',
        )
        cls.df = pd.read_csv(csv_path, sep=';')

    def test_returns_figure(self, _mock):
        fig = plot_density_seasons(self.df)
        self.assertIsNotNone(fig)

    def test_four_season_traces(self, _mock):
        fig = plot_density_seasons(self.df)
        self.assertEqual(len(fig.data), 4)

    def test_custom_title(self, _mock):
        fig = plot_density_seasons(self.df, title="My Density")
        self.assertIn("My Density", fig.layout.title.text)

    def test_trace_names(self, _mock):
        fig = plot_density_seasons(self.df)
        names = {t.name for t in fig.data}
        self.assertEqual(names, {"Spring", "Summer", "Fall", "Winter"})


class TestPlotSeasonalPlots(unittest.TestCase):
    """Tests for all 4 seasonal plot functions."""

    @classmethod
    def setUpClass(cls):
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', 'pyedautils', 'data',
            'seasonal_monthly.csv',
        )
        cls.df = pd.read_csv(csv_path, sep=';')

    def test_overlapping_returns_figure(self):
        fig = plot_seasonal_overlapping(self.df)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_overlapping_custom_title(self):
        fig = plot_seasonal_overlapping(self.df, title="Custom")
        self.assertIn("Custom", fig.layout.title.text)

    def test_miniplots_returns_figure(self):
        fig = plot_seasonal_miniplots(self.df)
        self.assertIsNotNone(fig)

    def test_miniplots_has_12_subplots(self):
        fig = plot_seasonal_miniplots(self.df)
        # Should have traces for 12 months
        self.assertGreater(len(fig.data), 12)

    def test_before_after_returns_figure(self):
        fig = plot_seasonal_before_after(
            self.df, date_optimization="2017-09-01")
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_before_after_custom_title(self):
        fig = plot_seasonal_before_after(
            self.df, date_optimization="2017-09-01", title="BefAft")
        self.assertIn("BefAft", fig.layout.title.text)

    def test_polar_returns_figure(self):
        fig = plot_seasonal_polar(self.df)
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_polar_uses_scatterpolar(self):
        fig = plot_seasonal_polar(self.df)
        self.assertEqual(fig.data[0].type, "scatterpolar")


class TestPlotDecomposition(unittest.TestCase):
    """Tests for plot_decomposition."""

    @classmethod
    def setUpClass(cls):
        # Long-term monthly data
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', 'pyedautils', 'data',
            'decomposition_long.csv',
        )
        cls.df_long = pd.read_csv(csv_path, sep=';')

        # Short-term 15-min data
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', 'pyedautils', 'data',
            'decomposition_short.csv',
        )
        cls.df_short = pd.read_csv(csv_path, sep=';')

    def test_long_returns_figure(self):
        fig = plot_decomposition(self.df_long, period=12, s_window=7)
        self.assertIsNotNone(fig)

    def test_long_has_four_panels(self):
        fig = plot_decomposition(self.df_long, period=12, s_window=7)
        self.assertEqual(len(fig.data), 4)

    def test_short_returns_figure(self):
        fig = plot_decomposition(self.df_short, period=96, s_window=193)
        self.assertIsNotNone(fig)

    def test_short_has_four_panels(self):
        fig = plot_decomposition(self.df_short, period=96, s_window=193)
        self.assertEqual(len(fig.data), 4)

    def test_custom_title(self):
        fig = plot_decomposition(self.df_long, period=12,
                                 title="Custom Title")
        self.assertIn("Custom Title", fig.layout.title.text)

    def test_auto_period_monthly(self):
        fig = plot_decomposition(self.df_long)
        self.assertEqual(len(fig.data), 4)

    def test_auto_period_15min(self):
        fig = plot_decomposition(self.df_short)
        self.assertEqual(len(fig.data), 4)


def _make_ts_dataframe(n=200, freq='h', seed=42):
    """Create a DataFrame with DatetimeIndex and two value columns."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    np.random.seed(seed)
    return pd.DataFrame({
        "temp": np.random.normal(20, 5, n),
        "humidity": np.random.normal(50, 10, n),
    }, index=idx)


class TestPlotTimeseries(unittest.TestCase):
    def test_returns_figure(self):
        df = _make_ts_dataframe()
        fig = plot_timeseries(df)
        self.assertIsNotNone(fig)

    def test_custom_columns(self):
        df = _make_ts_dataframe()
        fig = plot_timeseries(df, columns=["temp"])
        self.assertIsNotNone(fig)

    def test_no_range_slider(self):
        df = _make_ts_dataframe()
        fig = plot_timeseries(df, range_slider=False)
        self.assertIsNotNone(fig)


class TestPlotDistribution(unittest.TestCase):
    def test_returns_figure(self):
        df = _make_ts_dataframe()
        fig = plot_distribution(df)
        self.assertIsNotNone(fig)

    def test_custom_columns(self):
        df = _make_ts_dataframe()
        fig = plot_distribution(df, columns=["temp"])
        self.assertIsNotNone(fig)


class TestPlotBoxplot(unittest.TestCase):
    def test_returns_figure(self):
        df = _make_ts_dataframe()
        fig = plot_boxplot(df)
        self.assertIsNotNone(fig)

    def test_groupby_hour(self):
        df = _make_ts_dataframe()
        fig = plot_boxplot(df, groupby="hour")
        self.assertIsNotNone(fig)

    def test_groupby_weekday(self):
        df = _make_ts_dataframe()
        fig = plot_boxplot(df, groupby="weekday")
        self.assertIsNotNone(fig)

    def test_groupby_quarter(self):
        df = _make_ts_dataframe()
        fig = plot_boxplot(df, groupby="quarter")
        self.assertIsNotNone(fig)

    def test_invalid_groupby(self):
        df = _make_ts_dataframe()
        with self.assertRaises(ValueError):
            plot_boxplot(df, groupby="year")

    def test_custom_title(self):
        df = _make_ts_dataframe()
        fig = plot_boxplot(df, title="Custom Box")
        self.assertIn("Custom Box", fig.layout.title.text)

    def test_default_column(self):
        df = _make_ts_dataframe()
        fig = plot_boxplot(df)
        self.assertIsNotNone(fig)


class TestPlotOutliers(unittest.TestCase):
    def test_returns_figure(self):
        df = _make_ts_dataframe()
        # Add an outlier
        df.iloc[0, 0] = 200
        fig = plot_outliers(df)
        self.assertIsNotNone(fig)

    def test_custom_params(self):
        df = _make_ts_dataframe()
        df.iloc[0, 0] = 200
        fig = plot_outliers(df, column="temp", multiplier=1.0,
                            title="Custom Outliers", ylab="°C")
        self.assertIn("Custom Outliers", fig.layout.title.text)


class TestPlotCorrelation(unittest.TestCase):
    def test_returns_figure(self):
        df = _make_ts_dataframe()
        fig = plot_correlation(df)
        self.assertIsNotNone(fig)


class TestPlotScatter(unittest.TestCase):
    def test_returns_figure(self):
        df = _make_ts_dataframe()
        fig = plot_scatter(df)
        self.assertIsNotNone(fig)

    def test_custom_columns(self):
        df = _make_ts_dataframe()
        fig = plot_scatter(df, x_column="temp", y_column="humidity",
                           title="Custom", xlab="T", ylab="RH")
        self.assertIn("Custom", fig.layout.title.text)


class TestPlotAutocorrelation(unittest.TestCase):
    def test_returns_figure(self):
        df = _make_ts_dataframe(n=500)
        fig = plot_autocorrelation(df, lags=48)
        self.assertIsNotNone(fig)

    def test_custom_title(self):
        df = _make_ts_dataframe(n=500)
        fig = plot_autocorrelation(df, lags=48, title="Custom ACF")
        self.assertIn("Custom ACF", fig.layout.title.text)


@patch('pyedautils.data_prep.season.get_season', side_effect=_fast_get_season)
class TestPlotEmptySeasonBranches(unittest.TestCase):
    """Cover the `if subset.empty: continue` branches in season loops."""

    @classmethod
    def setUpClass(cls):
        # Data for only January (Winter) — Spring/Summer/Fall will be empty
        timestamps = pd.date_range('2023-01-01', '2023-01-31 23:00', freq='h')
        np.random.seed(42)
        n = len(timestamps)
        cls.df_simple = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.normal(5, 3, n),
            'value': np.random.uniform(10, 50, n),
        })
        cls.df_th = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.normal(20, 2, n),
            'humidity': np.random.normal(50, 5, n),
        })

    def test_energy_signature_missing_seasons(self, _mock):
        fig = plot_energy_signature(self.df_simple)
        self.assertIsNotNone(fig)

    def test_density_seasons_missing_seasons(self, _mock):
        fig = plot_density_seasons(self.df_simple[["timestamp", "value"]])
        self.assertIsNotNone(fig)

    def test_comfort_sia180_missing_seasons(self, _mock):
        fig = plot_comfort_sia180(
            self.df_simple[["timestamp", "temperature"]].rename(
                columns={"temperature": "value"}
            ),
            self.df_simple[["timestamp", "value"]],
        )
        self.assertIsNotNone(fig)

    def test_comfort_temp_humidity_missing_seasons(self, _mock):
        fig = plot_comfort_temp_humidity(self.df_th)
        self.assertIsNotNone(fig)

    @patch('pyedautils.energy_signature.compute_pes')
    def test_energy_signature_pes_missing_seasons(self, mock_pes, _mock):
        from pyedautils.energy_signature import PESResult
        mock_pes.return_value = PESResult(
            tb=15.0, q_tot=0.5, p_dhwc=1.0, p_dhw=2.0, p_ihg=0.0,
        )
        # Only January data -> Spring/Summer/Fall empty
        df_pes = pd.DataFrame({
            'timestamp': self.df_simple['timestamp'],
            'outside_temp': self.df_simple['temperature'],
            'power': self.df_simple['value'],
            'room_temp': np.full(len(self.df_simple), 21.0),
        })
        fig = plot_energy_signature_pes(df_pes)
        self.assertIsNotNone(fig)


class TestPlotDecompositionAutoDetect(unittest.TestCase):
    """Test auto-detection of period for hourly and daily data."""

    def test_auto_period_hourly(self):
        # Hourly data -> period=24
        df = _make_synthetic_data(n_days=60, freq='h')
        fig = plot_decomposition(df)
        self.assertEqual(len(fig.data), 4)

    def test_auto_period_daily(self):
        # Daily data -> period=7
        timestamps = pd.date_range("2023-01-01", periods=365, freq='D')
        np.random.seed(42)
        values = np.random.rand(365) * 100
        df = pd.DataFrame({'timestamp': timestamps, 'value': values})
        fig = plot_decomposition(df)
        self.assertEqual(len(fig.data), 4)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
