# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pyedautils.data_quality import (
    calc_gap_duration,
    fill_missing_values_with_na,
    calc_isna_percentage,
    plot_missing_values,
)


class TestCalcGapDuration(unittest.TestCase):
    def test_uniform_sampling(self):
        """Uniform 1-minute data should have constant gap durations."""
        idx = pd.date_range("2024-01-01", periods=100, freq="min")
        df = pd.DataFrame({"value": range(100)}, index=idx)
        result = calc_gap_duration(df)
        self.assertIn("gapDuration", result.columns)
        self.assertIn("gapDurationRollMedian", result.columns)
        # All gaps (except first NaN) should be 60 seconds
        gaps = result["gapDuration"].dropna()
        self.assertTrue((gaps == 60).all())

    def test_with_gap(self):
        """A gap in 1-minute data should show correct duration in seconds."""
        idx1 = pd.date_range("2024-01-01 00:00", periods=30, freq="min")
        # 00:29 -> 00:40 = 11 minutes = 660 seconds
        idx2 = pd.date_range("2024-01-01 00:40", periods=30, freq="min")
        idx = idx1.append(idx2)
        df = pd.DataFrame({"value": range(len(idx))}, index=idx)
        result = calc_gap_duration(df)
        self.assertEqual(result["gapDuration"].iloc[30], 660)

    def test_gap_over_24h(self):
        """Gaps >24h must be captured correctly (total_seconds, not .seconds)."""
        idx = pd.DatetimeIndex([
            "2024-01-01 00:00",
            "2024-01-01 00:01",
            "2024-01-03 00:01",  # 48h gap
        ])
        df = pd.DataFrame({"value": [1, 2, 3]}, index=idx)
        result = calc_gap_duration(df)
        self.assertEqual(result["gapDuration"].iloc[2], 48 * 3600)

    def test_custom_window(self):
        """Custom window parameter should be accepted."""
        idx = pd.date_range("2024-01-01", periods=50, freq="min")
        df = pd.DataFrame({"value": range(50)}, index=idx)
        result = calc_gap_duration(df, window=5)
        self.assertEqual(len(result), 50)


class TestFillMissingValuesWithNa(unittest.TestCase):
    def test_fills_gaps(self):
        """A gap should produce additional NaN rows."""
        idx1 = pd.date_range("2024-01-01 00:00", periods=30, freq="min")
        idx2 = pd.date_range("2024-01-01 00:40", periods=30, freq="min")
        idx = idx1.append(idx2)
        df = pd.DataFrame({"value": range(len(idx))}, index=idx)
        result = fill_missing_values_with_na(df)
        self.assertGreater(len(result), len(df))
        self.assertTrue(result["value"].isna().any())

    def test_no_gaps(self):
        """Uniform data should not gain extra rows."""
        idx = pd.date_range("2024-01-01", periods=100, freq="min")
        df = pd.DataFrame({"value": range(100)}, index=idx)
        result = fill_missing_values_with_na(df)
        self.assertEqual(len(result), 100)

    def test_preserves_original_data(self):
        """Original values must still be present after filling."""
        idx1 = pd.date_range("2024-01-01 00:00", periods=30, freq="min")
        idx2 = pd.date_range("2024-01-01 00:40", periods=30, freq="min")
        idx = idx1.append(idx2)
        df = pd.DataFrame({"value": range(len(idx))}, index=idx)
        result = fill_missing_values_with_na(df)
        for ts in df.index:
            self.assertIn(ts, result.index)

    def test_multiple_columns(self):
        """Multiple columns should all get NaN in filled rows."""
        idx1 = pd.date_range("2024-01-01 00:00", periods=30, freq="min")
        idx2 = pd.date_range("2024-01-01 00:40", periods=30, freq="min")
        idx = idx1.append(idx2)
        df = pd.DataFrame({
            "temp": np.random.randn(len(idx)),
            "humidity": np.random.randn(len(idx)),
        }, index=idx)
        result = fill_missing_values_with_na(df)
        new_rows = result.loc[~result.index.isin(df.index)]
        self.assertTrue(new_rows["temp"].isna().all())
        self.assertTrue(new_rows["humidity"].isna().all())


class TestCalcIsnaPercentage(unittest.TestCase):
    def test_zero_percent(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        self.assertEqual(calc_isna_percentage(df), 0.0)

    def test_hundred_percent(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        self.assertEqual(calc_isna_percentage(df), 100.0)

    def test_fifty_percent(self):
        df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2]})
        self.assertEqual(calc_isna_percentage(df), 50.0)

    def test_column_parameter(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
        self.assertEqual(calc_isna_percentage(df, column="a"), 0.0)
        self.assertEqual(calc_isna_percentage(df, column="b"), 100.0)

    def test_decimals_parameter(self):
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        result = calc_isna_percentage(df, decimals=1)
        self.assertEqual(result, 33.3)


class TestPlotMissingValues(unittest.TestCase):
    def test_returns_figure(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="min")
        values = list(range(100))
        values[40:50] = [np.nan] * 10
        df = pd.DataFrame({"temp": values}, index=idx)
        fig = plot_missing_values(df)
        self.assertIsInstance(fig, go.Figure)

    def test_title_contains_percentage(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="min")
        values = list(range(100))
        values[40:50] = [np.nan] * 10
        df = pd.DataFrame({"temp": values}, index=idx)
        fig = plot_missing_values(df)
        self.assertIn("10.0%", fig.layout.title.text)

    def test_custom_column(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="min")
        df = pd.DataFrame({"a": range(10), "b": [np.nan] * 10}, index=idx)
        fig = plot_missing_values(df, column="b")
        self.assertIn("100.0%", fig.layout.title.text)


class TestPlotMissingValuesHeatmap(unittest.TestCase):
    def test_returns_figure(self):
        from pyedautils.data_quality import plot_missing_values_heatmap
        idx = pd.date_range("2024-01-01", periods=100, freq="min")
        values = list(range(100))
        values[40:50] = [np.nan] * 10
        df = pd.DataFrame({"temp": values, "hum": values[::-1]}, index=idx)
        fig = plot_missing_values_heatmap(df)
        self.assertIsInstance(fig, go.Figure)

    def test_custom_params(self):
        from pyedautils.data_quality import plot_missing_values_heatmap
        idx = pd.date_range("2024-01-01", periods=50, freq="min")
        df = pd.DataFrame({"a": [np.nan] * 50}, index=idx)
        fig = plot_missing_values_heatmap(
            df, title="Custom", height=200, color_scale=["blue", "yellow"]
        )
        self.assertIn("Custom", fig.layout.title.text)


class TestCalcOutliers(unittest.TestCase):
    def test_returns_dict(self):
        from pyedautils.data_quality import calc_outliers
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        np.random.seed(42)
        values = np.random.normal(20, 2, 100).tolist()
        values[50] = 100  # outlier
        df = pd.DataFrame({"temp": values}, index=idx)
        result = calc_outliers(df)
        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertIn("outliers", result)
        self.assertIn("count", result)
        self.assertIn("percentage", result)
        self.assertGreater(result["count"], 0)

    def test_default_column(self):
        from pyedautils.data_quality import calc_outliers
        idx = pd.date_range("2024-01-01", periods=50, freq="h")
        df = pd.DataFrame({"a": range(50), "b": range(50)}, index=idx)
        result = calc_outliers(df)
        self.assertIsNotNone(result["lower"])

    def test_explicit_column(self):
        from pyedautils.data_quality import calc_outliers
        idx = pd.date_range("2024-01-01", periods=50, freq="h")
        df = pd.DataFrame({"a": range(50), "b": range(50)}, index=idx)
        result = calc_outliers(df, column="b", multiplier=2.0)
        self.assertIsNotNone(result["lower"])


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
