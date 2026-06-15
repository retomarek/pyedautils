# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyedautils.data_quality import (
    classify_quality_flags,
    detect_gaps,
    detect_outliers,
    detect_stuck,
    infer_interval,
    plot_data_quality,
)


def _series_with_gap_and_stuck():
    idx1 = pd.date_range("2024-01-01 00:00", periods=60, freq="10min")
    idx2 = pd.date_range("2024-01-01 12:00", periods=60, freq="10min")
    idx = idx1.append(idx2)
    val = list(np.arange(60.0)) + [50.0] * 60
    return pd.DataFrame({"timestamp": idx, "value": val})


class TestInferInterval(unittest.TestCase):
    def test_median_interval(self):
        ts = pd.date_range("2024-01-01", periods=50, freq="5min").to_series()
        self.assertEqual(infer_interval(ts), pd.Timedelta(minutes=5))

    def test_fallback(self):
        self.assertEqual(infer_interval(pd.Series([pd.Timestamp("2024-01-01")])),
                         pd.Timedelta(minutes=10))

    def test_fallback_all_duplicate(self):
        # all-equal timestamps -> no positive diff -> 10 min fallback
        dup = pd.Series([pd.Timestamp("2024-01-01")] * 5)
        self.assertEqual(infer_interval(dup), pd.Timedelta(minutes=10))


class TestDetectGaps(unittest.TestCase):
    def test_gap_found(self):
        df = _series_with_gap_and_stuck()
        gaps = detect_gaps(df)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps.iloc[0]["gap_duration_h"], 2.17, places=2)

    def test_series_input_matches(self):
        df = _series_with_gap_and_stuck()
        ser = pd.Series(df["value"].to_numpy(), index=df["timestamp"])
        self.assertEqual(len(detect_gaps(ser)), len(detect_gaps(df)))

    def test_no_gaps(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="10min"),
            "value": range(20),
        })
        self.assertTrue(detect_gaps(df).empty)

    def test_single_row(self):
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")],
                           "value": [1.0]})
        self.assertTrue(detect_gaps(df, pd.Timedelta(minutes=10)).empty)

    def test_datetimeindex_dataframe_input(self):
        # DataFrame indexed by timestamp (no 'timestamp' column)
        df = _series_with_gap_and_stuck().set_index("timestamp")
        self.assertEqual(len(detect_gaps(df)), 1)


class TestDetectStuck(unittest.TestCase):
    def test_stuck_found(self):
        df = _series_with_gap_and_stuck()
        stuck = detect_stuck(df, min_repeats=20, min_duration_h=6.0)
        self.assertEqual(len(stuck), 1)
        self.assertEqual(stuck.iloc[0]["stuck_value"], 50.0)
        self.assertEqual(stuck.iloc[0]["n_repeats"], 60)

    def test_min_duration_filters(self):
        df = _series_with_gap_and_stuck()
        # require 20 hours -> the 9.83h run no longer qualifies
        self.assertTrue(detect_stuck(df, min_repeats=20, min_duration_h=20.0).empty)

    def test_empty_input(self):
        empty = pd.DataFrame({"timestamp": pd.to_datetime([]), "value": []})
        self.assertTrue(detect_stuck(empty).empty)


class TestDetectOutliers(unittest.TestCase):
    def test_range(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "value": [-5, 10, 50, 105, 20],
        })
        out = detect_outliers(df, 0, 100)
        self.assertEqual(len(out), 2)
        self.assertSetEqual(set(out["reason"]), {"below 0", "above 100"})

    def test_no_outliers(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "value": [10, 20, 30],
        })
        self.assertTrue(detect_outliers(df, 0, 100).empty)


class TestClassifyFlags(unittest.TestCase):
    def test_levels(self):
        summary = pd.DataFrame({
            "coverage_pct": [95, 85, 60],
            "longest_gap_h": [1, 30, 200],
            "outlier_pct": [0, 2, 10],
            "n_stuck_periods": [0, 2, 9],
        })
        flags = classify_quality_flags(summary).tolist()
        self.assertEqual(flags, ["ok", "warning", "critical"])

    def test_custom_thresholds(self):
        summary = pd.DataFrame({"coverage_pct": [80]})
        flags = classify_quality_flags(summary, {"cov_warn": 70, "cov_crit": 50})
        self.assertEqual(flags.iloc[0], "ok")


class TestPlotDataQuality(unittest.TestCase):
    def test_returns_figure_with_gap_shapes(self):
        df = _series_with_gap_and_stuck()
        fig = plot_data_quality(df)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(len(fig.layout.shapes), 1)


if __name__ == "__main__":
    unittest.main()
