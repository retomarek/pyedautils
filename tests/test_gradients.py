# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyedautils.data_prep.gradients import (
    compute_gradients,
    mean_gradients_by_season,
    summarize_gradients,
)
from pyedautils.plots.gradients import plot_gradient_boxplots


def _signal():
    idx = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    val = 22 + 5 * np.sin(np.arange(len(idx)) / 3.0)
    return pd.Series(val, index=idx)


class TestComputeGradients(unittest.TestCase):
    def test_columns_and_directions(self):
        g = compute_gradients(_signal(), threshold=0.5)
        self.assertListEqual(list(g.columns),
                             ["timestamp", "gradient", "direction", "season"])
        self.assertSetEqual(set(g["direction"]), {"heating", "cooling"})
        # heating rows must have positive gradient, cooling negative
        self.assertTrue((g.loc[g["direction"] == "heating", "gradient"] > 0).all())
        self.assertTrue((g.loc[g["direction"] == "cooling", "gradient"] < 0).all())

    def test_threshold_filters(self):
        g_low = compute_gradients(_signal(), threshold=0.1)
        g_high = compute_gradients(_signal(), threshold=1.0)
        self.assertGreater(len(g_low), len(g_high))

    def test_dataframe_input_matches_series(self):
        s = _signal()
        df = pd.DataFrame({"timestamp": s.index, "value": s.to_numpy()})
        g_s = compute_gradients(s, threshold=0.5)
        g_df = compute_gradients(df, threshold=0.5)
        self.assertEqual(len(g_s), len(g_df))

    def test_custom_direction_labels(self):
        g = compute_gradients(_signal(), threshold=0.5,
                              direction_labels=("up", "down"))
        self.assertSetEqual(set(g["direction"]), {"up", "down"})

    def test_custom_season_labels(self):
        labels = ["Frühling", "Sommer", "Herbst", "Winter"]
        g = compute_gradients(_signal(), threshold=0.5, season_labels=labels)
        self.assertTrue(set(g["season"]) <= set(labels))

    def test_empty_result(self):
        # huge threshold -> nothing passes
        g = compute_gradients(_signal(), threshold=100.0)
        self.assertTrue(g.empty)
        self.assertListEqual(list(g.columns),
                             ["timestamp", "gradient", "direction", "season"])


class TestSummaries(unittest.TestCase):
    def test_summarize(self):
        g = compute_gradients(_signal(), threshold=0.5)
        s = summarize_gradients(g)
        self.assertListEqual(list(s["direction"]), ["heating", "cooling"])
        heating = s[s["direction"] == "heating"].iloc[0]
        self.assertGreater(heating["mean"], 0)
        self.assertGreater(heating["n"], 0)

    def test_summarize_empty_direction(self):
        g = compute_gradients(_signal(), threshold=0.5)
        only_up = g[g["direction"] == "heating"]
        s = summarize_gradients(only_up)
        cooling = s[s["direction"] == "cooling"].iloc[0]
        self.assertEqual(cooling["n"], 0)
        self.assertTrue(np.isnan(cooling["mean"]))

    def test_by_season(self):
        g = compute_gradients(_signal(), threshold=0.5)
        ms = mean_gradients_by_season(g)
        self.assertListEqual(list(ms.columns), ["season", "direction", "mean", "n"])

    def test_by_season_with_order(self):
        g = compute_gradients(_signal(), threshold=0.5)
        ms = mean_gradients_by_season(
            g, season_order=["Winter", "Spring", "Summer", "Fall"])
        self.assertFalse(ms.empty)

    def test_by_season_empty(self):
        ms = mean_gradients_by_season(pd.DataFrame(
            columns=["timestamp", "gradient", "direction", "season"]))
        self.assertTrue(ms.empty)


class TestPlot(unittest.TestCase):
    def test_boxplots_season(self):
        g = compute_gradients(_signal(), threshold=0.5)
        fig = plot_gradient_boxplots(g, groupby="season")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)

    def test_boxplots_quarter(self):
        g = compute_gradients(_signal(), threshold=0.5)
        fig = plot_gradient_boxplots(g, groupby="quarter")
        self.assertEqual(len(fig.data), 2)

    def test_empty(self):
        fig = plot_gradient_boxplots(pd.DataFrame(
            columns=["timestamp", "gradient", "direction", "season"]))
        self.assertEqual(len(fig.data), 0)

    def test_boxplots_other_groupings(self):
        g = compute_gradients(_signal(), threshold=0.5)
        for gb in ("month", "weekday", "hour"):
            fig = plot_gradient_boxplots(g, groupby=gb)
            self.assertEqual(len(fig.data), 2, gb)

    def test_invalid_groupby_raises(self):
        g = compute_gradients(_signal(), threshold=0.5)
        with self.assertRaises(ValueError):
            plot_gradient_boxplots(g, groupby="decade")

    def test_season_order_applied(self):
        g = compute_gradients(_signal(), threshold=0.5)
        order = ["Winter", "Spring", "Summer", "Fall"]
        fig = plot_gradient_boxplots(g, groupby="season", season_order=order)
        self.assertEqual(fig.layout.xaxis.categoryorder, "array")
        self.assertEqual(list(fig.layout.xaxis.categoryarray), order)

    def test_single_direction_only(self):
        g = compute_gradients(_signal(), threshold=0.5)
        fig = plot_gradient_boxplots(g[g["direction"] == "heating"])
        self.assertEqual(len(fig.data), 1)


if __name__ == "__main__":
    unittest.main()
