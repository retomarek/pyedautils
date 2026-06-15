# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyedautils.data_prep.solar_influence import analyze_solar_influence
from pyedautils.plots.solar import plot_solar_influence


def _sunny_dataset(days=10, seed=0):
    idx = pd.date_range("2024-06-01", periods=24 * days, freq="h")
    hour = idx.hour
    solar = np.clip(800 * np.sin((hour - 6) / 12 * np.pi), 0, None)
    t_out = 18 + 5 * np.sin(np.arange(len(idx)) / 24)
    rng = np.random.default_rng(seed)
    t_room = t_out + 0.01 * solar + rng.normal(0, 0.1, len(idx))
    mk = lambda v: pd.DataFrame({"timestamp": idx, "value": v})  # noqa: E731
    return mk(t_room), mk(t_out), mk(solar)


class TestAnalyzeSolarInfluence(unittest.TestCase):
    def test_metrics_for_sun_exposed_sensor(self):
        room, out, rad = _sunny_dataset()
        res = analyze_solar_influence(room, out, rad, local_tz="Europe/Zurich")
        self.assertGreater(res["pearson_r"], 0.5)
        self.assertIsNotNone(res["peak_hour_local"])
        # peak excess temperature should fall around midday
        self.assertTrue(10 <= res["peak_hour_local"] <= 16)
        self.assertGreaterEqual(res["n_solar_events"], 0)

    def test_insufficient_data_returns_none(self):
        idx = pd.date_range("2024-06-01", periods=10, freq="h")
        mk = lambda: pd.DataFrame({"timestamp": idx, "value": range(10)})  # noqa: E731
        res = analyze_solar_influence(mk(), mk(), mk(), min_hours=48)
        self.assertIsNone(res["pearson_r"])
        self.assertEqual(res["n_hours"], 10)

    def test_series_input(self):
        room, out, rad = _sunny_dataset()
        to_s = lambda d: pd.Series(d["value"].to_numpy(),  # noqa: E731
                                   index=d["timestamp"])
        res = analyze_solar_influence(to_s(room), to_s(out), to_s(rad))
        self.assertGreater(res["pearson_r"], 0.5)


class TestPlotSolarInfluence(unittest.TestCase):
    def test_dual_axis_traces(self):
        room, out, rad = _sunny_dataset()
        idx = room["timestamp"]
        joined = pd.DataFrame({
            "t_room": room["value"].to_numpy(),
            "solar": rad["value"].to_numpy(),
        }, index=idx)
        joined["is_event"] = joined["solar"] > 700
        fig = plot_solar_influence(joined, event_col="is_event")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.layout.yaxis2.side, "right")

    def test_no_events(self):
        idx = pd.date_range("2024-06-01", periods=48, freq="h")
        joined = pd.DataFrame({"t_room": np.arange(48.0),
                               "solar": np.zeros(48)}, index=idx)
        fig = plot_solar_influence(joined)
        self.assertEqual(len(fig.data), 2)


if __name__ == "__main__":
    unittest.main()
