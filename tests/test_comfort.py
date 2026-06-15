# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pyedautils import comfort
from pyedautils.plots.comfort import (
    plot_comfort_donuts,
    plot_comfort_sia180,
    plot_overheating_bar,
)


class TestSia180Curves(unittest.TestCase):
    def test_max_temp_plateaus_and_slope(self):
        """Upper boundary: 24.5 below 12 °C, 26.5 above 17.5 °C, linear between."""
        got = comfort.sia180_max_temp([-5, 12, 17.5, 40])
        self.assertTrue(np.allclose(got, [24.5, 24.5, 26.5, 26.5]))
        # midpoint 14.75 °C -> halfway between 24.5 and 26.5
        self.assertAlmostEqual(float(comfort.sia180_max_temp(14.75)), 25.5)

    def test_min_temp_plateaus_and_slope(self):
        """Lower boundary: 20.5 below 19 °C, 22.0 above 23.5 °C, linear between."""
        got = comfort.sia180_min_temp([-5, 19, 23.5, 40])
        self.assertTrue(np.allclose(got, [20.5, 20.5, 22.0, 22.0]))

    def test_scalar_input(self):
        self.assertEqual(float(comfort.sia180_max_temp(5.0)), 24.5)


class TestSummerSemester(unittest.TestCase):
    def test_boundaries_inclusive(self):
        ts = pd.to_datetime([
            "2024-04-15", "2024-04-16", "2024-07-01",
            "2024-10-15", "2024-10-16", "2024-01-01",
        ])
        mask = comfort.is_summer_semester_sia180(ts).tolist()
        self.assertEqual(mask, [False, True, True, True, False, False])


class TestAlignAndOverheating(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-07-01", periods=72, freq="h")
        self.room = pd.DataFrame({"timestamp": idx, "value": np.linspace(22, 30, 72)})
        self.outdoor = pd.DataFrame({"timestamp": idx, "value": np.linspace(15, 25, 72)})

    def test_align_hourly_columns(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        self.assertListEqual(list(al.columns), ["t_room", "t_oa", "t_oa_48h"])
        self.assertFalse(al.empty)

    def test_align_hourly_empty(self):
        al = comfort.align_hourly(pd.DataFrame(), self.outdoor)
        self.assertTrue(al.empty)

    def test_overheating_hours_adaptive(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        total, thr = comfort.overheating_hours(al, summer_only=True)
        self.assertGreater(total, 0)
        self.assertEqual(len(thr), len(thr.dropna()))

    def test_overheating_hours_fixed(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        total, thr = comfort.overheating_hours(al, method="fixed", summer_only=False)
        self.assertTrue((thr == comfort.FIXED_OVERHEATING_THRESHOLD).all())

    def test_comfort_kpis_keys(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        kpis = comfort.comfort_kpis(al)
        for k in ("n_hours", "h_cold", "h_warm", "h_ok", "pct_ok",
                  "overheating_h", "sia180_compliant", "minergie_compliant"):
            self.assertIn(k, kpis)

    def test_overheating_per_month(self):
        al = comfort.align_hourly(self.room, self.outdoor)
        _, thr = comfort.overheating_hours(al, summer_only=False)
        monthly = comfort.overheating_per_month(al, thr)
        self.assertListEqual(list(monthly.columns), ["month", "hours"])


class TestComfortPlots(unittest.TestCase):
    def test_sia180_plot_uses_curve_functions(self):
        """The refactored plot must draw the exact adaptive boundary y-values."""
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        outdoor = pd.DataFrame({"timestamp": idx, "value": np.linspace(-5, 30, 200)})
        room = pd.DataFrame({"timestamp": idx, "value": np.linspace(20, 27, 200)})
        fig = plot_comfort_sia180(outdoor, room)
        self.assertIsInstance(fig, go.Figure)
        lower = next(t for t in fig.data if t.name == "Lower limit SIA 180")
        self.assertTrue(np.allclose(
            comfort.sia180_min_temp(lower.x), lower.y))
        upper = next(t for t in fig.data if t.name == "Upper limit active cooling")
        self.assertTrue(np.allclose(
            comfort.sia180_max_temp(upper.x), upper.y))

    def test_donuts(self):
        d = pd.DataFrame({"temperature": [18, 22, 28, 21],
                          "humidity": [20, 50, 70, 45]})
        fig = plot_comfort_donuts(d)
        self.assertEqual(len(fig.data), 2)
        # temperature donut: 1 cold, 2 comfort, 1 warm
        self.assertEqual(list(fig.data[0].values), [1, 2, 1])

    def test_overheating_bar(self):
        bar = pd.DataFrame({"label": ["A", "B", "C"], "hours": [50, 200, 500]})
        fig = plot_overheating_bar(bar)
        self.assertEqual(len(fig.data), 1)
        # sorted ascending -> worst (C) last
        self.assertEqual(list(fig.data[0].y), ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
