# -*- coding: utf-8 -*-

import unittest
from datetime import datetime

import pandas as pd

from pyedautils.data_prep.sun import sun_position

# Zurich (Switzerland), close to sea level reference used for the expected angles.
LAT, LON = 47.3769, 8.5417


class TestSunPosition(unittest.TestCase):
    def test_summer_noon_high_and_south(self):
        # Summer solstice, near solar noon (~11:30 UTC for ~8.5 deg E).
        elevation, azimuth = sun_position("2025-06-21 11:30", LAT, LON)
        # Max elevation ~ 90 - (lat - 23.44) = ~66 deg.
        self.assertAlmostEqual(elevation, 66.0, delta=1.5)
        # Sun is due south at solar noon on the northern hemisphere.
        self.assertAlmostEqual(azimuth, 180.0, delta=3.0)

    def test_winter_noon_low(self):
        # Winter solstice noon elevation ~ 90 - (lat + 23.44) = ~19 deg.
        elevation, _ = sun_position("2025-12-21 11:30", LAT, LON)
        self.assertAlmostEqual(elevation, 19.0, delta=1.5)

    def test_midnight_below_horizon(self):
        elevation, _ = sun_position("2025-06-21 00:00", LAT, LON)
        self.assertLess(elevation, 0.0)

    def test_timezone_aware_matches_utc(self):
        # 13:30 CEST == 11:30 UTC -> identical position.
        aware = sun_position(pd.Timestamp("2025-06-21 13:30", tz="Europe/Zurich"), LAT, LON)
        naive_utc = sun_position("2025-06-21 11:30", LAT, LON)
        self.assertAlmostEqual(aware[0], naive_utc[0], places=6)
        self.assertAlmostEqual(aware[1], naive_utc[1], places=6)

    def test_azimuth_moves_east_to_west(self):
        morning = sun_position("2025-06-21 06:00", LAT, LON)[1]
        evening = sun_position("2025-06-21 17:00", LAT, LON)[1]
        # Morning sun in the east (< 180), evening sun in the west (> 180).
        self.assertLess(morning, 180.0)
        self.assertGreater(evening, 180.0)

    def test_datetimeindex_returns_dataframe(self):
        idx = pd.date_range("2025-06-21 03:00", "2025-06-21 20:00", freq="1h", tz="UTC")
        df = sun_position(idx, LAT, LON)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["elevation", "azimuth"])
        self.assertEqual(len(df), len(idx))
        pd.testing.assert_index_equal(df.index, idx)
        self.assertTrue((df["azimuth"] >= 0).all() and (df["azimuth"] <= 360).all())
        self.assertTrue(df["elevation"].between(-90, 90).all())

    def test_series_keeps_index(self):
        s = pd.Series(
            [datetime(2025, 6, 21, 6), datetime(2025, 6, 21, 12)],
            index=["a", "b"],
        )
        df = sun_position(s, LAT, LON)
        self.assertEqual(list(df.index), ["a", "b"])

    def test_refraction_raises_apparent_elevation_near_horizon(self):
        # Atmospheric refraction lifts the apparent position; the effect is
        # largest near the horizon.
        apparent, _ = sun_position("2025-06-21 04:00", LAT, LON, refraction=True)
        geometric, _ = sun_position("2025-06-21 04:00", LAT, LON, refraction=False)
        self.assertGreater(apparent, geometric)
        self.assertLess(apparent - geometric, 1.0)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
