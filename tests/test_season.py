# -*- coding: utf-8 -*-

import unittest
import pandas as pd
from pandas.testing import assert_series_equal
from datetime import datetime

from pyedautils.season import get_season

class TestSeasonFunction(unittest.TestCase):
    def test_season_northern_hemisphere_astronomical(self):
        # Test dates for the northern hemisphere
        test_cases = [
            (datetime(2024, 3, 20, 3, 5), "Winter"),
            (datetime(2024, 3, 20, 3, 7), "Spring"),
            (datetime(2024, 6, 20, 20, 50), "Spring"),
            (datetime(2024, 6, 20, 21, 52), "Summer"),
            (datetime(2024, 9, 22, 12, 42), "Summer"),
            (datetime(2024, 9, 22, 12, 44), "Fall"),
            (datetime(2024, 12, 21, 9, 19), "Fall"),
            (datetime(2024, 12, 21, 9, 21), "Winter"),
        ]
        for date, expected_season in test_cases:
            with self.subTest(date):
                self.assertEqual(get_season(date=date), expected_season)

    def test_season_northern_hemisphere_meteorological(self):
        # Test dates for the northern hemisphere with meteorological datechanges
        test_cases = [
            (datetime(2024, 2, 27), "Winter"),
            (datetime(2024, 3, 1), "Spring"),
            (datetime(2024, 5, 30), "Spring"),
            (datetime(2024, 6, 1), "Summer"),
            (datetime(2024, 8, 31), "Summer"),
            (datetime(2024, 9, 1), "Fall"),
            (datetime(2024, 11, 30), "Fall"),
            (datetime(2024, 12, 1), "Winter"),
        ]
        for date, expected_season in test_cases:
            with self.subTest(date):
                self.assertEqual(get_season(date=date, tracking_type="meteorological"), expected_season)
                
    def test_season_southern_hemisphere_astronomical(self):
        # Test dates for the southern hemisphere
        test_cases = [
            (datetime(2024, 3, 23), "Fall"),  # Spring in the Northern Hemisphere
            (datetime(2024, 6, 30), "Winter"),  # Summer in the Northern Hemisphere
            (datetime(2024, 9, 24), "Spring"),  # Fall in the Northern Hemisphere
            (datetime(2024, 12, 24), "Summer"),  # Winter in the Northern Hemisphere
        ]
        for date, expected_season in test_cases:
            with self.subTest(date):
                self.assertEqual(get_season(date=date, hemisphere="south"), expected_season)

    def test_custom_labels(self):
        # Test with custom labels
        custom_labels = ["frühling", "sommer", "herbst", "winter"]
        test_cases = [
            (datetime(2024, 3, 20, 3, 5), "winter"),
            (datetime(2024, 3, 20, 3, 7), "frühling"),
            (datetime(2024, 6, 20, 20, 50), "frühling"),
            (datetime(2024, 6, 20, 21, 52), "sommer"),
            (datetime(2024, 9, 22, 12, 42), "sommer"),
            (datetime(2024, 9, 22, 12, 44), "herbst"),
            (datetime(2024, 12, 21, 9, 19), "herbst"),
            (datetime(2024, 12, 21, 9, 21), "winter"),
        ]
        for date, expected_season in test_cases:
            with self.subTest(date):
                self.assertEqual(get_season(date=date, labels=custom_labels), expected_season)
       
    def test_pandas_series(self):
        # Test passing a pandas Series object
        pd_datetime_series = pd.Series([datetime(2024, 3, 31, 3, 5), datetime(2024, 6, 22), datetime(2024, 9, 24), datetime(2024, 12, 24)])
        expected_seasons = pd.Series(["Spring", "Summer", "Fall", "Winter"])
        assert_series_equal(get_season(date=pd_datetime_series), expected_seasons)
                
if __name__ == '__main__':
    unittest.main() # pragma: no cover
