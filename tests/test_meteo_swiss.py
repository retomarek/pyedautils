# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from pyedautils.weather.meteo_swiss import get_current_station_data, find_nearest_station


MOCK_STATION_DATA = pd.DataFrame({
    "Abk.": ["LUZ", "BAN", "FLU"],
    "Station": ["Luzern", "Bantiger", "Flühli"],
    "Breitengrad": ["47.036", "46.975", "46.889"],
    "Längengrad": ["8.301", "7.528", "8.015"],
    "Stationshöhe m ü. M.": ["454", "942", "940"],
    "Messungen": [
        "Temperatur, Feuchte, Niederschlag",
        "Globalstrahlung, Temperatur",
        "Temperatur, Feuchte, Niederschlag",
    ],
})


class TestGetCurrentStationData(unittest.TestCase):

    @patch('pyedautils.weather.meteo_swiss.pd.read_csv')
    def test_returns_dataframe(self, mock_read_csv):
        mock_read_csv.return_value = MOCK_STATION_DATA.copy()
        result = get_current_station_data()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    @patch('pyedautils.weather.meteo_swiss.pd.read_csv')
    def test_filters_nan_messungen(self, mock_read_csv):
        data = MOCK_STATION_DATA.copy()
        data.loc[2, "Messungen"] = None
        mock_read_csv.return_value = data
        result = get_current_station_data()
        self.assertEqual(len(result), 2)

    @patch('pyedautils.weather.meteo_swiss.pd.read_csv')
    def test_network_error(self, mock_read_csv):
        mock_read_csv.side_effect = Exception("Network error")
        with self.assertRaises(ValueError):
            get_current_station_data()


class TestFindNearestStation(unittest.TestCase):

    @patch('pyedautils.weather.meteo_swiss.get_current_station_data')
    def test_find_temp_station(self, mock_data):
        mock_data.return_value = MOCK_STATION_DATA.copy()
        result = find_nearest_station(47.01, 8.30, 450, sensor="temp")
        self.assertEqual(result, "LUZ")

    @patch('pyedautils.weather.meteo_swiss.get_current_station_data')
    def test_find_globrad_station(self, mock_data):
        mock_data.return_value = MOCK_STATION_DATA.copy()
        # From coordinates near Bantiger altitude range
        result = find_nearest_station(46.97, 7.53, 950, sensor="globrad")
        self.assertEqual(result, "BAN")

    @patch('pyedautils.weather.meteo_swiss.get_current_station_data')
    def test_find_relhum_station(self, mock_data):
        mock_data.return_value = MOCK_STATION_DATA.copy()
        result = find_nearest_station(47.01, 8.30, 450, sensor="relhum")
        self.assertEqual(result, "LUZ")

    @patch('pyedautils.weather.meteo_swiss.get_current_station_data')
    def test_find_rain_station(self, mock_data):
        mock_data.return_value = MOCK_STATION_DATA.copy()
        result = find_nearest_station(47.01, 8.30, 450, sensor="rain")
        self.assertEqual(result, "LUZ")

    def test_coordinates_out_of_range(self):
        with self.assertRaises(ValueError):
            find_nearest_station(8.3, 47.0, 450, sensor="temp")

    def test_invalid_sensor_type(self):
        with self.assertRaises(ValueError):
            find_nearest_station(47.0, 8.3, 450, sensor="windspeed")

    def test_missing_sensor_argument(self):
        with self.assertRaises(TypeError):
            find_nearest_station(47.0, 8.3, 450)

    @patch('pyedautils.weather.meteo_swiss.get_current_station_data')
    def test_no_station_in_altitude_range(self, mock_data):
        mock_data.return_value = MOCK_STATION_DATA.copy()
        # altitude=2000 won't match any station (max is 942)
        with self.assertRaises(ValueError):
            find_nearest_station(47.01, 8.30, 2000, sensor="temp")


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
