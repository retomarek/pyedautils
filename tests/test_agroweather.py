# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from pyedautils.agroweather import (
    get_station_data,
    find_nearest_station,
    download_data,
    download_data_by_plz,
)

MOCK_STATIONS_API = {
    "data": [
        {
            "id": 1,
            "name": "Agraro",
            "lat_dec": "46.0",
            "long_dec": "8.9",
            "altitude": 400,
            "sensors": [{"id": 1}, {"id": 4}, {"id": 6}],
        },
        {
            "id": 2,
            "name": "Berno",
            "lat_dec": "46.95",
            "long_dec": "7.45",
            "altitude": 550,
            "sensors": [{"id": 1}, {"id": 11}],
        },
        {
            "id": 3,
            "name": "Cevio",
            "lat_dec": "46.3",
            "long_dec": "8.6",
            "altitude": 450,
            "sensors": [{"id": 1}, {"id": 4}, {"id": 6}, {"id": 11}],
        },
    ]
}

MOCK_DOWNLOAD_API = {
    "data": [
        {"date": "2024-01-01 00:00:00", "1_1_avg": "2.5"},
        {"date": "2024-01-01 01:00:00", "1_1_avg": "2.3"},
    ]
}


class TestGetStationData(unittest.TestCase):

    @patch('pyedautils.agroweather.requests.get')
    def test_returns_dataframe(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_STATIONS_API
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_station_data()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("id", result.columns)
        self.assertIn("sensors", result.columns)
        self.assertAlmostEqual(result.iloc[0]["lat"], 46.0)
        self.assertAlmostEqual(result.iloc[0]["lon"], 8.9)

    @patch('pyedautils.agroweather.requests.get')
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        with self.assertRaises(ValueError):
            get_station_data()


class TestFindNearestStation(unittest.TestCase):

    @patch('pyedautils.agroweather.get_station_data')
    def test_find_nearest_temp(self, mock_data):
        mock_data.return_value = pd.DataFrame([
            {"id": 1, "name": "Agraro", "lat": 46.0, "lon": 8.9, "sensors": [1, 4, 6]},
            {"id": 2, "name": "Berno", "lat": 46.95, "lon": 7.45, "sensors": [1, 11]},
            {"id": 3, "name": "Cevio", "lat": 46.3, "lon": 8.6, "sensors": [1, 4, 6, 11]},
        ])
        result = find_nearest_station(46.95, 7.45, sensor="temp")
        self.assertEqual(result, 2)

    @patch('pyedautils.agroweather.get_station_data')
    def test_find_nearest_globrad(self, mock_data):
        mock_data.return_value = pd.DataFrame([
            {"id": 1, "name": "Agraro", "lat": 46.0, "lon": 8.9, "sensors": [1, 4, 6]},
            {"id": 2, "name": "Berno", "lat": 46.95, "lon": 7.45, "sensors": [1, 11]},
            {"id": 3, "name": "Cevio", "lat": 46.3, "lon": 8.6, "sensors": [1, 4, 6, 11]},
        ])
        result = find_nearest_station(46.0, 8.9, sensor="globrad")
        self.assertEqual(result, 3)

    @patch('pyedautils.agroweather.get_station_data')
    def test_find_nearest_no_sensor_filter(self, mock_data):
        mock_data.return_value = pd.DataFrame([
            {"id": 1, "name": "Agraro", "lat": 46.0, "lon": 8.9, "sensors": [1, 4, 6]},
            {"id": 2, "name": "Berno", "lat": 46.95, "lon": 7.45, "sensors": [1, 11]},
        ])
        result = find_nearest_station(46.0, 8.9)
        self.assertEqual(result, 1)

    @patch('pyedautils.agroweather.get_station_data')
    def test_no_station_with_sensor(self, mock_data):
        mock_data.return_value = pd.DataFrame([
            {"id": 1, "name": "Agraro", "lat": 46.0, "lon": 8.9, "sensors": [1, 4]},
        ])
        with self.assertRaises(ValueError):
            find_nearest_station(46.0, 8.9, sensor="globrad")

    def test_invalid_sensor_type(self):
        with self.assertRaises(ValueError):
            find_nearest_station(47.0, 8.3, sensor="windspeed")


class TestDownloadData(unittest.TestCase):

    @patch('pyedautils.agroweather.requests.get')
    def test_returns_dataframe(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_DOWNLOAD_API
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = download_data(1, "2024-01-01", "2024-01-02", sensors=["temp"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("temp", result.columns)
        self.assertEqual(result.iloc[0]["temp"], 2.5)

    @patch('pyedautils.agroweather.requests.get')
    def test_default_sensors(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = download_data(1, "2024-01-01", "2024-01-02")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('pyedautils.agroweather.requests.get')
    def test_empty_data(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = download_data(1, "2024-01-01", "2024-01-02", sensors=["temp"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('pyedautils.agroweather.requests.get')
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        with self.assertRaises(ValueError):
            download_data(1, "2024-01-01", "2024-01-02", sensors=["temp"])

    @patch('pyedautils.agroweather.requests.get')
    def test_parse_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": "not_a_list"}
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            download_data(1, "2024-01-01", "2024-01-02", sensors=["temp"])
        self.assertIn("parsing", str(ctx.exception))

    def test_invalid_sensor(self):
        with self.assertRaises(ValueError):
            download_data(1, "2024-01-01", "2024-01-02", sensors=["windspeed"])


class TestDownloadDataByPlz(unittest.TestCase):

    @patch('pyedautils.agroweather.download_data')
    @patch('pyedautils.agroweather.find_nearest_station')
    @patch('pyedautils.agroweather.get_coordindates_ch_plz')
    def test_returns_dataframe(self, mock_plz, mock_nearest, mock_download):
        mock_plz.return_value = (47.05, 8.31)
        mock_nearest.return_value = 1
        mock_download.return_value = pd.DataFrame(
            {"temp": [2.5, 2.3]},
            index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
        )
        mock_download.return_value.index.name = "datetime"

        result = download_data_by_plz(6048, "2024-01-01", "2024-01-02", sensors=["temp"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        mock_plz.assert_called_once_with(6048)

    @patch('pyedautils.agroweather.download_data')
    @patch('pyedautils.agroweather.find_nearest_station')
    @patch('pyedautils.agroweather.get_coordindates_ch_plz')
    def test_merges_multiple_stations(self, mock_plz, mock_nearest, mock_download):
        mock_plz.return_value = (47.05, 8.31)
        mock_nearest.side_effect = [1, 2, 1, 1]

        idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"])

        def download_side_effect(station_id, start, end, sensors):
            if station_id == 1:
                return pd.DataFrame(
                    {s: [1.0, 1.1] for s in sensors},
                    index=idx,
                )
            else:
                return pd.DataFrame(
                    {s: [2.0, 2.1] for s in sensors},
                    index=idx,
                )

        mock_download.side_effect = download_side_effect

        result = download_data_by_plz(6048, "2024-01-01", "2024-01-02")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch('pyedautils.agroweather.find_nearest_station')
    @patch('pyedautils.agroweather.get_coordindates_ch_plz')
    def test_empty_sensors_returns_empty(self, mock_plz, mock_nearest):
        mock_plz.return_value = (47.05, 8.31)
        result = download_data_by_plz(6048, "2024-01-01", "2024-01-02", sensors=[])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
