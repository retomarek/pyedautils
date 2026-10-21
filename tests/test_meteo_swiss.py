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


MOCK_STAC_ITEM = {
    "assets": {
        "ogd-smn_klo_d_historical.csv": {"href": "https://example.org/klo/ogd-smn_klo_d_historical.csv"},
        "ogd-smn_klo_d_recent.csv": {"href": "https://example.org/klo/ogd-smn_klo_d_recent.csv"},
        "ogd-smn_klo_h_recent.csv": {"href": "https://example.org/klo/ogd-smn_klo_h_recent.csv"},
    }
}

MOCK_HISTORICAL_CSV = (
    "station_abbr;reference_timestamp;tre200d0;gre000d0;ure200d0\n"
    "KLO;30.12.2022 00:00;1.5;20;80\n"
    "KLO;31.12.2022 00:00;2.5;25;81\n"
    "KLO;01.01.2023 00:00;3.5;30;82\n"
)

MOCK_RECENT_CSV = (
    "station_abbr;reference_timestamp;tre200d0;gre000d0;ure200d0\n"
    "KLO;01.01.2023 00:00;3.6;31;82\n"
    "KLO;02.01.2023 00:00;4.5;40;83\n"
    "KLO;03.01.2023 00:00;;45;84\n"
)

MOCK_STATIONS_CSV = "station_abbr;station_name;station_height_masl\nKLO;Zürich / Kloten;426\n"


def _ogd_get(url, timeout=60):
    from pyedautils.weather.meteo_swiss import SMN_STAC_URL
    response = MagicMock()
    response.raise_for_status.return_value = None
    if url.startswith(SMN_STAC_URL):
        response.json.return_value = MOCK_STAC_ITEM
    elif url.endswith("_historical.csv"):
        response.content = MOCK_HISTORICAL_CSV.encode("utf-8")
    elif url.endswith("_recent.csv"):
        response.content = MOCK_RECENT_CSV.encode("utf-8")
    elif url.endswith("meta_stations.csv"):
        response.content = MOCK_STATIONS_CSV.encode("cp1252")
    else:
        raise AssertionError(f"unexpected url {url}")
    return response


class TestOgdDownloads(unittest.TestCase):

    @patch('pyedautils.weather.meteo_swiss.requests.get', side_effect=_ogd_get)
    def test_get_smn_stations(self, mock_get):
        from pyedautils.weather.meteo_swiss import get_smn_stations
        df = get_smn_stations()
        self.assertEqual(df.loc[0, "station_abbr"], "KLO")
        self.assertEqual(df.loc[0, "station_name"], "Zürich / Kloten")

    @patch('pyedautils.weather.meteo_swiss.requests.get', side_effect=_ogd_get)
    def test_download_daily(self, mock_get):
        from pyedautils.weather.meteo_swiss import download_daily
        df = download_daily("KLO", "2022-12-31", "2023-01-03")
        self.assertListEqual(list(df.columns), ["tre200d0", "gre000d0"])
        self.assertEqual(len(df), 4)
        self.assertEqual(df.index.name, "datetime")
        # overlap resolved in favour of the recent file
        self.assertAlmostEqual(df.loc["2023-01-01", "tre200d0"], 3.6)
        self.assertTrue(pd.isna(df.loc["2023-01-03", "tre200d0"]))
        # only daily assets are fetched: STAC item + 2 CSV files
        self.assertEqual(mock_get.call_count, 3)

    @patch('pyedautils.weather.meteo_swiss.requests.get', side_effect=_ogd_get)
    def test_download_daily_missing_param(self, mock_get):
        from pyedautils.weather.meteo_swiss import download_daily
        with self.assertLogs("pyedautils.weather.meteo_swiss", level="WARNING"):
            df = download_daily("KLO", "2023-01-01", "2023-01-02", params=("tre200d0", "xyz"))
        self.assertListEqual(list(df.columns), ["tre200d0"])
        with self.assertRaises(ValueError):
            download_daily("KLO", "2023-01-01", "2023-01-02", params=("xyz",))

    @patch('pyedautils.weather.meteo_swiss.requests.get')
    def test_download_daily_network_error(self, mock_get):
        from pyedautils.weather.meteo_swiss import download_daily
        mock_get.side_effect = Exception("Network error")
        with self.assertRaises(ValueError):
            download_daily("KLO", "2023-01-01", "2023-01-02")

    @patch('pyedautils.weather.meteo_swiss.requests.get')
    def test_get_smn_stations_network_error(self, mock_get):
        from pyedautils.weather.meteo_swiss import get_smn_stations
        mock_get.side_effect = Exception("Network error")
        with self.assertRaises(ValueError):
            get_smn_stations()

    @patch('pyedautils.weather.meteo_swiss.requests.get')
    def test_download_daily_no_daily_assets(self, mock_get):
        from pyedautils.weather.meteo_swiss import download_daily
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"assets": {"ogd-smn_klo_h_recent.csv": {"href": "x"}}}
        mock_get.return_value = response
        with self.assertRaisesRegex(ValueError, "No daily OGD files"):
            download_daily("KLO", "2023-01-01", "2023-01-02")
