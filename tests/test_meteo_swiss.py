# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from pyedautils.weather.meteo_swiss import (
    get_smn_parameters,
    get_station_parameters,
    get_current_station_data,
    find_nearest_station,
    download_data,
    download_data_by_plz,
    get_smn_stations,
    _select_files,
)


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


STAC_ASSETS = {
    "assets": {
        f"ogd-smn_klo_h_{kind}.csv": {"href": f"https://example.org/ogd-smn_klo_h_{kind}.csv"}
        for kind in ["historical_2010-2019", "historical_2020-2029", "recent", "now"]
    }
}


def _mock_response(status_code=200, json_data=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status.return_value = None
    return response


def _mock_csv(url, **kwargs):
    if "historical_2020" in url:
        timestamps = ["31.12.2025 22:00", "31.12.2025 23:00"]
    elif "recent" in url:
        timestamps = ["01.01.2026 00:00", "01.01.2026 01:00"]
    else:
        timestamps = ["01.01.2026 01:00", "01.01.2026 02:00"]
    data = pd.DataFrame({
        "station_abbr": "KLO",
        "reference_timestamp": timestamps,
        "tre200h0": [1.0, 2.0],
        "gre000h0": [0, 5],
        "ure200h0": [80.0, 81.0],
        "rre150h0": [0.0, 0.2],
    })
    return data[[c for c in data.columns if kwargs["usecols"](c)]]


class TestSelectFiles(unittest.TestCase):

    FILES = {
        "x_h_historical_2010-2019.csv": "u2010",
        "x_h_historical_2020-2029.csv": "u2020",
        "x_h_recent.csv": "urecent",
        "x_h_now.csv": "unow",
        "x_d_historical.csv": "ud",
        "x_d_recent.csv": "udrecent",
        "x_m.csv": "um",
    }
    TODAY = pd.Timestamp("2026-10-22")

    def test_historical_only(self):
        urls = _select_files(self.FILES, "h", pd.Timestamp("2019-06-01"), pd.Timestamp("2020-02-01"), self.TODAY)
        self.assertEqual(urls, ["u2010", "u2020"])

    def test_historical_recent_now(self):
        urls = _select_files(self.FILES, "h", pd.Timestamp("2025-12-01"), pd.Timestamp("2026-10-23"), self.TODAY)
        self.assertEqual(urls, ["u2020", "urecent", "unow"])

    def test_daily_single_historical_file(self):
        urls = _select_files(self.FILES, "d", pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01"), self.TODAY)
        self.assertEqual(urls, ["ud"])


NOW = pd.Timestamp("2026-01-02 10:00", tz="UTC")


class TestDownloadData(unittest.TestCase):

    @patch('pyedautils.weather.meteo_swiss.pd.Timestamp.now', return_value=NOW)
    @patch('pyedautils.weather.meteo_swiss.pd.read_csv', side_effect=_mock_csv)
    @patch('pyedautils.weather.meteo_swiss.requests.get', return_value=_mock_response(json_data=STAC_ASSETS))
    def test_combines_files(self, mock_get, mock_read_csv, mock_now):
        df = download_data("KLO", "2025-12-31", "2026-01-02")
        self.assertEqual(list(df.columns), ["temp", "globrad", "relhum", "rain"])
        self.assertEqual(len(df), 5)  # duplicate 01:00 from recent/now removed
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertEqual(mock_read_csv.call_count, 3)

    @patch('pyedautils.weather.meteo_swiss.pd.Timestamp.now', return_value=NOW)
    @patch('pyedautils.weather.meteo_swiss.pd.read_csv', side_effect=_mock_csv)
    @patch('pyedautils.weather.meteo_swiss.requests.get', return_value=_mock_response(json_data=STAC_ASSETS))
    def test_filters_range_and_converts_tz(self, mock_get, mock_read_csv, mock_now):
        df = download_data("KLO", "2026-01-01", "2026-01-01", sensors=["temp", "rre150h0"], tz="Europe/Zurich")
        self.assertEqual(list(df.columns), ["temp", "rre150h0"])
        # 2026-01-01 local = 2025-12-31 23:00 UTC .. 2026-01-01 23:00 UTC
        self.assertEqual(len(df), 4)
        self.assertEqual(df.index[0], pd.Timestamp("2026-01-01 00:00", tz="Europe/Zurich"))

    @patch('pyedautils.weather.meteo_swiss.pd.read_csv', side_effect=_mock_csv)
    @patch('pyedautils.weather.meteo_swiss.requests.get')
    def test_falls_back_to_precip_collection(self, mock_get, mock_read_csv):
        mock_get.side_effect = [_mock_response(status_code=404), _mock_response(json_data=STAC_ASSETS)]
        df = download_data("ABE", "2019-01-01", "2019-12-31", sensors=["rain"])
        self.assertIn("ch.meteoschweiz.ogd-smn-precip/items/abe", mock_get.call_args_list[1].args[0])
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ["rain"])

    @patch('pyedautils.weather.meteo_swiss.requests.get', return_value=_mock_response(status_code=404))
    def test_unknown_station(self, mock_get):
        with self.assertRaises(ValueError):
            download_data("XXX", "2025-01-01", "2025-01-02")

    @patch('pyedautils.weather.meteo_swiss.requests.get', side_effect=Exception("Network error"))
    def test_network_error(self, mock_get):
        with self.assertRaises(ValueError):
            download_data("KLO", "2025-01-01", "2025-01-02")

    def test_invalid_sensor(self):
        with self.assertRaises(ValueError):
            download_data("KLO", "2025-01-01", "2025-01-02", sensors=["windspeed"])

    def test_invalid_granularity(self):
        with self.assertRaises(ValueError):
            download_data("KLO", "2025-01-01", "2025-01-02", granularity="m")

    def test_end_before_start(self):
        with self.assertRaises(ValueError):
            download_data("KLO", "2025-01-02", "2025-01-01")


class TestDownloadDataByPlz(unittest.TestCase):

    @patch('pyedautils.weather.meteo_swiss.download_data')
    @patch('pyedautils.weather.meteo_swiss.find_nearest_station')
    @patch('pyedautils.weather.meteo_swiss.get_altitude_lat_long', return_value=450.0)
    @patch('pyedautils.weather.meteo_swiss.get_coordindates_ch_plz', return_value=(47.01, 8.30))
    def test_groups_sensors_by_station(self, mock_plz, mock_alt, mock_find, mock_download):
        mock_find.side_effect = lambda lat, lon, alt, sensor: "BAN" if sensor == "globrad" else "LUZ"
        index = pd.date_range("2025-07-01", periods=2, freq="h", tz="UTC", name="datetime")
        mock_download.side_effect = lambda station, *args, sensors, **kwargs: pd.DataFrame(
            {s: [1.0, 2.0] for s in sensors}, index=index)

        df = download_data_by_plz(6048, "2025-07-01", "2025-07-01")
        self.assertEqual(list(df.columns), ["temp", "globrad", "relhum", "rain"])
        self.assertEqual(mock_download.call_count, 2)
        mock_find.assert_any_call(47.01, 8.30, 450.0, sensor="temp")

    @patch('pyedautils.weather.meteo_swiss.download_data')
    @patch('pyedautils.weather.meteo_swiss.find_nearest_station', return_value="LUZ")
    @patch('pyedautils.weather.meteo_swiss.get_altitude_lat_long')
    @patch('pyedautils.weather.meteo_swiss.get_coordindates_ch_plz', return_value=(47.01, 8.30))
    def test_explicit_altitude(self, mock_plz, mock_alt, mock_find, mock_download):
        mock_download.return_value = pd.DataFrame({"temp": [1.0]})
        download_data_by_plz(6048, "2025-07-01", "2025-07-01", sensors=["temp"], altitude=1000)
        mock_alt.assert_not_called()
        mock_find.assert_called_once_with(47.01, 8.30, 1000, sensor="temp")


DAILY_STAC_ASSETS = {
    "assets": {
        "ogd-smn_klo_d_historical.csv": {"href": "https://example.org/klo/ogd-smn_klo_d_historical.csv"},
        "ogd-smn_klo_d_recent.csv": {"href": "https://example.org/klo/ogd-smn_klo_d_recent.csv"},
        "ogd-smn_klo_h_recent.csv": {"href": "https://example.org/klo/ogd-smn_klo_h_recent.csv"},
    }
}

MOCK_DAILY = {
    "historical": pd.DataFrame({
        "reference_timestamp": ["30.12.2022 00:00", "31.12.2022 00:00", "01.01.2023 00:00"],
        "tre200d0": [1.5, 2.5, 3.5],
        "gre000d0": [20, 25, 30],
    }),
    "recent": pd.DataFrame({
        "reference_timestamp": ["01.01.2023 00:00", "02.01.2023 00:00", "03.01.2023 00:00"],
        "tre200d0": [3.6, 4.5, None],
        "gre000d0": [31, 40, 45],
    }),
}

MOCK_STATIONS_CSV = "station_abbr;station_name;station_height_masl\nKLO;Zürich / Kloten;426\n"


def _mock_daily_csv(url, **kwargs):
    data = MOCK_DAILY["historical" if "historical" in url else "recent"]
    return data[[c for c in data.columns if kwargs["usecols"](c)]]


class TestDownloadDataDaily(unittest.TestCase):

    @patch('pyedautils.weather.meteo_swiss.pd.Timestamp.now', return_value=pd.Timestamp("2023-01-05", tz="UTC"))
    @patch('pyedautils.weather.meteo_swiss.pd.read_csv', side_effect=_mock_daily_csv)
    @patch('pyedautils.weather.meteo_swiss.requests.get', return_value=_mock_response(json_data=DAILY_STAC_ASSETS))
    def test_daily_values(self, mock_get, mock_read_csv, mock_now):
        df = download_data("KLO", "2022-12-31", "2023-01-03", granularity="d",
                           sensors=["temp", "globrad"], tz="Europe/Zurich")
        self.assertListEqual(list(df.columns), ["temp", "globrad"])
        self.assertEqual(len(df), 4)
        self.assertIsNone(df.index.tz)  # daily values: tz-naive dates, tz ignored
        self.assertEqual(df.index[0], pd.Timestamp("2022-12-31"))
        # overlap resolved in favour of the recent file
        self.assertAlmostEqual(df.loc["2023-01-01", "temp"], 3.6)
        self.assertTrue(pd.isna(df.loc["2023-01-03", "temp"]))
        # only daily files are read
        self.assertEqual(mock_read_csv.call_count, 2)

    @patch('pyedautils.weather.meteo_swiss.pd.Timestamp.now', return_value=pd.Timestamp("2023-01-05", tz="UTC"))
    @patch('pyedautils.weather.meteo_swiss.pd.read_csv', side_effect=_mock_daily_csv)
    @patch('pyedautils.weather.meteo_swiss.requests.get', return_value=_mock_response(json_data=DAILY_STAC_ASSETS))
    def test_missing_param_warns(self, mock_get, mock_read_csv, mock_now):
        with self.assertLogs("pyedautils.weather.meteo_swiss", level="WARNING"):
            df = download_data("KLO", "2023-01-01", "2023-01-02", granularity="d", sensors=["temp", "xyz00000"])
        self.assertTrue(df["xyz00000"].isna().all())


class TestGetSmnStations(unittest.TestCase):

    @patch('pyedautils.weather.meteo_swiss.requests.get')
    def test_get_smn_stations(self, mock_get):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.content = MOCK_STATIONS_CSV.encode("cp1252")
        mock_get.return_value = response
        df = get_smn_stations()
        self.assertEqual(df.loc[0, "station_abbr"], "KLO")
        self.assertEqual(df.loc[0, "station_name"], "Zürich / Kloten")

    @patch('pyedautils.weather.meteo_swiss.requests.get', side_effect=Exception("Network error"))
    def test_get_smn_stations_network_error(self, mock_get):
        with self.assertRaises(ValueError):
            get_smn_stations()


if __name__ == '__main__':
    unittest.main()  # pragma: no cover


PARAMETERS_CSV = (
    "parameter_shortname;parameter_description_de;parameter_description_fr;"
    "parameter_description_it;parameter_description_en;parameter_group_de;parameter_group_fr;"
    "parameter_group_it;parameter_group_en;parameter_granularity;parameter_decimals;"
    "parameter_datatype;parameter_unit\n"
    "tre200h0;Lufttemperatur 2 m;Temp. 2 m;Temp. 2 m;Air temperature 2 m;Temperatur;"
    "Temperature;Temperatura;Temperature;H;1;Float;\u00b0C\n"
    "gre000h0;Globalstrahlung;Rayonnement;Radiazione;Global radiation;Strahlung;"
    "Rayonnement;Radiazione;Radiation;H;0;Integer;W/m\u00b2\n"
    "tre200d0;Lufttemperatur Tagesmittel;T;T;Air temperature daily mean;Temperatur;"
    "Temperature;Temperatura;Temperature;D;1;Float;\u00b0C\n"
)
INVENTORY_CSV = (
    "station_abbr;parameter_shortname;meas_cat_nr;data_since;data_till;owner\n"
    "LUZ;tre200h0;1;01.01.1981 00:00;;MeteoSchweiz\n"
    "LUZ;gre000h0;1;01.01.1981 00:00;31.12.2019 23:00;MeteoSchweiz\n"
    "LUZ;tre200d0;1;01.01.1864 00:00;;MeteoSchweiz\n"
    "KLO;tre200h0;1;01.01.1981 00:00;;MeteoSchweiz\n"
)


def _meta_text(url, timeout):
    return PARAMETERS_CSV if url.endswith("meta_parameters.csv") else INVENTORY_CSV


class TestStationParameters(unittest.TestCase):
    @patch("pyedautils.weather.meteo_swiss._get_text", side_effect=_meta_text)
    def test_parameters_of_station(self, mock_text):
        df = get_station_parameters("luz", granularity="h")
        self.assertEqual(list(df["parameter"]), ["gre000h0", "tre200h0"])  # by group
        temp = df[df["parameter"] == "tre200h0"].iloc[0]
        self.assertEqual(temp["description"], "Air temperature 2 m")
        self.assertEqual(temp["unit"], "\u00b0C")
        self.assertTrue(pd.isna(temp["data_till"]))

    @patch("pyedautils.weather.meteo_swiss._get_text", side_effect=_meta_text)
    def test_all_granularities_and_language(self, mock_text):
        df = get_station_parameters("LUZ", lang="de")
        self.assertEqual(len(df), 3)
        self.assertIn("Lufttemperatur 2 m", list(df["description"]))

    @patch("pyedautils.weather.meteo_swiss._get_text", side_effect=_meta_text)
    def test_unknown_station_is_empty(self, mock_text):
        self.assertTrue(get_station_parameters("XXX").empty)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            get_station_parameters("LUZ", lang="xx")

    @patch("pyedautils.weather.meteo_swiss._get_text", side_effect=Exception("offline"))
    def test_network_error(self, mock_text):
        with self.assertRaises(ValueError):
            get_smn_parameters()

