# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException

from pyedautils.geopy import (
    get_altitude_lat_long,
    get_lat_long_address,
    get_altitude_lv95,
    convert_wsg84_to_lv95,
    get_coordindates_ch_plz,
    get_distance_between_two_points,
    GeocodingError,
)


class TestConvertWsg84ToLv95(unittest.TestCase):

    @patch('pyedautils.geopy.requests.get')
    def test_valid_coordinates(self, mock_get):
        mock_get.return_value.json.return_value = {
            "coordinates": [2665960.0, 1207350.0]
        }
        result = convert_wsg84_to_lv95(47.013, 8.306)
        self.assertEqual(result, [2665960.0, 1207350.0])
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs['timeout'], 30)

    def test_lat_out_of_range(self):
        with self.assertRaises(GeocodingError):
            convert_wsg84_to_lv95(57.0, 8.3)

    def test_long_out_of_range(self):
        with self.assertRaises(GeocodingError):
            convert_wsg84_to_lv95(47.0, 15.0)

    @patch('pyedautils.geopy.requests.get')
    def test_request_failure(self, mock_get):
        mock_get.side_effect = RequestException("No internet")
        with self.assertRaises(GeocodingError):
            convert_wsg84_to_lv95(47.013, 8.306)


class TestGetAltitudeLv95(unittest.TestCase):

    @patch('pyedautils.geopy.requests.get')
    def test_valid_coordinates(self, mock_get):
        mock_get.return_value.json.return_value = {"height": "440.5"}
        result = get_altitude_lv95([2665960.0, 1207350.0])
        self.assertEqual(result, 440.5)

    @patch('pyedautils.geopy.requests.get')
    def test_request_failure(self, mock_get):
        mock_get.side_effect = RequestException("No internet")
        with self.assertRaises(GeocodingError):
            get_altitude_lv95([2665960.0, 1207350.0])


class TestGetAltitudeLatLong(unittest.TestCase):

    @patch('pyedautils.geopy.time.sleep')
    @patch('pyedautils.geopy.requests.get')
    def test_valid_coordinates(self, mock_get, mock_sleep):
        mock_get.return_value.json.return_value = {
            "results": [{"elevation": 440.2, "location": {"lat": 47.01, "lng": 8.31}}]
        }
        result = get_altitude_lat_long(47.01, 8.31)
        self.assertEqual(result, 440.2)
        mock_sleep.assert_called_once_with(1)

    @patch('pyedautils.geopy.time.sleep')
    @patch('pyedautils.geopy.requests.get')
    def test_elevation_none(self, mock_get, mock_sleep):
        mock_get.return_value.json.return_value = {
            "results": [{"elevation": None, "location": {"lat": 47.01, "lng": 8.31}}]
        }
        result = get_altitude_lat_long(47.01, 8.31)
        self.assertEqual(result, 0)

    def test_lat_too_high(self):
        with self.assertRaises(ValueError):
            get_altitude_lat_long(57.0, 8.3)

    def test_lat_too_low(self):
        with self.assertRaises(ValueError):
            get_altitude_lat_long(10.0, 8.3)

    def test_long_too_high(self):
        with self.assertRaises(ValueError):
            get_altitude_lat_long(47.0, 200.0)

    def test_long_too_low(self):
        with self.assertRaises(ValueError):
            get_altitude_lat_long(47.0, -200.0)

    @patch('pyedautils.geopy.requests.get')
    def test_request_failure(self, mock_get):
        mock_get.side_effect = RequestException("No internet")
        with self.assertRaises(GeocodingError):
            get_altitude_lat_long(47.01, 8.31)


class TestGetLatLongAddress(unittest.TestCase):

    @patch('pyedautils.geopy.Nominatim')
    def test_valid_address(self, mock_nominatim_cls):
        mock_location = MagicMock()
        mock_location.latitude = 47.0145
        mock_location.longitude = 8.3062
        mock_nom = MagicMock()
        mock_nom.geocode.return_value = mock_location
        mock_nominatim_cls.return_value = mock_nom

        result = get_lat_long_address("Technikumstrasse 21, 6048 Horw")
        self.assertAlmostEqual(result[0], 47.0145)
        self.assertAlmostEqual(result[1], 8.3062)

    @patch('pyedautils.geopy.time.sleep')
    @patch('pyedautils.geopy.Nominatim')
    def test_geocode_fails_all_attempts(self, mock_nominatim_cls, mock_sleep):
        mock_nom = MagicMock()
        mock_nom.geocode.return_value = None
        mock_nominatim_cls.return_value = mock_nom

        with self.assertRaises(GeocodingError):
            get_lat_long_address("Nonexistent place")

    @patch('pyedautils.geopy.Nominatim')
    def test_geocode_exception(self, mock_nominatim_cls):
        mock_nom = MagicMock()
        mock_nom.geocode.side_effect = Exception("Service unavailable")
        mock_nominatim_cls.return_value = mock_nom

        with self.assertRaises(GeocodingError):
            get_lat_long_address("Some address")


class TestGetCoordinatesChPlz(unittest.TestCase):

    @patch('pyedautils.geopy.pgeocode.Nominatim')
    def test_valid_plz(self, mock_nominatim_cls):
        mock_data = MagicMock()
        mock_data.latitude.astype.return_value = 47.0145
        mock_data.longitude.astype.return_value = 8.3062
        mock_nom = MagicMock()
        mock_nom.query_postal_code.return_value = mock_data
        mock_nominatim_cls.return_value = mock_nom

        result = get_coordindates_ch_plz(6048)
        self.assertEqual(result, (47.0145, 8.3062))

    @patch('pyedautils.geopy.pgeocode.Nominatim')
    def test_invalid_plz(self, mock_nominatim_cls):
        mock_data = MagicMock()
        mock_data.latitude.astype.return_value = float('nan')
        mock_data.longitude.astype.return_value = float('nan')
        mock_nom = MagicMock()
        mock_nom.query_postal_code.return_value = mock_data
        mock_nominatim_cls.return_value = mock_nom

        with self.assertRaises(GeocodingError):
            get_coordindates_ch_plz(424242)

    @patch('pyedautils.geopy.pgeocode.Nominatim')
    def test_pgeocode_exception(self, mock_nominatim_cls):
        mock_nominatim_cls.side_effect = Exception("pgeocode error")

        with self.assertRaises(GeocodingError):
            get_coordindates_ch_plz(6048)


class TestGetDistanceBetweenTwoPoints(unittest.TestCase):

    def test_known_distance(self):
        # Horw to Interlaken ~50km
        coord1 = (47.0145, 8.3062)
        coord2 = (46.6863, 7.8632)
        result = get_distance_between_two_points(coord1, coord2)
        self.assertAlmostEqual(result, 50.0, delta=5.0)

    def test_same_point(self):
        coord = (47.0, 8.3)
        result = get_distance_between_two_points(coord, coord)
        self.assertEqual(result, 0.0)

    def test_long_distance(self):
        # Zurich to New York ~6300km
        zurich = (47.3769, 8.5417)
        new_york = (40.7128, -74.0060)
        result = get_distance_between_two_points(zurich, new_york)
        self.assertAlmostEqual(result, 6324.0, delta=50.0)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
