# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from pyedautils.weather.open_data_ch import (
    download_degree_days_bfe,
    download_degree_days_zurich,
)

MOCK_BFE_CSV = "Jahr,Monat,HGT,CDD\n1980,1,657,0\n1980,2,488.8,0\n1980,7,20.5,30.1\n"

MOCK_ZURICH_CSV = (
    '﻿"Jahr","Monat","Jahr_Monat","Standort","Heizgradtag","akkumulierteTemperaturdifferenz"\n'
    '1864,1,"1864-01","Fluntern",797,549\n'
    '1864,2,"1864-02","Fluntern",598,366\n'
    '1864,1,"1864-01","Stampfenbachstrasse",780,540\n'
)


def _mock_response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.raise_for_status.return_value = None
    return response


class TestDownloadDegreeDaysBfe(unittest.TestCase):

    @patch("pyedautils.weather.open_data_ch.requests.get")
    def test_returns_monthly_frame(self, mock_get):
        mock_get.return_value = _mock_response(MOCK_BFE_CSV)
        df = download_degree_days_bfe()
        self.assertListEqual(list(df.columns), ["hgt", "cdd"])
        self.assertEqual(len(df), 3)
        self.assertEqual(df.index[0], pd.Timestamp("1980-01-01"))
        self.assertAlmostEqual(df["hgt"].iloc[1], 488.8)
        self.assertAlmostEqual(df["cdd"].iloc[2], 30.1)
        self.assertEqual(df.index.name, "period")

    @patch("pyedautils.weather.open_data_ch.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        with self.assertRaises(ValueError):
            download_degree_days_bfe()

    @patch("pyedautils.weather.open_data_ch.requests.get")
    def test_unexpected_format(self, mock_get):
        mock_get.return_value = _mock_response("a,b\n1,2\n")
        with self.assertRaises(ValueError):
            download_degree_days_bfe()


class TestDownloadDegreeDaysZurich(unittest.TestCase):

    @patch("pyedautils.weather.open_data_ch.requests.get")
    def test_all_stations(self, mock_get):
        mock_get.return_value = _mock_response(MOCK_ZURICH_CSV)
        df = download_degree_days_zurich()
        self.assertListEqual(list(df.columns), ["station", "hgt", "atd"])
        self.assertEqual(len(df), 3)
        self.assertEqual(df["hgt"].iloc[0], 797)
        # HGT 20/12 minus ATD 12 equals 8 K per heating day in January
        self.assertEqual(df["hgt"].iloc[0] - df["atd"].iloc[0], 31 * 8)

    @patch("pyedautils.weather.open_data_ch.requests.get")
    def test_station_filter(self, mock_get):
        mock_get.return_value = _mock_response(MOCK_ZURICH_CSV)
        df = download_degree_days_zurich("Fluntern")
        self.assertEqual(len(df), 2)
        self.assertTrue((df["station"] == "Fluntern").all())

    def test_unknown_station(self):
        with self.assertRaises(ValueError):
            download_degree_days_zurich("Bern")

    @patch("pyedautils.weather.open_data_ch.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        with self.assertRaises(ValueError):
            download_degree_days_zurich()

    @patch("pyedautils.weather.open_data_ch.requests.get")
    def test_unexpected_format(self, mock_get):
        mock_get.return_value = _mock_response("a,b\n1,2\n")
        with self.assertRaises(ValueError):
            download_degree_days_zurich()


if __name__ == "__main__":
    unittest.main()
