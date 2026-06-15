# -*- coding: utf-8 -*-

import unittest

import pandas as pd

from pyedautils.data_prep.reshape import (
    aggregate_long,
    remove_outliers_iqr,
    to_wide,
)


def _long():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    return pd.DataFrame({
        "datapoint_id": ["a"] * 5 + ["b"] * 5,
        "timestamp": list(idx) + list(idx),
        "value": [1, 2, 3, 100, 5, 10, 11, 12, 13, 14],
    })


class TestAggregateLong(unittest.TestCase):
    def test_resample_per_id(self):
        out = aggregate_long(_long(), "2h", "mean")
        self.assertIn("datapoint_id", out.columns)
        self.assertIn("timestamp", out.columns)
        # 5 hours -> 3 two-hour bins per id -> 6 rows
        self.assertEqual(len(out), 6)

    def test_none_freq_passthrough(self):
        df = _long()
        self.assertIs(aggregate_long(df, None), df)


class TestRemoveOutliersIqr(unittest.TestCase):
    def test_drops_outlier(self):
        out = remove_outliers_iqr(_long(), multiplier=1.5)
        # the 100 in series 'a' should be removed
        self.assertNotIn(100, out["value"].tolist())
        self.assertEqual(len(out), 9)

    def test_empty(self):
        empty = _long().iloc[0:0]
        self.assertTrue(remove_outliers_iqr(empty).empty)


class TestToWide(unittest.TestCase):
    def test_pivot_default_labels(self):
        wide = to_wide(_long())
        self.assertListEqual(sorted(wide.columns), ["a", "b"])
        self.assertEqual(wide.index.name, "timestamp")

    def test_label_mapping(self):
        wide = to_wide(_long(), {"a": "Room A", "b": "Room B"})
        self.assertListEqual(sorted(wide.columns), ["Room A", "Room B"])

    def test_label_mapping_drops_unmapped(self):
        wide = to_wide(_long(), {"a": "Room A"})
        self.assertListEqual(list(wide.columns), ["Room A"])

    def test_empty(self):
        self.assertTrue(to_wide(_long().iloc[0:0]).empty)


if __name__ == "__main__":
    unittest.main()
