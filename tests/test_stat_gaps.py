"""Gap handling before the ACF and the STL decomposition (plots/stat.py)."""

import unittest
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from pyedautils.plots import plot_autocorrelation, plot_decomposition
from pyedautils.plots.stat import (
    _fill_short_gaps, _infer_step, _longest_valid_run, _regularize,
)


def _hourly_sine(n_days=30, start="2024-01-01"):
    idx = pd.date_range(start, periods=n_days * 24, freq="h")
    return pd.Series(np.sin(2 * np.pi * np.arange(len(idx)) / 24), index=idx)


class TestInferStep(unittest.TestCase):
    def test_too_short(self):
        self.assertIsNone(_infer_step(pd.DatetimeIndex(["2024-01-01"])))

    def test_regular_alias(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        self.assertEqual(pd.Timedelta(pd.tseries.frequencies.to_offset(_infer_step(idx))),
                         pd.Timedelta("15min"))

    def test_two_points_timedelta(self):
        idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00"])
        self.assertEqual(_infer_step(idx), pd.Timedelta("1h"))

    def test_gappy_hourly_uses_most_common_step(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="h").delete([3, 4])
        self.assertEqual(_infer_step(idx), pd.Timedelta("1h"))

    def test_gappy_month_start(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"])
        self.assertIsInstance(_infer_step(idx), pd.offsets.MonthBegin)

    def test_gappy_month_end(self):
        idx = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-04-30", "2020-05-31"])
        self.assertIsInstance(_infer_step(idx), pd.offsets.MonthEnd)

    def test_thirty_days_not_on_month_boundary(self):
        idx = pd.DatetimeIndex(["2020-01-05", "2020-02-04", "2020-04-04", "2020-05-04"])
        self.assertEqual(_infer_step(idx), pd.Timedelta(days=30))


class TestFillShortGaps(unittest.TestCase):
    def test_long_gap_not_partly_filled(self):
        s = _hourly_sine(2)
        s.iloc[5:15] = np.nan   # 10 NaN
        out = _fill_short_gaps(s, 6)
        self.assertEqual(int(out.iloc[5:15].isna().sum()), 10)

    def test_short_gap_filled(self):
        s = pd.Series([0.0, np.nan, np.nan, 3.0],
                      index=pd.date_range("2024-01-01", periods=4, freq="h"))
        out = _fill_short_gaps(s, 6)
        np.testing.assert_allclose(out.to_numpy(), [0, 1, 2, 3])

    def test_edges_not_filled(self):
        s = pd.Series([np.nan, 1.0, 2.0, np.nan])
        out = _fill_short_gaps(s, 6)
        self.assertTrue(np.isnan(out.iloc[0]) and np.isnan(out.iloc[-1]))

    def test_positional_index(self):
        s = pd.Series([0.0, np.nan, 2.0])
        self.assertEqual(_fill_short_gaps(s, 1).iloc[1], 1.0)

    def test_disabled_and_no_gaps(self):
        s = pd.Series([0.0, np.nan, 2.0])
        self.assertTrue(np.isnan(_fill_short_gaps(s, 0).iloc[1]))
        full = pd.Series([1.0, 2.0])
        self.assertIs(_fill_short_gaps(full, 6), full)


class TestRegularize(unittest.TestCase):
    def test_missing_timestamps_become_gaps(self):
        s = _hourly_sine(2).drop(pd.date_range("2024-01-01 05:00", periods=10, freq="h"))
        out = _regularize(s, 6)
        self.assertEqual(len(out), 48)
        self.assertEqual(int(out.isna().sum()), 10)

    def test_short_missing_timestamps_filled(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        s = pd.Series(np.arange(10.0), index=idx)
        out = _regularize(s.drop(idx[[5, 6]]), 6)
        np.testing.assert_allclose(out.to_numpy(), np.arange(10.0))

    def test_unsorted_and_duplicates(self):
        idx = pd.DatetimeIndex(["2024-01-01 02:00", "2024-01-01 00:00",
                                "2024-01-01 01:00", "2024-01-01 01:00"])
        out = _regularize(pd.Series([3.0, 1.0, 2.0, 4.0], index=idx), 0)
        np.testing.assert_allclose(out.to_numpy(), [1.0, 3.0, 3.0])

    def test_monthly_gap(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-04-01"])
        out = _regularize(pd.Series([1.0, 2.0, 3.0], index=idx), 0)
        self.assertEqual(list(out.index.month), [1, 2, 3, 4])
        self.assertTrue(np.isnan(out.iloc[2]))

    def test_trims_edges_and_all_nan(self):
        s = pd.Series([np.nan, 1.0, 2.0, np.nan],
                      index=pd.date_range("2024-01-01", periods=4, freq="h"))
        self.assertEqual(len(_regularize(s, 6)), 2)
        self.assertEqual(len(_regularize(s * np.nan, 6)), 0)

    def test_single_value_and_positional(self):
        one = pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-01"]))
        self.assertEqual(len(_regularize(one, 6)), 1)
        pos = pd.Series([1.0, np.nan, 3.0])
        np.testing.assert_allclose(_regularize(pos, 6).to_numpy(), [1, 2, 3])


class TestLongestValidRun(unittest.TestCase):
    def test_picks_longest(self):
        s = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0])
        self.assertEqual(list(_longest_valid_run(s)), [3.0, 4.0, 5.0])


def _drop_half_periods(s, n_gaps=6, gap=12):
    """Remove ``n_gaps`` blocks of ``gap`` timestamps (half a daily period)."""
    drop = []
    for k in range(n_gaps):
        start = 50 + k * 100
        drop.extend(s.index[start:start + gap])
    return s.drop(drop)


class TestAutocorrelationGaps(unittest.TestCase):
    def test_lag_alignment_kept_across_missing_timestamps(self):
        s = _drop_half_periods(_hourly_sine(30))
        fig = plot_autocorrelation(s.to_frame("v"), lags=48)
        acf_values = np.asarray(fig.data[1].y)
        # Old behaviour: dropping the gaps puts half-period shifted samples
        # next to each other and smears the daily peak at lag 24.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = acf(s.interpolate(limit=6).dropna().to_numpy(), nlags=48)
        self.assertGreater(acf_values[24], 0.8)
        self.assertGreater(acf_values[24], old[24] + 0.2)
        self.assertLess(acf_values[12], -0.8)

    def test_long_nan_gap_not_partly_filled(self):
        s = _hourly_sine(30)
        s.iloc[100:130] = np.nan
        fig = plot_autocorrelation(s.to_frame("v"), lags=48)
        self.assertGreater(np.asarray(fig.data[1].y)[24], 0.9)

    def test_max_gap_argument(self):
        s = _hourly_sine(10)
        s.iloc[[10, 11]] = np.nan
        f0 = plot_autocorrelation(s.to_frame("v"), lags=24, max_gap=0)
        f6 = plot_autocorrelation(s.to_frame("v"), lags=24, max_gap=6)
        self.assertFalse(np.allclose(f0.data[1].y, f6.data[1].y))

    def test_regular_series_unchanged(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=300, freq="h")
        df = pd.DataFrame({"v": rng.normal(size=300)}, index=idx)
        fig = plot_autocorrelation(df, lags=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = acf(df["v"].to_numpy(), nlags=10)
        np.testing.assert_allclose(fig.data[1].y, expected)


class TestDecompositionGaps(unittest.TestCase):
    @staticmethod
    def _df(s):
        return pd.DataFrame({"timestamp": s.index, "value": s.to_numpy()})

    def test_short_gaps_filled_no_warning(self):
        s = _hourly_sine(10)
        s = s.drop(s.index[[30, 31, 32]])
        s.iloc[60:63] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_decomposition(self._df(s), period=24)
        self.assertEqual(len(fig.data[0].x), 240)
        self.assertFalse(np.isnan(np.asarray(fig.data[0].y, dtype=float)).any())

    def test_long_gap_uses_longest_segment(self):
        s = _hourly_sine(20)
        s = s.drop(s.index[100:130])    # 30 missing timestamps
        with self.assertWarnsRegex(UserWarning, "longest contiguous segment"):
            fig = plot_decomposition(self._df(s), period=24)
        x = pd.to_datetime(fig.data[0].x)
        self.assertEqual(len(x), 480 - 130)
        self.assertEqual(x[0], pd.Timestamp("2024-01-06 10:00"))
        self.assertFalse(np.isnan(np.asarray(fig.data[1].y, dtype=float)).any())

    def test_too_short_raises(self):
        s = _hourly_sine(10)
        s.iloc[30:200] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "two full periods"):
                plot_decomposition(self._df(s), period=24)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
