import unittest

import numpy as np
import pandas as pd

from pyedautils.degree_days import (
    degree_days,
    aggregate_degree_days,
    heating_degree_days,
    accumulated_temperature_difference,
    cooling_degree_days,
    ReferenceDegreeDays,
    climatology_from_monthly,
    reference_degree_days,
    climate_correction_factor,
    normalize_consumption,
    fit_degree_day_line,
    gts_correction_factor,
    GTS_MFH_2010_EXAMPLE,
    GTS_CH_2010_WEATHER,
)


def _const(value, start="2023-01-01", periods=31, freq="D", tz=None):
    idx = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    return pd.Series(float(value), index=idx)


class TestDegreeDayPrimitives(unittest.TestCase):

    def test_hgt_constant_january(self):
        res = heating_degree_days(_const(2.0))
        self.assertEqual(len(res), 1)
        self.assertAlmostEqual(res["hgt"].iloc[0], 31 * 18.0)
        self.assertEqual(res["ht"].iloc[0], 31)
        self.assertEqual(res["n_days"].iloc[0], 31)
        self.assertAlmostEqual(res["completeness"].iloc[0], 1.0)

    def test_hgt_above_heating_limit(self):
        res = heating_degree_days(_const(15.0))
        self.assertEqual(res["hgt"].iloc[0], 0.0)
        self.assertEqual(res["ht"].iloc[0], 0)

    def test_gt_without_heating_limit(self):
        res = heating_degree_days(_const(15.0, periods=30), heating_limit=None)
        self.assertAlmostEqual(res["hgt"].iloc[0], 30 * 5.0)
        self.assertEqual(res["ht"].iloc[0], 30)

    def test_heating_limit_inclusive(self):
        res = heating_degree_days(_const(12.0, periods=10))
        self.assertAlmostEqual(res["hgt"].iloc[0], 10 * 8.0)
        self.assertEqual(res["ht"].iloc[0], 10)

    def test_atd(self):
        res = accumulated_temperature_difference(_const(2.0, periods=10), base=12.0)
        self.assertAlmostEqual(res["atd"].iloc[0], 10 * 10.0)
        self.assertEqual(res["ht"].iloc[0], 10)
        # HGT 20/12 minus ATD 12 equals 8 K per heating day (Stadt Zürich data)
        hgt = heating_degree_days(_const(2.0, periods=10))["hgt"].iloc[0]
        self.assertAlmostEqual(hgt - res["atd"].iloc[0], 10 * 8.0)

    def test_cdd(self):
        res = cooling_degree_days(_const(25.0, start="2023-07-01", periods=10))
        self.assertAlmostEqual(res["cdd"].iloc[0], 10 * (25.0 - 18.3))
        self.assertEqual(res["cd"].iloc[0], 10)
        res = cooling_degree_days(_const(15.0, start="2023-07-01", periods=10))
        self.assertEqual(res["cdd"].iloc[0], 0.0)

    def test_input_variants_identical(self):
        s = _const(3.0, periods=20)
        pos = pd.DataFrame({"timestamp": s.index, "value": s.to_numpy()})
        idx_df = pd.DataFrame({"temp": s})
        a = heating_degree_days(s)["hgt"].iloc[0]
        b = heating_degree_days(pos)["hgt"].iloc[0]
        c = heating_degree_days(idx_df)["hgt"].iloc[0]
        self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(a, c)

    def test_daily_output(self):
        res = heating_degree_days(_const(2.0, periods=5), freq=None)
        self.assertEqual(len(res), 5)
        self.assertListEqual(list(res.columns), ["t_mean", "hgt", "is_dd_day", "coverage"])
        self.assertTrue(res["is_dd_day"].all())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            degree_days(_const(2.0), base=12.0, threshold=20.0)
        with self.assertRaises(ValueError):
            degree_days(pd.Series([1.0, 2.0]))
        with self.assertRaises(ValueError):
            degree_days(_const(2.0), mode="warming")
        with self.assertRaises(ValueError):
            degree_days(pd.DataFrame({"timestamp": [], "value": []}))
        with self.assertRaises(ValueError):
            degree_days(_const(2.0, periods=10, freq="W"))


class TestHourlyInput(unittest.TestCase):

    def setUp(self):
        idx = pd.date_range("2023-01-01", periods=24 * 5, freq="h")
        self.hourly = pd.Series(np.tile([10.0, 30.0], 24 * 5 // 2), index=idx)

    def test_daily_mean_vs_hourly_method(self):
        daily = heating_degree_days(self.hourly, room_temp=20.0, heating_limit=None, freq=None)
        self.assertAlmostEqual(daily["hgt"].iloc[0], 0.0)
        hourly = heating_degree_days(self.hourly, room_temp=20.0, heating_limit=None,
                                     freq=None, method="hourly")
        self.assertAlmostEqual(hourly["hgt"].iloc[0], 5.0)
        self.assertTrue(hourly["is_dd_day"].iloc[0])

    def test_low_coverage_day_is_nan(self):
        s = self.hourly.copy()
        s.iloc[6:24] = np.nan
        res = degree_days(s, base=20.0, threshold=None)
        self.assertTrue(np.isnan(res["dd"].iloc[0]))
        self.assertAlmostEqual(res["coverage"].iloc[0], 0.25)
        self.assertFalse(res["is_dd_day"].iloc[0])

    def test_hourly_method_requires_subdaily(self):
        with self.assertRaises(ValueError):
            degree_days(_const(2.0), method="hourly")


class TestAggregate(unittest.TestCase):

    def test_incomplete_month(self):
        res = heating_degree_days(_const(0.0, periods=15))
        self.assertAlmostEqual(res["hgt"].iloc[0], 300.0)
        self.assertEqual(res["n_days"].iloc[0], 31)
        self.assertEqual(res["n_valid"].iloc[0], 15)
        self.assertAlmostEqual(res["completeness"].iloc[0], 15 / 31)

    def test_scale_incomplete(self):
        res = heating_degree_days(_const(0.0, periods=15), scale_incomplete=True)
        self.assertAlmostEqual(res["hgt"].iloc[0], 620.0)
        self.assertAlmostEqual(res["ht"].iloc[0], 31.0)

    def test_yearly(self):
        res = heating_degree_days(_const(0.0, periods=365), freq="YS")
        self.assertEqual(len(res), 1)
        self.assertAlmostEqual(res["hgt"].iloc[0], 365 * 20.0)

    def test_season(self):
        s = pd.Series(5.0, index=pd.date_range("2022-07-01", "2023-06-30", freq="D"))
        res = heating_degree_days(s, freq="season")
        self.assertEqual(len(res), 1)
        self.assertEqual(res["season"].iloc[0], "2022/2023")
        self.assertEqual(res["n_days"].iloc[0], 365)
        self.assertAlmostEqual(res["completeness"].iloc[0], 1.0)

    def test_half_season(self):
        s = pd.Series(5.0, index=pd.date_range("2023-01-01", "2023-06-30", freq="D"))
        res = heating_degree_days(s, freq="season")
        self.assertEqual(res["season"].iloc[0], "2022/2023")
        self.assertAlmostEqual(res["completeness"].iloc[0], 181 / 365, places=3)

    def test_rejects_deprecated_freq(self):
        daily = degree_days(_const(0.0))
        with self.assertRaises(ValueError):
            aggregate_degree_days(daily, "M")

    def test_none_returns_daily(self):
        daily = degree_days(_const(0.0))
        self.assertIs(aggregate_degree_days(daily, None), daily)

    def test_tz_aware_local_day(self):
        idx = pd.date_range("2023-01-01 00:00", "2023-01-31 23:00", freq="h", tz="Europe/Zurich")
        s = pd.Series(0.0, index=idx)
        res = heating_degree_days(s)
        self.assertEqual(len(res), 1)
        self.assertEqual(res["n_valid"].iloc[0], 31)
        self.assertIsNotNone(res.index.tz)
        self.assertAlmostEqual(res["hgt"].iloc[0], 31 * 20.0)


class TestReference(unittest.TestCase):

    def setUp(self):
        idx = pd.date_range("2018-01-01", "2020-12-31", freq="D")
        self.temp = pd.Series(idx.month - 1.0, index=idx)

    def test_reference_degree_days(self):
        with self.assertLogs("pyedautils.degree_days", level="WARNING"):
            ref = reference_degree_days(self.temp, period=("2018", "2020"))
        self.assertIsInstance(ref, ReferenceDegreeDays)
        self.assertEqual(ref.n_years, 3)
        self.assertEqual(ref.period, ("2018", "2020"))
        self.assertAlmostEqual(ref.monthly[1], 31 * 20.0)
        self.assertAlmostEqual(ref.monthly[2], (28 + 28 + 29) / 3 * 19.0)
        self.assertAlmostEqual(ref.monthly[12], 31 * 9.0)
        self.assertAlmostEqual(ref.annual, ref.monthly.sum())
        self.assertAlmostEqual(ref.monthly_days[1], 31.0)

    def test_incomplete_month_excluded(self):
        temp = self.temp.copy()
        temp.loc["2020-01-01":"2020-01-20"] = np.nan
        ref = reference_degree_days(temp, period=("2018", "2020"))
        self.assertAlmostEqual(ref.monthly[1], 31 * 20.0)

    def test_climatology_from_series(self):
        monthly = heating_degree_days(self.temp)["hgt"]
        ref = climatology_from_monthly(monthly, period=None)
        self.assertAlmostEqual(ref.monthly[1], 31 * 20.0)
        self.assertTrue(np.isnan(ref.monthly_days[1]))

    def test_period_outside_data(self):
        with self.assertRaises(ValueError):
            reference_degree_days(self.temp, period=("1991", "2000"))


class TestCorrection(unittest.TestCase):

    def test_scalar_factor(self):
        self.assertAlmostEqual(climate_correction_factor(800, 1000), 1.25)

    def test_series_against_climatology(self):
        actual = pd.Series([400.0, 200.0], index=pd.to_datetime(["2023-01-01", "2023-02-01"]))
        clim = pd.Series(np.arange(1, 13) * 100.0, index=range(1, 13))
        f = climate_correction_factor(actual, clim)
        self.assertAlmostEqual(f.iloc[0], 0.25)
        self.assertAlmostEqual(f.iloc[1], 1.0)

    def test_reference_degree_days_object(self):
        monthly = pd.Series(100.0, index=pd.RangeIndex(1, 13))
        ref = ReferenceDegreeDays(monthly, monthly, 1200.0, 1200.0, 30, ("1991", "2020"))
        self.assertAlmostEqual(climate_correction_factor(1000.0, ref), 1.2)
        actual = pd.Series([50.0], index=pd.to_datetime(["2023-03-01"]))
        self.assertAlmostEqual(climate_correction_factor(actual, ref).iloc[0], 2.0)

    def test_zero_degree_days(self):
        with self.assertLogs("pyedautils.degree_days", level="WARNING"):
            self.assertTrue(np.isnan(climate_correction_factor(0.0, 1000.0)))

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            climate_correction_factor(-5.0, 1000.0)

    def test_normalize_consumption(self):
        self.assertAlmostEqual(normalize_consumption(10000, 800, 1000, 0.7), 11750.0)
        self.assertAlmostEqual(normalize_consumption(10000, 800, 1000), 12500.0)
        with self.assertRaises(ValueError):
            normalize_consumption(10000, 800, 1000, 1.2)

    def test_normalize_series(self):
        idx = pd.to_datetime(["2023-01-01", "2023-02-01"])
        energy = pd.Series([1000.0, 500.0], index=idx)
        actual = pd.Series([400.0, 250.0], index=idx)
        ref = pd.Series([500.0, 250.0], index=idx)
        res = normalize_consumption(energy, actual, ref, 0.5)
        self.assertAlmostEqual(res.iloc[0], 500.0 + 500.0 * 1.25)
        self.assertAlmostEqual(res.iloc[1], 500.0)


class TestFitDegreeDayLine(unittest.TestCase):

    def setUp(self):
        idx = pd.date_range("2023-01-01", periods=365, freq="D")
        self.temp = pd.Series(np.linspace(-5, 25, 365), index=idx)
        self.energy = 50.0 + 3.0 * (12.0 - self.temp).clip(lower=0)

    def test_daily_fit(self):
        line = fit_degree_day_line(self.energy, self.temp, base=12.0)
        self.assertAlmostEqual(line.intercept, 50.0, places=6)
        self.assertAlmostEqual(line.slope, 3.0, places=6)
        self.assertGreater(line.r2, 0.999)
        self.assertLess(abs(line.cvrmse), 1e-6)
        self.assertEqual(line.n, 365)
        self.assertEqual(line.freq, "D")
        self.assertIn("fitted", line.data.columns)

    def test_monthly_fit(self):
        monthly = self.energy.resample("MS").sum()
        line = fit_degree_day_line(monthly, self.temp, base=12.0, freq="MS")
        self.assertAlmostEqual(line.intercept, 50.0, places=6)
        self.assertAlmostEqual(line.slope, 3.0, places=6)
        self.assertEqual(line.n, 12)

    def test_noisy_fit(self):
        rng = np.random.default_rng(1)
        energy = self.energy + rng.normal(0, 2.0, len(self.energy))
        line = fit_degree_day_line(energy, self.temp, base=12.0)
        self.assertAlmostEqual(line.slope, 3.0, delta=0.15)
        self.assertGreater(line.r2, 0.9)

    def test_errors(self):
        with self.assertRaises(ValueError):
            fit_degree_day_line(self.energy, self.temp, freq="W")
        with self.assertRaises(ValueError):
            fit_degree_day_line(self.energy.iloc[:2], self.temp)
        tz_energy = self.energy.tz_localize("UTC")
        with self.assertRaises(ValueError):
            fit_degree_day_line(tz_energy, self.temp)


class TestGTS(unittest.TestCase):

    def _run(self, gt_actual, rad_actual, share, e_gt, e_rad):
        return gts_correction_factor(gt_actual, [100.0] * 12, rad_actual, [100.0] * 12, share, e_gt, e_rad)

    def test_zero_elasticities(self):
        res = self._run([110.0] * 12, [90.0] * 12, [1 / 12] * 12, ([0.0] * 12, [0.0] * 12), ([0.0] * 12, [0.0] * 12))
        self.assertAlmostEqual(res.ratio, 1.0)
        self.assertAlmostEqual(res.factor, 1.0)

    def test_single_month(self):
        share = [1.0] + [0.0] * 11
        gt = [110.0] + [100.0] * 11
        res = self._run(gt, [100.0] * 12, share, ([1.0] * 12, [0.0] * 12), ([0.0] * 12, [0.0] * 12))
        self.assertAlmostEqual(res.ratio, 1.1)
        self.assertAlmostEqual(res.factor, 1 / 1.1)
        self.assertAlmostEqual(res.monthly.loc[1, "d_gt"], 0.1)

    def test_mapping_elasticities(self):
        share = [1.0] + [0.0] * 11
        gt = [110.0] + [100.0] * 11
        e = {m: (1.0, 2.0) for m in range(1, 13)}
        res = self._run(gt, [100.0] * 12, share, e, {m: (0.0, 0.0) for m in range(1, 13)})
        self.assertAlmostEqual(res.monthly.loc[1, "e_gt"], 1.2)
        self.assertAlmostEqual(res.ratio, 1.12)

    def test_share_sum_validated(self):
        with self.assertRaises(ValueError):
            self._run([100.0] * 12, [100.0] * 12, [0.5] * 12, ([0.0] * 12, [0.0] * 12), ([0.0] * 12, [0.0] * 12))
        with self.assertRaises(ValueError):
            self._run([100.0] * 11, [100.0] * 12, [1 / 12] * 12, ([0.0] * 12, [0.0] * 12), ([0.0] * 12, [0.0] * 12))

    def test_prognos_mfh_2010_example(self):
        ex = GTS_MFH_2010_EXAMPLE
        res = gts_correction_factor(
            GTS_CH_2010_WEATHER["gt"], ex["gt_norm"], GTS_CH_2010_WEATHER["rad"], ex["rad_norm"],
            ex["load_share_norm"], ex["elasticity_gt"], ex["elasticity_rad"],
        )
        # Documented: 1.063 (January effect 12.7 %, January share 22.1 %)
        self.assertAlmostEqual(res.ratio, 1.063, delta=0.003)
        self.assertAlmostEqual(res.monthly.loc[1, "effect"], 0.127, delta=0.003)
        self.assertAlmostEqual(res.monthly.loc[1, "contribution"], 0.221, delta=0.002)

    def test_series_inputs(self):
        idx = pd.date_range("2010-01-01", periods=12, freq="MS")
        ex = GTS_MFH_2010_EXAMPLE
        res = gts_correction_factor(
            pd.Series(GTS_CH_2010_WEATHER["gt"], index=idx), pd.Series(ex["gt_norm"], index=range(1, 13)),
            pd.Series(GTS_CH_2010_WEATHER["rad"], index=idx), ex["rad_norm"],
            ex["load_share_norm"], ex["elasticity_gt"], ex["elasticity_rad"],
        )
        self.assertAlmostEqual(res.ratio, 1.063, delta=0.003)


if __name__ == "__main__":
    unittest.main()
