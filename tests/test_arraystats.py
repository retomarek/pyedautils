# -*- coding: utf-8 -*-

import math
import unittest

import numpy as np

from pyedautils.arraystats import (
    bucket_coverage,
    find_gaps,
    interval_overlap,
    interval_segments,
    local_interval,
    describe,
    detect_gaps,
    held_at,
    hold_weights,
    bucket_reduce,
    step_bucket_mean,
    step_bucket_reduce,
    weighted_stats,
    gap_indices,
    detect_stuck,
    infer_interval,
    outliers_hampel,
    outliers_iqr,
    outliers_zscore,
)

MIN = 60_000  # one minute in ms


class TestInferInterval(unittest.TestCase):
    def test_regular(self):
        ts = np.arange(100, dtype=np.int64) * MIN
        self.assertEqual(infer_interval(ts), MIN)

    def test_datetime64(self):
        ts = np.arange("2024-01-01", "2024-01-02", dtype="datetime64[m]")
        self.assertEqual(infer_interval(ts), 1)

    def test_too_short(self):
        self.assertIsNone(infer_interval(np.array([5])))
        self.assertIsNone(infer_interval(np.array([5, 5])))


class TestDescribe(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        values = rng.normal(20, 5, 10_000)
        d = describe(values)
        self.assertEqual(d["n"], 10_000)
        self.assertAlmostEqual(d["mean"], values.mean())
        self.assertAlmostEqual(d["median"], np.median(values))
        self.assertAlmostEqual(d["std"], values.std())
        self.assertAlmostEqual(d["p05"], np.percentile(values, 5))
        self.assertAlmostEqual(d["p95"], np.percentile(values, 95))
        self.assertTrue(math.isnan(d["coverage"]))

    def test_ignores_nan_and_reports_timestamps(self):
        ts = np.arange(5, dtype=np.int64) * MIN
        values = np.array([1.0, np.nan, 9.0, -3.0, np.nan])
        d = describe(values, ts)
        self.assertEqual(d["n"], 3)
        self.assertEqual(d["n_missing"], 2)
        self.assertEqual(d["min"], -3.0)
        self.assertEqual(d["ts_min"], 3 * MIN)
        self.assertEqual(d["ts_max"], 2 * MIN)
        self.assertEqual(d["ts_first"], 0)
        self.assertEqual(d["ts_last"], 3 * MIN)
        self.assertAlmostEqual(d["coverage"], 3 / 5)

    def test_coverage_for_explicit_range(self):
        ts = np.arange(60, dtype=np.int64) * MIN  # one hour of data
        d = describe(np.ones(60), ts, expected_interval=MIN, start=0, end=120 * MIN)
        self.assertAlmostEqual(d["coverage"], 0.5)

    def test_empty(self):
        d = describe(np.array([np.nan, np.nan]), np.array([0, MIN]))
        self.assertEqual(d["n"], 0)
        self.assertTrue(math.isnan(d["mean"]))
        self.assertIsNone(d["ts_min"])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            describe(np.ones(3), np.arange(2))

    def test_step_hold_weights_the_time_not_the_samples(self):
        # a setpoint: 20 for 50 minutes, 24 for 10 minutes (three quick samples)
        ts = np.array([0, 50, 52, 54], dtype=np.int64) * MIN
        values = np.array([20.0, 24.0, 24.0, 24.0])
        plain = describe(values, ts)
        self.assertAlmostEqual(plain["mean"], 23.0)
        d = describe(values, ts, hold="step", start=0, end=60 * MIN)
        self.assertEqual(d["n"], 4)
        self.assertAlmostEqual(d["mean"], (20 * 50 + 24 * 10) / 60)
        self.assertAlmostEqual(d["median"], 20.0)
        self.assertAlmostEqual(d["p95"], 24.0)
        self.assertEqual(d["min"], 20.0)

    def test_prior_value_holds_into_the_range(self):
        ts = np.array([30], dtype=np.int64) * MIN
        d = describe(np.array([10.0]), ts, hold="step", start=0, end=60 * MIN, prior=0.0)
        self.assertEqual(d["n"], 1)
        self.assertEqual(d["min"], 10.0)  # the prior sample is not one of the range
        self.assertAlmostEqual(d["mean"], 5.0)
        inside = describe(np.array([]), np.array([], dtype=np.int64), hold="step", start=0, end=MIN, prior=7.0)
        self.assertEqual(inside["n"], 0)
        self.assertAlmostEqual(inside["mean"], 7.0)  # a range inside one hold
        self.assertTrue(math.isnan(describe(np.array([]), np.array([]), hold="step", start=0, prior=7.0)["mean"]))

    def test_max_hold_cuts_an_outage(self):
        ts = np.array([0, 10, 1000], dtype=np.int64) * MIN  # the 10-minute value stops holding
        d = describe(np.array([0.0, 100.0, 0.0]), ts, hold="step", end=1010 * MIN, max_hold=10 * MIN)
        self.assertAlmostEqual(d["mean"], 100 / 3)
        caps = np.array([np.inf, 20 * MIN, np.inf])
        d = describe(np.array([0.0, 100.0, 0.0]), ts, hold="step", end=1010 * MIN, max_hold=caps)
        self.assertAlmostEqual(d["mean"], 2000 / 40)


class TestHoldWeights(unittest.TestCase):
    def test_modes(self):
        ts = np.array([0, 10, 30], dtype=np.int64) * MIN
        np.testing.assert_allclose(hold_weights(ts, "step", end=40 * MIN), np.array([10, 20, 10]) * MIN)
        np.testing.assert_allclose(hold_weights(ts, "step"), np.array([10, 20, 15]) * MIN)  # median
        np.testing.assert_allclose(hold_weights(ts, "step-back", start=-5 * MIN), np.array([5, 10, 20]) * MIN)
        np.testing.assert_allclose(hold_weights(ts, "linear"), np.array([5, 15, 10]) * MIN)
        self.assertEqual(hold_weights(np.array([], dtype=np.int64), "step").size, 0)
        with self.assertRaises(ValueError):
            hold_weights(ts, "spline")

    def test_weighted_stats(self):
        s = weighted_stats(np.array([1.0, 2.0, np.nan, 4.0]), np.array([1.0, 1.0, 5.0, 0.0]))
        self.assertAlmostEqual(s["mean"], 1.5)
        self.assertAlmostEqual(s["std"], 0.5)
        self.assertAlmostEqual(s["median"], 1.0)  # at or below 1 for half the weight
        rng = np.random.default_rng(1)
        values = rng.normal(size=5000)
        s = weighted_stats(values, np.ones(values.size))
        self.assertAlmostEqual(s["mean"], values.mean())
        self.assertAlmostEqual(s["std"], values.std())
        self.assertAlmostEqual(s["median"], np.median(values), delta=0.01)
        self.assertAlmostEqual(s["p05"], np.percentile(values, 5), delta=0.01)
        self.assertTrue(math.isnan(weighted_stats(np.array([1.0]), np.array([0.0]))["mean"]))

    def test_held_at(self):
        ts = np.array([0, 10, 30], dtype=np.int64) * MIN
        values = np.array([1.0, np.nan, 3.0])
        at = np.array([-1, 0, 5, 15, 29, 30, 100], dtype=np.int64) * MIN
        np.testing.assert_array_equal(held_at(ts, values, at), [np.nan, 1, 1, 1, 1, 3, 3])
        np.testing.assert_array_equal(
            held_at(ts, values, at, max_hold=20 * MIN), [np.nan, 1, 1, 1, np.nan, 3, np.nan]
        )
        caps = np.array([12 * MIN, np.inf, np.inf])
        np.testing.assert_array_equal(held_at(ts, values, at, max_hold=caps), [np.nan, 1, 1, np.nan, np.nan, 3, 3])

    def test_step_bucket_mean(self):
        ts = np.array([5, 10, 30], dtype=np.int64) * MIN
        values = np.array([2.0, 4.0, 8.0])
        edges = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64) * MIN
        means = step_bucket_mean(ts, values, edges)
        # bucket 0: 2 holds 5 of the 5 covered minutes; 1: 4; 2: 4; 3: 8 (until the last edge); 4: 8
        np.testing.assert_allclose(means, [2, 4, 4, 8, 8])
        capped = step_bucket_mean(ts, values, edges, max_hold=10 * MIN)
        np.testing.assert_allclose(capped, [2, 4, np.nan, 8, np.nan])  # 4 holds 10..20, 8 30..40
        self.assertEqual(step_bucket_mean(ts, values, edges[:1]).size, 0)
        self.assertTrue(np.isnan(step_bucket_mean(ts, np.full(3, np.nan), edges)).all())

    def test_bucket_reduce(self):
        index = np.array([0, 0, 2, 2, 2, 5])
        values = np.array([1.0, 3.0, np.nan, 5.0, 4.0, 9.0])
        nan = np.nan
        np.testing.assert_array_equal(bucket_reduce(index, values, 4, "mean"), [2, nan, 4.5, nan])
        np.testing.assert_array_equal(bucket_reduce(index, values, 4, "min"), [1, nan, 4, nan])
        np.testing.assert_array_equal(bucket_reduce(index, values, 4, "max"), [3, nan, 5, nan])
        np.testing.assert_array_equal(bucket_reduce(index, values, 4, "first"), [1, nan, 5, nan])
        np.testing.assert_array_equal(bucket_reduce(index, values, 4, "last"), [3, nan, 4, nan])
        with self.assertRaises(ValueError):
            bucket_reduce(index, values, 4, "median")

    def test_step_bucket_reduce(self):
        # a setpoint changed on the 5th, 12th and 20th of a 31-day month, monthly buckets
        day = 24 * 60 * MIN
        ts = np.array([5, 12, 20]) * day
        values = np.array([20.0, 22.0, 21.0])
        edges = np.array([0, 31 * day, 62 * day])
        nan = np.nan
        cap = day  # a value holds a day at most (an outage otherwise)
        np.testing.assert_array_equal(step_bucket_reduce(ts, values, edges, "first", cap), [20, nan])
        np.testing.assert_array_equal(step_bucket_reduce(ts, values, edges, "last", cap), [21, nan])
        np.testing.assert_array_equal(step_bucket_reduce(ts, values, edges, "min", cap), [20, nan])
        np.testing.assert_array_equal(step_bucket_reduce(ts, values, edges, "max", cap), [22, nan])
        np.testing.assert_allclose(step_bucket_reduce(ts, values, edges, "mean", cap), [21, nan])
        # without a cap the value holds into the next month, which has no sample of its own
        np.testing.assert_array_equal(step_bucket_reduce(ts, values, edges, "first"), [20, 21])
        np.testing.assert_array_equal(step_bucket_reduce(ts, values, edges, "last"), [21, 21])
        np.testing.assert_array_equal(step_bucket_reduce(ts, values, edges, "max"), [22, 21])
        self.assertEqual(step_bucket_reduce(ts, values, edges[:1], "max").size, 0)


class TestDetectGaps(unittest.TestCase):
    def test_finds_gap(self):
        ts = np.concatenate([np.arange(10), np.arange(30, 40)]).astype(np.int64) * MIN
        starts, ends = detect_gaps(ts)
        np.testing.assert_array_equal(starts, [9 * MIN])
        np.testing.assert_array_equal(ends, [30 * MIN])

    def test_min_gap(self):
        ts = np.array([0, 1, 2, 5, 6]) * MIN
        self.assertEqual(detect_gaps(ts)[0].size, 1)
        self.assertEqual(detect_gaps(ts, min_gap=10 * MIN)[0].size, 0)

    def test_datetime64_keeps_dtype(self):
        ts = np.array(["2024-01-01T00:00", "2024-01-01T00:01", "2024-01-01T05:00"],
                      dtype="datetime64[m]")
        starts, _ = detect_gaps(ts, expected_interval=1)
        self.assertEqual(starts.dtype, ts.dtype)


class TestGapIndices(unittest.TestCase):
    def test_indices(self):
        ts = np.array([0, 1, 2, 10, 11, 30]) * MIN
        np.testing.assert_array_equal(gap_indices(ts), [2, 4])
        self.assertEqual(gap_indices(ts).dtype, np.int64)
        self.assertEqual(gap_indices(ts[:1]).size, 0)


class TestDetectStuck(unittest.TestCase):
    def test_run(self):
        values = np.array([1.0, 2.0] + [5.0] * 30 + [3.0])
        ts = np.arange(values.size, dtype=np.int64) * MIN
        start, end, value, n = detect_stuck(values, ts, min_repeats=20)
        np.testing.assert_array_equal(start, [2 * MIN])
        np.testing.assert_array_equal(end, [31 * MIN])
        np.testing.assert_array_equal(value, [5.0])
        np.testing.assert_array_equal(n, [30])

    def test_min_duration(self):
        values = np.array([5.0] * 30)
        ts = np.arange(30, dtype=np.int64) * MIN
        self.assertEqual(detect_stuck(values, ts, 20, min_duration=60 * MIN)[0].size, 0)


class TestOutliers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.values = rng.normal(0, 1, 2_000)
        self.values[[100, 1500]] = [15.0, -12.0]
        self.values[50] = np.nan

    def test_iqr(self):
        mask = outliers_iqr(self.values, multiplier=3.0)
        self.assertTrue(mask[100] and mask[1500])
        self.assertFalse(mask[50])

    def test_zscore(self):
        mask = outliers_zscore(self.values, threshold=5.0)
        np.testing.assert_array_equal(np.flatnonzero(mask), [100, 1500])

    def test_hampel_robust_to_level_shift(self):
        values = np.concatenate([np.zeros(500), np.full(500, 100.0)])
        values += np.random.default_rng(2).normal(0, 0.1, 1000)
        values[700] = 110.0
        mask = outliers_hampel(values, window=10)
        self.assertTrue(mask[700])
        self.assertLess(mask.sum(), 50)  # few noise hits, no level flagged
        # global methods would flag nothing or a whole level
        self.assertFalse(outliers_zscore(values)[700])

    def test_hampel_large_input_is_fast(self):
        import time

        values = np.random.default_rng(3).normal(size=525_600)
        t0 = time.perf_counter()
        outliers_hampel(values, window=15)
        self.assertLess(time.perf_counter() - t0, 1.0)

    def test_all_nan(self):
        values = np.full(10, np.nan)
        self.assertFalse(outliers_iqr(values).any())
        self.assertFalse(outliers_zscore(values).any())
        self.assertFalse(outliers_hampel(values, window=2).any())


if __name__ == "__main__":
    unittest.main()


HOUR = 60 * MIN


def _regular(n, step, start=0):
    return start + np.arange(n, dtype=np.int64) * step


class TestLocalInterval(unittest.TestCase):
    def test_rate_change_is_not_a_gap(self):
        a = _regular(500, 10 * MIN)
        b = a[-1] + _regular(300, 60 * MIN, 60 * MIN)
        ts = np.r_[a, b]
        self.assertEqual(find_gaps(ts).index.size, 0)
        self.assertEqual(find_gaps(ts[::-1][::-1]).index.size, 0)
        back = np.r_[_regular(300, 60 * MIN), 300 * 60 * MIN + _regular(500, 10 * MIN)]
        self.assertEqual(find_gaps(back).index.size, 0)

    def test_outage_is_found(self):
        ts = _regular(1000, 10 * MIN)
        ts[600:] += 2 * HOUR
        gaps = find_gaps(ts)
        self.assertEqual(gaps.index.tolist(), [599])
        self.assertEqual(gaps.expected[0], 10 * MIN)
        self.assertEqual(gaps.duration[0], 2 * HOUR + 10 * MIN)

    def test_outage_at_rate_change_is_found(self):
        a = _regular(500, 10 * MIN)
        b = a[-1] + 6 * HOUR + _regular(300, 60 * MIN)
        self.assertEqual(find_gaps(np.r_[a, b]).index.tolist(), [499])

    def test_jitter_and_duplicates(self):
        rng = np.random.default_rng(0)
        ts = np.cumsum(rng.integers(9 * MIN, 11 * MIN, 2000)).astype(np.int64)
        self.assertEqual(find_gaps(ts, factor=2.5).index.size, 0)
        dup = np.r_[ts[:100], ts[99], ts[100:]]
        self.assertEqual(find_gaps(dup).index.size, 0)
        self.assertTrue(np.isnan(local_interval(dup)[99]))
        with self.assertRaises(ValueError):
            local_interval(ts[::-1])

    def test_short_and_datetime64(self):
        self.assertEqual(find_gaps(np.empty(0, dtype=np.int64)).index.size, 0)
        self.assertEqual(find_gaps(np.array([5])).index.size, 0)
        self.assertEqual(find_gaps(np.array([0, 10])).index.size, 0)
        ts = np.arange("2024-01-01", "2024-01-03", dtype="datetime64[h]")
        ts = np.delete(ts, slice(10, 20))
        gaps = find_gaps(ts)
        self.assertEqual(gaps.index.tolist(), [9])

    def test_trailing_gap(self):
        ts = _regular(100, 10 * MIN)
        self.assertEqual(find_gaps(ts, end=ts[-1] + 20 * MIN).index.size, 0)
        gaps = find_gaps(ts, end=ts[-1] + 5 * HOUR)
        self.assertEqual(gaps.index.tolist(), [99])
        self.assertEqual(gaps.end[0], ts[-1] + 5 * HOUR)

    def test_fixed_mode_unchanged(self):
        ts = np.r_[_regular(500, 10 * MIN), 4990 * MIN + _regular(300, 60 * MIN, 60 * MIN)]
        fixed = gap_indices(ts, factor=3.0)
        self.assertGreater(fixed.size, 200)  # the global median flags the slow part
        self.assertEqual(gap_indices(ts, factor=3.0, adaptive=True).size, 0)
        self.assertEqual(detect_gaps(ts, factor=3.0, adaptive=True)[0].size, 0)


class TestIntervalSegments(unittest.TestCase):
    def test_two_rates(self):
        a = _regular(500, 10 * MIN)
        b = a[-1] + _regular(300, 60 * MIN, 60 * MIN)
        seg = interval_segments(np.r_[a, b])
        self.assertEqual((seg.interval / MIN).tolist(), [10.0, 60.0])
        self.assertEqual(seg.n.sum(), 800)
        self.assertLessEqual(abs(seg.start[1] - a[-1]), 16 * 60 * MIN)

    def test_flicker_is_merged(self):
        ts = _regular(2000, 10 * MIN)
        ts[1000:1010] = ts[999] + np.arange(1, 11) * 30 * MIN
        ts[1010:] = ts[1009] + np.arange(1, 991) * 10 * MIN
        self.assertEqual(len(interval_segments(ts).interval), 1)

    def test_short(self):
        self.assertEqual(interval_segments(np.array([1])).n.size, 0)
        self.assertEqual(interval_segments(_regular(10, MIN)).n.tolist(), [10])


class TestBucketCoverage(unittest.TestCase):
    def test_gap_across_bucket_edge(self):
        day = 24 * HOUR
        ts = _regular(3 * 144, 10 * MIN)
        ts = ts[(ts < day - 3 * HOUR) | (ts >= day + 3 * HOUR)]  # 6 h gap around midnight
        edges = np.arange(0, 4 * day, day)
        c = bucket_coverage(edges, find_gaps(ts), ts[0], ts[-1] + 10 * MIN, ts)
        missing_h = c.missing / HOUR
        self.assertAlmostEqual(missing_h[0] + missing_h[1], 6.0, places=6)  # 6 h 10 min gap, the sample before covers 10 min
        self.assertEqual(c.samples.sum(), ts.size)
        self.assertEqual(c.n_gaps.tolist(), [1, 0, 0])
        self.assertTrue((c.missing <= c.observed).all())
        self.assertAlmostEqual(c.coverage[2], 1.0)

    def test_outside_window_and_trailing(self):
        ts = _regular(100, 10 * MIN, 10 * HOUR)
        edges = np.array([0, 10 * HOUR, 30 * HOUR, 40 * HOUR], dtype=np.float64)
        now = 30 * HOUR
        gaps = find_gaps(ts, end=now)
        c = bucket_coverage(edges, gaps, ts[0], now, ts)
        self.assertTrue(np.isnan(c.coverage[0]) and np.isnan(c.coverage[2]))
        stale = (now - ts[-1] - 10 * MIN) / (now - ts[0])
        self.assertAlmostEqual(c.coverage[1], 1 - stale)

    def test_interval_overlap(self):
        edges = np.array([0, 10, 20, 30])
        self.assertEqual(interval_overlap(np.array([5, 15]), np.array([12, 25]), edges).tolist(),
                         [5.0, 7.0, 5.0])
        self.assertEqual(interval_overlap(np.array([]), np.array([]), edges).tolist(), [0, 0, 0])
        with self.assertRaises(ValueError):
            interval_overlap(np.array([1]), np.array([2]), np.array([3, 1]))

    def test_speed(self):
        import time

        rng = np.random.default_rng(1)
        ts = np.cumsum(rng.integers(54_000, 66_000, 1_500_000)).astype(np.int64)
        t0 = time.perf_counter()
        find_gaps(ts)
        self.assertLess(time.perf_counter() - t0, 0.5)



class TestExclusions(unittest.TestCase):
    def test_excluded_time_is_not_missing(self):
        day = 24 * HOUR
        ts = _regular(10 * 144, 10 * MIN)
        ts = ts[(ts < 3 * day) | (ts >= 4 * day)]  # one day removed (excluded data)
        gaps = find_gaps(ts)
        self.assertEqual(gaps.index.size, 1)
        excluded = [[3 * day - 5 * MIN, 4 * day]]
        self.assertEqual(gaps.without(excluded).index.size, 0)
        self.assertEqual(gaps.without([[0, 1]]).index.size, 1)
        edges = np.array([0, 5 * day, 10 * day], dtype=np.float64)
        c = bucket_coverage(edges, gaps, ts[0], ts[-1] + 10 * MIN, ts, exclude=excluded)
        self.assertAlmostEqual(c.coverage[0], 1.0)
        self.assertAlmostEqual(c.observed[0], 4 * day - 5 * MIN)  # 5 days minus the exclusion
        plain = bucket_coverage(edges, gaps, ts[0], ts[-1] + 10 * MIN, ts)
        self.assertLess(plain.coverage[0], 0.81)

    def test_local_interval_with_and_without_duplicates(self):
        ts = _regular(1000, 10 * MIN)
        fast = local_interval(ts)
        dup = np.insert(ts, 500, ts[500])
        slow = local_interval(dup)
        self.assertTrue(np.isnan(slow[500]))
        self.assertEqual(np.nanmax(fast), 10 * MIN)
        self.assertEqual(fast.size, ts.size - 1)


class TestFindGapsFastPath(unittest.TestCase):
    """The candidate prefilter must give exactly what the full local estimate gives."""

    @staticmethod
    def reference(ts, factor=3.0, min_gap=None, block=16):
        expected = local_interval(ts, block)
        with np.errstate(invalid="ignore"):
            return np.flatnonzero(np.diff(ts) > np.maximum(factor * expected, min_gap or 0))

    def test_same_gaps_as_the_full_estimate(self):
        rng = np.random.default_rng(3)
        for trial in range(60):
            n = int(rng.integers(2, 3000))
            steps = rng.choice([10, 60], size=n - 1) * MIN if trial % 3 else np.full(n - 1, MIN)
            steps = steps + rng.integers(-5, 5, n - 1) * 1000  # jitter
            steps[rng.random(n - 1) < 0.02] *= int(rng.integers(2, 50))  # outages
            steps[rng.random(n - 1) < 0.01] = 0  # duplicates
            ts = np.r_[0, np.cumsum(np.abs(steps))].astype(np.int64)
            for block in (4, 16):
                for min_gap in (None, 30 * MIN):
                    got = find_gaps(ts, 3.0, min_gap, block)
                    ref = self.reference(ts, 3.0, min_gap, block)
                    np.testing.assert_array_equal(got.index, ref)
                    exp = local_interval(ts, block)[ref]
                    np.testing.assert_array_equal(got.expected, exp)

    def test_trailing_gap_uses_the_last_local_interval(self):
        ts = np.r_[np.arange(100) * 10 * MIN, 990 * MIN + np.arange(1, 60) * 60 * MIN]
        end = int(ts[-1]) + 10 * 60 * MIN
        gaps = find_gaps(ts, end=end)
        finite = local_interval(ts)
        self.assertEqual(gaps.index.tolist(), [ts.size - 1])
        self.assertEqual(gaps.expected[-1], finite[-1])
