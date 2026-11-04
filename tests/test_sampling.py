
import unittest

import numpy as np
from pyedautils.arraystats import infer_interval
from pyedautils.sampling import (
    _off_grid,
    interval_histogram,
    is_change_of_value,
    on_grid_share,
    sampling_profile,
)

SEC = 1_000
MIN = 60_000
HOUR = 3_600_000
DAY = 86_400_000


def _cov(signal_ts, signal, deadband, min_interval=0):
    """Change-of-value archive of a dense signal: store a sample when the value moved
    by at least ``deadband`` since the last stored one (and ``min_interval`` passed)."""
    keep = [0]
    last_v, last_t = signal[0], signal_ts[0]
    for i in range(1, signal.size):
        if abs(signal[i] - last_v) >= deadband and signal_ts[i] - last_t >= min_interval:
            keep.append(i)
            last_v, last_t = signal[i], signal_ts[i]
    idx = np.array(keep)
    return signal_ts[idx], signal[idx]


def _room_temperature(days=20, step=10 * SEC, seed=1):
    """Supply air temperature: daily course plus fast fluctuations while the unit
    runs (6-18 h), a constant value while it is off."""
    ts = np.arange(0, days * DAY, step, dtype=np.int64)
    rng = np.random.default_rng(seed)
    a = np.exp(-step / (2 * MIN))  # fluctuations: AR(1) with a 2-min time constant
    noise = rng.normal(0, 0.5 * np.sqrt(1 - a * a), ts.size)
    fluct = np.empty(ts.size)
    x = 0.0
    for i in range(ts.size):
        x = a * x + noise[i]
        fluct[i] = x
    running = (ts % DAY >= 8 * HOUR) & (ts % DAY < 18 * HOUR)
    v = np.where(running, 21 + 2 * np.sin(ts / DAY * 2 * np.pi) + fluct, 19.0)
    return ts, np.round(v, 2)


def _with_heartbeat(ts, v, beat):
    """Add a refresh of the held value ``beat`` after every stored sample, repeated
    while the value does not change (a COV sender with heartbeat)."""
    extra_ts, extra_v = [], []
    for i in range(ts.size):
        end = ts[i + 1] if i + 1 < ts.size else ts[i] + beat
        t = ts[i] + beat
        while t < end:
            extra_ts.append(t)
            extra_v.append(v[i])
            t += beat
    all_ts = np.concatenate([ts, np.array(extra_ts, dtype=np.int64)])
    order = np.argsort(all_ts, kind="stable")
    return all_ts[order], np.concatenate([v, np.array(extra_v)])[order]


class TestIsChangeOfValue(unittest.TestCase):
    def test_regular_with_outages_is_not_cov(self):
        ts = np.arange(10_000, dtype=np.int64) * MIN
        ts[5000:] += 3 * DAY  # one long outage
        self.assertFalse(is_change_of_value(ts))

    def test_cov_archive(self):
        ts, _ = _cov(*_room_temperature(), deadband=0.5)
        self.assertTrue(is_change_of_value(ts))

    def test_slow_signal_with_large_dead_band(self):
        # steps of 30 min to a day, none regular: the long-step test alone misses it
        rng = np.random.default_rng(4)
        steps = np.exp(rng.uniform(np.log(30 * MIN), np.log(DAY), 2000)).astype(np.int64)
        self.assertTrue(is_change_of_value(np.cumsum(steps)))

    def test_interval_with_rate_changes_stays_regular(self):
        rng = np.random.default_rng(5)
        fast = np.full(3000, 10 * MIN) + rng.integers(-5_000, 5_000, 3000)  # jitter
        slow = np.full(500, HOUR) + rng.integers(-5_000, 5_000, 500)
        ts = np.cumsum(np.concatenate([fast, slow, fast]))
        self.assertFalse(is_change_of_value(ts))
        self.assertGreater(on_grid_share(np.diff(ts).astype(float)), 0.9)
        self.assertEqual(on_grid_share(np.empty(0)), 1.0)

    def test_two_even_rates_stay_regular(self):
        ts = np.cumsum(np.tile([MIN, 15 * MIN], 2000)).astype(np.int64)
        self.assertFalse(is_change_of_value(ts))

    def test_few_samples(self):
        self.assertFalse(is_change_of_value(np.arange(10, dtype=np.int64) * HOUR))
        ts = np.zeros(100, dtype=np.int64)  # no positive step
        self.assertFalse(is_change_of_value(ts))

    def test_datetime64(self):
        ts = np.arange("2024-01-01", "2024-01-08", dtype="datetime64[m]")
        self.assertFalse(is_change_of_value(ts))

    def test_scattered_steps_with_narrow_spread_stay_regular(self):
        # steps spread evenly over 1-3 min: few at the median, but p90/p10 < 4
        rng = np.random.default_rng(6)
        steps = rng.integers(MIN, 3 * MIN, 2000)
        self.assertFalse(is_change_of_value(np.cumsum(steps)))

    def test_off_grid_needs_positive_median(self):
        self.assertFalse(_off_grid(np.array([0.0, 1.0, 2.0]), 0.0))


class TestIntervalHistogram(unittest.TestCase):
    def test_counts_and_clipping(self):
        ts = np.array([0, 1, 3, 13, 113, 10_113], dtype=np.int64)  # steps 1, 2, 10, 100, 10000
        counts = interval_histogram(ts, [2, 10, 1000])
        np.testing.assert_array_equal(counts, [2, 3])  # 1 clipped into the first bin

    def test_edges_needed(self):
        with self.assertRaises(ValueError):
            interval_histogram(np.arange(3), [1])


class TestSamplingProfile(unittest.TestCase):
    def test_regular_series(self):
        ts = np.arange(0, 7 * DAY, 5 * MIN, dtype=np.int64)
        v = np.round(np.random.default_rng(0).normal(20, 1, ts.size), 1)
        p = sampling_profile(ts, v)
        self.assertFalse(p.change_of_value)
        self.assertEqual(p.interval_min, 5 * MIN)
        self.assertEqual(p.interval_median, infer_interval(ts))
        self.assertIsNone(p.deadband)
        self.assertIsNone(p.heartbeat)
        self.assertAlmostEqual(p.resolution, 0.1, places=6)
        self.assertLess(p.compression, 1.2)
        self.assertEqual(p.n, ts.size)
        self.assertEqual(p.span, ts[-1])

    def test_cov_deadband_and_min_interval(self):
        ts, v = _cov(*_room_temperature(), deadband=0.5, min_interval=30 * SEC)
        p = sampling_profile(ts, v, long_hold=12 * HOUR)
        self.assertTrue(p.change_of_value)
        self.assertGreaterEqual(p.deadband, 0.5)
        self.assertLess(p.deadband, 0.56)
        self.assertGreaterEqual(p.interval_min, 30 * SEC)
        self.assertIsNone(p.heartbeat)
        # the unit is off for 14 h every night
        self.assertGreater(p.longest_hold, 13 * HOUR)
        self.assertGreater(p.hold_share, 0.5)
        self.assertLess(p.repeat_share, 0.01)

    def test_heartbeat(self):
        ts, v = _with_heartbeat(*_cov(*_room_temperature(), deadband=0.5), beat=15 * MIN)
        p = sampling_profile(ts, v)
        self.assertEqual(p.heartbeat, 15 * MIN)
        self.assertTrue(p.change_of_value)
        self.assertGreaterEqual(p.deadband, 0.5)
        self.assertLessEqual(p.interval_max, 15 * MIN)

    def test_irregular_repeats_are_no_heartbeat(self):
        ts, v = _cov(*_room_temperature(), deadband=0.5)
        rng = np.random.default_rng(5)
        extra = np.sort(rng.integers(ts[0], ts[-1], 400))  # repeats at random times
        held = v[np.searchsorted(ts, extra, side="right") - 1]
        all_ts = np.concatenate([ts, extra])
        order = np.argsort(all_ts, kind="stable")
        p = sampling_profile(all_ts[order], np.concatenate([v, held])[order])
        self.assertIsNone(p.heartbeat)

    def test_no_long_hold_share_without_threshold(self):
        ts = np.arange(0, 3 * DAY, HOUR, dtype=np.int64)
        p = sampling_profile(ts, np.zeros(ts.size))
        self.assertEqual(p.hold_share, 0.0)
        self.assertEqual(p.longest_hold, ts[-1])

    def test_periodic_mostly_constant_has_no_heartbeat(self):
        ts = np.arange(0, 7 * DAY, 20 * MIN, dtype=np.int64)
        v = np.where((ts // (6 * HOUR)) % 2 == 0, 1.0, 2.0)
        p = sampling_profile(ts, v)
        self.assertFalse(p.change_of_value)
        self.assertIsNone(p.heartbeat)
        self.assertGreater(p.repeat_share, 0.9)
        self.assertGreater(p.compression, 5)  # holds: hardly any curvature

    def test_compression_of_linear_segments(self):
        # an archive storing straight segments: second differences are zero inside
        ts = np.arange(0, 1000, dtype=np.int64) * MIN
        knots = np.random.default_rng(3).integers(-5, 5, 101) * 10.0  # exact slopes
        v = np.interp(np.arange(1000), np.arange(0, 1001, 10), knots)
        p = sampling_profile(ts, v)
        self.assertGreater(p.compression, 5)

    def test_resolution_ignores_denormal_changes(self):
        ts = np.arange(200, dtype=np.int64) * MIN
        v = np.round(np.random.default_rng(2).normal(0, 20, 200) / 8) * 8
        v[49], v[50], v[51] = 0.0, 1.4e-45, 0.0  # a float32 denormal read as a change
        p = sampling_profile(ts, v)
        self.assertAlmostEqual(p.resolution, 8.0)

    def test_no_resolution_for_continuous_values(self):
        ts = np.arange(500, dtype=np.int64) * MIN
        p = sampling_profile(ts, np.random.default_rng(4).normal(0, 1, 500))
        self.assertIsNone(p.resolution)

    def test_nan_dropped_and_empty(self):
        ts = np.arange(5, dtype=np.int64) * MIN
        p = sampling_profile(ts, np.array([1.0, np.nan, 2.0, np.nan, 3.0]))
        self.assertEqual(p.n, 3)
        self.assertEqual(p.interval_median, 2 * MIN)
        empty = sampling_profile(np.empty(0, np.int64), np.empty(0))
        self.assertEqual(empty.n, 0)
        self.assertIsNone(empty.interval_min)
        self.assertIsNone(empty.compression)
        self.assertEqual(empty.longest_hold, 0.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            sampling_profile(np.arange(3), np.arange(2.0))
        with self.assertRaises(ValueError):
            sampling_profile(np.array([2, 1, 3]), np.arange(3.0))

    def test_datetime64(self):
        ts = np.arange("2024-01-01", "2024-01-02", dtype="datetime64[m]")
        p = sampling_profile(ts, np.arange(ts.size, dtype=np.float64) % 7)
        self.assertEqual(p.interval_median, 1)


if __name__ == "__main__":
    unittest.main()
