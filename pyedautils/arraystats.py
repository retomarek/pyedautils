# -*- coding: utf-8 -*-
"""Fast statistics for (large) time series on plain NumPy arrays.

All functions operate on plain NumPy arrays and avoid Python loops over data
points, so they are suitable for interactive tools with millions of samples
per series. Timestamps are numeric (e.g. int64 epoch milliseconds) or
``datetime64``; intervals and durations are returned in the same unit as the
timestamps (``datetime64`` input is treated as its underlying int64 ticks).

Missing values are represented as ``NaN`` and ignored by all statistics.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, TypedDict, Union

import numpy as np

__all__ = [
    "Description",
    "describe",
    "HOLD_MODES",
    "hold_weights",
    "weighted_stats",
    "held_at",
    "step_bucket_mean",
    "bucket_reduce",
    "step_bucket_reduce",
    "bucket_reduce",
    "step_bucket_reduce",
    "infer_interval",
    "gap_indices",
    "detect_gaps",
    "local_interval",
    "Gaps",
    "find_gaps",
    "IntervalSegments",
    "interval_segments",
    "interval_overlap",
    "BucketCoverage",
    "bucket_coverage",
    "detect_stuck",
    "outliers_iqr",
    "outliers_zscore",
    "outliers_hampel",
]


#: ``[[start, end], ...]`` as a sequence of pairs or an array of shape (k, 2)
RangesLike = Union[np.ndarray, Sequence[Sequence[float]]]


class Description(TypedDict):
    """Result of :func:`describe`. Statistics are ``nan`` if ``n == 0``."""

    n: int
    n_missing: int
    coverage: float
    min: float
    max: float
    mean: float
    median: float
    std: float
    p05: float
    p95: float
    ts_min: Optional[int]
    ts_max: Optional[int]
    ts_first: Optional[int]
    ts_last: Optional[int]


def _as_ticks(ts: np.ndarray) -> np.ndarray:
    """Return timestamps as int64/float ticks (``datetime64`` -> int64 view)."""
    ts = np.asarray(ts)
    if np.issubdtype(ts.dtype, np.datetime64):
        return ts.view("int64")
    return ts


def infer_interval(ts: np.ndarray) -> Optional[float]:
    """Estimate the sampling interval as the median positive time difference.

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).

    Returns:
        The median positive interval in timestamp units, or ``None`` if there
        are fewer than two distinct timestamps.
    """
    ticks = _as_ticks(ts)
    if ticks.size < 2:
        return None
    diffs = np.diff(ticks)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


#: How a value is read between two samples: ``step`` holds until the next sample
#: (setpoints, states, change-of-value data), ``step-back`` holds until its own
#: timestamp (counter readings), ``linear`` is a continuous signal.
HOLD_MODES = ("step", "step-back", "linear")

#: A cap on the time a sample stands for: one number for all, or one per sample
#: (``inf`` where a hold is unlimited), e.g. the expected interval at a gap start.
MaxHold = Union[None, float, np.ndarray]


def _capped(dt: np.ndarray, max_hold: MaxHold, at: slice) -> np.ndarray:
    """``dt`` capped by ``max_hold`` (a number, or the per-sample caps ``at``)."""
    if max_hold is None:
        return dt
    cap = np.asarray(max_hold, dtype=np.float64)
    return np.minimum(dt, cap[at] if cap.ndim else cap)


def hold_weights(
    ts: np.ndarray,
    hold: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    expected_interval: Optional[float] = None,
    max_hold: MaxHold = None,
) -> np.ndarray:
    """The time each sample stands for, as a weight per sample.

    Args:
        ts: Sorted timestamps.
        hold: ``step``: a sample holds until the next one, the last one until
            ``end`` (else one ``expected_interval``). ``step-back``: a sample holds
            from the previous one, the first one from ``start``. ``linear``: a sample
            stands for half the interval to each neighbour (trapezoid rule).
        start, end: Bounds of the analysed range for the first / last hold.
        expected_interval: Fallback for the open end; inferred when ``None``.
        max_hold: Cap on a hold (a number or one cap per sample, ``inf`` = none):
            a value does not hold across an outage.

    Returns:
        Float weights, same length as ``ts`` (zeros for a single sample without
        bounds).
    """
    if hold not in HOLD_MODES:
        raise ValueError(f"hold must be one of {HOLD_MODES}, not {hold!r}")
    ticks = _as_ticks(ts).astype(np.float64)
    n = ticks.size
    if n == 0:
        return np.empty(0)

    def fallback() -> float:  # one interval for an open end (inferred only when needed)
        interval = expected_interval if expected_interval is not None else infer_interval(ticks)
        return float(interval) if interval else 0.0

    dt = np.diff(ticks)
    if hold == "step":
        tail = max(float(end) - ticks[-1], 0.0) if end is not None else fallback()
        w = np.append(_capped(dt, max_hold, slice(0, n - 1)), _capped(np.array([tail]), max_hold, slice(n - 1, n)))
    elif hold == "step-back":
        head = max(ticks[0] - float(start), 0.0) if start is not None else fallback()
        w = np.insert(_capped(dt, max_hold, slice(1, n)), 0, _capped(np.array([head]), max_hold, slice(0, 1)))
    else:
        dt = _capped(dt, max_hold, slice(0, n - 1))
        w = np.zeros(n)
        w[:-1] += dt / 2
        w[1:] += dt / 2
    return w


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict:
    """Mean, population std, median and the 5th / 95th percentile of ``values``
    weighted by ``weights`` (e.g. :func:`hold_weights`): the statistics of the
    signal over time rather than of the samples. Non-finite values and
    non-positive weights are ignored; all ``nan`` without any weight.

    Percentiles are the inverse of the weighted cumulative distribution (the smallest
    value the signal is at or below for the share ``p`` of the weight): exact for a
    step signal, within one sample of ``numpy.percentile`` for equal weights.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same length")
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    nan = float("nan")
    if not keep.any():
        return {"mean": nan, "std": nan, "median": nan, "p05": nan, "p95": nan}
    v, w = values[keep], weights[keep]
    total = float(w.sum())
    mean = float((v * w).sum() / total)
    std = float(np.sqrt(((v - mean) ** 2 * w).sum() / total))
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    at = np.minimum(np.searchsorted(cum, np.array([0.05, 0.5, 0.95]) * total, side="left"), v.size - 1)
    p05, median, p95 = v[at]
    return {"mean": mean, "std": std, "median": float(median), "p05": float(p05), "p95": float(p95)}


def held_at(ts: np.ndarray, values: np.ndarray, at: np.ndarray, max_hold: MaxHold = None) -> np.ndarray:
    """The value a step series holds at each time in ``at``: the last sample at or
    before it, ``nan`` before the first sample or once the hold is capped by
    ``max_hold`` (a number or one cap per sample). Non-finite samples do not hold."""
    ticks = _as_ticks(ts).astype(np.float64)
    values = np.asarray(values, dtype=np.float64)
    at = _as_ticks(np.asarray(at)).astype(np.float64)
    finite = np.isfinite(values)
    ticks, values = ticks[finite], values[finite]
    out = np.full(at.shape, np.nan)
    if ticks.size == 0:
        return out
    pos = np.searchsorted(ticks, at, side="right") - 1
    ok = pos >= 0
    src = np.clip(pos, 0, ticks.size - 1)
    if max_hold is not None:
        cap = np.asarray(max_hold, dtype=np.float64)
        cap = cap[finite][src] if cap.ndim else cap
        ok &= at - ticks[src] < cap
    out[ok] = values[src[ok]]
    return out


def step_bucket_mean(
    ts: np.ndarray, values: np.ndarray, edges: np.ndarray, max_hold: MaxHold = None
) -> np.ndarray:
    """Time-weighted mean of a step series (each sample holds until the next one) in
    the buckets ``[edges[i], edges[i + 1])``: the integral of the held value over the
    time the bucket is covered, divided by that time; ``nan`` where nothing holds.
    ``max_hold`` caps a hold (a number or one cap per sample), the last sample holds
    until the last edge at most."""
    ticks = _as_ticks(ts).astype(np.float64)
    values = np.asarray(values, dtype=np.float64)
    edges = _as_ticks(np.asarray(edges)).astype(np.float64)
    finite = np.isfinite(values)
    ticks, values = ticks[finite], values[finite]
    out = np.full(max(edges.size - 1, 0), np.nan)
    if ticks.size == 0 or edges.size < 2:
        return out
    cap = max_hold
    if cap is not None and np.ndim(cap):
        cap = np.asarray(cap, dtype=np.float64)[finite]
    hold = hold_weights(ticks, "step", end=float(edges[-1]), max_hold=cap)
    # cumulative integral and covered time up to each sample's start
    cum_v = np.concatenate(([0.0], np.cumsum(values * hold)))
    cum_t = np.concatenate(([0.0], np.cumsum(hold)))

    def upto(t: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        k = np.searchsorted(ticks, t, side="right") - 1
        src = np.clip(k, 0, ticks.size - 1)
        into = np.clip(t - ticks[src], 0.0, hold[src])
        into[k < 0] = 0.0
        base = np.where(k < 0, 0, k)
        return cum_v[base] + values[src] * into, cum_t[base] + into

    v_at, t_at = upto(edges)
    covered = np.diff(t_at)
    np.divide(np.diff(v_at), covered, out=out, where=covered > 0)
    return out


def _with_prior(
    values: np.ndarray,
    ticks: np.ndarray,
    weights: np.ndarray,
    hold: str,
    prior: Optional[float],
    start: Optional[float],
    end: Optional[float],
) -> "tuple[np.ndarray, np.ndarray]":
    """Values and weights with the prior value of a step series in front: it holds from
    ``start`` to the first sample (or to ``end`` when the range has no sample at all)."""
    if hold != "step" or prior is None or not np.isfinite(prior) or start is None:
        return values, weights
    until = float(ticks[0]) if ticks.size else (float(end) if end is not None else None)
    if until is None or until <= float(start):
        return values, weights
    return np.insert(values, 0, prior), np.insert(weights, 0, until - float(start))


def bucket_reduce(index: np.ndarray, values: np.ndarray, n: int, fn: str) -> np.ndarray:
    """One value per bucket ``0 .. n-1`` from samples with their bucket ``index``
    (sorted): ``mean``, ``min``, ``max``, ``first`` or ``last`` of the finite samples
    in the bucket, ``nan`` for a bucket without one. Samples outside ``0 .. n-1``
    are ignored."""
    index = np.asarray(index)
    values = np.asarray(values, dtype=np.float64)
    out = np.full(n, np.nan)
    keep = np.isfinite(values) & (index >= 0) & (index < n)
    index, values = index[keep], values[keep]
    if index.size == 0:
        return out
    if fn == "mean":
        sums = np.bincount(index, weights=values, minlength=n)
        counts = np.bincount(index, minlength=n)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out
    if fn == "last":
        out[index] = values  # later samples overwrite earlier ones
        return out
    starts = np.flatnonzero(np.concatenate(([True], np.diff(index) != 0)))
    if fn == "first":
        out[index[starts]] = values[starts]
    elif fn in ("min", "max"):
        reduce = np.minimum if fn == "min" else np.maximum
        out[index[starts]] = reduce.reduceat(values, starts)
    else:
        raise ValueError(f"fn must be mean, min, max, first or last, not {fn!r}")
    return out


def step_bucket_reduce(
    ts: np.ndarray, values: np.ndarray, edges: np.ndarray, fn: str, max_hold: MaxHold = None
) -> np.ndarray:
    """One value per bucket ``[edges[i], edges[i + 1])`` of a step series (each sample
    holds until the next one, for ``max_hold`` at most): ``mean`` is time-weighted
    (:func:`step_bucket_mean`); ``min`` / ``max`` take the samples in the bucket and
    the value entering it; ``first`` is the entering value, else the first sample;
    ``last`` the last sample, else the value held at the bucket's end. A bucket is
    ``nan`` only when nothing holds in it."""
    if fn == "mean":
        return step_bucket_mean(ts, values, edges, max_hold)
    ticks = _as_ticks(ts).astype(np.float64)
    edges = _as_ticks(np.asarray(edges)).astype(np.float64)
    n = max(edges.size - 1, 0)
    if n == 0:
        return np.full(0, np.nan)
    sampled = bucket_reduce(np.searchsorted(edges, ticks, side="right") - 1, values, n, fn)
    if fn == "last":
        at_end = held_at(ts, values, np.nextafter(edges[1:], -np.inf), max_hold)
        return np.where(np.isfinite(sampled), sampled, at_end)
    entering = held_at(ts, values, edges[:-1], max_hold)
    if fn == "first":
        return np.where(np.isfinite(entering), entering, sampled)
    return np.fmin(sampled, entering) if fn == "min" else np.fmax(sampled, entering)


def describe(
    values: np.ndarray,
    ts: Optional[np.ndarray] = None,
    expected_interval: Optional[float] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    hold: Optional[str] = None,
    max_hold: MaxHold = None,
    prior: Optional[float] = None,
) -> Description:
    """Descriptive statistics of a series, ignoring ``NaN`` values.

    Args:
        values: Sample values.
        ts: Timestamps belonging to ``values`` (same length, sorted). Needed
            for ``coverage`` and the ``ts_*`` fields.
        expected_interval: Expected sampling interval in timestamp units.
            Inferred via :func:`infer_interval` when ``None``.
        start: Start of the analysed range (inclusive) for ``coverage``.
            Defaults to the first timestamp.
        end: End of the analysed range (exclusive) for ``coverage``.
            Defaults to the last timestamp plus one interval.
        hold: One of :data:`HOLD_MODES`: mean, std, median and the percentiles are
            then weighted by the time each sample stands for (:func:`hold_weights`),
            i.e. they describe the signal over time, not the samples. ``None``:
            plain sample statistics.
        max_hold: Cap on a hold, see :func:`hold_weights`.
        prior: For ``hold="step"``: the value holding at ``start`` from before the
            range (the sample before the first one). It weighs in from ``start`` to
            the first timestamp but is not counted in ``n``, ``min``, ``max``.

    Returns:
        :class:`Description` with n, missing count, coverage (0..1, ``nan``
        without timestamps), min, max, mean, median, std (population),
        5th/95th percentile and the timestamps of min, max, first and last
        valid sample.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    valid = values[finite]
    n = int(valid.size)
    nan = float("nan")
    result: Description = {
        "n": n,
        "n_missing": int(values.size - n),
        "coverage": nan,
        "min": nan,
        "max": nan,
        "mean": nan,
        "median": nan,
        "std": nan,
        "p05": nan,
        "p95": nan,
        "ts_min": None,
        "ts_max": None,
        "ts_first": None,
        "ts_last": None,
    }
    if n > 0:
        p05, median, p95 = np.percentile(valid, [5.0, 50.0, 95.0])
        result.update(
            min=float(valid.min()),
            max=float(valid.max()),
            mean=float(valid.mean()),
            median=float(median),
            std=float(valid.std()),
            p05=float(p05),
            p95=float(p95),
        )
    if ts is None:
        return result

    ticks = _as_ticks(ts)
    if ticks.size != values.size:
        raise ValueError("ts and values must have the same length")
    valid_ticks = ticks[finite]
    if n > 0:
        result.update(
            ts_min=int(valid_ticks[np.argmin(valid)]),
            ts_max=int(valid_ticks[np.argmax(valid)]),
            ts_first=int(valid_ticks[0]),
            ts_last=int(valid_ticks[-1]),
        )
    interval = expected_interval if expected_interval is not None else infer_interval(ticks)
    if hold is not None and (n > 0 or prior is not None):
        caps = np.asarray(max_hold, dtype=np.float64)[finite] if np.ndim(max_hold) else max_hold
        weights = hold_weights(valid_ticks, hold, start, end, interval, caps) if n else np.empty(0)
        weighted = _with_prior(valid, valid_ticks, weights, hold, prior, start, end)
        if weighted[1].size:
            result.update(weighted_stats(*weighted))  # type: ignore[call-arg]
    if interval is None or interval <= 0:
        return result
    range_start = float(start) if start is not None else float(ticks[0]) if ticks.size else None
    range_end = float(end) if end is not None else float(ticks[-1]) + interval if ticks.size else None
    if range_start is None or range_end is None or range_end <= range_start:
        return result
    expected = (range_end - range_start) / interval
    result["coverage"] = min(1.0, n / expected)
    return result


def gap_indices(
    ts: np.ndarray,
    expected_interval: Optional[float] = None,
    factor: float = 2.0,
    min_gap: Optional[float] = None,
    adaptive: bool = False,
    block: int = 16,
) -> np.ndarray:
    """Indices ``i`` where a gap lies between ``ts[i]`` and ``ts[i + 1]``.

    A gap is a time difference larger than ``max(factor * expected_interval,
    min_gap)``; the interval is inferred via :func:`infer_interval` when
    ``None``. With ``adaptive=True`` (and no ``expected_interval``) the expected
    interval is the local one (:func:`local_interval`), so a change of the
    sampling rate is not a gap. Returns an int64 array (empty if no gaps).
    """
    if adaptive and expected_interval is None:
        return find_gaps(ts, factor=factor, min_gap=min_gap, block=block).index
    ticks = _as_ticks(ts)
    empty = np.empty(0, dtype=np.int64)
    if ticks.size < 2:
        return empty
    interval = expected_interval if expected_interval is not None else infer_interval(ticks)
    if interval is None:
        return empty
    threshold = max(interval * factor, min_gap or 0)
    return np.flatnonzero(np.diff(ticks) > threshold).astype(np.int64)


def detect_gaps(
    ts: np.ndarray,
    expected_interval: Optional[float] = None,
    factor: float = 2.0,
    min_gap: Optional[float] = None,
    adaptive: bool = False,
    block: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Find gaps between consecutive timestamps.

    A gap is a time difference larger than ``max(factor * expected_interval,
    min_gap)``.

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).
        expected_interval: Expected sampling interval in timestamp units;
            inferred via :func:`infer_interval` when ``None``.
        factor: Multiplier on the expected interval. Default 2.0.
        min_gap: Minimum gap length in timestamp units. Default 0.
        adaptive: Compare against the local interval (:func:`local_interval`)
            instead of one interval for the whole series. Default False.
        block: Block size of the local interval. Default 16.

    Returns:
        ``(gap_start, gap_end)``: arrays with the last timestamp before and
        the first timestamp after each gap (same dtype as ``ts``).
    """
    ts = np.asarray(ts)
    idx = gap_indices(ts, expected_interval, factor, min_gap, adaptive, block)
    return ts[idx], ts[idx + 1]


def _block_medians(dp: np.ndarray, block: int) -> np.ndarray:
    """Medians of consecutive blocks of ``dp``; a short remainder (less than half a
    block) joins the last block."""
    k = dp.size // block
    medians = np.median(dp[: k * block].reshape(k, block), axis=1)
    rest = dp.size - k * block
    if rest >= block // 2:
        medians = np.append(medians, np.median(dp[k * block :]))
    elif rest:
        medians[-1] = np.median(dp[(k - 1) * block :])
    return medians


def local_interval(ts: np.ndarray, block: int = 16) -> np.ndarray:
    """Expected sampling interval before each step, robust to rate changes.

    The positive time differences are split into blocks of ``block`` steps; the
    expected interval of a step is the largest median of its own block and the two
    neighbouring blocks. A switch from 10 to 60 minutes (e.g. a LoRaWAN downlink)
    therefore does not make the first 60-minute steps look like gaps, while a long
    outage between regular samples stands out. Fewer than ``2 * block`` steps fall
    back to the global median. Vectorised, O(n).

    Limitation: if more than half of the packets of a block are lost, the local
    interval reads too long and those losses are not flagged.

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).
        block: Steps per block. Default 16.

    Returns:
        float64 array of length ``len(ts) - 1``; ``nan`` for duplicate timestamps.

    Raises:
        ValueError: if ``ts`` is not sorted.
    """
    ticks = _as_ticks(ts)
    d = np.diff(ticks).astype(np.float64)
    out = np.full(d.size, np.nan)
    if d.size == 0:
        return out
    if (d < 0).any():
        raise ValueError("ts must be sorted")
    pos = d > 0
    all_pos = bool(pos.all())
    dp = d if all_pos else d[pos]
    if dp.size == 0:
        return out
    if dp.size < 2 * block:
        out[pos] = np.median(dp)
        return out
    medians = _block_medians(dp, block)
    padded = np.concatenate([medians[:1], medians, medians[-1:]])
    expected = np.maximum(np.maximum(padded[:-2], padded[1:-1]), padded[2:])
    # per step: the value of its block (the last block also takes a short remainder)
    per_step = np.repeat(expected, block)[: dp.size]
    if per_step.size < dp.size:
        per_step = np.r_[per_step, np.full(dp.size - per_step.size, expected[-1])]
    else:
        per_step[(medians.size - 1) * block :] = expected[-1]
    if all_pos:
        return per_step
    out[pos] = per_step
    return out


def _local_interval_at(dp: np.ndarray, at: np.ndarray, block: int) -> np.ndarray:
    """:func:`local_interval` only for the positive steps ``at`` (indices into the
    positive time differences ``dp``): medians are computed just for the blocks
    around them, which makes a sparse query much cheaper than the full estimate."""
    if at.size == 0:
        return np.empty(0, dtype=np.float64)
    if dp.size < 2 * block:
        return np.full(at.size, float(np.median(dp)))
    k, rest = divmod(dp.size, block)
    n_blocks = k + 1 if rest >= block // 2 else k
    own = np.minimum(at // block, n_blocks - 1)
    needed = np.unique(np.clip(np.concatenate([own - 1, own, own + 1]), 0, n_blocks - 1))
    medians = np.empty(n_blocks)
    for j in needed.tolist():  # the last block also takes a short remainder
        stop = dp.size if j == n_blocks - 1 else (j + 1) * block
        medians[j] = np.median(dp[j * block : stop])
    lo = medians[np.maximum(own - 1, 0)]
    hi = medians[np.minimum(own + 1, n_blocks - 1)]
    return np.maximum(np.maximum(lo, medians[own]), hi)


class Gaps(NamedTuple):
    """Gaps of a series: ``index`` i means between ``ts[i]`` and ``ts[i + 1]``
    (``len(ts) - 1`` for a trailing gap up to ``end``)."""

    index: np.ndarray  # int64
    start: np.ndarray  # last timestamp before the gap (ticks)
    end: np.ndarray  # first timestamp after the gap (or ``end``)
    expected: np.ndarray  # float64: expected interval at the gap

    @property
    def duration(self) -> np.ndarray:
        return self.end - self.start

    def without(self, exclude: "RangesLike") -> "Gaps":
        """Drop gaps whose missing part (``[start + expected, end)``) lies completely
        inside the excluded ranges (``[[start, end], ...]``, e.g. removed test data)."""
        ranges = _merged_ranges(exclude)
        if ranges.size == 0 or self.index.size == 0:
            return self
        miss_start = self.start.astype(np.float64) + self.expected
        miss_end = self.end.astype(np.float64)
        k = np.searchsorted(ranges[:, 0], miss_start, side="right") - 1
        inside = (k >= 0) & (miss_end <= ranges[np.maximum(k, 0), 1])
        keep = ~inside
        return Gaps(self.index[keep], self.start[keep], self.end[keep], self.expected[keep])


def find_gaps(
    ts: np.ndarray,
    factor: float = 3.0,
    min_gap: Optional[float] = None,
    block: int = 16,
    expected_interval: Optional[float] = None,
    end: Optional[float] = None,
) -> Gaps:
    """Gaps longer than ``max(factor * expected interval, min_gap)``.

    The expected interval is the local one (:func:`local_interval`) unless
    ``expected_interval`` is given. With ``end`` (e.g. "now"), the time after the
    last sample counts as a trailing gap if it is that long.

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).
        factor: Multiplier on the expected interval. Default 3.0.
        min_gap: Minimum gap length in timestamp units. Default 0.
        block: Block size of :func:`local_interval`. Default 16.
        expected_interval: Fixed expected interval (disables the local estimate).
        end: Timestamp up to which data is expected (ticks).

    Returns:
        :class:`Gaps` (start and end in ticks).
    """
    ticks = _as_ticks(ts)
    n = ticks.size
    if n < 2:
        empty_f = np.empty(0, dtype=np.float64)
        return Gaps(np.empty(0, dtype=np.int64), ticks[:0], ticks[:0], empty_f)
    d = np.diff(ticks)
    floor = min_gap or 0
    last: Optional[float] = None
    if expected_interval is not None:
        idx = np.flatnonzero(d > max(factor * float(expected_interval), floor)).astype(np.int64)
        exp = np.full(idx.size, float(expected_interval))
        last = float(expected_interval)
    else:
        if (d < 0).any():
            raise ValueError("ts must be sorted")
        pos = d > 0
        all_pos = bool(pos.all())
        dp = (d if all_pos else d[pos]).astype(np.float64)
        if dp.size == 0:
            idx, exp = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        else:
            # every local interval is at least the shortest step, so only steps longer
            # than factor * shortest step can be gaps: check just those (fast path)
            cand = np.flatnonzero(d > max(factor * float(dp.min()), floor)).astype(np.int64)
            rank = cand if all_pos else np.cumsum(pos)[cand] - 1
            exp_c = _local_interval_at(dp, rank, block)
            keep = d[cand] > np.maximum(factor * exp_c, floor)
            idx, exp = cand[keep], exp_c[keep]
            last = float(_local_interval_at(dp, np.array([dp.size - 1]), block)[0])
    starts, ends = ticks[idx], ticks[idx + 1]
    if end is not None:
        if last is not None and end - ticks[-1] > max(factor * last, min_gap or 0):
            idx = np.append(idx, n - 1)
            starts = np.append(starts, ticks[-1])
            ends = np.append(ends, np.asarray(end).astype(ticks.dtype))
            exp = np.append(exp, last)
    return Gaps(idx, starts, ends, exp)


class IntervalSegments(NamedTuple):
    """Periods with one sampling interval (e.g. 10 min, then 60 min)."""

    start: np.ndarray  # first timestamp of the segment (ticks)
    end: np.ndarray  # last timestamp of the segment
    interval: np.ndarray  # float64 median interval in the segment
    n: np.ndarray  # int64 samples in the segment


def interval_segments(
    ts: np.ndarray, block: int = 16, tolerance: float = 0.25, min_blocks: int = 3
) -> IntervalSegments:
    """Split a series into periods with a stable sampling interval.

    Block medians (see :func:`local_interval`) are put into logarithmic classes of
    relative width ``tolerance``; runs of the same class form segments, and runs
    shorter than ``min_blocks`` blocks (flicker, outages) join the previous segment.
    The interval of a segment is the median of its positive time differences.

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).
        block: Steps per block. Default 16.
        tolerance: Relative difference treated as the same interval. Default 0.25.
        min_blocks: Minimum length of a segment in blocks. Default 3.

    Returns:
        :class:`IntervalSegments` (empty for fewer than two distinct timestamps).
    """
    ticks = _as_ticks(ts)
    empty = IntervalSegments(ticks[:0], ticks[:0], np.empty(0), np.empty(0, dtype=np.int64))
    if ticks.size < 2:
        return empty
    d = np.diff(ticks).astype(np.float64)
    pos_idx = np.flatnonzero(d > 0)
    if pos_idx.size == 0:
        return empty
    dp = d[pos_idx]
    if dp.size < 2 * block:
        return IntervalSegments(ticks[:1], ticks[-1:], np.array([np.median(dp)]),
                                np.array([ticks.size], dtype=np.int64))
    medians = _block_medians(dp, block)
    classes = np.round(np.log(medians) / np.log1p(tolerance)).astype(np.int64)
    # runs of equal class; short runs join the previous long run
    starts = np.flatnonzero(np.r_[True, classes[1:] != classes[:-1]])
    lengths = np.diff(np.r_[starts, classes.size])
    long_run = lengths >= min_blocks
    if not long_run.any():
        long_run[np.argmax(lengths)] = True
    keep = starts[long_run]
    # neighbouring long runs of the same class (split by a dropped short run) are one
    keep = keep[np.r_[True, classes[keep[1:]] != classes[keep[:-1]]]]
    if keep[0] != 0:  # leading short runs belong to the first long run
        keep[0] = 0
    block_start = keep * block  # index into dp
    step_start = pos_idx[block_start]  # index into ts
    seg_start = ticks[step_start]
    next_start = np.r_[step_start[1:], ticks.size - 1]
    seg_end = ticks[next_start]
    bounds = np.r_[block_start, dp.size]
    interval = np.array([np.median(dp[a:b]) for a, b in zip(bounds[:-1], bounds[1:])])
    n = np.diff(np.r_[step_start, ticks.size]).astype(np.int64)
    return IntervalSegments(seg_start, seg_end, interval, n)


def interval_overlap(starts: np.ndarray, ends: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Total length of sorted, non-overlapping intervals ``[starts, ends)`` inside
    each bucket ``[edges[k], edges[k + 1])``. Vectorised, O((n + k) log n)."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 2:
        return np.empty(0)
    if (np.diff(edges) < 0).any():
        raise ValueError("edges must be sorted")
    s = np.asarray(starts, dtype=np.float64)
    e = np.asarray(ends, dtype=np.float64)
    if s.size == 0:
        return np.zeros(edges.size - 1)
    length = np.maximum(e - s, 0)
    before = np.r_[0.0, np.cumsum(length)]

    def covered(t: np.ndarray) -> np.ndarray:
        k = np.searchsorted(s, t, side="right") - 1
        inside = np.where(k >= 0, np.clip(t - s[np.maximum(k, 0)], 0, length[np.maximum(k, 0)]), 0)
        return before[np.maximum(k, 0)] * (k >= 0) + inside

    return np.diff(covered(edges))


class BucketCoverage(NamedTuple):
    """Per bucket: observed time, missing time (in gaps), samples and gap count.
    All fields are additive, e.g. over the series of a building."""

    observed: np.ndarray
    missing: np.ndarray
    samples: np.ndarray
    n_gaps: np.ndarray

    @property
    def coverage(self) -> np.ndarray:
        """Share of observed time without gap (``nan`` where nothing was observed)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.observed > 0, 1 - self.missing / self.observed, np.nan)


def _merged_ranges(ranges: "RangesLike") -> np.ndarray:
    """``[[start, end], ...]`` as a sorted, non-overlapping float64 array of shape (k, 2)."""
    arr = np.asarray(ranges, dtype=np.float64).reshape(-1, 2)
    arr = arr[arr[:, 1] > arr[:, 0]]
    if arr.shape[0] < 2:
        return arr
    arr = arr[np.argsort(arr[:, 0])]
    ends = np.maximum.accumulate(arr[:, 1])
    new = np.r_[True, arr[1:, 0] > ends[:-1]]
    starts = arr[new, 0]
    group_end = np.r_[np.flatnonzero(new)[1:], arr.shape[0]] - 1
    return np.column_stack([starts, ends[group_end]])


def _subtract(starts: np.ndarray, ends: np.ndarray, ranges: np.ndarray, edges: np.ndarray):
    """Per bucket: overlap of the intervals with the (merged) ranges."""
    total = np.zeros(edges.size - 1)
    for a, b in ranges:
        s = np.maximum(starts, a)
        e = np.minimum(ends, b)
        ok = e > s
        if ok.any():
            total += interval_overlap(s[ok], e[ok], edges)
    return total


def bucket_coverage(
    edges: np.ndarray,
    gaps: Gaps,
    first: float,
    obs_end: float,
    ts: Optional[np.ndarray] = None,
    exclude: Optional["RangesLike"] = None,
) -> BucketCoverage:
    """Time-based coverage per bucket ``[edges[k], edges[k + 1])``.

    The observed period is ``[first, obs_end)`` (e.g. first sample to now). Of each
    gap, ``[start + expected, end)`` is missing: the sample before a gap still
    covers one interval.

    Args:
        edges: Sorted bucket edges (ticks).
        gaps: Result of :func:`find_gaps`.
        first: Start of the observed period.
        obs_end: End of the observed period.
        ts: Timestamps for the sample counts (optional).
        exclude: Ranges ``[[start, end], ...]`` that are neither observed nor missing
            (e.g. excluded test data). Optional.

    Returns:
        :class:`BucketCoverage`.
    """
    edges = np.asarray(edges, dtype=np.float64)
    observed = interval_overlap(np.array([first]), np.array([max(first, obs_end)]), edges)
    miss_start = np.maximum(gaps.start.astype(np.float64) + gaps.expected, first)
    miss_end = np.minimum(gaps.end.astype(np.float64), obs_end)
    valid = miss_end > miss_start
    missing = interval_overlap(miss_start[valid], miss_end[valid], edges)
    if exclude is not None:
        ranges = _merged_ranges(exclude)
        if ranges.size:
            observed = np.maximum(
                observed - _subtract(np.array([first]), np.array([max(first, obs_end)]),
                                     ranges, edges), 0)
            missing = np.clip(
                missing - _subtract(miss_start[valid], miss_end[valid], ranges, edges),
                0, observed)
    if ts is not None:
        ticks = _as_ticks(ts)
        # integer edges: comparing int64 ticks with float edges would copy all ticks
        bounds = np.ceil(edges).astype(ticks.dtype)
        samples = np.diff(np.searchsorted(ticks, bounds, side="left")).astype(np.int64)
    else:
        samples = np.zeros(edges.size - 1, dtype=np.int64)
    n_gaps = np.diff(np.searchsorted(gaps.start.astype(np.float64), edges, side="left"))
    return BucketCoverage(observed, missing, samples, n_gaps.astype(np.int64))


def detect_stuck(
    values: np.ndarray,
    ts: np.ndarray,
    min_repeats: int = 20,
    min_duration: float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find runs of identical consecutive values (run-length encoding).

    ``NaN`` values are dropped before the runs are formed.

    Args:
        values: Sample values.
        ts: Sorted timestamps (numeric or ``datetime64``), same length.
        min_repeats: Minimum number of identical consecutive samples.
        min_duration: Minimum run duration (last - first timestamp) in
            timestamp units.

    Returns:
        ``(run_start, run_end, run_value, n_repeats)`` arrays.
    """
    values = np.asarray(values, dtype=np.float64)
    ts = np.asarray(ts)
    present = ~np.isnan(values)
    values, ts = values[present], ts[present]
    if values.size == 0:
        return ts[:0], ts[:0], values[:0], np.zeros(0, dtype=np.int64)
    ticks = _as_ticks(ts)
    run_starts = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1))
    run_ends = np.concatenate((run_starts[1:] - 1, [values.size - 1]))
    counts = run_ends - run_starts + 1
    durations = ticks[run_ends] - ticks[run_starts]
    keep = (counts >= min_repeats) & (durations >= min_duration)
    s, e = run_starts[keep], run_ends[keep]
    return ts[s], ts[e], values[s], counts[keep].astype(np.int64)


def outliers_iqr(values: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """Flag values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` (Tukey fences).

    Returns:
        Boolean mask, ``False`` for ``NaN`` values.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=bool)
    q1, q3 = np.percentile(values[finite], [25.0, 75.0])
    iqr = q3 - q1
    with np.errstate(invalid="ignore"):
        return (values < q1 - multiplier * iqr) | (values > q3 + multiplier * iqr)


def outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Flag values whose absolute z-score exceeds ``threshold``.

    Returns:
        Boolean mask, ``False`` for ``NaN`` values or zero variance.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=bool)
    valid = values[finite]
    std = valid.std()
    if std == 0:
        return np.zeros(values.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        return np.abs(values - valid.mean()) > threshold * std


def outliers_hampel(
    values: np.ndarray,
    window: int = 15,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Hampel filter: flag values deviating from the rolling median.

    A value is an outlier if ``|x - median| > n_sigma * 1.4826 * MAD`` within
    a centred window of ``2 * window + 1`` samples. Robust against level
    changes, unlike global IQR/z-score methods. ``NaN`` values are never
    flagged; for the window medians they are replaced by the previous valid
    value (or the first valid value at the start).

    Returns:
        Boolean mask.
    """
    from scipy.ndimage import median_filter  # C implementation, ~20x faster than rolling

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=bool)
    filled = _fill_forward(values, finite)
    width = 2 * window + 1
    med = median_filter(filled, size=width, mode="nearest")
    deviation = np.abs(filled - med)
    mad = median_filter(deviation, size=width, mode="nearest")
    return (deviation > n_sigma * 1.4826 * mad) & finite


def _fill_forward(values: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Replace non-finite values by the previous finite one (first finite at the start)."""
    if finite.all():
        return values
    idx = np.where(finite, np.arange(values.size), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(finite))
    idx[: first] = first
    return values[idx]
