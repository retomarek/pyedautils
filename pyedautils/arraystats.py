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

from typing import Optional, TypedDict

import numpy as np

__all__ = [
    "Description",
    "describe",
    "infer_interval",
    "gap_indices",
    "detect_gaps",
    "detect_stuck",
    "outliers_iqr",
    "outliers_zscore",
    "outliers_hampel",
]


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


def describe(
    values: np.ndarray,
    ts: Optional[np.ndarray] = None,
    expected_interval: Optional[float] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
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
) -> np.ndarray:
    """Indices ``i`` where a gap lies between ``ts[i]`` and ``ts[i + 1]``.

    A gap is a time difference larger than ``max(factor * expected_interval,
    min_gap)``; the interval is inferred via :func:`infer_interval` when
    ``None``. Returns an int64 array (empty if there are no gaps).
    """
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

    Returns:
        ``(gap_start, gap_end)``: arrays with the last timestamp before and
        the first timestamp after each gap (same dtype as ``ts``).
    """
    ts = np.asarray(ts)
    idx = gap_indices(ts, expected_interval, factor, min_gap)
    return ts[idx], ts[idx + 1]


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
