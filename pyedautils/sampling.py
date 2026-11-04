"""How a time series was sampled and stored, estimated from the data alone.

Building automation archives store values either at a fixed interval or only on a
change of value (COV: a new sample when the value moved by more than a dead band,
sometimes with a periodic "heartbeat" of an unchanged value). Which of the two, and
with which settings, decides which analyses are meaningful (e.g. oscillation
indices need samples well below the oscillation period; a compressed archive biases
spread and spectra, see Thornhill et al. 2004, doi:10.1016/j.jprocont.2003.06.003).

:func:`sampling_profile` estimates these settings from the timestamps and values.
Timestamps are numeric (e.g. int64 epoch milliseconds) or ``datetime64``; durations
are returned in the same unit. ``NaN`` values are dropped.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

from pyedautils.arraystats import _as_ticks

__all__ = [
    "SamplingProfile",
    "interval_histogram",
    "is_change_of_value",
    "on_grid_share",
    "sampling_profile",
]

#: a step this many times the median step is "long"
COV_RATIO = 10.0
#: long steps covering more than this share of the time: change-of-value sampling
COV_SHARE = 0.5
COV_MIN_SAMPLES = 64
#: fewer long steps are outages, not a sampling style
COV_MIN_LONG = 8
#: an interval puts most steps near a few lengths (one rate or a few rates); a step
#: within GRID_TOLERANCE of the median counts as on the grid at once
GRID_TOLERANCE = 0.05
#: width of the log-step bins that find the common step lengths, and how many of them
GRID_BIN = 0.1
GRID_MODES = 3
#: fewer steps on the grid than this share and a wide spread of the steps (p90 / p10
#: above OFF_GRID_SPREAD): no interval, the sender stores on change
GRID_SHARE = 0.5
OFF_GRID_SPREAD = 4.0

#: a heartbeat needs this share of repeated values (and at least HEARTBEAT_MIN of them)
HEARTBEAT_SHARE = 0.01
HEARTBEAT_MIN = 16
#: repeat intervals of a heartbeat are regular: P90 / P10 at most this
HEARTBEAT_SPREAD = 2.0
#: value changes come at most this x the heartbeat apart (median), else periodic sampling
HEARTBEAT_CHANGE_RATIO = 0.5

#: the dead band is this percentile of the non-zero value changes
DEADBAND_PERCENTILE = 5.0
#: resolution: share of value changes that must be whole multiples of the step
RESOLUTION_FIT = 0.95
RESOLUTION_TOLERANCE = 1e-3  # |r - round(r)| for a multiple r of the step (float32 noise)
#: candidate steps far below the largest change are numerical noise (denormals)
RESOLUTION_MAX_RATIO = 1e6
RESOLUTION_CANDIDATES = 8
RESOLUTION_MIN_CHANGES = 16


def is_change_of_value(
    ts: np.ndarray,
    ratio: float = COV_RATIO,
    share: float = COV_SHARE,
    min_samples: int = COV_MIN_SAMPLES,
    min_long: int = COV_MIN_LONG,
) -> bool:
    """Change-of-value sampling: most of the time passes in steps far longer than the
    typical one (a regular series with a few outages does not qualify), or the steps
    follow no interval at all - few lie near a common length and they spread widely
    (a slow signal with a large dead band: long steps, but none of them regular).

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).
        ratio: A step longer than ``ratio`` x the median step is "long".
        share: Long steps must cover more than this share of the time.
        min_samples: Fewer (positive) steps are never change of value.
        min_long: At least this many long steps.

    Returns:
        ``True`` for change-of-value sampling.
    """
    ticks = _as_ticks(ts)
    if ticks.size < min_samples:
        return False
    d = np.diff(ticks).astype(np.float64)
    d = d[d > 0]
    if d.size < min_samples:
        return False
    return _cov_from_steps(d, float(np.median(d)), ratio, share, min_long)


def _cov_from_steps(
    steps: np.ndarray,
    median: float,
    ratio: float = COV_RATIO,
    share: float = COV_SHARE,
    min_long: int = COV_MIN_LONG,
) -> bool:
    """:func:`is_change_of_value` on the positive steps and their median."""
    long = steps > ratio * median
    if int(long.sum()) >= min_long and float(steps[long].sum()) > share * float(steps.sum()):
        return True
    return _off_grid(steps, median)


def on_grid_share(steps: np.ndarray, modes: int = GRID_MODES) -> float:
    """Share of the (positive) steps near one of the ``modes`` most common step lengths
    (``GRID_BIN`` wide on a log scale): about 1 for sampling at one or a few intervals,
    small for change-of-value data."""
    if steps.size == 0:
        return 1.0
    b = np.floor(np.log(steps) / np.log1p(GRID_BIN)).astype(np.int64)
    counts = np.bincount(b - b.min())
    pairs = counts[:-1] + counts[1:] if counts.size > 1 else counts  # a jittered interval
    return min(1.0, float(np.sort(pairs)[-modes:].sum()) / steps.size)


def _off_grid(steps: np.ndarray, median: float) -> bool:
    """No interval: few steps near the common lengths and a wide spread."""
    if median <= 0:
        return False
    if float(np.mean(np.abs(steps - median) <= GRID_TOLERANCE * median)) >= GRID_SHARE:
        return False  # most steps at the median: an interval (the cheap test first)
    p10, p90 = (float(q) for q in np.percentile(steps, [10, 90]))
    if p10 <= 0 or p90 / p10 <= OFF_GRID_SPREAD:
        return False
    return on_grid_share(steps) < GRID_SHARE


def interval_histogram(ts: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Number of sampling intervals (positive time steps) per bin.

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).
        edges: Increasing bin edges in timestamp units; steps below the first edge
            count in the first bin, steps above the last edge in the last bin.

    Returns:
        ``len(edges) - 1`` counts.
    """
    edges_arr = np.asarray(edges, dtype=np.float64)
    if edges_arr.size < 2:
        raise ValueError("edges needs at least two values")
    d = np.diff(_as_ticks(ts))
    d = d[d > 0]
    # bin index by the inner edges: below the first and above the last edge clip
    index = np.searchsorted(edges_arr[1:-1], d, side="right")
    return np.bincount(index, minlength=edges_arr.size - 1).astype(np.int64)


class SamplingProfile(NamedTuple):
    """Result of :func:`sampling_profile`. Durations are in timestamp units; an
    estimate that the data does not support is ``None``."""

    n: int  #: samples (without ``NaN``)
    span: float  #: last - first timestamp
    interval_min: Optional[float]  #: 1st percentile of the steps (minimum send interval)
    interval_p10: Optional[float]
    interval_median: Optional[float]
    interval_p90: Optional[float]
    interval_max: Optional[float]
    change_of_value: bool  #: see :func:`is_change_of_value`
    repeat_share: float  #: share of steps that repeat the previous value
    deadband: Optional[float]  #: smallest typical value change (change of value only)
    resolution: Optional[float]  #: every value change is a multiple of this
    heartbeat: Optional[float]  #: refresh interval of an unchanged value (COV)
    compression: Optional[float]  #: N / m, m = non-zero second differences (Thornhill)
    longest_hold: float  #: longest time a value stood unchanged
    hold_share: float  #: share of the span in holds longer than ``long_hold``


#: the resolution fit is tested on at most this many changes (evenly spread)
RESOLUTION_SAMPLE = 100_000
#: candidate steps are taken from at most this many of the smallest changes
RESOLUTION_SMALLEST = 4096


def _percentile(a: np.ndarray, q: float) -> Optional[float]:
    return float(np.percentile(a, q)) if a.size else None


def _resolution(changes: np.ndarray) -> Optional[float]:
    """The largest step of which (nearly) all value changes are whole multiples."""
    if changes.size < RESOLUTION_MIN_CHANGES:
        return None
    found: Optional[float] = None
    largest = float(changes.max())
    k = min(changes.size, RESOLUTION_SMALLEST)
    smallest = np.partition(changes, k - 1)[:k] if k < changes.size else changes
    sample = changes[:: max(1, changes.size // RESOLUTION_SAMPLE)]
    for step in np.unique(smallest)[:RESOLUTION_CANDIDATES]:
        if largest / step > RESOLUTION_MAX_RATIO:
            continue
        r = sample / step
        if np.mean(np.abs(r - np.round(r)) <= RESOLUTION_TOLERANCE) >= RESOLUTION_FIT:
            found = float(step)
    return found


def _heartbeat(repeat_steps: np.ndarray, dt: np.ndarray, change_mask: np.ndarray) -> Optional[float]:
    """The refresh interval of an unchanged value, if the repeats are regular and the
    changes come much faster: periodic sampling repeats at the same interval as it
    changes, a COV sender changes at any time and refreshes at a fixed interval."""
    p10, beat, p90 = (float(q) for q in np.percentile(repeat_steps, [10, 50, 90]))
    if p10 <= 0 or p90 / p10 > HEARTBEAT_SPREAD:
        return None
    change_steps = dt[change_mask]
    if change_steps.size and float(np.median(change_steps)) < HEARTBEAT_CHANGE_RATIO * beat:
        return beat
    return None


def _holds(ticks: np.ndarray, changed: np.ndarray) -> np.ndarray:
    """How long each value stood: until the next different value (the last one until
    the end)."""
    run_starts = np.flatnonzero(changed) + 1
    starts = np.empty(run_starts.size + 2, dtype=ticks.dtype)
    starts[0], starts[1:-1], starts[-1] = ticks[0], ticks[run_starts], ticks[-1]
    return np.diff(starts).astype(np.float64)


def sampling_profile(
    ts: np.ndarray, values: np.ndarray, long_hold: Optional[float] = None
) -> SamplingProfile:
    """Estimate how a series was sampled and stored.

    - Intervals: percentiles of the positive time steps; ``interval_min`` (1st
      percentile) is the shortest interval the sender typically uses.
    - ``heartbeat``: the median interval between two samples with the same value,
      when such repeats are common and regular while changes come much faster (a
      change-of-value sender refreshing an unchanged value; periodic sampling of a
      mostly constant value repeats at its sampling interval and has no heartbeat).
    - Change of value: :func:`is_change_of_value` or a heartbeat. Then
      ``deadband`` is the 5th percentile of the non-zero value changes, i.e. what a
      value must move before a new sample is stored.
    - ``resolution``: a quantisation step, if (nearly) all value changes are whole
      multiples of it (e.g. 0.1 for values stored with one decimal).
    - ``compression``: N / m with m the number of non-zero second differences of
      the values (Thornhill et al. 2004): about 1 for uncompressed data, larger
      when an archive stores straight segments or holds. Change-of-value data is
      about 1 by construction; there, the dead band describes the compression.
    - ``longest_hold``, ``hold_share``: how long values stood unchanged (from a
      sample to the next sample with a different value); the share counts holds
      longer than ``long_hold`` (``None``: no share, 0).

    Args:
        ts: Sorted timestamps (numeric or ``datetime64``).
        values: Sample values, same length.
        long_hold: Hold duration from which ``hold_share`` counts.

    Returns:
        :class:`SamplingProfile`.
    """
    ticks = _as_ticks(np.asarray(ts))
    vals = np.asarray(values, dtype=np.float64)
    if ticks.shape != vals.shape:
        raise ValueError("ts and values must have the same shape")
    missing = np.isnan(vals)
    if missing.any():
        ticks, vals = ticks[~missing], vals[~missing]
    n = int(ticks.size)
    span = float(ticks[-1] - ticks[0]) if n else 0.0
    dt = np.diff(ticks).astype(np.float64)
    dv = np.diff(vals)
    if (dt < 0).any():
        raise ValueError("ts must be sorted")
    pos = dt > 0
    all_pos = bool(pos.all())
    steps = dt if all_pos else dt[pos]
    quantiles = (
        [float(q) for q in np.percentile(steps, [1, 10, 50, 90])] if steps.size else [None] * 4
    )

    changed = dv != 0
    all_changed = bool(changed.all())
    repeated = pos & ~changed
    n_repeated = int(repeated.sum())
    repeat_share = n_repeated / steps.size if steps.size else 0.0
    changes = np.abs(dv if all_changed else dv[changed])
    heartbeat = None
    if n_repeated >= max(HEARTBEAT_MIN, HEARTBEAT_SHARE * steps.size):
        heartbeat = _heartbeat(dt[repeated], dt, pos & changed)
    # a heartbeat caps the steps, so the long-step test alone misses such COV data
    cov = heartbeat is not None or (
        n >= COV_MIN_SAMPLES
        and steps.size >= COV_MIN_SAMPLES
        and _cov_from_steps(steps, float(quantiles[2]))
    )

    m = int(np.count_nonzero(np.diff(dv))) if n >= 3 else 0
    compression = n / m if m else None

    longest_hold, hold_share = 0.0, 0.0
    if n >= 2:
        holds = dt if all_changed else _holds(ticks, changed)
        longest_hold = float(holds.max())
        if long_hold is not None and span > 0:
            hold_share = float(holds[holds > long_hold].sum() / span)
    return SamplingProfile(
        n=n,
        span=span,
        interval_min=quantiles[0],
        interval_p10=quantiles[1],
        interval_median=quantiles[2],
        interval_p90=quantiles[3],
        interval_max=float(steps.max()) if steps.size else None,
        change_of_value=cov,
        repeat_share=repeat_share,
        deadband=_percentile(changes, DEADBAND_PERCENTILE) if cov else None,
        resolution=_resolution(changes),
        heartbeat=heartbeat,
        compression=compression,
        longest_hold=longest_hold,
        hold_share=hold_share,
    )
