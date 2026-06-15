"""Heating and cooling gradients (rate of change per hour) of a time series.

Sign-based method: the series is resampled to a fixed frequency (hourly
by default), then the step-to-step difference (the gradient) is taken.
Differencing does *not* bridge data gaps — missing steps become NaN and
are dropped. A fixed threshold separates relevant heating/cooling phases
from the noise of small fluctuations.

Seasons are classified via :func:`pyedautils.data_prep.season.get_season`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pyedautils.data_prep.season import get_season

_OUTPUT_COLUMNS = ["timestamp", "gradient", "direction", "season"]


def _as_series(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Coerce input to a Series with a DatetimeIndex.

    Accepts a Series (DatetimeIndex) or a two-column ``[timestamp, value]``
    DataFrame.
    """
    if isinstance(data, pd.Series):
        s = data.copy()
        s.index = pd.to_datetime(s.index)
        return s
    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")["value"]


def compute_gradients(
    data: Union[pd.Series, pd.DataFrame],
    *,
    threshold: float = 0.0,
    freq: str = "1h",
    tracking_type: str = "astronomical",
    direction_labels: Tuple[str, str] = ("heating", "cooling"),
    season_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Step-to-step gradients of a time series, classified by direction and season.

    Args:
        data: A pandas Series with a DatetimeIndex, or a two-column
            ``[timestamp, value]`` DataFrame.
        threshold: Absolute magnitude threshold (per *freq* step); only
            ``|gradient| > threshold`` is kept. Default 0 (keep all
            non-zero steps).
        freq: Resampling frequency for the mean before differencing.
            Default ``"1h"`` (gradient is then per hour).
        tracking_type: Season definition passed to :func:`get_season`
            (``"astronomical"`` or ``"meteorological"``).
        direction_labels: ``(rising, falling)`` labels. Default
            ``("heating", "cooling")``.
        season_labels: Optional custom season labels (order: spring,
            summer, fall, winter) passed to :func:`get_season`.

    Returns:
        pd.DataFrame: One row per kept step with columns
        ``[timestamp, gradient, direction, season]``. ``direction`` is the
        first label for ``gradient > 0`` and the second for
        ``gradient < 0``.
    """
    s = _as_series(data)
    resampled = s.sort_index().resample(freq).mean()
    grad = resampled.diff().dropna()
    grad = grad[grad.abs() > threshold]
    if grad.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    rising, falling = direction_labels
    out = pd.DataFrame({
        "timestamp": grad.index,
        "gradient": grad.to_numpy(),
    })
    out["direction"] = np.where(out["gradient"] > 0, rising, falling)

    # Compute the season only once per unique date, then merge back.
    out["_date"] = out["timestamp"].dt.normalize()
    date_df = pd.DataFrame({"_date": pd.unique(out["_date"])})
    season_kwargs = {"tracking_type": tracking_type}
    if season_labels is not None:
        season_kwargs["labels"] = season_labels
    date_df["season"] = get_season(date_df["_date"], **season_kwargs)
    out = out.merge(date_df, on="_date", how="left").drop(columns="_date")
    return out[_OUTPUT_COLUMNS]


def summarize_gradients(
    gradients: pd.DataFrame,
    direction_labels: Tuple[str, str] = ("heating", "cooling"),
    decimals: int = 2,
) -> pd.DataFrame:
    """Aggregate gradients by direction (mean / min / max / count).

    Args:
        gradients: Output of :func:`compute_gradients`.
        direction_labels: ``(rising, falling)`` labels, used to order rows
            and to guarantee both rows exist even if one direction is empty.
        decimals: Rounding for the statistics. Default 2.

    Returns:
        pd.DataFrame: One row per direction with columns
        ``[direction, mean, min, max, n]``. Statistics are NaN when a
        direction has no steps.
    """
    cols = ["direction", "mean", "min", "max", "n"]
    rows = []
    for label in direction_labels:
        vals = gradients.loc[gradients["direction"] == label, "gradient"] \
            if not gradients.empty else pd.Series(dtype=float)
        if len(vals):
            rows.append({
                "direction": label,
                "mean": round(float(vals.mean()), decimals),
                "min": round(float(vals.min()), decimals),
                "max": round(float(vals.max()), decimals),
                "n": int(len(vals)),
            })
        else:
            rows.append({"direction": label, "mean": np.nan,
                         "min": np.nan, "max": np.nan, "n": 0})
    return pd.DataFrame(rows, columns=cols)


def mean_gradients_by_season(
    gradients: pd.DataFrame,
    season_order: Optional[List[str]] = None,
    direction_order: Optional[Tuple[str, str]] = ("heating", "cooling"),
) -> pd.DataFrame:
    """Mean gradient and count per season and direction.

    Args:
        gradients: Output of :func:`compute_gradients`.
        season_order: Optional ordered list of season labels for sorting.
        direction_order: Optional ordered direction labels for sorting.

    Returns:
        pd.DataFrame: Columns ``[season, direction, mean, n]``.
    """
    cols = ["season", "direction", "mean", "n"]
    if gradients.empty:
        return pd.DataFrame(columns=cols)
    agg = (
        gradients.groupby(["season", "direction"])["gradient"]
        .agg(mean="mean", n="size")
        .reset_index()
    )
    if season_order is not None:
        agg["season"] = pd.Categorical(agg["season"], categories=season_order,
                                       ordered=True)
    if direction_order is not None:
        agg["direction"] = pd.Categorical(agg["direction"],
                                          categories=list(direction_order),
                                          ordered=True)
    return agg.sort_values(["season", "direction"]).reset_index(drop=True)[cols]
