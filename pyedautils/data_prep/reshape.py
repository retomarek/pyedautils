"""Reshaping helpers for long-format sensor data.

Long format here means one row per ``(datapoint_id, timestamp)`` with a
``value`` column — the typical shape of data pulled from a time-series
database before pivoting to a wide, plot-ready frame.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def aggregate_long(
    df: pd.DataFrame,
    freq: Optional[str],
    func: str = "mean",
    id_col: str = "datapoint_id",
    time_col: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """Resample long-format data per id.

    Args:
        df: Long-format DataFrame with id, timestamp and value columns.
        freq: A pandas offset alias (e.g. ``"1h"``, ``"1D"``, ``"1W"``,
            ``"1ME"``). *None* returns *df* unchanged (raw resolution).
        func: Aggregation function name (``"mean"``, ``"min"``, ``"max"``,
            ``"median"``, ``"sum"``, …). Default ``"mean"``.
        id_col: Id column name. Default ``"datapoint_id"``.
        time_col: Timestamp column name. Default ``"timestamp"``.
        value_col: Value column name. Default ``"value"``.

    Returns:
        pd.DataFrame: Same long format, resampled per id.
    """
    if freq is None:
        return df
    out = (
        df.set_index(time_col)
        .groupby(id_col)[value_col]
        .resample(freq)
        .agg(func)
        .reset_index()
    )
    return out


def remove_outliers_iqr(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    id_col: str = "datapoint_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Drop outliers per id using the IQR fence.

    Values outside ``[Q1 - multiplier·IQR, Q3 + multiplier·IQR]`` are
    removed, computed independently per id.

    Args:
        df: Long-format DataFrame.
        multiplier: IQR multiplier. Default 3.0.
        id_col: Id column name. Default ``"datapoint_id"``.
        value_col: Value column name. Default ``"value"``.

    Returns:
        pd.DataFrame: Filtered DataFrame (same columns).
    """
    if df.empty:
        return df
    out = []
    for _, group in df.groupby(id_col, sort=False):
        q1, q3 = group[value_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        out.append(group[(group[value_col] >= lo) & (group[value_col] <= hi)])
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


def to_wide(
    df_long: pd.DataFrame,
    id_to_label: Optional[Dict[str, str]] = None,
    id_col: str = "datapoint_id",
    time_col: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """Pivot long-format data to wide (timestamp index, one column per id).

    Args:
        df_long: Long-format DataFrame.
        id_to_label: Optional mapping from id (as ``str``) to a column
            label. Ids not present in the mapping are dropped. If *None*,
            the ids themselves are used as column labels.
        id_col: Id column name. Default ``"datapoint_id"``.
        time_col: Timestamp column name. Default ``"timestamp"``.
        value_col: Value column name. Default ``"value"``.

    Returns:
        pd.DataFrame: Wide DataFrame indexed by timestamp. Duplicate
        ``(timestamp, id)`` pairs keep the first value.
    """
    if df_long.empty:
        return pd.DataFrame()
    df = df_long.copy()
    if id_to_label is not None:
        df["__label"] = df[id_col].astype(str).map(id_to_label)
        df = df.dropna(subset=["__label"])
        col = "__label"
    else:
        df["__label"] = df[id_col].astype(str)
        col = "__label"
    wide = df.pivot_table(
        index=time_col, columns=col, values=value_col, aggfunc="first",
    )
    wide.columns.name = None
    return wide
