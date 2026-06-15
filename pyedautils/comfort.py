"""Thermal comfort and overheating metrics per SIA 180:2014.

Provides the adaptive SIA 180 comfort boundary curves (lower heating
setpoint and upper cooling limit as a function of the 48-hour running
mean outdoor temperature), a summer-half-year filter, and overheating
metrics (overheating hours, monthly breakdown, comfort KPIs) including
the Minergie and SIA 180 residential limits.

The plotting counterparts live in :mod:`pyedautils.plots.comfort`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SIA 180 constants
# ---------------------------------------------------------------------------
#: Minergie overheating limit: max 100 h/year above the adaptive boundary.
MINERGIE_OVERHEATING_LIMIT_H = 100.0
#: SIA 180 residential overheating limit: max 400 h/year.
SIA180_RESIDENTIAL_OVERHEATING_LIMIT_H = 400.0
#: Fixed overheating threshold [°C] (Minergie / SIA 180 fixed criterion).
FIXED_OVERHEATING_THRESHOLD = 26.5

# Adaptive maximum comfort temperature (mechanically ventilated / actively
# cooled):
#   T_oa_48h <= 12 °C       ->  T_max = 24.5 °C
#   12 °C < T_oa_48h < 17.5 °C  ->  linear interpolation
#   T_oa_48h >= 17.5 °C     ->  T_max = 26.5 °C
ADAPTIVE_MAX_T_OA_LO = 12.0
ADAPTIVE_MAX_T_OA_HI = 17.5
ADAPTIVE_MAX_T_ROOM_LO = 24.5
ADAPTIVE_MAX_T_ROOM_HI = 26.5

# Adaptive minimum comfort temperature (heating setpoint):
#   T_oa_48h <= 19 °C       ->  T_min = 20.5 °C
#   19 °C < T_oa_48h < 23.5 °C  ->  linear interpolation
#   T_oa_48h >= 23.5 °C     ->  T_min = 22.0 °C
ADAPTIVE_MIN_T_OA_LO = 19.0
ADAPTIVE_MIN_T_OA_HI = 23.5
ADAPTIVE_MIN_T_ROOM_LO = 20.5
ADAPTIVE_MIN_T_ROOM_HI = 22.0


# ---------------------------------------------------------------------------
# SIA 180 boundary curves
# ---------------------------------------------------------------------------
def sia180_max_temp(t_oa_48h) -> np.ndarray:
    """SIA 180 adaptive maximum comfort temperature [°C].

    Upper comfort boundary as a function of the 48-hour running mean
    outdoor temperature. Below :data:`ADAPTIVE_MAX_T_OA_LO` the limit is
    constant at :data:`ADAPTIVE_MAX_T_ROOM_LO`, above
    :data:`ADAPTIVE_MAX_T_OA_HI` it is constant at
    :data:`ADAPTIVE_MAX_T_ROOM_HI`, and linear in between.

    Args:
        t_oa_48h: Scalar or array of 48-hour running mean outdoor
            temperatures [°C].

    Returns:
        np.ndarray: Maximum comfort temperature [°C], same shape as input.
    """
    t = np.asarray(t_oa_48h, dtype=float)
    slope = (ADAPTIVE_MAX_T_ROOM_HI - ADAPTIVE_MAX_T_ROOM_LO) / (
        ADAPTIVE_MAX_T_OA_HI - ADAPTIVE_MAX_T_OA_LO
    )
    return np.where(
        t <= ADAPTIVE_MAX_T_OA_LO, ADAPTIVE_MAX_T_ROOM_LO,
        np.where(
            t >= ADAPTIVE_MAX_T_OA_HI, ADAPTIVE_MAX_T_ROOM_HI,
            ADAPTIVE_MAX_T_ROOM_LO + slope * (t - ADAPTIVE_MAX_T_OA_LO),
        ),
    )


def sia180_min_temp(t_oa_48h) -> np.ndarray:
    """SIA 180 adaptive minimum comfort (heating setpoint) temperature [°C].

    Lower comfort boundary as a function of the 48-hour running mean
    outdoor temperature. Below :data:`ADAPTIVE_MIN_T_OA_LO` the limit is
    constant at :data:`ADAPTIVE_MIN_T_ROOM_LO`, above
    :data:`ADAPTIVE_MIN_T_OA_HI` it is constant at
    :data:`ADAPTIVE_MIN_T_ROOM_HI`, and linear in between.

    Args:
        t_oa_48h: Scalar or array of 48-hour running mean outdoor
            temperatures [°C].

    Returns:
        np.ndarray: Minimum comfort temperature [°C], same shape as input.
    """
    t = np.asarray(t_oa_48h, dtype=float)
    slope = (ADAPTIVE_MIN_T_ROOM_HI - ADAPTIVE_MIN_T_ROOM_LO) / (
        ADAPTIVE_MIN_T_OA_HI - ADAPTIVE_MIN_T_OA_LO
    )
    return np.where(
        t <= ADAPTIVE_MIN_T_OA_LO, ADAPTIVE_MIN_T_ROOM_LO,
        np.where(
            t >= ADAPTIVE_MIN_T_OA_HI, ADAPTIVE_MIN_T_ROOM_HI,
            ADAPTIVE_MIN_T_ROOM_LO + slope * (t - ADAPTIVE_MIN_T_OA_LO),
        ),
    )


def is_summer_semester_sia180(timestamps) -> pd.Series:
    """Boolean mask for the SIA 180 summer half-year (16 Apr – 15 Oct).

    Args:
        timestamps: Anything accepted by :func:`pandas.to_datetime`
            (Series, DatetimeIndex, list of timestamps).

    Returns:
        pd.Series: Boolean Series, ``True`` for timestamps within the
        summer half-year (inclusive).
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    month = ts.dt.month
    day = ts.dt.day
    return (
        ((month == 4) & (day >= 16))
        | (month.isin([5, 6, 7, 8, 9]))
        | ((month == 10) & (day <= 15))
    )


# ---------------------------------------------------------------------------
# Indoor + outdoor alignment
# ---------------------------------------------------------------------------
def align_hourly(
    room: pd.DataFrame,
    outdoor: pd.DataFrame,
    rolling_window: int = 48,
    min_periods: int = 12,
) -> pd.DataFrame:
    """Resample room and outdoor temperature to hourly means and join them.

    Computes the 48-hour running mean of the outdoor temperature used by
    the adaptive SIA 180 boundary curves.

    Args:
        room: DataFrame ``[timestamp, value]`` with room temperature [°C].
        outdoor: DataFrame ``[timestamp, value]`` with outdoor
            temperature [°C].
        rolling_window: Window length (hours) for the outdoor running mean.
            Default 48.
        min_periods: Minimum number of observations in the rolling window.
            Default 12.

    Returns:
        pd.DataFrame: Indexed by hourly timestamp with columns
        ``[t_room, t_oa, t_oa_48h]``. Empty if either input is empty.
    """
    cols = ["t_room", "t_oa", "t_oa_48h"]
    if room.empty or outdoor.empty:
        return pd.DataFrame(columns=cols)

    r = room.copy()
    r.columns = ["timestamp", "value"]
    r["timestamp"] = pd.to_datetime(r["timestamp"])
    r = r.set_index("timestamp")["value"].resample("1h").mean().rename("t_room")

    o = outdoor.copy()
    o.columns = ["timestamp", "value"]
    o["timestamp"] = pd.to_datetime(o["timestamp"])
    o = o.set_index("timestamp")["value"].resample("1h").mean().rename("t_oa")

    joined = pd.concat([r, o], axis=1).dropna()
    joined["t_oa_48h"] = joined["t_oa"].rolling(
        rolling_window, min_periods=min_periods
    ).mean()
    return joined.dropna(subset=["t_oa_48h"])


# ---------------------------------------------------------------------------
# Comfort / overheating metrics
# ---------------------------------------------------------------------------
def comfort_kpis(aligned: pd.DataFrame, summer_only: bool = False) -> Dict[str, float]:
    """Comfort key performance indicators from aligned hourly data.

    Counts hours below the adaptive lower limit (too cold), above the
    adaptive upper limit (too warm / overheating), and within the comfort
    band, and checks the Minergie and SIA 180 residential overheating
    limits.

    Args:
        aligned: DataFrame indexed by timestamp with columns
            ``[t_room, t_oa, t_oa_48h]`` (see :func:`align_hourly`).
        summer_only: Restrict to the SIA 180 summer half-year. Default
            ``False``.

    Returns:
        dict: ``n_hours``, ``h_cold``, ``h_warm``, ``h_ok``, ``pct_ok``,
        ``overheating_h``, ``sia180_compliant``, ``minergie_compliant``.
        One sample equals one hour.
    """
    df = aligned.copy()
    if summer_only:
        df = df[is_summer_semester_sia180(df.index).to_numpy()]
    if df.empty:
        return {
            "n_hours": 0,
            "h_cold": 0.0, "h_warm": 0.0, "h_ok": 0.0,
            "pct_ok": 0.0,
            "overheating_h": 0.0,
            "sia180_compliant": False,
            "minergie_compliant": False,
        }
    t_max = sia180_max_temp(df["t_oa_48h"])
    t_min = sia180_min_temp(df["t_oa_48h"])
    cold = df["t_room"] < t_min
    warm = df["t_room"] > t_max
    n = len(df)
    n_cold = int(cold.sum())
    n_warm = int(warm.sum())
    n_ok = n - n_cold - n_warm
    return {
        "n_hours":            n,
        "h_cold":             float(n_cold),
        "h_warm":             float(n_warm),
        "h_ok":               float(n_ok),
        "pct_ok":             round(100.0 * n_ok / n, 1) if n else 0.0,
        "overheating_h":      float(n_warm),
        "sia180_compliant":   n_warm <= SIA180_RESIDENTIAL_OVERHEATING_LIMIT_H,
        "minergie_compliant": n_warm <= MINERGIE_OVERHEATING_LIMIT_H,
    }


def overheating_hours(
    aligned: pd.DataFrame,
    method: str = "adaptive",
    summer_only: bool = True,
    business_hours_only: bool = False,
) -> Tuple[float, pd.Series]:
    """Overheating hours and the per-hour threshold series.

    Args:
        aligned: DataFrame indexed by timestamp with columns
            ``[t_room, t_oa, t_oa_48h]`` (see :func:`align_hourly`).
        method: ``"adaptive"`` uses the SIA 180 upper boundary curve
            (:func:`sia180_max_temp`); ``"fixed"`` uses
            :data:`FIXED_OVERHEATING_THRESHOLD`. Default ``"adaptive"``.
        summer_only: Restrict to the SIA 180 summer half-year. Default
            ``True``.
        business_hours_only: Restrict to 07:00–22:00. Default ``False``.

    Returns:
        tuple: ``(total_hours, threshold_series)`` where *threshold_series*
        is the per-hour comfort threshold (useful for plotting). One
        exceeded sample equals one overheating hour.
    """
    df = aligned.copy()
    if summer_only:
        df = df[is_summer_semester_sia180(df.index).to_numpy()]
    if business_hours_only:
        h = df.index.hour
        df = df[(h >= 7) & (h < 22)]
    if df.empty:
        return 0.0, pd.Series(dtype=float)

    if method == "adaptive":
        threshold = pd.Series(sia180_max_temp(df["t_oa_48h"]), index=df.index)
    else:
        threshold = pd.Series(FIXED_OVERHEATING_THRESHOLD, index=df.index)
    too_warm = df["t_room"] > threshold
    return float(too_warm.sum()), threshold


def overheating_per_month(aligned: pd.DataFrame, threshold: pd.Series) -> pd.DataFrame:
    """Overheating hours grouped by calendar month.

    Args:
        aligned: DataFrame indexed by timestamp with a ``t_room`` column.
        threshold: Per-hour comfort threshold (the second return value of
            :func:`overheating_hours`).

    Returns:
        pd.DataFrame: Columns ``[month, hours]`` (month as 1–12).
    """
    if aligned.empty or threshold.empty:
        return pd.DataFrame(columns=["month", "hours"])
    df = aligned.loc[threshold.index].copy()
    df["over"] = (df["t_room"] > threshold).astype(int)
    df["month"] = df.index.month
    out = df.groupby("month")["over"].sum().reset_index()
    out.columns = ["month", "hours"]
    return out
