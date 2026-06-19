"""Thermal comfort and overheating metrics per SIA 180:2014.

Provides the adaptive SIA 180 comfort boundary curves (lower heating
setpoint and upper cooling limit as a function of the 48-hour running
mean outdoor temperature), a summer-half-year filter, and overheating
metrics (overheating hours, monthly breakdown, comfort KPIs) including
the Minergie and SIA 180 residential limits.

The plotting counterparts live in :mod:`pyedautils.plots.comfort`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
def _to_datetime_ns(values) -> pd.Series:
    """Parse to datetime and normalise to nanosecond resolution.

    pandas >= 2.0 preserves non-nanosecond resolutions (e.g. ``datetime64[us]``
    from Parquet). Two series with different units fail to align in
    :func:`pandas.concat`, silently producing an almost-empty join. Casting
    both to nanoseconds avoids that. Works for tz-naive and tz-aware input;
    on pandas < 2.0 (always nanoseconds) it is a no-op.
    """
    s = pd.to_datetime(values)
    as_unit = getattr(getattr(s, "dt", None), "as_unit", None)
    return as_unit("ns") if as_unit is not None else s


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
    r["timestamp"] = _to_datetime_ns(r["timestamp"])
    r = r.set_index("timestamp")["value"].resample("1h").mean().rename("t_room")

    o = outdoor.copy()
    o.columns = ["timestamp", "value"]
    o["timestamp"] = _to_datetime_ns(o["timestamp"])
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


# ---------------------------------------------------------------------------
# Comfort compass — distribution over 9 states x 3 severity stages
# ---------------------------------------------------------------------------
#: The 8 deviation directions (excluding the central "ok"), in the order used
#: by :func:`pyedautils.plots.comfort.plot_comfort_compass` (0 deg = humid,
#: counter-clockwise): humid, warm+humid, warm, warm+dry, dry, cold+dry, cold,
#: cold+humid.
_COMPASS_DIRECTIONS: Tuple[str, ...] = ("f", "wf", "w", "wt", "t", "ct", "c", "kf")
#: Severity stages (innermost -> outermost): light / marked / strong.
_COMPASS_STAGES: Tuple[str, ...] = ("l", "d", "s")
#: (temp_class, hum_class) -> state, where class is -1 (low), 0 (ok), +1 (high).
_COMPASS_STATE_MAP = {
    (0, 0): "ok", (1, 0): "w", (1, 1): "wf", (0, 1): "f", (-1, 1): "kf",
    (-1, 0): "c", (-1, -1): "ct", (0, -1): "t", (1, -1): "wt",
}


def comfort_compass_categories() -> List[str]:
    """Ordered category keys of the comfort-compass distribution.

    ``["ok"]`` followed by ``"<direction>_<stage>"`` for every direction in
    :data:`_COMPASS_DIRECTIONS` and stage in :data:`_COMPASS_STAGES` (25 keys).
    These are exactly the keys produced by :func:`comfort_compass_distribution`
    and consumed by :func:`pyedautils.plots.comfort.plot_comfort_compass`.
    """
    return ["ok"] + [f"{k}_{s}" for k in _COMPASS_DIRECTIONS
                     for s in _COMPASS_STAGES]


def comfort_compass_distribution(
    data: pd.DataFrame,
    temp_band: Tuple[float, float],
    hum_band: Tuple[float, float],
    hum_abs_band: Optional[Tuple[float, float]] = None,
    pressure: float = 101325.0,
    temp_col: str = "temperature",
    hum_col: str = "humidity",
    aggregate_daily: bool = True,
    temp_stage_limits: Tuple[float, float] = (1.0, 2.5),
    hum_stage_limits: Tuple[float, float] = (5.0, 10.0),
    hum_abs_stage_limits: Tuple[float, float] = (0.001, 0.0025),
) -> Dict[str, int]:
    """Count samples per comfort-compass category (one count per day by default).

    Each sample is classified against a fixed comfort band into one of nine
    states — the cross product of temperature {cold, ok, warm} and humidity
    {dry, ok, humid} — and graded into three severity stages from the
    exceedance over the band (temperature in K, humidity in %-points): *mild*
    ``<= limit1``, *moderate* ``<= limit2``, *severe* otherwise (the worse axis
    wins). The result is the input of
    :func:`pyedautils.plots.comfort.plot_comfort_compass`.

    Humidity is judged by **relative** humidity against ``hum_band``. Pass
    ``hum_abs_band`` to *also* judge it by **absolute** humidity: a sample then
    counts as too humid / too dry if it exceeds the relative **or** the absolute
    limit (e.g. so a warm point above the comfort zone's absolute-humidity cap
    reads as "warm + humid" even when its relative humidity is still in range).
    The absolute humidity is derived from temperature, relative humidity and
    ``pressure``.

    By default the input is first averaged to **daily means** (one row per
    calendar day), so the counts are *days* — matching the per-day reading of
    the compass. Pass ``aggregate_daily=False`` to count the rows as given (e.g.
    when the input is already daily).

    Args:
        data: DataFrame holding temperature and humidity columns. With
            ``aggregate_daily=True`` it must have a ``DatetimeIndex`` (used to
            resample to daily means).
        temp_band: ``(low, high)`` comfort temperature band [°C].
        hum_band: ``(low, high)`` comfort *relative* humidity band [%rH].
        hum_abs_band: Optional ``(low, high)`` comfort *absolute* humidity band
            [kg/kg]. When given, "too humid"/"too dry" is the relative **or**
            the absolute criterion. ``None`` (default) judges humidity by
            relative humidity only.
        pressure: Air pressure [Pa], used to derive absolute humidity when
            ``hum_abs_band`` is set. Default 101325.
        temp_col: Temperature column name [°C]. Default ``"temperature"``.
        hum_col: Humidity column name [%rH]. Default ``"humidity"``.
        aggregate_daily: Average to daily means before classifying, so the
            counts are days. Default ``True``.
        temp_stage_limits: ``(light, marked)`` K over the band for the
            temperature severity stages. Default ``(1.0, 2.5)``.
        hum_stage_limits: ``(mild, moderate)`` %-points over the band for the
            relative-humidity severity stages. Default ``(5.0, 10.0)``.
        hum_abs_stage_limits: ``(mild, moderate)`` kg/kg over the band for the
            absolute-humidity severity stages (used only with ``hum_abs_band``).
            Default ``(0.001, 0.0025)`` (1 and 2.5 g/kg).

    Returns:
        dict mapping each key of :func:`comfort_compass_categories` to the
        number of (daily) samples in that category (all zeros when there is no
        valid sample).
    """
    from collections import Counter

    df = data[[temp_col, hum_col]].rename(columns={temp_col: "t", hum_col: "h"})
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["h"] = pd.to_numeric(df["h"], errors="coerce")
    if aggregate_daily:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "aggregate_daily=True requires `data` to have a DatetimeIndex")
        df = df.resample("D").mean()
    df = df.dropna(subset=["t", "h"])

    out = {c: 0 for c in comfort_compass_categories()}
    if df.empty:
        return out

    t = df["t"].to_numpy()
    h = df["h"].to_numpy()
    t_lo, t_hi = float(temp_band[0]), float(temp_band[1])
    h_lo, h_hi = float(hum_band[0]), float(hum_band[1])

    def _level(exc, lims):
        return np.where(exc <= 0, 0,
                        np.where(exc <= lims[0], 1,
                                 np.where(exc <= lims[1], 2, 3)))

    ti = np.where(t < t_lo, -1, np.where(t > t_hi, 1, 0))
    t_exc = np.where(ti == 1, t - t_hi, np.where(ti == -1, t_lo - t, 0.0))
    t_level = _level(t_exc, temp_stage_limits)

    if hum_abs_band is not None:
        # judge humidity by relative OR absolute (the absolute-humidity cap of
        # the comfort zone), so warm points above the cap read as warm+humid.
        from pyedautils._mollier import get_x_y
        x_abs, _ = get_x_y(t, h / 100.0, pressure)        # [kg/kg]
        xa_lo, xa_hi = float(hum_abs_band[0]), float(hum_abs_band[1])
        hi = np.where((h > h_hi) | (x_abs > xa_hi), 1,
                      np.where((h < h_lo) | (x_abs < xa_lo), -1, 0))
        rel_exc = np.where(hi == 1, h - h_hi, np.where(hi == -1, h_lo - h, 0.0))
        abs_exc = np.where(hi == 1, x_abs - xa_hi,
                           np.where(hi == -1, xa_lo - x_abs, 0.0))
        h_level = np.maximum(_level(rel_exc, hum_stage_limits),
                             _level(abs_exc, hum_abs_stage_limits))
    else:
        hi = np.where(h < h_lo, -1, np.where(h > h_hi, 1, 0))
        h_exc = np.where(hi == 1, h - h_hi, np.where(hi == -1, h_lo - h, 0.0))
        h_level = _level(h_exc, hum_stage_limits)

    lvl = np.maximum(t_level, h_level)
    stage = np.where(lvl <= 1, "l", np.where(lvl <= 2, "d", "s"))

    counts: "Counter[str]" = Counter()
    for a, b, sg in zip(ti, hi, stage):
        st = _COMPASS_STATE_MAP[(int(a), int(b))]
        counts["ok" if st == "ok" else f"{st}_{sg}"] += 1
    for cat, cnt in counts.items():
        out[cat] = int(cnt)
    return out
