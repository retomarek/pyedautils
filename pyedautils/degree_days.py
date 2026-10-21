"""Degree days and climate correction (Witterungsbereinigung) of heating energy.

Provides the Swiss degree-day conventions and the classic degree-day based
climate correction of measured heating energy:

- **HGT 20/12** heating degree days per SIA 381/3 and MeteoSwiss: sum of
  ``20 °C − T_mean`` over all days with ``T_mean <= 12 °C`` (heating days HT).
  Used by the BFE (population-weighted Swiss series since 1980) and most
  cantonal energy statistics.
- **ATD** accumulated temperature difference per SIA 380:2015 and
  SN EN ISO 15927-6: sum of ``max(0, T_base − T_mean)`` with a single base
  (heating-limit) temperature, ideally building-specific.
- **GT** plain degree days ``20 °C − T_mean`` without heating limit (basis
  of the BFE/Prognos GT&S method) and **CDD** cooling degree days
  (BFE: base 18.3 °C).
- A reference climatology (monthly means over a climate-normal period such
  as 1991–2020), the correction factor ``DD_ref / DD_actual`` and the
  normalization of a consumption whose weather-dependent share is known.
- A fixed-base degree-day regression (``E = a + b · HDD``) with ASHRAE
  Guideline 14 fit metrics, usable with the balance temperature from
  :func:`pyedautils.energy_signature.compute_pes`.
- The generic GT&S correction factor (Prognos 2003/2015 for the BFE:
  degree days *and* solar radiation with monthly elasticities and a
  normal-year load curve).

Plotting counterparts live in :mod:`pyedautils.plots.degree_days`; Swiss
open-data sources for reference values in
:mod:`pyedautils.weather.open_data_ch` and
:mod:`pyedautils.weather.meteo_swiss`.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pyedautils.comfort import _to_datetime_ns
from pyedautils.data_quality import infer_interval

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Room temperature of the Swiss HGT 20/12 convention (SIA 381/3) [°C].
HGT_ROOM_TEMP_CH = 20.0
#: Heating limit of the Swiss HGT 20/12 convention (SIA 381/3) [°C].
HGT_HEATING_LIMIT_CH = 12.0
#: Base temperature of the BFE cooling degree days [°C].
CDD_BASE_BFE = 18.3
#: Base of the plain degree days ("Gradtage") used by the GT&S method [°C].
GT_BASE = 20.0

_ONE_DAY = pd.Timedelta("1D")
_AGG_FREQS = ("MS", "YS", "season")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def _to_series(data, name: str = "value") -> pd.Series:
    """Normalise user input to a float Series with a sorted DatetimeIndex.

    Accepts a Series with a DatetimeIndex, a DataFrame with a DatetimeIndex
    and one column, or a positional ``[timestamp, value]`` DataFrame.
    Duplicate timestamps are averaged, timestamps are cast to nanoseconds.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] >= 2:
            ts = _to_datetime_ns(data.iloc[:, 0])
            values = pd.to_numeric(data.iloc[:, 1], errors="coerce")
            s = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(ts), name=name)
        elif data.shape[1] == 1 and isinstance(data.index, pd.DatetimeIndex):
            s = pd.to_numeric(data.iloc[:, 0], errors="coerce").astype(float).rename(name)
        else:
            raise ValueError(
                "DataFrame must have a DatetimeIndex and one column or "
                "positional [timestamp, value] columns"
            )
    elif isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex")
        s = pd.to_numeric(data, errors="coerce").astype(float).rename(name)
    else:
        raise ValueError("Input must be a pandas Series or DataFrame")

    s.index = pd.DatetimeIndex(_to_datetime_ns(pd.Series(s.index)))
    s = s[s.index.notna()]
    if s.empty:
        raise ValueError("Input contains no timestamps")
    s = s.sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    return s


def _interval(s: pd.Series) -> pd.Timedelta:
    """Sampling interval of a Series (1 day for a single observation)."""
    if len(s) < 2:
        return _ONE_DAY
    return infer_interval(pd.Series(s.index))


def _daily_mean(s: pd.Series, min_coverage: float = 0.75) -> pd.DataFrame:
    """Daily mean temperature with per-day coverage.

    Sub-daily input is resampled to daily means; days with a sample
    coverage below ``min_coverage`` are set to NaN. Daily input is returned
    on normalised (midnight) timestamps.

    Returns:
        pd.DataFrame: Columns ``t_mean`` and ``coverage`` indexed by day.
    """
    interval = _interval(s)
    if interval > _ONE_DAY:
        raise ValueError(
            f"Temperature resolution of {interval} is coarser than daily; "
            "daily or finer data is required"
        )
    if interval == _ONE_DAY:
        daily = s.groupby(s.index.normalize()).mean()
        coverage = daily.notna().astype(float)
    else:
        expected = _ONE_DAY / interval
        grouped = s.resample("D")
        daily = grouped.mean()
        coverage = (grouped.count() / expected).clip(upper=1.0)
        daily[coverage < min_coverage] = np.nan
    return pd.DataFrame({"t_mean": daily, "coverage": coverage})


def _period_start(index: pd.DatetimeIndex, freq: str, season_start_month: int) -> pd.DatetimeIndex:
    """Start timestamp of the aggregation period for every day in ``index``."""
    year = index.year
    if freq == "MS":
        month = index.month
    elif freq == "YS":
        month = np.ones(len(index), dtype=int)
    else:  # season
        year = np.where(index.month >= season_start_month, year, year - 1)
        month = np.full(len(index), season_start_month)
    starts = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))
    starts = pd.DatetimeIndex(starts)
    if index.tz is not None:
        starts = starts.tz_localize(index.tz)
    return starts


def _calendar_days(index: pd.DatetimeIndex, freq: str, season_start_month: int) -> pd.DatetimeIndex:
    """Complete daily calendar from the first to the last aggregation period."""
    starts = _period_start(index[[0, -1]], freq, season_start_month)
    offset = pd.DateOffset(months=1) if freq == "MS" else pd.DateOffset(years=1)
    end = starts[-1] + offset - _ONE_DAY
    return pd.date_range(starts[0], end, freq="D", tz=index.tz)


# ---------------------------------------------------------------------------
# 1. Degree-day primitives
# ---------------------------------------------------------------------------
def degree_days(
    temperature,
    base: float = HGT_ROOM_TEMP_CH,
    threshold: Optional[float] = HGT_HEATING_LIMIT_CH,
    mode: str = "heating",
    method: str = "daily_mean",
    min_coverage: float = 0.75,
) -> pd.DataFrame:
    """Daily degree days of an outdoor temperature series.

    Heating: a day counts as degree-day day when ``t_mean <= threshold``
    (``threshold=None``: when ``t_mean < base``); its degree days are
    ``base − t_mean``. Cooling: a day counts when ``t_mean >= threshold``
    (``None``: ``base``); its degree days are ``t_mean − base``.

    Args:
        temperature: Outdoor temperature [°C] as Series with DatetimeIndex,
            DataFrame with DatetimeIndex and one column, or positional
            ``[timestamp, value]`` DataFrame. Daily or finer resolution.
        base: Reference (room) temperature [°C]. Default 20.
        threshold: Heating limit [°C] deciding whether a day counts.
            Default 12 (Swiss HGT 20/12). ``None`` uses ``base``.
        mode: ``"heating"`` or ``"cooling"``.
        method: ``"daily_mean"`` (degree days from the daily mean) or
            ``"hourly"`` (ISO 15927-6 style: mean of the positive hourly
            differences, i.e. degree-hours / hours; requires sub-daily data).
            With a ``threshold`` the day still has to qualify via its daily
            mean; without, every hour below ``base`` counts.
        min_coverage: Minimum fraction of expected samples per day for
            sub-daily data; days below are NaN. Default 0.75.

    Returns:
        pd.DataFrame: Indexed by day with columns ``t_mean`` [°C],
        ``dd`` [Kd] (0 on non-degree-day days, NaN on missing days),
        ``is_dd_day`` (bool) and ``coverage`` (0–1).

    Raises:
        ValueError: On invalid mode/method, ``base <= threshold`` for
            heating (``base < threshold`` for cooling), coarser-than-daily
            input or ``method="hourly"`` with daily input.
    """
    if mode not in ("heating", "cooling"):
        raise ValueError("mode must be 'heating' or 'cooling'")
    if method not in ("daily_mean", "hourly"):
        raise ValueError("method must be 'daily_mean' or 'hourly'")
    if threshold is not None:
        if mode == "heating" and base <= threshold:
            raise ValueError("base must be greater than threshold for heating degree days")
        if mode == "cooling" and base > threshold:
            raise ValueError("base must not exceed threshold for cooling degree days")

    s = _to_series(temperature, "temperature")
    daily = _daily_mean(s, min_coverage)
    t_mean = daily["t_mean"]

    if mode == "heating":
        is_dd_day = (t_mean < base) if threshold is None else (t_mean <= threshold)
        diff_daily = base - t_mean
        diff_fine = base - s
    else:
        is_dd_day = t_mean >= (base if threshold is None else threshold)
        diff_daily = t_mean - base
        diff_fine = s - base
    is_dd_day = is_dd_day.astype(bool)

    if method == "hourly":
        if _interval(s) >= _ONE_DAY:
            raise ValueError("method='hourly' requires sub-daily temperature data")
        dd_raw = diff_fine.clip(lower=0).resample("D").mean().reindex(t_mean.index)
        if threshold is None:
            # ISO 15927-6: every hour below the base counts, no daily gating
            is_dd_day = (dd_raw > 0) & t_mean.notna()
    else:
        dd_raw = diff_daily.clip(lower=0)

    dd = pd.Series(np.where(is_dd_day, dd_raw, 0.0), index=t_mean.index)
    dd[t_mean.isna()] = np.nan

    return pd.DataFrame({
        "t_mean": t_mean,
        "dd": dd,
        "is_dd_day": is_dd_day,
        "coverage": daily["coverage"],
    })


def aggregate_degree_days(
    daily: pd.DataFrame,
    freq: Optional[str] = "MS",
    scale_incomplete: bool = False,
    season_start_month: int = 7,
) -> pd.DataFrame:
    """Aggregate daily degree days to months, years or heating seasons.

    Args:
        daily: Output of :func:`degree_days`.
        freq: ``"MS"`` (calendar months), ``"YS"`` (calendar years),
            ``"season"`` (heating seasons starting at ``season_start_month``)
            or ``None`` (return ``daily`` unchanged).
        scale_incomplete: Scale ``dd`` and ``dd_days`` of periods with
            missing days by ``n_days / n_valid`` (IWU rule for incomplete
            months). Default False.
        season_start_month: First month of a heating season (default 7,
            i.e. July–June, labelled e.g. ``"2022/2023"``).

    Returns:
        pd.DataFrame: Indexed by period start with columns ``dd``,
        ``dd_days`` (number of degree-day days), ``n_days`` (calendar days
        of the period), ``n_valid`` (days with data), ``completeness``
        (``n_valid / n_days``) and ``t_mean``. ``"season"`` adds a
        ``season`` label column. Periods without any data have NaN ``dd``.

    Raises:
        ValueError: On an unsupported ``freq`` (note that ``"M"``/``"Y"``
            are rejected; use ``"MS"``/``"YS"``).
    """
    if freq is None:
        return daily
    if freq not in _AGG_FREQS:
        raise ValueError(f"freq must be one of {_AGG_FREQS} or None, got {freq!r}")
    if daily.empty:
        raise ValueError("daily degree days are empty")

    calendar = _calendar_days(daily.index, freq, season_start_month)
    full = daily.reindex(calendar)
    full["is_dd_day"] = full["is_dd_day"].eq(True)
    keys = _period_start(full.index, freq, season_start_month)

    grouped = full.groupby(keys)
    out = pd.DataFrame({
        "dd": grouped["dd"].sum(min_count=1),
        "dd_days": grouped["is_dd_day"].sum().astype(float),
        "n_days": grouped.size().astype(int),
        "n_valid": grouped["t_mean"].count().astype(int),
        "t_mean": grouped["t_mean"].mean(),
    })
    out["completeness"] = out["n_valid"] / out["n_days"]
    out = out[["dd", "dd_days", "n_days", "n_valid", "completeness", "t_mean"]]

    if scale_incomplete:
        factor = (out["n_days"] / out["n_valid"].replace(0, np.nan))
        out["dd"] = out["dd"] * factor
        out["dd_days"] = (out["dd_days"] * factor).clip(upper=out["n_days"])

    if freq == "season":
        out["season"] = [f"{ts.year}/{ts.year + 1}" for ts in out.index]
    out.index.name = "period"
    return out


def _dd_wrapper(daily: pd.DataFrame, freq, scale_incomplete, dd_name: str, days_name: str) -> pd.DataFrame:
    agg = aggregate_degree_days(daily, freq, scale_incomplete)
    return agg.rename(columns={"dd": dd_name, "dd_days": days_name})


def heating_degree_days(
    temperature,
    room_temp: float = HGT_ROOM_TEMP_CH,
    heating_limit: Optional[float] = HGT_HEATING_LIMIT_CH,
    freq: Optional[str] = "MS",
    method: str = "daily_mean",
    min_coverage: float = 0.75,
    scale_incomplete: bool = False,
) -> pd.DataFrame:
    """Heating degree days HGT and heating days HT (Swiss convention HGT 20/12).

    ``HGT = Σ (room_temp − t_mean)`` over all days with
    ``t_mean <= heating_limit`` (SIA 381/3, MeteoSwiss, BFE).
    ``heating_limit=None`` gives the plain degree days GT of the BFE/Prognos
    GT&S method (all days with ``t_mean < room_temp``).

    Args:
        temperature: Outdoor temperature, see :func:`degree_days`.
        room_temp: Room temperature [°C]. Default 20.
        heating_limit: Heating limit [°C]. Default 12. ``None`` for GT.
        freq: Aggregation, see :func:`aggregate_degree_days` (``None`` for
            daily values).
        method: ``"daily_mean"`` or ``"hourly"``, see :func:`degree_days`.
        min_coverage: Minimum daily sample coverage for sub-daily input.
        scale_incomplete: Scale incomplete periods, see
            :func:`aggregate_degree_days`.

    Returns:
        pd.DataFrame: As :func:`aggregate_degree_days` with ``dd`` renamed
        to ``hgt`` [Kd] and ``dd_days`` to ``ht`` [d].
    """
    daily = degree_days(temperature, room_temp, heating_limit, "heating", method, min_coverage)
    return _dd_wrapper(daily, freq, scale_incomplete, "hgt", "ht")


def accumulated_temperature_difference(
    temperature,
    base: float = HGT_HEATING_LIMIT_CH,
    freq: Optional[str] = "MS",
    method: str = "daily_mean",
    min_coverage: float = 0.75,
    scale_incomplete: bool = False,
) -> pd.DataFrame:
    """Accumulated temperature difference ATD per SIA 380:2015 / ISO 15927-6.

    ``ATD = Σ max(0, base − t_mean)`` with a single base temperature (the
    heating limit). SIA 380 recommends a building-specific base; the
    balance temperature ``tb`` of
    :func:`pyedautils.energy_signature.compute_pes` can be used directly.
    The Stadt Zürich open data uses ``base = 12 °C``.

    Args:
        temperature: Outdoor temperature, see :func:`degree_days`.
        base: Base / heating-limit temperature [°C]. Default 12.
        freq: Aggregation, see :func:`aggregate_degree_days`.
        method: ``"daily_mean"`` or ``"hourly"``.
        min_coverage: Minimum daily sample coverage for sub-daily input.
        scale_incomplete: Scale incomplete periods.

    Returns:
        pd.DataFrame: As :func:`aggregate_degree_days` with ``dd`` renamed
        to ``atd`` [Kd] and ``dd_days`` to ``ht`` [d].
    """
    daily = degree_days(temperature, base, None, "heating", method, min_coverage)
    return _dd_wrapper(daily, freq, scale_incomplete, "atd", "ht")


def cooling_degree_days(
    temperature,
    base: float = CDD_BASE_BFE,
    threshold: Optional[float] = None,
    freq: Optional[str] = "MS",
    method: str = "daily_mean",
    min_coverage: float = 0.75,
    scale_incomplete: bool = False,
) -> pd.DataFrame:
    """Cooling degree days CDD and cooling days CD.

    ``CDD = Σ (t_mean − base)`` over all days with ``t_mean >= threshold``
    (BFE: base and threshold 18.3 °C).

    Args:
        temperature: Outdoor temperature, see :func:`degree_days`.
        base: Base temperature [°C]. Default 18.3.
        threshold: Day-counting threshold [°C]; ``None`` uses ``base``.
        freq: Aggregation, see :func:`aggregate_degree_days`.
        method: ``"daily_mean"`` or ``"hourly"``.
        min_coverage: Minimum daily sample coverage for sub-daily input.
        scale_incomplete: Scale incomplete periods.

    Returns:
        pd.DataFrame: As :func:`aggregate_degree_days` with ``dd`` renamed
        to ``cdd`` [Kd] and ``dd_days`` to ``cd`` [d].
    """
    daily = degree_days(temperature, base, threshold, "cooling", method, min_coverage)
    return _dd_wrapper(daily, freq, scale_incomplete, "cdd", "cd")


# ---------------------------------------------------------------------------
# 2. Reference climatology, correction factor, normalization
# ---------------------------------------------------------------------------
class ReferenceDegreeDays(NamedTuple):
    """Monthly degree-day climatology of a reference period.

    Attributes:
        monthly: Mean degree days per calendar month [Kd], index 1–12.
        monthly_days: Mean degree-day days per calendar month [d] (NaN if
            the source had no day counts).
        annual: Sum of ``monthly`` [Kd].
        annual_days: Sum of ``monthly_days`` [d].
        n_years: Number of years contributing data.
        period: ``(first_year, last_year)`` as strings.
    """

    monthly: pd.Series
    monthly_days: pd.Series
    annual: float
    annual_days: float
    n_years: int
    period: Tuple[str, str]


_DD_COLUMNS = ("dd", "hgt", "atd", "cdd")
_DAYS_COLUMNS = ("dd_days", "ht", "cd")


def _split_monthly_frame(monthly_dd) -> Tuple[pd.Series, Optional[pd.Series], pd.Series]:
    """Return (values, day counts or None, completeness) from monthly input."""
    if isinstance(monthly_dd, pd.Series):
        values = pd.to_numeric(monthly_dd, errors="coerce").astype(float)
        return values, None, pd.Series(1.0, index=values.index)
    if not isinstance(monthly_dd, pd.DataFrame):
        raise ValueError("monthly_dd must be a pandas Series or DataFrame")
    value_col = next((c for c in _DD_COLUMNS if c in monthly_dd.columns), None)
    if value_col is None:
        numeric = monthly_dd.select_dtypes("number").columns
        if len(numeric) == 0:
            raise ValueError("monthly_dd has no numeric degree-day column")
        value_col = numeric[0]
    values = monthly_dd[value_col].astype(float)
    days_col = next((c for c in _DAYS_COLUMNS if c in monthly_dd.columns), None)
    days = monthly_dd[days_col].astype(float) if days_col else None
    if "completeness" in monthly_dd.columns:
        completeness = monthly_dd["completeness"].astype(float)
    else:
        completeness = pd.Series(1.0, index=monthly_dd.index)
    return values, days, completeness


def climatology_from_monthly(
    monthly_dd,
    period: Optional[Tuple[str, str]] = ("1991", "2020"),
    min_completeness: float = 0.9,
) -> ReferenceDegreeDays:
    """Monthly degree-day climatology from a monthly time series.

    Args:
        monthly_dd: Monthly degree days with a DatetimeIndex: the output of
            :func:`aggregate_degree_days` / :func:`heating_degree_days`
            (uses its ``completeness`` and day-count columns) or a plain
            Series, e.g. ``download_degree_days_bfe()["hgt"]``.
        period: ``(first_year, last_year)`` inclusive, e.g. the climate
            normal ``("1991", "2020")``; ``None`` uses all data.
        min_completeness: Months with a lower ``completeness`` are ignored.

    Returns:
        ReferenceDegreeDays: Climatology with 12 monthly means.

    Raises:
        ValueError: If no month of the period has data, or a calendar month
            has no valid value.
    """
    values, days, completeness = _split_monthly_frame(monthly_dd)
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("monthly_dd must have a DatetimeIndex")
    mask = values.notna() & (completeness >= min_completeness)
    if period is not None:
        years = values.index.year
        mask &= (years >= int(period[0])) & (years <= int(period[1]))
    values = values[mask]
    if values.empty:
        raise ValueError("No valid monthly degree days in the reference period")

    months = values.index.month
    monthly = values.groupby(months).mean().reindex(range(1, 13))
    monthly.index.name = "month"
    missing = monthly[monthly.isna()].index.tolist()
    if missing:
        raise ValueError(f"No valid data for calendar months {missing} in the reference period")
    if days is not None:
        monthly_days = days[mask].groupby(months).mean().reindex(range(1, 13))
    else:
        monthly_days = pd.Series(np.nan, index=monthly.index)
    monthly_days.index.name = "month"

    years_present = np.unique(values.index.year)
    n_years = int(len(years_present))
    if n_years < 10:
        logger.warning("Reference climatology based on only %d year(s) of data", n_years)
    return ReferenceDegreeDays(
        monthly=monthly.rename("dd"),
        monthly_days=monthly_days.rename("dd_days"),
        annual=float(monthly.sum()),
        annual_days=float(monthly_days.sum()) if monthly_days.notna().all() else float("nan"),
        n_years=n_years,
        period=(str(years_present.min()), str(years_present.max())),
    )


def reference_degree_days(
    temperature,
    period: Optional[Tuple[str, str]] = ("1991", "2020"),
    base: float = HGT_ROOM_TEMP_CH,
    threshold: Optional[float] = HGT_HEATING_LIMIT_CH,
    mode: str = "heating",
    method: str = "daily_mean",
    min_coverage: float = 0.75,
    min_completeness: float = 0.9,
) -> ReferenceDegreeDays:
    """Degree-day climatology computed from a long outdoor temperature series.

    Convenience wrapper: :func:`degree_days` → monthly
    :func:`aggregate_degree_days` → :func:`climatology_from_monthly`.

    Args:
        temperature: Long outdoor temperature series (e.g. MeteoSwiss daily
            values via :func:`pyedautils.weather.meteo_swiss.download_data`
            with ``granularity="d"``).
        period: Reference years, inclusive. Default climate normal 1991–2020.
        base: Base temperature [°C], see :func:`degree_days`.
        threshold: Day-counting threshold [°C], see :func:`degree_days`.
        mode: ``"heating"`` or ``"cooling"``.
        method: ``"daily_mean"`` or ``"hourly"``.
        min_coverage: Minimum daily sample coverage for sub-daily input.
        min_completeness: Minimum monthly completeness.

    Returns:
        ReferenceDegreeDays: Climatology with 12 monthly means.
    """
    daily = degree_days(temperature, base, threshold, mode, method, min_coverage)
    monthly = aggregate_degree_days(daily, "MS")
    return climatology_from_monthly(monthly, period, min_completeness)


def _is_month_climatology(s: pd.Series) -> bool:
    idx = s.index
    return (
        not isinstance(idx, pd.DatetimeIndex)
        and pd.api.types.is_integer_dtype(idx)
        and idx.min() >= 1 and idx.max() <= 12
    )


def _align_reference(dd_reference, dd_actual):
    """Bring ``dd_reference`` into the shape of ``dd_actual``."""
    if isinstance(dd_reference, ReferenceDegreeDays):
        dd_reference = dd_reference.monthly if isinstance(dd_actual, pd.Series) else dd_reference.annual
    if isinstance(dd_actual, pd.Series) and isinstance(dd_reference, pd.Series):
        if _is_month_climatology(dd_reference):
            if not isinstance(dd_actual.index, pd.DatetimeIndex):
                raise ValueError("dd_actual needs a DatetimeIndex to be matched with a monthly climatology")
            ref = pd.Series(dd_reference.reindex(dd_actual.index.month).to_numpy(), index=dd_actual.index)
        else:
            ref = dd_reference.reindex(dd_actual.index)
        return ref.astype(float)
    if isinstance(dd_reference, pd.Series) and not isinstance(dd_actual, pd.Series):
        raise ValueError("A Series reference requires a Series dd_actual")
    return dd_reference


def climate_correction_factor(dd_actual, dd_reference):
    """Degree-day based climate correction factor ``f = DD_ref / DD_actual``.

    Multiply the weather-dependent consumption of a period by ``f`` to get
    the consumption under reference (normal-year) weather. ``f < 1`` means
    the period was colder than the reference.

    Args:
        dd_actual: Degree days of the analysed period(s): scalar or Series
            with a DatetimeIndex (e.g. monthly or yearly ``hgt``).
        dd_reference: Reference degree days: scalar, Series indexed by
            calendar month 1–12 (matched via ``index.month``), Series with
            the same DatetimeIndex, or a :class:`ReferenceDegreeDays`
            (``monthly`` for Series input, ``annual`` for scalar input).

    Returns:
        float or pd.Series: Correction factor(s); NaN where ``dd_actual``
        is 0 (a warning is logged).

    Raises:
        ValueError: On negative degree days or incompatible inputs.
    """
    ref = _align_reference(dd_reference, dd_actual)
    actual = dd_actual.astype(float) if isinstance(dd_actual, pd.Series) else float(dd_actual)
    if np.nanmin(np.atleast_1d(np.asarray(actual, dtype=float))) < 0:
        raise ValueError("Degree days must not be negative")
    if np.nanmin(np.atleast_1d(np.asarray(ref, dtype=float))) < 0:
        raise ValueError("Reference degree days must not be negative")

    if isinstance(actual, pd.Series):
        zero = actual == 0
        if zero.any():
            logger.warning("%d period(s) with zero degree days; factor set to NaN", int(zero.sum()))
        factor = ref / actual.replace(0, np.nan)
        return factor.rename("factor")
    if actual == 0:
        logger.warning("Zero degree days; correction factor is NaN")
        return float("nan")
    return float(ref) / actual


def normalize_consumption(energy, dd_actual, dd_reference, weather_dependent_share: float = 1.0):
    """Climate-correct a measured consumption to reference weather.

    ``E_norm = E · (1 − s) + E · s · f`` with ``s`` the weather-dependent
    (space-heating) share and ``f`` from :func:`climate_correction_factor`.
    Domestic hot water and process shares are left unchanged, as in SIA 380
    and the cantonal energy statistics.

    The share can be derived from the energy signature, e.g. with
    ``pes = compute_pes(data)`` and ``E_total`` the measured annual energy:
    ``s = 1 − (pes.p_dhw + pes.p_dhwc) · hours / E_total``.

    Args:
        energy: Measured consumption: scalar or Series aligned with
            ``dd_actual`` (same DatetimeIndex).
        dd_actual: Degree days of the same period(s).
        dd_reference: Reference degree days, see
            :func:`climate_correction_factor`.
        weather_dependent_share: Share ``s`` in 0–1. Default 1 (everything
            is corrected).

    Returns:
        float or pd.Series: Climate-corrected consumption.

    Raises:
        ValueError: If ``weather_dependent_share`` is outside 0–1.
    """
    s = float(weather_dependent_share)
    if not 0.0 <= s <= 1.0:
        raise ValueError("weather_dependent_share must be between 0 and 1")
    factor = climate_correction_factor(dd_actual, dd_reference)
    if isinstance(energy, pd.Series):
        energy = energy.astype(float)
        if isinstance(factor, pd.Series):
            factor = factor.reindex(energy.index)
    elif isinstance(factor, pd.Series):
        raise ValueError("Series degree days require a Series energy aligned on the same index")
    result = energy * (1.0 - s) + energy * s * factor
    return result.rename("energy_normalized") if isinstance(result, pd.Series) else float(result)


# ---------------------------------------------------------------------------
# 3. Fixed-base degree-day regression
# ---------------------------------------------------------------------------
class DegreeDayLine(NamedTuple):
    """Result of :func:`fit_degree_day_line`.

    Attributes:
        base: Base temperature [°C] used for the heating degree days.
        intercept: Weather-independent consumption per day [kWh/d].
        slope: Consumption per degree day [kWh/Kd].
        r2: Coefficient of determination.
        cvrmse: Coefficient of variation of the RMSE [%] (ASHRAE G14).
        nmbe: Normalized mean bias error [%] (ASHRAE G14).
        n: Number of periods used.
        freq: ``"D"`` or ``"MS"``.
        data: Per-period DataFrame with ``t_mean``, ``hdd`` (per day),
            ``energy`` (per day), ``n_days`` and ``fitted``.
    """

    base: float
    intercept: float
    slope: float
    r2: float
    cvrmse: float
    nmbe: float
    n: int
    freq: str
    data: pd.DataFrame


def _energy_daily(energy) -> pd.Series:
    """Energy as daily sums (sub-daily input) or on daily timestamps."""
    e = _to_series(energy, "energy")
    interval = _interval(e)
    if interval < _ONE_DAY:
        return e.resample("D").sum(min_count=1)
    if interval > _ONE_DAY:
        raise ValueError("energy must be daily or finer for freq='D'; use freq='MS' for monthly data")
    return e.groupby(e.index.normalize()).sum(min_count=1)


def _energy_monthly(energy) -> pd.Series:
    """Energy as monthly sums; monthly input is snapped to month starts."""
    e = _to_series(energy, "energy")
    if _interval(e) >= pd.Timedelta("28D"):
        starts = _period_start(e.index, "MS", 7)
        return e.groupby(starts).sum(min_count=1)
    return e.resample("MS").sum(min_count=1)


def _check_tz(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> None:
    if (a.tz is None) != (b.tz is None):
        raise ValueError("energy and temperature must both be tz-naive or both tz-aware")


def fit_degree_day_line(
    energy,
    temperature,
    base: float = HGT_HEATING_LIMIT_CH,
    freq: str = "D",
    min_coverage: float = 0.75,
) -> DegreeDayLine:
    """Fit ``E = a + b · HDD(base)`` with a fixed base temperature.

    Daily: ordinary least squares of daily energy on daily heating degree
    days ``max(0, base − t_mean)``. Monthly (``freq="MS"``): weighted
    least squares of energy per day on degree days per day, weighted by
    the number of valid days (CalTRACK billing convention). The base
    temperature is not searched; pass the balance temperature from
    :func:`pyedautils.energy_signature.compute_pes` (``pes.tb``) or a
    known heating limit.

    Args:
        energy: Consumption [kWh] (sub-daily, daily or monthly) as Series
            or ``[timestamp, value]`` DataFrame; sub-daily values are summed
            per day.
        temperature: Outdoor temperature [°C], daily or finer.
        base: Base temperature [°C]. Default 12.
        freq: ``"D"`` or ``"MS"``.
        min_coverage: Minimum daily sample coverage for sub-daily
            temperature.

    Returns:
        DegreeDayLine: Fit parameters, ASHRAE Guideline 14 metrics and the
        per-period data.

    Raises:
        ValueError: On unsupported ``freq``, timezone mismatch or fewer
            than three usable periods.
    """
    if freq not in ("D", "MS"):
        raise ValueError("freq must be 'D' or 'MS'")
    t_daily = _daily_mean(_to_series(temperature, "temperature"), min_coverage)["t_mean"]
    hdd_daily = (base - t_daily).clip(lower=0)

    if freq == "D":
        e = _energy_daily(energy)
        _check_tz(e.index, t_daily.index)
        df = pd.DataFrame({"t_mean": t_daily, "hdd": hdd_daily, "energy": e}).dropna()
        df["n_days"] = 1
    else:
        e = _energy_monthly(energy)
        _check_tz(e.index, t_daily.index)
        monthly_t = t_daily.resample("MS")
        df = pd.DataFrame({
            "t_mean": monthly_t.mean(),
            "hdd": hdd_daily.resample("MS").sum(min_count=1),
            "n_days": monthly_t.count(),
            "energy": e,
        }).dropna()
        df = df[df["n_days"] > 0]
        df["hdd"] = df["hdd"] / df["n_days"]
        df["energy"] = df["energy"] / df["n_days"]

    if len(df) < 3:
        raise ValueError("At least three periods with energy and temperature data are required")

    x = sm.add_constant(df["hdd"].to_numpy(), has_constant="add")
    y = df["energy"].to_numpy()
    if freq == "D":
        fit = sm.OLS(y, x).fit()
    else:
        fit = sm.WLS(y, x, weights=df["n_days"].to_numpy()).fit()

    fitted = fit.predict(x)
    resid = y - fitted
    n, p = len(y), 2
    y_mean = float(np.mean(y))
    cvrmse = 100.0 * float(np.sqrt(np.sum(resid ** 2) / (n - p))) / y_mean if y_mean else float("nan")
    nmbe = 100.0 * float(np.sum(resid)) / ((n - p) * y_mean) if y_mean else float("nan")
    df["fitted"] = fitted

    return DegreeDayLine(
        base=float(base),
        intercept=float(fit.params[0]),
        slope=float(fit.params[1]),
        r2=float(fit.rsquared),
        cvrmse=cvrmse,
        nmbe=nmbe,
        n=int(n),
        freq=freq,
        data=df,
    )


# ---------------------------------------------------------------------------
# 4. GT&S method (Prognos / BFE)
# ---------------------------------------------------------------------------
class GTSResult(NamedTuple):
    """Result of :func:`gts_correction_factor`.

    Attributes:
        ratio: Weather-weighted annual load relative to the normal year
            (the Prognos "Korrekturfaktor"): *divide* the measured
            consumption by it. ``> 1`` means colder and/or less sunny than
            the reference.
        factor: ``1 / ratio``; *multiply* the measured consumption by it
            (same convention as :func:`climate_correction_factor`).
        monthly: DataFrame indexed by month 1–12 with ``share`` (normal-year
            load share), ``d_gt`` and ``d_rad`` (relative deviations of
            degree days and radiation from the reference), ``e_gt`` and
            ``e_rad`` (resulting elasticities), ``effect`` (relative change
            of the monthly demand) and ``contribution`` (weather-weighted
            monthly share; ``ratio`` is their sum).
    """

    ratio: float
    factor: float
    monthly: pd.DataFrame


ElasticityInput = Union[Mapping[int, Tuple[float, float]], Tuple[Sequence[float], Sequence[float]]]


def _monthly_vector(values, name: str) -> np.ndarray:
    """Twelve monthly values as array ordered January–December."""
    if isinstance(values, pd.Series):
        if isinstance(values.index, pd.DatetimeIndex):
            values = values.groupby(values.index.month).sum()
        if _is_month_climatology(values):
            values = values.reindex(range(1, 13))
        arr = values.to_numpy(dtype=float)
    elif isinstance(values, Mapping):
        arr = np.array([values[m] for m in range(1, 13)], dtype=float)
    else:
        arr = np.asarray(list(values), dtype=float)
    if arr.shape != (12,) or np.isnan(arr).any():
        raise ValueError(f"{name} must provide one value for each of the 12 months")
    return arr


def _elasticity_arrays(elasticity: ElasticityInput, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Constants ``c`` and slopes ``a`` of functional elasticities per month."""
    if isinstance(elasticity, Mapping):
        c = np.array([elasticity[m][0] for m in range(1, 13)], dtype=float)
        a = np.array([elasticity[m][1] for m in range(1, 13)], dtype=float)
    else:
        c_seq, a_seq = elasticity
        c = _monthly_vector(c_seq, f"{name} constant")
        a = _monthly_vector(a_seq, f"{name} slope")
    return c, a


def gts_correction_factor(
    gt_actual,
    gt_norm,
    rad_actual,
    rad_norm,
    load_share_norm,
    elasticity_gt: ElasticityInput,
    elasticity_rad: ElasticityInput,
) -> GTSResult:
    """Annual climate correction factor of the BFE/Prognos GT&S method.

    Degree-day and radiation method ("Gradtag- und Strahlungsverfahren",
    Prognos 2003/2015 for the BFE): for every month the relative deviations
    of degree days ``ΔGT`` and solar radiation ``ΔS`` from the reference
    period are weighted with functional elasticities
    ``e = c + a · Δ`` and applied to the normal-year load share:

    ``ratio = Σ_m share_m · (1 + e_GT,m · ΔGT_m + e_S,m · ΔS_m)``

    The elasticities and load shares depend on the building category and
    year (derived from SIA 380/1 model buildings) and are not published in
    full; :data:`GTS_MFH_2010_EXAMPLE` holds the documented example for
    multi-family houses in 2010.

    Args:
        gt_actual: Monthly degree days GT (base 20 °C, no heating limit) of
            the analysed year: 12 values, Series indexed by month or by
            month-start timestamps, or mapping ``{month: value}``.
        gt_norm: Monthly reference degree days (same shape).
        rad_actual: Monthly global radiation of the analysed year
            [MJ/m² or any consistent unit].
        rad_norm: Monthly reference global radiation.
        load_share_norm: Normal-year monthly shares of the annual space
            heating demand (sum 1).
        elasticity_gt: Degree-day elasticities per month as
            ``{month: (c, a)}`` or ``(c_values, a_values)``.
        elasticity_rad: Radiation elasticities per month, same format.

    Returns:
        GTSResult: ``ratio``, ``factor`` and the monthly breakdown.

    Raises:
        ValueError: If any input does not cover 12 months or the load
            shares do not sum to 1 (±1 %).
    """
    gt_a = _monthly_vector(gt_actual, "gt_actual")
    gt_n = _monthly_vector(gt_norm, "gt_norm")
    rad_a = _monthly_vector(rad_actual, "rad_actual")
    rad_n = _monthly_vector(rad_norm, "rad_norm")
    share = _monthly_vector(load_share_norm, "load_share_norm")
    if abs(share.sum() - 1.0) > 0.01:
        raise ValueError(f"load_share_norm must sum to 1, got {share.sum():.3f}")
    if (gt_n <= 0).any() or (rad_n <= 0).any():
        raise ValueError("Reference degree days and radiation must be positive")

    c_gt, a_gt = _elasticity_arrays(elasticity_gt, "elasticity_gt")
    c_rad, a_rad = _elasticity_arrays(elasticity_rad, "elasticity_rad")

    d_gt = (gt_a - gt_n) / gt_n
    d_rad = (rad_a - rad_n) / rad_n
    e_gt = c_gt + a_gt * d_gt
    e_rad = c_rad + a_rad * d_rad
    effect = e_gt * d_gt + e_rad * d_rad
    contribution = share * (1.0 + effect)
    ratio = float(contribution.sum())

    monthly = pd.DataFrame({
        "share": share,
        "d_gt": d_gt,
        "d_rad": d_rad,
        "e_gt": e_gt,
        "e_rad": e_rad,
        "effect": effect,
        "contribution": contribution,
    }, index=pd.RangeIndex(1, 13, name="month"))
    return GTSResult(ratio=ratio, factor=1.0 / ratio, monthly=monthly)


#: Documented GT&S parameters for the category "Wohnen Mehrfamilienhäuser",
#: year 2010 (Prognos 2015, "Witterungsbereinigung auf Basis von Gradtagen
#: und Solarstrahlung", Tabelle 2-9). Reference period 1984–2002, Swiss
#: mean of 53 MeteoSwiss stations; radiation in MJ/m². Elasticities are
#: ``(c, a)`` pairs of ``e = c + a · Δ`` with ``Δ`` as a fraction.
GTS_MFH_2010_EXAMPLE: Dict[str, object] = {
    "gt_norm": [605, 515, 452, 348, 205, 126, 65, 69, 177, 305, 469, 562],
    "rad_norm": [112, 176, 323, 423, 545, 579, 609, 526, 360, 224, 117, 86],
    "load_share_norm": [0.196, 0.153, 0.118, 0.080, 0.023, 0.002,
                        0.001, 0.001, 0.022, 0.074, 0.146, 0.185],
    "elasticity_gt": (
        [1.22, 1.30, 1.43, 1.58, 3.00, 4.89, 4.89, 4.89, 2.86, 1.56, 1.25, 1.20],
        [0.007, 0.054, 0.136, -0.090, 1.373, 7.711, 7.711, 7.711, 1.034, -0.043, 0.047, 0.025],
    ),
    "elasticity_rad": (
        [-0.11, -0.19, -0.31, -0.42, -1.44, -2.80, -2.80, -2.80, -1.26, -0.34, -0.12, -0.08],
        [0.002, 0.008, 0.034, -0.025, 0.660, 4.044, 4.044, 4.044, 0.471, 0.010, 0.003, 0.001],
    ),
}

#: Swiss mean weather of the year 2010 from the same source (Tabelle 2-9):
#: monthly degree days GT and global radiation [MJ/m²]. Together with
#: :data:`GTS_MFH_2010_EXAMPLE` this reproduces the documented ratio 1.063.
GTS_CH_2010_WEATHER: Dict[str, Sequence[float]] = {
    "gt": [668, 535, 476, 298, 249, 101, 39, 90, 197, 340, 442, 642],
    "rad": [110, 185, 350, 534, 464, 587, 688, 495, 422, 238, 131, 95],
}
