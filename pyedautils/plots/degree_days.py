"""Degree-day and climate-correction plots."""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from pyedautils.plots._constants import DEFAULT_SEASON_COLORS

_DEFAULT_BAR_COLORS = {"actual": "#365c8d", "reference": "#b0b0b0"}
_DD_COLUMNS = ("hgt", "atd", "cdd", "dd")
_DAYS_COLUMNS = ("ht", "cd", "dd_days")


def _actual_frame(actual, value_col: Optional[str]) -> pd.DataFrame:
    """Degree days as DataFrame with ``value`` (+ optional ``days``, ``completeness``)."""
    if isinstance(actual, pd.Series):
        return pd.DataFrame({"value": actual.astype(float)})
    if not isinstance(actual, pd.DataFrame):
        raise ValueError("actual must be a pandas Series or DataFrame")
    col = value_col or next((c for c in _DD_COLUMNS if c in actual.columns), None)
    if col is None:
        raise ValueError("actual has no degree-day column; pass value_col")
    df = pd.DataFrame({"value": actual[col].astype(float)})
    days_col = next((c for c in _DAYS_COLUMNS if c in actual.columns), None)
    if days_col:
        df["days"] = actual[days_col]
    if "completeness" in actual.columns:
        df["completeness"] = actual["completeness"]
    return df


def _reference_series(reference, index: pd.DatetimeIndex) -> Optional[pd.Series]:
    """Reference degree days aligned with ``index``."""
    if reference is None:
        return None
    monthly = getattr(reference, "monthly", reference)
    if not isinstance(monthly, pd.Series):
        raise ValueError("reference must be a Series or ReferenceDegreeDays")
    if isinstance(monthly.index, pd.DatetimeIndex):
        return monthly.reindex(index).astype(float)
    return pd.Series(monthly.reindex(index.month).to_numpy(dtype=float), index=index)


def plot_degree_days(
    actual,
    reference=None,
    value_col: Optional[str] = None,
    title: str = "Heating Degree Days",
    xlab: str = "Month",
    ylab: str = "HGT [Kd]",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Grouped bar chart of degree days per period, optionally against a reference.

    Args:
        actual: Monthly (or yearly) degree days: output of
            :func:`pyedautils.degree_days.heating_degree_days` /
            ``accumulated_temperature_difference`` / ``cooling_degree_days``
            (``freq="MS"`` or ``"YS"``) or a Series with a DatetimeIndex.
        reference: Reference degree days: a
            :class:`pyedautils.degree_days.ReferenceDegreeDays`, a Series
            indexed by calendar month 1–12 (matched by month), or a Series
            on the same DatetimeIndex. Default None (no reference bars).
        value_col: Column of ``actual`` to plot; auto-detected
            (``hgt``/``atd``/``cdd``/``dd``) if None.
        title: Plot title.
        xlab: X-axis label.
        ylab: Y-axis label.
        colors: Color overrides keyed ``"actual"`` and ``"reference"``.

    Returns:
        go.Figure: Plotly figure with one bar trace per series.
    """
    c = {**_DEFAULT_BAR_COLORS, **(colors or {})}
    df = _actual_frame(actual, value_col)
    ref = _reference_series(reference, df.index)
    labels = df.index.strftime("%Y-%m") if isinstance(df.index, pd.DatetimeIndex) else df.index.astype(str)

    hover_extra = ""
    customdata = None
    if "days" in df.columns or "completeness" in df.columns:
        cols = [col for col in ("days", "completeness") if col in df.columns]
        customdata = df[cols].to_numpy()
        for i, col in enumerate(cols):
            fmt = ".0f" if col == "days" else ".0%"
            hover_extra += f"{col.capitalize()}: %{{customdata[{i}]:{fmt}}}<br>"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=df["value"], name="Actual",
        marker_color=c["actual"], customdata=customdata,
        hovertemplate=("%{x}<br>" + f"{ylab}: " + "%{y:.0f}<br>" + hover_extra + "<extra>Actual</extra>"),
    ))
    if ref is not None:
        fig.add_trace(go.Bar(
            x=labels, y=ref, name="Reference",
            marker_color=c["reference"],
            hovertemplate=("%{x}<br>" + f"{ylab}: " + "%{y:.0f}<extra>Reference</extra>"),
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        yaxis_title=ylab,
        barmode="group",
    )
    return fig


def plot_degree_day_regression(
    energy,
    temperature,
    base: float = 12.0,
    freq: str = "D",
    title: str = "Degree-Day Regression",
    xlab: str = "Outside Temperature [°C]",
    ylab: str = "Energy [kWh/d]",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Scatter of energy per day vs outside temperature with a fixed-base
    degree-day regression line.

    Fits :func:`pyedautils.degree_days.fit_degree_day_line` and draws the
    heating line (``intercept + slope · (base − T)`` below ``base``) and
    the flat base load above it. Points are colored by season (daily) or
    shown as a single series (monthly).

    Args:
        energy: Consumption [kWh], see ``fit_degree_day_line``.
        temperature: Outdoor temperature [°C], daily or finer.
        base: Base temperature [°C] of the heating degree days. Default 12;
            e.g. ``compute_pes(...).tb``.
        freq: ``"D"`` (daily points) or ``"MS"`` (monthly means per day).
        title: Plot title.
        xlab: X-axis label.
        ylab: Y-axis label.
        colors: Season color overrides keyed by season name (daily) or
            ``"points"`` (monthly).

    Returns:
        go.Figure: Plotly figure with scatter, regression line and
        annotation of base temperature, slope and fit metrics.
    """
    from pyedautils.degree_days import fit_degree_day_line

    c = {**DEFAULT_SEASON_COLORS, "points": "#365c8d", **(colors or {})}
    line = fit_degree_day_line(energy, temperature, base=base, freq=freq)
    pts = line.data.copy()

    fig = go.Figure()
    if freq == "D":
        from pyedautils.data_prep.season import get_season

        pts["season"] = pd.Series(get_season(pd.Series(pts.index))).to_numpy()
        for season in DEFAULT_SEASON_COLORS:
            subset = pts[pts["season"] == season]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["t_mean"], y=subset["energy"], mode="markers", name=season,
                marker=dict(color=c.get(season, "#999"), size=6, opacity=0.7),
                customdata=subset.index.strftime("%Y-%m-%d"),
                hovertemplate=("Date: %{customdata}<br>" + f"{xlab}: " + "%{x:.1f}<br>"
                               + f"{ylab}: " + "%{y:.1f}<br>" + f"Season: {season}<extra></extra>"),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pts["t_mean"], y=pts["energy"], mode="markers", name="Months",
            marker=dict(color=c["points"], size=8, opacity=0.8),
            customdata=pts.index.strftime("%Y-%m"),
            hovertemplate=("Month: %{customdata}<br>" + f"{xlab}: " + "%{x:.1f}<br>"
                           + f"{ylab}: " + "%{y:.1f}<extra></extra>"),
        ))

    t_min, t_max = float(pts["t_mean"].min()), float(pts["t_mean"].max())
    x_line = [t_min, line.base, max(t_max, line.base)]
    y_line = [line.intercept + line.slope * max(0.0, line.base - t_min), line.intercept, line.intercept]
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name="Regression",
        line=dict(color="red", width=2),
    ))
    fig.add_annotation(
        x=line.base, y=line.intercept,
        text=f"T<sub>b</sub> = {line.base:.1f} °C",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.98, xanchor="right", showarrow=False, align="left",
        text=(f"slope = {line.slope:.2f} kWh/Kd<br>intercept = {line.intercept:.1f} kWh/d<br>"
              f"R² = {line.r2:.3f}, CV(RMSE) = {line.cvrmse:.1f} %"),
        bgcolor="rgba(255,255,255,0.8)",
    )

    y_vals = list(pts["energy"]) + y_line
    x_pad = (max(x_line) - t_min) * 0.05 or 1.0
    y_pad = (max(y_vals) - min(y_vals)) * 0.05 or 1.0
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        yaxis_title=ylab,
        xaxis_range=[t_min - x_pad, max(x_line) + x_pad],
        yaxis_range=[min(y_vals) - y_pad, max(y_vals) + y_pad],
    )
    return fig
