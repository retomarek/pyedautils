"""Seasonal and temporal pattern plots."""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from pyedautils.plots._constants import DEFAULT_SEASON_COLORS


def plot_seasonal_overlapping(
    data: pd.DataFrame,
    title: str = "Seasonal Plot — Overlapping",
    ylab: str = "Value",
) -> go.Figure:
    """
    Seasonal plot with one line per year overlaid on the same x-axis (month).

    Args:
        data: DataFrame with columns ``[timestamp, value]``.
            Monthly data; value should be normalised (e.g. kWh/day).
        title: Plot title.
        ylab: Y-axis label.

    Returns:
        go.Figure
    """
    import plotly.express as px

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    years = sorted(df["year"].unique())
    colors = px.colors.sample_colorscale(
        "Viridis", [i / max(len(years) - 1, 1) for i in range(len(years))]
    )
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure()
    for i, year in enumerate(years):
        subset = df[df["year"] == year].sort_values("month")
        fig.add_trace(go.Scatter(
            x=subset["month"], y=subset["value"],
            mode="lines+markers", name=str(year),
            line=dict(color=colors[i]),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis=dict(
            tickvals=list(range(1, 13)), ticktext=month_names,
            title="Month",
        ),
        yaxis_title=ylab,
    )
    return fig


def plot_seasonal_miniplots(
    data: pd.DataFrame,
    title: str = "Seasonal Miniplots",
    ylab: str = "Value",
) -> go.Figure:
    """
    Seasonal subseries plot — one mini panel per month showing values across years.

    A horizontal blue line shows the per-month mean.

    Args:
        data: DataFrame with columns ``[timestamp, value]``.
        title: Plot title.
        ylab: Y-axis label.

    Returns:
        go.Figure
    """
    import plotly.express as px
    from plotly.subplots import make_subplots

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    years = sorted(df["year"].unique())
    colors = px.colors.sample_colorscale(
        "Viridis", [i / max(len(years) - 1, 1) for i in range(len(years))]
    )
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = make_subplots(
        rows=1, cols=12,
        shared_yaxes=True,
        subplot_titles=month_names,
        horizontal_spacing=0.02,
    )

    for month in range(1, 13):
        subset = df[df["month"] == month].sort_values("year")
        mean_val = subset["value"].mean()

        for i, year in enumerate(years):
            row = subset[subset["year"] == year]
            if row.empty:
                continue
            fig.add_trace(go.Scatter(
                x=[year], y=row["value"].values,
                mode="markers", name=str(year),
                marker=dict(color=colors[i], size=6),
                showlegend=(month == 1),
                legendgroup=str(year),
            ), row=1, col=month)

        # Connect points with lines
        fig.add_trace(go.Scatter(
            x=subset["year"], y=subset["value"],
            mode="lines",
            line=dict(color="grey", width=1),
            showlegend=False,
        ), row=1, col=month)

        # Mean line
        fig.add_trace(go.Scatter(
            x=[years[0], years[-1]], y=[mean_val, mean_val],
            mode="lines",
            line=dict(color="blue", width=2),
            showlegend=False,
        ), row=1, col=month)

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        height=400,
        yaxis_title=ylab,
    )
    return fig


def plot_seasonal_before_after(
    data: pd.DataFrame,
    date_optimization: str,
    title: str = "Seasonal — Before/After Optimization",
    ylab: str = "Value",
    confidence: float = 95.0,
) -> go.Figure:
    """
    Seasonal plot highlighting before/after an optimization date.

    Years before the optimization are shown in grey with a confidence
    band; years after are shown in colour.

    Args:
        data: DataFrame with columns ``[timestamp, value]``.
        date_optimization: Date string (e.g. ``"2017-09-01"``).
            Years starting after this date are "after".
        title: Plot title.
        ylab: Y-axis label.
        confidence: Confidence level for the "before" band (0-100).

    Returns:
        go.Figure
    """
    import plotly.express as px

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    opt_date = pd.to_datetime(date_optimization)
    before = df[df["timestamp"] < opt_date]
    after = df[df["timestamp"] >= opt_date]

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure()

    # Before years in grey
    for year in sorted(before["year"].unique()):
        subset = before[before["year"] == year].sort_values("month")
        fig.add_trace(go.Scatter(
            x=subset["month"], y=subset["value"],
            mode="lines", name="Before",
            line=dict(color="lightgrey", width=1),
            legendgroup="before",
            showlegend=bool(year == before["year"].min()),
        ))

    # Confidence band for before period
    q_lo = (100 - confidence) / 200
    q_hi = 1 - q_lo
    monthly_stats = before.groupby("month")["value"].agg(
        ["mean", lambda x: x.quantile(q_lo), lambda x: x.quantile(q_hi)]
    )
    monthly_stats.columns = ["mean", "lo", "hi"]
    months = list(range(1, 13))

    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=list(monthly_stats.reindex(months)["hi"])
        + list(monthly_stats.reindex(months)["lo"][::-1]),
        fill="toself", fillcolor="rgba(150,150,150,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"Before ({confidence:.0f}% CI)",
        showlegend=True,
    ))

    # Mean line for before
    fig.add_trace(go.Scatter(
        x=months,
        y=list(monthly_stats.reindex(months)["mean"]),
        mode="lines",
        line=dict(color="grey", width=2, dash="dash"),
        name="Before (mean)",
    ))

    # After years in colour
    after_years = sorted(after["year"].unique())
    colors = px.colors.sample_colorscale(
        "Viridis", [i / max(len(after_years) - 1, 1)
                    for i in range(len(after_years))]
    )
    for i, year in enumerate(after_years):
        subset = after[after["year"] == year].sort_values("month")
        fig.add_trace(go.Scatter(
            x=subset["month"], y=subset["value"],
            mode="lines+markers", name=str(year),
            line=dict(color=colors[i], width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis=dict(
            tickvals=list(range(1, 13)), ticktext=month_names,
            title="Month",
        ),
        yaxis_title=ylab,
    )
    return fig


def plot_seasonal_polar(
    data: pd.DataFrame,
    title: str = "Seasonal Plot — Polar",
    ylab: str = "Value",
) -> go.Figure:
    """
    Seasonal plot in polar coordinates — one line per year.

    Args:
        data: DataFrame with columns ``[timestamp, value]``.
        title: Plot title.
        ylab: Radial axis label.

    Returns:
        go.Figure
    """
    import plotly.express as px

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    years = sorted(df["year"].unique())
    colors = px.colors.sample_colorscale(
        "Viridis", [i / max(len(years) - 1, 1) for i in range(len(years))]
    )
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure()
    for i, year in enumerate(years):
        subset = df[df["year"] == year].sort_values("month")
        # Close the loop
        theta = [month_names[m - 1] for m in subset["month"]]
        r = list(subset["value"])
        if len(theta) == 12:
            theta.append(theta[0])
            r.append(r[0])
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta,
            mode="lines", name=str(year),
            line=dict(color=colors[i]),
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        polar=dict(radialaxis=dict(title=ylab)),
    )
    return fig


def plot_density_seasons(
    data: pd.DataFrame,
    title: str = "Density Plot by Season",
    xlab: str = "Value",
    ylab: str = "Density",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Create a kernel density plot of a value column, one curve per season.

    Args:
        data: DataFrame with two columns: timestamp and value.
        title: Plot title.
        xlab: X-axis label.
        ylab: Y-axis label.
        colors: Season color overrides keyed by season name.
            Default uses ``DEFAULT_SEASON_COLORS``.

    Returns:
        go.Figure: Plotly figure with one density trace per season.
    """
    import numpy as np
    from pyedautils.season import get_season

    c = {**DEFAULT_SEASON_COLORS, **(colors or {})}

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["season"] = get_season(df["timestamp"])

    fig = go.Figure()

    for season in ["Spring", "Summer", "Fall", "Winter"]:
        subset = df[df["season"] == season]["value"].dropna()
        if subset.empty:
            continue

        # Kernel density estimation using numpy histogram
        values = subset.values
        n_bins = 200
        x_min, x_max = values.min(), values.max()
        padding = (x_max - x_min) * 0.15
        x_grid = np.linspace(x_min - padding, x_max + padding, n_bins)

        # Gaussian KDE via scipy-free approach: histogram + smoothing
        from numpy import exp, sqrt, pi
        bandwidth = 1.06 * values.std() * len(values) ** (-1 / 5)
        density = np.zeros_like(x_grid)
        for v in values:
            density += exp(-0.5 * ((x_grid - v) / bandwidth) ** 2)
        density /= (len(values) * bandwidth * sqrt(2 * pi))

        fig.add_trace(go.Scatter(
            x=x_grid,
            y=density,
            mode="lines",
            name=season,
            line=dict(color=c.get(season, "#999"), width=2),
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        yaxis_title=ylab,
    )

    return fig
