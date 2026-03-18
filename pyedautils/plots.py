"""Plot functions for energy data analysis and visualization."""

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyedautils._plot_utils import (
    DEFAULT_SEASONS,
    DEFAULT_WEEKDAYS,
    DEFAULT_XTICKS,
    add_confidence_band,
    create_seasonal_weekday_subplots,
    prepare_hourly_seasonal_data,
    style_subplot_axes,
)


def plot_daily_profiles_overview(
    data: pd.DataFrame,
    title: str = "Daily Profiles Overview by Weekday and Season",
    ylab: str = "Value",
    confidence: float = 95.0,
    colors: Optional[Dict[str, str]] = None,
    seasons: Optional[List[str]] = None,
    weekdays: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create a 4x7 subplot grid showing daily profiles by season and weekday.

    Each subplot shows the median line with a confidence band based on quantiles.
    Rows represent seasons (Spring, Summer, Fall, Winter) and columns represent
    weekdays (Monday through Sunday).

    Args:
        data: DataFrame with two columns: timestamp and value.
        title: Plot title.
        ylab: Y-axis label (shown on first column).
        confidence: Confidence level for quantile bands (0-100). Default 95.
        colors: Optional color overrides. Keys: "median", "bounds", "fill".
        seasons: Custom season names in row order. Default: Spring, Summer, Fall, Winter.
        weekdays: Custom weekday names in column order. Default: Monday through Sunday.

    Returns:
        go.Figure: Plotly figure with the subplot grid.
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS
    if weekdays is None:
        weekdays = DEFAULT_WEEKDAYS

    df = prepare_hourly_seasonal_data(data, confidence=confidence, seasons=seasons)
    fig, seasons, weekdays = create_seasonal_weekday_subplots(
        title=title, seasons=seasons, weekdays=weekdays,
    )

    for i, season in enumerate(seasons):
        for j, weekday in enumerate(weekdays):
            data_subset = df[(df["season"] == season) & (df["weekday"] == weekday)]
            row, col = i + 1, j + 1

            add_confidence_band(fig, data_subset, row=row, col=col, colors=colors)
            style_subplot_axes(fig, row=row, col=col)

            if j == 0:
                fig.update_yaxes(title_text=season, row=row, col=col)

    return fig


def plot_daily_profiles_decomposed(
    data: pd.DataFrame,
    loc_time_zone: str = "UTC",
    title: str = "Daily Profiles - Decomposed",
    ylab: str = "delta Energy Consumption",
    k: int = 672,
    digits: int = 1,
) -> go.Figure:
    """
    Create a decomposed daily profile plot showing the seasonal component per weekday.

    The time series is detrended using a rolling mean, then the seasonal (weekly)
    pattern is extracted by averaging across all weeks. Each weekday is shown as
    a separate line.

    Args:
        data: DataFrame with two columns: timestamp and value.
        loc_time_zone: Timezone for localization (e.g. "Europe/Zurich"). Default "UTC".
        title: Plot title.
        ylab: Y-axis label.
        k: Rolling window size for trend calculation. Default 672 (4 weeks at hourly).
        digits: Number of decimal places for rounding. Default 1.

    Returns:
        go.Figure: Plotly figure with one line per weekday.
    """
    df = data.copy()
    df.columns = ["timestamp", "value"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    df = df.tz_localize(tz="UTC")
    df = df.tz_convert(tz=loc_time_zone)

    # Detrend
    roll_mean = df["value"].rolling(window=k, min_periods=1).mean()
    df["trend"] = roll_mean
    df["valueDetrended"] = df["value"] - roll_mean

    # Seasonal component
    df["weekday"] = df.index.dayofweek
    df["dayhour"] = df.index.hour
    df["dayminute"] = df.index.minute
    seasonal = df.groupby(["weekday", "dayhour", "dayminute"])["valueDetrended"].mean().reset_index()

    final_data = seasonal.copy()
    min_value = final_data["valueDetrended"].min()
    final_data["value"] = final_data["valueDetrended"] - min_value

    # Prepare plot data
    df_plot = final_data.copy()
    df_plot["time"] = df_plot["dayhour"] + df_plot["dayminute"] / 60
    df_plot["value"] = df_plot["value"].round(digits)
    df_plot = df_plot.sort_values(by=["weekday", "time"])

    weekdays = DEFAULT_WEEKDAYS
    df_plot["weekday"] = df_plot["weekday"].map(lambda x: weekdays[x])

    fig = make_subplots(rows=1, cols=1)

    for day in weekdays:
        subset = df_plot[df_plot["weekday"] == day]
        fig.add_trace(go.Scatter(
            x=subset["time"], y=subset["value"],
            mode="lines", name=str(day),
        ), row=1, col=1)

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Hour of Day",
        yaxis_title=ylab,
    )
    fig.update_xaxes(tickvals=DEFAULT_XTICKS)

    return fig
