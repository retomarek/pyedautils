"""Shared utilities for plot functions."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Default layout constants
DEFAULT_SEASONS = ["Spring", "Summer", "Fall", "Winter"]
DEFAULT_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DEFAULT_XTICKS = list(range(0, 24, 6))

DEFAULT_COLORS = {
    "median": "black",
    "bounds": "darkgrey",
    "fill": "lightgrey",
    "grid": "darkgrey",
}


def prepare_hourly_seasonal_data(
    data: pd.DataFrame,
    confidence: float = 95.0,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare hourly data grouped by season and weekday with median and quantile bands.

    Args:
        data: DataFrame with columns ["timestamp", "value"].
        confidence: Confidence level for quantile bands (0-100).
        seasons: List of season names in display order.

    Returns:
        Aggregated DataFrame with columns: season, weekday, dayhour,
        valueMedian, valueUpper, valueLower.
    """
    from pyedautils.data_prep.season import get_season

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("h")

    df_h = df.groupby("hour").agg({"value": "sum"}).reset_index()
    df_h["weekday"] = df_h["hour"].dt.day_name()
    df_h["dayhour"] = df_h["hour"].dt.hour
    df_h["season"] = get_season(df_h["hour"])

    # Rename internal English season names to custom labels
    if seasons is not None:
        rename_map = dict(zip(DEFAULT_SEASONS, seasons))
        df_h["season"] = df_h["season"].map(rename_map).fillna(df_h["season"])

    df_agg = df_h.groupby(["season", "weekday", "dayhour"]).agg(
        valueMedian=("value", lambda x: x.quantile(0.5)),
        valueUpper=("value", lambda x: x.quantile(confidence / 100)),
        valueLower=("value", lambda x: x.quantile((100 - confidence) / 100)),
    ).reset_index()

    return df_agg


def create_seasonal_weekday_subplots(
    title: str = "",
    seasons: Optional[List[str]] = None,
    weekdays: Optional[List[str]] = None,
) -> Tuple[go.Figure, List[str], List[str]]:
    """
    Create a 4x7 subplot grid (seasons x weekdays).

    Args:
        title: Plot title.
        seasons: List of season names for rows.
        weekdays: List of weekday names for columns.

    Returns:
        Tuple of (figure, seasons list, weekdays list).
    """
    fig = make_subplots(
        rows=len(seasons), cols=len(weekdays),
        subplot_titles=weekdays,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.025, horizontal_spacing=0.025,
    )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        template="plotly_white",
        title_x=0.5,
        title_font=dict(size=20),
    )

    return fig, seasons, weekdays


def add_confidence_band(
    fig: go.Figure,
    data_subset: pd.DataFrame,
    row: int,
    col: int,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Add median line with confidence band traces to a subplot.

    Args:
        fig: Plotly figure.
        data_subset: DataFrame with dayhour, valueMedian, valueUpper, valueLower.
        row: Subplot row (1-indexed).
        col: Subplot column (1-indexed).
        colors: Color overrides (keys: median, bounds, fill).
    """
    c = {**DEFAULT_COLORS, **(colors or {})}

    # Fill area
    fig.add_trace(go.Scatter(
        x=np.concatenate([data_subset["dayhour"], data_subset["dayhour"][::-1]]),
        y=np.concatenate([data_subset["valueLower"], data_subset["valueUpper"][::-1]]),
        fill="toself", opacity=0.3, fillcolor=c["fill"],
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
    ), row=row, col=col)

    # Bounds and median
    fig.add_trace(go.Scatter(
        x=data_subset["dayhour"], y=data_subset["valueUpper"],
        mode="lines", line=dict(color=c["bounds"]),
        showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=data_subset["dayhour"], y=data_subset["valueMedian"],
        mode="lines", line=dict(color=c["median"]),
        showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=data_subset["dayhour"], y=data_subset["valueLower"],
        mode="lines", line=dict(color=c["bounds"]),
        showlegend=False,
    ), row=row, col=col)


def style_subplot_axes(
    fig: go.Figure,
    row: int,
    col: int,
    xticks: Optional[List[int]] = None,
    grid_color: str = "darkgrey",
) -> None:
    """Apply consistent axis styling to a subplot."""
    if xticks is None:
        xticks = DEFAULT_XTICKS

    fig.update_xaxes(tickvals=xticks, row=row, col=col)
    fig.update_xaxes(showline=True, linewidth=1, linecolor=grid_color, mirror=True, row=row, col=col)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=grid_color, mirror=True, row=row, col=col)
