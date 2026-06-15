"""Plots for heating/cooling gradients (see :mod:`pyedautils.data_prep.gradients`)."""

from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go


def plot_gradient_boxplots(
    gradients: pd.DataFrame,
    groupby: str = "season",
    direction_labels: Tuple[str, str] = ("heating", "cooling"),
    title: str = "Temperature Gradients",
    ylab: str = "Gradient [K/h]",
    colors: Tuple[str, str] = ("#E74C3C", "#3498DB"),
    season_order: Optional[list] = None,
) -> go.Figure:
    """Grouped boxplots of gradients, split by direction.

    Draws one box per group and direction (e.g. heating vs cooling per
    season or per quarter), so heating and cooling spreads are directly
    comparable.

    Args:
        gradients: Output of
            :func:`pyedautils.data_prep.gradients.compute_gradients`
            (columns ``[timestamp, gradient, direction, season]``).
        groupby: Grouping on the x-axis. ``"season"`` uses the ``season``
            column; ``"quarter"``, ``"month"``, ``"weekday"`` and
            ``"hour"`` are derived from ``timestamp``.
        direction_labels: ``(rising, falling)`` labels matching
            :func:`compute_gradients`.
        title: Plot title.
        ylab: Y-axis label.
        colors: ``(rising_color, falling_color)``.
        season_order: Optional ordered season labels for the x-axis when
            ``groupby="season"``.

    Returns:
        go.Figure: Plotly grouped boxplot figure.
    """
    df = gradients.copy()
    if df.empty:
        return go.Figure()

    if groupby == "season":
        df["_group"] = df["season"]
        category_order = season_order
    else:
        ts = pd.to_datetime(df["timestamp"])
        if groupby == "quarter":
            df["_group"] = "Q" + ts.dt.quarter.astype(str)
        elif groupby == "month":
            df["_group"] = ts.dt.month
        elif groupby == "weekday":
            df["_group"] = ts.dt.day_name()
        elif groupby == "hour":
            df["_group"] = ts.dt.hour
        else:
            raise ValueError(f"Unknown groupby: {groupby!r}")
        category_order = None

    fig = go.Figure()
    for label, color in zip(direction_labels, colors):
        sub = df[df["direction"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            x=sub["_group"], y=sub["gradient"],
            name=label, marker_color=color, boxmean=True,
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        boxmode="group",
        yaxis_title=ylab,
    )
    if category_order is not None:
        fig.update_xaxes(categoryorder="array", categoryarray=category_order)
    return fig
