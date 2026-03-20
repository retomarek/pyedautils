"""Statistical and exploratory plots."""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_sum_frequency(
    data: pd.DataFrame,
    resolution: str = "hourly",
    year: Optional[int] = None,
    reverse: bool = False,
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: str = "Value",
) -> go.Figure:
    """
    Create a sum frequency (duration curve) plot.

    Values are sorted and plotted against their rank (frequency).
    Useful for temperature duration curves or load duration curves.

    Args:
        data: DataFrame with two columns: timestamp and value.
        resolution: ``"hourly"`` or ``"daily"``. Controls aggregation.
        year: Filter to a specific year. If *None*, use all data.
        reverse: If *True*, the highest value is at x=0
            (classic duration curve / Jahresdauerlinie).
            If *False* (default), lowest value at x=0 (R default).
        title: Plot title. Auto-generated if *None*.
        xlab: X-axis label. Auto-generated if *None*.
        ylab: Y-axis label.

    Returns:
        go.Figure
    """
    import numpy as np

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if year is not None:
        df = df[df["timestamp"].dt.year == year]

    if resolution == "daily":
        df["day"] = df["timestamp"].dt.date
        agg = df.groupby("day")["value"].mean().dropna()
        freq_label = "days"
    else:
        df["hour"] = df["timestamp"].dt.floor("h")
        agg = df.groupby("hour")["value"].mean().dropna()
        freq_label = "hours"

    sorted_vals = np.sort(agg.values)
    if reverse:
        sorted_vals = sorted_vals[::-1]

    freq = np.arange(1, len(sorted_vals) + 1)

    if title is None:
        yr = f" ({year})" if year else ""
        title = f"Sum Frequency Plot — {resolution.title()}{yr}"
    if xlab is None:
        xlab = f"Frequency ({freq_label})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freq, y=sorted_vals,
        mode="markers",
        marker=dict(color="orange", size=3, opacity=0.6),
        name="",
        showlegend=False,
    ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        yaxis_title=ylab,
    )
    return fig


def plot_decomposition(
    data: pd.DataFrame,
    period: Optional[int] = None,
    s_window: Optional[int] = None,
    title: str = "Time Series Decomposition",
    ylab: str = "Value",
    digits: int = 1,
) -> go.Figure:
    """
    Decompose a time series into trend, seasonal, and remainder using STL.

    Works for both long-term (e.g. monthly over years, period=12) and
    short-term data (e.g. 15-min over days, period=96).

    Args:
        data: DataFrame with two columns: timestamp and value.
        period: Seasonal period in number of observations. If *None*,
            auto-detected from the data frequency (e.g. 12 for monthly,
            96 for 15-min, 24 for hourly).
        s_window: Seasonal smoothing window for STL. If *None*, defaults
            to ``2 * period + 1`` (robust default).
        title: Plot title.
        ylab: Y-axis label for the raw data panel.
        digits: Decimal places for rounding. Default 1.

    Returns:
        go.Figure: Plotly figure with 4 subplot rows
        (observed, trend, seasonal, remainder).
    """
    from statsmodels.tsa.seasonal import STL

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Auto-detect period from frequency
    if period is None:
        if len(df) >= 2:
            delta = df["timestamp"].diff().median()
            seconds = delta.total_seconds()
            if seconds <= 960:       # ~15 min -> daily period
                period = int(86400 / seconds)
            elif seconds <= 3660:    # ~1 hour -> daily period
                period = 24
            elif seconds <= 90000:   # ~1 day -> weekly period
                period = 7
            else:                    # monthly or longer -> yearly
                period = 12

    if s_window is None:
        s_window = 2 * period + 1

    ts = df.set_index("timestamp")["value"]

    stl_result = STL(ts, period=period, seasonal=s_window).fit()

    components = {
        "Observed": ts.values,
        "Trend": stl_result.trend,
        "Seasonal": stl_result.seasonal,
        "Remainder": stl_result.resid,
    }

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=list(components.keys()),
    )

    timestamps = df["timestamp"]

    for i, (name, values) in enumerate(components.items(), start=1):
        rounded = [round(v, digits) for v in values]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=rounded,
                mode="lines",
                name=name,
                line=dict(color="black", width=1),
                showlegend=False,
            ),
            row=i, col=1,
        )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        height=700,
    )

    # Y-axis label only on first row
    fig.update_yaxes(title_text=ylab, row=1, col=1)

    return fig


def plot_timeseries(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Time Series",
    ylab: str = "Value",
    range_slider: bool = True,
    height: int = 400,
) -> go.Figure:
    """
    Create an interactive line plot with an optional range slider.

    Supports plotting one or multiple columns from a DataFrame with a
    DatetimeIndex.

    Args:
        data: DataFrame with a DatetimeIndex and one or more value columns.
        columns: Columns to plot. If *None*, plots all columns.
        title: Plot title.
        ylab: Y-axis label.
        range_slider: Show a range slider below the plot. Default True.
        height: Figure height in pixels. Default 400.

    Returns:
        go.Figure: Plotly figure with line traces and optional range slider.
    """
    import plotly.express as px

    df = data.copy()

    if columns is None:
        columns = list(df.columns)

    fig = px.line(df, y=columns, title=title)

    if range_slider:
        fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        height=height,
        yaxis_title=ylab,
        legend_title="Sensor",
    )

    return fig


def plot_distribution(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    nbins: int = 50,
    opacity: float = 0.6,
    title: str = "Distribution",
    xlab: str = "Value",
) -> go.Figure:
    """
    Create overlaid histograms for one or more columns.

    Args:
        data: DataFrame with a DatetimeIndex and one or more value columns.
        columns: Columns to include. If *None*, uses all columns.
        nbins: Number of histogram bins. Default 50.
        opacity: Bar opacity. Default 0.6.
        title: Plot title.
        xlab: X-axis label.

    Returns:
        go.Figure: Plotly histogram figure.
    """
    import plotly.express as px

    df = data.copy()

    if columns is None:
        columns = list(df.columns)

    df_melt = df[columns].melt(var_name="variable", value_name="value")

    fig = px.histogram(
        df_melt,
        x="value",
        color="variable",
        barmode="overlay",
        opacity=opacity,
        nbins=nbins,
        labels={"value": xlab, "variable": "Sensor"},
    )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
    )

    return fig


def plot_boxplot(
    data: pd.DataFrame,
    column: Optional[str] = None,
    groupby: str = "month",
    title: Optional[str] = None,
    ylab: str = "Value",
) -> go.Figure:
    """
    Create a box plot grouped by a time unit extracted from the DatetimeIndex.

    Args:
        data: DataFrame with a DatetimeIndex and one or more value columns.
        column: Column to plot. If *None*, uses the first column.
        groupby: Time grouping — ``"month"``, ``"hour"``, ``"weekday"``,
            or ``"quarter"``. Default ``"month"``.
        title: Plot title. Auto-generated if *None*.
        ylab: Y-axis label.

    Returns:
        go.Figure: Plotly box plot figure.
    """
    import plotly.express as px

    df = data.copy()

    if column is None:
        column = df.columns[0]

    group_map = {
        "month": ("month", df.index.month),
        "hour": ("hour", df.index.hour),
        "weekday": ("weekday", df.index.dayofweek),
        "quarter": ("quarter", df.index.quarter),
    }

    if groupby not in group_map:
        raise ValueError(
            f"groupby must be one of {list(group_map.keys())}, got '{groupby}'"
        )

    label, values = group_map[groupby]
    df[label] = values

    if title is None:
        title = f"{column} by {groupby.title()}"

    fig = px.box(
        df,
        x=label,
        y=column,
        labels={column: ylab, label: groupby.title()},
    )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
    )

    return fig


def plot_outliers(
    data: pd.DataFrame,
    column: Optional[str] = None,
    multiplier: float = 1.5,
    title: Optional[str] = None,
    ylab: str = "Value",
    height: int = 400,
) -> go.Figure:
    """
    Visualize IQR-based outliers on a time series plot.

    Plots the time series as a line with outlier points highlighted in red
    and dashed horizontal lines at the IQR fences.

    Args:
        data: DataFrame with a DatetimeIndex and one or more value columns.
        column: Column to analyse. If *None*, uses the first column.
        multiplier: IQR multiplier for the fence. Default 1.5.
        title: Plot title. Auto-generated if *None*.
        ylab: Y-axis label.
        height: Figure height in pixels. Default 400.

    Returns:
        go.Figure: Plotly figure with time series, outlier markers, and fences.
    """
    from pyedautils.data_quality import calc_outliers

    if column is None:
        column = data.columns[0]

    result = calc_outliers(data, column=column, multiplier=multiplier)

    if title is None:
        title = (
            f"{column} — Outlier Detection (IQR×{multiplier})"
            f"  [{result['count']} outliers, {result['percentage']}%]"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[column],
        mode="lines",
        name="Normal",
        line=dict(width=0.5),
    ))

    outliers = result["outliers"]
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers.index,
            y=outliers[column],
            mode="markers",
            name="Outlier",
            marker=dict(color="red", size=4),
        ))

    fig.add_hline(
        y=result["upper"],
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Upper: {result['upper']:.1f}",
    )
    fig.add_hline(
        y=result["lower"],
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Lower: {result['lower']:.1f}",
    )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        yaxis_title=ylab,
        height=height,
    )

    return fig


def plot_correlation(
    data: pd.DataFrame,
    color_scale: str = "RdBu_r",
    text_format: str = ".2f",
    title: str = "Correlation Matrix",
    height: int = 500,
    width: int = 600,
) -> go.Figure:
    """
    Create a correlation matrix heatmap.

    Args:
        data: DataFrame with numeric columns (DatetimeIndex is ignored).
        color_scale: Plotly colorscale name. Default ``"RdBu_r"``.
        text_format: Number format for cell annotations. Default ``".2f"``.
        title: Plot title.
        height: Figure height in pixels. Default 500.
        width: Figure width in pixels. Default 600.

    Returns:
        go.Figure: Plotly heatmap figure of the correlation matrix.
    """
    import plotly.express as px

    corr = data.corr()

    fig = px.imshow(
        corr,
        text_auto=text_format,
        color_continuous_scale=color_scale,
        zmin=-1,
        zmax=1,
    )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        height=height,
        width=width,
    )

    return fig


def plot_scatter(
    data: pd.DataFrame,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    opacity: float = 0.2,
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
) -> go.Figure:
    """
    Create a scatter plot of two variables.

    Low default opacity helps reveal density in large datasets.

    Args:
        data: DataFrame with a DatetimeIndex and at least two value columns.
        x_column: Column for the x-axis. If *None*, uses the first column.
        y_column: Column for the y-axis. If *None*, uses the second column.
        opacity: Marker opacity. Default 0.2.
        title: Plot title. Auto-generated if *None*.
        xlab: X-axis label. Defaults to the column name.
        ylab: Y-axis label. Defaults to the column name.

    Returns:
        go.Figure: Plotly scatter plot figure.
    """
    import plotly.express as px

    cols = list(data.columns)
    if x_column is None:
        x_column = cols[0]
    if y_column is None:
        y_column = cols[1]
    if xlab is None:
        xlab = x_column
    if ylab is None:
        ylab = y_column
    if title is None:
        title = f"{x_column} vs. {y_column}"

    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        opacity=opacity,
        labels={x_column: xlab, y_column: ylab},
    )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
    )

    return fig


def plot_autocorrelation(
    data: pd.DataFrame,
    column: Optional[str] = None,
    lags: int = 168,
    title: Optional[str] = None,
    height: int = 300,
) -> go.Figure:
    """
    Create an autocorrelation plot with a confidence band using Plotly.

    Args:
        data: DataFrame with a DatetimeIndex and one or more value columns.
        column: Column to analyse. If *None*, uses the first column.
        lags: Number of lags to compute. Default 168 (one week at hourly).
        title: Plot title. Auto-generated if *None*.
        height: Figure height in pixels. Default 300.

    Returns:
        go.Figure: Plotly bar chart of autocorrelation values with a
        95% confidence band.
    """
    import numpy as np
    from statsmodels.tsa.stattools import acf

    df = data.copy()

    if column is None:
        column = df.columns[0]

    series = df[column].interpolate(limit=6).dropna()
    acf_values, confint = acf(series, nlags=lags, alpha=0.05)

    if title is None:
        title = f"Autocorrelation — {column}"

    lag_axis = np.arange(len(acf_values))
    ci_lower = confint[:, 0] - acf_values
    ci_upper = confint[:, 1] - acf_values

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([lag_axis, lag_axis[::-1]]),
        y=np.concatenate([ci_upper, ci_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(135,206,250,0.3)",
        line=dict(color="rgba(135,206,250,0)"),
        name="95% CI",
        showlegend=False,
    ))

    # ACF bars
    fig.add_trace(go.Bar(
        x=lag_axis,
        y=acf_values,
        marker_color="steelblue",
        width=1.0,
        name="ACF",
        showlegend=False,
    ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        height=height,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        yaxis_range=[-1, 1],
    )

    return fig
