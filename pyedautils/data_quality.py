"""Functions for detecting and visualizing gaps in time series data."""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go


def calc_gap_duration(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Calculate the time difference between consecutive index entries.

    Computes the gap duration in seconds between rows using the
    DatetimeIndex, plus a rolling median for comparison.

    Args:
        df: DataFrame with a DatetimeIndex.
        window: Rolling median window size. Default 20.

    Returns:
        DataFrame with columns ``gapDuration`` (seconds between rows)
        and ``gapDurationRollMedian`` (rolling median of gap durations).
    """
    gap = df.index.to_series().diff().dt.total_seconds()
    roll_median = gap.rolling(window=window, min_periods=1).median()
    return pd.DataFrame(
        {"gapDuration": gap, "gapDurationRollMedian": roll_median},
        index=df.index,
    )


def fill_missing_values_with_na(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Detect gaps in a time series and fill them with NaN rows.

    A gap is detected when the time difference between consecutive rows
    exceeds the rolling median gap duration. For each detected gap, rows
    are inserted at the expected timestamps with NaN values.

    Args:
        df: DataFrame with a DatetimeIndex.
        window: Rolling median window size passed to
            :func:`calc_gap_duration`. Default 20.

    Returns:
        DataFrame with original data plus NaN-filled rows at missing
        timestamps. The result is sorted by index.
    """
    gaps = calc_gap_duration(df, window=window)
    median_gap = gaps["gapDurationRollMedian"]

    result = df.copy()
    new_rows = []

    for i in range(1, len(gaps)):
        duration = gaps["gapDuration"].iloc[i]
        expected = median_gap.iloc[i]
        if pd.notna(duration) and pd.notna(expected) and duration > expected:
            freq = pd.Timedelta(seconds=expected)
            start = df.index[i - 1] + freq
            end = df.index[i]
            missing_times = pd.date_range(start=start, end=end, freq=freq)
            # Exclude timestamps already in the index
            missing_times = missing_times.difference(df.index)
            if len(missing_times) > 0:
                nan_df = pd.DataFrame(
                    index=missing_times,
                    columns=df.columns,
                )
                new_rows.append(nan_df)

    if new_rows:
        result = pd.concat([result] + new_rows)
    result = result.sort_index()
    return result


def calc_isna_percentage(
    df: pd.DataFrame,
    column: Optional[str] = None,
    decimals: int = 3,
) -> float:
    """
    Calculate the percentage of NaN values in a DataFrame or column.

    Args:
        df: DataFrame to check.
        column: If given, only check this column. Otherwise check
            the entire DataFrame.
        decimals: Number of decimal places for rounding. Default 3.

    Returns:
        Percentage of NaN values (0–100).
    """
    if column is not None:
        na_count = df[column].isna().sum()
        total = len(df[column])
    else:
        na_count = df.isna().sum().sum()
        total = df.size
    pct = (na_count / total) * 100 if total > 0 else 0.0
    return round(pct, decimals)


def plot_missing_values(
    df: pd.DataFrame,
    column: Optional[str] = None,
    title: Optional[str] = None,
    xlab: str = "Time",
    ylab: str = "Value",
    missing_color: str = "rgba(255,0,0,0.2)",
    line_color: str = "green",
) -> go.Figure:
    """
    Plot a time series and highlight regions with missing (NaN) values.

    Creates a Plotly step plot of the data with red vertical rectangles
    marking NaN regions.

    Args:
        df: DataFrame with a DatetimeIndex.
        column: Column to plot. If *None*, uses the first column.
        title: Plot title. Auto-generated with NaN percentage if *None*.
        xlab: X-axis label. Default ``"Time"``.
        ylab: Y-axis label. Default ``"Value"``.
        missing_color: Fill color for NaN regions.
            Default ``"rgba(255,0,0,0.2)"``.
        line_color: Line color for the data trace. Default ``"green"``.

    Returns:
        go.Figure: Plotly figure with the time series and NaN highlights.
    """
    if column is None:
        column = df.columns[0]

    series = df[column]
    na_pct = calc_isna_percentage(df, column=column)

    if title is None:
        title = f"Missing Values — {column} ({na_pct}% NaN)"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=series,
        mode="lines",
        line=dict(color=line_color, width=1, shape="hv"),
        name=column,
    ))

    # Find NaN regions and add vertical rectangles
    is_na = series.isna()
    if is_na.any():
        # Identify contiguous NaN blocks
        blocks = is_na.ne(is_na.shift()).cumsum()
        for _, group in is_na[is_na].groupby(blocks):
            x0 = group.index[0]
            x1 = group.index[-1]
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=missing_color,
                line_width=0,
                layer="below",
            )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        yaxis_title=ylab,
    )

    return fig


def plot_missing_values_heatmap(
    df: pd.DataFrame,
    title: str = "Missing Values Over Time",
    height: int = 300,
    color_scale: Optional[list] = None,
) -> go.Figure:
    """
    Show a heatmap of missing values across all columns.

    Each row represents a column, each position along the x-axis a timestamp.
    Red cells indicate missing (NaN) values, white cells indicate present values.

    Args:
        df: DataFrame with a DatetimeIndex and one or more value columns.
        title: Plot title.
        height: Figure height in pixels. Default 300.
        color_scale: Two-element color scale list. Default ``["white", "red"]``.

    Returns:
        go.Figure: Plotly heatmap figure.
    """
    if color_scale is None:
        color_scale = [[0, "white"], [1, "red"]]

    na_pct = (df.isna().mean() * 100).round(1)
    subtitle = f"{na_pct.mean():.1f}% missing on average"

    na_matrix = df.isna().astype(int)

    fig = go.Figure(
        go.Heatmap(
            z=na_matrix.T.values,
            x=df.index,
            y=list(df.columns),
            colorscale=color_scale,
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate="Time: %{x}<br>Sensor: %{y}<br>Missing: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title_text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        height=height,
        xaxis_title="Time",
        yaxis_title="Sensor",
    )

    return fig


def calc_outliers(
    df: pd.DataFrame,
    column: Optional[str] = None,
    multiplier: float = 1.5,
) -> Dict:
    """
    Detect outliers using the IQR method.

    Computes the interquartile range and flags values outside
    [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

    Args:
        df: DataFrame with a DatetimeIndex.
        column: Column to analyse. If *None*, uses the first column.
        multiplier: IQR multiplier for the fence. Default 1.5.

    Returns:
        Dict with keys:

        - ``lower`` (float): Lower fence value.
        - ``upper`` (float): Upper fence value.
        - ``outliers`` (pd.DataFrame): Rows flagged as outliers.
        - ``count`` (int): Number of outliers.
        - ``percentage`` (float): Percentage of outliers.
    """
    if column is None:
        column = df.columns[0]

    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    mask = (df[column] < lower) | (df[column] > upper)
    outlier_df = df[mask]

    return {
        "lower": lower,
        "upper": upper,
        "outliers": outlier_df,
        "count": len(outlier_df),
        "percentage": round(len(outlier_df) / len(df) * 100, 2) if len(df) > 0 else 0.0,
    }
