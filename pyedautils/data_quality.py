"""Functions for detecting and visualizing gaps in time series data."""

from typing import Dict, Optional, Union

import numpy as np
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


# ---------------------------------------------------------------------------
# Interval / gap / stuck / range-outlier detection (timestamp-based)
#
# These complement the rolling-median helpers above. They accept either a
# DataFrame with a DatetimeIndex, a DataFrame with a ``timestamp`` column,
# or a Series with a DatetimeIndex, and operate on a single sensor series.
# ---------------------------------------------------------------------------
def _ts_value(data: Union[pd.Series, pd.DataFrame],
              column: Optional[str] = None):
    """Return ``(timestamps, values)`` from flexible single-series input.

    Accepts a Series (DatetimeIndex), a DataFrame with a ``timestamp``
    column, or a DataFrame with a DatetimeIndex. ``column`` selects the
    value column (first column by default).
    """
    if isinstance(data, pd.Series):
        ts = pd.Series(pd.to_datetime(data.index), index=data.index)
        return ts.reset_index(drop=True), data.reset_index(drop=True)
    df = data
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    else:
        ts = pd.Series(pd.to_datetime(df.index)).reset_index(drop=True)
    if column is None:
        value_cols = [c for c in df.columns if c != "timestamp"]
        column = value_cols[0]
    values = df[column].reset_index(drop=True)
    return ts, values


def infer_interval(timestamps: pd.Series) -> pd.Timedelta:
    """Estimate a sensor's sampling interval from the median time difference.

    Args:
        timestamps: Series (or DatetimeIndex) of timestamps.

    Returns:
        pd.Timedelta: Median positive interval; falls back to 10 minutes
        if there are fewer than two usable timestamps.
    """
    sorted_ts = pd.Series(pd.to_datetime(timestamps)).dropna().sort_values()
    if len(sorted_ts) < 2:
        return pd.Timedelta(minutes=10)
    diffs = sorted_ts.diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return pd.Timedelta(minutes=10)
    return positive.median()


def detect_gaps(
    data: Union[pd.Series, pd.DataFrame],
    expected_interval: Optional[pd.Timedelta] = None,
    factor: float = 2.0,
    min_floor: pd.Timedelta = pd.Timedelta(minutes=30),
) -> pd.DataFrame:
    """Find gaps (interruptions) in a single sensor time series.

    A gap is an interval larger than ``factor × expected_interval`` but at
    least *min_floor*. Unlike :func:`calc_gap_duration`/
    :func:`fill_missing_values_with_na` (rolling-median based, NaN-filling),
    this returns an explicit table of gap intervals.

    Args:
        data: Series (DatetimeIndex), or DataFrame with a ``timestamp``
            column or DatetimeIndex.
        expected_interval: Expected sampling interval. Inferred via
            :func:`infer_interval` when *None*.
        factor: Multiplier on the expected interval. Default 2.0.
        min_floor: Minimum gap length to report. Default 30 minutes.

    Returns:
        pd.DataFrame: Columns ``[gap_start, gap_end, gap_duration_h]``.
    """
    ts, _ = _ts_value(data)
    ts = ts.dropna().sort_values().reset_index(drop=True)
    if expected_interval is None:
        expected_interval = infer_interval(ts)
    min_gap = max(expected_interval * factor, min_floor)
    if len(ts) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_duration_h"])

    diffs = ts.diff()
    gap_mask = diffs > min_gap
    gap_starts = ts[gap_mask.shift(-1, fill_value=False)].values
    gap_ends = ts[gap_mask].values

    records = []
    for start, end in zip(gap_starts, gap_ends):
        duration_h = (end - start) / pd.Timedelta(hours=1)
        records.append({"gap_start": start, "gap_end": end,
                        "gap_duration_h": round(duration_h, 2)})
    return pd.DataFrame(records, columns=["gap_start", "gap_end", "gap_duration_h"])


def detect_stuck(
    data: Union[pd.Series, pd.DataFrame],
    min_repeats: int = 20,
    min_duration_h: float = 6.0,
    column: Optional[str] = None,
) -> pd.DataFrame:
    """Find periods where a sensor repeatedly reports the same value.

    Uses run-length encoding: consecutive identical values form a run, and
    runs that are both long enough (``n_repeats >= min_repeats``) and last
    long enough (``stuck_duration_h >= min_duration_h``) are flagged.

    Args:
        data: Series (DatetimeIndex), or DataFrame with a ``timestamp``
            column / DatetimeIndex plus a value column.
        min_repeats: Minimum number of identical consecutive samples.
            Default 20.
        min_duration_h: Minimum duration of the run in hours. Default 6.0.
        column: Value column name (DataFrame input only); first non-
            timestamp column by default.

    Returns:
        pd.DataFrame: Columns ``[stuck_start, stuck_end, stuck_value,
        stuck_duration_h, n_repeats]``.
    """
    out_cols = ["stuck_start", "stuck_end", "stuck_value",
                "stuck_duration_h", "n_repeats"]
    ts, values = _ts_value(data, column=column)
    work = pd.DataFrame({"timestamp": ts, "value": values}).dropna()
    work = work.sort_values("timestamp").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    work["_grp"] = (work["value"] != work["value"].shift()).cumsum()
    runs = (
        work.groupby("_grp")
        .agg(
            stuck_start=("timestamp", "first"),
            stuck_end=("timestamp", "last"),
            stuck_value=("value", "first"),
            n_repeats=("value", "count"),
        )
        .reset_index(drop=True)
    )
    runs["stuck_duration_h"] = (runs["stuck_end"] - runs["stuck_start"]) / pd.Timedelta(hours=1)
    stuck = runs[
        (runs["n_repeats"] >= min_repeats) & (runs["stuck_duration_h"] >= min_duration_h)
    ].copy()
    stuck["stuck_duration_h"] = stuck["stuck_duration_h"].round(2)
    return stuck[out_cols].reset_index(drop=True)


def detect_outliers(
    data: Union[pd.Series, pd.DataFrame],
    lo: float,
    hi: float,
    column: Optional[str] = None,
) -> pd.DataFrame:
    """Find values outside a plausibility range ``[lo, hi]``.

    Complements the IQR-based :func:`calc_outliers` with a fixed,
    physically motivated range (e.g. relative humidity 0–100 %).

    Args:
        data: Series (DatetimeIndex), or DataFrame with a ``timestamp``
            column / DatetimeIndex plus a value column.
        lo: Lower plausibility bound (inclusive).
        hi: Upper plausibility bound (inclusive).
        column: Value column name (DataFrame input only).

    Returns:
        pd.DataFrame: Columns ``[timestamp, value, reason]`` sorted by
        timestamp; empty if there are no out-of-range values.
    """
    ts, values = _ts_value(data, column=column)
    work = pd.DataFrame({"timestamp": ts, "value": values}).dropna()
    below = work[work["value"] < lo].copy()
    below["reason"] = f"below {lo}"
    above = work[work["value"] > hi].copy()
    above["reason"] = f"above {hi}"
    result = pd.concat([below, above], ignore_index=True)
    if result.empty:
        return pd.DataFrame(columns=["timestamp", "value", "reason"])
    return result.sort_values("timestamp").reset_index(drop=True)[
        ["timestamp", "value", "reason"]]


#: Default thresholds for :func:`classify_quality_flags`.
DEFAULT_QUALITY_THRESHOLDS: Dict[str, float] = {
    "cov_warn":   90.0,  "cov_crit":   70.0,  # coverage [%], lower bound
    "gap_warn":   24.0,  "gap_crit":  168.0,  # longest gap [h], upper bound
    "out_warn":    1.0,  "out_crit":    5.0,  # outliers [%], upper bound
    "stuck_warn":  1.0,  "stuck_crit":  5.0,  # stuck periods [count], upper bound
}


def classify_quality_flags(
    summary: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Classify each row of a quality summary as ok / warning / critical.

    A row is *critical* if any metric crosses its critical threshold, else
    *warning* if any crosses its warning threshold, else *ok*.

    Args:
        summary: DataFrame with (any subset of) columns ``coverage_pct``,
            ``longest_gap_h``, ``outlier_pct``, ``n_stuck_periods``.
        thresholds: Overrides merged onto
            :data:`DEFAULT_QUALITY_THRESHOLDS`.

    Returns:
        pd.Series: ``"ok"`` / ``"warning"`` / ``"critical"`` per row,
        named ``quality_flag``.
    """
    t = {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})}

    def _col(name: str, default: float) -> pd.Series:
        # Missing columns default to a neutral value so they never trigger a flag.
        if name not in summary.columns:
            return pd.Series(default, index=summary.index)
        return pd.to_numeric(summary[name], errors="coerce").fillna(default)

    cov = _col("coverage_pct", 100.0)
    gap = _col("longest_gap_h", 0.0)
    out = _col("outlier_pct", 0.0)
    stuck = _col("n_stuck_periods", 0.0)
    is_crit = (
        (cov < t["cov_crit"]) | (gap >= t["gap_crit"]) |
        (out >= t["out_crit"]) | (stuck >= t["stuck_crit"])
    )
    is_warn = (
        (cov < t["cov_warn"]) | (gap >= t["gap_warn"]) |
        (out >= t["out_warn"]) | (stuck >= t["stuck_warn"])
    )
    return pd.Series(
        np.where(is_crit, "critical", np.where(is_warn, "warning", "ok")),
        index=summary.index,
        name="quality_flag",
    )


def plot_data_quality(
    data: Union[pd.Series, pd.DataFrame],
    expected_interval: Optional[pd.Timedelta] = None,
    column: Optional[str] = None,
    title: str = "Data Quality",
    ylab: str = "Value",
    line_color: str = "#0D7377",
    gap_color: str = "rgba(239,68,68,0.25)",
    height: int = 300,
) -> go.Figure:
    """Plot a sensor series with detected gaps shaded.

    Draws the time series as points (WebGL) with red background rectangles
    over the gaps found by :func:`detect_gaps`.

    Args:
        data: Series (DatetimeIndex), or DataFrame with a ``timestamp``
            column / DatetimeIndex plus a value column.
        expected_interval: Passed to :func:`detect_gaps`; inferred if
            *None*.
        column: Value column name (DataFrame input only).
        title: Plot title.
        ylab: Y-axis label.
        line_color: Marker/line color for the data trace.
        gap_color: Fill color for gap rectangles.
        height: Figure height in pixels. Default 300.

    Returns:
        go.Figure
    """
    ts, values = _ts_value(data, column=column)
    chart = pd.DataFrame({"timestamp": ts, "value": values}).sort_values("timestamp")
    gaps = detect_gaps(data, expected_interval=expected_interval)

    gap_shapes = [
        {
            "type": "rect", "xref": "x", "yref": "paper",
            "x0": g["gap_start"], "x1": g["gap_end"], "y0": 0, "y1": 1,
            "fillcolor": gap_color, "line": {"width": 0}, "layer": "below",
        }
        for _, g in gaps.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=chart["timestamp"], y=chart["value"],
        mode="markers", marker=dict(color=line_color, size=3),
        name="Value",
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        shapes=gap_shapes,
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        height=height,
        yaxis_title=ylab,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
