"""Daily profile analysis plots."""

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


def plot_daily_profiles(
    data: pd.DataFrame,
    method: str = "mean",
    loc_time_zone: str = "UTC",
    title: Optional[str] = None,
    ylab: Optional[str] = None,
    k: int = 672,
    digits: int = 1,
):
    """
    Create a daily profile plot with one line per weekday.

    Supports three methods:

    - ``"mean"``: Simple mean per weekday and time-of-day.
    - ``"decomposed"``: Detrend with a rolling mean, extract the seasonal
      component, and normalise each weekday to start at 0 at midnight.
    - ``"overlayed"``: All individual daily profiles overlaid in blue.
      Hovering highlights the entire day-line in orange (requires JS).

    Args:
        data: DataFrame with two columns: timestamp and value.
        method: Aggregation method — ``"mean"``, ``"decomposed"``, or
            ``"overlayed"``.
        loc_time_zone: Timezone for localization (e.g. "Europe/Zurich"). Default "UTC".
        title: Plot title. Auto-set per method if *None*.
        ylab: Y-axis label. Auto-set per method if *None*.
        k: Rolling window size for trend (only used with ``"decomposed"``).
            Default 672 (7 days at 15-min resolution).
        digits: Number of decimal places for rounding. Default 1.

    Returns:
        go.Figure for ``"mean"`` / ``"decomposed"``, or
        str (self-contained HTML) for ``"overlayed"``.
    """
    valid_methods = ("mean", "decomposed", "overlayed")
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )

    if title is None:
        title = {
            "mean": "Daily Profiles - Mean",
            "decomposed": "Daily Profiles - Decomposed",
            "overlayed": "Daily Profiles - Overlayed",
        }[method]
    if ylab is None:
        ylab = {
            "mean": "Energy Consumption",
            "decomposed": "delta Energy Consumption",
            "overlayed": "Energy Consumption",
        }[method]

    import plotly.express as px

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    weekdays = DEFAULT_WEEKDAYS
    viridis_colors = px.colors.sample_colorscale(
        "Viridis", [i / max(len(weekdays) - 1, 1) for i in range(len(weekdays))]
    )

    if method == "overlayed":
        import json as _json
        import uuid

        df["x"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        df["weekday"] = df["timestamp"].dt.day_name()
        df["day"] = df["timestamp"].dt.date

        # Set value at exact midnight to NaN (avoid line jumping back to 0)
        df.loc[df["x"] == 0.0, "value"] = None

        base_color = "rgba(31, 119, 180, 0.15)"
        color_map = dict(zip(weekdays, viridis_colors))

        fig = go.Figure()

        for day_name in weekdays:
            day_data = df[df["weekday"] == day_name]
            legend_shown = False
            for date, grp in day_data.groupby("day"):
                grp = grp.sort_values("x")
                fig.add_trace(go.Scatter(
                    x=grp["x"],
                    y=grp["value"],
                    mode="lines",
                    name=f"{day_name} ({date})",
                    legendgroup=day_name,
                    showlegend=not legend_shown,
                    line=dict(color=base_color, width=1),
                    hoverinfo="none",
                ))
                legend_shown = True

        max_val = df["value"].max()
        fig.update_layout(
            title_text=f"<b>{title}</b>",
            title_font=dict(size=20),
            template="plotly_white",
            title_x=0.5,
            xaxis_title="Hour of Day",
            yaxis_title=ylab,
            xaxis=dict(tickvals=list(range(0, 25, 3))),
            yaxis=dict(range=[0, max_val * 1.05 if max_val and max_val > 0 else None]),
            hovermode="closest",
        )

        # Build traces as plain JSON (avoid plotly's binary encoding)
        traces_json = []
        for trace in fig.data:
            x_vals = [float(v) for v in trace.x] if trace.x is not None else []
            y_vals = [None if (v is None or pd.isna(v)) else float(v)
                      for v in trace.y] if trace.y is not None else []
            traces_json.append({
                "x": x_vals, "y": y_vals,
                "mode": trace.mode, "name": trace.legendgroup if trace.showlegend else trace.name,
                "type": "scatter",
                "legendgroup": trace.legendgroup,
                "showlegend": trace.showlegend,
                "line": {"color": trace.line.color, "width": trace.line.width},
                "hoverinfo": "none",
            })
        layout_json = _json.loads(fig.to_json())["layout"]
        fig_json = _json.dumps({"data": traces_json, "layout": layout_json})
        uid = uuid.uuid4().hex[:8]
        div_id = f"overlayed_{uid}"
        tt_id = f"tt_{uid}"
        color_map_json = _json.dumps(color_map)

        return f"""<div id="{div_id}" style="width:100%;position:relative;"></div>
<div id="{tt_id}" style="position:fixed;background:rgba(255,255,255,0.95);
border:1px solid #ccc;border-radius:4px;padding:6px 10px;pointer-events:none;
font-family:Arial,sans-serif;font-size:12px;box-shadow:2px 2px 6px rgba(0,0,0,0.15);
display:none;z-index:9999;"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
(function() {{
  var figData = {fig_json};
  var baseColor = "{base_color}";
  var highlightColor = "orange";
  var el = document.getElementById("{div_id}");
  var tt = document.getElementById("{tt_id}");
  var colorMap = {color_map_json};
  var prevIdx = -1;

  Plotly.newPlot(el, figData.data, figData.layout, {{
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  }}).then(function() {{

    el.on("plotly_hover", function(eventData) {{
      var traceIdx = eventData.points[0].curveNumber;
      if (traceIdx === prevIdx) return;
      prevIdx = traceIdx;

      var traceName = el.data[traceIdx].name;
      var parts = traceName.match(/^(.+?)\\s*\\((.+)\\)$/);
      var dayName = parts ? parts[1] : traceName;
      var date = parts ? parts[2] : "";
      var val = eventData.points[0].y;
      var hour = eventData.points[0].x;

      // Show tooltip
      tt.innerHTML = "<b>" + date + " (" + dayName + ")</b><br>"
        + "Time: " + hour.toFixed(2) + "h<br>"
        + "Value: " + (val != null ? val.toFixed(0) : "—");
      tt.style.display = "block";

      // Highlight hovered trace
      var colors = [];
      var widths = [];
      var opacities = [];
      for (var i = 0; i < el.data.length; i++) {{
        if (i === traceIdx) {{
          colors.push(highlightColor);
          widths.push(3);
          opacities.push(1.0);
        }} else {{
          colors.push(baseColor);
          widths.push(1);
          opacities.push(1.0);
        }}
      }}
      Plotly.restyle(el, {{"line.color": colors, "line.width": widths, "opacity": opacities}});
    }});

    el.on("plotly_unhover", function() {{
      prevIdx = -1;
      tt.style.display = "none";
      var colors = [];
      var widths = [];
      for (var i = 0; i < el.data.length; i++) {{
        colors.push(baseColor);
        widths.push(1);
      }}
      Plotly.restyle(el, {{"line.color": colors, "line.width": widths}});
    }});

    // Move tooltip with mouse
    el.addEventListener("mousemove", function(e) {{
      tt.style.left = (e.clientX + 15) + "px";
      tt.style.top = (e.clientY - 40) + "px";
    }});

    // Double-click to reset
    el.on("plotly_doubleclick", function() {{
      var colors = [];
      var widths = [];
      for (var i = 0; i < el.data.length; i++) {{
        colors.push(baseColor);
        widths.push(1);
      }}
      Plotly.restyle(el, {{"line.color": colors, "line.width": widths}});
    }});
  }});
}})();
</script>"""

    # --- mean / decomposed ---
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    df = df.tz_localize(tz=loc_time_zone, ambiguous=True, nonexistent="shift_forward")

    df["weekday"] = df.index.dayofweek
    df["dayhour"] = df.index.hour
    df["dayminute"] = df.index.minute

    if method == "decomposed":
        # Detrend
        roll_mean = df["value"].rolling(window=k, center=True).mean()
        df["valueDetrended"] = df["value"] - roll_mean
        df = df.dropna(subset=["valueDetrended"])

        # Seasonal component
        seasonal = df.groupby(["weekday", "dayhour", "dayminute"])["valueDetrended"].mean().reset_index()

        # Per-weekday: subtract value at midnight so each day starts at 0
        start_values = seasonal.loc[
            (seasonal["dayhour"] == 0) & (seasonal["dayminute"] == 0),
            ["weekday", "valueDetrended"],
        ].rename(columns={"valueDetrended": "valueStart"})
        seasonal = seasonal.merge(start_values, on="weekday")
        seasonal["value"] = seasonal["valueDetrended"] - seasonal["valueStart"]
        df_plot = seasonal
    else:
        # Simple mean
        df_plot = df.groupby(["weekday", "dayhour", "dayminute"])["value"].mean().reset_index()

    # Prepare plot data
    df_plot["time"] = df_plot["dayhour"] + df_plot["dayminute"] / 60
    df_plot["value"] = df_plot["value"].round(digits)
    df_plot = df_plot.sort_values(by=["weekday", "time"])

    df_plot["weekday"] = df_plot["weekday"].map(lambda x: weekdays[x])

    fig = make_subplots(rows=1, cols=1)

    for i, day in enumerate(weekdays):
        subset = df_plot[df_plot["weekday"] == day]
        fig.add_trace(go.Scatter(
            x=subset["time"], y=subset["value"],
            mode="lines", name=str(day),
            line=dict(color=viridis_colors[i]),
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
