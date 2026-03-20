"""Plot functions for energy data analysis and visualization."""

from typing import Dict, List, Optional, Tuple

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

DEFAULT_SEASON_COLORS = {
    "Winter": "#365c8d",
    "Spring": "#2db27d",
    "Summer": "#febc2b",
    "Fall": "#824b04",
}

_SEASON_LABELS_DE = {
    "Winter": "Winter",
    "Spring": "Frühling",
    "Summer": "Sommer",
    "Fall": "Herbst",
}


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


def plot_heatmap_median_weeks(
    data: pd.DataFrame,
    title: str = "Heatmap Median per Hour by Weekday and Season",
    ylab: str = "Energy Consumption (kWh/h)",
    seasons: Optional[List[str]] = None,
    colorscale: str = "Magma",
) -> go.Figure:
    """
    Create a heatmap of median values per hour, grouped by weekday and season.

    Each season is shown as a separate subplot column. The y-axis shows weekdays
    (Monday at top, Sunday at bottom), the x-axis shows the hour of day (0-23),
    and the color intensity represents the median value.

    Args:
        data: DataFrame with two columns: timestamp and value.
        title: Plot title.
        ylab: Colorbar title / value label.
        seasons: Custom season names in column order.
            Default: Spring, Summer, Fall, Winter.
        colorscale: Plotly colorscale name. Default "Magma" (≈ viridis option B).

    Returns:
        go.Figure: Plotly figure with one heatmap subplot per season.
    """
    from pyedautils.season import get_season

    if seasons is None:
        seasons = DEFAULT_SEASONS

    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("h")

    df_h = df.groupby("hour").agg({"value": "sum"}).reset_index()
    df_h["weekday"] = df_h["hour"].dt.day_name()
    df_h["dayhour"] = df_h["hour"].dt.hour
    df_h["season"] = get_season(df_h["hour"])

    # Rename internal English season names to custom labels
    rename_map = dict(zip(DEFAULT_SEASONS, seasons))
    df_h["season"] = df_h["season"].map(rename_map).fillna(df_h["season"])

    df_median = (
        df_h.groupby(["season", "weekday", "dayhour"])["value"]
        .median()
        .reset_index()
    )

    weekday_full = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    weekday_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = make_subplots(
        rows=1, cols=len(seasons),
        subplot_titles=seasons,
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    # Build a full grid per season so empty cells show as blank
    zmin = df_median["value"].min()
    zmax = df_median["value"].max()

    for col_idx, season in enumerate(seasons, start=1):
        subset = df_median[df_median["season"] == season]

        # Pivot to weekday × hour matrix
        pivot = subset.pivot(index="weekday", columns="dayhour", values="value")
        pivot = pivot.reindex(index=weekday_full, columns=range(24))

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=list(range(24)),
                y=weekday_abbr,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=(col_idx == len(seasons)),
                colorbar=dict(title="Legend") if col_idx == len(seasons) else None,
                hovertemplate=(
                    "Hour: %{x}<br>%{y}<br>Median: %{z:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            tickvals=[0, 6, 12, 18, 24],
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        height=500,
        yaxis_title=ylab,
        yaxis_autorange="reversed",
    )
    # Shared x-axis label at bottom center
    fig.add_annotation(
        text="Hour of day",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=14),
    )

    return fig


def plot_heatmap_calendar(
    data: pd.DataFrame,
    title: str = "Calendar Heatmap",
    ylab: str = "Energy Consumption (kWh/d)",
    colorscale: str = "Magma",
) -> go.Figure:
    """
    Create a calendar heatmap of daily aggregated values.

    Each cell represents one day, coloured by the aggregated daily value.
    Rows are weekdays (Mon–Sun), columns are calendar weeks, and each year
    is shown as a separate subplot row.

    Args:
        data: DataFrame with two columns: timestamp and value.
        title: Plot title.
        ylab: Colorbar title / value label.
        colorscale: Plotly colorscale name. Default "Magma" (≈ viridis option B).

    Returns:
        go.Figure: Plotly figure with one subplot row per year.
    """
    df = data.copy()
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.floor("D")

    df_daily = df.groupby("day")["value"].sum().reset_index()
    df_daily["weekday"] = df_daily["day"].dt.dayofweek  # 0=Mon
    df_daily["year"] = df_daily["day"].dt.year
    df_daily["month"] = df_daily["day"].dt.month

    # Use ISO week but fix year boundaries:
    # - Dec days in ISO week 1 → show as week 53 of current year
    # - Jan days in ISO week 52/53 → show as week 0 of current year
    iso_week = df_daily["day"].dt.isocalendar().week.astype(int)
    df_daily["week"] = iso_week
    df_daily.loc[
        (df_daily["month"] == 12) & (df_daily["week"] == 1), "week"
    ] = 53
    df_daily.loc[
        (df_daily["month"] == 1) & (df_daily["week"] >= 52), "week"
    ] = 0

    years = sorted(df_daily["year"].unique())
    weekday_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    zmin = df_daily["value"].min()
    zmax = df_daily["value"].max()

    fig = make_subplots(
        rows=len(years), cols=1,
        subplot_titles=[str(y) for y in years],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # Month labels: find the week where each month starts
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    for row_idx, year in enumerate(years, start=1):
        year_data = df_daily[df_daily["year"] == year]

        # Pivot to weekday × week matrix
        all_weeks = list(range(0, 54))
        pivot = year_data.pivot_table(
            index="weekday", columns="week", values="value", aggfunc="sum",
        ).reindex(index=range(7), columns=all_weeks)

        day_pivot = year_data.pivot_table(
            index="weekday", columns="week", values="day", aggfunc="first",
        ).reindex(index=range(7), columns=all_weeks)

        hover_text = []
        for wd in range(7):
            row_text = []
            for wk in all_weeks:
                val = pivot.loc[wd, wk] if wd in pivot.index else None
                day = day_pivot.loc[wd, wk] if wd in day_pivot.index else None
                if pd.notna(val) and pd.notna(day):
                    row_text.append(
                        f"{pd.Timestamp(day).strftime('%Y-%m-%d')} ({weekday_abbr[wd]})"
                        f"<br>Value: {val:.0f}"
                    )
                else:
                    row_text.append("")
            hover_text.append(row_text)

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=all_weeks,
                y=weekday_abbr,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=(row_idx == len(years)),
                colorbar=dict(title="Legend") if row_idx == len(years) else None,
                hoverinfo="text",
                text=hover_text,
                xgap=1,
                ygap=1,
            ),
            row=row_idx,
            col=1,
        )

        # Month tick labels and border lines
        month_ticks = []
        month_labels = []
        for m in range(1, 13):
            month_data = year_data[year_data["month"] == m]
            if not month_data.empty:
                week_start = month_data["week"].min()
                month_ticks.append(week_start)
                month_labels.append(month_names[m - 1])

                # Add vertical month separator line
                if m > 1:
                    # Line at left edge of first week of new month
                    x_pos = week_start - 0.5
                    # Find which weekday the month starts on
                    first_day = month_data.loc[
                        month_data["week"] == week_start, "weekday"
                    ].min()
                    # Draw line from top to the start weekday
                    # then horizontal to previous week, then down
                    ax = fig.get_subplot(row_idx, 1)
                    xref = ax.xaxis.plotly_name.replace("axis", "")
                    yref = ax.yaxis.plotly_name.replace("axis", "")

                    if first_day > 0:
                        # Partial week: L-shaped border
                        # Vertical line from top to first_day
                        fig.add_shape(
                            type="line",
                            x0=x_pos, x1=x_pos,
                            y0=-0.5, y1=first_day - 0.5,
                            xref=xref, yref=yref,
                            line=dict(color="white", width=4),
                        )
                        # Horizontal line at first_day to previous week
                        fig.add_shape(
                            type="line",
                            x0=x_pos, x1=x_pos - 1,
                            y0=first_day - 0.5, y1=first_day - 0.5,
                            xref=xref, yref=yref,
                            line=dict(color="white", width=4),
                        )
                        # Vertical line from first_day to bottom
                        fig.add_shape(
                            type="line",
                            x0=x_pos - 1, x1=x_pos - 1,
                            y0=first_day - 0.5, y1=6.5,
                            xref=xref, yref=yref,
                            line=dict(color="white", width=4),
                        )
                    else:
                        # Month starts on Monday: simple vertical line
                        fig.add_shape(
                            type="line",
                            x0=x_pos, x1=x_pos,
                            y0=-0.5, y1=6.5,
                            xref=xref, yref=yref,
                            line=dict(color="white", width=4),
                        )

        fig.update_xaxes(
            tickvals=month_ticks,
            ticktext=month_labels,
            row=row_idx, col=1,
        )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        height=250 * len(years) + 100,
        yaxis_autorange="reversed",
    )

    # Reverse y-axis for all subplots
    for row_idx in range(2, len(years) + 1):
        fig.update_yaxes(autorange="reversed", row=row_idx, col=1)

    return fig


def plot_energy_signature(
    data: pd.DataFrame,
    title: str = "Building Energy Signature",
    xlab: str = "Outside Temperature [°C]",
    ylab: str = "Energy Consumption [kWh/d]",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Create a scatter plot of daily energy consumption vs outside temperature.

    Each point represents one day, colored by season. The input DataFrame
    is aggregated to daily resolution (mean temperature, sum of value).

    Args:
        data: DataFrame with columns ``[timestamp, temperature, value]``.
        title: Plot title.
        xlab: X-axis label.
        ylab: Y-axis label.
        colors: Season color overrides keyed by season name.
            Default uses ``DEFAULT_SEASON_COLORS``.

    Returns:
        go.Figure: Plotly figure with the scatter plot.
    """
    from pyedautils.season import get_season

    c = {**DEFAULT_SEASON_COLORS, **(colors or {})}

    df = data.copy()
    df.columns = ["timestamp", "temperature", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.date

    daily = df.groupby("day").agg(
        temperature=("temperature", "mean"),
        value=("value", "sum"),
    ).reset_index()

    daily["timestamp"] = pd.to_datetime(daily["day"])
    daily["season"] = get_season(daily["timestamp"])

    fig = go.Figure()

    for season in DEFAULT_SEASON_COLORS:
        subset = daily[daily["season"] == season]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["temperature"],
            y=subset["value"],
            mode="markers",
            name=season,
            marker=dict(color=c.get(season, "#999"), size=6, opacity=0.7),
            hovertemplate=(
                "Date: %{customdata}<br>"
                f"{xlab}: " + "%{x:.1f}<br>"
                f"{ylab}: " + "%{y:.1f}<br>"
                f"Season: {season}"
                "<extra></extra>"
            ),
            customdata=subset["day"].astype(str),
        ))

    # Fix axis ranges so toggling seasons doesn't rescale
    x_pad = (daily["temperature"].max() - daily["temperature"].min()) * 0.05
    y_pad = (daily["value"].max() - daily["value"].min()) * 0.05
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        yaxis_title=ylab,
        xaxis_range=[daily["temperature"].min() - x_pad,
                     daily["temperature"].max() + x_pad],
        yaxis_range=[daily["value"].min() - y_pad,
                     daily["value"].max() + y_pad],
    )

    return fig


def plot_energy_signature_pes(
    data: pd.DataFrame,
    p_ihg: float = 0.0,
    title: str = "Proposed Energy Signature",
    xlab: str = "Outside Temperature [°C]",
    ylab: str = "Power [kW]",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Create a Proposed Energy Signature plot with regression lines.

    Computes the PES parameters (balance temperature, heat loss coefficient,
    standby and hot-water power) and overlays the characteristic lines on
    a daily scatter plot colored by season.

    Args:
        data: DataFrame with columns
            ``[timestamp, outside_temp, power, room_temp]``.
        p_ihg: Internal heat gains [kW]. Default 0.
        title: Plot title.
        xlab: X-axis label.
        ylab: Y-axis label.
        colors: Season color overrides keyed by season name.

    Returns:
        go.Figure: Plotly figure with scatter and regression lines.
    """
    from pyedautils.energy_signature import compute_pes
    from pyedautils.season import get_season

    c = {**DEFAULT_SEASON_COLORS, **(colors or {})}

    pes = compute_pes(data, p_ihg=p_ihg)

    df = data.copy()
    df.columns = ["timestamp", "outside_temp", "power", "room_temp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.date

    daily = df.groupby("day").agg(
        outside_temp=("outside_temp", "mean"),
        power=("power", "mean"),
    ).reset_index()

    daily["timestamp"] = pd.to_datetime(daily["day"])
    daily["season"] = get_season(daily["timestamp"])

    fig = go.Figure()

    # Scatter by season
    for season in DEFAULT_SEASON_COLORS:
        subset = daily[daily["season"] == season]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["outside_temp"],
            y=subset["power"],
            mode="markers",
            name=season,
            marker=dict(color=c.get(season, "#999"), size=6, opacity=0.7),
            hovertemplate=(
                "Date: %{customdata}<br>"
                f"{xlab}: " + "%{x:.1f}<br>"
                f"{ylab}: " + "%{y:.2f}<br>"
                f"Season: {season}"
                "<extra></extra>"
            ),
            customdata=subset["day"].astype(str),
        ))

    # Regression lines (Eq. 9: P = Q_tot * (Tb - T_oa)+ + P_dhw + P_dhwc)
    t_min = daily["outside_temp"].min()
    base_power = pes.p_dhwc + pes.p_dhw

    # Heating line: from t_min to Tb
    t_heat = [t_min, pes.tb]
    p_heat = [base_power + pes.q_tot * (pes.tb - t_min), base_power]

    fig.add_trace(go.Scatter(
        x=t_heat, y=p_heat,
        mode="lines",
        name="Heating line",
        line=dict(color="red", width=2),
    ))

    # Base load line (P_dhw + P_dhwc): from Tb to max temp
    t_max = daily["outside_temp"].max()
    fig.add_trace(go.Scatter(
        x=[pes.tb, t_max], y=[base_power, base_power],
        mode="lines",
        name=f"P_dhw + P_dhwc = {base_power:.2f} kW",
        line=dict(color="blue", width=2),
    ))

    # DHWC line (standby)
    fig.add_trace(go.Scatter(
        x=[pes.tb, t_max], y=[pes.p_dhwc, pes.p_dhwc],
        mode="lines",
        name=f"P_dhwc = {pes.p_dhwc:.2f} kW",
        line=dict(color="green", width=2, dash="dash"),
    ))

    # Annotations
    fig.add_annotation(
        x=pes.tb, y=base_power,
        text=f"T<sub>b</sub> = {pes.tb:.1f} °C",
        showarrow=True, arrowhead=2,
        ax=40, ay=-30,
    )
    fig.add_annotation(
        x=(t_min + pes.tb) / 2,
        y=(p_heat[0] + p_heat[1]) / 2,
        text=f"Q<sub>tot</sub> = {pes.q_tot:.3f} kW/K",
        showarrow=False,
        yshift=15,
    )

    # Fix axis ranges so toggling seasons doesn't rescale
    all_x = [daily["outside_temp"].min(), t_min, pes.tb, t_max]
    all_y = [daily["power"].min(), daily["power"].max(),
             p_heat[0], base_power, pes.p_dhwc]
    x_lo, x_hi = min(all_x), max(all_x)
    y_lo, y_hi = min(all_y), max(all_y)
    x_pad = (x_hi - x_lo) * 0.05
    y_pad = (y_hi - y_lo) * 0.05
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        yaxis_title=ylab,
        xaxis_range=[x_lo - x_pad, x_hi + x_pad],
        yaxis_range=[y_lo - y_pad, y_hi + y_pad],
    )

    return fig


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


def plot_comfort_sia180(
    data_outdoor: pd.DataFrame,
    data_room: pd.DataFrame,
    title: str = "Thermal Comfort according to SIA 180:2014",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    SIA 180:2014 thermal comfort plot.

    Scatter of room temperature vs 48-hour rolling mean outdoor
    temperature, colored by season, with SIA 180 comfort boundaries.

    Args:
        data_outdoor: DataFrame ``[timestamp, value]`` with outdoor temp.
        data_room: DataFrame ``[timestamp, value]`` with room temp.
        title: Plot title.
        colors: Season color overrides.

    Returns:
        go.Figure
    """
    from pyedautils.season import get_season

    c = {**DEFAULT_SEASON_COLORS, **(colors or {})}

    # Outdoor: hourly mean, fill gaps, 48h rolling mean
    df_oa = data_outdoor.copy()
    df_oa.columns = ["timestamp", "value"]
    df_oa["timestamp"] = pd.to_datetime(df_oa["timestamp"])
    df_oa["hour"] = df_oa["timestamp"].dt.floor("h")
    df_oa = df_oa.groupby("hour")["value"].mean().reset_index()
    df_oa.columns = ["timestamp", "temp_oa"]
    full = pd.date_range(df_oa["timestamp"].min(),
                         df_oa["timestamp"].max(), freq="h")
    df_oa = df_oa.set_index("timestamp").reindex(full).interpolate()
    df_oa["temp_oa_48h"] = df_oa["temp_oa"].rolling(48, min_periods=1).mean()
    df_oa = df_oa.dropna(subset=["temp_oa_48h"]).reset_index()
    df_oa.columns = ["timestamp", "temp_oa", "temp_oa_48h"]

    # Room: hourly mean
    df_r = data_room.copy()
    df_r.columns = ["timestamp", "value"]
    df_r["timestamp"] = pd.to_datetime(df_r["timestamp"])
    df_r["hour"] = df_r["timestamp"].dt.floor("h")
    df_r = df_r.groupby("hour")["value"].mean().reset_index()
    df_r.columns = ["timestamp", "temp_r"]

    # Merge
    data = df_oa[["timestamp", "temp_oa_48h"]].merge(
        df_r, on="timestamp", how="inner"
    ).dropna()
    data["season"] = get_season(data["timestamp"])

    # Axis ranges
    min_x = min(0, data["temp_oa_48h"].min())
    max_x = max(28, data["temp_oa_48h"].max())
    min_y = min(21, data["temp_r"].min()) - 1
    max_y = max(32, data["temp_r"].max()) + 1

    fig = go.Figure()

    # SIA 180 boundaries
    # Lower limit (heating setpoint)
    fig.add_trace(go.Scatter(
        x=[min_x, 19, 23.5, max_x],
        y=[20.5, 20.5, 22, 22],
        mode="lines", name="Lower limit SIA 180",
        line=dict(color="#440154", width=2),
    ))
    # Upper limit active cooling
    fig.add_trace(go.Scatter(
        x=[min_x, 12, 17.5, max_x],
        y=[24.5, 24.5, 26.5, 26.5],
        mode="lines", name="Upper limit active cooling",
        line=dict(color="#1E9B8A", width=2),
    ))
    # Upper limit passive cooling
    fig.add_trace(go.Scatter(
        x=[min_x, 10, max_x],
        y=[25, 25, 0.33 * max_x + 21.8],
        mode="lines", name="Upper limit passive cooling",
        line=dict(color="#FDE725", width=2),
    ))

    # Scatter by season
    for season in ["Spring", "Summer", "Fall", "Winter"]:
        s = data[data["season"] == season]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["temp_oa_48h"], y=s["temp_r"],
            mode="markers", name=season,
            marker=dict(color=c.get(season, "#999"), size=5, opacity=0.3),
            hovertemplate=(
                "T_room: %{y:.1f} °C<br>"
                "T_oa (48h): %{x:.1f} °C<br>"
                "Date: %{customdata}<br>"
                f"Season: {season}<extra></extra>"
            ),
            customdata=s["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis=dict(
            title="Moving avg outdoor temp (48h) [°C]",
            range=[min_x, max_x], dtick=2,
        ),
        yaxis=dict(
            title="Room Temperature [°C]",
            range=[min_y, max_y], dtick=1,
        ),
    )
    return fig


def plot_comfort_temp_humidity(
    data: pd.DataFrame,
    title: str = "Temperature vs Humidity Comfort Plot",
    colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Scatter of daily mean temperature vs humidity with comfort zones.

    Shows two comfort zone polygons: "comfortable" (green) and
    "still comfortable" (orange), based on common building standards.

    Args:
        data: DataFrame ``[timestamp, temperature, humidity]``.
            Humidity in %rH, temperature in °C.
        title: Plot title.
        colors: Season color overrides.

    Returns:
        go.Figure
    """
    from pyedautils.season import get_season

    c = {**DEFAULT_SEASON_COLORS, **(colors or {})}

    df = data.copy()
    df.columns = ["timestamp", "temperature", "humidity"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.date

    daily = df.groupby("day").agg(
        temperature=("temperature", "mean"),
        humidity=("humidity", "mean"),
    ).reset_index()
    daily["timestamp"] = pd.to_datetime(daily["day"])
    daily["season"] = get_season(daily["timestamp"])

    fig = go.Figure()

    # Comfort zones (polygons)
    # "Still comfortable"
    still_t = [20, 17, 16, 17, 21.5, 25, 27, 25.5, 20]
    still_h = [20, 40, 75, 85, 80, 60, 30, 20, 20]
    fig.add_trace(go.Scatter(
        x=still_t, y=still_h,
        mode="lines",
        fill="toself", fillcolor="rgba(255,165,0,0.25)",
        line=dict(color="orange"),
        name="Still comfortable",
    ))

    # "Comfortable"
    comf_t = [19, 17.5, 22.5, 24, 19]
    comf_h = [38, 74, 65, 35, 38]
    fig.add_trace(go.Scatter(
        x=comf_t, y=comf_h,
        mode="lines",
        fill="toself", fillcolor="rgba(154,205,50,0.4)",
        line=dict(color="yellowgreen"),
        name="Comfortable",
    ))

    # Scatter by season
    for season in ["Spring", "Summer", "Fall", "Winter"]:
        s = daily[daily["season"] == season]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["temperature"], y=s["humidity"],
            mode="markers", name=season,
            marker=dict(color=c.get(season, "#999"), size=6, opacity=0.5),
            hovertemplate=(
                "Temp: %{x:.1f} °C<br>"
                "Hum: %{y:.1f} %rH<br>"
                "Date: %{customdata}<br>"
                f"Season: {season}<extra></extra>"
            ),
            customdata=s["day"].astype(str),
        ))

    min_x = min(14, daily["temperature"].min())
    max_x = max(28, daily["temperature"].max())

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis=dict(title="Temperature [°C]",
                   range=[min_x, max_x], dtick=2),
        yaxis=dict(title="Humidity [%rH]",
                   range=[0, 100], dtick=20),
    )
    return fig


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


def _load_d3_js():
    """Load bundled D3.js source files for the Mollier diagram."""
    from importlib import resources as _res
    d3_dir = _res.files("pyedautils") / "data" / "d3_mollier"
    return {
        "mollier_functions": (d3_dir / "mollier_functions.js").read_text(encoding="utf-8"),
        "coordinate_generator": (d3_dir / "CoordinateGenerator.js").read_text(encoding="utf-8"),
        "draw_comfort": (d3_dir / "drawComfort.js").read_text(encoding="utf-8"),
    }


def _get_season_fast(dt):
    """Fast season detection without ephem (for D3 data prep)."""
    m = dt.month
    if m in (12, 1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Spring"
    elif m in (6, 7, 8):
        return "Summer"
    return "Fall"


def plot_mollier_hx(
    data: Optional[pd.DataFrame] = None,
    pressure: float = 101325.0,
    domain_x: Tuple[float, float] = (0.0, 0.020),
    domain_y: Tuple[float, float] = (-20.0, 50.0),
    comfort_zone: Optional[Dict[str, Tuple[float, float]]] = None,
    height: int = 700,
) -> str:
    """
    Create a Mollier h,x-diagram (psychrometric chart) as self-contained HTML.

    Uses D3.js for fast SVG rendering with iso-lines for temperature, enthalpy,
    relative humidity and density, a comfort zone, and optional measured data
    points colour-coded by season with interactive hover tooltips.

    Args:
        data: Optional DataFrame with columns [timestamp, humidity, temperature].
            humidity in %, temperature in °C.
        pressure: Air pressure in Pa. Default 101325 (sea level).
        domain_x: Range of absolute humidity [kg/kg] for the x-axis.
        domain_y: Range of the y-coordinate (≈ temperature at x=0) for the y-axis.
        comfort_zone: Dict with keys "temperature", "rel_humidity", "abs_humidity",
            each a (min, max) tuple. Defaults: T=[20, 26], phi=[0.30, 0.65],
            x=[0, 0.0115]. Pass ``False`` to disable the comfort zone.
        height: Diagram height in pixels. Default 700.

    Returns:
        str: Self-contained HTML string with inline D3.js rendering.
            Can be saved to a file, used with ``streamlit.components.v1.html()``,
            or displayed in a Jupyter notebook via ``IPython.display.HTML()``.
    """
    import json

    from pyedautils._mollier import (
        get_x_y,
        rel_humidity as m_rel_humidity,
        temperature as m_temperature,
    )

    js = _load_d3_js()

    # Prepare data JSON
    data_json = "null"
    if data is not None and not data.empty:
        df = data.copy()
        df.columns = ["timestamp", "humidity", "temperature"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["humidity", "temperature"])
        if not df.empty:
            t_arr = df["temperature"].values
            phi_arr = df["humidity"].values / 100.0
            x_arr, y_arr = get_x_y(t_arr, phi_arr, pressure)
            df["season"] = df["timestamp"].apply(_get_season_fast)
            records = []
            for i in range(len(df)):
                ts = df.iloc[i]["timestamp"]
                xv, yv = float(x_arr[i]), float(y_arr[i])
                records.append({
                    "x": xv, "y": yv,
                    "season": _SEASON_LABELS_DE.get(df.iloc[i]["season"], "?"),
                    "ts": ts.strftime("%Y-%m-%d %H:%M"),
                    "temp": round(float(m_temperature(xv, yv)), 2),
                    "phi": round(float(m_rel_humidity(xv, yv, pressure) * 100), 2),
                    "xg": round(xv * 1000, 2),
                })
            data_json = json.dumps(records)

    if comfort_zone is False:
        comfort_t, comfort_phi, comfort_x = "[0,0]", "[0,0]", "[0,0]"
    else:
        cz = comfort_zone or {}
        comfort_t = json.dumps(list(cz.get("temperature", (20, 26))))
        comfort_phi = json.dumps(list(cz.get("rel_humidity", (0.30, 0.65))))
        comfort_x = json.dumps(list(cz.get("abs_humidity", (0, 0.0115))))
    domain_x_js = json.dumps(list(domain_x))
    domain_y_js = json.dumps(list(domain_y))

    import uuid
    uid = uuid.uuid4().hex[:8]
    diagram_id = f"mollier_{uid}"
    tooltip_id = f"tt_{uid}"
    plot_id = f"plot_{uid}"
    clip_id = f"clip_{uid}"

    season_colors = json.dumps(
        {v: DEFAULT_SEASON_COLORS[k] for k, v in _SEASON_LABELS_DE.items()})

    return f"""<div id="{diagram_id}" style="width:100%;background:white;"></div>
<div id="{tooltip_id}" style="position:absolute;background:rgba(255,255,255,0.9);\
border-radius:4px;padding:6px 8px;pointer-events:none;\
font-family:Tahoma,Geneva,sans-serif;font-size:11px;\
box-shadow:2px 2px 6px rgba(0,0,0,0.2);opacity:0;"></div>
<script src="https://d3js.org/d3.v5.min.js"></script>
<script>
(function() {{
{js["mollier_functions"]}
{js["coordinate_generator"]}
{js["draw_comfort"]}
  let p = {pressure};
  let domainX = {domain_x_js};
  let domainY = {domain_y_js};
  let rangeT = {comfort_t};
  let rangePhi = {comfort_phi};
  let rangeX = {comfort_x};
  let dataRecords = {data_json};
  let colorMap = {season_colors};

  let Height = {height};
  let container = document.getElementById("{diagram_id}");
  let Width = container.getBoundingClientRect().width || 900;

  let margin = {{top: 30, right: 70, bottom: 35, left: 50}};
  let width = Width - margin.left - margin.right;
  let height = Height - margin.top - margin.bottom;

  let svg = d3.select("#{diagram_id}").append("svg")
    .attr("width", Width).attr("height", Height);
  let bg = svg.append("g").attr("id", "{plot_id}");
  let clip = svg.append("defs").append("svg:clipPath")
    .attr("id", "{clip_id}").append("svg:rect")
    .attr("width", width).attr("height", height);
  let plot = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .attr("clip-path", "url(#{clip_id})");

  drawHXCoordinates(bg, Width, Height, margin, domainX, domainY, p);

  let x = d3.scaleLinear().range([0, width]).domain(domainX);
  let y = d3.scaleLinear().range([height, 0]).domain(domainY);

  let line = d3.line().x(d => x(d.x)).y(d => y(d.y));
  let pathos = createComfort(rangeT, rangePhi, rangeX, p);
  if (pathos && pathos.length > 0) {{
    plot.append("path").datum(pathos).attr("d", line)
      .attr("fill", "yellowgreen").attr("fill-opacity", 0.4)
      .attr("stroke", "yellowgreen");
  }}

  if (dataRecords && dataRecords.length > 0) {{
    let tooltip = d3.select("#{tooltip_id}");

    for (let i = dataRecords.length - 1; i > 0; i--) {{
      let j = Math.floor(Math.random() * (i + 1));
      [dataRecords[i], dataRecords[j]] = [dataRecords[j], dataRecords[i]];
    }}

    plot.selectAll("circle").data(dataRecords).enter().append("circle")
      .attr("cx", d => x(d.x)).attr("cy", d => y(d.y))
      .attr("r", 5).attr("fill", d => colorMap[d.season] || "#999")
      .attr("opacity", 0.4).attr("shape-rendering", "optimizeSpeed")
      .on("mouseover", function(d) {{
        d3.select(this).attr("r", 10).attr("opacity", 0.9);
        tooltip.style("opacity", 1)
          .style("background-color", colorMap[d.season] || "#999")
          .style("color",
            (d.season === "Winter" || d.season === "Herbst") ? "white" : "black")
          .html(d.ts + "<br>x: " + d.xg + " g/kg<br>T: " + d.temp
            + " °C<br>φ: " + d.phi + " %")
          .style("left", (d3.event.pageX + 15) + "px")
          .style("top", (d3.event.pageY - 40) + "px");
      }})
      .on("mouseout", function(d) {{
        d3.select(this).attr("r", 5).attr("opacity", 0.4);
        tooltip.style("opacity", 0);
      }});

    let legendItems = [
      {{label: "Komfortzone", color: "#9ACD32", type: "rect"}},
      {{label: "Frühling", color: colorMap["Frühling"], type: "circle"}},
      {{label: "Sommer", color: colorMap["Sommer"], type: "circle"}},
      {{label: "Herbst", color: colorMap["Herbst"], type: "circle"}},
      {{label: "Winter", color: colorMap["Winter"], type: "circle"}}
    ];
    let legend = svg.append("g")
      .attr("transform", "translate(" + (margin.left + 10) + ","
        + (margin.top + 10) + ")");
    legend.append("rect").attr("x", -5).attr("y", -5)
      .attr("width", 120).attr("height", legendItems.length * 20 + 10)
      .attr("fill", "white").attr("opacity", 0.7);
    legendItems.forEach((item, i) => {{
      let g = legend.append("g")
        .attr("transform", "translate(0," + (i * 20) + ")");
      if (item.type === "rect")
        g.append("rect").attr("width", 14).attr("height", 14)
          .attr("fill", item.color).attr("opacity", 0.7);
      else
        g.append("circle").attr("cx", 7).attr("cy", 7).attr("r", 5)
          .attr("fill", item.color).attr("opacity", 0.7);
      g.append("text").attr("x", 20).attr("y", 11).text(item.label)
        .style("font-family", "Tahoma, Geneva, sans-serif")
        .style("font-size", "12px");
    }});
  }}
}})();
</script>"""
