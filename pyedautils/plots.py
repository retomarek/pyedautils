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
