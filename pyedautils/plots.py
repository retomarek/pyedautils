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
            x=[0, 0.0115].
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

    cz = comfort_zone or {}
    comfort_t = json.dumps(list(cz.get("temperature", (20, 26))))
    comfort_phi = json.dumps(list(cz.get("rel_humidity", (0.30, 0.65))))
    comfort_x = json.dumps(list(cz.get("abs_humidity", (0, 0.0115))))
    domain_x_js = json.dumps(list(domain_x))
    domain_y_js = json.dumps(list(domain_y))

    season_colors = json.dumps(
        {v: DEFAULT_SEASON_COLORS[k] for k, v in _SEASON_LABELS_DE.items()})

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin: 0; padding: 0; background: white; overflow: hidden; }}
  #diagram {{ width: 100%; }}
  .tooltip {{
    position: absolute; background: rgba(255,255,255,0.9);
    border-radius: 4px; padding: 6px 8px; pointer-events: none;
    font-family: Tahoma, Geneva, sans-serif; font-size: 11px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2); opacity: 0;
  }}
</style>
</head>
<body>
<div id="diagram"></div>
<div class="tooltip" id="tt"></div>

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
  let container = document.getElementById("diagram");
  let Width = container.getBoundingClientRect().width || 900;

  let margin = {{top: 30, right: 70, bottom: 35, left: 50}};
  let width = Width - margin.left - margin.right;
  let height = Height - margin.top - margin.bottom;

  let svg = d3.select("#diagram").append("svg")
    .attr("width", Width).attr("height", Height);
  let bg = svg.append("g").attr("id", "theplot");
  let clip = svg.append("defs").append("svg:clipPath")
    .attr("id", "clip").append("svg:rect")
    .attr("width", width).attr("height", height);
  let plot = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .attr("clip-path", "url(#clip)");

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
    let tooltip = d3.select("#tt");

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
</script>
</body>
</html>"""
