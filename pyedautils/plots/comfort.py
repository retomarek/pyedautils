"""Thermal comfort and psychrometric chart plots."""

from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from pyedautils.plots._constants import DEFAULT_SEASON_COLORS, _SEASON_LABELS_DE


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
    from pyedautils.data_prep.season import get_season

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
    from pyedautils.data_prep.season import get_season

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
