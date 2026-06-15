"""Thermal comfort and psychrometric chart plots."""

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

from pyedautils.comfort import (
    ADAPTIVE_MAX_T_OA_HI,
    ADAPTIVE_MAX_T_OA_LO,
    ADAPTIVE_MIN_T_OA_HI,
    ADAPTIVE_MIN_T_OA_LO,
    MINERGIE_OVERHEATING_LIMIT_H,
    SIA180_RESIDENTIAL_OVERHEATING_LIMIT_H,
    sia180_max_temp,
    sia180_min_temp,
)
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

    # SIA 180 boundaries (adaptive curves from pyedautils.comfort —
    # single source of truth, see sia180_min_temp / sia180_max_temp).
    # Lower limit (heating setpoint): constant below 19 °C, linear to
    # 23.5 °C, constant above.
    lower_x = [min_x, ADAPTIVE_MIN_T_OA_LO, ADAPTIVE_MIN_T_OA_HI, max_x]
    fig.add_trace(go.Scatter(
        x=lower_x,
        y=sia180_min_temp(lower_x),
        mode="lines", name="Lower limit SIA 180",
        line=dict(color="#440154", width=2),
    ))
    # Upper limit active cooling: constant below 12 °C, linear to
    # 17.5 °C, constant above.
    upper_x = [min_x, ADAPTIVE_MAX_T_OA_LO, ADAPTIVE_MAX_T_OA_HI, max_x]
    fig.add_trace(go.Scatter(
        x=upper_x,
        y=sia180_max_temp(upper_x),
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
    convention: str = 'classical',
    highlight_latest: bool = True,
    highlight_color: Optional[str] = 'black',
    states: Optional[Sequence] = None,
    labels: Optional[List[str]] = None,
    comfort_label: str = 'Comfort zone',
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
        convention: Mollier coordinate convention. ``'classical'`` (default)
            normalises enthalpy per kg of dry air (Recknagel/Sprenger style,
            isotherms tilt slightly up with x). ``'glueck'`` normalises per
            kg of moist air (per the Glück book, isotherms tilt slightly down).
        highlight_latest: When ``True`` (default), the row with the most recent
            timestamp is overlaid as a circle on top of the seasonal scatter
            points. Set to ``False`` to disable.
        highlight_color: CSS colour used for the latest-row overlay. Default
            ``'black'``. Pass ``None`` to use that row's season colour (so the
            highlight is visible only by size/order, not by colour).
        states: Optional sequence of ``MoistAirState`` objects (from
            :func:`pyedautils._mollier.state`). When supplied, each state is
            drawn as a numbered point and consecutive states are connected by
            arrows — a psychrometric process chain in the spirit of
            psychrosim.com. Works alongside ``data`` (historical scatter is
            drawn underneath).
        labels: Optional sequence of strings used as point labels (default
            is the index ``"0"``, ``"1"``, …). Length must match ``states``.
        comfort_label: Legend caption for the comfort-zone polygon. Default
            ``"Comfort zone"``. Override e.g. to ``"Komfortzone"`` for German
            output or to a custom designation like ``"DIN EN 16798 Cat II"``.

    Returns:
        str: Self-contained HTML string with inline D3.js rendering.
            Can be saved to a file, used with ``streamlit.components.v1.html()``,
            or displayed in a Jupyter notebook via ``IPython.display.HTML()``.
    """
    import json

    from pyedautils._mollier import (
        _check_convention,
        get_x_y,
        rel_humidity as m_rel_humidity,
        temperature as m_temperature,
    )

    convention = _check_convention(convention)
    js = _load_d3_js()

    # Prepare data JSON
    data_json = "null"
    current_json = "null"
    if data is not None and not data.empty:
        df = data.copy()
        df.columns = ["timestamp", "humidity", "temperature"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["humidity", "temperature"])
        if not df.empty:
            t_arr = df["temperature"].values
            phi_arr = df["humidity"].values / 100.0
            x_arr, y_arr = get_x_y(t_arr, phi_arr, pressure, convention=convention)
            df["season"] = df["timestamp"].apply(_get_season_fast)
            records = []
            for i in range(len(df)):
                ts = df.iloc[i]["timestamp"]
                xv, yv = float(x_arr[i]), float(y_arr[i])
                records.append({
                    "x": xv, "y": yv,
                    "season": _SEASON_LABELS_DE.get(df.iloc[i]["season"], "?"),
                    "ts": ts.strftime("%Y-%m-%d %H:%M"),
                    "temp": round(float(m_temperature(xv, yv, convention=convention)), 2),
                    "phi": round(float(m_rel_humidity(
                        xv, yv, pressure, convention=convention) * 100), 2),
                    "xg": round(xv * 1000, 2),
                })
            data_json = json.dumps(records)
            if highlight_latest:
                # Find the record with the most recent timestamp (don't assume
                # the input DataFrame is sorted).
                idx_latest = df["timestamp"].values.argmax()
                current_json = json.dumps(records[idx_latest])

    # Process-chain points. Each record carries both a short integer ``number``
    # (1-based, drawn inside the chart circle) and a free-form ``label``
    # (shown in the legend). The legend is rendered only when the user
    # supplied explicit labels — otherwise the labels would just repeat the
    # numbers.
    states_json = "null"
    state_legend_js = "false"
    if states is not None and len(states) > 0:
        if labels is not None and len(labels) != len(states):
            raise ValueError(
                f"labels length ({len(labels)}) must match states length "
                f"({len(states)})"
            )
        state_records = []
        for i, s in enumerate(states):
            number = str(i + 1)
            lab = labels[i] if labels is not None else number
            state_records.append({
                "x": float(s.x),
                "y": float(s.y),
                "number": number,
                "label": str(lab),
                "t": round(float(s.t), 2),
                "phi": round(float(s.phi) * 100, 2),
                "xg": round(float(s.x) * 1000, 2),
                "h": round(float(s.h), 2),
                "t_wb": round(float(s.t_wb), 2),
                "t_dp": round(float(s.t_dp), 2),
            })
        states_json = json.dumps(state_records)
        state_legend_js = "true" if labels is not None else "false"

    if comfort_zone is False:
        comfort_t, comfort_phi, comfort_x = "[0,0]", "[0,0]", "[0,0]"
        has_comfort_js = "false"
    else:
        cz = comfort_zone or {}
        comfort_t = json.dumps(list(cz.get("temperature", (20, 26))))
        comfort_phi = json.dumps(list(cz.get("rel_humidity", (0.30, 0.65))))
        comfort_x = json.dumps(list(cz.get("abs_humidity", (0, 0.0115))))
        has_comfort_js = "true"
    domain_x_js = json.dumps(list(domain_x))
    domain_y_js = json.dumps(list(domain_y))
    highlight_color_js = json.dumps(highlight_color)
    comfort_label_js = json.dumps(comfort_label)

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
  let convention = "{convention}";
  let mollier = createMollier(convention);
  let p = {pressure};
  let domainX = {domain_x_js};
  let domainY = {domain_y_js};
  let rangeT = {comfort_t};
  let rangePhi = {comfort_phi};
  let rangeX = {comfort_x};
  let dataRecords = {data_json};
  let currentRecord = {current_json};
  let statePoints = {states_json};
  let showStateLegend = {state_legend_js};
  let hasComfort = {has_comfort_js};
  let comfortLabel = {comfort_label_js};
  let highlightColor = {highlight_color_js};
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
  let defs = svg.append("defs");
  defs.append("svg:clipPath")
    .attr("id", "{clip_id}").append("svg:rect")
    .attr("width", width).attr("height", height);
  // Arrow head for process-chain segments.
  defs.append("marker")
    .attr("id", "arrow-{uid}")
    .attr("viewBox", "0 0 10 10")
    .attr("refX", 9).attr("refY", 5)
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", "#444");
  let plot = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .attr("clip-path", "url(#{clip_id})");

  drawHXCoordinates(bg, Width, Height, margin, domainX, domainY, p, mollier);

  let x = d3.scaleLinear().range([0, width]).domain(domainX);
  let y = d3.scaleLinear().range([height, 0]).domain(domainY);

  let line = d3.line().x(d => x(d.x)).y(d => y(d.y));
  let pathos = hasComfort ? createComfort(rangeT, rangePhi, rangeX, p, mollier) : null;
  if (hasComfort && pathos && pathos.length > 0) {{
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

    // Latest record drawn on top of all seasonal points.
    if (currentRecord) {{
      let latestFill = highlightColor !== null
        ? highlightColor
        : (colorMap[currentRecord.season] || "#999");
      plot.append("g").attr("id", "current-point")
        .append("circle")
          .datum(currentRecord)
          .attr("cx", x(currentRecord.x)).attr("cy", y(currentRecord.y))
          .attr("r", 6).attr("fill", latestFill)
          .attr("stroke", "white").attr("stroke-width", 1.5)
          .style("cursor", "pointer")
          .on("mouseover", function(d) {{
            d3.select(this).attr("r", 10);
            tooltip.style("opacity", 1)
              .style("background-color", latestFill)
              .style("color", "white")
              .html("Latest: " + d.ts
                + "<br>x: " + d.xg + " g/kg<br>T: " + d.temp
                + " °C<br>φ: " + d.phi + " %")
              .style("left", (d3.event.pageX + 15) + "px")
              .style("top", (d3.event.pageY - 40) + "px");
          }})
          .on("mouseout", function() {{
            d3.select(this).attr("r", 6);
            tooltip.style("opacity", 0);
          }});
    }}

  }}

  // Process chain: numbered state points joined by arrows.
  if (statePoints && statePoints.length > 0) {{
    let tooltip = d3.select("#{tooltip_id}");
    let chain = plot.append("g").attr("id", "process-chain");

    for (let i = 1; i < statePoints.length; i++) {{
      let a = statePoints[i - 1];
      let b = statePoints[i];
      chain.append("line")
        .attr("x1", x(a.x)).attr("y1", y(a.y))
        .attr("x2", x(b.x)).attr("y2", y(b.y))
        .attr("stroke", "#444").attr("stroke-width", 2)
        .attr("marker-end", "url(#arrow-{uid})");
    }}

    let nodes = chain.selectAll("g.state").data(statePoints).enter()
      .append("g").attr("class", "state")
      .attr("transform", d => "translate(" + x(d.x) + "," + y(d.y) + ")")
      .style("cursor", "pointer")
      .on("mouseover", function(d) {{
        tooltip.style("opacity", 1)
          .style("background-color", "#222")
          .style("color", "white")
          .html(d.label
            + "<br>T: " + d.t + " °C"
            + "<br>φ: " + d.phi + " %"
            + "<br>x: " + d.xg + " g/kg"
            + "<br>h: " + d.h + " kJ/kg")
          .style("left", (d3.event.pageX + 15) + "px")
          .style("top", (d3.event.pageY - 40) + "px");
      }})
      .on("mouseout", function() {{ tooltip.style("opacity", 0); }});

    nodes.append("circle")
      .attr("r", 11).attr("fill", "#222")
      .attr("stroke", "white").attr("stroke-width", 2);
    nodes.append("text")
      .attr("text-anchor", "middle").attr("dy", "0.35em")
      .attr("fill", "white").attr("font-size", "11px")
      .attr("font-weight", "bold")
      .text(d => d.number);
  }}

  // ----- Combined legend (bottom-right) ----------------------------------
  let legendItems = [];
  if (statePoints && statePoints.length > 0 && showStateLegend) {{
    for (let sp of statePoints) {{
      legendItems.push({{type: "process", number: sp.number, label: sp.label}});
    }}
  }}
  if (hasComfort && pathos && pathos.length > 0) {{
    legendItems.push({{type: "rect", color: "#9ACD32", label: comfortLabel}});
  }}
  if (dataRecords && dataRecords.length > 0) {{
    legendItems.push({{type: "circle", color: colorMap["Frühling"], label: "Frühling"}});
    legendItems.push({{type: "circle", color: colorMap["Sommer"], label: "Sommer"}});
    legendItems.push({{type: "circle", color: colorMap["Herbst"], label: "Herbst"}});
    legendItems.push({{type: "circle", color: colorMap["Winter"], label: "Winter"}});
  }}

  if (legendItems.length > 0) {{
    let rowH = 20;
    let legendH = legendItems.length * rowH + 10;
    let legendW = 150;
    // Anchor bottom-right of the plot area, but keep enough margin from the
    // right and bottom edges so that the enthalpy-axis labels (which sit at
    // plot-local x = width-20 and y = height-20) stay visible.
    let lx = margin.left + width - legendW - 50;
    let ly = margin.top + height - legendH - 35;
    let legendG = svg.append("g")
      .attr("transform", "translate(" + lx + "," + ly + ")");
    legendG.append("rect")
      .attr("x", -6).attr("y", -6)
      .attr("width", legendW + 12).attr("height", legendH + 6)
      .attr("fill", "white").attr("stroke", "#bbb").attr("rx", 3);
    legendItems.forEach((item, i) => {{
      let g = legendG.append("g")
        .attr("transform", "translate(0," + (i * rowH + 5) + ")");
      if (item.type === "process") {{
        g.append("circle")
          .attr("cx", 10).attr("cy", 8).attr("r", 10)
          .attr("fill", "#222").attr("stroke", "white").attr("stroke-width", 1.5);
        g.append("text")
          .attr("x", 10).attr("y", 8).attr("text-anchor", "middle").attr("dy", "0.35em")
          .attr("fill", "white").attr("font-size", "10px").attr("font-weight", "bold")
          .text(item.number);
        g.append("text").attr("x", 26).attr("y", 12)
          .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
          .text(item.label);
      }} else if (item.type === "rect") {{
        g.append("rect").attr("x", 2).attr("y", 1).attr("width", 14).attr("height", 14)
          .attr("fill", item.color).attr("opacity", 0.7);
        g.append("text").attr("x", 22).attr("y", 12)
          .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
          .text(item.label);
      }} else {{
        g.append("circle").attr("cx", 9).attr("cy", 8).attr("r", 5)
          .attr("fill", item.color).attr("opacity", 0.7);
        g.append("text").attr("x", 22).attr("y", 12)
          .style("font-family", "Tahoma, Geneva, sans-serif").style("font-size", "12px")
          .text(item.label);
      }}
    }});
  }}
}})();
</script>"""


def plot_comfort_donuts(
    data: pd.DataFrame,
    temp_range: Tuple[float, float] = (20.0, 26.0),
    hum_range: Tuple[float, float] = (30.0, 65.0),
    title: Optional[str] = None,
    temp_colors: Tuple[str, str, str] = ("#3498DB", "#2ECC71", "#E74C3C"),
    hum_colors: Tuple[str, str, str] = ("#F39C12", "#2ECC71", "#3498DB"),
    labels_de: bool = False,
) -> go.Figure:
    """Two donut charts showing time spent below / within / above comfort.

    The left donut splits temperature into *too cold* / *comfort* / *too
    warm*, the right donut splits humidity into *too dry* / *comfort* /
    *too humid*, based on the given comfort ranges.

    Args:
        data: DataFrame with columns ``temperature`` [°C] and
            ``humidity`` [%rH] (extra columns are ignored).
        temp_range: ``(min, max)`` comfort temperature band [°C].
        hum_range: ``(min, max)`` comfort humidity band [%rH].
        title: Overall figure title. Default *None*.
        temp_colors: Colors for (cold, comfort, warm).
        hum_colors: Colors for (dry, comfort, humid).
        labels_de: Use German slice labels. Default *False* (English).

    Returns:
        go.Figure: Plotly figure with two donut subplots.
    """
    from plotly.subplots import make_subplots

    df = data.copy()
    t = pd.to_numeric(df["temperature"], errors="coerce").dropna()
    h = pd.to_numeric(df["humidity"], errors="coerce").dropna()

    t_lo, t_hi = temp_range
    h_lo, h_hi = hum_range
    temp_vals = [int((t < t_lo).sum()), int(((t >= t_lo) & (t <= t_hi)).sum()),
                 int((t > t_hi).sum())]
    hum_vals = [int((h < h_lo).sum()), int(((h >= h_lo) & (h <= h_hi)).sum()),
                int((h > h_hi).sum())]

    if labels_de:
        temp_labels = [f"< {t_lo:g} °C", "Komfort", f"> {t_hi:g} °C"]
        hum_labels = ["Zu trocken", "Komfort", "Zu feucht"]
        sub_titles = ("Temperaturverteilung", "Feuchtigkeitsverteilung")
    else:
        temp_labels = [f"< {t_lo:g} °C", "Comfort", f"> {t_hi:g} °C"]
        hum_labels = ["Too dry", "Comfort", "Too humid"]
        sub_titles = ("Temperature distribution", "Humidity distribution")

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=sub_titles,
    )
    fig.add_trace(go.Pie(
        labels=temp_labels, values=temp_vals, hole=0.5,
        marker_colors=list(temp_colors), name="Temperature",
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=hum_labels, values=hum_vals, hole=0.5,
        marker_colors=list(hum_colors), name="Humidity",
    ), row=1, col=2)

    fig.update_layout(
        title_text=f"<b>{title}</b>" if title else None,
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
    )
    return fig


def plot_overheating_bar(
    data: pd.DataFrame,
    label_col: str = "label",
    value_col: str = "hours",
    limits: Tuple[float, float] = (
        MINERGIE_OVERHEATING_LIMIT_H,
        SIA180_RESIDENTIAL_OVERHEATING_LIMIT_H,
    ),
    limit_labels: Tuple[str, str] = ("Minergie (100 h)", "SIA 180 (400 h)"),
    title: str = "Overheating hours per room",
    xlab: str = "Overheating hours [h]",
) -> go.Figure:
    """Horizontal bar chart of overheating hours per room with limit lines.

    Bars are sorted ascending (worst room on top) and colored green /
    orange / red depending on whether they fall below the lower limit,
    between the limits, or above the upper limit. Two vertical reference
    lines mark the limits (e.g. Minergie 100 h and SIA 180 400 h).

    Args:
        data: DataFrame with one row per room.
        label_col: Column with the room label. Default ``"label"``.
        value_col: Column with the overheating hours. Default ``"hours"``.
        limits: ``(lower, upper)`` reference limits [h].
        limit_labels: Annotations for the two reference lines.
        title: Plot title.
        xlab: X-axis label.

    Returns:
        go.Figure: Plotly horizontal bar figure.
    """
    df = data[[label_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    df = df.sort_values(value_col, ascending=True).reset_index(drop=True)

    lo, hi = limits

    def _color(v: float) -> str:
        if v > hi:
            return "#ef4444"   # red
        if v > lo:
            return "#f59e0b"   # orange
        return "#22c55e"       # green

    colors = [_color(v) for v in df[value_col]]

    fig = go.Figure(go.Bar(
        x=df[value_col], y=df[label_col],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f} h" for v in df[value_col]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f} h<extra></extra>",
    ))

    for lim, lab, col in zip(limits, limit_labels, ("#16a34a", "#dc2626")):
        fig.add_vline(
            x=lim, line=dict(color=col, width=1.5, dash="dash"),
            annotation_text=lab, annotation_position="top",
        )

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        xaxis_title=xlab,
        height=max(300, 40 * len(df) + 120),
        showlegend=False,
        margin=dict(l=10, r=40, t=60, b=10),
    )
    return fig
