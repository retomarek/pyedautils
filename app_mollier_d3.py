"""Streamlit + D3.js Mollier h,x-diagram — original D3 rendering, blazing fast."""

import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from importlib import resources
from pathlib import Path

from pyedautils._mollier import get_x_y, temperature as m_temperature, rel_humidity as m_rel_humidity

# ======================================================================
# Load JS sources once
# ======================================================================
_JS_DIR = Path(r"C:\Repositories\gitlab\IGE\LAES\lcm-fieldtest-shiny-influxdb\app\R\d3MollierGraph")
_JS_ORIG = Path(r"C:\Repositories\github\hslu-ige-laes\d3-mollierhx\src")


def _read(path):
    return path.read_text(encoding="utf-8")


JS_MOLLIER_FUNCTIONS = _read(_JS_DIR / "mollier_functions.js")
JS_COORDINATE_GEN = _read(_JS_DIR / "CoordinateGenerator.js")
JS_DRAW_COMFORT = _read(_JS_DIR / "drawComfort.js")


def _get_season_fast(dt):
    m = dt.month
    if m in (12, 1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Frühling"
    elif m in (6, 7, 8):
        return "Sommer"
    return "Herbst"


def build_html(domain_x, domain_y, pressure, comfort, data_df=None, height=700):
    """Build a self-contained HTML string with inline D3 Mollier diagram."""

    # Prepare data JSON
    data_json = "null"
    if data_df is not None and not data_df.empty:
        df = data_df.copy()
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
                records.append({
                    "x": float(x_arr[i]),
                    "y": float(y_arr[i]),
                    "season": df.iloc[i]["season"],
                    "ts": ts.strftime("%Y-%m-%d %H:%M"),
                    "temp": round(float(m_temperature(x_arr[i], y_arr[i])), 2),
                    "phi": round(float(m_rel_humidity(x_arr[i], y_arr[i], pressure) * 100), 2),
                    "xg": round(float(x_arr[i] * 1000), 2),
                })
            data_json = json.dumps(records)

    comfort_t = json.dumps(list(comfort.get("temperature", (20, 26))))
    comfort_phi = json.dumps(list(comfort.get("rel_humidity", (0.30, 0.65))))
    comfort_x = json.dumps(list(comfort.get("abs_humidity", (0, 0.0115))))
    domain_x_js = json.dumps(list(domain_x))
    domain_y_js = json.dumps(list(domain_y))

    html = f"""<!DOCTYPE html>
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
// === Mollier Functions ===
{JS_MOLLIER_FUNCTIONS}

// === Coordinate Generator ===
{JS_COORDINATE_GEN}

// === Draw Comfort ===
{JS_DRAW_COMFORT}

// === Main ===
(function() {{
  let p = {pressure};
  let domainX = {domain_x_js};
  let domainY = {domain_y_js};
  let rangeT = {comfort_t};
  let rangePhi = {comfort_phi};
  let rangeX = {comfort_x};
  let dataRecords = {data_json};

  let Height = {height};
  let container = document.getElementById("diagram");
  let Width = container.getBoundingClientRect().width || 900;

  let margin = {{top: 30, right: 70, bottom: 35, left: 50}};
  let width = Width - margin.left - margin.right;
  let height = Height - margin.top - margin.bottom;

  // Main SVG
  let svg = d3.select("#diagram").append("svg")
    .attr("width", Width).attr("height", Height);

  let bg = svg.append("g").attr("id", "theplot");

  let clip = svg.append("defs").append("svg:clipPath")
    .attr("id", "clip").append("svg:rect")
    .attr("width", width).attr("height", height);

  let plot = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .attr("clip-path", "url(#clip)");

  // Draw coordinate grid
  drawHXCoordinates(bg, Width, Height, margin, domainX, domainY, p);

  // Scales for data overlay
  let x = d3.scaleLinear().range([0, width]).domain(domainX);
  let y = d3.scaleLinear().range([height, 0]).domain(domainY);

  // Draw comfort zone
  let line = d3.line().x(d => x(d.x)).y(d => y(d.y));
  let pathos = createComfort(rangeT, rangePhi, rangeX, p);
  if (pathos && pathos.length > 0) {{
    plot.append("path")
      .datum(pathos)
      .attr("d", line)
      .attr("fill", "yellowgreen")
      .attr("fill-opacity", 0.4)
      .attr("stroke", "yellowgreen");
  }}

  // Draw data points
  if (dataRecords && dataRecords.length > 0) {{
    let colorMap = {{
      "Winter": "#365c8d",
      "Frühling": "#2db27d",
      "Sommer": "#febc2b",
      "Herbst": "#824b04"
    }};

    let tooltip = d3.select("#tt");

    // Shuffle for better visual layering
    for (let i = dataRecords.length - 1; i > 0; i--) {{
      let j = Math.floor(Math.random() * (i + 1));
      [dataRecords[i], dataRecords[j]] = [dataRecords[j], dataRecords[i]];
    }}

    plot.selectAll("circle")
      .data(dataRecords)
      .enter().append("circle")
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .attr("r", 5)
        .attr("fill", d => colorMap[d.season] || "#999")
        .attr("opacity", 0.4)
        .attr("shape-rendering", "optimizeSpeed")
        .on("mouseover", function(d) {{
          d3.select(this).attr("r", 10).attr("opacity", 0.9);
          tooltip.style("opacity", 1)
            .style("background-color", colorMap[d.season] || "#999")
            .style("color", (d.season === "Winter" || d.season === "Herbst") ? "white" : "black")
            .html(d.ts + "<br>x: " + d.xg + " g/kg<br>T: " + d.temp + " °C<br>φ: " + d.phi + " %")
            .style("left", (d3.event.pageX + 15) + "px")
            .style("top", (d3.event.pageY - 40) + "px");
        }})
        .on("mouseout", function(d) {{
          d3.select(this).attr("r", 5).attr("opacity", 0.4);
          tooltip.style("opacity", 0);
        }});

    // Legend
    let legendItems = [
      {{label: "Komfortzone", color: "#9ACD32", type: "rect"}},
      {{label: "Frühling", color: "#2db27d", type: "circle"}},
      {{label: "Sommer", color: "#febc2b", type: "circle"}},
      {{label: "Herbst", color: "#824b04", type: "circle"}},
      {{label: "Winter", color: "#365c8d", type: "circle"}}
    ];

    let legend = svg.append("g")
      .attr("transform", "translate(" + (margin.left + 10) + "," + (margin.top + 10) + ")");

    legend.append("rect").attr("x", -5).attr("y", -5)
      .attr("width", 120).attr("height", legendItems.length * 20 + 10)
      .attr("fill", "white").attr("opacity", 0.7);

    legendItems.forEach((item, i) => {{
      let g = legend.append("g").attr("transform", "translate(0," + (i * 20) + ")");
      if (item.type === "rect") {{
        g.append("rect").attr("width", 14).attr("height", 14)
          .attr("fill", item.color).attr("opacity", 0.7);
      }} else {{
        g.append("circle").attr("cx", 7).attr("cy", 7).attr("r", 5)
          .attr("fill", item.color).attr("opacity", 0.7);
      }}
      g.append("text").attr("x", 20).attr("y", 11)
        .text(item.label)
        .style("font-family", "Tahoma, Geneva, sans-serif")
        .style("font-size", "12px");
    }});
  }}
}})();
</script>
</body>
</html>"""
    return html


# ======================================================================
# Streamlit UI
# ======================================================================
st.set_page_config(page_title="Mollier h,x-Diagram (D3)", layout="wide")

st.markdown(
    "<h1 style='font-family: monospace; font-weight: normal;'>Mollier-hx-diagram</h1>",
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.markdown("## Control Panel")

st.sidebar.markdown("##### x-range [g/kg]")
x_min, x_max = st.sidebar.slider(
    "x-range", min_value=0.0, max_value=35.0, value=(0.0, 20.0),
    step=0.5, format="%.1f", label_visibility="collapsed",
)

st.sidebar.markdown("##### y-range [°C]")
y_min, y_max = st.sidebar.slider(
    "y-range", min_value=-40.0, max_value=80.0, value=(-20.0, 50.0),
    step=1.0, format="%.0f", label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("##### Comfort Zone")

st.sidebar.markdown("Temperature [°C]")
comfort_t_min, comfort_t_max = st.sidebar.slider(
    "temperature-range", min_value=-20.0, max_value=50.0, value=(20.0, 26.0),
    step=0.5, format="%.1f", label_visibility="collapsed",
)

st.sidebar.markdown("Rel. Humidity [%]")
comfort_phi_min, comfort_phi_max = st.sidebar.slider(
    "rel-humidity-range", min_value=0, max_value=100, value=(30, 65),
    step=1, label_visibility="collapsed",
)

st.sidebar.markdown("Abs. Humidity [g/kg]")
comfort_x_min, comfort_x_max = st.sidebar.slider(
    "abs-humidity-range", min_value=0.0, max_value=35.0, value=(0.0, 11.5),
    step=0.5, format="%.1f", label_visibility="collapsed",
)

st.sidebar.markdown("---")

pressure = st.sidebar.number_input(
    "Pressure [Pa]", min_value=50000, max_value=120000, value=101325, step=100,
)

st.sidebar.markdown("---")
st.sidebar.markdown("##### Data")

use_sample = st.sidebar.checkbox("Load sample data", value=False)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (timestamp, humidity, temperature)", type=["csv"],
)

chart_height = st.sidebar.slider(
    "Diagram height [px]", min_value=400, max_value=1200, value=700, step=50,
)

# --- Load data ---
data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    data_path = resources.files("pyedautils") / "data" / "mollier_sample.csv"
    data = pd.read_csv(data_path)

# --- Build & render ---
domain_x = (x_min / 1000.0, x_max / 1000.0)
domain_y = (y_min, y_max)

html = build_html(
    domain_x=domain_x,
    domain_y=domain_y,
    pressure=float(pressure),
    comfort={
        "temperature": (comfort_t_min, comfort_t_max),
        "rel_humidity": (comfort_phi_min / 100.0, comfort_phi_max / 100.0),
        "abs_humidity": (comfort_x_min / 1000.0, comfort_x_max / 1000.0),
    },
    data_df=data,
    height=chart_height,
)

components.html(html, height=chart_height + 10, scrolling=False)
