"""Streamlit + ECharts Mollier h,x-diagram — fast canvas rendering."""

import math

import numpy as np
import pandas as pd
import streamlit as st
from importlib import resources
from streamlit_echarts import st_echarts

from pyedautils._mollier import (
    create_comfort,
    density as m_density,
    enthalpy as m_enthalpy,
    get_x_y,
    get_x_y_tx,
    rel_humidity as m_rel_humidity,
    temperature as m_temperature,
    x_hy,
    y_hx,
    y_rhox,
)

# -- Colors from rShiny d3MollierGraph --
COLORS = {
    "temperature": "#63c1ff",
    "density": "#888888",
    "rel_humidity": "#555555",
    "enthalpy": "#CCCCCC",
    "comfort_fill": "rgba(154,205,50,0.4)",
    "comfort_stroke": "yellowgreen",
}
SEASON_COLORS = {
    "Winter": "#365c8d",
    "Spring": "#2db27d",
    "Summer": "#febc2b",
    "Fall": "#824b04",
}


def _nice_ticks(vmin, vmax, n):
    rng = vmax - vmin
    if rng == 0:
        return [vmin]
    step = rng / n
    mag = 10 ** math.floor(math.log10(step))
    residual = step / mag
    if residual <= 1.5:
        nice_step = mag
    elif residual <= 3:
        nice_step = 2 * mag
    elif residual <= 7:
        nice_step = 5 * mag
    else:
        nice_step = 10 * mag
    start = math.ceil(vmin / nice_step) * nice_step
    ticks = []
    v = start
    while v <= vmax + nice_step * 0.01:
        ticks.append(round(v, 10))
        v += nice_step
    return ticks


def _get_season(dt):
    m = dt.month
    if m in (3, 4, 5):
        return "Spring"
    elif m in (6, 7, 8):
        return "Summer"
    elif m in (9, 10, 11):
        return "Fall"
    return "Winter"


def build_echarts_option(domain_x, domain_y, pressure, comfort_zone, data, num_points=100):
    """Build a complete ECharts option dict for the Mollier diagram."""
    dx = (domain_x[1] - domain_x[0]) / num_points

    corners = [
        (domain_x[0], domain_y[0]), (domain_x[1], domain_y[0]),
        (domain_x[0], domain_y[1]), (domain_x[1], domain_y[1]),
    ]
    corner_t = [m_temperature(cx, cy) for cx, cy in corners]
    domain_t = (min(corner_t), max(corner_t))
    dt = (domain_t[1] - domain_t[0]) / num_points

    series = []
    # helper to make x-axis in g/kg for display
    def xg(v):
        return round(v * 1000, 4)

    # ---- 1. Temperature iso-lines ----
    for t_val in _nice_ticks(domain_t[0], domain_t[1], 40):
        pts = []
        xv = domain_x[0]
        while xv <= domain_x[1] + dx * 0.5:
            _, yv = get_x_y_tx(t_val, xv, pressure)
            pts.append([xg(xv), round(yv, 4)])
            xv += dx
        series.append({
            "type": "line", "data": pts, "showSymbol": False,
            "lineStyle": {"color": COLORS["temperature"], "width": 1},
            "silent": True, "animation": False, "z": 1,
        })

    # ---- 2. Density iso-lines ----
    corner_rho = [m_density(cx, cy, pressure) for cx, cy in corners]
    rho_ticks = _nice_ticks(min(corner_rho), max(corner_rho), 8)
    dim_x = domain_x[1] - domain_x[0]
    label_x_rho = domain_x[0] + 0.03 * dim_x
    for rho_val in rho_ticks:
        pts = []
        xv = domain_x[0]
        while xv <= domain_x[1] + dx * 0.5:
            pts.append([xg(xv), round(y_rhox(rho_val, xv, pressure), 4)])
            xv += dx
        series.append({
            "type": "line", "data": pts, "showSymbol": False,
            "lineStyle": {"color": COLORS["density"], "width": 1},
            "silent": True, "animation": False, "z": 2,
        })

    # ---- 3. Enthalpy iso-lines (before phi so cover hides them) ----
    corner_h = [m_enthalpy(cx, cy) for cx, cy in corners]
    h_ticks = _nice_ticks(min(corner_h), max(corner_h), 20)
    for h_val in h_ticks:
        x0, y0 = domain_x[0], y_hx(h_val, domain_x[0])
        x1, y1 = x_hy(h_val, domain_y[0]), domain_y[0]
        if y0 > domain_y[1]:
            x0, y0 = x_hy(h_val, domain_y[1]), domain_y[1]
        if x1 > domain_x[1]:
            x1, y1 = domain_x[1], y_hx(h_val, domain_x[1])
        series.append({
            "type": "line", "data": [[xg(x0), round(y0, 4)], [xg(x1), round(y1, 4)]],
            "showSymbol": False,
            "lineStyle": {"color": COLORS["enthalpy"], "width": 1},
            "silent": True, "animation": False, "z": 3,
        })

    # ---- 4. Relative humidity iso-lines ----
    corner_phi = [m_rel_humidity(cx, cy, pressure) for cx, cy in corners]
    phi_max = min(max(corner_phi), 1.0)
    if corner_phi[1] < 1 and m_rel_humidity(
            domain_x[0] + dim_x * 0.99, domain_y[0], pressure) > corner_phi[1]:
        phi_max = 1.0
    phi_ticks = _nice_ticks(min(corner_phi), phi_max, 10)

    phi_sat_pts = []
    for phi_val in phi_ticks:
        pts = []
        t_sweep = domain_t[0]
        while t_sweep <= domain_t[1] + dt * 0.5:
            xv, yv = get_x_y(t_sweep, phi_val, pressure)
            if xv > domain_x[1]:
                break
            pts.append([xg(xv), round(yv, 4)])
            t_sweep += dt
        if pts:
            series.append({
                "type": "line", "data": pts, "showSymbol": False,
                "lineStyle": {"color": COLORS["rel_humidity"], "width": 1.5},
                "silent": True, "animation": False, "z": 4,
            })
            if phi_val == phi_ticks[-1]:
                phi_sat_pts = list(pts)

    # ---- 5. Saturation cover (white polygon) ----
    if phi_sat_pts:
        dim_y = domain_y[1] - domain_y[0]
        cover = list(phi_sat_pts) + [
            [xg(domain_x[1] + 0.1 * dim_x), phi_sat_pts[-1][1]],
            [xg(domain_x[1] + 0.1 * dim_x), round(domain_y[0] - 0.1 * dim_y, 4)],
            [phi_sat_pts[0][0], round(domain_y[0] - 0.1 * dim_y, 4)],
            list(phi_sat_pts[0]),
        ]
        series.append({
            "type": "line", "data": cover, "showSymbol": False,
            "lineStyle": {"color": "black", "width": 1.5},
            "areaStyle": {"color": "white", "opacity": 1},
            "silent": True, "animation": False, "z": 5,
        })

    # ---- 6. Comfort zone ----
    cz = comfort_zone or {}
    polygon = create_comfort(
        cz.get("temperature", (20, 26)),
        cz.get("rel_humidity", (0.30, 0.65)),
        cz.get("abs_humidity", (0, 0.0115)),
        pressure,
    )
    if polygon:
        pts = [[xg(px), round(py, 4)] for px, py in polygon]
        series.append({
            "type": "line", "data": pts, "showSymbol": False,
            "lineStyle": {"color": COLORS["comfort_stroke"], "width": 1},
            "areaStyle": {"color": COLORS["comfort_fill"]},
            "silent": True, "animation": False, "z": 6,
        })

    # ---- 7. Data points ----
    legend_data = []
    if data is not None and not data.empty:
        df = data.copy()
        df.columns = ["timestamp", "humidity", "temperature"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["humidity", "temperature"])
        if not df.empty:
            t_arr = df["temperature"].values
            phi_arr = df["humidity"].values / 100.0
            x_arr, y_arr = get_x_y(t_arr, phi_arr, pressure)
            df["x_coord"] = x_arr
            df["y_coord"] = y_arr
            df["season"] = df["timestamp"].apply(_get_season)

            for season_name, color in SEASON_COLORS.items():
                subset = df[df["season"] == season_name]
                if subset.empty:
                    continue
                legend_data.append(season_name)
                pts = []
                for _, row in subset.iterrows():
                    xv, yv = row["x_coord"], row["y_coord"]
                    tv = m_temperature(xv, yv)
                    phiv = m_rel_humidity(xv, yv, pressure) * 100
                    ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
                    pts.append({
                        "value": [xg(xv), round(yv, 4)],
                        "ts": ts, "temp": round(tv, 2),
                        "phi": round(phiv, 2), "xg": round(xv * 1000, 2),
                    })
                series.append({
                    "type": "scatter", "name": season_name, "data": pts,
                    "symbolSize": 6,
                    "itemStyle": {"color": color, "opacity": 0.4},
                    "animation": False, "z": 10,
                })

    # ---- Annotations (markPoint on a dummy series) ----
    # Density labels
    rho_labels = []
    for rho_val in rho_ticks:
        ly = y_rhox(rho_val, label_x_rho, pressure)
        if domain_y[0] <= ly <= domain_y[1]:
            rho_labels.append({
                "coord": [xg(label_x_rho), round(ly, 4)],
                "value": f"{rho_val:.2f}",
                "itemStyle": {"color": "transparent"},
                "label": {"show": True, "formatter": f"{rho_val:.2f}",
                           "color": COLORS["density"], "fontSize": 10,
                           "backgroundColor": "rgba(255,255,255,0.7)", "padding": 2},
                "symbol": "none",
            })

    # Enthalpy labels
    h_threshold = m_enthalpy(domain_x[1] - 0.03 * dim_x, domain_y[0])
    h_labels = []
    for h_val in h_ticks:
        if h_val < h_threshold:
            lx = x_hy(h_val, domain_y[0])
            ly = domain_y[0]
        else:
            lx = domain_x[1] - 0.03 * dim_x
            ly = y_hx(h_val, lx)
        if domain_x[0] <= lx <= domain_x[1] and domain_y[0] <= ly <= domain_y[1]:
            h_labels.append({
                "coord": [xg(lx), round(ly, 4)],
                "value": f"{h_val:.0f}",
                "itemStyle": {"color": "transparent"},
                "label": {"show": True, "formatter": f"{h_val:.0f}",
                           "color": COLORS["enthalpy"], "fontSize": 10,
                           "backgroundColor": "rgba(255,255,255,0.7)", "padding": 2},
                "symbol": "none",
            })

    # Phi labels
    phi_labels = []
    for phi_val in phi_ticks:
        # place at the last point of each line (near right/top edge)
        pts = []
        t_sweep = domain_t[0]
        while t_sweep <= domain_t[1] + dt * 0.5:
            xv, yv = get_x_y(t_sweep, phi_val, pressure)
            if xv > domain_x[1]:
                break
            pts.append((xv, yv))
            t_sweep += dt
        if pts:
            lx, ly = pts[-1]
            if domain_x[0] <= lx <= domain_x[1] and domain_y[0] <= ly <= domain_y[1]:
                phi_labels.append({
                    "coord": [xg(lx), round(ly, 4)],
                    "value": f"{phi_val * 100:.0f} %",
                    "itemStyle": {"color": "transparent"},
                    "label": {"show": True, "formatter": f"{phi_val * 100:.0f} %",
                               "color": COLORS["rel_humidity"], "fontSize": 10,
                               "backgroundColor": "rgba(255,255,255,0.7)", "padding": 2},
                    "symbol": "none",
                })

    all_labels = rho_labels + h_labels + phi_labels
    if all_labels:
        series.append({
            "type": "scatter", "data": [],
            "markPoint": {"data": all_labels, "animation": False},
            "silent": True, "animation": False, "z": 20,
        })

    # ---- Build option ----
    option = {
        "animation": False,
        "grid": {"left": 60, "right": 70, "top": 40, "bottom": 50, "containLabel": False},
        "xAxis": {
            "type": "value",
            "name": "Absolute Humidity [g/kg]",
            "nameLocation": "middle", "nameGap": 30,
            "nameTextStyle": {"color": "black", "fontSize": 14},
            "min": xg(domain_x[0]), "max": xg(domain_x[1]),
            "axisLine": {"show": True, "lineStyle": {"color": "black"}},
            "axisTick": {"show": True},
            "axisLabel": {"color": "black", "fontSize": 12},
            "splitLine": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "name": "Temperature [°C]",
            "nameLocation": "middle", "nameGap": 40,
            "nameTextStyle": {"color": "#63c1ff", "fontSize": 14},
            "min": domain_y[0], "max": domain_y[1],
            "axisLine": {"show": True, "lineStyle": {"color": "#63c1ff"}},
            "axisTick": {"show": True, "lineStyle": {"color": "#63c1ff"}},
            "axisLabel": {"color": "#63c1ff", "fontSize": 12},
            "splitLine": {"show": False},
        },
        "tooltip": {
            "trigger": "item",
            "formatter": """function(params) {
                if (params.data && params.data.ts) {
                    return params.data.ts + '<br/>'
                        + 'x: ' + params.data.xg + ' g/kg<br/>'
                        + 'T: ' + params.data.temp + ' °C<br/>'
                        + 'φ: ' + params.data.phi + ' %';
                }
                return '';
            }""",
        },
        "legend": {
            "data": legend_data,
            "top": 5, "right": 80,
            "textStyle": {"fontSize": 12},
        } if legend_data else {"show": False},
        "series": series,
    }

    return option


# ======================================================================
# Streamlit UI
# ======================================================================
st.set_page_config(page_title="Mollier h,x-Diagram (ECharts)", layout="wide")

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

option = build_echarts_option(
    domain_x=domain_x,
    domain_y=domain_y,
    pressure=float(pressure),
    comfort_zone={
        "temperature": (comfort_t_min, comfort_t_max),
        "rel_humidity": (comfort_phi_min / 100.0, comfort_phi_max / 100.0),
        "abs_humidity": (comfort_x_min / 1000.0, comfort_x_max / 1000.0),
    },
    data=data,
)

st_echarts(options=option, height="700px", theme="white")
