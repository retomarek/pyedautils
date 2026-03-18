"""Streamlit app — Mollier h,x-diagram using D3.js (via pyedautils)."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from importlib import resources

from pyedautils.plots import plot_mollier_hx

st.set_page_config(page_title="Mollier h,x-Diagram", layout="wide")

st.markdown(
    "<h1 style='font-family: monospace; font-weight: normal;'>Mollier-hx-diagram</h1>",
    unsafe_allow_html=True,
)

# --- Sidebar: Control Panel ---
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
    "Upload CSV (timestamp, humidity, temperature)",
    type=["csv"],
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
html = plot_mollier_hx(
    data=data,
    pressure=float(pressure),
    domain_x=(x_min / 1000.0, x_max / 1000.0),
    domain_y=(y_min, y_max),
    comfort_zone={
        "temperature": (comfort_t_min, comfort_t_max),
        "rel_humidity": (comfort_phi_min / 100.0, comfort_phi_max / 100.0),
        "abs_humidity": (comfort_x_min / 1000.0, comfort_x_max / 1000.0),
    },
    height=chart_height,
)

full_html = f"<!DOCTYPE html><html><body style='margin:0;background:white'>{html}</body></html>"
components.html(full_html, height=chart_height + 10, scrolling=False)
