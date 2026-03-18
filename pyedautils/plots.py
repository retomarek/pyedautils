"""Plot functions for energy data analysis and visualization."""

from typing import Dict, List, Optional, Tuple

import numpy as np
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

DEFAULT_MOLLIER_COLORS = {
    "temperature": "#63c1ff",
    "density": "grey",
    "rel_humidity": "green",
    "enthalpy": "#e3362d",
    "comfort": "rgba(0,0,255,0.2)",
}

DEFAULT_SEASON_COLORS = {
    "Winter": "lightblue",
    "Spring": "lightgreen",
    "Summer": "orange",
    "Fall": "brown",
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


def _nice_ticks(vmin, vmax, n):
    """Generate *n* nicely spaced tick values in [*vmin*, *vmax*]."""
    import math as _m
    rng = vmax - vmin
    if rng == 0:
        return [vmin]
    step = rng / n
    mag = 10 ** _m.floor(_m.log10(step))
    residual = step / mag
    if residual <= 1.5:
        nice_step = 1 * mag
    elif residual <= 3:
        nice_step = 2 * mag
    elif residual <= 7:
        nice_step = 5 * mag
    else:
        nice_step = 10 * mag
    start = _m.ceil(vmin / nice_step) * nice_step
    ticks = []
    v = start
    while v <= vmax + nice_step * 0.01:
        ticks.append(round(v, 10))
        v += nice_step
    return ticks


def _sweep_x_range(domain_x, dx, y_func):
    """Sweep x from domain_x[0] to domain_x[1] and collect (x, y) points."""
    xs, ys = [], []
    xv = domain_x[0]
    while xv <= domain_x[1] + dx * 0.5:
        xs.append(xv)
        ys.append(y_func(xv))
        xv += dx
    return xs, ys


def _add_iso_line(fig, xs, ys, color, width=1):
    """Add a single iso-line trace to *fig*."""
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=width),
        showlegend=False, hoverinfo="skip",
    ))


def _add_phi_isolines(fig, c, corners, domain_x, domain_y, domain_t, dt, pressure):
    """Draw relative-humidity iso-lines and the saturation cover."""
    from pyedautils._mollier import get_x_y, rel_humidity as m_rel_humidity

    dim_x = domain_x[1] - domain_x[0]
    dim_y = domain_y[1] - domain_y[0]

    corner_phi = [m_rel_humidity(cx, cy, pressure) for cx, cy in corners]
    phi_max = min(max(corner_phi), 1.0)
    if (corner_phi[1] < 1
            and m_rel_humidity(
                domain_x[0] + dim_x * 0.99, domain_y[0], pressure
            ) > corner_phi[1]):
        phi_max = 1.0
    phi_ticks = _nice_ticks(min(corner_phi), phi_max, 10)

    phi_sat_xs, phi_sat_ys = [], []
    for phi_val in phi_ticks:
        xs, ys = [], []
        t_sweep = domain_t[0]
        while t_sweep <= domain_t[1] + dt * 0.5:
            xv, yv = get_x_y(t_sweep, phi_val, pressure)
            if xv > domain_x[1]:
                break
            xs.append(xv)
            ys.append(yv)
            t_sweep += dt
        if xs:
            _add_iso_line(fig, xs, ys, c["rel_humidity"], width=1.5)
            if phi_val == phi_ticks[-1]:
                phi_sat_xs, phi_sat_ys = list(xs), list(ys)

    # Saturation cover
    if phi_sat_xs:
        cx = list(phi_sat_xs) + [
            domain_x[1] + 0.1 * dim_x, domain_x[1] + 0.1 * dim_x,
            phi_sat_xs[0], phi_sat_xs[0],
        ]
        cy = list(phi_sat_ys) + [
            phi_sat_ys[-1], domain_y[0] - 0.1 * dim_y,
            domain_y[0] - 0.1 * dim_y, phi_sat_ys[0],
        ]
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            fill="toself", fillcolor="white",
            line=dict(color="black", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))


def _add_mollier_isolines(fig, show, c, domain_x, domain_y, pressure, num_points):
    """Draw iso-lines (temperature, density, humidity, enthalpy) onto *fig*."""
    from pyedautils._mollier import (
        density as m_density,
        enthalpy as m_enthalpy,
        get_x_y_tx,
        temperature as m_temperature,
        x_hy,
        y_hx,
        y_rhox,
    )

    dx = (domain_x[1] - domain_x[0]) / num_points

    corners = [
        (domain_x[0], domain_y[0]), (domain_x[1], domain_y[0]),
        (domain_x[0], domain_y[1]), (domain_x[1], domain_y[1]),
    ]
    corner_t = [m_temperature(cx, cy) for cx, cy in corners]
    domain_t = (min(corner_t), max(corner_t))
    dt = (domain_t[1] - domain_t[0]) / num_points

    if show["temperature"]:
        for t_val in _nice_ticks(domain_t[0], domain_t[1], 40):
            xs, ys = _sweep_x_range(domain_x, dx,
                                    lambda xv, t=t_val: get_x_y_tx(t, xv, pressure)[1])
            _add_iso_line(fig, xs, ys, c["temperature"])

    if show["density"]:
        corner_rho = [m_density(cx, cy, pressure) for cx, cy in corners]
        for rho_val in _nice_ticks(min(corner_rho), max(corner_rho), 8):
            xs, ys = _sweep_x_range(domain_x, dx,
                                    lambda xv, r=rho_val: y_rhox(r, xv, pressure))
            _add_iso_line(fig, xs, ys, c["density"])

    if show["rel_humidity"]:
        _add_phi_isolines(fig, c, corners, domain_x, domain_y, domain_t, dt, pressure)

    if show["enthalpy"]:
        corner_h = [m_enthalpy(cx, cy) for cx, cy in corners]
        for h_val in _nice_ticks(min(corner_h), max(corner_h), 20):
            fig.add_trace(go.Scatter(
                x=[domain_x[0], x_hy(h_val, domain_y[0])],
                y=[y_hx(h_val, domain_x[0]), domain_y[0]],
                mode="lines", line=dict(color=c["enthalpy"], width=1),
                showlegend=False, hoverinfo="skip",
            ))


def _add_mollier_data(fig, data, pressure):
    """Add season-coloured data markers to a Mollier figure."""
    from pyedautils._mollier import get_x_y
    from pyedautils.season import get_season

    df = data.copy()
    df.columns = ["timestamp", "humidity", "temperature"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["humidity", "temperature"])
    if df.empty:
        return

    t_arr = df["temperature"].values
    phi_arr = df["humidity"].values / 100.0
    x_arr, y_arr = get_x_y(t_arr, phi_arr, pressure)

    df["x_coord"] = x_arr
    df["y_coord"] = y_arr
    df["season"] = get_season(df["timestamp"])

    for season_name in df["season"].unique():
        subset = df[df["season"] == season_name]
        fig.add_trace(go.Scatter(
            x=subset["x_coord"], y=subset["y_coord"],
            mode="markers",
            marker=dict(
                size=4,
                color=DEFAULT_SEASON_COLORS.get(season_name, "grey"),
                opacity=0.6,
            ),
            name=season_name,
        ))


def plot_mollier_hx(
    data: Optional[pd.DataFrame] = None,
    pressure: float = 101325.0,
    domain_x: Tuple[float, float] = (0.0, 0.020),
    domain_y: Tuple[float, float] = (-20.0, 50.0),
    comfort_zone: Optional[Dict[str, Tuple[float, float]]] = None,
    colors: Optional[Dict[str, str]] = None,
    title: str = "Mollier h,x-Diagram",
    show_isolines: Optional[Dict[str, bool]] = None,
    num_points: int = 100,
) -> go.Figure:
    """
    Create a Mollier h,x-diagram (psychrometric chart) with iso-lines.

    The diagram visualises the state of moist air using iso-lines for
    temperature, enthalpy, relative humidity and density, plus an optional
    comfort zone and measured data points colour-coded by season.

    Args:
        data: Optional DataFrame with columns [timestamp, humidity, temperature].
            humidity in %, temperature in °C.
        pressure: Air pressure in Pa. Default 101325 (sea level).
        domain_x: Range of absolute humidity [kg/kg] for the x-axis.
        domain_y: Range of the y-coordinate (≈ temperature at x=0) for the y-axis.
        comfort_zone: Dict with keys "temperature", "rel_humidity", "abs_humidity",
            each a (min, max) tuple. Defaults: T=[20, 26], phi=[0.30, 0.65],
            x=[0, 0.0115].
        colors: Colour overrides for iso-lines and comfort zone.
            Keys: "temperature", "density", "rel_humidity", "enthalpy", "comfort".
        title: Plot title.
        show_isolines: Enable/disable iso-line families. Keys: "temperature",
            "density", "rel_humidity", "enthalpy". All True by default.
        num_points: Number of points per curve (smoothness). Default 100.

    Returns:
        go.Figure: Interactive Plotly figure.
    """
    from pyedautils._mollier import create_comfort

    c = {**DEFAULT_MOLLIER_COLORS, **(colors or {})}
    show = {
        "temperature": True, "density": True,
        "rel_humidity": True, "enthalpy": True,
        **(show_isolines or {}),
    }

    fig = go.Figure()

    # Iso-lines
    _add_mollier_isolines(fig, show, c, domain_x, domain_y, pressure, num_points)

    # Comfort zone
    cz = comfort_zone or {}
    polygon = create_comfort(
        cz.get("temperature", (20, 26)),
        cz.get("rel_humidity", (0.30, 0.65)),
        cz.get("abs_humidity", (0, 0.0115)),
        pressure,
    )
    if polygon:
        fig.add_trace(go.Scatter(
            x=[pt[0] for pt in polygon],
            y=[pt[1] for pt in polygon],
            mode="lines", fill="toself", fillcolor=c["comfort"],
            line=dict(color="black", width=1),
            showlegend=False, name="Comfort Zone",
        ))

    # Data points
    if data is not None and not data.empty:
        _add_mollier_data(fig, data, pressure)

    # Layout
    x_tick_vals = np.arange(domain_x[0], domain_x[1] + 0.001, 0.002)
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20),
        title_x=0.5,
        template="plotly_white",
        xaxis=dict(
            title="Absolute Humidity [g/kg]",
            range=domain_x,
            tickvals=x_tick_vals.tolist(),
            ticktext=[f"{v * 1000:.0f}" for v in x_tick_vals],
        ),
        yaxis=dict(title="Temperature [°C]", range=domain_y),
    )

    return fig
