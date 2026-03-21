"""Energy signature plots."""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from pyedautils.plots._constants import DEFAULT_SEASON_COLORS


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
    from pyedautils.data_prep.season import get_season

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
    from pyedautils.data_prep.season import get_season

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
