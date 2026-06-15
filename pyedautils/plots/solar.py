"""Solar-influence plot (see :mod:`pyedautils.data_prep.solar_influence`)."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

COLOR_TEMP = "#0D7377"
COLOR_SOLAR = "#FBC02D"
COLOR_EVENT = "#E53935"


def plot_solar_influence(
    data: pd.DataFrame,
    room_col: str = "t_room",
    radiation_col: str = "solar",
    event_col: Optional[str] = None,
    title: str = "Solar Influence",
    temp_label: str = "Room temperature (°C)",
    radiation_label: str = "Global radiation (W/m²)",
    height: int = 360,
) -> go.Figure:
    """Dual-axis time series of room temperature and global radiation.

    Room temperature is drawn on the left axis, global radiation on the
    right axis. Optional steep-rise events are marked with triangles.

    Args:
        data: DataFrame indexed by timestamp (or with a ``timestamp``
            column) containing the room-temperature and radiation columns.
        room_col: Room-temperature column. Default ``"t_room"``.
        radiation_col: Global-radiation column. Default ``"solar"``.
        event_col: Optional boolean column flagging steep-rise events;
            flagged rows are overlaid as triangles. Default *None*.
        title: Plot title.
        temp_label: Left y-axis label.
        radiation_label: Right y-axis label.
        height: Figure height in pixels. Default 360.

    Returns:
        go.Figure: Plotly dual-axis figure.
    """
    df = data.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df.index, y=df[radiation_col],
        mode="lines", line=dict(color=COLOR_SOLAR, width=0.8),
        name="Global radiation", yaxis="y2", opacity=0.6,
        hovertemplate="%{x}<br>%{y:.0f} W/m²<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=df.index, y=df[room_col],
        mode="lines", line=dict(color=COLOR_TEMP, width=1.2),
        name="Room temperature",
        hovertemplate="%{x}<br>%{y:.1f} °C<extra></extra>",
    ))
    if event_col is not None and event_col in df.columns:
        events = df[df[event_col].astype(bool)]
        if not events.empty:
            fig.add_trace(go.Scattergl(
                x=events.index, y=events[room_col],
                mode="markers",
                marker=dict(color=COLOR_EVENT, size=8, symbol="triangle-up"),
                name="Steep-rise event",
                hovertemplate="%{x}<br>%{y:.1f} °C<extra></extra>",
            ))

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=20), title_x=0.5,
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(title=temp_label),
        yaxis2=dict(title=radiation_label, overlaying="y", side="right",
                    showgrid=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig
