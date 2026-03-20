"""Heatmap visualization plots."""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyedautils._plot_utils import DEFAULT_SEASONS


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
