import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pyedautils.season import get_season

def plot_daily_profiles_overview(data,
                                 title="Daily Profiles Overview by Weekday and Season",
                                 ylab="Value",
                                 confidence=95.0):
    # function code
    data.columns = ["timestamp", "value"]

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # aggregate hours
    data["hour"] = data["timestamp"].dt.floor("H")

    df_h = data.groupby("hour").agg({"value": "sum"}).reset_index()

    df_h["weekday"] = df_h["hour"].dt.day_name()
    df_h["dayhour"] = df_h["hour"].dt.hour
    df_h["season"] = get_season(df_h["hour"])

    # create factors for correct order in plot
    df_h["season_fac"] = pd.Categorical(df_h["season"], categories=["Spring", "Summer", "Fall", "Winter"])

    df = df_h.groupby(["season", "weekday", "dayhour"]).agg(
        valueMedian=("value", lambda x: x.quantile(0.5)),
        valueUpper=("value", lambda x: x.quantile(confidence / 100)),
        valueLower=("value", lambda x: x.quantile((100 - confidence) / 100))
    ).reset_index()


    # Define seasons and weekdays
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Create subplots
    fig = make_subplots(rows=4, cols=7, subplot_titles=(weekdays), shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.025, horizontal_spacing=0.025)   
    
    # Define x ticks
    xticks = np.arange(0, 24, 6)
    thick_ticks = np.arange(0, 24, 6)
    
    # Iterate over seasons
    for i, season in enumerate(seasons):
        # Iterate over weekdays
        for j, weekday in enumerate(weekdays):
            # Filter DataFrame for the current season and weekday
            data_subset = df[(df['season'] == season) & (df['weekday'] == weekday)]
    
            # Fill the area between upper and lower lines
            fig.add_trace(go.Scatter(x=np.concatenate([data_subset['dayhour'], data_subset['dayhour'][::-1]]),
                                     y=np.concatenate([data_subset['valueLower'], data_subset['valueUpper'][::-1]]),
                                     fill='toself', opacity=0.3, fillcolor='lightgrey', line=dict(color='rgba(255,255,255,0)'),
                                     showlegend=False), row=i+1, col=j+1)
            
            # Add upper and lower bounds
            fig.add_trace(go.Scatter(x=data_subset['dayhour'], y=data_subset['valueUpper'], mode='lines', line=dict(color='darkgrey'), name='Upper', showlegend=False), row=i+1, col=j+1)
            fig.add_trace(go.Scatter(x=data_subset['dayhour'], y=data_subset['valueMedian'], mode='lines', line=dict(color='black'), name='Median', showlegend=False), row=i+1, col=j+1)
            fig.add_trace(go.Scatter(x=data_subset['dayhour'], y=data_subset['valueLower'], mode='lines', line=dict(color='darkgrey'), name='Lower', showlegend=False), row=i+1, col=j+1)
            
            # Update x-axis tick labels
            fig.update_xaxes(tickvals=xticks, row=i + 1, col=j + 1)
            fig.update_xaxes(tickvals=thick_ticks, row=i + 1, col=j + 1)
            
            # Add frame around subplot on all sides
            fig.update_xaxes(showline=True, linewidth=1, linecolor='darkgrey', row=i + 1, col=j + 1)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='darkgrey', row=i + 1, col=j + 1)
            fig.update_xaxes(showline=True, linewidth=1, linecolor='darkgrey', mirror=True, row=i + 1, col=j + 1)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='darkgrey', mirror=True, row=i + 1, col=j + 1)

            # Update layout
            fig.update_layout(
                title_text=f"<b>{title}</b>", 
                template='plotly_white', 
                title_x=0.5,
                title_font=dict(size=20)
            )
            
            # Add season titles
            if (j == 0):
                fig.update_yaxes(title_text=season, row=i+1, col=j+1)
    
    return fig
