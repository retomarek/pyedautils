# plots

Plotly-based visualizations for energy time-series data.

## Examples

### Daily profiles overview

Creates a 4x7 subplot grid (seasons x weekdays) with median lines and confidence bands:

```python
from pyedautils.plots import plot_daily_profiles_overview
from pyedautils.data_io import load_data

df = load_data("pyedautils/data/ele_meter.csv")
df["value"] = df["value"].diff()
df = df.dropna()

fig = plot_daily_profiles_overview(df, title="Electricity Daily Profiles", ylab="Energy [kWh]")
fig.show()
```

### Custom colors and confidence level

```python
fig = plot_daily_profiles_overview(
    df,
    title="90% Confidence Band",
    confidence=90.0,
    colors={"median": "red", "bounds": "orange", "fill": "rgba(255,165,0,0.2)"},
)
fig.show()
```

### Decomposed weekly pattern

Shows the seasonal component per weekday after detrending:

```python
from pyedautils.plots import plot_daily_profiles_decomposed

fig = plot_daily_profiles_decomposed(
    df,
    loc_time_zone="Europe/Zurich",
    title="Decomposed Weekly Pattern",
    ylab="delta Energy [kWh]",
)
fig.show()
```

See {doc}`../examples/daily_profiles` for interactive versions of these plots.

### Mollier h,x-Diagram

Creates a psychrometric chart (Mollier h,x-diagram) with iso-lines for temperature,
enthalpy, relative humidity and density, a comfort zone, and optional measured data
points colour-coded by season:

```python
from pyedautils.plots import plot_mollier_hx

# Empty diagram with default comfort zone
fig = plot_mollier_hx(title="Mollier h,x-Diagram")
fig.show()
```

With measured data:

```python
import pandas as pd
from importlib import resources

data_path = resources.files("pyedautils") / "data" / "mollier_sample.csv"
df = pd.read_csv(data_path)

fig = plot_mollier_hx(data=df, title="Indoor Climate Measurements")
fig.show()
```

Custom comfort zone and pressure:

```python
fig = plot_mollier_hx(
    pressure=95000.0,
    comfort_zone={
        "temperature": (18, 24),
        "rel_humidity": (0.20, 0.70),
        "abs_humidity": (0, 0.012),
    },
)
fig.show()
```

See {doc}`../examples/mollier_hx` for interactive examples.

## API Reference

```{eval-rst}
.. automodule:: pyedautils.plots
   :members:
   :undoc-members:
   :show-inheritance:
```
