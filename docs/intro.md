# pyedautils

**Python Energy Data Analysis Utilities** -- a pip-installable library of compact utility functions for analyzing and visualizing energy and comfort time-series data.

## Features

- **{doc}`Data I/O <api/data_io>`** -- Save and load DataFrames in CSV, pickle, compressed pickle, and JSON formats with automatic directory creation, timing, and file-size logging.
- **{doc}`Geocoding & coordinates <api/geopy>`** -- Address geocoding (Nominatim), WGS84-to-Swiss-LV95 conversion, altitude lookup via opentopodata.org and geo.admin.ch, Swiss postal code resolution, and Haversine distance calculation.
- **{doc}`Plotting <api/plots>`** -- Plotly-based daily profile visualizations: a 4x7 subplot grid (seasons x weekdays) with median lines and quantile confidence bands, plus a decomposed seasonal profile view.
- **{doc}`Data quality <api/data_quality>`** -- Gap, stuck-value and range-outlier detection, interval inference, ok/warning/critical quality flags, and missing-value visualizations.
- **{doc}`Thermal comfort <api/comfort>`** -- SIA 180:2014 adaptive comfort boundary curves, summer-half-year filter, overheating hours and KPIs (Minergie / SIA 180 limits), plus comfort donuts and an overheating bar chart.
- **{doc}`Gradients <api/gradients>`** -- Heating and cooling gradients (K/h) of a time series, classified by direction and season, with grouped boxplots.
- **{doc}`Solar influence <api/solar_influence>`** -- Detect direct solar influence on a sensor (radiation-vs-excess-temperature correlation, orientation hint, steep-rise events) with a dual-axis plot.
- **{doc}`Season detection <api/season>`** -- Determine the season for any date using astronomical (ephem-based equinox/solstice) or meteorological definitions, with support for both hemispheres and custom labels.
- **{doc}`Solar position <api/sun>`** -- Sun elevation and azimuth (ephem-based) for a location and time, as a single value or vectorized over a pandas Series/DatetimeIndex.
- **{doc}`MeteoSwiss stations <api/meteo_swiss>`** -- Find the nearest MeteoSwiss weather station matching a given sensor type within an altitude tolerance.

## Quick start

```python
from pyedautils.data_prep.season import get_season
from datetime import datetime

season = get_season(datetime(2024, 7, 15))
print(season)  # "Summer"
```

See the {doc}`installation` page for setup instructions and the API Reference for detailed documentation of each module.
