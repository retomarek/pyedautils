# pyedautils

**Python Energy Data Analysis Utilities** -- a pip-installable library of compact utility functions for analyzing and visualizing energy and comfort time-series data.

## Features

- **{doc}`Data I/O <api/data_io>`** -- Save and load DataFrames in CSV, pickle, compressed pickle, and JSON formats with automatic directory creation, timing, and file-size logging.
- **{doc}`Data quality <api/data_quality>`** -- Gap, stuck-value and range-outlier detection, interval inference, ok/warning/critical quality flags, and missing-value visualizations.
- **{doc}`Thermal comfort <api/comfort>`** -- SIA 180:2014 adaptive comfort boundary curves, summer-half-year filter, overheating hours and KPIs (Minergie / SIA 180 limits), and the comfort-compass distribution of temperature/humidity states.
- **{doc}`Psychrometrics <api/psychrometrics>`** -- Moist-air state calculations (dew point, wet bulb, enthalpy, density, pressure from altitude) and air-handling processes (heating, cooling, humidification, mixing, heat recovery) for the Mollier h,x-diagram.
- **{doc}`Energy signature <api/energy_signature>`** -- Proposed Energy Signature (Eriksson et al. 2020): balance temperature, heat loss coefficient, and domestic hot water / circulation demand from hourly data.
- **{doc}`Unit conversion <api/units>`** -- SI ↔ IP conversions for temperature, humidity ratio, enthalpy, pressure, flow, power, density and more, including whole moist-air states.
- **{doc}`Degree days & climate correction <api/degree_days>`** -- Swiss heating degree days HGT 20/12, accumulated temperature differences ATD (SIA 380), cooling degree days, reference climatology, SIA-380-style normalization of heating energy, fixed-base degree-day regression (ASHRAE Guideline 14 metrics) and the BFE GT&S method.
- **{doc}`Geocoding & coordinates <api/geopy>`** -- Address geocoding (Nominatim), WGS84-to-Swiss-LV95 conversion, altitude lookup via opentopodata.org and geo.admin.ch, Swiss postal code resolution, and Haversine distance calculation.
- **{doc}`Plotting <api/plots>`** -- Plotly- and D3-based visualizations:
  - daily profiles (seasons x weekdays grid with median lines and quantile bands), median-week and calendar heatmaps
  - energy signature (simple and PES), seasonal plots (overlapping, miniplots, before/after, polar) and density by season
  - comfort plots: SIA 180 comfort, Mollier h,x-diagram (incl. a fast interactive variant), comfort compass, comfort donuts, overheating time series and bar chart, temperature/humidity time series
  - statistical plots: sum frequency (duration curve), STL decomposition, time series, distribution, boxplot, outliers, correlation, scatter, autocorrelation
  - heating/cooling gradient boxplots and solar-influence dual-axis plot
  - degree days vs. reference and degree-day regression
- **Data preparation**
  - {doc}`Season detection <api/season>` -- Astronomical (ephem-based equinox/solstice) or meteorological seasons, for both hemispheres and with custom labels.
  - {doc}`Solar position <api/sun>` -- Sun elevation and azimuth for a location and time, as a single value or vectorized over a pandas Series/DatetimeIndex.
  - {doc}`Gradients <api/gradients>` -- Heating and cooling gradients (K/h) of a time series, classified by direction and season.
  - {doc}`Solar influence <api/solar_influence>` -- Detect direct solar influence on a sensor (radiation-vs-excess-temperature correlation, orientation hint, steep-rise events).
  - {doc}`Reshape <api/reshape>` -- Per-id resampling of long-format sensor data, IQR outlier removal, and long-to-wide pivoting.
- **Swiss weather data**
  - {doc}`MeteoSwiss <api/meteo_swiss>` -- Download of historical and recent measurements (10-minute, hourly, daily) from the MeteoSwiss open data portal by station or Swiss postal code, current measurements of all automatic stations, and the nearest station for a sensor type within an altitude tolerance.
  - {doc}`Agrometeo.ch <api/agroweather>` -- Station list, nearest station for a sensor type, and download of hourly temperature, global radiation, humidity and rain data by station or Swiss postal code.
  - {doc}`Swiss open data <api/open_data_ch>` -- Monthly degree-day series of the BFE (population-weighted, since 1980) and the Stadt Zürich (since 1864).

## Quick start

```python
from pyedautils.data_prep.season import get_season
from datetime import datetime

season = get_season(datetime(2024, 7, 15))
print(season)  # "Summer"
```

See the {doc}`installation` page for setup instructions and the API Reference for detailed documentation of each module.
