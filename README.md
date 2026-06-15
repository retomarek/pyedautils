# pyedautils
**Python Energy Data Analysis Utilities**

[![CI - Test](https://github.com/retomarek/pyedautils/actions/workflows/python-unittest.yml/badge.svg)](https://github.com/retomarek/pyedautils/actions/workflows/python-unittest.yml) [![Coverage](https://codecov.io/github/retomarek/pyedautils/coverage.svg?branch=main)](https://codecov.io/gh/retomarek/pyedautils)
[![PyPI Latest Release](https://img.shields.io/pypi/v/pyedautils.svg)](https://pypi.org/project/pyedautils) [![PyPI Downloads](https://img.shields.io/pypi/dd/pyedautils.svg?label=PyPI%20downloads)](https://pypi.org/project/pyedautils/)

A pip-installable library of compact utility functions for analyzing and visualizing energy and comfort time-series data.

## Features

- **Plotting** — Plotly-based daily profile visualizations with confidence bands and decomposed weekly patterns
- **Data quality** — Gap, stuck-value and range-outlier detection, interval inference, ok/warning/critical flags
- **Thermal comfort** — SIA 180:2014 adaptive comfort curves, overheating hours/KPIs, comfort donuts, overheating bar
- **Gradients** — Heating/cooling gradients (K/h) by direction and season, with grouped boxplots
- **Solar influence** — Detect direct solar influence on a sensor, with a dual-axis plot
- **Data I/O** — Save/load DataFrames in CSV, pickle, compressed pickle, and JSON formats
- **Geocoding** — Address geocoding, WGS84/LV95 conversion, altitude lookup, Swiss postal codes, Haversine distance
- **Season detection** — Astronomical or meteorological season classification for any date
- **Solar position** — Sun elevation and azimuth for a location and time (single timestamp or vectorized over a pandas Series/DatetimeIndex)
- **MeteoSwiss** — Find nearest weather station by sensor type and altitude

## Installation

```bash
pip install pyedautils
```

## Quick start

```python
from pyedautils.plots import plot_daily_profiles_overview
from pyedautils.data_io import load_data

df = load_data("my_data.csv")
fig = plot_daily_profiles_overview(df)
fig.show()
```

## Documentation

Full API reference, examples with interactive plots, and usage guides:

**[retomarek.github.io/pyedautils](https://retomarek.github.io/pyedautils/)**

## License

**Disclaimer** — The author declines any liability or responsibility in connection with the published code and documentation.
