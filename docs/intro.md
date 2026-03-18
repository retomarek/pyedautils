# pyedautils

**Python Energy Data Analysis Utilities** -- a pip-installable library of compact utility functions for analyzing and visualizing energy and comfort time-series data.

## Features

- **Data I/O** -- Save and load DataFrames in CSV, pickle, compressed pickle, and JSON formats with automatic directory creation, timing, and file-size logging.
- **Geocoding & coordinates** -- Address geocoding (Nominatim), WGS84-to-Swiss-LV95 conversion, altitude lookup via opentopodata.org and geo.admin.ch, Swiss postal code resolution, and Haversine distance calculation.
- **Plotting** -- Plotly-based daily profile visualizations: a 4x7 subplot grid (seasons x weekdays) with median lines and quantile confidence bands, plus a decomposed seasonal profile view.
- **Season detection** -- Determine the season for any date using astronomical (ephem-based equinox/solstice) or meteorological definitions, with support for both hemispheres and custom labels.
- **MeteoSwiss stations** -- Find the nearest MeteoSwiss weather station matching a given sensor type within an altitude tolerance.

## Quick start

```python
from pyedautils.season import get_season
from datetime import datetime

season = get_season(datetime(2024, 7, 15))
print(season)  # "Summer"
```

See the {doc}`installation` page for setup instructions and the API Reference for detailed documentation of each module.
