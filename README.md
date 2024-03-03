# pyedautils
**Python Energy Data Analysis Utilities**

[![CI - Test](https://github.com/retomarek/pyedautils/actions/workflows/python-unittest.yml/badge.svg)](https://github.com/retomarek/pyedautils/actions/workflows/python-unittests.yml) [![Coverage](https://codecov.io/github/retomarek/pyedautils/coverage.svg?branch=main)](https://codecov.io/gh/retomarek/pyedautils)
[![PyPI Latest Release](https://img.shields.io/pypi/v/pyedautils.svg)](https://pypi.org/project/pyedautils) [![PyPI Downloads](https://img.shields.io/pypi/dd/pyedautils.svg?label=PyPI%20downloads)](https://pypi.org/project/pyedautils/)

## What is it?

**pyedautils** is a python package designed for analyzing and visualizing comfort and energy time series data. It provides a compact yet expanding collection of utility functions and visualizations aimed at streamlining data exploration.

## Installation

The package is available on [PyPi.org](https://pypi.org/) and can be installed with:

``` python
pip install pyedautils
```

## Plots

### plot_daily_profiles_overview()
This function creates an overview of typical daily profiles per weekday and season of year with a confidence band where most of the values lie.

```html
<iframe src="docs/images/plot_daily_profiles_overview.html" height="600" width="800"></iframe>
```

``` python
from pyedautils.data_io import load_data
from pyedautils.plots import load_data

file_path = "pyedautils/data/ele_meter.csv"
df = load_data(file_path=)
df['value'] = df['value'].diff()
df = df.dropna()

plot_daily_profiles_overview(df)
```

## Functions

### data_io.py
File handling utilities for loading and saving data.

#### save_data()
Saves data to a file in various formats (CSV, Pickle, JSON) based on the given file extension of pile_path.

``` python
from pyedautils.data_io import save_data

file_path = "./my_filename.json"
save_data(df, file_path)

```

#### load_data()
Loads data from a file in various formats (CSV, Pickle, JSON) based on the given file extension of file_path.

``` python
from pyedautils.data_io import load_data

file_path = "pyedautils/data/ele_meter.csv"
df = load_data(file_path)

```

### geopy.py
Helper funtions to find the coordinates from an address, convert lat/long values to swiss WGS84 coordinates and get the altitude from coordinates.

#### get_lat_long()
Returns latitude and longitude coordinates for the given address.

``` python
from pyedautils.geopy import get_lat_long

get_lat_long("Technikumstrasse 21, 6048 Horw, Switzerland")

# Out: [47.0143233, 8.305245521466286]
```

#### get_altitude_lat_long()
Returns altitude in meters above sea level for the given WGS84 coordinates. The opentopodata.org api gets used.

``` python
from pyedautils.geopy import get_altitude_lat_long

get_altitude_lat_long(47.0132975, 8.3059169)

# Out: 444.9
```

#### convert_wsg84_to_lv95()
Converts WGS84 latitude and longitude coordinates to Swiss coordinate system LV95.
``` python
from pyedautils.geopy import convert_wsg84_to_lv95

convert_wsg84_to_lv95(47.0132975, 8.3059169)

# Out: [2665945.104007165, 1207280.4252477456]
```

#### get_altitude_lv95()
Returns altitude in meters above sea level for the given LV95 coordinates. The geo.admin.ch api gets used.

``` python
from pyedautils.geopy import get_altitude_lv95

get_altitude_lv95([2665960.531, 1207281.985])

# Out: 442.6
```

#### get_coordindates_ch_plz()
Returns latitude and longitude for a Swiss postal code.

``` python
from pyedautils.geopy import get_coordindates_ch_plz

get_coordindates_ch_plz(6048)

# Out: (47.0108, 8.3039)
```

#### get_distance_between_two_points()
Calculates the distance in km between two points on the Earth's surface given their latitude and longitude coordinates.

``` python
from pyedautils.geopy import get_coordindates_ch_plz, get_distance_between_two_points

coord1 = get_coordindates_ch_plz(6048) # Horw
coord2 = get_coordindates_ch_plz(3800) # Interlaken

get_distance_between_two_points(coord1, coord2)

# Out: 50.301
```

### meteo_swiss.py

#### find_nearest_station()

Returns station id of closest meteo swiss station to a coordinate.

``` python
from pyedautils.meteo_swiss import find_nearest_station
from pyedautils.geopy import get_coordindates_ch_plz, get_altitude_lat_long

coord = get_coordindates_ch_plz(6197)
altitude = get_altitude_lat_long(coord[0], coord[1])

find_nearest_station(coord[0], coord[1], altitude, sensor="temp")

# Out: "FLU"
```

### season.py

#### get_season()

Get the season name out of a date for filter and grouping purposes.

``` python
from pyedautils.season import get_season
from datetime import datetime

get_season(datetime(2024,5,5))

# Out: 'Spring'
```

Default language of the returned strings is English. You can change that by passing the argument `labels`:

``` python
get_season(datetime(2024,5,5), labels=["Frühling", "Sommer", "Herbst", "Winter"])

# Out: 'Frühling'
```

<hr>

**Disclaimer**<br> The author declines any liability or responsibility in connection with the published code and documentation
