# pyedautils
**Python Energy Data Analysis Utilities**

[![CI - Test](https://github.com/retomarek/pyedautils/actions/workflows/python-unittest.yml/badge.svg)](https://github.com/retomarek/pyedautils/actions/workflows/python-unittest.yml) [![Coverage](https://codecov.io/github/retomarek/pyedautils/coverage.svg?branch=main)](https://codecov.io/gh/retomarek/pyedautils)
[![PyPI Latest Release](https://img.shields.io/pypi/v/pyedautils.svg)](https://pypi.org/project/pyedautils) [![PyPI Downloads](https://img.shields.io/pypi/dd/pyedautils.svg?label=PyPI%20downloads)](https://pypi.org/project/pyedautils/)

## What is it?

**pyedautils** is a python package that provides frequently used utility functions for the analysis and visualization of comfort and energy time series data. These functions reduce the complexity of the analysis and visualization of the data.

## Installation

You can install the package from [PyPi.org](https://pypi.org/) with:

``` python
pip install pyedautils
```

## Functions

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

### season.py

#### get_season()

Get the season name out of a date for filter and grouping purposes.

``` python
from pyedautils.season import get_season
from datetime import datetime

get_season(datetime(2024,5,5))

#Out: 'Spring'
```

Default language of the returned strings is English. You can change that by passing the argument `labels`:

``` python
get_season(datetime(2024,5,5), labels=["Frühling", "Sommer", "Herbst", "Winter"])

#Out: 'Frühling'
```

<hr>

**Disclaimer**<br> The author declines any liability or responsibility in connection with the published code and documentation
