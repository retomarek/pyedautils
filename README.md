# pyedautils
**Python Energy Data Analysis Utilities**

| --- | --- |
| Testing | [![CI - Test](https://github.com/retomarek/pyedautils/actions/workflows/python-unittest.yml/badge.svg)](https://github.com/retomarek/pyedautils/actions/workflows/python-unittests.yml) [![Coverage](https://codecov.io/github/retomarek/pyedautils/coverage.svg?branch=main)](https://codecov.io/gh/retomarek/pyedautils) |
| Package | [![PyPI Latest Release](https://img.shields.io/pypi/v/pyedautils.svg)](https://pypi.org/project/pyedautils) [![PyPI Downloads](https://img.shields.io/pypi/dd/pyedautils.svg?label=PyPI%20downloads)](https://pypi.org/project/pyedautils/) |

## What is it?

**pyedautils** is a python package that provides frequently used utility functions for the analysis and visualization of comfort and energy time series data. These functions reduce the complexity of the analysis and visualization of the data.

## Installation

You can install the package from [PyPi.org](https://pypi.org/) with:

``` python
pip install pyedautils
```

## Functions

### get_season()

Get the season name out of a date for filter and grouping purposes.

``` python
from pyedautils.season import get_season
from datetime import datetime

get_season(datetime(2024,5,5))

#Out[0]: 'Spring'
```

Default language of the returned strings is English. You can change that by passing the argument `labels`:

``` python
get_season(datetime(2024,5,5), labels=["Frühling", "Sommer", "Herbst", "Winter"])

#Out[1]: 'Frühling'
```

<hr>

**Disclaimer**<br> The author declines any liability or responsibility in connection with the published code and documentation
