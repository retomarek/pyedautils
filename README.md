# pyedautils
Python Energy Data Analysis Utilities

The Python-package ‘pyedautils’ provides frequently used utility functions for
the analysis and visualization of comfort and energy data in python. These
functions reduce the complexity of the analysis task and allow a fast
visualization of the data.

## Installation

You can install the package from [GitHub](https://github.com/) with:

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

## Publishing notes for author
``` python
# Install package locally
python setup.py install

# Execute tests
cd tests
python -m unittest discover
cd ..

# push to testpypi
python setup.py sdist bdist_wheel
python -m twine upload -r testpypi dist/*

# finally push to pypi
python setup.py sdist bdist_wheel
python -m twine upload dist/*
```


<hr>

**Disclaimer**<br> The author declines any liability or responsibility in connection with the published code and documentation
