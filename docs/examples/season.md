# season examples

See {doc}`../api/season` for the full API reference.

## Basic usage

```python
from pyedautils.season import get_season
from datetime import datetime

print(get_season(datetime(2024, 7, 15)))   # "Summer"
print(get_season(datetime(2024, 1, 10)))   # "Winter"
print(get_season(datetime(2024, 4, 5)))    # "Spring"
```

## Meteorological vs. astronomical seasons

```python
# Meteorological: fixed dates (Mar 1, Jun 1, Sep 1, Dec 1)
print(get_season(datetime(2024, 3, 15), tracking_type="meteorological"))  # "Spring"

# Astronomical: based on equinox/solstice (default)
print(get_season(datetime(2024, 3, 15), tracking_type="astronomical"))    # "Winter"
```

## Custom season labels (e.g. German)

```python
get_season(datetime(2024, 7, 15), labels=["Frühling", "Sommer", "Herbst", "Winter"])
# "Sommer"
```

## Southern hemisphere

```python
get_season(datetime(2024, 7, 15), hemisphere="south")
# "Winter" (July is winter in the southern hemisphere)
```

## Vectorized with pandas

```python
import pandas as pd

dates = pd.date_range("2024-01-01", periods=365, freq="D")
df = pd.DataFrame({"date": dates})
df["season"] = get_season(df["date"])
print(df["season"].value_counts())
```
