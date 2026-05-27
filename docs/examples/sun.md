# sun

See {doc}`../api/sun` for the full API reference.

## Basic usage

```python
from pyedautils.data_prep.sun import sun_position

# Zurich, summer solstice around solar noon (UTC)
elevation, azimuth = sun_position("2025-06-21 11:30", latitude=47.3769, longitude=8.5417)
print(round(elevation, 1), round(azimuth, 1))   # 66.1 181.3  (high, due south)
```

## Convention

Elevation is the angle above the horizon (negative when the sun is below it).
Azimuth is measured clockwise from geographic north: 0=N, 90=E, 180=S, 270=W.

```python
# Winter solstice noon: sun much lower, still due south
elevation, azimuth = sun_position("2025-12-21 11:30", 47.3769, 8.5417)
print(round(elevation, 1), round(azimuth, 1))   # 19.2 181.5
```

## Timezone handling

Timezone-aware timestamps are converted to UTC; naive timestamps are assumed to be UTC.

```python
import pandas as pd

# 13:30 local (CEST) == 11:30 UTC -> identical result
sun_position(pd.Timestamp("2025-06-21 13:30", tz="Europe/Zurich"), 47.3769, 8.5417)
```

## Vectorized with pandas

Pass a `Series` or `DatetimeIndex` to get a `DataFrame` with `elevation` and `azimuth`
columns, indexed like the input:

```python
import pandas as pd

idx = pd.date_range("2025-06-21 04:00", "2025-06-21 20:00", freq="1h", tz="UTC")
df = sun_position(idx, latitude=47.3769, longitude=8.5417)
print(df.round(1))
```

## Geometric vs. apparent position

By default the apparent position (including atmospheric refraction) is returned. Set
`refraction=False` for the true geometric position (the difference is largest near
the horizon):

```python
sun_position("2025-06-21 04:00", 47.3769, 8.5417, refraction=False)
```
