# agroweather

See {doc}`../api/agroweather` for the full API reference.

## Find the nearest temperature station

```python
from pyedautils.agroweather import find_nearest_station

station_id = find_nearest_station(47.05, 8.31, sensor="temp")
print(station_id)  # 190
```

## Download hourly data for a station

```python
from pyedautils.agroweather import download_data

df = download_data(station_id, "2024-01-01", "2024-01-03", sensors=["temp", "relhum"])
print(df.head())
#                      temp  relhum
# datetime
# 2024-01-01 00:00:00   1.7   100.0
# 2024-01-01 01:00:00   1.3   100.0
# 2024-01-01 02:00:00   0.7   100.0
# 2024-01-01 03:00:00   0.9   100.0
# 2024-01-01 04:00:00   1.0   100.0
```

## Download data by postal code

```python
from pyedautils.agroweather import download_data_by_plz

df = download_data_by_plz(6048, "2024-01-01", "2024-01-03")
print(df.head())
#                      temp  relhum  rain  globrad
# datetime
# 2024-01-01 00:00:00   1.7   100.0   0.0      0.0
# 2024-01-01 01:00:00   1.3   100.0   0.0      0.0
# 2024-01-01 02:00:00   0.7   100.0   0.0      0.0
# 2024-01-01 03:00:00   0.9   100.0   0.0      0.0
# 2024-01-01 04:00:00   1.0   100.0   0.0      0.0
```

## Get all station data

```python
from pyedautils.agroweather import get_station_data

stations = get_station_data()
print(f"{len(stations)} stations found")
# 217 stations found
print(stations[["id", "name", "lat", "lon"]].head())
#     id                   name        lat       lon
# 0  189               AESCH-BL  47.466863  7.575196
# 1  360  AGNO-Progetto ViSo-KS  45.997000  8.893944
# 2   94                  AIGLE  46.314537  6.979451
# 3   78               AMBURNEX  46.536925  6.215849
# 4   79         AMBURNEX-COMBE  46.535419  6.218099
```
