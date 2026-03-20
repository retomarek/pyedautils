# agroweather

See {doc}`../api/agroweather` for the full API reference.

## Find the nearest temperature station

```python
from pyedautils.agroweather import find_nearest_station

station_id = find_nearest_station(47.05, 8.31, sensor="temp")
print(station_id)
```

## Download hourly data for a station

```python
from pyedautils.agroweather import download_data

df = download_data(station_id, "2024-01-01", "2024-01-31", sensors=["temp", "relhum"])
print(df.head())
```

## Download data by postal code

```python
from pyedautils.agroweather import download_data_by_plz

df = download_data_by_plz(6048, "2024-01-01", "2024-01-31")
print(df.head())
```

## Get all station data

```python
from pyedautils.agroweather import get_station_data

stations = get_station_data()
print(f"{len(stations)} stations found")
print(stations[["id", "name", "lat", "lon"]].head())
```
