# meteo_swiss

See {doc}`../api/meteo_swiss` for the full API reference.

## Download historical data

Measurement data is downloaded from the
[MeteoSwiss open data portal](https://opendatadocs.meteoswiss.ch/a-data-groundbased/a1-automatic-weather-stations)
(OGD). The function combines the historical files (per decade), the file of
the current year and the file of the last day, so any period from the start of
the measurements up to now can be requested.

Time stamps are in UTC and mark the end of the measurement interval (the
hourly value at 16:00 covers 15:00–16:00). With `tz` the dates are
interpreted in, and the index is converted to, another time zone.

```python
from pyedautils.weather.meteo_swiss import download_data

# Hourly temperature, global radiation, relative humidity and rain at Zurich-Kloten
df = download_data("KLO", "2025-01-01", "2025-12-31", tz="Europe/Zurich")
print(df.head(3))
#                            temp  globrad  relhum  rain
# datetime
# 2025-01-01 00:00:00+01:00  -2.1        2    96.0   0.0
# 2025-01-01 01:00:00+01:00  -2.1        2    96.8   0.0
# 2025-01-01 02:00:00+01:00  -2.4        2    95.8   0.0
```

Available granularities: `t` (10 minutes), `h` (hourly, default), `d` (daily).
Besides the sensor types `temp`, `globrad`, `relhum` and `rain`, any
MeteoSwiss parameter ID can be requested (see
[ogd-smn_meta_parameters.csv](https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_parameters.csv)):

```python
# Daily mean, maximum and minimum temperature
df = download_data("LUZ", "2020-01-01", "2024-12-31", granularity="d",
                   sensors=["temp", "tre200dx", "tre200dn"])

# 10-minute values
df = download_data("BER", "2026-06-01", "2026-06-07", granularity="t", sensors=["temp", "globrad"])
```

## Download data by postal code

For each sensor the nearest station with that sensor within 150 m altitude
difference is used; the results are merged into one DataFrame.

```python
from pyedautils.weather.meteo_swiss import download_data_by_plz

df = download_data_by_plz(6048, "2025-07-01", "2025-07-31", tz="Europe/Zurich")
```

## Find the nearest temperature station

```python
from pyedautils.weather.meteo_swiss import find_nearest_station
from pyedautils.geopy import get_coordindates_ch_plz, get_altitude_lat_long

# Get coordinates and altitude for Horw (PLZ 6048)
coord = get_coordindates_ch_plz(6048)
altitude = get_altitude_lat_long(coord[0], coord[1])

station = find_nearest_station(coord[0], coord[1], altitude, sensor="temp")
print(station)  # "LUZ"
```

## Find station for different sensor types

Available sensors: `temp`, `globrad`, `relhum`, `rain`

```python
coord = get_coordindates_ch_plz(6197)
altitude = get_altitude_lat_long(coord[0], coord[1])

print(find_nearest_station(coord[0], coord[1], altitude, sensor="temp"))     # "FLU"
print(find_nearest_station(coord[0], coord[1], altitude, sensor="globrad"))  # "BAN"
print(find_nearest_station(coord[0], coord[1], altitude, sensor="relhum"))   # "FLU"
print(find_nearest_station(coord[0], coord[1], altitude, sensor="rain"))     # "FLU"
```

## Get all station data

```python
from pyedautils.weather.meteo_swiss import get_current_station_data

df = get_current_station_data()
print(f"{len(df)} stations found")
print(df[["Abk.", "Station", "Stationshöhe m ü. M."]].head())
```

## SwissMetNet station list

Station metadata of the Open Government Data (abbreviation, name, canton,
altitude, coordinates, ...). Cite as "Source: MeteoSwiss" when reproducing
the data.

```python
from pyedautils.weather.meteo_swiss import get_smn_stations

stations = get_smn_stations()
print(stations[["station_abbr", "station_name", "station_height_masl"]].head())
```

## Daily values for degree days

Daily values have a tz-naive date index (one row per day):

```python
from pyedautils.weather.meteo_swiss import download_data

klo = download_data("KLO", "2023-01-01", "2023-12-31", granularity="d", sensors=["temp", "globrad"])
print(klo.head(3))
#             temp  globrad
# datetime
# 2023-01-01  11.5     64.0
# 2023-01-02   7.9     45.0
# 2023-01-03   7.0     20.0
```

Long series are the input for a station-specific degree-day climatology,
see {doc}`../api/degree_days`:

```python
from pyedautils.degree_days import reference_degree_days

klo_long = download_data("KLO", "1991-01-01", "2020-12-31", granularity="d", sensors=["temp"])
ref = reference_degree_days(klo_long["temp"], period=("1991", "2020"))
print(ref.annual, ref.n_years)  # 3271.46 30
```
