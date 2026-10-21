# meteo_swiss

See {doc}`../api/meteo_swiss` for the full API reference.

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

## Download daily values (Open Government Data)

Daily SwissMetNet values per station, combined from the `historical` and
`recent` OGD files. Default parameters: `tre200d0` (daily mean air
temperature 2 m, °C) and `gre000d0` (global radiation daily mean, W/m²).
Cite as "Source: MeteoSwiss" when reproducing the data.

```python
from pyedautils.weather.meteo_swiss import get_smn_stations, download_daily

stations = get_smn_stations()
print(stations[["station_abbr", "station_name", "station_height_masl"]].head())

klo = download_daily("KLO", "2023-01-01", "2023-12-31")
print(klo.head(3))
#             tre200d0  gre000d0
# datetime
# 2023-01-01      11.5      64.0
# 2023-01-02       7.9      45.0
# 2023-01-03       7.5      42.0
```

Long series are the input for a station-specific degree-day climatology,
see {doc}`../api/degree_days`:

```python
from pyedautils.degree_days import reference_degree_days

klo_long = download_daily("KLO", "1991-01-01", "2020-12-31")
ref = reference_degree_days(klo_long["tre200d0"], period=("1991", "2020"))
print(ref.annual, ref.n_years)
```
