# meteo_swiss examples

See {doc}`../api/meteo_swiss` for the full API reference.

## Find the nearest temperature station

```python
from pyedautils.meteo_swiss import find_nearest_station
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
from pyedautils.meteo_swiss import get_current_station_data

df = get_current_station_data()
print(f"{len(df)} stations found")
print(df[["Abk.", "Station", "Stationshöhe m ü. M."]].head())
```
