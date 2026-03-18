# geopy examples

See {doc}`../api/geopy` for the full API reference.

## Get coordinates from an address

```python
from pyedautils.geopy import get_lat_long_address

coords = get_lat_long_address("Technikumstrasse 21, 6048 Horw, Switzerland")
print(coords)  # [47.014, 8.305]
```

## Get coordinates from a Swiss postal code

```python
from pyedautils.geopy import get_coordindates_ch_plz

lat, lon = get_coordindates_ch_plz(6048)
print(f"Horw: {lat:.4f}, {lon:.4f}")  # Horw: 47.0108, 8.3039
```

## Convert WGS84 to Swiss LV95

```python
from pyedautils.geopy import convert_wsg84_to_lv95

lv95 = convert_wsg84_to_lv95(47.0133, 8.3061)
print(f"E: {lv95[0]:.0f}, N: {lv95[1]:.0f}")  # E: 2665945, N: 1207280
```

## Get altitude from coordinates

```python
from pyedautils.geopy import get_altitude_lat_long, get_altitude_lv95

# From WGS84 (uses opentopodata.org)
alt = get_altitude_lat_long(47.0145, 8.3062)
print(f"Altitude: {alt} m")  # ~440 m

# From LV95 (uses geo.admin.ch)
alt_lv95 = get_altitude_lv95([2665960.0, 1207350.0])
print(f"Altitude: {alt_lv95} m")  # ~440 m
```

## Calculate distance between two locations

```python
from pyedautils.geopy import get_coordindates_ch_plz, get_distance_between_two_points

horw = get_coordindates_ch_plz(6048)
interlaken = get_coordindates_ch_plz(3800)
dist = get_distance_between_two_points(horw, interlaken)
print(f"Distance: {dist} km")  # ~50 km
```
