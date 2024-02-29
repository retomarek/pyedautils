from geopy.geocoders import Nominatim
import requests
import pandas as pd
import time
from typing import List, Union

class GeocodingError(Exception):
    pass

def convert_wsg84_to_lv95(lat: float, long: float) -> List[float]:
    """
    Converts WGS84 latitude and longitude coordinates to Swiss coordinate system LV95.

    Args:
        lat (float): Latitude in decimal degrees.
        long (float): Longitude in decimal degrees.

    Returns:
        List[float]: A list containing x and y coordinates in LV95 system, e.g., [xcoord, ycoord].
    """
    if not (5.2 <= long <= 11) or not (45.4 <= lat <= 48.2):
        raise GeocodingError("Coordinates not in range for Swiss coordinate system LV95")

    latitude = str(lat)
    longitude = str(long)
    query = f'http://geodesy.geo.admin.ch/reframe/wgs84tolv95?easting={longitude}&northing={latitude}'
    try:
        r = requests.get(query).json()
        coord_list_lv95 = r.get("coordinates", [])
        return coord_list_lv95
    except Exception as e:
        raise GeocodingError(f"Failed to convert WGS84 to LV95: {e}") from e

def get_altitude_lv95(coord_list_lv95: List[float]) -> float:
    """
    Returns altitude in meters above sea level for the given LV95 coordinates.
    The geo.admin.ch api gets used.

    Args:
        coord_list_lv95 (List[float]): LV95 coordinates as [xcoord, ycoord].

    Returns:
        float: Altitude in meters above sea level.
    """
    query = f'https://api3.geo.admin.ch/rest/services/height?easting={coord_list_lv95[0]}&northing={coord_list_lv95[1]}'
    try:
        r = requests.get(query).json()
        altitude = float(r.get("height", 0))
        return altitude
    except Exception as e:
        raise GeocodingError(f"Failed to get altitude for LV95 coordinates: {e}") from e

def get_altitude_lat_long(lat: float, long: float) -> float:
    """
    Returns altitude in meters above sea level for the given WGS84 coordinates.
    The opentopodata.org api gets used.

    Args:
        lat (float): Latitude in decimal degrees.
        long (float): Longitude in decimal degrees.

    Returns:
        float: Altitude in meters above sea level.
    """
    latitude = str(lat)
    longitude = str(long)
    query = f'https://api.opentopodata.org/v1/eudem25m?locations={latitude},{longitude}'
    try:
        r = requests.get(query).json()
        altitude = float(pd.json_normalize(r, 'results')['elevation'].values[0])
        time.sleep(1)  # API rate limiting
        return round(altitude,1)
    except Exception as e:
        raise GeocodingError(f"Failed to get altitude for WGS84 coordinates: {e}") from e

def get_lat_long(address: str) -> Union[List[float], None]:
    """
    Returns latitude and longitude coordinates for the given address.

    Args:
        address (str): The address to geocode.

    Returns:
        Union[List[float], None]: A list containing latitude and longitude coordinates, e.g., [latitude, longitude], or None if the address could not be geocoded.
    """
    try:
        nom = Nominatim(user_agent="myPythonScript")
        for _ in range(3):
            n = nom.geocode(address)
            if n is not None:
                break
            time.sleep(1)  # API rate limiting
        else:
            print("Failed to geocode address after 3 attempts")
            return None

        return [n.latitude, n.longitude]
    except Exception as e:
        raise GeocodingError(f"Failed to geocode address: {e}") from e
