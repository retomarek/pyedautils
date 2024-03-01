from geopy.geocoders import Nominatim
import pgeocode
import requests
import pandas as pd
import time
from typing import List, Union, Tuple
import math

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
        raise GeocodingError(f"Failed to get altitude for LV95 coordinates, Error: {e}") from e

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
    switzerland_lat_min = 45.67
    switzerland_lat_max = 47.92
    switzerland_long_min = 5.7
    switzerland_long_max = 10.7
    
    if(not((switzerland_lat_min <= lat <= switzerland_lat_max) and (switzerland_long_min <= long <= switzerland_long_max))):
        raise ValueError("Coordinates not in range for Swiss coordinate system LV95")
    
    latitude = str(lat)
    longitude = str(long)
    query = f'https://api.opentopodata.org/v1/eudem25m?locations={latitude},{longitude}'
    try:
        r = requests.get(query).json()
        if(pd.json_normalize(r, 'results')['elevation'][0] == None):
            altitude = 0
        else:
            altitude = float(pd.json_normalize(r, 'results')['elevation'].values[0])
        time.sleep(1)  # API rate limiting
        return round(altitude,1)
    except Exception as e:
        raise GeocodingError(f"Failed to get altitude for WGS84 coordinates: {e}") from e

def get_lat_long_address(address: str) -> Union[List[float], None]:
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
            raise GeocodingError("Failed to geocode address after 3 attempts")

        return [n.latitude, n.longitude]
    except Exception as e:
        raise GeocodingError(f"Failed to geocode address, Error: {e}") from e

def get_coordindates_ch_plz(plz: int) -> Tuple[float, float]:
    """
    Returns latitude and longitude for a Swiss postal code.

    Args:
        plz (int): Postal Code (Postleitzahl)

    Returns:
        tuple (lat: float, lon: float)
    """
    try:
        nomi = pgeocode.Nominatim('ch') 
        data=nomi.query_postal_code(plz)
        coordinates = data.latitude.astype(float), data.longitude.astype(float)
        if(str(coordinates[0]) == "nan"):
            raise GeocodingError(f"Failed to get lat/long from plz {plz}")
    except Exception:
        raise GeocodingError(f"Failed to get lat/long for plz {plz}")
     
    return coordinates

def get_distance_between_two_points(coord1, coord2):
    """
    Calculate the distance between two points on the Earth's surface given their latitude and longitude coordinates.
    
    Args:
        coord1 (tuple): Latitude and longitude of the first point in degrees, as a tuple (lat1, lon1).
        coord2 (tuple): Latitude and longitude of the second point in degrees, as a tuple (lat2, lon2).
        
    Returns:
        float: Distance between the two points in kilometers.
    
    """
    # Radius of Earth in km
    R = 6373.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    # Calculate differences in longitude and latitude
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Apply Haversine formula to calculate distance
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance