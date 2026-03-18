# -*- coding: utf-8 -*-

import logging

import pandas as pd
from pyedautils.geopy import get_distance_between_two_points

logger = logging.getLogger(__name__)


def get_current_station_data() -> pd.DataFrame:
    """
    Gets current measurement data of all Meteo Swiss stations.

    Returns:
        pd.DataFrame: DataFrame with station data.
    """
    try:
        endpoint = "https://data.geo.admin.ch/ch.meteoschweiz.messnetz-automatisch/ch.meteoschweiz.messnetz-automatisch_de.csv"
        data = pd.read_csv(endpoint, encoding='unicode_escape', sep=";")
        df = data[data['Messungen'].notna()]
        return df
    except Exception as e:
        raise ValueError(f"Error in getting data: {e}") from e


def find_nearest_station(lat: float, long: float, altitude: float, sensor: str) -> str:
    """
    Returns station id of closest meteo swiss station to a coordinate.

    Args:
        lat (float): Latitude in decimal degrees.
        long (float): Longitude in decimal degrees.
        altitude (float): Altitude in meters above sea level.
        sensor (str): temp, globrad, relhum or rain

    Returns:
        str: Meteo Swiss station ID
    """
    try:
        switzerland_lat_min = 45.67
        switzerland_lat_max = 47.92
        switzerland_long_min = 5.7
        switzerland_long_max = 10.7

        if not (switzerland_lat_min <= lat <= switzerland_lat_max and
                switzerland_long_min <= long <= switzerland_long_max):
            raise ValueError("Coordinates not in range for Swiss coordinate system")

        all_station_data = get_current_station_data()

        lats = all_station_data["Breitengrad"].astype("float")
        lons = all_station_data["Längengrad"].astype("float")

        distances = pd.DataFrame(
            [get_distance_between_two_points((lat, long), (lats[i], lons[i]))
             for i in range(len(lats))]
        )
        distances.columns = ['value']
        distances = distances.sort_values(by='value')
        distances = distances.reset_index()

        sensor_map = {
            "temp": "Temperatur",
            "globrad": "Globalstrahlung",
            "relhum": "Feuchte",
            "rain": "Niederschlag",
        }

        if sensor not in sensor_map:
            raise ValueError(f"Unknown sensor type: {sensor}. Must be one of {list(sensor_map.keys())}")

        sensor_keyword = sensor_map[sensor]

        i = 0
        while i < len(distances):
            id = distances.loc[i, "index"]
            sensors = all_station_data.loc[id, "Messungen"]
            stationID = all_station_data.loc[id, "Abk."]
            stationIDString = all_station_data.loc[id, "Station"]
            stationAltitude = float(all_station_data.loc[id, "Stationshöhe m ü. M."])
            if (float(altitude) > (stationAltitude - 150.0)) and (float(altitude) < (stationAltitude + 150.0)):
                if sensor_keyword in sensors:
                    logger.info("Closest station for %s: %s", sensor, stationIDString)
                    return stationID
            i += 1

        raise ValueError(
            f"No station found for sensor '{sensor}' within 150m altitude of {altitude}m"
        )

    except Exception as e:
        raise ValueError(f"Error in getting data: {e}") from e
