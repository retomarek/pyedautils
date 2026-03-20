# -*- coding: utf-8 -*-

import logging

import pandas as pd
import requests

from pyedautils.geopy import (
    get_distance_between_two_points,
    get_coordindates_ch_plz,
)

logger = logging.getLogger(__name__)

SENSOR_MAP = {
    "temp": 1,
    "globrad": 11,
    "relhum": 4,
    "rain": 6,
}


def get_station_data() -> pd.DataFrame:
    """
    Fetches all agrometeo.ch weather stations.

    Returns:
        pd.DataFrame: DataFrame with station info (id, name, lat, lon, sensors).
    """
    try:
        endpoint = "https://www.agrometeo.ch/backend/api/map/models/17/stations"
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        data = response.json()

        rows = []
        for station in data:
            rows.append({
                "id": station["id"],
                "name": station.get("name", ""),
                "lat": station.get("lat"),
                "lon": station.get("lng"),
                "sensors": [s["id"] for s in station.get("sensors", [])],
            })

        df = pd.DataFrame(rows)
        logger.info("Fetched %d agrometeo stations", len(df))
        return df
    except Exception as e:
        raise ValueError(f"Error fetching agrometeo station data: {e}") from e


def find_nearest_station(lat: float, lon: float, sensor: str = None) -> int:
    """
    Returns station ID of the closest agrometeo.ch station to a coordinate.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        sensor (str, optional): Filter by sensor type: temp, globrad, relhum, or rain.

    Returns:
        int: Agrometeo station ID.
    """
    if sensor is not None and sensor not in SENSOR_MAP:
        raise ValueError(
            f"Unknown sensor type: {sensor}. Must be one of {list(SENSOR_MAP.keys())}"
        )

    stations = get_station_data()

    if sensor is not None:
        sensor_id = SENSOR_MAP[sensor]
        stations = stations[stations["sensors"].apply(lambda s: sensor_id in s)]
        if stations.empty:
            raise ValueError(f"No agrometeo station found with sensor '{sensor}'")

    stations = stations.copy()
    stations["distance"] = stations.apply(
        lambda row: get_distance_between_two_points(
            (lat, lon), (row["lat"], row["lon"])
        ),
        axis=1,
    )
    stations = stations.sort_values("distance")
    nearest = stations.iloc[0]

    logger.info(
        "Nearest agrometeo station for %s: %s (id=%d, %.1f km)",
        sensor or "any",
        nearest["name"],
        nearest["id"],
        nearest["distance"],
    )
    return int(nearest["id"])


def download_data(
    station_id: int,
    start_date: str,
    end_date: str,
    sensors: list = None,
) -> pd.DataFrame:
    """
    Downloads hourly weather data from agrometeo.ch for a station.

    Args:
        station_id (int): Agrometeo station ID.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        sensors (list, optional): List of sensor types to download.
            Defaults to ["temp", "globrad", "relhum", "rain"].

    Returns:
        pd.DataFrame: DataFrame with datetime index and sensor columns.
    """
    if sensors is None:
        sensors = ["temp", "globrad", "relhum", "rain"]

    for s in sensors:
        if s not in SENSOR_MAP:
            raise ValueError(
                f"Unknown sensor type: {s}. Must be one of {list(SENSOR_MAP.keys())}"
            )

    sensor_ids = [str(SENSOR_MAP[s]) for s in sensors]
    sensor_param = "%2C".join(sensor_ids)

    url = (
        f"https://www.agrometeo.ch/backend/api/meteo/station/{station_id}"
        f"/overview?from={start_date}&to={end_date}"
        f"&sensors={sensor_param}&scale=hour&groupBy=station"
    )

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise ValueError(f"Error downloading agrometeo data: {e}") from e

    try:
        rows = []
        for entry in data.get("data", []):
            row = {"datetime": entry.get("date")}
            for sensor_data in entry.get("sensors", []):
                sid = sensor_data.get("id")
                # Reverse lookup sensor name from id
                name = next(
                    (k for k, v in SENSOR_MAP.items() if v == sid), str(sid)
                )
                row[name] = sensor_data.get("avg")
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")

        logger.info(
            "Downloaded %d rows for station %d (%s to %s)",
            len(df), station_id, start_date, end_date,
        )
        return df
    except Exception as e:
        raise ValueError(f"Error parsing agrometeo data: {e}") from e


def download_data_by_plz(
    plz: int,
    start_date: str,
    end_date: str,
    sensors: list = None,
) -> pd.DataFrame:
    """
    Downloads hourly weather data from agrometeo.ch for the nearest station to a Swiss postal code.

    For each requested sensor type, finds the nearest station that has that sensor
    and merges all results into a single DataFrame.

    Args:
        plz (int): Swiss postal code (Postleitzahl).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        sensors (list, optional): List of sensor types to download.
            Defaults to ["temp", "globrad", "relhum", "rain"].

    Returns:
        pd.DataFrame: DataFrame with datetime index and sensor columns.
    """
    if sensors is None:
        sensors = ["temp", "globrad", "relhum", "rain"]

    lat, lon = get_coordindates_ch_plz(plz)
    logger.info("PLZ %d resolved to lat=%.4f, lon=%.4f", plz, lat, lon)

    # Group sensors by nearest station to minimize API calls
    station_sensors = {}
    for sensor in sensors:
        station_id = find_nearest_station(lat, lon, sensor=sensor)
        station_sensors.setdefault(station_id, []).append(sensor)

    result = None
    for station_id, sensor_list in station_sensors.items():
        df = download_data(station_id, start_date, end_date, sensors=sensor_list)
        if result is None:
            result = df
        else:
            result = result.join(df, how="outer")

    if result is None:
        return pd.DataFrame()

    return result
