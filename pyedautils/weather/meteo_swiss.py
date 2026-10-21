# -*- coding: utf-8 -*-

"""MeteoSwiss station lookup and Open Government Data (OGD) downloads.

Daily SwissMetNet values (collection ``ch.meteoschweiz.ogd-smn``) are
available per station as ``historical`` and ``recent`` CSV files. When
reproducing or redistributing the data the source must be cited as
"Source: MeteoSwiss".
"""

import io
import logging
from typing import Sequence

import pandas as pd
import requests

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


# ---------------------------------------------------------------------------
# Open Government Data (SwissMetNet daily values)
# ---------------------------------------------------------------------------
#: Base URL of the SwissMetNet OGD files.
SMN_BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
#: STAC collection of the SwissMetNet OGD files (lists the per-station assets).
SMN_STAC_URL = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
#: Daily parameter codes: mean air temperature 2 m [°C], global radiation [W/m²].
SMN_DAILY_PARAMS = ("tre200d0", "gre000d0")


def _get_text(url: str, timeout: int) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.content.decode("cp1252")


def get_smn_stations(timeout: int = 60) -> pd.DataFrame:
    """
    Gets the SwissMetNet station list of the MeteoSwiss OGD.

    Args:
        timeout (int): Request timeout in seconds. Default 60.

    Returns:
        pd.DataFrame: Station metadata (``station_abbr``, ``station_name``,
        coordinates, altitude, ...).
    """
    try:
        text = _get_text(f"{SMN_BASE_URL}/ogd-smn_meta_stations.csv", timeout)
        return pd.read_csv(io.StringIO(text), sep=";")
    except Exception as e:
        raise ValueError(f"Error in getting station list: {e}") from e


def _daily_asset_urls(station: str, timeout: int) -> list:
    item = requests.get(f"{SMN_STAC_URL}/items/{station.lower()}", timeout=timeout)
    item.raise_for_status()
    assets = item.json().get("assets", {})
    urls = [a["href"] for key, a in assets.items() if "_d_" in key and key.endswith(".csv")]
    if not urls:
        raise ValueError(f"No daily OGD files found for station {station!r}")
    return sorted(urls)


def _read_smn_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=";")
    ts = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M")
    return df.drop(columns=["reference_timestamp", "station_abbr"], errors="ignore").set_index(ts)


def download_daily(
    station: str,
    start_date: str,
    end_date: str,
    params: Sequence[str] = SMN_DAILY_PARAMS,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Downloads daily SwissMetNet values of a station from the MeteoSwiss OGD.

    Combines the ``historical`` and ``recent`` files of the station. Use
    the result as long temperature series for
    :func:`pyedautils.degree_days.reference_degree_days`.
    Source: MeteoSwiss.

    Args:
        station (str): Station abbreviation, e.g. ``"KLO"`` (Zürich-Kloten);
            see :func:`get_smn_stations`.
        start_date (str): First day, ``'YYYY-MM-DD'``.
        end_date (str): Last day (inclusive), ``'YYYY-MM-DD'``.
        params (Sequence[str]): Daily parameter codes to keep. Default
            ``("tre200d0", "gre000d0")``: mean air temperature 2 m [°C]
            and global radiation daily mean [W/m²].
        timeout (int): Request timeout in seconds. Default 60.

    Returns:
        pd.DataFrame: Daily values indexed by ``datetime`` with one column
        per parameter (missing parameters are skipped with a warning).
    """
    try:
        urls = _daily_asset_urls(station, timeout)
        frames = [_read_smn_csv(_get_text(url, timeout)) for url in urls]
    except Exception as e:
        raise ValueError(f"Error downloading MeteoSwiss daily data: {e}") from e

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "datetime"
    missing = [p for p in params if p not in df.columns]
    if missing:
        logger.warning("Parameters %s not available for station %s", missing, station)
    keep = [p for p in params if p in df.columns]
    if not keep:
        raise ValueError(f"None of the parameters {list(params)} available for station {station!r}")
    df = df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date), keep].apply(pd.to_numeric, errors="coerce")
    logger.info("Downloaded %d daily rows for station %s", len(df), station)
    return df
