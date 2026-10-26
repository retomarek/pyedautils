# -*- coding: utf-8 -*-

"""MeteoSwiss station lookup and Open Government Data (OGD) downloads.

SwissMetNet values (collection ``ch.meteoschweiz.ogd-smn``, precipitation-only
stations in ``ch.meteoschweiz.ogd-smn-precip``) are available per station as
``historical``, ``recent`` and ``now`` CSV files in 10-minute, hourly and daily
granularity and are downloaded with :func:`download_data`. When reproducing
or redistributing the data the source must be cited as "Source: MeteoSwiss".
"""

import io
import logging
import re

import pandas as pd
import requests

from pyedautils.geopy import (
    get_altitude_lat_long,
    get_coordindates_ch_plz,
    get_distance_between_two_points,
)

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
# Open Government Data (SwissMetNet 10-minute, hourly and daily values)
# ---------------------------------------------------------------------------
#: Base URL of the SwissMetNet OGD files.
SMN_BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
#: Root of the STAC collections on data.geo.admin.ch.
OGD_STAC_URL = "https://data.geo.admin.ch/api/stac/v1/collections"
#: SwissMetNet weather stations; precipitation-only stations live in a separate collection.
OGD_COLLECTIONS = ["ch.meteoschweiz.ogd-smn", "ch.meteoschweiz.ogd-smn-precip"]
#: MeteoSwiss parameter IDs per granularity: t = 10 minutes, h = hourly, d = daily.
SENSOR_PARAMS = {
    "t": {"temp": "tre200s0", "globrad": "gre000z0", "relhum": "ure200s0", "rain": "rre150z0"},
    "h": {"temp": "tre200h0", "globrad": "gre000h0", "relhum": "ure200h0", "rain": "rre150h0"},
    "d": {"temp": "tre200d0", "globrad": "gre000d0", "relhum": "ure200d0", "rain": "rre150d0"},
}


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


#: ``parameter_granularity`` in the parameter metadata -> ``granularity`` of :func:`download_data`
GRANULARITY_CODES = {"T": "t", "H": "h", "D": "d"}


def get_smn_parameters(timeout: int = 60) -> pd.DataFrame:
    """
    Gets the parameter list (metadata) of the SwissMetNet data of the MeteoSwiss OGD.

    Args:
        timeout (int): Request timeout in seconds. Default 60.

    Returns:
        pd.DataFrame: One row per parameter with ``parameter_shortname`` (the ID
        used by :func:`download_data`, e.g. "tre200h0"), descriptions and groups
        in de/fr/it/en, ``parameter_granularity`` (T = 10 min, H = hourly,
        D = daily, M = monthly, Y = yearly), ``parameter_datatype`` and
        ``parameter_unit``.
    """
    try:
        text = _get_text(f"{SMN_BASE_URL}/ogd-smn_meta_parameters.csv", timeout)
        return pd.read_csv(io.StringIO(text), sep=";")
    except Exception as e:
        raise ValueError(f"Error in getting parameter list: {e}") from e


def get_station_parameters(
    station: str, granularity: str = None, lang: str = "en", timeout: int = 60
) -> pd.DataFrame:
    """
    Lists the parameters a SwissMetNet station measures, with description and unit.

    Args:
        station (str): Station abbreviation, e.g. "LUZ".
        granularity (str, optional): Only parameters of this resolution: "t"
            (10 minutes), "h" (hourly) or "d" (daily). Default: all.
        lang (str, optional): Language of description and group: "de", "fr",
            "it" or "en". Default "en".
        timeout (int): Request timeout in seconds. Default 60.

    Returns:
        pd.DataFrame: Columns ``parameter`` (ID for :func:`download_data`),
        ``description``, ``group``, ``unit``, ``granularity`` ("t", "h", "d",
        "m", "y"), ``data_since`` and ``data_till`` (NaT while still measured),
        sorted by group and description. Empty if the station is unknown.
    """
    if lang not in ("de", "fr", "it", "en"):
        raise ValueError(f"Unknown language: {lang}. Must be one of de, fr, it, en")
    try:
        text = _get_text(f"{SMN_BASE_URL}/ogd-smn_meta_datainventory.csv", timeout)
        inventory = pd.read_csv(io.StringIO(text), sep=";")
    except Exception as e:
        raise ValueError(f"Error in getting data inventory: {e}") from e
    params = get_smn_parameters(timeout)
    rows = inventory[inventory["station_abbr"] == station.upper()]
    df = rows.merge(params, on="parameter_shortname", how="inner")
    df = pd.DataFrame(
        {
            "parameter": df["parameter_shortname"],
            "description": df[f"parameter_description_{lang}"],
            "group": df[f"parameter_group_{lang}"],
            "unit": df["parameter_unit"],
            "granularity": df["parameter_granularity"].str.lower(),
            "data_since": pd.to_datetime(df["data_since"], format="%d.%m.%Y %H:%M"),
            "data_till": pd.to_datetime(df["data_till"], format="%d.%m.%Y %H:%M"),
        }
    )
    if granularity is not None:
        if granularity not in GRANULARITY_CODES.values():
            raise ValueError(f"Unknown granularity: {granularity}. Must be one of t, h, d")
        df = df[df["granularity"] == granularity]
    return df.sort_values(["group", "description"]).reset_index(drop=True)


def _list_data_files(station: str) -> dict:
    """Returns {file name: url} of all OGD data files of a station."""
    for collection in OGD_COLLECTIONS:
        response = requests.get(f"{OGD_STAC_URL}/{collection}/items/{station.lower()}", timeout=60)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        assets = response.json().get("assets", {})
        return {name: asset["href"] for name, asset in assets.items()}
    raise ValueError(f"Unknown MeteoSwiss station: {station}")


def _select_files(files: dict, granularity: str, start: pd.Timestamp, end: pd.Timestamp,
                  today: pd.Timestamp) -> list:
    """Returns the urls of the files covering [start, end), ordered historical, recent, now."""
    pattern = re.compile(rf"_{granularity}_(historical|recent|now)(?:_(\d{{4}})-(\d{{4}}))?\.csv$")
    order = {"historical": 0, "recent": 1, "now": 2}
    selected = []
    for name, url in files.items():
        match = pattern.search(name)
        if match is None:
            continue
        kind, first_year, last_year = match.groups()
        if kind == "historical":
            first = int(first_year) if first_year else 0
            last = int(last_year) if last_year else today.year - 1
            use = start.year <= last and end.year >= first
        elif kind == "recent":
            # recent: from January 1st of the current year until yesterday
            use = end.year >= today.year
        else:
            # now: since yesterday 12:00 UTC
            use = end >= today - pd.Timedelta(days=1)
        if use:
            selected.append((order[kind], name, url))
    return [url for _, _, url in sorted(selected)]


def _resolve_params(sensors: list, granularity: str) -> dict:
    """Returns {MeteoSwiss parameter ID: column name} for the requested sensors."""
    if granularity not in SENSOR_PARAMS:
        raise ValueError(f"Unknown granularity: {granularity}. Must be one of {list(SENSOR_PARAMS.keys())}")

    params = {}
    for s in sensors:
        if s in SENSOR_PARAMS[granularity]:
            params[SENSOR_PARAMS[granularity][s]] = s
        elif re.fullmatch(r"[a-z0-9]{8}", s):
            params[s] = s
        else:
            raise ValueError(
                f"Unknown sensor type: {s}. Must be one of {list(SENSOR_PARAMS[granularity].keys())} "
                "or a MeteoSwiss parameter ID"
            )
    return params


def _utc_range(start_date: str, end_date: str, tz) -> tuple:
    """Returns [start, end) as tz-naive UTC timestamps; dates are interpreted in tz (None: as is)."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    if tz is not None:
        start = start.tz_localize(tz).tz_convert("UTC").tz_localize(None)
        end = end.tz_localize(tz).tz_convert("UTC").tz_localize(None)
    return start, end


def download_data(
    station: str,
    start_date: str,
    end_date: str,
    sensors: list = None,
    granularity: str = "h",
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Downloads historical measurement data of a MeteoSwiss station from the
    MeteoSwiss open data portal (OGD, data.geo.admin.ch).

    The data is assembled from the historical (per decade), recent (current
    year) and now (last day) files of the station. MeteoSwiss time stamps are
    in UTC and mark the end of the measurement interval (e.g. the hourly value
    at 16:00 covers 15:00-16:00). Daily values are returned with a tz-naive
    date index (one row per day), e.g. as input for
    :func:`pyedautils.degree_days.reference_degree_days`.

    Args:
        station (str): MeteoSwiss station abbreviation, e.g. "KLO" (see
            :func:`find_nearest_station`).
        start_date (str): Start date in 'YYYY-MM-DD' format (inclusive).
        end_date (str): End date in 'YYYY-MM-DD' format (inclusive).
        sensors (list, optional): Sensor types temp, globrad, relhum, rain,
            or raw MeteoSwiss parameter IDs (e.g. "tre200hx"). Defaults to
            ["temp", "globrad", "relhum", "rain"].
        granularity (str, optional): "t" (10 minutes), "h" (hourly) or "d"
            (daily). Defaults to "h".
        tz (str, optional): Time zone of start_date/end_date and of the
            returned index, e.g. "Europe/Zurich". Ignored for daily values.
            Defaults to "UTC".

    Returns:
        pd.DataFrame: DataFrame with datetime index (tz-aware for "t" and
        "h", tz-naive dates for "d") and one column per sensor.
    """
    if sensors is None:
        sensors = ["temp", "globrad", "relhum", "rain"]
    params = _resolve_params(sensors, granularity)
    if granularity == "d":
        tz = None

    start, end = _utc_range(start_date, end_date, tz)
    if end <= start:
        raise ValueError("end_date must not be before start_date")
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    try:
        files = _list_data_files(station)
        urls = _select_files(files, granularity, start, end, today)
        frames = [
            pd.read_csv(
                url, sep=";", encoding="cp1252",
                usecols=lambda c: c == "reference_timestamp" or c in params,
            )
            for url in urls
        ]
    except Exception as e:
        raise ValueError(f"Error downloading MeteoSwiss data: {e}") from e

    columns = list(params.values())
    if not frames:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz=tz, name="datetime"))

    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df.pop("reference_timestamp"), format="%d.%m.%Y %H:%M")
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= start) & (df.index < end)]

    for param in params:
        if param not in df.columns:
            logger.warning("Parameter %s not available for station %s", param, station)
    df = df.rename(columns=params).reindex(columns=columns)
    df = df.apply(pd.to_numeric, errors="coerce")
    if tz is not None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)

    logger.info(
        "Downloaded %d rows for station %s (%s to %s)",
        len(df), station, start_date, end_date,
    )
    return df


def download_data_by_plz(
    plz: int,
    start_date: str,
    end_date: str,
    sensors: list = None,
    granularity: str = "h",
    tz: str = "UTC",
    altitude: float = None,
) -> pd.DataFrame:
    """
    Downloads historical measurement data from the nearest MeteoSwiss
    stations to a Swiss postal code.

    For each requested sensor type, finds the nearest station that has that
    sensor within 150 m altitude difference (see :func:`find_nearest_station`)
    and merges all results into a single DataFrame.

    Args:
        plz (int): Swiss postal code (Postleitzahl).
        start_date (str): Start date in 'YYYY-MM-DD' format (inclusive).
        end_date (str): End date in 'YYYY-MM-DD' format (inclusive).
        sensors (list, optional): Sensor types temp, globrad, relhum, rain.
            Defaults to ["temp", "globrad", "relhum", "rain"].
        granularity (str, optional): "t" (10 minutes), "h" (hourly) or "d"
            (daily). Defaults to "h".
        tz (str, optional): Time zone of start_date/end_date and of the
            returned index. Defaults to "UTC".
        altitude (float, optional): Altitude in meters above sea level used
            for the station search. Defaults to the terrain altitude at the
            postal code's coordinates.

    Returns:
        pd.DataFrame: DataFrame with datetime index (tz-aware for "t" and
        "h", tz-naive dates for "d") and one column per sensor.
    """
    if sensors is None:
        sensors = ["temp", "globrad", "relhum", "rain"]

    lat, lon = get_coordindates_ch_plz(plz)
    if altitude is None:
        altitude = get_altitude_lat_long(lat, lon)
    logger.info("PLZ %d resolved to lat=%.4f, lon=%.4f, altitude=%.0f m", plz, lat, lon, altitude)

    # Group sensors by nearest station to minimize downloads
    station_sensors = {}
    for sensor in sensors:
        station = find_nearest_station(lat, lon, altitude, sensor=sensor)
        station_sensors.setdefault(station, []).append(sensor)

    result = None
    for station, sensor_list in station_sensors.items():
        df = download_data(station, start_date, end_date, sensors=sensor_list,
                           granularity=granularity, tz=tz)
        result = df if result is None else result.join(df, how="outer")

    return result[sensors]
