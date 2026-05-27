# -*- coding: utf-8 -*-

"""Solar geometry helpers (sun elevation and azimuth) based on PyEphem."""

import math
from datetime import datetime
from typing import Tuple, Union

import ephem
import pandas as pd

__all__ = ["sun_position"]

# Standard sea-level pressure in mbar/hPa. ephem uses this to model atmospheric
# refraction; setting the observer pressure to 0 disables refraction and yields
# the true geometric position.
_STANDARD_PRESSURE = 1013.25


def _make_observer(latitude: float, longitude: float, altitude: float, refraction: bool) -> ephem.Observer:
    obs = ephem.Observer()
    # ephem interprets float angles as radians but strings as degrees, so pass
    # the coordinates as strings to keep them in decimal degrees.
    obs.lat = str(float(latitude))
    obs.lon = str(float(longitude))
    obs.elevation = float(altitude)
    obs.pressure = _STANDARD_PRESSURE if refraction else 0
    return obs


def _to_utc_naive_index(timestamp) -> pd.DatetimeIndex:
    """Normalise any datetime-like input to a timezone-naive UTC DatetimeIndex.

    Timezone-aware values are converted to UTC; naive values are assumed to
    already be UTC (ephem expects UTC).
    """
    dti = pd.DatetimeIndex(pd.to_datetime(timestamp))
    if dti.tz is not None:
        dti = dti.tz_convert("UTC").tz_localize(None)
    return dti


def sun_position(
    timestamp,
    latitude: float,
    longitude: float,
    *,
    altitude: float = 0.0,
    refraction: bool = True,
) -> Union[Tuple[float, float], pd.DataFrame]:
    """Solar elevation and azimuth for the given time(s) and location.

    Args:
        timestamp: A single datetime/``pandas.Timestamp`` or a collection of
            timestamps (``pandas.Series``, ``DatetimeIndex``, list or array).
            Timezone-aware values are converted to UTC; naive values are
            assumed to be UTC.
        latitude: Observer latitude in decimal degrees (north positive).
        longitude: Observer longitude in decimal degrees (east positive).
        altitude: Observer height above sea level in meters. Default 0.
        refraction: If True (default), return the apparent position including
            atmospheric refraction. If False, return the true geometric
            position (no refraction).

    Returns:
        For scalar input: a ``(elevation, azimuth)`` tuple in degrees.
        For a collection: a ``pandas.DataFrame`` with columns ``elevation`` and
        ``azimuth`` (degrees), indexed like the input (a ``Series`` keeps its
        index, otherwise a UTC-naive ``DatetimeIndex`` is used).

        Elevation is the angle above the horizon (negative when the sun is
        below the horizon). Azimuth is measured clockwise from geographic
        north (0=N, 90=E, 180=S, 270=W).
    """
    obs = _make_observer(latitude, longitude, altitude, refraction)
    sun = ephem.Sun()

    # Scalar input -> return a single (elevation, azimuth) tuple.
    if isinstance(timestamp, (datetime, pd.Timestamp, str)):
        ts = pd.Timestamp(timestamp)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        obs.date = ts.to_pydatetime()
        sun.compute(obs)
        return math.degrees(float(sun.alt)), math.degrees(float(sun.az))

    # Collection input -> compute per timestamp and return a DataFrame.
    # Keep the caller's original index/labels; only the values fed to ephem are
    # normalised to UTC.
    if isinstance(timestamp, pd.Series):
        out_index = timestamp.index
    elif isinstance(timestamp, pd.DatetimeIndex):
        out_index = timestamp
    else:
        out_index = None
    dti = _to_utc_naive_index(timestamp)
    if out_index is None:
        out_index = dti

    elevations = []
    azimuths = []
    for dt in dti.to_pydatetime():
        obs.date = dt
        sun.compute(obs)
        elevations.append(math.degrees(float(sun.alt)))
        azimuths.append(math.degrees(float(sun.az)))

    return pd.DataFrame(
        {"elevation": elevations, "azimuth": azimuths},
        index=out_index,
    )
