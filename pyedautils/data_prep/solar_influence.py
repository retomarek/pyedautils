"""Detect direct solar influence on a room temperature sensor.

Given a room temperature series, the local outdoor temperature and the
global radiation at the same location, three signals are computed:

1. **Pearson correlation** between global radiation and the excess
   temperature ``dT = t_room - t_outdoor`` (hourly means). A high positive
   correlation indicates direct sun exposure.
2. **Time-of-day peak**: the local hour with the highest mean ``dT``
   during sunny hours, hinting at the sensor orientation (morning ≈ east,
   noon ≈ south, evening ≈ west).
3. **Steep-rise events**: hours with a strong room-temperature rise,
   strong radiation and a large excess temperature — the typical pattern
   of sun hitting a sensor directly.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


def _ns(x):
    """Return a nanosecond-resolution DatetimeIndex from a Series or Index.

    Different datetime units (e.g. ``[us]`` from Parquet) fail to align in
    :func:`pandas.concat`; nanoseconds is the common ground. No-op on
    pandas < 2.0. Works for tz-naive and tz-aware input.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(x))
    as_unit = getattr(idx, "as_unit", None)
    return as_unit("ns") if as_unit is not None else idx


def _hourly(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Hourly mean of a ``[timestamp, value]`` DataFrame or a Series."""
    if isinstance(data, pd.Series):
        s = data.copy()
        s.index = _ns(s.index)
    else:
        df = data.copy()
        df.columns = ["timestamp", "value"]
        df = df.set_index(_ns(df["timestamp"]))
        s = df["value"]
    return s.resample("1h").mean()


def analyze_solar_influence(
    room: Union[pd.Series, pd.DataFrame],
    outdoor: Union[pd.Series, pd.DataFrame],
    radiation: Union[pd.Series, pd.DataFrame],
    *,
    local_tz: Optional[str] = None,
    daytime_radiation: float = 50.0,
    sunny_radiation: float = 100.0,
    event_radiation: float = 200.0,
    event_gradient: float = 1.5,
    event_dt: float = 3.0,
    min_hours: int = 48,
) -> Dict[str, Optional[float]]:
    """Quantify direct solar influence on a room temperature sensor.

    Args:
        room: Room temperature, ``[timestamp, value]`` or Series [°C].
        outdoor: Outdoor temperature, ``[timestamp, value]`` or Series [°C].
        radiation: Global radiation, ``[timestamp, value]`` or Series
            [W/m²].
        local_tz: Optional timezone (e.g. ``"Europe/Zurich"``) used only
            for the ``peak_hour_local``. Input is assumed UTC; if *None*
            the peak hour is reported in the input's own clock.
        daytime_radiation: Radiation above which an hour counts as daytime
            (denominator of the event rate). Default 50 W/m².
        sunny_radiation: Radiation above which an hour counts as sunny
            (time-of-day profile). Default 100 W/m².
        event_radiation: Radiation threshold for a steep-rise event.
            Default 200 W/m².
        event_gradient: Room-temperature rise threshold [K/h] for an event.
            Default 1.5.
        event_dt: Excess-temperature threshold [°C] for an event.
            Default 3.0.
        min_hours: Minimum overlapping hours required; below this all
            metrics are *None*. Default 48.

    Returns:
        dict: ``n_hours``, ``pearson_r``, ``peak_hour_local``,
        ``n_solar_events``, ``n_daytime_hours``, ``event_rate``. Values are
        *None* when fewer than *min_hours* overlapping hours are available.
    """
    room_h = _hourly(room).rename("t_room")
    out_h = _hourly(outdoor).rename("t_out")
    sol_h = _hourly(radiation).rename("solar")

    joined = pd.concat([room_h, out_h, sol_h], axis=1, sort=True).dropna()

    if len(joined) < min_hours:
        return {
            "n_hours": int(len(joined)),
            "pearson_r": None,
            "peak_hour_local": None,
            "n_solar_events": None,
            "n_daytime_hours": None,
            "event_rate": None,
        }

    joined["dT"] = joined["t_room"] - joined["t_out"]
    joined["dT_dt"] = joined["t_room"].diff()

    # 1. Pearson correlation (scipy-free).
    pearson_r = float(np.corrcoef(joined["solar"], joined["dT"])[0, 1])

    # 2. Time-of-day profile during sunny hours.
    sunny = joined[joined["solar"] > sunny_radiation]
    if not sunny.empty:
        idx = sunny.index
        if local_tz is not None:
            tz_idx = idx.tz_localize("UTC").tz_convert(local_tz) \
                if idx.tz is None else idx.tz_convert(local_tz)
            hours = tz_idx.hour
        else:
            hours = idx.hour
        hourly_dt = sunny.groupby(hours)["dT"].mean()
        peak_hour = int(hourly_dt.idxmax())
    else:
        peak_hour = None

    # 3. Steep-rise events.
    events = joined[
        (joined["dT_dt"] > event_gradient)
        & (joined["solar"] > event_radiation)
        & (joined["dT"] > event_dt)
    ]
    n_daytime = int((joined["solar"] > daytime_radiation).sum())
    n_events = int(len(events))
    event_rate = n_events / n_daytime if n_daytime > 0 else 0.0

    return {
        "n_hours": int(len(joined)),
        "pearson_r": round(pearson_r, 3),
        "peak_hour_local": peak_hour,
        "n_solar_events": n_events,
        "n_daytime_hours": n_daytime,
        "event_rate": round(event_rate, 4),
    }
