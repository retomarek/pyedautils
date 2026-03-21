"""Proposed Energy Signature (PES) computation.

Implements the iterative algorithm from Eriksson et al. (2020),
"Development and validation of energy signature method", Energy &
Buildings 210, 109756.

Determines balance temperature, heat loss coefficient, domestic hot
water circulation (DHWC) demand, and domestic hot water (DHW) demand
from hourly time-series data.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from pyedautils.data_prep.season import get_season


class PESResult(NamedTuple):
    """Result of the Proposed Energy Signature computation.

    Attributes:
        tb: Balance temperature [°C].
        q_tot: Total heat loss coefficient [kW/K].
        p_dhwc: Domestic hot water circulation demand [kW] (standby).
        p_dhw: Domestic hot water demand [kW].
        p_ihg: Internal heat gains [kW] (input parameter echoed back).
    """

    tb: float
    q_tot: float
    p_dhwc: float
    p_dhw: float
    p_ihg: float


# Keep old field names accessible
PESResult.p_stby = property(lambda self: self.p_dhwc)
PESResult.p_hw = property(lambda self: self.p_dhw)


def compute_pes(
    data: pd.DataFrame,
    p_ihg: float = 0.0,
    max_iter: int = 50,
) -> PESResult:
    """Compute the Proposed Energy Signature parameters.

    The algorithm iteratively determines the balance temperature and
    derives the heat loss coefficient, DHWC demand (P_dhwc) and DHW
    demand (P_dhw) from hourly building data.

    Following Eriksson et al. (2020):

    - **P_dhwc** is the mean of daily minimum power on days where at
      least one hour has T_oa > T_b (Section 3.2.2).
    - **P_dhw** is the mean of all hourly power at T_oa > T_b, minus
      P_dhwc (Section 3.2.3).
    - **Q_tot** is computed from nighttime hours (0:00–4:59) in
      December–February using Eq. (7):
      ``Q_tot = (P_dh + P_ihg - P_dhwc) / (T_room - T_oa)``
      (Section 3.2.1).
    - **T_b** is scanned from 10–20 °C in 0.1 °C steps; the value
      where calculated annual energy is closest to measured is chosen
      (Section 3.2.4, E_tot criterion).

    Args:
        data: DataFrame with columns
            ``[timestamp, outside_temp, power, room_temp]``.
            *timestamp* must be parseable by ``pd.to_datetime``,
            *outside_temp* and *room_temp* in °C, *power* in kW.
        p_ihg: Internal heat gains [kW]. Default 0.
        max_iter: Maximum number of iterations. Default 50.

    Returns:
        PESResult with the computed parameters.

    Raises:
        ValueError: If convergence is not reached within *max_iter*.
    """
    df = data.copy()
    df.columns = ["timestamp", "outside_temp", "power", "room_temp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["season"] = get_season(df["timestamp"])
    df["day"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour

    # Total actual power sum (all hours) for Tb scan
    total_power_actual = df["power"].sum()
    t_oa_all = df["outside_temp"].values

    # Pre-compute nighttime winter data (Dec-Feb, hours 0-4)
    # per Eriksson Section 3.2.1: "12:00 AM – 5:00 AM"
    night_winter = df[
        (df["month"].isin([12, 1, 2])) & (df["hour"].between(0, 4))
    ].copy()

    # Initial balance temperature
    tb = 12.0

    for _ in range(max_iter):
        # === P_dhwc (Section 3.2.2) ===
        # Days with at least one hour where T_oa > Tb
        daily_max_temp_all = df.groupby("day")["outside_temp"].max()
        warm_days = daily_max_temp_all[daily_max_temp_all > tb].index
        warm_day_data = df[df["day"].isin(warm_days)]

        if warm_day_data.empty:
            raise ValueError(
                f"No warm days found with Tb={tb:.1f}. "
                "Check that data spans warm periods."
            )

        # Min power per warm day, then average
        daily_min_power = warm_day_data.groupby("day")["power"].min()
        p_dhwc = daily_min_power.mean()

        # === P_dhw (Section 3.2.3) ===
        # Mean of all hourly power where T_oa > Tb, minus P_dhwc
        warm_hours = df[df["outside_temp"] > tb]
        if warm_hours.empty:  # pragma: no cover — guarded by warm_days check above
            p_dhw = 0.0
        else:
            p_dhw = warm_hours["power"].mean() - p_dhwc

        # === Q_tot (Section 3.2.1, Eq. 7) ===
        # Nighttime (0-4h), Dec-Feb, T_oa < Tb
        night_cold = night_winter[night_winter["outside_temp"] < tb]

        if night_cold.empty:
            raise ValueError(
                f"No nighttime winter data with T_oa < Tb={tb:.1f}. "
                "Check that data includes Dec-Feb."
            )

        # Eq. (7): Q_tot = (P_dh,sup + P_ihg - P_dhwc) /
        #                   (T_indoors - T_outdoors)
        # Use 1-day averaged outdoor temps to account for thermal
        # mass (Section 3.2.1, Table 3)
        night_cold = night_cold.copy()
        daily_mean_t_oa = df.groupby("day")["outside_temp"].mean()
        night_cold["daily_t_oa"] = (
            night_cold["day"].map(daily_mean_t_oa)
        )
        denom = night_cold["room_temp"] - night_cold["daily_t_oa"]
        numer = night_cold["power"] + p_ihg - p_dhwc
        valid = denom.abs() > 0.01
        q_tot = (numer[valid] / denom[valid]).mean()

        if np.isnan(q_tot) or q_tot <= 0:
            raise ValueError(
                "Could not compute a valid heat loss coefficient. "
                "Check input data quality."
            )

        # === T_b (Section 3.2.4) ===
        # Scan 10-20°C in 0.1 steps (paper range)
        # Eq. (9): P_dh,sup = Q_tot * max(0, Tb - T_oa) + P_dhw + P_dhwc
        # Find Tb where sum(P_calc) / sum(P_actual) closest to 100%
        tb_candidates = np.arange(10.0, 20.05, 0.1)
        best_tb = tb
        best_err = float("inf")

        for tb_c in tb_candidates:
            temp_diff = tb_c - t_oa_all
            heating_mask = temp_diff > 0
            power_calc_sum = (
                (q_tot * temp_diff[heating_mask] + p_dhwc + p_dhw).sum()
            )
            perc_diff = 100.0 / total_power_actual * power_calc_sum
            err = abs(perc_diff - 100.0)
            if err < best_err:
                best_err = err
                best_tb = round(tb_c, 1)

        delta = best_tb - tb
        tb = best_tb

        if abs(delta) < 0.1:
            return PESResult(
                tb=round(tb, 1),
                q_tot=round(q_tot, 4),
                p_dhwc=round(p_dhwc, 4),
                p_dhw=round(p_dhw, 4),
                p_ihg=round(p_ihg, 4),
            )

    raise ValueError(
        f"PES algorithm did not converge after {max_iter} iterations."
    )
