"""Proposed Energy Signature (PES) computation.

Implements the iterative algorithm from Eriksson et al. (2020) to find
the balance temperature, heat loss coefficient, standby power, and
hot-water power of a building from hourly time-series data.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from pyedautils.season import get_season


class PESResult(NamedTuple):
    """Result of the Proposed Energy Signature computation.

    Attributes:
        tb: Balance temperature [°C].
        q_tot: Total heat loss coefficient [kW/K].
        p_stby: Standby power [kW].
        p_hw: Hot-water power [kW].
        p_ihg: Internal heat gains [kW] (input parameter echoed back).
    """

    tb: float
    q_tot: float
    p_stby: float
    p_hw: float
    p_ihg: float


def compute_pes(
    data: pd.DataFrame,
    p_ihg: float = 0.0,
    max_iter: int = 50,
) -> PESResult:
    """Compute the Proposed Energy Signature parameters.

    The algorithm iteratively determines the balance temperature and
    derives the heat loss coefficient, standby power, and hot-water
    power from hourly building data.

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
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["year"] = df["timestamp"].dt.year
    df["hour"] = df["timestamp"].dt.hour

    # Pre-compute weekly aggregates
    weekly_mean_temp = df.groupby(["year", "week"])["outside_temp"].mean()
    weekly_min_power = df.groupby(["year", "week"])["power"].min()
    weekly_mean_power = df.groupby(["year", "week"])["power"].mean()

    # Pre-compute heating season data (Jan/Feb/Mar)
    heating_all = df[df["timestamp"].dt.month.isin([1, 2, 3])].copy()
    daily_max_temp = heating_all.groupby("day")["outside_temp"].max()

    # Initial balance temperature
    tb = 12.0

    for _ in range(max_iter):
        # --- Warm weeks: weekly mean outside temp > Tb ---
        warm_mask = weekly_mean_temp > tb
        if not warm_mask.any():
            raise ValueError(
                f"No warm weeks found with Tb={tb:.1f}. "
                "Check that data spans warm periods."
            )

        # pStby = mean of (min power per warm week)
        p_stby = weekly_min_power[warm_mask].mean()

        # pHw = mean of (mean power per warm week) - pStby
        p_hw = weekly_mean_power[warm_mask].mean() - p_stby

        # --- Heating season filter: daily max(TOa) < Tb ---
        cold_days = daily_max_temp[daily_max_temp < tb].index
        heating = heating_all[heating_all["day"].isin(cold_days)]

        if heating.empty:
            raise ValueError(
                f"No cold days found with Tb={tb:.1f}. "
                "Check that data includes winter months."
            )

        # Daily means for heating period
        daily = heating.groupby("day").agg(
            mean_power=("power", "mean"),
            mean_t_oa=("outside_temp", "mean"),
            mean_t_room=("room_temp", "mean"),
        )

        # q_tot from daily means
        numerator = daily["mean_power"] - p_stby - p_hw + p_ihg
        denominator = daily["mean_t_room"] - daily["mean_t_oa"]
        valid = denominator.abs() > 0.01
        q_tot = (numerator[valid] / denominator[valid]).mean()

        if np.isnan(q_tot) or q_tot <= 0:
            raise ValueError(
                "Could not compute a valid heat loss coefficient. "
                "Check input data quality."
            )

        # --- Find new Tb by scanning 10..30 in 0.1 steps ---
        # At balance temperature, heating power = 0, so:
        #   P_actual = q_tot * (Tb - T_oa) + P_stby + P_hw - P_ihg
        # Solve for Tb from mean heating period data:
        mean_power = daily["mean_power"].mean()
        mean_t_oa = daily["mean_t_oa"].mean()
        # Tb = (mean_power - p_stby - p_hw + p_ihg) / q_tot + mean_t_oa
        tb_candidates = np.arange(10.0, 30.05, 0.1)
        best_tb = tb
        best_err = float("inf")

        for tb_c in tb_candidates:
            p_calc = q_tot * (tb_c - mean_t_oa) + p_stby + p_hw - p_ihg
            err = abs(p_calc - mean_power)
            if err < best_err:
                best_err = err
                best_tb = round(tb_c, 1)

        delta = best_tb - tb
        tb = best_tb

        if abs(delta) < 0.1:
            return PESResult(
                tb=round(tb, 1),
                q_tot=round(q_tot, 4),
                p_stby=round(p_stby, 4),
                p_hw=round(p_hw, 4),
                p_ihg=round(p_ihg, 4),
            )

    raise ValueError(
        f"PES algorithm did not converge after {max_iter} iterations."
    )
