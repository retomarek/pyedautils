# -*- coding: utf-8 -*-
"""Swiss open-data degree-day series (BFE and Stadt Zürich).

Monthly heating degree days published as Open Government Data, useful as
reference series for :mod:`pyedautils.degree_days`:

- **BFE** (Bundesamt für Energie, OGD105): population-weighted Swiss
  heating degree days HGT 20/12 and cooling degree days CDD 18.3 per
  month since 1980, computed by MeteoSwiss per postal-code area.
- **Stadt Zürich** (UMW103OD1031): heating degree days HGT 20/12 and
  accumulated temperature differences ATD (base 12 °C) per month since
  1864 for the sites Fluntern, Stampfenbachstrasse, Schimmelstrasse,
  Rosengartenstrasse and Heubeeribühl.
"""

import io
import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

#: BFE OGD105 "Monatliche Heizgradtage und Kühlgradtage in der Schweiz (gewichtet)".
BFE_DEGREE_DAYS_URL = "https://www.bfe-ogd.ch/ogd105_heizgradtage_k%C3%BChlgradtage.csv"
#: Stadt Zürich "Heizgradtage und akkumulierte Temperaturdifferenzen".
ZURICH_DEGREE_DAYS_URL = (
    "https://data.stadt-zuerich.ch/dataset/umw_heizgradtage_standort_jahr_monat_od1031/"
    "download/UMW103OD1031.csv"
)
#: Measurement sites of the Stadt Zürich degree-day series.
ZURICH_STATIONS = (
    "Fluntern", "Stampfenbachstrasse", "Schimmelstrasse", "Rosengartenstrasse", "Heubeeribühl",
)


def _fetch_csv(url: str, timeout: int, **read_kwargs) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        response.encoding = "utf-8-sig"
        return pd.read_csv(io.StringIO(response.text), **read_kwargs)
    except Exception as e:
        raise ValueError(f"Error downloading degree days from {url}: {e}") from e


def _month_index(df: pd.DataFrame, year_col: str, month_col: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime({
        "year": df[year_col].astype(int), "month": df[month_col].astype(int), "day": 1,
    }), name="period")


def download_degree_days_bfe(timeout: int = 60) -> pd.DataFrame:
    """
    Downloads the monthly population-weighted Swiss degree days of the BFE.

    HGT: heating degree days 20/12 (sum of ``20 − T_mean`` on days with
    ``T_mean <= 12 °C``). CDD: cooling degree days with base 18.3 °C.
    Source: Bundesamt für Energie / MeteoSchweiz (OGD105).

    Args:
        timeout (int): Request timeout in seconds. Default 60.

    Returns:
        pd.DataFrame: Indexed by month start (``period``) with columns
        ``hgt`` and ``cdd`` [Kd]. Pass ``df["hgt"]`` to
        :func:`pyedautils.degree_days.climatology_from_monthly`.
    """
    raw = _fetch_csv(BFE_DEGREE_DAYS_URL, timeout)
    try:
        df = pd.DataFrame({
            "hgt": pd.to_numeric(raw["HGT"], errors="coerce").to_numpy(),
            "cdd": pd.to_numeric(raw["CDD"], errors="coerce").to_numpy(),
        }, index=_month_index(raw, "Jahr", "Monat"))
    except KeyError as e:
        raise ValueError(f"Unexpected BFE CSV format, missing column {e}") from e
    df = df.sort_index()
    logger.info("Downloaded %d months of BFE degree days (%s to %s)",
                len(df), df.index.min().date(), df.index.max().date())
    return df


def download_degree_days_zurich(station: Optional[str] = None, timeout: int = 60) -> pd.DataFrame:
    """
    Downloads the monthly degree days of the Stadt Zürich open data.

    HGT: heating degree days 20/12. ATD: accumulated temperature difference
    with base 12 °C (SIA 380). Source: Stadt Zürich, Umwelt- und
    Gesundheitsschutz / MeteoSchweiz (UMW103OD1031).

    Args:
        station (str, optional): One of :data:`ZURICH_STATIONS`
            (e.g. ``"Fluntern"``). Default None returns all sites with a
            ``station`` column.
        timeout (int): Request timeout in seconds. Default 60.

    Returns:
        pd.DataFrame: Indexed by month start (``period``) with columns
        ``station``, ``hgt`` and ``atd`` [Kd].
    """
    if station is not None and station not in ZURICH_STATIONS:
        raise ValueError(f"Unknown station {station!r}. Must be one of {ZURICH_STATIONS}")
    raw = _fetch_csv(ZURICH_DEGREE_DAYS_URL, timeout)
    try:
        df = pd.DataFrame({
            "station": raw["Standort"].astype(str).to_numpy(),
            "hgt": pd.to_numeric(raw["Heizgradtag"], errors="coerce").to_numpy(),
            "atd": pd.to_numeric(raw["akkumulierteTemperaturdifferenz"], errors="coerce").to_numpy(),
        }, index=_month_index(raw, "Jahr", "Monat"))
    except KeyError as e:
        raise ValueError(f"Unexpected Stadt Zürich CSV format, missing column {e}") from e
    if station is not None:
        df = df[df["station"] == station]
    df = df.sort_index()
    logger.info("Downloaded %d rows of Stadt Zürich degree days", len(df))
    return df
