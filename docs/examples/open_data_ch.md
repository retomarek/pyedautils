# open_data_ch

See {doc}`../api/open_data_ch` for the full API reference.

## BFE: Swiss heating and cooling degree days

Population-weighted monthly HGT 20/12 and CDD 18.3 for Switzerland since
1980 (Bundesamt für Energie / MeteoSchweiz, OGD105).

```python
from pyedautils.weather import download_degree_days_bfe

hgt_ch = download_degree_days_bfe()
print(hgt_ch.tail(3))
#               hgt    cdd
# period
# 2026-07-01    0.2  136.4
# 2026-08-01    0.3  137.4
# 2026-09-01    3.7   25.8
```

## Stadt Zürich: HGT and ATD per site

```python
from pyedautils.weather import download_degree_days_zurich

zh = download_degree_days_zurich("Fluntern")
print(zh.loc["2023"].head(3))
#              station    hgt    atd
# period
# 2023-01-01  Fluntern  522.0  282.0
# 2023-02-01  Fluntern  461.0  237.0
# 2023-03-01  Fluntern  381.0  157.0
```

## Build a reference climatology and correct a consumption

```python
from pyedautils.degree_days import climatology_from_monthly, normalize_consumption

ref = climatology_from_monthly(hgt_ch["hgt"], period=("1991", "2020"))
print(round(ref.annual), ref.n_years)   # 3281 30

# Heating season 2022/2023 of a building: 85 MWh, 80 % space heating
hgt_season = hgt_ch.loc["2022-07":"2023-06", "hgt"].sum()
e_norm = normalize_consumption(85_000, hgt_season, ref.annual, weather_dependent_share=0.8)
print(round(hgt_season), round(e_norm))  # 2830 95858  (season was ~14 % milder than the normal)
```

## Daily MeteoSwiss values for a station-specific reference

```python
from pyedautils.weather import download_daily
from pyedautils.degree_days import reference_degree_days

klo = download_daily("KLO", "1991-01-01", "2020-12-31")
ref_klo = reference_degree_days(klo["tre200d0"], period=("1991", "2020"))
print(ref_klo.monthly.round(0))
```

Source of the daily values: MeteoSwiss (cite as "Source: MeteoSwiss").
