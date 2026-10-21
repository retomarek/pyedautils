# degree_days

Degree days and climate correction (Witterungsbereinigung) of heating
energy — Swiss heating degree days HGT 20/12 (SIA 381/3), accumulated
temperature differences ATD (SIA 380:2015 / ISO 15927-6), plain degree
days GT and cooling degree days CDD, a reference climatology, the
correction factor `DD_ref / DD_actual`, a fixed-base degree-day
regression with ASHRAE Guideline 14 metrics, and the BFE/Prognos GT&S
method (degree days and solar radiation).

See {doc}`../examples/degree_days` for usage examples. The plotting
counterparts live in {doc}`plots`; Swiss reference series can be
downloaded with {doc}`open_data_ch` and {doc}`meteo_swiss`.

## Background

| Method | Definition | Source |
|---|---|---|
| HGT 20/12 | Σ (20 °C − T_mean) over days with T_mean ≤ 12 °C; heating days HT | SIA 381/3, MeteoSwiss, BFE |
| ATD | Σ max(0, T_base − T_mean) with one base (heating limit), ideally building-specific | SIA 380:2015, SN EN ISO 15927-6 |
| GT | Σ (20 °C − T_mean) over all days below 20 °C, no heating limit | BFE/Prognos GT&S |
| CDD | Σ (T_mean − 18.3 °C) over days with T_mean ≥ 18.3 °C | BFE |
| Correction | `E_norm = E · (1 − s) + E · s · DD_ref / DD_actual`, with `s` the weather-dependent share | SIA 380, cantonal statistics |
| GT&S | Σ_m share_m · (1 + e_GT,m · ΔGT_m + e_S,m · ΔS_m) with functional elasticities `e = c + a · Δ` | Prognos 2003/2015 for the BFE |

## API Reference

```{eval-rst}
.. automodule:: pyedautils.degree_days
   :members:
   :undoc-members:
   :show-inheritance:
```
