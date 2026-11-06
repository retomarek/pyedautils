# comfort

Thermal comfort and overheating metrics per SIA 180:2014 — adaptive
boundary curves, summer-half-year filter, overheating hours and KPIs
(including the Minergie and SIA 180 residential limits).

It also classifies samples for the comfort compass
(`comfort_compass_distribution`, with an optional absolute-humidity limit) and
turns its severity-stage limits into the ± offsets of the orange / red warning
bands of `plot_mollier_hx` (`comfort_warning_bands`), so the h,x diagram and
the compass share one definition.

See {doc}`../examples/comfort` for usage examples. The plotting
counterparts live in {doc}`plots`.

## API Reference

```{eval-rst}
.. automodule:: pyedautils.comfort
   :members:
   :undoc-members:
   :show-inheritance:
```
