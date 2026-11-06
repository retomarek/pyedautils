# psychrometrics

Moist-air thermodynamics behind the Mollier h,x-diagram: saturation
pressure, dew point, wet-bulb temperature, enthalpy and density, the
{func}`~pyedautils.psychrometrics.state` constructor for a complete moist-air
state, and air-handling processes (heating, cooling, isothermal and
adiabatic humidification, mixing, heat recovery) that can be chained and
summarised.

The public module is `pyedautils.psychrometrics`:

```python
from pyedautils.psychrometrics import state, heat, mix, pressure_from_altitude

p = pressure_from_altitude(450)   # ISA barometric formula: 96 035 Pa at 450 m
```

`pressure_from_altitude` is also what {func}`~pyedautils.plots.comfort.plot_mollier_hx`,
{func}`~pyedautils.plots.comfort.plot_mollier_hx_fast` and `state(altitude=...)` use
when only an altitude is given, so passing `altitude=450` is enough there.

```{note}
The implementation lives in the private module `pyedautils._mollier`. Imports
from there keep working, but only the names exported by
`pyedautils.psychrometrics` are public API.
```

See {doc}`../examples/mollier_hx` for usage examples.

```{tip}
New to the h,x diagram? {doc}`../guides/mollier_hx` explains {ref}`how to read it correctly <hx-reading-the-diagram>` (relative vs. absolute humidity, pressure, mould at cold surfaces), {ref}`comfort <hx-comfort>` and {ref}`saving energy <hx-saving-energy>`, with worked examples computed with these functions.
```

## API Reference

```{eval-rst}
.. automodule:: pyedautils.psychrometrics
   :members:
   :undoc-members:
   :show-inheritance:
```
