# psychrometrics

Moist-air thermodynamics behind the Mollier h,x-diagram: saturation
pressure, dew point, wet-bulb temperature, enthalpy and density, the
{func}`~pyedautils._mollier.state` constructor for a complete moist-air
state, and air-handling processes (heating, cooling, isothermal and
adiabatic humidification, mixing, heat recovery) that can be chained and
summarised.

The functions live in the module `pyedautils._mollier`:

```python
from pyedautils._mollier import state, heat, mix, pressure_from_altitude
```

See {doc}`../examples/mollier_hx` for usage examples.

## API Reference

```{eval-rst}
.. automodule:: pyedautils._mollier
   :members:
   :undoc-members:
   :show-inheritance:
```
