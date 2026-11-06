# The h,x diagram in five minutes

The [Mollier h,x diagram](https://en.wikipedia.org/wiki/Psychrometrics#Mollier_diagram)
is the map engineers use for moist air. In English-speaking countries the same
physics is usually drawn as the psychrometric chart, rotated and mirrored.
pyedautils draws it with {func}`~pyedautils.plots.comfort.plot_mollier_hx` (and the
lighter {func}`~pyedautils.plots.comfort.plot_mollier_hx_fast`); the numbers behind it
come from the {doc}`psychrometric functions <../api/psychrometrics>` in
`pyedautils._mollier`.

```{admonition} How the numbers on this page were computed
:class: note
All values on this page were calculated with pyedautils itself, at
**450 m above sea level**: `pressure_from_altitude(450)` gives
**96 035 Pa**. Saturation pressure comes from the piecewise polynomial in
Glück, *Zustands- und Stoffwerte* (ch. 2.1), **not** from the Magnus
formula; between 0 and 25 °C the two differ by less than 0.2 %.
At sea level (101 325 Pa) the same air holds about 5 % less water per kg
(for example 7.7 instead of 8.2 g/kg at 21 °C and 50 %). Dew points and
relative humidities do not depend on the pressure.
```

## Why temperature and humidity belong together

Comfort, health and the risk of mould always depend on temperature **and**
humidity at the same time. Looking at only one of them, or at two separate
graphs, can easily mislead you:

- **Relative humidity changes with temperature.** Air at 21 °C and 60 % holds
  about 9.8 g of water per kg of dry air. Warm the same air to 26 °C and it
  reads only 44 %. Cool it at a 15 °C outside wall and it reaches about 87 %,
  enough for mould. The amount of water did not change at all.
- **The same reading means different things.** 23 °C feels pleasant at 45 %,
  but sticky at 70 %.

The h,x diagram shows both in one point. Its position gives temperature,
relative and absolute humidity at the same time, and how far each
measurement was from the comfort zone.

## Reading the diagram

Every state of the air is one point:

- **Up and down** is the temperature (°C).
- **Left and right** is the absolute humidity *x*, the grams of water in one
  kilogram of dry air. Left is dry, right is humid.
- **The curved lines** are the relative humidity φ. The bottom curve, 100 %,
  is saturation: air cannot hold more water. Cool air down to this curve and
  water condenses on windows and cold walls, or forms fog. That temperature is
  the **dew point** (see below).
- The air pressure, which depends on the altitude, shifts the curves a little.
  Pass `pressure=` (in Pa) or `altitude=` to `plot_mollier_hx`, and `p=` or
  `altitude=` to `state()`.

How processes move the point (each has a function in `pyedautils._mollier`):

- **Heating or cooling** without adding or removing water moves the point
  straight **up or down** ({func}`~pyedautils._mollier.heat`,
  {func}`~pyedautils._mollier.cool` above the dew point): *x* stays the same
  and only the relative humidity changes. Heating makes air relatively drier,
  cooling makes it relatively more humid.
- **Adding moisture** (people, showering, cooking, plants, a humidifier) moves
  the point to the **right** ({func}`~pyedautils._mollier.humidify_isothermal`,
  {func}`~pyedautils._mollier.humidify_adiabatic`). **Removing moisture**
  (ventilating with dry air, a dehumidifier) moves it to the **left**.
- **Mixing** two air streams, for example room air and outside air when you
  ventilate, puts the result on the straight line between the two points
  ({func}`~pyedautils._mollier.mix`).

```python
from pyedautils._mollier import state

s = state(t=21, phi=0.60, altitude=450)
print(f"x = {s.x * 1000:.1f} g/kg, dew point = {s.t_dp:.1f} °C")
# x = 9.8 g/kg, dew point = 12.9 °C
```

## Dew point: where condensation and mould start

To find the dew point of the room air, go **straight down** from the point to
the 100 % curve: the temperature there is the dew point. Any surface colder
than that gets wet: the inside of single glazing, window frames, thermal
bridges, outside corners, walls behind wardrobes.

Mould does not even need visible water. It starts to grow when the air right
at a surface stays above about **80 % relative humidity** for some time.
That happens a few degrees **above** the dew point:

| Room air | Water content | Dew point (condensation) | Mould risk at surfaces below |
| --- | --- | --- | --- |
| 20 °C, 40 % | 6.1 g/kg | 6.0 °C | 9.3 °C |
| 21 °C, 50 % | 8.2 g/kg | 10.2 °C | 13.6 °C |
| 21 °C, 60 % | 9.8 g/kg | 12.9 °C | 16.4 °C |

The last column is the surface temperature at which the same air reaches 80 %:

```python
from pyedautils._mollier import state, temperature_p_sat

s = state(t=21, phi=0.50, altitude=450)
t_mould = temperature_p_sat(s.p_v / 0.80)   # surface temperature at phi = 80 %
print(f"{s.x * 1000:.1f} g/kg, dew point {s.t_dp:.1f} °C, mould below {t_mould:.1f} °C")
# 8.2 g/kg, dew point 10.2 °C, mould below 13.6 °C
```

On a cold winter day, the inside of an outside wall corner or an old window
frame can easily be 12–15 °C. At 21 °C and 60 % that is already critical; at
21 °C and 50 % it is borderline; at 40 % it is safe. This is why it pays to
give the comfort zone an absolute-humidity limit (`"abs_humidity"` in
`comfort_zone`, for example `(0, 0.012)` for 12 g/kg) and to watch the points
near the right-hand edge of the zone.

## Example: ventilating in winter makes the room drier

Cold outside air holds little water. At 0 °C and 80 % it carries about
3.2 g/kg. Heated to 21 °C in the room, the same air has only about **20 %**
relative humidity, because warm air could hold far more. Ventilating
therefore moves the room point to the left: the room gets drier. That is why
rooms are often too dry in winter, and why airing works so well against too
much humidity. Short, wide-open airing (a few minutes, with cross-draught)
exchanges the humid air without cooling down the walls.

```python
from pyedautils._mollier import state, heat
from pyedautils.plots import plot_mollier_hx

outside = state(t=0, phi=0.80, altitude=450)
heated, _ = heat(outside, t_out=21)
print(f"{outside.x * 1000:.1f} g/kg -> {heated.phi:.0%} at 21 °C")
# 3.2 g/kg -> 20% at 21 °C

html = plot_mollier_hx(altitude=450, pressure=outside.p,
                       states=[outside, heated], labels=["outside", "heated"])
```

## Example: heating less raises the humidity and saves energy

Turn the heating down and the point moves straight down: the water content
stays the same and the relative humidity rises. Room air with 6.4 g/kg reads
about **35 %** at 23 °C, **39 %** at 21 °C and **42 %** at 20 °C. Relative
humidity between about 40 and 60 % is pleasant for most people and easier on
eyes and airways; viruses and dust mites also do worse in that range than in
very dry or very humid air.

It also helps the environment. As a **rule of thumb**, every degree less room
temperature saves around **6 % of heating energy** (the exact figure depends
on the building and the climate; it is roughly one over the mean
indoor-outdoor temperature difference of the heating season). A slightly
cooler room with comfortable humidity often feels better than a warm, dry
one, with a smaller energy bill and lower emissions.

The limit is on the other side: if the relative humidity at cold surfaces
(outside walls, window frames, corners behind furniture) climbs above about
80 %, mould can grow. A comfort zone with warning bands
(`comfort_zone_orange`, `comfort_zone_red` in `plot_mollier_hx`) helps to
stay in between.

## Example: airing a cool cellar in summer can make it damp

Warm summer air carries a lot of water: at 25 °C and 70 % about 14.7 g/kg.
Let it into a cellar at 18 °C and it cools below its dew point (about 19 °C),
so water condenses on the walls (about 1 g per kg of air, since saturated air
at 18 °C holds only 13.7 g/kg). In summer, air a cool cellar in the early
morning or at night, when the outside air holds less water. The diagram shows
it at a glance: is the outside point to the **left** of the room point? Then
airing dries the room, even when the outside air reads 90 % relative humidity.

```python
from pyedautils._mollier import state

outside = state(t=25, phi=0.70, altitude=450)
cellar = state(t=18, phi=0.65, altitude=450)
print(f"outside {outside.x * 1000:.1f} g/kg (dew point {outside.t_dp:.1f} °C), "
      f"cellar {cellar.x * 1000:.1f} g/kg")
print("airing dries" if outside.x < cellar.x else "airing adds moisture")
# outside 14.7 g/kg (dew point 19.1 °C), cellar 8.8 g/kg
# airing adds moisture
```

See also {doc}`comfort_compass`, which counts how many days were off in which
direction, and the {doc}`../examples/mollier_hx` examples for more
diagram options.
