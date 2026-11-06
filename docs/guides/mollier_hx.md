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

(hx-reading-the-diagram)=
## Reading the diagram

How to read the diagram, and the most common misreadings.

### Axes and curves

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

### How processes move the point

Each process has a function in `pyedautils._mollier`:

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

(hx-misreadings)=
### Common misreadings

- **Relative humidity is not water content.** A high relative humidity does
  not mean a lot of water. Outside air at 0 °C and 80 % holds 3.2 g/kg, less
  than room air at 21 °C and 30 % (4.9 g/kg). To compare two airs, compare
  their *x*, not their φ: the point further **left** is the drier one.
- **Heating and cooling move the point vertically, not sideways.** In the
  h,x diagram the temperature is on the vertical axis, so heating or cooling
  without condensation keeps *x* and moves the point straight up or down,
  across the φ curves. (In the psychrometric chart, which is rotated and
  mirrored, the same process is a horizontal line.) Only when cooling reaches
  the 100 % curve does the point follow that curve to the left, and water
  condenses.
- **The curves depend on the air pressure.** Use the pressure of your site
  (`pressure=` or `altitude=`). The same air at 21 °C and 50 % holds
  7.7 g/kg at sea level (101 325 Pa), 8.2 g/kg at 450 m (96 035 Pa) and
  8.7 g/kg at 1000 m (89 875 Pa). Its dew point stays the same (10.2 °C),
  but a fixed limit in g/kg means a different relative humidity at a
  different altitude.
- **Mould starts at cold surfaces, not in the room air.** A room at 21 °C and
  50 % looks fine on the diagram, but the air right at a 13 °C window frame is
  above 80 %. Read the risk off at the **surface temperature**, see
  {ref}`hx-dew-point` below.
- **Single points are not the whole story.** Short peaks (showering, cooking)
  show up as single points to the right; what matters for mould is how long
  the surfaces stay humid, so look at where most of the points are.

(hx-dew-point)=
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

(hx-comfort)=
## Comfort

The **comfort zone** is the area of the diagram where most people feel
comfortable and where buildings stay healthy. pyedautils draws it from a
temperature range, a relative-humidity range and, optionally, an upper limit
of the absolute humidity (`comfort_zone={"temperature": (20, 26),
"rel_humidity": (0.30, 0.65), "abs_humidity": (0, 0.012)}` in
`plot_mollier_hx`). Typical limits for living rooms and offices are
**20–26 °C and 30–65 %** relative humidity, with an absolute limit of about
**11.5–12 g/kg** (`plot_mollier_hx` uses 11.5 g/kg by default). The exact
values depend on the standard, the use of the room and the season.

Why both limits matter, and why the zone is not a rectangle:

- **Temperature alone is not enough.** 23 °C feels pleasant at 45 % and sticky
  at 70 %.
- **Relative humidity alone is not enough.** At 20 °C, 65 % is 10.0 g/kg; at
  26 °C the same 65 % is 14.5 g/kg, warm and muggy. The absolute limit of
  12 g/kg cuts off that corner: at 24 °C it is reached at 61 %, at 26 °C
  already at 54 %.
- **Too dry is uncomfortable too.** Below about 30 % eyes, skin and airways
  dry out; 30 % is 4.6 g/kg at 20 °C and 6.6 g/kg at 26 °C.

**Warning bands** show how far a point is outside the zone. Draw them as
dashed outlines with `comfort_zone_orange` and `comfort_zone_red`, for example
the zone widened by ±1 K and ±5 %-points (orange) and by ±2.5 K and
±10 %-points (red). These are the same steps that
{func}`~pyedautils.comfort.comfort_compass_distribution` uses for the stages
mild, moderate and severe, so the diagram and the
{doc}`comfort compass <comfort_compass>` tell the same story.

The comfort zone is about the room air. For mould, also look at the coldest
surfaces ({ref}`hx-dew-point`).

(hx-saving-energy)=
## Saving energy

The diagram also shows where energy goes, and where you can save it without
losing comfort. The examples below have the numbers.

- **Heat a little less.** As a rule of thumb, every degree less saves around
  6 % of heating energy, and the relative humidity rises at the same time
  ({ref}`hx-heating-less`).
- **Ventilate briefly and intensively in winter.** A few minutes with the
  windows wide open exchange the humid air without cooling down the walls and
  furniture; windows tilted for hours waste heat and cool the surfaces next to
  them ({ref}`hx-winter-ventilation`).
- **Do not over-humidify.** Every gram of water a humidifier evaporates takes
  energy, and too much water raises the mould risk at cold surfaces
  ({ref}`hx-humidifying`).
- **Air a cellar only when the outside air holds less water** (outside point
  to the left of the room point); otherwise you bring water in and pay for
  removing it again ({ref}`hx-cellar`).
- **Cool with night air in summer** instead of an air conditioner, when the
  night air is cool and holds no more water (in g/kg) than the room air
  ({ref}`hx-night-ventilation`).

(hx-winter-ventilation)=
### Example: ventilating in winter makes the room drier

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

(hx-heating-less)=
### Example: heating less raises the humidity and saves energy

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

(hx-humidifying)=
### Example: humidifying costs energy and raises the mould risk

Raising a room at 21 °C from 30 % to 50 % adds 3.3 g of water per kg of air
(4.9 to 8.2 g/kg). Evaporating water takes about 2500 kJ, or **0.7 kWh, per
kg of water**, whether it comes from a humidifier or a drying rack. At 50 %
the mould limit at surfaces is 13.6 °C, at 60 % already 16.4 °C, instead of
10.2 °C at 40 %. In winter, 30–40 % is usually enough. First check whether
the room is simply overheated, because heating less raises the relative
humidity for free.

```python
from pyedautils._mollier import state, humidify_isothermal

room = state(t=21, phi=0.30, altitude=450, m_dot_dry=1.0)
humid, balance = humidify_isothermal(room, phi_out=0.50)
print(f"{room.x * 1000:.1f} -> {humid.x * 1000:.1f} g/kg, "
      f"{balance.dh:.1f} kJ per kg of air")
# 4.9 -> 8.2 g/kg, 8.4 kJ per kg of air
```

(hx-cellar)=
### Example: airing a cool cellar in summer can make it damp

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

(hx-night-ventilation)=
### Example: cooling with night air in summer

After a hot day the room is at 26 °C and 50 % (11.1 g/kg). The night air at
16 °C and 85 % reads very humid, but it holds only 10.2 g/kg, less than the
room air, and its enthalpy is 42 instead of 55 kJ/kg. Opening the windows at
night therefore cools the room **and** dries it slightly: warmed to 26 °C, the
night air would read only 46 %. Close the windows and shade them in the
morning, before the outside air gets warmer than the room.

```python
from pyedautils._mollier import state

room = state(t=26, phi=0.50, altitude=450)
night = state(t=16, phi=0.85, altitude=450)
print(f"room {room.x * 1000:.1f} g/kg, {room.h:.0f} kJ/kg; "
      f"night {night.x * 1000:.1f} g/kg, {night.h:.0f} kJ/kg")
# room 11.1 g/kg, 55 kJ/kg; night 10.2 g/kg, 42 kJ/kg
```

See also {doc}`comfort_compass`, which counts how many days were off in which
direction, and the {doc}`../examples/mollier_hx` examples for more
diagram options.
