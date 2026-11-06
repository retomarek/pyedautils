# Reading the comfort compass

{func}`~pyedautils.plots.comfort.plot_comfort_compass` sums up how a room's days
split up around a comfort zone: how many days were in range, and how many
were too warm, too humid, too dry or too cold, or a mix of two. The counts
come from {func}`~pyedautils.comfort.comfort_compass_distribution`, which
takes the daily means of temperature and relative humidity and compares
every day with the comfort zone. The compass shows *how often*; the
{doc}`h,x diagram <mollier_hx>` ({func}`~pyedautils.plots.comfort.plot_mollier_hx`)
shows *where* and *why*.

```python
from importlib import resources

import pandas as pd

from pyedautils._mollier import pressure_from_altitude
from pyedautils.comfort import comfort_compass_distribution
from pyedautils.plots import plot_comfort_compass

df = pd.read_csv(resources.files("pyedautils") / "data" / "flat_temp_hum.csv",
                 sep=";", parse_dates=["timestamp"], index_col="timestamp")

dist = comfort_compass_distribution(
    df,
    temp_band=(20, 26),           # °C
    hum_band=(30, 65),            # % relative humidity
    hum_abs_band=(0, 0.012),      # kg/kg, optional absolute-humidity limit
    pressure=pressure_from_altitude(450),
)
fig = plot_comfort_compass(dist, title="Living room")
fig.show()
```

## Why temperature and humidity belong together

Comfort, health and the risk of mould always depend on temperature **and**
humidity at the same time. Looking at only one of them, or at two separate
graphs, can easily mislead you:

- **Relative humidity changes with temperature.** Air at 21 °C and 60 % holds
  about 9.8 g of water per kg of dry air (at 450 m a.s.l.). Warm the same air
  to 26 °C and it reads only 44 %. Cool it at a 15 °C outside wall and it
  reaches about 87 %, enough for mould. The amount of water did not change.
- **The same reading means different things.** 23 °C feels pleasant at 45 %,
  but sticky at 70 %.

That is why the compass judges every day on both together. A day only counts
as *in range* when temperature and humidity are both in the comfort zone, and
the direction of an arm tells you which combination was off. *Warm + humid*
calls for cooling or ventilation, *cold + humid* points to condensation and
mould risk, *warm + dry* to overheated, dry winter air.

With `hum_abs_band`, a day counts as humid when it is above the relative
**or** the absolute limit, so a warm day above 12 g/kg counts as *warm +
humid* even when its relative humidity is still in range. Without
`hum_abs_band` there is no absolute limit and humidity is judged by relative
humidity only.

## Reading the compass

The dashed circle has the area of **100 % of the days**. The green disc and
the arms always add up to exactly that area:

- All days in range: the green disc fills the dashed circle exactly, and
  there are no arms.
- Some days off: the disc shrinks by the share of those days, and the same
  area reappears in the arms. A long arm is therefore not "more than 100 %";
  it is area moved out of the disc.
- Each arm points in the direction of the deviation, as in the h,x diagram:
  warm up, cold down, humid right, dry left, mixes diagonally. From the
  inside out its segments are mild, moderate and severe, each with an area
  equal to its share of days.

The stages come from ± offsets around the comfort zone. Mild is up to the
first limit, moderate up to the second, severe beyond; the worse of
temperature and humidity decides. The defaults of
`comfort_compass_distribution` are 1 / 2.5 K (`temp_stage_limits`),
5 / 10 %-points (`hum_stage_limits`) and 1 / 2.5 g/kg
(`hum_abs_stage_limits`, used only with `hum_abs_band`). Use the same
offsets for the orange and red warning bands of `plot_mollier_hx`, so the
diagram and the compass tell the same story.

Hover over a segment to see its number of days and its share.

## Comparing rooms

By default (`fixed_scale=True`) the radial axis is fixed to the largest
possible arm. The dashed 100 % circle then has the same size in every chart,
and **equal areas mean equal numbers of days**, so compasses of different
rooms or periods can be compared directly. Draw them at the same `height` so
they come out at the same size. With mostly good days the glyph is small in
this mode.

With `fixed_scale=False` every compass is zoomed to its own glyph, so it fills
the figure. The dashed circle then has a different size in each chart, and
**two compasses cannot be compared by area**, only by their percentages.

```python
rooms = {"Living room": df_living, "Bedroom": df_bedroom}
for name, room in rooms.items():
    dist = comfort_compass_distribution(room, temp_band=(20, 26),
                                        hum_band=(30, 65))
    plot_comfort_compass(dist, title=name, height=460).show()  # fixed_scale=True
```

## Good for people and the environment

The h,x diagram explains why the compass looks the way it does, and what you
can do about it (all numbers at 450 m a.s.l., 96 035 Pa; see
{doc}`mollier_hx` for how they were computed):

- **Too dry in winter** (left arms) usually comes from ventilating with cold
  outside air: at 0 °C and 80 % it carries only about 3.2 g/kg of water,
  which is just **20 %** relative humidity once heated to 21 °C.
- **Heating less** moves days towards *in range*: the water content stays,
  the relative humidity rises. Air with 6.4 g/kg reads about 35 % at 23 °C,
  but 39 % at 21 °C. As a rule of thumb, every degree less saves around
  **6 % of heating energy**, good for comfort, the bill and the climate.
- **Too humid or cold + humid** (right and lower right arms) means a risk of
  condensation and mould at cold surfaces: at 21 °C and 60 % the dew point is
  12.9 °C, and mould can grow on surfaces below about 16 °C, such as outside
  wall corners in winter. Short, wide-open airing helps, as long as the
  outside air is drier (holds fewer g/kg) than the room air.

{doc}`mollier_hx` explains the h,x diagram step by step, with these examples;
more background on
[Wikipedia](https://en.wikipedia.org/wiki/Psychrometrics#Mollier_diagram).
