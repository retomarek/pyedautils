# plots

Visualizations for energy time-series data and psychrometric charts.

See the {doc}`../examples/index` section for interactive usage examples.

```{tip}
Reading the comfort plots: {doc}`../guides/mollier_hx` ({ref}`reading the diagram <hx-reading-the-diagram>`, {ref}`comfort <hx-comfort>`, {ref}`saving energy <hx-saving-energy>`) for `plot_mollier_hx` / `plot_mollier_hx_fast`, and {doc}`../guides/comfort_compass` ({ref}`reading the compass <compass-reading-the-diagram>`, {ref}`comfort <compass-comfort>`, {ref}`saving energy <compass-saving-energy>`) for `plot_comfort_compass`. The comfort-zone and warning-band ideas apply to the other comfort plots too.
```

## Daily Profiles

```{eval-rst}
.. automodule:: pyedautils.plots.profiles
   :members:
   :undoc-members:
   :show-inheritance:
```

## Heatmaps

```{eval-rst}
.. automodule:: pyedautils.plots.heatmaps
   :members:
   :undoc-members:
   :show-inheritance:
```

## Energy Signature

```{eval-rst}
.. automodule:: pyedautils.plots.energy
   :members:
   :undoc-members:
   :show-inheritance:
```

## Seasonal Patterns

```{eval-rst}
.. automodule:: pyedautils.plots.seasonal
   :members:
   :undoc-members:
   :show-inheritance:
```

## Thermal Comfort

**Comfort compass** ({func}`pyedautils.plots.comfort.plot_comfort_compass`):
the dashed circle is the area of 100 % of the days; the green disc (share in
range) and the eight arms (deviation directions, mild / moderate / severe from
the inside out) always add up to exactly that area.

- The green disc says what it means: share plus the in-range word
  (`names["ok"]`). When the disc is too small only the share is printed, or
  nothing; without the legend a caption below the glyph gives it instead
  (`● 8% in range`).
- Each direction label sits at the end of its own arm, anchored outward.
- `scale="fixed"` (default, same as `fixed_scale=True`): the dashed 100 %
  circle has the same size in every chart, so rooms can be compared by area.
  `scale="fit"`: zoomed to this glyph and its labels; compare such charts by
  their percentages only. `scale` takes precedence over `fixed_scale`.
- Every visible text can be passed: `title`, `names`, `direction_labels`,
  `stage_names`, `count_label` / `count_label_singular`, `stats_text`. German
  presets: `_COMPASS_NAMES_DE`, `_COMPASS_LABELS_DE`, `_COMPASS_STAGE_NAMES_DE`
  in `pyedautils.plots.comfort`.

```{eval-rst}
.. automodule:: pyedautils.plots.comfort
   :members:
   :undoc-members:
   :show-inheritance:
```

## Statistical / Exploratory

```{eval-rst}
.. automodule:: pyedautils.plots.stat
   :members:
   :undoc-members:
   :show-inheritance:
```

## Gradients

```{eval-rst}
.. automodule:: pyedautils.plots.gradients
   :members:
   :undoc-members:
   :show-inheritance:
```

## Degree Days

```{eval-rst}
.. automodule:: pyedautils.plots.degree_days
   :members:
   :undoc-members:
   :show-inheritance:
```

## Solar Influence

```{eval-rst}
.. automodule:: pyedautils.plots.solar
   :members:
   :undoc-members:
   :show-inheritance:
```
