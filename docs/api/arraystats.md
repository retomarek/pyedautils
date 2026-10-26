# arraystats

Fast, NumPy-only statistics for large time series (millions of samples):
descriptive statistics with coverage ({func}`~pyedautils.arraystats.describe`),
sampling interval, gap and stuck-value detection, and outlier detection
(IQR, z-score, Hampel filter). No pandas required; designed for interactive
tools where every millisecond counts. The pandas-based functions in
{mod}`pyedautils.data_quality` use these as their core.

```python
import numpy as np
from pyedautils.arraystats import describe, outliers_hampel

ts = np.arange(525_600, dtype=np.int64) * 60_000      # one year, 1-min, epoch ms
values = 20 + 5 * np.sin(ts / 8.64e7 * 2 * np.pi)
stats = describe(values, ts)                          # n, coverage, min, max, mean, median, ...
mask = outliers_hampel(values, window=30)             # boolean outlier mask
```

## API Reference

```{eval-rst}
.. automodule:: pyedautils.arraystats
   :members:
   :undoc-members:
   :show-inheritance:
```
