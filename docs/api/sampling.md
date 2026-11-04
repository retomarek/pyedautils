# sampling

How a series was sampled and stored, estimated from the data alone: interval
percentiles, change-of-value (COV) detection, dead band, heartbeat, quantisation
step, compression factor (Thornhill et al. 2004) and how long values stood
unchanged. Building automation archives often store only changes, which biases
spread, spectra and oscillation indices; the profile shows which analyses a
series supports. NumPy only.

```python
import numpy as np
from pyedautils.sampling import sampling_profile, interval_histogram

p = sampling_profile(ts, values, long_hold=86_400_000)   # ts: epoch ms
p.change_of_value, p.deadband, p.interval_min, p.compression
interval_histogram(ts, [1_000, 10_000, 60_000, 600_000, 3_600_000, 86_400_000])
```

## API Reference

```{eval-rst}
.. automodule:: pyedautils.sampling
   :members:
   :undoc-members:
   :show-inheritance:
```
