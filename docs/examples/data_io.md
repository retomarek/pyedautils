# data_io

See {doc}`../api/data_io` for the full API reference.

## Save and load a DataFrame as CSV

```python
import pandas as pd
from pyedautils.data_io import save_data, load_data

df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                    "value": range(24)})

save_data(df, "output/measurements.csv", index=False)
df_loaded = load_data("output/measurements.csv")
```

## Save and load as compressed pickle

```python
save_data(df, "output/measurements.pklz")
df_loaded = load_data("output/measurements.pklz")
```

## Save a dictionary as JSON

```python
config = {"building": "Office A", "sensors": ["temp", "relhum"]}
save_data(config, "output/config.json")
config_loaded = load_data("output/config.json")
```
