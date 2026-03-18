# Installation

## From PyPI

```bash
pip install pyedautils
```

## Development install

Clone the repository and install in editable mode:

```bash
git clone https://github.com/retomarek/pyedautils.git
cd pyedautils
pip install -r requirements.txt
pip install -e .
```

## Requirements

- Python >= 3.10
- Key dependencies: `ephem`, `pandas`, `geopy`, `requests`, `pgeocode`, `plotly`

## Running tests

```bash
pytest
```

## Building the documentation locally

```bash
pip install "jupyter-book<2" sphinx-autodoc2
jupyter-book build docs/
```

The built HTML will be in `docs/_build/html/`.
