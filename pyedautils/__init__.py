from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyedautils")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
