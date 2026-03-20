from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyedautils")
except PackageNotFoundError:
    __version__ = "unknown"
