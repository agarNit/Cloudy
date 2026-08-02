from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cloudy")
except PackageNotFoundError:
    # Running from source without an installed/editable distribution record.
    __version__ = "0.0.0-dev"
