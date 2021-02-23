try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Can remove when we require Python > 3.7
    from importlib_metadata import version, PackageNotFoundError

__version__ = version("pvlib")

from pvlib import (  # noqa: F401
    atmosphere,
    bifacial,
    clearsky,
    # forecast
    iam,
    inverter,
    iotools,
    irradiance,
    ivtools,
    location,
    modelchain,
    pvsystem,
    scaling,
    shading,
    singlediode,
    snow,
    soiling,
    solarposition,
    spa,
    spectrum,
    temperature,
    tools,
    tracking,
)
