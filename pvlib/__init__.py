import logging
logging.basicConfig()
from pvlib.version import __version__

# make sure this is consistent with api.rst
from pvlib import atmosphere
from pvlib import clearsky
# from pvlib import forecast  # requires optional dependencies, not stable
from pvlib import irradiance
from pvlib import location
from pvlib import modelchain
from pvlib import pvsystem
from pvlib import solarposition
from pvlib import spa
from pvlib import tmy
from pvlib import tools
from pvlib import tracking

from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem, LocalizedPVSystem
from pvlib.tracking import SingleAxisTracker, LocalizedSingleAxisTracker

# consider adding some of the get_* functions e.g. get_solarposition
