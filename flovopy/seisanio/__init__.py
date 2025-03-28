"""
seisanio: Tools for reading and processing enhanced Seisan S-files used at MVO.

Modules:
- core.sfile: Main Sfile parser
- core.wavfile: WAV file wrapper
- core.aeffile: AEF file parser
- utils.helpers: Helper functions (e.g., datetime conversion, file path formatting)
- utils.inventory: StationXML and STATION0.HYP handling
"""

from .core.sfile import Sfile
from .core.wavfile import Wavfile
from .core.aeffile import AEFfile
from .utils.helpers import (
    parse_string,
    spath2datetime,
    filetime2spath,
    filetime2wavpath,
    read_pickle
)
from .utils.inventory import (
    set_globals,
    parse_STATION0HYP,
    add_station_locations
)

# Optional: define the version
__version__ = "0.1.0"

# Optional: define what gets imported with `from seisan_parser import *`
__all__ = [
    "Sfile",
    "Wavfile",
    "AEFfile",
    "parse_string",
    "spath2datetime",
    "filetime2spath",
    "filetime2wavpath",
    "read_pickle",
    "set_globals",
    "parse_STATION0HYP",
    "add_station_locations"
]