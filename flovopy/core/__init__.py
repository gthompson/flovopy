# flovopy/core/__init__.py

"""
Core module of FLOVOpy: foundational tools for volcano-seismology and seismic monitoring.
Includes waveform processing, event detection, feature extraction, and related utilities.
"""

from .preprocessing import *
from .detection import *
from .mvo import *
from .legacy import *
from .features import *
from .magnitude import *
from .plotting import *