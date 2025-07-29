
"""
Diarization Simulation Package

A Python package for simulating speaker diarization with LENA and VTC
from ground truth vocalization data.
"""

from .diarization import DiarizationSimulator, simulate_diarization

__version__ = "0.1.0"
__author__ = "LAAC-LSCP"

__all__ = [
    "DiarizationSimulator",
    "simulate_diarization",
]