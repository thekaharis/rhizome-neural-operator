"""Energy-parameterized EDM and recurrent local–Fourier field models."""

from .energy import EnergyEDM, build_model
from .recurrent import RecurrentFNO2d
from .rhizome import RhizomeOperator2d

__all__ = ["EnergyEDM", "build_model", "RecurrentFNO2d", "RhizomeOperator2d"]
