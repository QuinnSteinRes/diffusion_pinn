# src/diffusion_pinn/__init__.py

from .config import DiffusionConfig
from .models.pinn import OpenSystemDiffusionPINN
from .data.processor import DiffusionDataProcessor
from .variables import PINN_VARIABLES

# Backward compatibility alias
DiffusionPINN = OpenSystemDiffusionPINN

__version__ = '0.3.1'

# Base components that should always be available
__all__ = [
    'DiffusionConfig',
    'OpenSystemDiffusionPINN',
    'DiffusionDataProcessor',
    'PINN_VARIABLES',
    'DiffusionPINN',  # Compatibility alias
]
