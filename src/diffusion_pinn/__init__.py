# src/diffusion_pinn/__init__.py

from .config import DiffusionConfig
from .models.pinn import DiffusionPINN
from .data.processor import DiffusionDataProcessor
from .utils.visualization import plot_loss_history, plot_diffusion_convergence
from .variables import PINN_VARIABLES

# Conditional imports for optional dependencies
try:
    from .optimization.bayesian_opt import PINNBayesianOptimizer
    from .optimization.config import OPTIMIZATION_SETTINGS
    _has_optimization = True
except ImportError:
    # Create placeholder for missing components
    PINNBayesianOptimizer = None
    OPTIMIZATION_SETTINGS = {}
    _has_optimization = False

__version__ = '0.2.27.0'

# Base components that should always be available
__all__ = [
    'DiffusionConfig',
    'DiffusionPINN',
    'DiffusionDataProcessor',
    'plot_loss_history',
    'plot_diffusion_convergence',
    'PINN_VARIABLES',
]

# Add optimization components if available
if _has_optimization:
    __all__.extend([
        'PINNBayesianOptimizer',
        'OPTIMIZATION_SETTINGS'
    ])