# src/diffusion_pinn/__init__.py

from .config import DiffusionConfig
from .models.pinn import DiffusionPINN
from .data.processor import DiffusionDataProcessor
from .utils.visualization import plot_loss_history, plot_diffusion_convergence
from .variables import PINN_VARIABLES
from .optimization.bayesian_opt import PINNBayesianOptimizer
from .optimization.config import OPTIMIZATION_SETTINGS

__version__ = '0.2.7'

__all__ = [
    'DiffusionConfig',
    'DiffusionPINN',
    'DiffusionDataProcessor',
    'plot_loss_history',
    'plot_diffusion_convergence',
    'PINN_VARIABLES',
    'PINNBayesianOptimizer',
    'OPTIMIZATION_SETTINGS'
]