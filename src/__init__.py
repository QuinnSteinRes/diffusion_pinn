from .config import DiffusionConfig
from .models.pinn import DiffusionPINN
from .data.processor import DiffusionDataProcessor
from .plot_results import plot_loss_history, plot_d_convergence
__version__ = '0.1.0'
