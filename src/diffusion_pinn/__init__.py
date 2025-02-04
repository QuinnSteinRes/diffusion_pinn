from .config import DiffusionConfig
from .models.pinn import DiffusionPINN
from .data.processor import DiffusionDataProcessor
from .utils.visualization import plot_loss_history, plot_diffusion_convergence  # Fixed function name

__version__ = '0.1.1'
