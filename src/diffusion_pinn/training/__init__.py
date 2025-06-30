# src/diffusion_pinn/training/__init__.py

from .trainer import (
    create_open_system_pinn,
    train_open_system_pinn,
    save_open_system_checkpoint,
    # Backward compatibility (deprecated)
    train_pinn,
    create_and_initialize_pinn
)

__all__ = [
    'create_open_system_pinn',
    'train_open_system_pinn', 
    'save_open_system_checkpoint',
    # Deprecated but kept for compatibility
    'train_pinn',
    'create_and_initialize_pinn'
]
