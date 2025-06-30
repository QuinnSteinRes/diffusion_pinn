# src/diffusion_pinn/models/__init__.py

from .pinn import OpenSystemDiffusionPINN

# Keep old name for backward compatibility during transition
DiffusionPINN = OpenSystemDiffusionPINN  # Alias for compatibility

__all__ = ['OpenSystemDiffusionPINN', 'DiffusionPINN']
