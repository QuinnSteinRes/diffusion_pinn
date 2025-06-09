# src/diffusion_pinn/training/__init__.py
# Updated to include hybrid training functions

from .trainer import (
    train_pinn,
    create_and_initialize_pinn,
    load_pretrained_pinn,
    save_checkpoint
)

# Try to import hybrid training functions
try:
    from .hybrid_trainer import (
        hybrid_train_pinn_v1,
        hybrid_train_pinn_v2,
        v026_with_enhanced_interior,
        run_phased_training,
        analyze_final_performance
    )
    _has_hybrid = True
    print("Hybrid training functions loaded successfully")
except ImportError as e:
    _has_hybrid = False
    print(f"Warning: Hybrid training not available - {str(e)}")

    # Create placeholder functions to prevent import errors
    def hybrid_train_pinn_v1(*args, **kwargs):
        raise ImportError("Hybrid training not available. Check hybrid_trainer.py")

    def hybrid_train_pinn_v2(*args, **kwargs):
        raise ImportError("Hybrid training not available. Check hybrid_trainer.py")

    def v026_with_enhanced_interior(*args, **kwargs):
        raise ImportError("Hybrid training not available. Check hybrid_trainer.py")

# Base exports (always available)
__all__ = [
    'train_pinn',
    'create_and_initialize_pinn',
    'load_pretrained_pinn',
    'save_checkpoint'
]

# Add hybrid functions if available
if _has_hybrid:
    __all__.extend([
        'hybrid_train_pinn_v1',
        'hybrid_train_pinn_v2',
        'v026_with_enhanced_interior',
        'run_phased_training',
        'analyze_final_performance'
    ])
else:
    # Still export the placeholder functions
    __all__.extend([
        'hybrid_train_pinn_v1',
        'hybrid_train_pinn_v2',
        'v026_with_enhanced_interior'
    ])