# Try to import the optimization components
# This is wrapped in try/except to handle missing optional dependencies

try:
    from .bayesian_opt import PINNBayesianOptimizer
    from .config import OPTIMIZATION_SETTINGS

    __all__ = [
        'PINNBayesianOptimizer',
        'OPTIMIZATION_SETTINGS'
    ]

except ImportError as e:
    # Create a warning message but don't crash
    import warnings
    warnings.warn(f"Could not import optimization components: {str(e)}\n"
                  f"Bayesian optimization will not be available.")

    # Empty __all__ since we couldn't import anything
    __all__ = []