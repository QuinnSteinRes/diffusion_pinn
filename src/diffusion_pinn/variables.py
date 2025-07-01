# src/diffusion_pinn/variables.py
# Updated for open system with boundary flux

PINN_VARIABLES = {
    # Network Architecture - MUCH SMALLER
    'hidden_layers': [20, 40, 20],      # Was [32, 64, 64, 32]
    'activation': 'tanh',

    # Training - SHORTER, FASTER
    'epochs': 5000,                     # Was 15000
    'learning_rate': 0.001,             # Was 0.0003
    'random_seed': 42,

    # Sampling - MORE DATA FITTING
    'N_u': 5000,                        # Was 2000
    'N_f': 8000,                        # Was 20000
    'N_i': 8000,                        # Was 5000

    # Loss weights - DATA FITTING ONLY
    'loss_weights': {
        'initial': 1000.0,              # Was 10.0
        'boundary': 0.0,                # Was 5.0 - DISABLE
        'interior': 100.0,              # Was 3.0
        'physics': 0.0                  # Was 5.0 - DISABLE
    },

    # Keep these the same
    'initial_D': 0.0001,
    'initial_k': 0.001,

    # Open system specific parameters
    'c_external': 0.0,                       # External concentration (usually 0)
    'boundary_flux_weight': 1.0,             # Weight for Robin boundary terms

    # Bounds for logarithmic parameterization
    'log_D_bounds': (-16.0, -6.0),           # log(D) bounds
    'log_k_bounds': (-20.0, -2.0),           # log(k) bounds

    # Convergence criteria for two parameters
    'convergence_threshold': 0.01,           # Relative std dev threshold
    'convergence_window': 100,               # Window for convergence check

    # Model type identifier
    'model_type': 'OpenSystemDiffusionPINN'
}