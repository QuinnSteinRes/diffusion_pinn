# src/diffusion_pinn/variables.py
# Updated for open system with boundary flux

PINN_VARIABLES = {
    # Network Architecture
    'hidden_layers': [32, 64, 64, 32],      # Slightly larger for two-parameter problem
    'activation': 'tanh',                    # Still best for PDEs

    # Training - increased for more complex physics
    'epochs': 15000,                         # More epochs needed for boundary physics
    'learning_rate': 0.0003,                 # Slightly lower for stability
    'random_seed': 42,                       # For reproducibility

    # Physical Parameters - now learning TWO parameters
    'initial_D': 0.0001,                     # Diffusion coefficient starting point
    'initial_k': 0.001,                      # Boundary permeability starting point

    # Sampling points - rebalanced for open system
    'N_u': 2000,     # More initial condition points (critical for open system)
    'N_f': 20000,    # More physics points (need boundary + interior)
    'N_i': 5000,     # Fewer interior data points (let physics dominate)

    # Loss weights - rebalanced for open system physics
    'loss_weights': {
        'initial': 10.0,    # Initial condition is critical
        'boundary': 5.0,    # Robin boundary conditions
        'interior': 3.0,    # Interior physics
        'physics': 5.0      # Overall physics weight
    },

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