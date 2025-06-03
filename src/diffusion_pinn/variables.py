PINN_VARIABLES = {
    # Network Architecture
    'hidden_layers': [16, 32, 64, 32, 16],  # Balanced architecture
    'activation': 'tanh',                   # Best for PDEs

    # Training
    'epochs': 12000,                        # Enough for convergence
    'learning_rate': 0.0005,                # Stable learning rate
    'random_seed': 42,                      # For reproducibility

    # Diffusion coefficient
    'initial_D': 0.0005,                    # Reasonable starting point

    # Sampling points
    'N_u': 1500,    # Boundary points
    'N_f': 20000,   # Physics points (PDE)
    'N_i': 15000,   # Interior data points

    # FIXED Loss weights - prioritize data learning first
    'loss_weights': {
        'initial': 5.0,     # Increased from 1.0
        'boundary': 5.0,    # Increased from 1.0
        'interior': 10.0,   # Increased from 3.0
        'physics': 1.0      # Decreased from 5.0
    }
}