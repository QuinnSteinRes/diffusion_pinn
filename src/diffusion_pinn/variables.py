PINN_VARIABLES = {
    # Network Architecture
    'hidden_layers': [16, 32, 64, 32, 16],  # Balanced architecture
    'activation': 'tanh',                   # Best for PDEs

    # Training
    'epochs': 2500,                        # Enough for convergence
    'learning_rate': 0.0005,                # Stable learning rate
    'random_seed': 42,                      # For reproducibility

    # Diffusion coefficient
    'initial_D': 0.0005,                    # Reasonable starting point

    # Sampling points
    'N_u': 1000,    # Boundary points
    'N_f': 15000,   # Physics points (PDE)
    'N_i': 8000,    # Interior data points

    # Loss weights - balanced for stability
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 3.0,
        'physics': 5.0
    }
}