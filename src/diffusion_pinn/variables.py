PINN_VARIABLES = {
    # Network Architecture - v0.2.21 proven hourglass design
    'hidden_layers': [16, 32, 64, 32, 16],  # Balanced hourglass architecture that worked
    'activation': 'tanh',                   # Best activation for PDEs

    # Training Parameters - v0.2.21 proven stable settings
    'epochs': 1000,                        # Sufficient epochs for convergence
    'learning_rate': 0.0005,                # Stable learning rate (not 0.001)
    'decay_steps': 500,                     # Learning rate decay schedule
    'decay_rate': 0.95,
    'random_seed': 42,                      # Fixed seed for reproducibility

    # Diffusion coefficient - v0.2.21 proven initial value
    'initial_D': 0.0005,                    # Proven starting point (not 0.00009)

    # Sampling Points - v0.2.21 proven distribution
    'N_u': 1000,    # Boundary points
    'N_f': 15000,   # Physics collocation points (PDE enforcement)
    'N_i': 8000,    # Interior data supervision points

    # Loss Weights - v0.2.21 proven balanced weighting
    'loss_weights': {
        'initial': 1.0,     # Initial condition weight
        'boundary': 1.0,    # Boundary condition weight
        'interior': 3.0,    # Interior data fitting weight (balanced, not 10.0)
        'physics': 5.0      # Physics loss weight (balanced, not 8.0)
    }
}