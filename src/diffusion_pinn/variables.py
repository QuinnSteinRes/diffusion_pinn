PINN_VARIABLES = {
    # Network Architecture - Deeper but with reducing width
    'hidden_layers': [40, 40, 40, 40],
    'activation': 'tanh',  # tanh works well for PDEs

    # Training Parameters - More careful learning rate decay
    'epochs': 10000,
    'learning_rate': 0.001,
    'decay_steps': 500,
    'decay_rate': 0.95,
    'initial_D': 0.005,  # Better initial guess #Range

    # Sampling Points - Balanced boundary and interior
    'N_u': 1000,    # boundary points
    'N_f': 15000,   # collocation points (reduced to avoid memory issues)
    'N_i': 8000,    # interior points

    # Loss Weights - Enhanced weighting for physics
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 10.0,
        'physics': 8.0  # Increased physics weight to enforce PDE
    }
}
