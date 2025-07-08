PINN_VARIABLES = {
    # Network Architecture - Using v0.2.14 proven architecture
    'hidden_layers': [40, 40, 40, 40],  # Consistent width that worked well testing testing
    'activation': 'tanh',               # tanh works well for PDEs

    # Training Parameters - Using v0.2.14 proven settings
    'epochs': 10000,                    # Sufficient for convergence
    'learning_rate': 0.001,             # Stable learning rate
    'decay_steps': 500,                 # Learning rate decay
    'decay_rate': 0.95,
    'random_seed': 42,                  # For reproducibility

    # Diffusion coefficient - Using v0.2.14 proven initial value
    'initial_D': 0.00009,               # Better initial guess from v0.2.14

    # Sampling Points - Using v0.2.14 proven distribution
    'N_u': 1000,    # boundary points
    'N_f': 15000,   # collocation points (reduced to avoid memory issues)
    'N_i': 8000,    # interior points

    # Loss Weights - Using v0.2.14 proven weighting that ensures good concentration fields
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 1,   # Strong emphasis on data fitting
        'physics': 1      # Strong physics enforcement
    }
}