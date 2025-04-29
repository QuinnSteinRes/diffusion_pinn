PINN_VARIABLES = {
    # Network Architecture - Balanced for diffusion problems
    'hidden_layers': [20, 40, 40, 20],  # Keep the proven architecture
    'activation': 'tanh',  # Maintain tanh activation

    # Training Parameters - Optimized for convergence
    'epochs': 15000,      # Keep your increased epochs
    'learning_rate': 0.001,  # Slightly higher learning rate for faster initial convergence
    'decay_steps': 1200,  # Keep moderate decay steps
    'decay_rate': 0.96,   # Balanced decay rate between current and your proposal
    'initial_D': 0.00009,  # Start slightly below expected value to prevent overshooting

    # Sampling Points - Balanced distribution
    'N_u': 1000,    # Keep boundary points consistent
    'N_f': 18000,   # Moderate increase in collocation points
    'N_i': 9000,    # Moderate increase in interior points

    # Loss Weights - Balanced weights with phase-based approach
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 6.0,   # Moderate weight on interior points
        'physics': 10.0    # Strong but not overwhelming physics weight
    }
}