PINN_VARIABLES = {
    # Network Architecture - Further refined for diffusion problems
    'hidden_layers': [24, 48, 48, 24],  # Slightly larger but same shape
    'activation': 'tanh',  # Keep tanh which works well for PDEs

    # Training Parameters - Even more stable learning
    'epochs': 15000,      # Increase epochs further
    'learning_rate': 0.0006,  # Further reduce learning rate
    'decay_steps': 1500,  # Even slower decay
    'decay_rate': 0.97,   # More gentle decay (closer to 1.0)
    'initial_D': 0.0001,  # Keep same initial value

    # Sampling Points - Increase physics points
    'N_u': 1200,    # More boundary points
    'N_f': 20000,   # More collocation points for better physics enforcement
    'N_i': 10000,   # More interior points

    # Loss Weights - Further emphasize physics
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 5.0,   # Reduced weight on interior points
        'physics': 15.0    # Increased physics weight further
    }
}