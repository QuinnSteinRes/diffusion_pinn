PINN_VARIABLES = {
    'hidden_layers': [16, 32, 64, 32, 16],
    'activation': 'tanh',
    'epochs': 12000,  # Quick test of Stage 1 approach
    'learning_rate': 0.0005,
    'decay_steps': 500,
    'decay_rate': 0.95,
    'random_seed': 42,
    'initial_D': 0.0005,
    'N_u': 1000,
    'N_f': 15000,
    'N_i': 8000,

    # EXTREME physics focus - even more than v0.2.21
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 2.0,    # VERY LOW - minimize data interference with D convergence
        'physics': 6.0      # HIGH - maximize physics constraint for D
    }
}