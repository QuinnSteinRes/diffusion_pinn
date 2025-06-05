# variables_diagnostic.py
# Test intermediate loss weighting to balance convergence vs field prediction

PINN_VARIABLES = {
    # v0.2.21 proven architecture
    'hidden_layers': [16, 32, 64, 32, 16],
    'activation': 'tanh',

    # Quick test parameters
    'epochs': 12000,  # Enough to see trends
    'learning_rate': 0.0005,  # v0.2.21 stable rate
    'decay_steps': 500,
    'decay_rate': 0.95,
    'random_seed': 42,

    # v0.2.21 proven initial value
    'initial_D': 0.0005,

    # v0.2.21 proven sampling
    'N_u': 1000,
    'N_f': 15000,
    'N_i': 8000,

    # INTERMEDIATE loss weights - testing hypothesis
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 6.0,    # Between v0.2.21 (3.0) and v0.2.26 (10.0)
        'physics': 6.5      # Between v0.2.21 (5.0) and v0.2.26 (8.0), slightly favoring physics
    }
}