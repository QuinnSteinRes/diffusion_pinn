# variables_lr_test.py
# Testing ultra-conservative learning rate with log(D) parameterization
# Hypothesis: log(D) gradients are more sensitive, need slower learning

PINN_VARIABLES = {
    # Network Architecture - v0.2.21 proven hourglass design
    'hidden_layers': [16, 32, 64, 32, 16],  # Keep proven architecture
    'activation': 'tanh',

    # Training Parameters - ULTRA-CONSERVATIVE learning rate
    'epochs': 1000,                         # Quick test
    'learning_rate': 0.0002,                # Even more conservative than v0.2.21's 0.0005
    'decay_steps': 500,
    'decay_rate': 0.95,
    'random_seed': 42,

    # Diffusion coefficient - v0.2.21 proven initial value
    'initial_D': 0.0005,                    # Keep proven starting point

    # Sampling Points - v0.2.21 proven distribution
    'N_u': 1000,    # Boundary points
    'N_f': 15000,   # Physics collocation points
    'N_i': 8000,    # Interior data supervision points

    # Loss Weights - v0.2.21 proven balanced weighting
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 3.0,    # v0.2.21 balance
        'physics': 5.0      # v0.2.21 balance
    }
}