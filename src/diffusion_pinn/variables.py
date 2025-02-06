"""
Configuration variables for PINN training
Save as variables.py in your project
"""

PINN_VARIABLES = {
    # Network Architecture
    'hidden_layers': [40, 40, 40],
    'activation': 'tanh',

    # Training Parameters
    'epochs': 50000,
    'learning_rate': 0.001,
    'decay_steps': 1000,
    'decay_rate': 0.95,
    'initial_D': 0.5,

    # Sampling Points
    'N_u': 1000,    # boundary points
    'N_f': 20000,   # collocation points
    'N_i': 10000,   # interior points

    # Loss Weights
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 10.0,
        'physics': 5.0
    }
}
