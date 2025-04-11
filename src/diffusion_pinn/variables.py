"""
Configuration variables for PINN training
"""

PINN_VARIABLES = {
    # Network Architecture
    'hidden_layers': [40, 40, 40, 40, 40], # 0.2.7 [40, 40, 40]
    'activation': 'tanh',

    # Training Parameters
    'epochs': 50001,
    'learning_rate': 0.0005, # 0.2.7 0.001
    'decay_steps': 2000, # 0.2.7 1000
    'decay_rate': 0.90,  # 0.2.7 0.95
    'initial_D': 0.0005, # 0.2.7 0.001

    # Sampling Points
    'N_u': 2000,    # boundary points       0.2.7 1000
    'N_f': 20000,   # collocation points
    'N_i': 10000,   # interior points

    # Loss Weights
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 10.0,
        'physics': 10.0    #0.2.7 5.0
    }
}
