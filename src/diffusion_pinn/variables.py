PINN_VARIABLES = {
    # Network Architecture - More stable architecture
    'hidden_layers': [20, 40, 40, 20],  # Wider in middle, narrow at ends
    'activation': 'tanh',  # Keep tanh which works well for PDEs

    # Training Parameters - More conservative learning
    'epochs': 12000,      # More epochs for proper convergence
    'learning_rate': 0.0008,  # Slightly lower initial learning rate
    'decay_steps': 1200,  # Slower decay
    'decay_rate': 0.95,   # More gentle decay
    'initial_D': 0.0001,  # Start from a reasonable middle value

    # Sampling Points - Keep as is
    'N_u': 1000,    # boundary points
    'N_f': 15000,   # collocation points
    'N_i': 8000,    # interior points

    # Loss Weights - Emphasize physics more
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 8.0,
        'physics': 12.0  # Higher physics weight
    }
}




#PINN_VARIABLES = {
#    # Network Architecture - Deeper but with reducing width
#    'hidden_layers': [40, 40, 40, 40],
#    'activation': 'tanh',  # tanh works well for PDEs#
#
#    # Training Parameters - More careful learning rate decay
#    'epochs': 10000,
#    'learning_rate': 0.001,
#    'decay_steps': 500,
#    'decay_rate': 0.95,
#    'initial_D': 0.00009,  # Better initial guess #Range#

    # Sampling Points - Balanced boundary and interior#
#    'N_u': 1000,    # boundary points
#    'N_f': 15000,   # collocation points (reduced to avoid memory issues)
#    'N_i': 8000,    # interior points

    # Loss Weights - Enhanced weighting for physics
 #   'loss_weights': {
 #       'initial': 1.0,
 #       'boundary': 1.0,
 #       'interior': 10.0,
 #       'physics': 8.0  # Increased physics weight to enforce PDE
 #   }
#}
