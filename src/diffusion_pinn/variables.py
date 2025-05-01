PINN_VARIABLES = {
    # Network Architecture - Better balanced architecture
    'hidden_layers': [20, 40, 60, 40, 20],  # Deeper, gradually expanding then contracting
    'activation': 'tanh',  # tanh is generally better for PDEs

    # Training Parameters
    'epochs': 15000,      # More epochs with multi-stage training
    'learning_rate': 0.0005,  # Slightly lower initial learning rate for stability
    'decay_steps': 1000,  # Will be used with custom scheduler
    'decay_rate': 0.95,
    'initial_D': 0.0001,  # Start from expected value range

    # Sampling Points - Increased for better coverage
    'N_u': 1500,    # More boundary points
    'N_f': 20000,   # More collocation points
    'N_i': 10000,   # More interior points

    # Loss Weights - Better balanced for physics and data
    'loss_weights': {
        'initial': 1.0,
        'boundary': 1.0,
        'interior': 5.0,  # Higher weight on interior data
        'physics': 8.0    # Balanced physics weight
    }
}