# src/diffusion_pinn/optimization/config.py

"""
Centralized configuration settings for PINN optimization
"""

OPTIMIZATION_SETTINGS = {
    ##### INPUT/OUTPUT SETTINGS #####
    "inputFile": "intensity_time_series_spatial_temporal.csv",
    "resultFolder": "trainingResults",

    ##### DATA PREPROCESSING SETTINGS #####
    "centering_method": "mean",
    "scaling_method": "auto",
    "training_ratio": 70,  # %

    ##### OPTIMIZER SETTINGS #####
    # Number of iterations for the Bayesian optimizer
    "iterations_optimizer": 20,
    # Acquisition function for Bayesian optimization
    "acquisitionFunction": "EI",
    # Number of random starts (as fraction of total iterations)
    "random_starts_fraction": 0.3,

    ##### NEURAL NETWORK TRAINING SETTINGS #####
    # Number of epochs for each trial
    "network_epochs": 10000,
    # Batch size
    "batchSize": 64,  # Increased from 32
    # Loss function
    "lossFunction": "mae",
    # Early stopping patience
    "earlyStop_patience": 8,
    # Learning rate scheduler factor
    "lr_reduction_factor": 0.5,
    # Learning rate scheduler patience
    "lr_patience": 3,

    ##### INITIAL PARAMETER VALUES #####
    "initial_neurons": 16,
    "initial_layers": 1,
    "initial_activation": "tanh",
    "initial_learningRate": 1e-4,

    ##### PARAMETER SEARCH RANGES #####
    # Layers search range
    "layers_lowerBound": 1,
    "layers_upperBound": 10,  # Reduced from 25
    # Neurons search range
    "neurons_lowerBound": 8,
    "neurons_upperBound": 256,  # Reduced from 512
    # Learning rate search range
    "learning_lowerBound": 1e-6,  # Increased from 1e-8
    "learning_upperBound": 1e-4,  # Reduced from 1e-3

    ##### STABILITY ENHANCEMENTS #####
    # L2 regularization strength
    "regularization_strength": 1e-5,
    # Gradient clipping norm
    "gradient_clip_norm": 1.0,
    # Gradient clipping value
    "gradient_clip_value": 0.5,
    # Max weight value to prevent extreme weights
    "max_weight_value": 10.0,
    # Output clipping to prevent extreme predictions
    "output_clip_min": -1e6,
    "output_clip_max": 1e6,

    ##### PINN SAMPLING POINTS #####
    "N_u": 1000,    # boundary points
    "N_f": 20000,   # collocation points
    "N_i": 10000,   # interior points

    ##### LOSS WEIGHTS #####
    "loss_weights": {
        "data": 1.0,
        "initial": 1.0,
        "boundary": 3.0,
        "physics": 5.0
    },

    ##### ADVANCED SETTINGS #####
    "dimension": 3,
    "code_debugging": True,
    "plot_results": True,
    "save_model": True,
    "checkpoint_interval": 5  # Save checkpoints every N iterations
}

def get_config():
    """Get a copy of the configuration settings"""
    import copy
    return copy.deepcopy(OPTIMIZATION_SETTINGS)

def update_config(updates):
    """Update configuration with custom values"""
    config = get_config()
    config.update(updates)
    return config
