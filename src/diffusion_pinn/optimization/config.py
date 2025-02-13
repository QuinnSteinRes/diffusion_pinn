# src/diffusion_pinn/optimization/config.py

"""
Configuration settings for PINN optimization, following the original structure
from trainingBayesian 1.py
"""

OPTIMIZATION_SETTINGS = {
    ##### CASE #####
    "inputFile"                 : "src/diffusion_pinn/data/intensity_time_series_spatial_temporal.csv",
    "resultFolder"             : "trainingResults",

    ##### DATA PREPROCESSING SETTINGS #####
    # Centering and scaling options
    "centering_method"         : "mean",
    "scaling_method"           : "auto",
    # Percentage of input data to be used for training
    "training_ratio"           : 70,  # %

    ##### ANN AND OPTIMIZER SETTINGS #####
    # Number of epochs for the ANN
    "network_epochs"           : 500,
    # Number of iterations for the optimizer
    "iterations_optimizer"     : 30,
    # Acquisition function to be utilized
    "acquisitionFunction"      : "EI",
    # Settings for the first iteration of the optimizer
    "initial_neurons"          : 16,
    "initial_layers"           : 1,
    "initial_activation"       : "tanh",
    "initial_learningRate"     : 1e-4,
    # Loss
    "lossFunction"             : "mae",
    # Batch size
    "batchSize"               : 32,

    ##### DESIGN SPACE SETTINGS #####
    # Lower and upper bound for number of layers
    "layers_lowerBound"        : 1,
    "layers_upperBound"        : 25,
    # Lower and upper bound for number of neurons
    "neurons_lowerBound"       : 5,
    "neurons_upperBound"       : 512,
    # Lower and upper bound for learning rate
    "learning_lowerBound"      : 1e-8,
    "learning_upperBound"      : 1e-3,

    ##### OTHER UTILITIES #####
    "plot_results"             : True,
    "save_model"              : True,
    # Early stop to avoid overfit
    "earlyStop_patience"       : 5,

    ##### PHYSICS-INFORMED SETTINGS #####
    "dimension"                : 3,
    "code_debugging"           : True,
    "enforce_realizability"    : False,
    "num_realizability_its"    : 5,
    "capping"                 : False,
    "cappingValue"            : 1e+6,
    "timeScale"               : 1
}

# For easier access to common settings groups
def get_network_settings():
    """Get neural network related settings"""
    return {
        "initial_neurons": OPTIMIZATION_SETTINGS["initial_neurons"],
        "initial_layers": OPTIMIZATION_SETTINGS["initial_layers"],
        "initial_activation": OPTIMIZATION_SETTINGS["initial_activation"],
        "initial_learningRate": OPTIMIZATION_SETTINGS["initial_learningRate"],
        "network_epochs": OPTIMIZATION_SETTINGS["network_epochs"],
        "batchSize": OPTIMIZATION_SETTINGS["batchSize"],
        "lossFunction": OPTIMIZATION_SETTINGS["lossFunction"]
    }

def get_optimization_bounds():
    """Get bounds for optimization parameters"""
    return {
        "layers": (
            OPTIMIZATION_SETTINGS["layers_lowerBound"],
            OPTIMIZATION_SETTINGS["layers_upperBound"]
        ),
        "neurons": (
            OPTIMIZATION_SETTINGS["neurons_lowerBound"],
            OPTIMIZATION_SETTINGS["neurons_upperBound"]
        ),
        "learning_rate": (
            OPTIMIZATION_SETTINGS["learning_lowerBound"],
            OPTIMIZATION_SETTINGS["learning_upperBound"]
        )
    }

def get_data_settings():
    """Get data processing related settings"""
    return {
        "centering_method": OPTIMIZATION_SETTINGS["centering_method"],
        "scaling_method": OPTIMIZATION_SETTINGS["scaling_method"],
        "training_ratio": OPTIMIZATION_SETTINGS["training_ratio"],
        "dimension": OPTIMIZATION_SETTINGS["dimension"]
    }

def get_physics_settings():
    """Get physics-related settings"""
    return {
        "dimension": OPTIMIZATION_SETTINGS["dimension"],
        "enforce_realizability": OPTIMIZATION_SETTINGS["enforce_realizability"],
        "num_realizability_its": OPTIMIZATION_SETTINGS["num_realizability_its"],
        "capping": OPTIMIZATION_SETTINGS["capping"],
        "cappingValue": OPTIMIZATION_SETTINGS["cappingValue"],
        "timeScale": OPTIMIZATION_SETTINGS["timeScale"]
    }