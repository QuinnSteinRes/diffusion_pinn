from .processor import DiffusionDataProcessor

import os

def get_data_path(filename):
    """Return the absolute path to a data file in the data directory"""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(data_dir, filename)
