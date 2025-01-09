from typing import List
from dataclasses import dataclass, field
from pathlib import Path

def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent.parent

def default_hidden_layers() -> List[int]:
    """Default hidden layer configuration"""
    return [20, 20, 20]

@dataclass
class DiffusionConfig:
    """Configuration for diffusion PINN

    Attributes:
        hidden_layers: List of integers defining the size of each hidden layer
        activation: Activation function to use ('tanh', 'sin', or 'relu')
        initialization: Weight initialization method ('glorot' or 'he')
        diffusion_trainable: Whether to train the diffusion coefficient
        use_physics_loss: Whether to use physics-informed loss term
    """
    hidden_layers: List[int] = field(default_factory=default_hidden_layers)
    activation: str = 'tanh'
    initialization: str = 'glorot'
    diffusion_trainable: bool = True
    use_physics_loss: bool = True

    def get_data_dir(self) -> Path:
        """Get the path to the data directory."""
        return get_project_root() / "src" / "diffusion_pinn" / "data"

    def get_ground_truth_dir(self) -> Path:
        """Get the path to the ground truth directory."""
        return get_project_root() / "src" / "diffusion_pinn" / "ground_truth"

    def get_saved_models_dir(self) -> Path:
        """Get the path to the saved models directory."""
        return get_project_root() / "examples" / "saved_models"

    def get_data_file(self, filename: str) -> Path:
        """Get the full path to a data file"""
        #return self.get_data_dir() / filename
        return filename

    def get_ground_truth_file(self, filename: str) -> Path:
        """Get the full path to a ground truth file"""
        #return self.get_ground_truth_dir() / filename
        return filename

    def get_saved_model_file(self, filename: str) -> Path:
        """Get the full path to a saved model file"""
        #return self.get_saved_models_dir() / filename
        return filename
