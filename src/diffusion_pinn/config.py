from typing import List
from dataclasses import dataclass, field

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
