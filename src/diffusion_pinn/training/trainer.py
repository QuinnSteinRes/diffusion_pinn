import tensorflow as tf
from typing import Dict, List, Tuple
import os
import json

def create_and_initialize_pinn(inputfile: str, 
                             N_u: int, 
                             N_f: int, 
                             N_i: int,
                             initial_D: float = 1.0) -> Tuple['DiffusionPINN', Dict[str, tf.Tensor]]:
    """
    Create and initialize PINN with data
    
    Args:
        inputfile: Path to data file
        N_u: Number of boundary/initial condition points
        N_f: Number of collocation points
        N_i: Number of interior supervision points
        initial_D: Initial guess for diffusion coefficient
        
    Returns:
        Tuple of (initialized PINN, training data dictionary)
    """
    from ..data.processor import DiffusionDataProcessor
    from ..models.pinn import DiffusionPINN
    from ..config import DiffusionConfig
    
    # Process data
    data_processor = DiffusionDataProcessor(inputfile)
    
    # Get domain information
    domain_info = data_processor.get_domain_info()
    
    # Create PINN configuration
    config = DiffusionConfig(
        hidden_layers=[40, 40, 40, 40, 40],
        activation='tanh',
        diffusion_trainable=True,
        use_physics_loss=True
    )
    
    # Initialize PINN
    pinn = DiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=initial_D,
        config=config
    )
    
    # Prepare training data
    training_data = data_processor.prepare_training_data(N_u, N_f, N_i)
    
    return pinn, training_data

def train_pinn(pinn: 'DiffusionPINN', 
               data: Dict[str, tf.Tensor], 
               optimizer: tf.keras.optimizers.Optimizer,
               epochs: int = 100,
               save_dir: str = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Training function with interrupt handling and intermediate plotting
    
    Args:
        pinn: PINN model to train
        data: Dictionary containing training data
        optimizer: TensorFlow optimizer
        epochs: Number of training epochs
        save_dir: Optional directory to save model checkpoints
        
    Returns:
        Tuple of (diffusion coefficient history, loss history)
    """
    D_history = []
    loss_history = []
    
    try:
        for epoch in range(epochs):
            print(f"\rTraining progress: {epoch+1}/{epochs} epochs", end="")
            
            with tf.GradientTape() as tape:
                # Compute boundary/initial losses
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )
                
                # Compute interior supervision loss
                c_pred_interior = pinn.forward_pass(data['X_i_train'])
                interior_loss = tf.reduce_mean(tf.square(c_pred_interior - data['u_i_train']))
                losses['interior'] = interior_loss
                
                # Update total loss
                total_loss = losses['total'] + pinn.loss_weights['interior'] * interior_loss
                losses['total'] = total_loss
            
            gradients = tape.gradient(total_loss, pinn.get_trainable_variables())
            optimizer.apply_gradients(zip(gradients, pinn.get_trainable_variables()))
            
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: v.numpy() for k, v in losses.items()})
            
            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}")
                for key, value in losses.items():
                    print(f"{key.capitalize()} loss: {value.numpy():.6f}")
                print(f"Current D = {D_history[-1]:.6f}\n")
                
                # Save checkpoint if directory provided
                if save_dir:
                    save_checkpoint(pinn, save_dir, epoch)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    return D_history, loss_history

def save_checkpoint(pinn: 'DiffusionPINN', save_dir: str, epoch: int) -> None:
    """
    Save model checkpoint
    
    Args:
        pinn: PINN model to save
        save_dir: Directory to save checkpoint
        epoch: Current epoch number
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_dict = {
        'hidden_layers': pinn.config.hidden_layers,
        'activation': pinn.config.activation,
        'initialization': pinn.config.initialization,
        'diffusion_trainable': pinn.config.diffusion_trainable,
        'use_physics_loss': pinn.config.use_physics_loss,
        'spatial_bounds': {
            'x': [float(pinn.x_bounds[0]), float(pinn.x_bounds[1])],
            'y': [float(pinn.y_bounds[0]), float(pinn.y_bounds[1])]
        },
        'time_bounds': [float(pinn.t_bounds[0]), float(pinn.t_bounds[1])],
        'D_value': float(pinn.get_diffusion_coefficient())
    }
    
    with open(os.path.join(save_dir, f'config_{epoch}.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Save weights and biases
    weights_dict = {f'weight_{i}': w.numpy().tolist() 
                   for i, w in enumerate(pinn.weights)}
    biases_dict = {f'bias_{i}': b.numpy().tolist() 
                   for i, b in enumerate(pinn.biases)}
    
    with open(os.path.join(save_dir, f'weights_{epoch}.json'), 'w') as f:
        json.dump(weights_dict, f)
    with open(os.path.join(save_dir, f'biases_{epoch}.json'), 'w') as f:
        json.dump(biases_dict, f)

def load_pretrained_pinn(load_dir: str, data_path: str) -> Tuple['DiffusionPINN', 'DiffusionDataProcessor']:
    """
    Load a pretrained PINN model
    
    Args:
        load_dir: Directory containing saved model
        data_path: Path to data file
        
    Returns:
        Tuple of (loaded PINN, data processor)
    """
    from ..data.processor import DiffusionDataProcessor
    from ..models.pinn import DiffusionPINN
    from ..config import DiffusionConfig
    
    with open(os.path.join(load_dir, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    
    data_processor = DiffusionDataProcessor(data_path, normalize_spatial=True)
    
    config = DiffusionConfig(
        hidden_layers=config_dict['hidden_layers'],
        activation=config_dict['activation'],
        initialization=config_dict['initialization'],
        diffusion_trainable=config_dict['diffusion_trainable'],
        use_physics_loss=config_dict['use_physics_loss']
    )
    
    pinn = DiffusionPINN(
        spatial_bounds=config_dict['spatial_bounds'],
        time_bounds=tuple(config_dict['time_bounds']),
        initial_D=config_dict['D_value'],
        config=config
    )
    
    with open(os.path.join(load_dir, 'weights.json'), 'r') as f:
        weights_dict = json.load(f)
    with open(os.path.join(load_dir, 'biases.json'), 'r') as f:
        biases_dict = json.load(f)
    
    for i in range(len(pinn.weights)):
        pinn.weights[i].assign(weights_dict[f'weight_{i}'])
        pinn.biases[i].assign(biases_dict[f'bias_{i}'])
    
    return pinn, data_processor