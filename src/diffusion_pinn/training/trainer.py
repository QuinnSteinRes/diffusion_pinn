import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import os
import json
from ..variables import PINN_VARIABLES

def create_and_initialize_pinn(inputfile: str,
                             N_u: int = PINN_VARIABLES['N_u'],
                             N_f: int = PINN_VARIABLES['N_f'],
                             N_i: int = PINN_VARIABLES['N_i'],
                             initial_D: float = PINN_VARIABLES['initial_D']) -> Tuple['DiffusionPINN', Dict[str, tf.Tensor]]:
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
        #hidden_layers=[40, 40, 40],  # Reduced from [40, 40, 40, 40, 40]
        #activation='tanh',
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
              save_dir: str = None,
              checkpoint_frequency: int = 1000) -> Tuple[List[float], List[Dict[str, float]]]:
    """Training function with adaptive learning and constraints"""
    D_history = []
    loss_history = []

    # Define acceptable range for diffusion coefficient
    D_min = 0
    D_max = 0.5

    # Use two-phase training for better convergence
    try:
        # Phase 1: Initial training with strong regularization
        print("Phase 1: Initial training...")
        phase1_epochs = min(epochs // 3, 1000)

        for epoch in range(phase1_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            print(f"\rPhase 1: {epoch+1}/{phase1_epochs}", end="", flush=True)

            with tf.GradientTape() as tape:
                # Compute losses
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )

                interior_loss = tf.reduce_mean(tf.square(
                    pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                ))
                losses['interior'] = interior_loss

                # Add L2 regularization for weights in phase 1
                l2_loss = 0.0001 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                # Strong weight on physics in phase 1
                total_loss = (
                    losses['total'] +
                    pinn.loss_weights['interior'] * interior_loss +
                    l2_loss
                )
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Enforce diffusion coefficient constraints
            if pinn.config.diffusion_trainable:
                D_value = pinn.D.numpy()
                if D_value < D_min or D_value > D_max:
                    constrained_D = np.clip(D_value, D_min, D_max)
                    pinn.D.assign(constrained_D)

            # Record history
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            if epoch % 100 == 0:
                print(f"\nPhase 1 - Epoch {epoch}, D = {D_history[-1]:.6f}")

        # Phase 2: Fine-tuning with reduced regularization
        print("\nPhase 2: Fine-tuning...")
        phase2_epochs = epochs - phase1_epochs

        for epoch in range(phase2_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            print(f"\rPhase 2: {epoch+1}/{phase2_epochs}", end="", flush=True)

            with tf.GradientTape() as tape:
                # Same loss computation but with reduced regularization
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )

                interior_loss = tf.reduce_mean(tf.square(
                    pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                ))
                losses['interior'] = interior_loss

                # No L2 regularization in phase 2
                total_loss = losses['total'] + pinn.loss_weights['interior'] * interior_loss
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Reduced gradient clipping in phase 2
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Enforce diffusion coefficient constraints
            if pinn.config.diffusion_trainable:
                D_value = pinn.D.numpy()
                if D_value < D_min or D_value > D_max:
                    constrained_D = np.clip(D_value, D_min, D_max)
                    pinn.D.assign(constrained_D)

            # Record history
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            if epoch % 100 == 0:
                print(f"\nPhase 2 - Epoch {epoch}, D = {D_history[-1]:.6f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")

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
