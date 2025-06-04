import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import os
import json
import gc
from ..variables import PINN_VARIABLES

def create_and_initialize_pinn(inputfile: str,
                             N_u: int = PINN_VARIABLES['N_u'],
                             N_f: int = PINN_VARIABLES['N_f'],
                             N_i: int = PINN_VARIABLES['N_i'],
                             initial_D: float = PINN_VARIABLES['initial_D'],
                             seed: int = PINN_VARIABLES['random_seed']) -> Tuple['DiffusionPINN', Dict[str, tf.Tensor]]:
    """
    Create and initialize PINN with data - SAME AS V0.2.14 except seed parameter

    Args:
        inputfile: Path to data file
        N_u: Number of boundary/initial condition points
        N_f: Number of collocation points
        N_i: Number of interior supervision points
        initial_D: Initial guess for diffusion coefficient
        seed: Random seed for reproducibility

    Returns:
        Tuple of (initialized PINN, training data dictionary)
    """
    from ..data.processor import DiffusionDataProcessor
    from ..models.pinn import DiffusionPINN
    from ..config import DiffusionConfig

    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Process data
    data_processor = DiffusionDataProcessor(inputfile, seed=seed)

    # Get domain information
    domain_info = data_processor.get_domain_info()

    # Create PINN configuration
    config = DiffusionConfig(
        diffusion_trainable=True,
        use_physics_loss=True
    )

    # Initialize PINN
    pinn = DiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=initial_D,
        config=config,
        seed=seed,
        data_processor=data_processor
    )

    # Prepare training data
    training_data = data_processor.prepare_training_data(N_u, N_f, N_i, seed=seed)

    return pinn, training_data

def train_pinn(pinn: 'DiffusionPINN',
              data: Dict[str, tf.Tensor],
              optimizer: tf.keras.optimizers.Optimizer,
              epochs: int = 100,
              save_dir: str = None,
              checkpoint_frequency: int = 1000,
              seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """Training function with adaptive learning and constraints - V0.2.14 + minimal log(D)"""

    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    loss_history = []

    # MINIMAL CHANGE: Define acceptable range based on log bounds
    D_min = np.exp(pinn.log_D_min)  # ~2e-9
    D_max = np.exp(pinn.log_D_max)  # ~0.018

    print(f"Training with log(D) parameterization")
    print(f"Monitoring D range: [{D_min:.2e}, {D_max:.2e}]")

    # Use two-phase training for better convergence - SAME AS V0.2.14
    try:
        # Phase 1: Initial training with strong regularization
        print("Phase 1: Initial training...")
        phase1_epochs = min(epochs // 3, 1000)

        for epoch in range(phase1_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            print(f"\rPhase 1: {epoch+1}/{phase1_epochs}", end="", flush=True)

            with tf.GradientTape() as tape:
                # Compute losses - SAME AS V0.2.14
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

            # MINIMAL CHANGE: No explicit D constraint clipping since log(D) handles bounds naturally

            # Record history
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            if epoch % 100 == 0:
                print(f"\nPhase 1 - Epoch {epoch}, D = {D_history[-1]:.6e}")

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

            # MINIMAL CHANGE: No explicit D constraint clipping

            # Record history
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            if epoch % 100 == 0:
                print(f"\nPhase 2 - Epoch {epoch}, D = {D_history[-1]:.6e}")

        # Final summary
        final_D = D_history[-1]
        print(f"\nTraining completed. Final D: {final_D:.8e}")

        # Check if D stayed in reasonable bounds
        if final_D < D_min * 10 or final_D > D_max / 10:
            print(f"WARNING: Final D may be outside expected range")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")

    return D_history, loss_history

def save_checkpoint(pinn: 'DiffusionPINN', save_dir: str, epoch: str) -> None:
    """
    Save model checkpoint - MINIMAL CHANGE: Save log(D) info

    Args:
        pinn: PINN model to save
        save_dir: Directory to save checkpoint
        epoch: Epoch identifier
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
        'D_value': float(pinn.get_diffusion_coefficient()),
        'log_D_value': float(pinn.log_D.numpy()),  # ADDED: log(D) value
        'log_D_bounds': [float(pinn.log_D_min), float(pinn.log_D_max)],  # ADDED: log(D) bounds
        'parameterization': 'logarithmic'  # ADDED: parameterization type
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
    Load a pretrained PINN model - SAME AS V0.2.14

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