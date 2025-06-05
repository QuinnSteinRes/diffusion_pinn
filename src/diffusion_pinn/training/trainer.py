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
                             seed: int = None) -> Tuple['DiffusionPINN', Dict[str, tf.Tensor]]:
    """
    Create and initialize PINN with data - now with logarithmic D parameterization

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

    # Set random seeds
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Process data with seed for reproducibility
    data_processor = DiffusionDataProcessor(inputfile, seed=seed)

    # Get domain information
    domain_info = data_processor.get_domain_info()

    # Create PINN configuration (keeping v0.2.14 defaults)
    config = DiffusionConfig(
        diffusion_trainable=True,
        use_physics_loss=True
    )

    # Initialize PINN with logarithmic parameterization
    pinn = DiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=initial_D,
        config=config,
        seed=seed
    )

    # Prepare training data with v0.2.14 temporal density
    training_data = data_processor.prepare_training_data(N_u, N_f, N_i, temporal_density=10, seed=seed)

    return pinn, training_data

def train_pinn(pinn: 'DiffusionPINN',
              data: Dict[str, tf.Tensor],
              optimizer: tf.keras.optimizers.Optimizer,
              epochs: int = 100,
              save_dir: str = None,
              checkpoint_frequency: int = 1000,
              seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Training function with v0.2.14 two-phase approach but adapted for logarithmic D
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    log_D_history = []
    loss_history = []

    # Define acceptable range for diffusion coefficient (wider for log D)
    D_min = 1e-8
    D_max = 1e-1

    print(f"\nStarting training with logarithmic D parameterization for {epochs} epochs")
    print(f"Initial D: {pinn.get_diffusion_coefficient():.8e}")
    print(f"Initial log(D): {pinn.get_log_diffusion_coefficient():.6f}")

    # Use v0.2.14 two-phase training approach but adapted for log D
    try:
        # Phase 1: Initial training with strong regularization
        print("Phase 1: Initial training with logarithmic D...")
        phase1_epochs = min(epochs // 3, 1000)

        for epoch in range(phase1_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            if epoch % 100 == 0:
                print(f"Phase 1: {epoch+1}/{phase1_epochs}, D={pinn.get_diffusion_coefficient():.6e}, log(D)={pinn.get_log_diffusion_coefficient():.4f}")

            with tf.GradientTape() as tape:
                # Compute losses using v0.2.14 approach
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )

                # Interior loss (v0.2.14 style)
                interior_loss = tf.reduce_mean(tf.square(
                    pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                ))
                losses['interior'] = interior_loss

                # Add L2 regularization for weights in phase 1 (v0.2.14 style)
                l2_loss = 0.0001 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                # Total loss with v0.2.14 weighting but updated for log D
                total_loss = (
                    losses['total'] +
                    pinn.loss_weights['interior'] * interior_loss +
                    l2_loss
                )
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping (v0.2.14 style)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # UPDATED: Logarithmic D constraint (replaces v0.2.14 D clipping)
            # Soft constraint for log(D) to prevent extreme values
            if pinn.config.diffusion_trainable:
                current_log_D = pinn.log_D.numpy()
                if current_log_D < pinn.log_D_min or current_log_D > pinn.log_D_max:
                    # Apply soft constraint - nudge towards bounds rather than hard clip
                    if current_log_D < pinn.log_D_min:
                        nudged_log_D = 0.9 * current_log_D + 0.1 * pinn.log_D_min
                    else:
                        nudged_log_D = 0.9 * current_log_D + 0.1 * pinn.log_D_max
                    pinn.log_D.assign(nudged_log_D)

            # Record history
            current_D = pinn.get_diffusion_coefficient()
            current_log_D = pinn.get_log_diffusion_coefficient()
            D_history.append(current_D)
            log_D_history.append(current_log_D)
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

        # Phase 2: Fine-tuning with reduced regularization (v0.2.14 style)
        print("Phase 2: Fine-tuning with logarithmic D...")
        phase2_epochs = epochs - phase1_epochs

        for epoch in range(phase2_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            if epoch % 100 == 0:
                print(f"Phase 2: {epoch+1}/{phase2_epochs}, D={pinn.get_diffusion_coefficient():.6e}, log(D)={pinn.get_log_diffusion_coefficient():.4f}")

            with tf.GradientTape() as tape:
                # Same loss computation but with reduced regularization (v0.2.14 style)
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )

                interior_loss = tf.reduce_mean(tf.square(
                    pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                ))
                losses['interior'] = interior_loss

                # No L2 regularization in phase 2 (v0.2.14 style)
                total_loss = losses['total'] + pinn.loss_weights['interior'] * interior_loss
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Reduced gradient clipping in phase 2 (v0.2.14 style)
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # UPDATED: Softer constraint in phase 2 for log(D)
            if pinn.config.diffusion_trainable:
                current_log_D = pinn.log_D.numpy()
                # Only apply very soft constraints in phase 2
                if current_log_D < pinn.log_D_min - 2.0:
                    pinn.log_D.assign(pinn.log_D_min - 2.0)
                elif current_log_D > pinn.log_D_max + 2.0:
                    pinn.log_D.assign(pinn.log_D_max + 2.0)

            # Record history
            current_D = pinn.get_diffusion_coefficient()
            current_log_D = pinn.get_log_diffusion_coefficient()
            D_history.append(current_D)
            log_D_history.append(current_log_D)
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

        # Final summary
        final_D = pinn.get_diffusion_coefficient()
        final_log_D = pinn.get_log_diffusion_coefficient()
        print(f"\nTraining completed:")
        print(f"Final D: {final_D:.8e}")
        print(f"Final log(D): {final_log_D:.6f}")

        # Check convergence
        if len(log_D_history) >= 100:
            recent_log_d = log_D_history[-100:]
            log_d_std = np.std(recent_log_d)
            print(f"log(D) convergence metric (last 100 epochs): {log_d_std:.6f}")
            print(f"Converged: {'Yes' if log_d_std < 0.05 else 'No'}")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    return D_history, loss_history

def save_checkpoint(pinn: 'DiffusionPINN', save_dir: str, epoch: int) -> None:
    """
    Save model checkpoint with logarithmic D information

    Args:
        pinn: PINN model to save
        save_dir: Directory to save checkpoint
        epoch: Current epoch number
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration with log D info
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
        'log_D_value': float(pinn.get_log_diffusion_coefficient()) if hasattr(pinn, 'log_D') else None,
        'parameterization': 'logarithmic' if hasattr(pinn, 'log_D') else 'standard'
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