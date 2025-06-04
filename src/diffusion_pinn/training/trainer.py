import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import os
import json
import gc
from ..variables import PINN_VARIABLES
from ..models.pinn import DiffusionPINN

def create_and_initialize_pinn(inputfile: str,
                             N_u: int = PINN_VARIABLES['N_u'],
                             N_f: int = PINN_VARIABLES['N_f'],
                             N_i: int = PINN_VARIABLES['N_i'],
                             initial_D: float = PINN_VARIABLES['initial_D'],
                             seed: int = PINN_VARIABLES['random_seed']) -> Tuple['DiffusionPINN', Dict[str, tf.Tensor]]:
    """
    Create and initialize PINN with data - HYBRID approach

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

    # Process data
    data_processor = DiffusionDataProcessor(inputfile, seed=seed)

    # Get domain information
    domain_info = data_processor.get_domain_info()

    # Create PINN configuration
    config = DiffusionConfig(
        diffusion_trainable=True,
        use_physics_loss=True
    )

    # Initialize PINN with hybrid approach
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

def compute_stable_interior_loss(pinn, x_interior, c_interior):
    """Compute interior loss with stability enhancements"""
    if x_interior.shape[0] == 0:
        return tf.constant(0.0, dtype=tf.float32)

    c_pred = pinn.forward_pass(x_interior)
    raw_errors = c_pred - c_interior

    # Apply Huber loss for robustness
    delta = 0.1
    abs_errors = tf.abs(raw_errors)
    quadratic = tf.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    loss_value = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)

    return loss_value

def train_pinn(pinn: 'DiffusionPINN',
              data: Dict[str, tf.Tensor],
              optimizer: tf.keras.optimizers.Optimizer,
              epochs: int = 100,
              save_dir: str = None,
              checkpoint_frequency: int = 1000,
              seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    HYBRID training function: V0.2.14's two-phase approach with log(D) parameterization
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    log_D_history = []
    loss_history = []

    # Define acceptable range for monitoring (converted from log bounds)
    D_min = np.exp(pinn.log_D_min)  # ~2e-9
    D_max = np.exp(pinn.log_D_max)  # ~0.018

    print(f"\nStarting hybrid training with log(D) parameterization for {epochs} epochs")
    print(f"Monitoring D range: [{D_min:.2e}, {D_max:.2e}]")

    # Print initial state
    initial_D = pinn.get_diffusion_coefficient()
    initial_log_D = pinn.get_log_diffusion_coefficient()
    print(f"Initial D: {initial_D:.8e}")
    print(f"Initial log(D): {initial_log_D:.6f}")

    try:
        # Phase 1: Initial training with strong regularization (like V0.2.14)
        print("\n" + "="*60)
        print("Phase 1: Initial training with strong physics focus")
        print("="*60)
        phase1_epochs = min(epochs // 3, 2000)

        for epoch in range(phase1_epochs):
            # Clear memory periodically
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()
                gc.collect()

            with tf.GradientTape() as tape:
                # Compute losses using the hybrid loss function
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )

                # Add interior loss
                interior_loss = compute_stable_interior_loss(
                    pinn, data['X_i_train'], data['u_i_train']
                )
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
                losses['l2_reg'] = l2_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping
            gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Record history
            current_D = pinn.get_diffusion_coefficient()
            current_log_D = pinn.get_log_diffusion_coefficient()

            D_history.append(current_D)
            log_D_history.append(current_log_D)
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            # Progress reporting
            if epoch % 100 == 0:
                print(f"Phase 1 - Epoch {epoch:4d}: D={current_D:.6e}, log(D)={current_log_D:.4f}, "
                      f"Loss={losses['total']:.6f}")
                print(f"    Initial={losses['initial']:.6f}, Boundary={losses['boundary']:.6f}, "
                      f"Interior={interior_loss:.6f}, Physics={losses['physics']:.6f}")

                # Check for extreme values
                if current_D < D_min * 10 or current_D > D_max / 10:
                    print(f"    WARNING: D approaching bounds!")

        print(f"Phase 1 completed - Final D: {D_history[-1]:.6e}")

        # Phase 2: Fine-tuning with reduced regularization (like V0.2.14)
        print("\n" + "="*60)
        print("Phase 2: Fine-tuning with reduced regularization")
        print("="*60)
        phase2_epochs = epochs - phase1_epochs

        for epoch in range(phase2_epochs):
            # Clear memory periodically
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()
                gc.collect()

            with tf.GradientTape() as tape:
                # Same loss computation but with reduced regularization
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )

                interior_loss = compute_stable_interior_loss(
                    pinn, data['X_i_train'], data['u_i_train']
                )
                losses['interior'] = interior_loss

                # Reduced L2 regularization in phase 2
                l2_loss = 0.00001 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                total_loss = (
                    losses['total'] +
                    pinn.loss_weights['interior'] * interior_loss +
                    l2_loss
                )
                losses['total'] = total_loss
                losses['l2_reg'] = l2_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Reduced gradient clipping in phase 2
            gradients, global_norm = tf.clip_by_global_norm(gradients, 2.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Record history
            current_D = pinn.get_diffusion_coefficient()
            current_log_D = pinn.get_log_diffusion_coefficient()

            D_history.append(current_D)
            log_D_history.append(current_log_D)
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            # Progress reporting
            if epoch % 100 == 0:
                print(f"Phase 2 - Epoch {epoch:4d}: D={current_D:.6e}, log(D)={current_log_D:.4f}, "
                      f"Loss={losses['total']:.6f}")

                # Check convergence
                if len(D_history) >= 100:
                    recent_d = D_history[-100:]
                    d_std = np.std(recent_d)
                    d_mean = np.mean(recent_d)
                    relative_std = d_std / d_mean if d_mean > 0 else float('inf')
                    print(f"    Convergence metric: {relative_std:.6f}")

        # Final training summary
        final_D = D_history[-1]
        final_log_D = log_D_history[-1]
        final_loss = loss_history[-1]['total']

        print(f"\n{'='*60}")
        print("HYBRID TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total epochs: {len(D_history)}")
        print(f"Final diffusion coefficient: {final_D:.8e}")
        print(f"Final log(D): {final_log_D:.6f}")
        print(f"Final loss: {final_loss:.6f}")

        # Test network predictions
        test_points = tf.constant([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 2.5],
            [1.0, 1.0, 5.0]
        ], dtype=tf.float32)

        test_preds = pinn.predict(test_points)
        pred_range = tf.reduce_max(test_preds) - tf.reduce_min(test_preds)

        print(f"Prediction range test: {pred_range.numpy():.6f}")
        if pred_range < 1e-6:
            print("WARNING: Network predicting uniform values!")
        else:
            print("SUCCESS: Network predicting varying values!")

        # Check final convergence
        if len(D_history) >= 100:
            recent_d = D_history[-100:]
            final_std = np.std(recent_d)
            final_mean = np.mean(recent_d)
            final_relative_std = final_std / final_mean if final_mean > 0 else float('inf')
            print(f"Final convergence metric: {final_relative_std:.6f}")
            print(f"Converged: {'Yes' if final_relative_std < 0.05 else 'No'}")

        # Save final model
        if save_dir:
            save_checkpoint(pinn, save_dir, "final_hybrid")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    return D_history, loss_history

def save_checkpoint(pinn: 'DiffusionPINN', save_dir: str, epoch: str) -> None:
    """
    Save model checkpoint with hybrid information

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
        'log_D_value': float(pinn.get_log_diffusion_coefficient()),
        'log_D_bounds': [float(pinn.log_D_min), float(pinn.log_D_max)],
        'parameterization': 'hybrid_logarithmic'
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

    # Save log(D) information
    log_d_info = {
        'log_D_value': float(pinn.get_log_diffusion_coefficient()),
        'D_value': float(pinn.get_diffusion_coefficient()),
        'log_D_bounds': [float(pinn.log_D_min), float(pinn.log_D_max)]
    }

    with open(os.path.join(save_dir, f'log_d_info_{epoch}.json'), 'w') as f:
        json.dump(log_d_info, f, indent=4)

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