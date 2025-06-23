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
    Create and initialize PINN with data - includes logarithmic D parameterization

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

    # Initialize PINN with logarithmic parameterization
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

    # Forward pass predictions
    c_pred = pinn.forward_pass(x_interior)

    # Calculate raw errors
    raw_errors = c_pred - c_interior

    # Apply Huber loss for robustness to outliers
    delta = 0.1
    abs_errors = tf.abs(raw_errors)
    quadratic = tf.minimum(abs_errors, delta)
    linear = abs_errors - quadratic

    # Combine for final loss
    return tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)

def deterministic_train_pinn_log_d(pinn: 'DiffusionPINN',
                                 data: Dict[str, tf.Tensor],
                                 optimizer: tf.keras.optimizers.Optimizer,
                                 epochs: int = 100,
                                 save_dir: str = None,
                                 checkpoint_frequency: int = 1000,
                                 seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Deterministic training with logarithmic D parameterization - UNBIASED VERSION
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    log_D_history = []
    loss_history = []

    # Enhanced training schedule for logarithmic parameterization
    phase_configs = [
        {
            'name': 'Phase 1: Physics Learning (Log D)',
            'epochs': epochs // 4,
            'weights': {'initial': 2.0, 'boundary': 2.0, 'interior': 3.0, 'physics': 5.0},
            'lr_schedule': lambda epoch, total: 1e-3 * (0.95 ** (epoch // 100)),
            'regularization': 0.001,
            'description': 'Focus on learning physics with log(D) parameterization'
        },
        {
            'name': 'Phase 2: Data Fitting (Log D)',
            'epochs': epochs // 2,
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 5.0, 'physics': 3.0},
            'lr_schedule': lambda epoch, total: 5e-4 * (0.98 ** (epoch // 50)),
            'regularization': 0.0005,
            'description': 'Balance physics with data fitting using log(D)'
        },
        {
            'name': 'Phase 3: Fine Tuning (Log D)',
            'epochs': epochs // 4,
            'weights': {'initial': 0.5, 'boundary': 0.5, 'interior': 10.0, 'physics': 1.0},
            'lr_schedule': lambda epoch, total: 1e-4 * (0.99 ** (epoch // 25)),
            'regularization': 0.0001,
            'description': 'Fine-tune with emphasis on data accuracy'
        }
    ]

    epoch_counter = 0
    convergence_history = []

    # Log(D) bounds for monitoring
    log_D_min_monitor = -25.0
    log_D_max_monitor = -1.0

    print(f"\nStarting deterministic training with logarithmic D parameterization for {epochs} epochs")
    print(f"Phase breakdown: {[phase['epochs'] for phase in phase_configs]} epochs")

    # Print initial state
    initial_D = pinn.get_diffusion_coefficient()
    initial_log_D = pinn.get_log_diffusion_coefficient()
    print(f"Initial D: {initial_D:.8e}")
    print(f"Initial log(D): {initial_log_D:.6f}")

    try:
        for phase_idx, phase in enumerate(phase_configs):
            print(f"\n{'='*60}")
            print(f"{phase['name']} - {phase['epochs']} epochs")
            print(f"Description: {phase['description']}")
            print(f"Loss weights: {phase['weights']}")
            print(f"{'='*60}")

            phase_start_epoch = epoch_counter

            for epoch in range(phase['epochs']):
                # Deterministic learning rate
                current_lr = phase['lr_schedule'](epoch, phase['epochs'])

                # Set learning rate
                if hasattr(optimizer, 'learning_rate'):
                    if hasattr(optimizer.learning_rate, 'assign'):
                        optimizer.learning_rate.assign(current_lr)

                # Training step with logarithmic D
                with tf.GradientTape() as tape:
                    # Compute losses with fixed weights
                    losses = pinn.loss_fn(
                        x_data=data['X_u_train'],
                        c_data=data['u_train'],
                        x_physics=data['X_f_train'],
                        weights=phase['weights']
                    )

                    # Compute interior loss separately
                    interior_loss = compute_stable_interior_loss(
                        pinn, data['X_i_train'], data['u_i_train']
                    )

                    # L2 regularization on network weights
                    l2_loss = phase['regularization'] * sum(
                        tf.reduce_sum(tf.square(w)) for w in pinn.weights
                    )

                    # UNBIASED: Remove all log(D) regularization that creates bias
                    log_d_reg = tf.constant(0.0, dtype=tf.float32)

                    # Total loss calculation
                    total_loss = (
                        losses['total'] +
                        interior_loss +
                        l2_loss +
                        log_d_reg
                    )

                    losses['total'] = total_loss
                    losses['interior'] = interior_loss
                    losses['l2_reg'] = l2_loss
                    losses['log_d_reg_unbiased'] = log_d_reg

                # Apply gradients with consistent clipping
                trainable_vars = pinn.get_trainable_variables()
                gradients = tape.gradient(total_loss, trainable_vars)

                # Gradient clipping
                gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Record history
                current_D = pinn.get_diffusion_coefficient()
                current_log_D = pinn.get_log_diffusion_coefficient()

                D_history.append(current_D)
                log_D_history.append(current_log_D)
                loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

                # Enhanced convergence checking for log(D)
                if epoch_counter % 100 == 0:
                    # Check convergence over last 50 epochs using log(D)
                    if len(log_D_history) >= 50:
                        recent_log_d = log_D_history[-50:]
                        log_d_std = np.std(recent_log_d)
                        log_d_mean = np.mean(recent_log_d)

                        convergence_metric = log_d_std
                        convergence_history.append(convergence_metric)

                        print(f"Epoch {epoch_counter:5d}: D={current_D:.6e}, log(D)={current_log_D:.6f}, "
                              f"Loss={losses['total']:.6f}, LR={current_lr:.2e}, "
                              f"log(D)_std={convergence_metric:.6f}")

                        # Convergence criteria
                        if (convergence_metric < 0.01 and
                            len(convergence_history) >= 3 and
                            all(conv < 0.01 for conv in convergence_history[-3:])):
                            print(f"Converged at epoch {epoch_counter} (log(D) std < 0.01)")

                        # Warning for extreme values
                        if current_log_D < log_D_min_monitor + 1.0:
                            print(f"WARNING: log(D) approaching lower bound: {current_log_D:.6f}")
                        elif current_log_D > log_D_max_monitor - 1.0:
                            print(f"WARNING: log(D) approaching upper bound: {current_log_D:.6f}")

                epoch_counter += 1

                # Memory cleanup
                if epoch_counter % 100 == 0:
                    gc.collect()

            # Phase completion summary
            phase_final_D = D_history[-1]
            phase_final_log_D = log_D_history[-1]
            phase_final_loss = loss_history[-1]['total']
            print(f"\nPhase {phase_idx + 1} completed:")
            print(f"  Final D: {phase_final_D:.6e}")
            print(f"  Final log(D): {phase_final_log_D:.6f}")
            print(f"  Final Loss: {phase_final_loss:.6f}")
            print(f"  Epochs: {phase_start_epoch} to {epoch_counter-1}")

            # Save checkpoint after each phase
            if save_dir:
                save_checkpoint_log_d(pinn, save_dir, f"phase_{phase_idx+1}_final",
                                     log_D_history, D_history)

        # Final training summary
        final_D = D_history[-1]
        final_log_D = log_D_history[-1]
        final_loss = loss_history[-1]['total']

        print(f"\n{'='*60}")
        print("LOGARITHMIC D TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total epochs: {epoch_counter}")
        print(f"Final diffusion coefficient: {final_D:.8e}")
        print(f"Final log(D): {final_log_D:.6f}")
        print(f"Final loss: {final_loss:.6f}")

        # Enhanced prediction range test
        test_points = tf.constant([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 2.5],
            [1.0, 1.0, 5.0]
        ], dtype=tf.float32)

        test_preds = pinn.predict(test_points)
        pred_range = tf.reduce_max(test_preds) - tf.reduce_min(test_preds)

        print(f"Prediction range test: {pred_range.numpy():.6f}")
        if pred_range < 1e-6:
            print("WARNING: Still predicting uniform values!")
        else:
            print("SUCCESS: Network predicting varying values!")

        # Check final convergence using log(D)
        if len(log_D_history) >= 100:
            recent_log_d = log_D_history[-100:]
            final_log_d_std = np.std(recent_log_d)
            print(f"Final log(D) convergence metric: {final_log_d_std:.6f}")
            print(f"Converged: {'Yes' if final_log_d_std < 0.05 else 'No'}")

        # Log(D) training statistics
        log_d_range = max(log_D_history) - min(log_D_history)
        print(f"log(D) exploration range: {log_d_range:.6f}")
        print(f"D exploration range: {max(D_history):.2e} to {min(D_history):.2e}")

        # Save final model
        if save_dir:
            save_checkpoint_log_d(pinn, save_dir, "final_log_d", log_D_history, D_history)

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    return D_history, loss_history

def deterministic_train_pinn(pinn: 'DiffusionPINN',
                           data: Dict[str, tf.Tensor],
                           optimizer: tf.keras.optimizers.Optimizer,
                           epochs: int = 100,
                           save_dir: str = None,
                           checkpoint_frequency: int = 1000,
                           seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Standard deterministic training (fallback for non-logarithmic PINNs)
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    loss_history = []

    # Standard training schedule
    phase_configs = [
        {
            'name': 'Phase 1: Physics Learning',
            'epochs': epochs // 4,
            'weights': {'initial': 2.0, 'boundary': 2.0, 'interior': 0.5, 'physics': 10.0},
            'lr_schedule': lambda epoch, total: 1e-3 * (0.95 ** (epoch // 100)),
            'regularization': 0.001,
            'description': 'Focus on learning physics constraints'
        },
        {
            'name': 'Phase 2: Data Fitting',
            'epochs': epochs // 2,
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 5.0, 'physics': 3.0},
            'lr_schedule': lambda epoch, total: 5e-4 * (0.98 ** (epoch // 50)),
            'regularization': 0.0005,
            'description': 'Balance physics with data fitting'
        },
        {
            'name': 'Phase 3: Fine Tuning',
            'epochs': epochs // 4,
            'weights': {'initial': 0.5, 'boundary': 0.5, 'interior': 10.0, 'physics': 1.0},
            'lr_schedule': lambda epoch, total: 1e-4 * (0.99 ** (epoch // 25)),
            'regularization': 0.0001,
            'description': 'Fine-tune with emphasis on data accuracy'
        }
    ]

    epoch_counter = 0
    convergence_history = []

    # Define acceptable range for diffusion coefficient
    D_min = 1e-6
    D_max = 1e-2

    print(f"\nStarting deterministic training for {epochs} epochs")
    print(f"Phase breakdown: {[phase['epochs'] for phase in phase_configs]} epochs")

    try:
        for phase_idx, phase in enumerate(phase_configs):
            print(f"\n{'='*60}")
            print(f"{phase['name']} - {phase['epochs']} epochs")
            print(f"Description: {phase['description']}")
            print(f"Loss weights: {phase['weights']}")
            print(f"{'='*60}")

            phase_start_epoch = epoch_counter

            for epoch in range(phase['epochs']):
                # Deterministic learning rate
                current_lr = phase['lr_schedule'](epoch, phase['epochs'])

                # Set learning rate
                if hasattr(optimizer, 'learning_rate'):
                    if hasattr(optimizer.learning_rate, 'assign'):
                        optimizer.learning_rate.assign(current_lr)

                # Training step
                with tf.GradientTape() as tape:
                    # Compute losses with fixed weights
                    losses = pinn.loss_fn(
                        x_data=data['X_u_train'],
                        c_data=data['u_train'],
                        x_physics=data['X_f_train'],
                        weights=phase['weights']
                    )

                    # Compute interior loss separately
                    interior_loss = compute_stable_interior_loss(
                        pinn, data['X_i_train'], data['u_i_train']
                    )

                    # L2 regularization
                    l2_loss = phase['regularization'] * sum(
                        tf.reduce_sum(tf.square(w)) for w in pinn.weights
                    )

                    # UNBIASED: Remove diffusion coefficient bias regularization
                    d_reg = tf.constant(0.0, dtype=tf.float32)

                    # Total loss calculation
                    total_loss = (
                        losses['total'] +
                        l2_loss +
                        d_reg
                    )

                    losses['total'] = total_loss
                    losses['interior'] = interior_loss
                    losses['l2_reg'] = l2_loss
                    losses['d_reg'] = d_reg

                # Apply gradients
                trainable_vars = pinn.get_trainable_variables()
                gradients = tape.gradient(total_loss, trainable_vars)

                # Gradient clipping
                gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Record history
                current_D = pinn.get_diffusion_coefficient()
                D_history.append(current_D)
                loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

                # Convergence checking
                if epoch_counter % 100 == 0:
                    if len(D_history) >= 50:
                        recent_d = D_history[-50:]
                        d_std = np.std(recent_d)
                        d_mean = np.mean(recent_d)
                        relative_std = d_std / d_mean if d_mean > 0 else float('inf')
                        convergence_history.append(relative_std)
                        print(f"Epoch {epoch_counter:5d}: D={current_D:.6f}, "
                              f"Loss={losses['total']:.6f}, LR={current_lr:.2e}, "
                              f"RelStd={relative_std:.6f}")

                epoch_counter += 1

                # Memory cleanup
                if epoch_counter % 100 == 0:
                    gc.collect()

            # Phase completion summary
            phase_final_D = D_history[-1]
            phase_final_loss = loss_history[-1]['total']
            print(f"\nPhase {phase_idx + 1} completed:")
            print(f"  Final D: {phase_final_D:.6f}")
            print(f"  Final Loss: {phase_final_loss:.6f}")

            # Save checkpoint after each phase
            if save_dir:
                save_checkpoint(pinn, save_dir, f"phase_{phase_idx+1}_final")

        # Final training summary
        final_D = D_history[-1]
        final_loss = loss_history[-1]['total']

        print(f"\n{'='*60}")
        print("DETERMINISTIC TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total epochs: {epoch_counter}")
        print(f"Final diffusion coefficient: {final_D:.8f}")
        print(f"Final loss: {final_loss:.6f}")

        # Save final model
        if save_dir:
            save_checkpoint(pinn, save_dir, "final_deterministic")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    return D_history, loss_history

def train_pinn(pinn: 'DiffusionPINN',
              data: Dict[str, tf.Tensor],
              optimizer: tf.keras.optimizers.Optimizer,
              epochs: int = 100,
              save_dir: str = None,
              checkpoint_frequency: int = 1000,
              seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Main training function - uses logarithmic D training if PINN has log_D attribute
    """
    # Check if this PINN uses logarithmic parameterization
    if hasattr(pinn, 'log_D'):
        print("Detected logarithmic D parameterization - using enhanced training...")
        return deterministic_train_pinn_log_d(
            pinn=pinn,
            data=data,
            optimizer=optimizer,
            epochs=epochs,
            save_dir=save_dir,
            checkpoint_frequency=checkpoint_frequency,
            seed=seed
        )
    else:
        print("Using standard deterministic training...")
        return deterministic_train_pinn(
            pinn=pinn,
            data=data,
            optimizer=optimizer,
            epochs=epochs,
            save_dir=save_dir,
            checkpoint_frequency=checkpoint_frequency,
            seed=seed
        )

def save_checkpoint_log_d(pinn: 'DiffusionPINN', save_dir: str, epoch: str,
                         log_D_history: List[float], D_history: List[float]) -> None:
    """Save model checkpoint with log(D) information"""
    os.makedirs(save_dir, exist_ok=True)

    # Save enhanced configuration including log(D) information
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
        'parameterization': 'logarithmic'
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

    # Save log(D) history for analysis
    log_d_history_dict = {
        'log_D_history': log_D_history,
        'D_history': D_history,
        'final_D': float(pinn.get_diffusion_coefficient()),
        'final_log_D': float(pinn.get_log_diffusion_coefficient())
    }

    with open(os.path.join(save_dir, f'log_d_history_{epoch}.json'), 'w') as f:
        json.dump(log_d_history_dict, f, indent=4)

def save_checkpoint(pinn: 'DiffusionPINN', save_dir: str, epoch: str) -> None:
    """
    Save model checkpoint (standard version)

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
        'D_value': float(pinn.get_diffusion_coefficient())
    }

    # Add log_D info if available
    if hasattr(pinn, 'log_D'):
        config_dict['log_D_value'] = float(pinn.get_log_diffusion_coefficient())
        config_dict['log_D_bounds'] = [float(pinn.log_D_min), float(pinn.log_D_max)]
        config_dict['parameterization'] = 'logarithmic'
    else:
        config_dict['parameterization'] = 'standard'

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