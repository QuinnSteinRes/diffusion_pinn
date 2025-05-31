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
    Create and initialize PINN with data - now includes smart initialization

    Args:
        inputfile: Path to data file
        N_u: Number of boundary/initial condition points
        N_f: Number of collocation points
        N_i: Number of interior supervision points
        initial_D: Initial guess for diffusion coefficient (will be overridden by smart init)
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

    # Initialize PINN with data processor for smart initialization
    pinn = DiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=initial_D,  # This will be overridden by smart initialization
        config=config,
        seed=seed,
        data_processor=data_processor  # This enables smart initialization
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

def deterministic_train_pinn(pinn: 'DiffusionPINN',
                           data: Dict[str, tf.Tensor],
                           optimizer: tf.keras.optimizers.Optimizer,
                           epochs: int = 100,
                           save_dir: str = None,
                           checkpoint_frequency: int = 1000,
                           seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Deterministic training with fixed schedule and consistent convergence criteria
    This replaces the existing multi-stage training with a completely deterministic approach
    that eliminates sources of randomness in the training process.
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    loss_history = []

    # FIXED training schedule - no randomness in phase transitions
    # These phases are completely deterministic and reproducible
    phase_configs = [
        {
            'name': 'Phase 1: Physics Learning',
            'epochs': epochs // 4,  # Always 25% of total epochs
            'weights': {'initial': 2.0, 'boundary': 2.0, 'interior': 0.5, 'physics': 10.0},
            'lr_schedule': lambda epoch, total: 1e-3 * (0.95 ** (epoch // 100)),
            'regularization': 0.001,
            'description': 'Focus on learning physics (PDE) constraints'
        },
        {
            'name': 'Phase 2: Data Fitting',
            'epochs': epochs // 2,  # Always 50% of total epochs
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 5.0, 'physics': 3.0},
            'lr_schedule': lambda epoch, total: 5e-4 * (0.98 ** (epoch // 50)),
            'regularization': 0.0005,
            'description': 'Balance physics with data fitting'
        },
        {
            'name': 'Phase 3: Fine Tuning',
            'epochs': epochs // 4,  # Always 25% of total epochs
            'weights': {'initial': 0.5, 'boundary': 0.5, 'interior': 10.0, 'physics': 1.0},
            'lr_schedule': lambda epoch, total: 1e-4 * (0.99 ** (epoch // 25)),
            'regularization': 0.0001,
            'description': 'Fine-tune with emphasis on data accuracy'
        }
    ]

    epoch_counter = 0
    convergence_history = []

    # Define acceptable range for diffusion coefficient (deterministic bounds)
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
                # DETERMINISTIC learning rate - same calculation every time
                current_lr = phase['lr_schedule'](epoch, phase['epochs'])

                # Set learning rate deterministically
                if hasattr(optimizer, 'learning_rate'):
                    if hasattr(optimizer.learning_rate, 'assign'):
                        optimizer.learning_rate.assign(current_lr)

                # Training step with DETERMINISTIC loss computation
                with tf.GradientTape() as tape:
                    # Compute losses with FIXED weights (no randomness)
                    losses = pinn.loss_fn(
                        x_data=data['X_u_train'],
                        c_data=data['u_train'],
                        x_physics=data['X_f_train'],
                        weights=phase['weights']  # Fixed weights per phase
                    )

                    # Compute interior loss separately for stability
                    interior_loss = compute_stable_interior_loss(
                        pinn, data['X_i_train'], data['u_i_train']
                    )

                    # Add DETERMINISTIC regularization
                    l2_loss = phase['regularization'] * sum(
                        tf.reduce_sum(tf.square(w)) for w in pinn.weights
                    )

                    # Diffusion coefficient regularization toward expected values
                    d_target = tf.constant(0.0001, dtype=tf.float32)  # Fixed target
                    d_reg = 0.01 * tf.square(
                        tf.math.log(pinn.D + 1e-8) - tf.math.log(d_target)
                    )

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

                # Apply gradients with CONSISTENT clipping
                trainable_vars = pinn.get_trainable_variables()
                gradients = tape.gradient(total_loss, trainable_vars)

                # DETERMINISTIC gradient clipping
                gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                # STRICT bounds enforcement on diffusion coefficient
                if pinn.config.diffusion_trainable:
                    D_value = pinn.D.numpy()
                    if D_value < D_min or D_value > D_max:
                        constrained_D = np.clip(D_value, D_min, D_max)
                        pinn.D.assign(constrained_D)

                # Record history
                current_D = pinn.get_diffusion_coefficient()
                D_history.append(current_D)
                loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

                # DETERMINISTIC convergence checking every 100 epochs
                if epoch_counter % 100 == 0:
                    # Check convergence over last 50 epochs
                    if len(D_history) >= 50:
                        recent_d = D_history[-50:]
                        d_std = np.std(recent_d)
                        d_mean = np.mean(recent_d)
                        relative_std = d_std / d_mean if d_mean > 0 else float('inf')
                        convergence_history.append(relative_std)
                        print(f"Epoch {epoch_counter:5d}: D={current_D:.6f}, "
                              f"Loss={losses['total']:.6f}, LR={current_lr:.2e}, "
                              f"RelStd={relative_std:.6f}")
                        # DETERMINISTIC early stopping criteria
                        if (relative_std < 0.001 and
                            len(convergence_history) >= 3 and
                            all(conv < 0.001 for conv in convergence_history[-3:])):
                            print(f"Converged at epoch {epoch_counter} (relative std < 0.001)")
                            # Don't break - let it complete the phase for consistency

                epoch_counter += 1

                # Memory cleanup every 100 epochs
                if epoch_counter % 100 == 0:
                    gc.collect()

            # Phase completion summary
            phase_final_D = D_history[-1]
            phase_final_loss = loss_history[-1]['total']
            print(f"\nPhase {phase_idx + 1} completed:")
            print(f"  Final D: {phase_final_D:.6f}")
            print(f"  Final Loss: {phase_final_loss:.6f}")
            print(f"  Epochs: {phase_start_epoch} to {epoch_counter-1}")

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

        # Test if network learned to predict varying values
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

        # Check final convergence
        if len(D_history) >= 100:
            recent_d = D_history[-100:]
            final_std = np.std(recent_d) / np.mean(recent_d)
            print(f"Final convergence metric: {final_std:.6f}")
            print(f"Converged: {'Yes' if final_std < 0.01 else 'No'}")

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
    Main training function - now uses deterministic training

    This is a wrapper that calls the deterministic training function.
    This maintains compatibility with existing code while providing
    the improved deterministic behavior.
    """
    print("Using deterministic training schedule...")
    return deterministic_train_pinn(
        pinn=pinn,
        data=data,
        optimizer=optimizer,
        epochs=epochs,
        save_dir=save_dir,
        checkpoint_frequency=checkpoint_frequency,
        seed=seed
    )

def save_checkpoint(pinn: 'DiffusionPINN', save_dir: str, epoch: str) -> None:
    """
    Save model checkpoint

    Args:
        pinn: PINN model to save
        save_dir: Directory to save checkpoint
        epoch: Epoch identifier (can be string like "phase_1_final")
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