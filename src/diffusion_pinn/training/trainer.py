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

def debug_data_distribution(data: Dict[str, tf.Tensor]):
    """Debug function to check data distribution - NO UNICODE CHARACTERS"""
    print("\n=== DATA DISTRIBUTION DEBUG ===")

    x_u_train = data['X_u_train'].numpy()
    x_i_train = data['X_i_train'].numpy()
    x_f_train = data['X_f_train'].numpy()

    print(f"Boundary/Initial data shape: {x_u_train.shape}")
    print(f"Interior data shape: {x_i_train.shape}")
    print(f"Physics data shape: {x_f_train.shape}")

    # Check coordinate ranges
    print(f"\nBoundary/Initial data ranges:")
    print(f"  x: [{x_u_train[:, 0].min():.4f}, {x_u_train[:, 0].max():.4f}]")
    print(f"  y: [{x_u_train[:, 1].min():.4f}, {x_u_train[:, 1].max():.4f}]")
    print(f"  t: [{x_u_train[:, 2].min():.4f}, {x_u_train[:, 2].max():.4f}]")

    print(f"\nInterior data ranges:")
    print(f"  x: [{x_i_train[:, 0].min():.4f}, {x_i_train[:, 0].max():.4f}]")
    print(f"  y: [{x_i_train[:, 1].min():.4f}, {x_i_train[:, 1].max():.4f}]")
    print(f"  t: [{x_i_train[:, 2].min():.4f}, {x_i_train[:, 2].max():.4f}]")

    # Check for time = 0 points (initial conditions) - FIXED: NO UNICODE
    t_zero_boundary = np.sum(np.abs(x_u_train[:, 2]) < 1e-6)
    t_zero_interior = np.sum(np.abs(x_i_train[:, 2]) < 1e-6)

    print(f"\nInitial condition points (t=0):")
    print(f"  In boundary data: {t_zero_boundary}")
    print(f"  In interior data: {t_zero_interior}")

    # Check for boundary points (x=0, x=1, y=0, y=1)
    boundary_x = np.logical_or(
        np.abs(x_u_train[:, 0]) < 1e-6,
        np.abs(x_u_train[:, 0] - 1.0) < 1e-6
    )
    boundary_y = np.logical_or(
        np.abs(x_u_train[:, 1]) < 1e-6,
        np.abs(x_u_train[:, 1] - 1.0) < 1e-6
    )
    boundary_points = np.logical_or(boundary_x, boundary_y)

    print(f"\nSpatial boundary points:")
    print(f"  In boundary data: {np.sum(boundary_points)}")

    # Check time distribution
    print(f"\nTime distribution in boundary data:")
    unique_times = np.unique(x_u_train[:, 2])
    for t_val in unique_times:
        count = np.sum(np.abs(x_u_train[:, 2] - t_val) < 1e-6)
        print(f"  t={t_val:.3f}: {count} points")

    # ADD THIS: TARGETED DIAGNOSTIC FOR THE REAL ISSUE
    print(f"\nTARGETED DIAGNOSTIC - Exact t=0 check:")
    t_vals = x_u_train[:, 2]

    # Check for exactly 0.0 (no tolerance)
    exactly_zero = np.sum(t_vals == 0.0)
    print(f"  Points with exactly t=0.0: {exactly_zero}")

    # Check for very close to zero with different tolerances
    print(f"  Points within different tolerances:")
    for tol in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        count = np.sum(np.abs(t_vals) < tol)
        print(f"    |t| < {tol:.0e}: {count} points")

    # Show actual values of the "zero" points
    zero_positions = np.where(np.abs(t_vals) < 1e-3)[0]
    if len(zero_positions) > 0:
        zero_values = t_vals[zero_positions]
        print(f"  Actual 'zero' values (first 10): {zero_values[:10]}")
        print(f"  Min 'zero' value: {zero_values.min():.12f}")
        print(f"  Max 'zero' value: {zero_values.max():.12f}")
        print(f"  All exactly 0.0: {np.all(zero_values == 0.0)}")
    else:
        print(f"  ERROR: No values found near t=0!")

    print("=== END DEBUG ===\n")

def compute_stable_interior_loss(pinn, x_interior, c_interior):
    """Compute interior loss with stability enhancements"""
    if x_interior.shape[0] == 0:
        print("Warning: No interior points provided")
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
    loss_value = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)

    return loss_value

def fixed_train_pinn_with_debugging(pinn: 'DiffusionPINN',
                                  data: Dict[str, tf.Tensor],
                                  optimizer: tf.keras.optimizers.Optimizer,
                                  epochs: int = 100,
                                  save_dir: str = None,
                                  checkpoint_frequency: int = 1000,
                                  seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    FIXED training with enhanced debugging and corrected loss computation
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Debug data distribution
    debug_data_distribution(data)

    D_history = []
    log_D_history = []
    loss_history = []

    # FIXED TRAINING SCHEDULE - prioritize data learning first
    phase_configs = [
        {
            'name': 'Phase 1: Data Learning Priority',
            'epochs': epochs // 3,
            'weights': {'initial': 10.0, 'boundary': 10.0, 'interior': 20.0, 'physics': 0.5},
            'lr': 5e-4,
            'description': 'Focus on fitting data first, minimal physics'
        },
        {
            'name': 'Phase 2: Gradual Physics Integration',
            'epochs': epochs // 3,
            'weights': {'initial': 5.0, 'boundary': 5.0, 'interior': 10.0, 'physics': 3.0},
            'lr': 2e-4,
            'description': 'Gradually introduce physics constraints'
        },
        {
            'name': 'Phase 3: Balanced Optimization',
            'epochs': epochs // 3,
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 5.0, 'physics': 5.0},
            'lr': 1e-4,
            'description': 'Balance all loss components'
        }
    ]

    epoch_counter = 0
    convergence_history = []

    print(f"\nStarting FIXED training with debugging for {epochs} epochs")
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
            print(f"Learning rate: {phase['lr']}")
            print(f"{'='*60}")

            # Set learning rate for this phase
            optimizer.learning_rate.assign(phase['lr'])

            phase_start_epoch = epoch_counter

            for epoch in range(phase['epochs']):
                with tf.GradientTape() as tape:
                    # Extract coordinates for proper classification
                    t = data['X_u_train'][:, 2]
                    x_coord = data['X_u_train'][:, 0]
                    y_coord = data['X_u_train'][:, 1]

                    # FIXED: Use EXACT equality instead of tolerance-based comparison
                    ic_mask = tf.equal(t, tf.cast(pinn.t_bounds[0], tf.float32))

                    # Boundary masks with exact equality
                    bc_x_mask = tf.logical_or(
                        tf.equal(x_coord, tf.cast(pinn.x_bounds[0], tf.float32)),
                        tf.equal(x_coord, tf.cast(pinn.x_bounds[1], tf.float32))
                    )
                    bc_y_mask = tf.logical_or(
                        tf.equal(y_coord, tf.cast(pinn.y_bounds[0], tf.float32)),
                        tf.equal(y_coord, tf.cast(pinn.y_bounds[1], tf.float32))
                    )
                    bc_mask = tf.logical_or(bc_x_mask, bc_y_mask)
                    bc_mask = tf.logical_and(bc_mask, tf.logical_not(ic_mask))

                    # Debug mask counts on first epoch
                    if epoch_counter == 0:
                        ic_count = tf.reduce_sum(tf.cast(ic_mask, tf.int32))
                        bc_count = tf.reduce_sum(tf.cast(bc_mask, tf.int32))
                        total_points = t.shape[0]
                        print(f"Mask validation - IC: {ic_count}, BC: {bc_count}, Total: {total_points}")

                        if ic_count == 0:
                            print("CRITICAL ERROR: No initial condition points found!")
                            print("This will prevent proper training. Check data preprocessing.")
                        if bc_count == 0:
                            print("CRITICAL ERROR: No boundary condition points found!")

                    # Initialize losses dictionary
                    losses = {}

                    # Initial condition loss
                    if tf.reduce_any(ic_mask):
                        c_pred_ic = pinn.forward_pass(tf.boolean_mask(data['X_u_train'], ic_mask))
                        c_true_ic = tf.boolean_mask(data['u_train'], ic_mask)
                        losses['initial'] = tf.reduce_mean(tf.square(c_pred_ic - c_true_ic))
                    else:
                        losses['initial'] = tf.constant(0.0, dtype=tf.float32)
                        if epoch_counter % 100 == 0:
                            print("WARNING: No initial condition points found!")

                    # Boundary condition loss
                    if tf.reduce_any(bc_mask):
                        c_pred_bc = pinn.forward_pass(tf.boolean_mask(data['X_u_train'], bc_mask))
                        c_true_bc = tf.boolean_mask(data['u_train'], bc_mask)
                        losses['boundary'] = tf.reduce_mean(tf.square(c_pred_bc - c_true_bc))
                    else:
                        losses['boundary'] = tf.constant(0.0, dtype=tf.float32)
                        if epoch_counter % 100 == 0:
                            print("WARNING: No boundary condition points found!")

                    # Interior data loss
                    if 'X_i_train' in data and data['X_i_train'].shape[0] > 0:
                        interior_loss = compute_stable_interior_loss(
                            pinn, data['X_i_train'], data['u_i_train']
                        )
                        losses['interior'] = interior_loss
                    else:
                        losses['interior'] = tf.constant(0.0, dtype=tf.float32)

                    # FIXED Physics loss - ALWAYS compute if we have physics points
                    if data['X_f_train'].shape[0] > 0:
                        try:
                            pde_residual = pinn.compute_pde_residual(data['X_f_train'])

                            # Enhanced physics loss computation with debugging
                            residual_mean = tf.reduce_mean(tf.abs(pde_residual))

                            if epoch_counter % 100 == 0:
                                print(f"Physics residual mean: {residual_mean.numpy():.8e}")

                            if tf.math.is_finite(residual_mean) and residual_mean > 1e-15:
                                # Stabilized physics loss computation
                                delta = tf.maximum(0.01, residual_mean)
                                abs_residual = tf.abs(pde_residual)

                                # Prevent extreme residuals from dominating
                                abs_residual_capped = tf.minimum(abs_residual, 100.0 * delta)
                                quadratic = tf.minimum(abs_residual_capped, delta)
                                linear = abs_residual_capped - quadratic

                                physics_loss = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)
                                losses['physics'] = physics_loss

                                if epoch_counter % 100 == 0:
                                    print(f"Computed physics loss: {physics_loss.numpy():.8e}")
                            else:
                                if epoch_counter % 100 == 0:
                                    print("WARNING: Physics residual is zero, NaN, or too small!")
                                # Force a small non-zero physics loss to encourage learning
                                losses['physics'] = tf.constant(0.01, dtype=tf.float32)

                        except Exception as e:
                            if epoch_counter % 100 == 0:
                                print(f"ERROR in physics loss computation: {str(e)}")
                            # Use fallback physics loss
                            losses['physics'] = tf.constant(0.1, dtype=tf.float32)
                    else:
                        losses['physics'] = tf.constant(0.0, dtype=tf.float32)

                    # L2 regularization on network weights only (not diffusion coefficient)
                    l2_loss = 1e-6 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                    # MINIMAL diffusion coefficient regularization
                    log_d_reg_losses = pinn.compute_minimal_log_d_regularization()
                    log_d_reg_total = sum(log_d_reg_losses.values())

                    # Total loss with phase-specific weights
                    total_loss = (
                        losses['initial'] * phase['weights']['initial'] +
                        losses['boundary'] * phase['weights']['boundary'] +
                        losses['interior'] * phase['weights']['interior'] +
                        losses['physics'] * phase['weights']['physics'] +
                        l2_loss +
                        log_d_reg_total
                    )

                    losses['total'] = total_loss
                    losses['l2_reg'] = l2_loss
                    losses['log_d_reg'] = log_d_reg_total

                # ENHANCED GRADIENT COMPUTATION AND DEBUGGING
                trainable_vars = pinn.get_trainable_variables()
                gradients = tape.gradient(total_loss, trainable_vars)

                # Check for None gradients
                if any(g is None for g in gradients):
                    print(f"WARNING: Some gradients are None at epoch {epoch_counter}")
                    gradients = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, trainable_vars)]

                # DEBUG: Check gradients for diffusion coefficient (should be last variable)
                if epoch_counter % 100 == 0 and len(gradients) > 0:
                    # Check if last gradient corresponds to log_D
                    if pinn.config.diffusion_trainable and len(gradients) == len(pinn.weights) + len(pinn.biases) + 1:
                        log_d_grad = gradients[-1]
                        if log_d_grad is not None:
                            grad_magnitude = tf.reduce_mean(tf.abs(log_d_grad)).numpy()
                            print(f"log_D gradient magnitude: {grad_magnitude:.8e}")
                            if grad_magnitude < 1e-15:
                                print("WARNING: log_D gradient is essentially zero!")
                        else:
                            print("ERROR: log_D gradient is None!")
                    else:
                        print("WARNING: Unexpected number of gradients or diffusion not trainable")

                # Gradient clipping
                gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

                if epoch_counter % 100 == 0:
                    print(f"Gradient global norm: {global_norm.numpy():.8e}")

                # Apply gradients
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Record history
                current_D = pinn.get_diffusion_coefficient()
                current_log_D = pinn.get_log_diffusion_coefficient()

                D_history.append(current_D)
                log_D_history.append(current_log_D)
                loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

                # ENHANCED MONITORING - Report every 50 epochs
                if epoch_counter % 50 == 0:
                    ic_count = tf.reduce_sum(tf.cast(ic_mask, tf.int32))
                    bc_count = tf.reduce_sum(tf.cast(bc_mask, tf.int32))

                    print(f"\nEpoch {epoch_counter:5d}: D={current_D:.8e}, log(D)={current_log_D:.6f}")
                    print(f"  Total Loss={losses['total']:.8f}, LR={optimizer.learning_rate.numpy():.2e}")
                    print(f"  Initial={losses['initial']:.8f} (IC points: {ic_count})")
                    print(f"  Boundary={losses['boundary']:.8f} (BC points: {bc_count})")
                    print(f"  Interior={losses['interior']:.8f}")
                    print(f"  Physics={losses['physics']:.8f}")

                    # Check if D is actually changing
                    if len(D_history) > 10:
                        recent_d_change = abs(D_history[-1] - D_history[-10])
                        print(f"  D change (last 10 epochs): {recent_d_change:.2e}")
                        if recent_d_change < 1e-12:
                            print("  WARNING: D is not changing! Check gradient flow.")

                if epoch_counter % 200 == 0:
                    # Test prediction variability
                    test_points = tf.constant([
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 2.5],
                        [1.0, 1.0, 5.0]
                    ], dtype=tf.float32)

                    test_preds = pinn.predict(test_points)
                    pred_range = tf.reduce_max(test_preds) - tf.reduce_min(test_preds)
                    print(f"  Prediction range test: {pred_range.numpy():.8f}")

                    if pred_range < 1e-6:
                        print("  WARNING: Network still predicting uniform values!")
                    else:
                        print("  GOOD: Network predicting varying values")

                    # Check convergence over last 100 epochs using log(D)
                    if len(log_D_history) >= 100:
                        recent_log_d = log_D_history[-100:]
                        log_d_std = np.std(recent_log_d)
                        log_d_mean = np.mean(recent_log_d)

                        convergence_metric = log_d_std
                        convergence_history.append(convergence_metric)

                        print(f"  log(D) convergence metric (last 100 epochs): {convergence_metric:.6f}")

                        # Convergence criteria
                        if (convergence_metric < 0.01 and
                            len(convergence_history) >= 3 and
                            all(conv < 0.01 for conv in convergence_history[-3:])):
                            print(f"  CONVERGENCE ACHIEVED: log(D) has stabilized")

                epoch_counter += 1

                # Memory cleanup
                if epoch_counter % 100 == 0:
                    gc.collect()

            # Phase completion summary
            phase_final_D = D_history[-1]
            phase_final_log_D = log_D_history[-1]
            phase_final_loss = loss_history[-1]['total']
            print(f"\nPhase {phase_idx + 1} completed:")
            print(f"  Final D: {phase_final_D:.8e}")
            print(f"  Final log(D): {phase_final_log_D:.6f}")
            print(f"  Final Loss: {phase_final_loss:.8f}")
            print(f"  Epochs: {phase_start_epoch} to {epoch_counter-1}")

        # Final training summary
        final_D = D_history[-1]
        final_log_D = log_D_history[-1]
        final_loss = loss_history[-1]['total']

        print(f"\n{'='*60}")
        print("FIXED TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total epochs: {epoch_counter}")
        print(f"Final diffusion coefficient: {final_D:.8e}")
        print(f"Final log(D): {final_log_D:.6f}")
        print(f"Final loss: {final_loss:.8f}")

        # Check for D value changes
        if len(D_history) > 100:
            initial_d = D_history[0]
            final_d = D_history[-1]
            total_change = abs(final_d - initial_d)
            relative_change = total_change / initial_d if initial_d > 0 else 0

            print(f"D value evolution:")
            print(f"  Initial: {initial_d:.8e}")
            print(f"  Final: {final_d:.8e}")
            print(f"  Absolute change: {total_change:.8e}")
            print(f"  Relative change: {relative_change:.2%}")

            if relative_change < 0.01:
                print("  WARNING: D changed by less than 1% - poor convergence!")
            else:
                print("  GOOD: D showed significant learning")

        # Final convergence analysis
        if len(log_D_history) >= 100:
            recent_log_d = log_D_history[-100:]
            final_log_d_std = np.std(recent_log_d)
            final_log_d_mean = np.mean(recent_log_d)

            print(f"Final convergence analysis:")
            print(f"  log(D) std (last 100 epochs): {final_log_d_std:.8f}")
            converged = final_log_d_std < 0.05
            print(f"  Converged: {'YES' if converged else 'NO'}")

        return D_history, loss_history

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        return D_history, loss_history
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
    Main training function - uses FIXED training with enhanced debugging
    """
    print("Using FIXED training with enhanced debugging...")
    return fixed_train_pinn_with_debugging(
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