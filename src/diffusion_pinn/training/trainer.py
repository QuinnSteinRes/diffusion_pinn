import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import gc
from ..variables import PINN_VARIABLES

def create_and_initialize_pinn(inputfile: str,
                             N_boundary: int,
                             N_interior: int,
                             N_collocation: int,
                             temporal_density: int = 5,
                             initial_D: float = PINN_VARIABLES['initial_D'],
                             seed: int = None) -> Tuple['DiffusionPINN', Dict]:
    """
    Create and initialize PINN with labeled training data

    Args:
        inputfile: Path to data file
        N_boundary: Number of boundary/initial condition points
        N_interior: Number of interior supervision points
        N_collocation: Number of physics collocation points
        temporal_density: Temporal density multiplier for physics points
        initial_D: Initial guess for diffusion coefficient
        seed: Random seed for reproducibility

    Returns:
        Tuple of (initialized PINN, labeled training data)
    """
    from ..data.processor import DiffusionDataProcessor
    from ..models.pinn import DiffusionPINN
    from ..config import DiffusionConfig

    print(f"Initializing PINN with labeled point sets...")
    print(f"  Boundary points: {N_boundary}")
    print(f"  Interior points: {N_interior}")
    print(f"  Collocation points: {N_collocation}")
    print(f"  Temporal density: {temporal_density}")
    print(f"  Initial D: {initial_D}")
    print(f"  Seed: {seed}")

    # Set random seeds
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Process data with seed for reproducibility
    data_processor = DiffusionDataProcessor(inputfile, seed=seed)
    data_processor.print_data_summary()

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
        seed=seed
    )

    # Prepare labeled training data - clean interface!
    labeled_data = data_processor.prepare_labeled_training_data(
        N_boundary=N_boundary,
        N_interior=N_interior,
        N_collocation=N_collocation,
        temporal_density=temporal_density
    )

    print(f"PINN initialization complete!")
    pinn.print_diagnostics()

    return pinn, labeled_data

def train_pinn(pinn: 'DiffusionPINN',
              labeled_data: Dict,
              optimizer: tf.keras.optimizers.Optimizer,
              epochs: int = PINN_VARIABLES['epochs'],
              progress_frequency: int = 10,
              save_dir: str = None,
              seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Train PINN using labeled point sets - clean and simple interface

    Args:
        pinn: PINN model to train
        labeled_data: Labeled training data from processor containing:
            'boundary': (coords, values) - boundary/initial conditions
            'interior': (coords, values) - interior supervision points
            'physics': coords - physics collocation points
        optimizer: TensorFlow optimizer
        epochs: Number of training epochs
        progress_frequency: How often to print progress
        save_dir: Directory to save intermediate models (optional)
        seed: Random seed (optional)

    Returns:
        Tuple of (D_history, loss_history)
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Initialize tracking
    D_history = []
    loss_history = []

    print(f"\n" + "="*60)
    print(f"TRAINING PINN WITH LABELED POINT SETS")
    print(f"="*60)
    print(f"Epochs: {epochs}")
    print(f"Initial D: {pinn.get_diffusion_coefficient():.2e}")
    print(f"Optimizer: {optimizer.__class__.__name__}")

    # Print data summary
    boundary_coords, boundary_values = labeled_data['boundary']
    interior_coords, interior_values = labeled_data['interior']
    physics_coords = labeled_data['physics']

    print(f"Training data:")
    print(f"  Boundary points: {boundary_coords.shape[0]} (with values)")
    print(f"  Interior points: {interior_coords.shape[0]} (with values)")
    print(f"  Physics points: {physics_coords.shape[0]} (collocation)")
    print(f"="*60 + "\n")

    try:
        for epoch in range(epochs):
            # Periodic memory cleanup
            if epoch % 100 == 0:
                tf.keras.backend.clear_session()
                gc.collect()

            # Training step with labeled data
            with tf.GradientTape() as tape:
                # Compute losses using labeled point sets - much cleaner!
                losses = pinn.loss_fn(labeled_data)
                total_loss = losses['total']

            # Calculate gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping for stability
            if gradients is not None:
                gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

                # Check for NaN gradients
                if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None]):
                    print(f"Warning: NaN gradients detected at epoch {epoch}")
                    continue

                # Apply gradients
                optimizer.apply_gradients(zip(gradients, trainable_vars))
            else:
                print(f"Warning: No gradients computed at epoch {epoch}")
                continue

            # Apply parameter constraints
            pinn.apply_parameter_constraints()

            # Record history
            current_D = pinn.get_diffusion_coefficient()
            D_history.append(current_D)

            # Convert loss tensors to python floats
            loss_dict = {k: float(v.numpy()) for k, v in losses.items()}
            loss_history.append(loss_dict)

            # Progress reporting with clear loss breakdown
            if epoch % progress_frequency == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:6d}/{epochs}: D={current_D:.2e}, "
                      f"Total={total_loss:.2e} "
                      f"(B:{losses['boundary']:.2e}, "
                      f"I:{losses['interior']:.2e}, "
                      f"P:{losses['physics']:.2e})")

        # Training completion
        final_D = pinn.get_diffusion_coefficient()
        print(f"\n" + "="*60)
        print(f"TRAINING COMPLETED")
        print(f"="*60)
        print(f"Final D: {final_D:.2e}")
        print(f"Total epochs: {len(D_history)}")

        # Basic convergence analysis
        if len(D_history) >= 100:
            recent_D = D_history[-100:]
            D_std = np.std(recent_D)
            D_mean = np.mean(recent_D)
            relative_variation = D_std / D_mean if D_mean > 0 else float('inf')
            print(f"Convergence (last 100 epochs): relative variation = {relative_variation:.6f}")

            # Simple convergence check
            converged = relative_variation < 0.001
            print(f"Converged: {converged} (threshold: 0.001)")

        print(f"="*60 + "\n")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Completed {len(D_history)} epochs")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"Completed {len(D_history)} epochs before error")

    finally:
        # Final cleanup
        gc.collect()

    return D_history, loss_history

def evaluate_pinn(pinn: 'DiffusionPINN',
                 labeled_data: Dict,
                 save_results: bool = True,
                 save_dir: str = 'results') -> Dict[str, float]:
    """
    Evaluate trained PINN performance on test data

    Args:
        pinn: Trained PINN model
        labeled_data: Labeled data including test set
        save_results: Whether to save evaluation results
        save_dir: Directory to save results

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating PINN performance...")

    # Extract test data
    test_coords, test_values = labeled_data['test']
    test_coords = tf.convert_to_tensor(test_coords, dtype=tf.float32)
    test_values = tf.convert_to_tensor(test_values, dtype=tf.float32)

    # Get predictions
    predictions = pinn.predict(test_coords)

    # Calculate metrics
    mse = tf.reduce_mean(tf.square(predictions - test_values))
    mae = tf.reduce_mean(tf.abs(predictions - test_values))

    # Calculate R²
    ss_res = tf.reduce_sum(tf.square(test_values - predictions))
    ss_tot = tf.reduce_sum(tf.square(test_values - tf.reduce_mean(test_values)))
    r2 = 1 - ss_res / ss_tot

    # Relative error
    relative_error = tf.reduce_mean(tf.abs((predictions - test_values) / test_values))

    metrics = {
        'mse': float(mse.numpy()),
        'mae': float(mae.numpy()),
        'r2': float(r2.numpy()),
        'relative_error': float(relative_error.numpy()),
        'diffusion_coefficient': pinn.get_diffusion_coefficient()
    }

    print(f"Evaluation Results:")
    print(f"  MSE: {metrics['mse']:.2e}")
    print(f"  MAE: {metrics['mae']:.2e}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Relative Error: {metrics['relative_error']:.4f}")
    print(f"  Final D: {metrics['diffusion_coefficient']:.2e}")

    # Save results if requested
    if save_results:
        import os
        import json

        os.makedirs(save_dir, exist_ok=True)

        # Save metrics as JSON
        with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save predictions vs actual for plotting
        np.savez(
            os.path.join(save_dir, 'predictions.npz'),
            coordinates=test_coords.numpy(),
            true_values=test_values.numpy(),
            predictions=predictions.numpy()
        )

        print(f"Results saved to {save_dir}")

    return metrics

def create_optimizer(learning_rate: float = PINN_VARIABLES['learning_rate'],
                    decay_steps: int = PINN_VARIABLES['decay_steps'],
                    decay_rate: float = PINN_VARIABLES['decay_rate']) -> tf.keras.optimizers.Optimizer:
    """
    Create optimizer with learning rate schedule

    Args:
        learning_rate: Initial learning rate
        decay_steps: Steps for learning rate decay
        decay_rate: Decay rate

    Returns:
        Configured optimizer
    """
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    # Create Adam optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    print(f"Created optimizer: Adam with exponential decay")
    print(f"  Initial LR: {learning_rate}")
    print(f"  Decay steps: {decay_steps}")
    print(f"  Decay rate: {decay_rate}")

    return optimizer

# Convenience function for complete training workflow
def train_diffusion_pinn(inputfile: str,
                        epochs: int = PINN_VARIABLES['epochs'],
                        N_boundary: int = PINN_VARIABLES['N_u'],
                        N_interior: int = PINN_VARIABLES['N_i'],
                        N_collocation: int = PINN_VARIABLES['N_f'],
                        temporal_density: int = 5,
                        initial_D: float = PINN_VARIABLES['initial_D'],
                        learning_rate: float = PINN_VARIABLES['learning_rate'],
                        save_dir: str = 'results',
                        seed: int = None) -> Tuple[List[float], List[Dict[str, float]], Dict[str, float]]:
    """
    Complete training workflow - one function call for full PINN training

    Args:
        inputfile: Path to data file
        epochs: Number of training epochs
        N_boundary: Number of boundary points
        N_interior: Number of interior points
        N_collocation: Number of collocation points
        temporal_density: Temporal density for physics points
        initial_D: Initial diffusion coefficient guess
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save results
        seed: Random seed for reproducibility

    Returns:
        Tuple of (D_history, loss_history, evaluation_metrics)
    """
    print(f"\n" + "="*80)
    print(f"COMPLETE DIFFUSION PINN TRAINING WORKFLOW")
    print(f"="*80)
    print(f"Input file: {inputfile}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {seed}")
    print(f"Save directory: {save_dir}")
    print(f"="*80)

    try:
        # Step 1: Create and initialize PINN with labeled data
        print(f"\n>>> STEP 1: Creating PINN and preparing labeled training data...")
        pinn, labeled_data = create_and_initialize_pinn(
            inputfile=inputfile,
            N_boundary=N_boundary,
            N_interior=N_interior,
            N_collocation=N_collocation,
            temporal_density=temporal_density,
            initial_D=initial_D,
            seed=seed
        )

        # Step 2: Create optimizer
        print(f"\n>>> STEP 2: Creating optimizer...")
        optimizer = create_optimizer(learning_rate=learning_rate)

        # Step 3: Train the PINN
        print(f"\n>>> STEP 3: Training PINN...")
        D_history, loss_history = train_pinn(
            pinn=pinn,
            labeled_data=labeled_data,
            optimizer=optimizer,
            epochs=epochs,
            save_dir=save_dir,
            seed=seed
        )

        # Step 4: Evaluate performance
        print(f"\n>>> STEP 4: Evaluating performance...")
        evaluation_metrics = evaluate_pinn(
            pinn=pinn,
            labeled_data=labeled_data,
            save_results=True,
            save_dir=save_dir
        )

        # Step 5: Save final model
        print(f"\n>>> STEP 5: Saving final model...")
        import os
        os.makedirs(save_dir, exist_ok=True)
        final_model_path = os.path.join(save_dir, 'final_pinn_model.npz')
        pinn.save_model(final_model_path)

        print(f"\n" + "="*80)
        print(f"TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
        print(f"="*80)
        print(f"Final diffusion coefficient: {evaluation_metrics['diffusion_coefficient']:.2e}")
        print(f"Final R²: {evaluation_metrics['r2']:.4f}")
        print(f"Results saved to: {save_dir}")
        print(f"="*80 + "\n")

        return D_history, loss_history, evaluation_metrics

    except Exception as e:
        print(f"\n" + "="*80)
        print(f"ERROR IN TRAINING WORKFLOW")
        print(f"="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"="*80 + "\n")
        raise

def check_convergence(D_history: List[float],
                     loss_history: List[Dict[str, float]],
                     window: int = 100,
                     D_threshold: float = 0.001,
                     loss_threshold: float = 0.01) -> Dict[str, bool]:
    """
    Check convergence status of PINN training

    Args:
        D_history: History of diffusion coefficient values
        loss_history: History of loss values
        window: Window size for convergence analysis
        D_threshold: Relative variation threshold for D convergence
        loss_threshold: Relative variation threshold for loss convergence

    Returns:
        Dictionary with convergence status for different metrics
    """
    convergence_status = {
        'D_converged': False,
        'loss_converged': False,
        'overall_converged': False
    }

    if len(D_history) < window or len(loss_history) < window:
        print(f"Insufficient data for convergence analysis (need {window} points)")
        return convergence_status

    # Check D convergence
    recent_D = D_history[-window:]
    D_mean = np.mean(recent_D)
    D_std = np.std(recent_D)
    D_relative_var = D_std / D_mean if D_mean > 0 else float('inf')

    convergence_status['D_converged'] = D_relative_var < D_threshold

    # Check loss convergence
    recent_losses = [loss['total'] for loss in loss_history[-window:]]
    loss_mean = np.mean(recent_losses)
    loss_std = np.std(recent_losses)
    loss_relative_var = loss_std / loss_mean if loss_mean > 0 else float('inf')

    convergence_status['loss_converged'] = loss_relative_var < loss_threshold

    # Overall convergence
    convergence_status['overall_converged'] = (
        convergence_status['D_converged'] and
        convergence_status['loss_converged']
    )

    # Print convergence report
    print(f"\nConvergence Analysis (last {window} epochs):")
    print(f"  D coefficient:")
    print(f"    Mean: {D_mean:.2e}")
    print(f"    Std: {D_std:.2e}")
    print(f"    Relative variation: {D_relative_var:.6f} (threshold: {D_threshold})")
    print(f"    Converged: {convergence_status['D_converged']}")

    print(f"  Total loss:")
    print(f"    Mean: {loss_mean:.2e}")
    print(f"    Std: {loss_std:.2e}")
    print(f"    Relative variation: {loss_relative_var:.6f} (threshold: {loss_threshold})")
    print(f"    Converged: {convergence_status['loss_converged']}")

    print(f"  Overall converged: {convergence_status['overall_converged']}")

    return convergence_status

def save_training_history(D_history: List[float],
                         loss_history: List[Dict[str, float]],
                         save_dir: str = 'results'):
    """
    Save training history to files for analysis

    Args:
        D_history: Diffusion coefficient history
        loss_history: Loss history
        save_dir: Directory to save files
    """
    import os
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # Save D history
    d_df = pd.DataFrame({
        'epoch': range(len(D_history)),
        'diffusion_coefficient': D_history
    })
    d_df.to_csv(os.path.join(save_dir, 'd_history.csv'), index=False)

    # Save loss history
    loss_df = pd.DataFrame(loss_history)
    loss_df['epoch'] = range(len(loss_history))

    # Reorder columns to put epoch first
    cols = ['epoch'] + [col for col in loss_df.columns if col != 'epoch']
    loss_df = loss_df[cols]

    loss_df.to_csv(os.path.join(save_dir, 'loss_history.csv'), index=False)

    print(f"Training history saved to {save_dir}")
    print(f"  D history: d_history.csv ({len(D_history)} epochs)")
    print(f"  Loss history: loss_history.csv ({len(loss_history)} epochs)")

# Helper function for debugging training issues
def diagnose_training_issues(pinn: 'DiffusionPINN',
                           labeled_data: Dict,
                           sample_size: int = 100) -> Dict[str, float]:
    """
    Diagnose potential training issues by analyzing gradients and losses

    Args:
        pinn: PINN model
        labeled_data: Labeled training data
        sample_size: Number of points to sample for diagnosis

    Returns:
        Dictionary with diagnostic information
    """
    print(f"Diagnosing potential training issues...")

    # Sample a subset of data for quick diagnosis
    boundary_coords, boundary_values = labeled_data['boundary']
    interior_coords, interior_values = labeled_data['interior']
    physics_coords = labeled_data['physics']

    # Sample smaller sets
    if boundary_coords.shape[0] > sample_size:
        indices = np.random.choice(boundary_coords.shape[0], sample_size, replace=False)
        boundary_coords = boundary_coords[indices]
        boundary_values = boundary_values[indices]

    if interior_coords.shape[0] > sample_size:
        indices = np.random.choice(interior_coords.shape[0], sample_size, replace=False)
        interior_coords = interior_coords[indices]
        interior_values = interior_values[indices]

    if physics_coords.shape[0] > sample_size:
        indices = np.random.choice(physics_coords.shape[0], sample_size, replace=False)
        physics_coords = physics_coords[indices]

    # Create sampled labeled data
    sampled_data = {
        'boundary': (boundary_coords, boundary_values),
        'interior': (interior_coords, interior_values),
        'physics': physics_coords
    }

    # Compute losses
    losses = pinn.loss_fn(sampled_data)

    # Check gradients
    with tf.GradientTape() as tape:
        total_loss = losses['total']

    trainable_vars = pinn.get_trainable_variables()
    gradients = tape.gradient(total_loss, trainable_vars)

    # Analyze gradients
    grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients]

    diagnostics = {
        'boundary_loss': float(losses['boundary'].numpy()),
        'interior_loss': float(losses['interior'].numpy()),
        'physics_loss': float(losses['physics'].numpy()),
        'total_loss': float(losses['total'].numpy()),
        'max_gradient_norm': max(grad_norms),
        'min_gradient_norm': min(grad_norms),
        'mean_gradient_norm': np.mean(grad_norms),
        'diffusion_coefficient': pinn.get_diffusion_coefficient(),
        'num_zero_gradients': sum(1 for g in grad_norms if g == 0.0)
    }

    print(f"Diagnostic Results:")
    print(f"  Losses:")
    print(f"    Boundary: {diagnostics['boundary_loss']:.2e}")
    print(f"    Interior: {diagnostics['interior_loss']:.2e}")
    print(f"    Physics: {diagnostics['physics_loss']:.2e}")
    print(f"    Total: {diagnostics['total_loss']:.2e}")
    print(f"  Gradients:")
    print(f"    Max norm: {diagnostics['max_gradient_norm']:.2e}")
    print(f"    Min norm: {diagnostics['min_gradient_norm']:.2e}")
    print(f"    Mean norm: {diagnostics['mean_gradient_norm']:.2e}")
    print(f"    Zero gradients: {diagnostics['num_zero_gradients']}")
    print(f"  Current D: {diagnostics['diffusion_coefficient']:.2e}")

    # Warning checks
    if diagnostics['max_gradient_norm'] > 1e3:
        print(f"  ⚠️  WARNING: Very large gradients detected - may need gradient clipping")

    if diagnostics['num_zero_gradients'] > len(grad_norms) // 2:
        print(f"  ⚠️  WARNING: Many zero gradients - check network connectivity")

    if diagnostics['physics_loss'] > 100 * diagnostics['boundary_loss']:
        print(f"  ⚠️  WARNING: Physics loss much larger than data loss - check loss weights")

    return diagnostics