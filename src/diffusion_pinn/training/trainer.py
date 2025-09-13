import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import gc
from ..variables import PINN_VARIABLES

def create_and_initialize_pinn(inputfile: str,
                             N_u: int = PINN_VARIABLES['N_u'],
                             N_f: int = PINN_VARIABLES['N_f'],
                             N_i: int = PINN_VARIABLES['N_i'],
                             initial_D: float = PINN_VARIABLES['initial_D'],
                             seed: int = None) -> Tuple['DiffusionPINN', Dict[str, tf.Tensor]]:
    """
    Create and initialize PINN with data - simplified version

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

    # Create PINN configuration
    config = DiffusionConfig(
        diffusion_trainable=True,
        use_physics_loss=True
    )

    # Initialize PINN with direct D optimization
    pinn = DiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=initial_D,
        config=config,
        seed=seed
    )

    # Prepare training data - use natural temporal sampling (no temporal_density)
    training_data = data_processor.prepare_training_data(N_u, N_f, N_i, seed=seed)

    return pinn, training_data

def train_pinn(pinn: 'DiffusionPINN',
              data: Dict[str, tf.Tensor],
              optimizer: tf.keras.optimizers.Optimizer,
              epochs: int = 100,
              progress_frequency: int = 1000,
              seed: int = None) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Simplified training function - single phase, direct D optimization

    Args:
        pinn: PINN model to train
        data: Training data dictionary
        optimizer: TensorFlow optimizer
        epochs: Number of training epochs
        progress_frequency: How often to print progress
        seed: Random seed (optional)

    Returns:
        Tuple of (D_history, loss_history)
    """
    # Set random seeds if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    loss_history = []

    print(f"Starting simplified training for {epochs} epochs")
    print(f"Initial D: {pinn.get_diffusion_coefficient():.2e}")

    try:
        for epoch in range(epochs):
            # Clean up memory periodically
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            # Training step
            with tf.GradientTape() as tape:
                # Compute all losses
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )

                # Add interior loss if we have interior training data
                if 'X_i_train' in data and 'u_i_train' in data:
                    interior_loss = tf.reduce_mean(tf.square(
                        pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                    ))
                    losses['interior'] = interior_loss

                    # Add interior loss to total with weight
                    total_loss = losses['total'] + pinn.loss_weights.get('interior', 1.0) * interior_loss
                    losses['total'] = total_loss

                total_loss = losses['total']

            # Calculate gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping for stability
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Apply gradients
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Apply simple constraints (keep D positive)
            pinn.apply_parameter_constraints()

            # Record history
            current_D = pinn.get_diffusion_coefficient()
            D_history.append(current_D)

            # Convert loss tensors to python floats
            loss_dict = {k: float(v.numpy()) for k, v in losses.items()}
            loss_history.append(loss_dict)

            # Progress reporting
            if epoch % progress_frequency == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: D={current_D:.2e}, Loss={total_loss:.2e}")

        # Final results
        final_D = pinn.get_diffusion_coefficient()
        print(f"Training completed - Final D: {final_D:.2e}")

        # Check for basic convergence
        if len(D_history) >= 100:
            recent_D = D_history[-100:]
            D_std = np.std(recent_D)
            D_mean = np.mean(recent_D)
            relative_std = D_std / D_mean if D_mean > 0 else float('inf')
            print(f"D convergence metric (last 100 epochs): {relative_std:.6f}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

    return D_history, loss_history