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
    """Training function with adaptive learning, constraints, and multi-stage approach"""
    D_history = []
    loss_history = []

    # Define acceptable range for diffusion coefficient
    D_min = 5e-7  # Lower minimum to allow more exploration
    D_max = 2e-3  # Slightly higher maximum

    # Create learning rate schedule with warm-up and decay
    initial_learning_rate = optimizer.learning_rate.initial_learning_rate if hasattr(optimizer.learning_rate, 'initial_learning_rate') else 0.001

    def lr_schedule(epoch):
        warmup_epochs = min(200, epochs // 20)
        if epoch < warmup_epochs:
            # Linear warm-up
            return initial_learning_rate * (epoch + 1) / warmup_epochs
        else:
            # Cosine decay with restarts
            decay_epochs = epochs - warmup_epochs
            epoch_in_cycle = (epoch - warmup_epochs) % (decay_epochs // 3)
            cycle_progress = epoch_in_cycle / (decay_epochs // 3)
            return 0.1 * initial_learning_rate + 0.9 * initial_learning_rate * 0.5 * (1 + np.cos(np.pi * cycle_progress))

    # Multi-stage training approach
    try:
        # Phase 1: PDE physics emphasis
        print("Phase 1: Physics-based initialization...")
        phase1_epochs = min(epochs // 5, 1000)

        # Custom loss weights for phase 1
        phase1_weights = {
            'initial': 2.0,      # Emphasize initial conditions
            'boundary': 2.0,     # Emphasize boundary conditions
            'interior': 1.0,     # Less weight on interior data points initially
            'physics': 10.0      # Very high weight on physics (PDE)
        }

        for epoch in range(phase1_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            # Apply learning rate schedule
            if hasattr(optimizer, 'learning_rate'):
                optimizer.learning_rate = lr_schedule(epoch)

            print(f"\rPhase 1: {epoch+1}/{phase1_epochs}, LR: {optimizer.learning_rate.numpy():.2e}", end="", flush=True)

            with tf.GradientTape() as tape:
                # Compute losses with phase 1 weights
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=phase1_weights
                )

                interior_loss = tf.reduce_mean(tf.square(
                    pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                ))
                losses['interior'] = interior_loss

                # Add L2 regularization for weights
                l2_loss = 0.0005 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                # Add regiteration to encourage reasonable Diffusion values early
                d_reg = 0.1 * tf.square(tf.math.log(pinn.D + 1e-6) - tf.math.log(tf.constant(0.0001, dtype=tf.float32)))

                # Total loss
                total_loss = (
                    losses['total'] +
                    phase1_weights['interior'] * interior_loss +
                    l2_loss + d_reg
                )
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)  # More aggressive clipping in phase 1
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
                print(f"\nPhase 1 - Epoch {epoch}, D = {D_history[-1]:.6f}, Loss = {losses['total']:.6f}")

        # Phase 2: Data-fitting and refinement
        print("\nPhase 2: Data-fitting and refinement...")
        phase2_epochs = min(epochs // 3, 2000)

        # Custom loss weights for phase 2
        phase2_weights = {
            'initial': 1.0,      # Balanced initial conditions
            'boundary': 1.0,     # Balanced boundary conditions
            'interior': 5.0,     # Higher weight on interior data points
            'physics': 5.0       # Slightly reduced physics weight
        }

        for epoch in range(phase2_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            # Apply learning rate schedule
            if hasattr(optimizer, 'learning_rate'):
                optimizer.learning_rate = lr_schedule(phase1_epochs + epoch)

            print(f"\rPhase 2: {epoch+1}/{phase2_epochs}, LR: {optimizer.learning_rate.numpy():.2e}", end="", flush=True)

            with tf.GradientTape() as tape:
                # Compute losses with phase 2 weights
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=phase2_weights
                )

                interior_loss = tf.reduce_mean(tf.square(
                    pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                ))
                losses['interior'] = interior_loss

                # Reduced L2 regularization
                l2_loss = 0.0001 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                # Total loss
                total_loss = (
                    losses['total'] +
                    phase2_weights['interior'] * interior_loss +
                    l2_loss
                )
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  # Less aggressive clipping in phase 2
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
                print(f"\nPhase 2 - Epoch {epoch}, D = {D_history[-1]:.6f}, Loss = {losses['total']:.6f}")

        # Phase 3: Fine-tuning
        print("\nPhase 3: Fine-tuning...")
        phase3_epochs = epochs - phase1_epochs - phase2_epochs

        # Adaptive loss weighting based on previous loss values
        avg_losses = {k: 0.0 for k in ['initial', 'boundary', 'interior', 'physics']}
        for loss_dict in loss_history[-100:]:
            for k in avg_losses.keys():
                if k in loss_dict:
                    avg_losses[k] += loss_dict[k] / 100

        # Set weights inversely proportional to loss magnitude
        max_loss = max(avg_losses.values())
        phase3_weights = {k: max(1.0, max_loss / (v + 1e-8)) for k, v in avg_losses.items()}

        # Normalize weights to reasonable range
        weight_sum = sum(phase3_weights.values())
        phase3_weights = {k: 10.0 * v / weight_sum for k, v in phase3_weights.items()}

        print(f"Phase 3 adaptive weights: {phase3_weights}")

        # Exponential moving average of parameters for stability
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        for epoch in range(phase3_epochs):
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            # Apply learning rate schedule
            if hasattr(optimizer, 'learning_rate'):
                optimizer.learning_rate = lr_schedule(phase1_epochs + phase2_epochs + epoch)

            print(f"\rPhase 3: {epoch+1}/{phase3_epochs}, LR: {optimizer.learning_rate.numpy():.2e}", end="", flush=True)

            with tf.GradientTape() as tape:
                # Compute losses with adaptive weights
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=phase3_weights
                )

                interior_loss = tf.reduce_mean(tf.square(
                    pinn.forward_pass(data['X_i_train']) - data['u_i_train']
                ))
                losses['interior'] = interior_loss

                # Minimal regularization in final phase
                l2_loss = 1e-5 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                # Total loss
                total_loss = (
                    losses['total'] +
                    phase3_weights['interior'] * interior_loss +
                    l2_loss
                )
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 2.0)  # Least aggressive clipping in phase 3
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Apply exponential moving average to parameters
            ema.apply(trainable_vars)

            # Every 100 epochs, update with EMA values temporarily for evaluation
            if epoch % 100 == 0 or epoch == phase3_epochs - 1:
                # Store original values
                original_vars = [var.numpy() for var in trainable_vars]

                # Apply EMA values
                for i, var in enumerate(trainable_vars):
                    var.assign(ema.average(var).numpy())

                # Record history with EMA values
                D_history.append(pinn.get_diffusion_coefficient())
                loss_current = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train']
                )
                loss_history.append({k: float(v.numpy()) for k, v in loss_current.items()})

                print(f"\nPhase 3 - Epoch {epoch}, D = {D_history[-1]:.6f}, Loss = {loss_current['total']:.6f}")

                # Restore original values
                for i, var in enumerate(trainable_vars):
                    var.assign(original_vars[i])
            else:
                # Record history with current values
                D_history.append(pinn.get_diffusion_coefficient())
                loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

        # Apply EMA values at the end for final model
        for var in trainable_vars:
            var.assign(ema.average(var).numpy())

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

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
