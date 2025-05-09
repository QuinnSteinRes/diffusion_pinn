import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import os
import json
from ..variables import PINN_VARIABLES
from ..models.pinn import DiffusionPINN

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
    """
    Training function with improved stability and EMA implementation

    Args:
        pinn: The PINN model to train
        data: Dictionary of training data tensors
        optimizer: TensorFlow optimizer
        epochs: Number of training epochs
        save_dir: Directory to save checkpoints
        checkpoint_frequency: How often to save checkpoints

    Returns:
        Tuple of (diffusion coefficient history, loss history)
    """
    # Main history tracking
    D_history = []
    loss_history = []

    # Define acceptable range for diffusion coefficient
    D_min = 5e-7  # Lower minimum to allow more exploration
    D_max = 2e-3  # Slightly higher maximum

    # Create EMA model - a completely separate model instance
    ema_pinn = create_ema_model(pinn)
    ema_decay = 0.995  # Higher value for more stability (closer to 1.0)
    ema_active = False  # Flag to track if we're using EMA values for evaluation

    # Multi-stage training approach
    total_epochs = epochs
    phase1_epochs = min(epochs // 5, 1000)
    phase2_epochs = min(epochs // 3, 2000)
    phase3_epochs = epochs - phase1_epochs - phase2_epochs

    # Phase 1: PDE physics emphasis
    print(f"Phase 1: Physics-based initialization ({phase1_epochs} epochs)...")

    # Initial loss weights
    phase1_weights = {
        'initial': 2.0,      # Emphasize initial conditions
        'boundary': 2.0,     # Emphasize boundary conditions
        'interior': 1.0,     # Less weight on interior data points initially
        'physics': 10.0      # High weight on physics (PDE)
    }

    try:
        # Phase 1 training loop
        for epoch in range(phase1_epochs):
            # Periodic memory cleanup
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            # Calculate smooth learning rate
            current_lr = get_smooth_lr(epoch, phase1_epochs, initial_lr=1e-4, min_lr=1e-5)

            # Apply learning rate if optimizer supports it
            if hasattr(optimizer, 'learning_rate'):
                if hasattr(optimizer.learning_rate, 'assign'):
                    # For Variable learning rates
                    optimizer.learning_rate.assign(current_lr)
                elif isinstance(optimizer.learning_rate, tf.Variable):
                    # Another way to check for Variable
                    optimizer.learning_rate.assign(current_lr)
                else:
                    # For schedule-based or tensor learning rates
                    # Skip the direct assignment and just log the value
                    print(f"Using schedule-based learning rate: {optimizer.get_config()['learning_rate']}")
                    print(f"Current recommended lr value: {current_lr:.6f}")

            # Training step with gradient tape
            with tf.GradientTape() as tape:
                # Compute losses with phase 1 weights
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=phase1_weights
                )

                # Compute interior loss separately for better stability
                interior_loss = compute_stable_interior_loss(
                    pinn, data['X_i_train'], data['u_i_train']
                )
                losses['interior'] = interior_loss

                # Add L2 regularization for weights
                l2_loss = 0.0005 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                # Add regularization to encourage reasonable diffusion values
                d_reg = 0.1 * tf.square(tf.math.log(pinn.D + 1e-6) - tf.math.log(tf.constant(0.0001, dtype=tf.float32)))

                # Total loss
                total_loss = (
                    losses['total'] +
                    phase1_weights['interior'] * interior_loss +
                    l2_loss + d_reg
                )
                losses['total'] = total_loss

            # Calculate and apply gradients with stability measures
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply consistent gradient clipping
            gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

            # Log warning for large gradients
            if global_norm > 10.0:
                print(f"Warning: Large gradient norm: {global_norm:.2f}")

            # Apply gradients
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Enforce diffusion coefficient constraints
            if pinn.config.diffusion_trainable:
                D_value = pinn.D.numpy()
                if D_value < D_min or D_value > D_max:
                    constrained_D = np.clip(D_value, D_min, D_max)
                    pinn.D.assign(constrained_D)

            # Update EMA model - without affecting main training
            update_ema_model(ema_pinn, pinn, ema_decay)

            # Record primary model history only
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            # Logging and checkpoints
            if epoch % 100 == 0:
                print(f"\nPhase 1 - Epoch {epoch}, D = {D_history[-1]:.6f}, Loss = {losses['total']:.6f}, LR = {current_lr:.6f}")

                # Evaluate EMA model separately (for monitoring only)
                ema_losses = ema_pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=phase1_weights
                )
                ema_D = ema_pinn.get_diffusion_coefficient()
                print(f"EMA model - D = {ema_D:.6f}, Loss = {float(ema_losses['total']):.6f}")

                # Save checkpoint if requested
                if save_dir and epoch % checkpoint_frequency == 0:
                    save_checkpoint(pinn, save_dir, f"phase1_{epoch}")

        # Phase 2: Data-fitting and refinement
        print(f"\nPhase 2: Data-fitting and refinement ({phase2_epochs} epochs)...")

        # Transition weights gradually
        phase2_weights = {
            'initial': 1.0,      # Balanced initial conditions
            'boundary': 1.0,     # Balanced boundary conditions
            'interior': 3.0,     # Gradually increased interior weight
            'physics': 6.0       # Slightly reduced physics weight
        }

        for epoch in range(phase2_epochs):
            # Periodic memory cleanup
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            # Calculate smooth learning rate - slightly lower than phase 1
            current_lr = get_smooth_lr(epoch, phase2_epochs, initial_lr=5e-4, min_lr=1e-5)

            # Apply learning rate if optimizer supports it
            if hasattr(optimizer, 'learning_rate'):
                if hasattr(optimizer.learning_rate, 'assign'):
                    # For Variable learning rates
                    optimizer.learning_rate.assign(current_lr)
                elif isinstance(optimizer.learning_rate, tf.Variable):
                    # Another way to check for Variable
                    optimizer.learning_rate.assign(current_lr)
                else:
                    # For schedule-based or tensor learning rates
                    # Skip the direct assignment and just log the value
                    print(f"Using schedule-based learning rate: {optimizer.get_config()['learning_rate']}")
                    print(f"Current recommended lr value: {current_lr:.6f}")

            # Phase transition factor for smooth weight transition
            transition_factor = min(1.0, epoch / (phase2_epochs * 0.7))

            # Smooth weight transition between phases
            current_weights = {}
            for key in phase1_weights:
                current_weights[key] = phase1_weights[key] + (phase2_weights[key] - phase1_weights[key]) * transition_factor

            with tf.GradientTape() as tape:
                # Compute losses with transitioning weights
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=current_weights
                )

                # Compute interior loss with stability measures
                interior_loss = compute_stable_interior_loss(
                    pinn, data['X_i_train'], data['u_i_train']
                )
                losses['interior'] = interior_loss

                # Reduced L2 regularization
                l2_loss = 0.0001 * sum(tf.reduce_sum(tf.square(w)) for w in pinn.weights)

                # Reduced D regularization
                d_reg = 0.05 * tf.square(tf.math.log(pinn.D + 1e-6) - tf.math.log(tf.constant(0.0001, dtype=tf.float32)))

                # Total loss
                total_loss = (
                    losses['total'] +
                    current_weights['interior'] * interior_loss +
                    l2_loss + d_reg
                )
                losses['total'] = total_loss

            # Calculate and apply gradients
            trainable_vars = pinn.get_trainable_variables()
            gradients = tape.gradient(total_loss, trainable_vars)

            # Apply consistent gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Enforce diffusion coefficient constraints
            if pinn.config.diffusion_trainable:
                D_value = pinn.D.numpy()
                if D_value < D_min or D_value > D_max:
                    constrained_D = np.clip(D_value, D_min, D_max)
                    pinn.D.assign(constrained_D)

            # Update EMA model
            update_ema_model(ema_pinn, pinn, ema_decay)

            # Record history
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            # Logging and checkpoints
            if epoch % 100 == 0:
                print(f"\nPhase 2 - Epoch {epoch}, D = {D_history[-1]:.6f}, Loss = {losses['total']:.6f}, LR = {current_lr:.6f}")

                # Evaluate EMA model separately
                ema_losses = ema_pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=current_weights
                )
                ema_D = ema_pinn.get_diffusion_coefficient()
                print(f"EMA model - D = {ema_D:.6f}, Loss = {float(ema_losses['total']):.6f}")

                # Save checkpoint if requested
                if save_dir and epoch % checkpoint_frequency == 0:
                    save_checkpoint(pinn, save_dir, f"phase2_{epoch}")

        # Phase 3: Fine-tuning
        print(f"\nPhase 3: Fine-tuning ({phase3_epochs} epochs)...")

        # Final phase weights with emphasis on data fitting
        phase3_weights = {
            'initial': 1.0,      # Balanced initial conditions
            'boundary': 1.0,     # Balanced boundary conditions
            'interior': 10.0,     # Higher weight on interior data points
            'physics': 3.0       # Reduced physics weight to prioritize data
        }

        for epoch in range(phase3_epochs):
            # Periodic memory cleanup
            if epoch % 50 == 0:
                tf.keras.backend.clear_session()

            # Calculate smooth learning rate - low and stable for fine-tuning
            current_lr = get_smooth_lr(epoch, phase3_epochs, initial_lr=1e-4, min_lr=5e-6)

            # Apply learning rate if optimizer supports it
            if hasattr(optimizer, 'learning_rate'):
                if hasattr(optimizer.learning_rate, 'assign'):
                    # For Variable learning rates
                    optimizer.learning_rate.assign(current_lr)
                elif isinstance(optimizer.learning_rate, tf.Variable):
                    # Another way to check for Variable
                    optimizer.learning_rate.assign(current_lr)
                else:
                    # For schedule-based or tensor learning rates
                    # Skip the direct assignment and just log the value
                    print(f"Using schedule-based learning rate: {optimizer.get_config()['learning_rate']}")
                    print(f"Current recommended lr value: {current_lr:.6f}")

            with tf.GradientTape() as tape:
                # Compute losses with phase 3 weights
                losses = pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=phase3_weights
                )

                # Compute interior loss with stability measures
                interior_loss = compute_stable_interior_loss(
                    pinn, data['X_i_train'], data['u_i_train']
                )
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

            # Apply consistent gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Enforce diffusion coefficient constraints
            if pinn.config.diffusion_trainable:
                D_value = pinn.D.numpy()
                if D_value < D_min or D_value > D_max:
                    constrained_D = np.clip(D_value, D_min, D_max)
                    pinn.D.assign(constrained_D)

            # Update EMA model
            update_ema_model(ema_pinn, pinn, ema_decay)

            # Record history
            D_history.append(pinn.get_diffusion_coefficient())
            loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

            # Logging and checkpoints
            if epoch % 100 == 0:
                print(f"\nPhase 3 - Epoch {epoch}, D = {D_history[-1]:.6f}, Loss = {losses['total']:.6f}, LR = {current_lr:.6f}")

                # Evaluate EMA model separately
                ema_losses = ema_pinn.loss_fn(
                    x_data=data['X_u_train'],
                    c_data=data['u_train'],
                    x_physics=data['X_f_train'],
                    weights=phase3_weights
                )
                ema_D = ema_pinn.get_diffusion_coefficient()
                print(f"EMA model - D = {ema_D:.6f}, Loss = {float(ema_losses['total']):.6f}")

                # Save checkpoint if requested
                if save_dir and epoch % checkpoint_frequency == 0:
                    save_checkpoint(pinn, save_dir, f"phase3_{epoch}")

        # Final evaluation
        print("\nTraining complete. Evaluating final models...")

        # Evaluate primary model
        final_losses = pinn.loss_fn(
            x_data=data['X_u_train'],
            c_data=data['u_train'],
            x_physics=data['X_f_train'],
            weights=phase3_weights
        )
        final_D = pinn.get_diffusion_coefficient()
        print(f"Final model - D = {final_D:.6f}, Loss = {float(final_losses['total']):.6f}")

        # Evaluate EMA model
        ema_final_losses = ema_pinn.loss_fn(
            x_data=data['X_u_train'],
            c_data=data['u_train'],
            x_physics=data['X_f_train'],
            weights=phase3_weights
        )
        ema_final_D = ema_pinn.get_diffusion_coefficient()
        print(f"EMA model - D = {ema_final_D:.6f}, Loss = {float(ema_final_losses['total']):.6f}")

        # Replace with EMA model if it performs better
        if float(ema_final_losses['total']) < float(final_losses['total']):
            print("EMA model performed better. Replacing primary model with EMA model.")
            copy_model_parameters(pinn, ema_pinn)

            # Add final EMA model value to history
            D_history.append(ema_final_D)
            loss_history.append({k: float(v.numpy()) for k, v in ema_final_losses.items()})

        # Save final model if directory provided
        if save_dir:
            save_checkpoint(pinn, save_dir, "final")

            # Also save a separate checkpoint for the EMA model
            save_checkpoint(ema_pinn, save_dir, "final_ema")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    return D_history, loss_history

# Helper functions for the train_pinn function

def create_ema_model(source_model):
    """Create a copy of the model for EMA tracking"""
    # Get model configuration
    config = source_model.config

    # Create new model with same configuration
    ema_model = DiffusionPINN(
        spatial_bounds={
            'x': source_model.x_bounds,
            'y': source_model.y_bounds
        },
        time_bounds=source_model.t_bounds,
        initial_D=source_model.D.numpy(),
        config=config
    )

    # Copy initial weights
    for i, w in enumerate(source_model.weights):
        ema_model.weights[i].assign(w.numpy())
    for i, b in enumerate(source_model.biases):
        ema_model.biases[i].assign(b.numpy())

    return ema_model

def update_ema_model(ema_model, source_model, decay):
    """Update EMA model weights based on source model"""
    for i, w in enumerate(source_model.weights):
        ema_model.weights[i].assign(decay * ema_model.weights[i] + (1 - decay) * w)
    for i, b in enumerate(source_model.biases):
        ema_model.biases[i].assign(decay * ema_model.biases[i] + (1 - decay) * b)

    # Update diffusion coefficient if trainable
    if source_model.config.diffusion_trainable:
        ema_model.D.assign(decay * ema_model.D + (1 - decay) * source_model.D)

def copy_model_parameters(target_model, source_model):
    """Copy parameters from source model to target model"""
    for i, w in enumerate(source_model.weights):
        target_model.weights[i].assign(w.numpy())
    for i, b in enumerate(source_model.biases):
        target_model.biases[i].assign(b.numpy())

    if source_model.config.diffusion_trainable:
        target_model.D.assign(source_model.D.numpy())

def get_smooth_lr(epoch, total_epochs, initial_lr=1e-4, min_lr=1e-5):
    """Smooth learning rate schedule without restarts"""
    # Warmup phase
    warmup_epochs = min(200, total_epochs // 10)

    if epoch < warmup_epochs:
        # Linear warm-up
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Smooth exponential decay
        decay_epochs = total_epochs - warmup_epochs
        decay_position = (epoch - warmup_epochs) / max(1, decay_epochs)

        # Exponential decay with floor
        decay_factor = tf.exp(-4.0 * decay_position)
        return tf.maximum(min_lr, initial_lr * decay_factor)

def compute_stable_interior_loss(pinn, x_interior, c_interior):
    """Compute interior loss with stability enhancements"""
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
