import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
import os
import json
import gc
from ..variables import PINN_VARIABLES

def create_open_system_pinn(inputfile: str,
                           N_u: int = PINN_VARIABLES['N_u'],
                           N_f: int = PINN_VARIABLES['N_f'],
                           N_i: int = PINN_VARIABLES['N_i'],
                           initial_D: float = PINN_VARIABLES['initial_D'],
                           initial_k: float = 0.001,
                           seed: int = PINN_VARIABLES['random_seed']) -> Tuple['OpenSystemDiffusionPINN', Dict[str, tf.Tensor]]:
    """Create PINN for open diffusion system with boundary flux"""
    from ..data.processor import DiffusionDataProcessor
    from ..models.pinn import OpenSystemDiffusionPINN
    from ..config import DiffusionConfig

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Process data
    data_processor = DiffusionDataProcessor(inputfile, seed=seed)
    domain_info = data_processor.get_domain_info()

    # Check mass conservation
    total_mass_over_time = []
    for t_idx in range(len(data_processor.t)):
        total_mass = np.sum(data_processor.usol[:, :, t_idx])
        total_mass_over_time.append(total_mass)

    mass_loss_rate = (total_mass_over_time[-1] - total_mass_over_time[0]) / (data_processor.t[-1] - data_processor.t[0])
    print(f"Mass loss analysis:")
    print(f"  Initial total mass: {total_mass_over_time[0]:.3f}")
    print(f"  Final total mass: {total_mass_over_time[-1]:.3f}")
    print(f"  Mass loss rate: {mass_loss_rate:.6f} units/time")
    print(f"  Relative mass loss: {(total_mass_over_time[0] - total_mass_over_time[-1])/total_mass_over_time[0]*100:.1f}%")

    # Estimate initial boundary permeability from mass loss
    if mass_loss_rate < 0:  # Mass is decreasing
        characteristic_concentration = np.mean(data_processor.usol[:, :, 0])
        boundary_length = 2 * (domain_info['spatial_bounds']['x'][1] - domain_info['spatial_bounds']['x'][0] +
                              domain_info['spatial_bounds']['y'][1] - domain_info['spatial_bounds']['y'][0])
        estimated_k = abs(mass_loss_rate) / (boundary_length * characteristic_concentration)
        initial_k = max(estimated_k, 1e-6)
        print(f"  Estimated initial k: {initial_k:.6e}")

    # Create PINN config
    config = DiffusionConfig(
        diffusion_trainable=True,
        use_physics_loss=True
    )

    # Initialize open system PINN
    pinn = OpenSystemDiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=initial_D,
        initial_k=initial_k,
        c_external=0.0,  # Assume external concentration is zero
        config=config,
        seed=seed
    )

    # Prepare training data for open system
    training_data = prepare_open_system_data(data_processor, N_u, N_f, N_i, seed)

    return pinn, training_data

def prepare_open_system_data(data_processor, N_u: int, N_f: int, N_i: int, seed: int = None) -> Dict[str, tf.Tensor]:
    """Prepare training data specifically for open system"""
    if seed is not None:
        np.random.seed(seed)

    print(f"Preparing open system training data:")
    print(f"  N_u (boundary points): {N_u}")
    print(f"  N_f (physics points): {N_f}")
    print(f"  N_i (interior data): {N_i}")

    # Get all data points
    all_coords, all_values = data_processor.get_boundary_and_interior_points()

    # Separate initial condition data (t = t_min only)
    t_min = data_processor.t.min()
    initial_mask = np.abs(all_coords[:, 2] - t_min) < 1e-6

    initial_coords = all_coords[initial_mask]
    initial_values = all_values[initial_mask]

    print(f"  Initial condition points available: {len(initial_coords)}")

    # Subsample initial condition if needed
    if len(initial_coords) > N_u:
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(len(initial_coords), N_u, replace=False)
        indices.sort()
        X_u_train = initial_coords[indices]
        u_train = initial_values[indices]
    else:
        X_u_train = initial_coords
        u_train = initial_values

    print(f"  Using {len(X_u_train)} initial condition points")

    # Generate physics points (interior + boundary) for all times
    # These will be used to enforce PDE and boundary conditions
    X_f_train = data_processor.create_deterministic_collocation_points(N_f, seed=seed)

    # Add some points specifically on boundaries for Robin conditions
    boundary_points = generate_boundary_physics_points(data_processor, N_f // 4, seed)
    X_f_train = np.vstack([X_f_train, boundary_points])

    print(f"  Physics points (total): {len(X_f_train)}")

    # Interior data points for additional supervision (optional, can be reduced)
    non_initial_mask = ~initial_mask
    if np.any(non_initial_mask) and N_i > 0:
        non_initial_coords = all_coords[non_initial_mask]
        non_initial_values = all_values[non_initial_mask]

        if len(non_initial_coords) > N_i:
            if seed is not None:
                np.random.seed(seed + 1)
            indices = np.random.choice(len(non_initial_coords), N_i, replace=False)
            indices.sort()
            X_i_train = non_initial_coords[indices]
            u_i_train = non_initial_values[indices]
        else:
            X_i_train = non_initial_coords
            u_i_train = non_initial_values
    else:
        X_i_train = np.empty((0, 3))
        u_i_train = np.empty((0, 1))

    print(f"  Interior supervision points: {len(X_i_train)}")

    # Convert to tensors
    training_data = {
        'X_u_train': tf.convert_to_tensor(X_u_train, dtype=tf.float32),
        'u_train': tf.convert_to_tensor(u_train, dtype=tf.float32),
        'X_i_train': tf.convert_to_tensor(X_i_train, dtype=tf.float32),
        'u_i_train': tf.convert_to_tensor(u_i_train, dtype=tf.float32),
        'X_f_train': tf.convert_to_tensor(X_f_train, dtype=tf.float32),
        # Test data for evaluation
        'X_u_test': tf.convert_to_tensor(data_processor.X_u_test, dtype=tf.float32),
        'u_test': tf.convert_to_tensor(data_processor.u, dtype=tf.float32)
    }

    return training_data

def generate_boundary_physics_points(data_processor, n_points: int, seed: int = None) -> np.ndarray:
    """Generate points specifically on boundaries for Robin condition enforcement"""
    if seed is not None:
        np.random.seed(seed + 2)

    x_bounds = [data_processor.x.min(), data_processor.x.max()]
    y_bounds = [data_processor.y.min(), data_processor.y.max()]
    t_bounds = [data_processor.t.min(), data_processor.t.max()]

    boundary_points = []
    points_per_boundary = n_points // 4

    # Generate time and one spatial coordinate randomly
    t_vals = np.random.uniform(t_bounds[0], t_bounds[1], points_per_boundary * 4)

    # X boundaries (x = x_min, x = x_max)
    y_vals_x = np.random.uniform(y_bounds[0], y_bounds[1], points_per_boundary * 2)

    # x = x_min boundary
    x_min_points = np.column_stack([
        np.full(points_per_boundary, x_bounds[0]),
        y_vals_x[:points_per_boundary],
        t_vals[:points_per_boundary]
    ])

    # x = x_max boundary
    x_max_points = np.column_stack([
        np.full(points_per_boundary, x_bounds[1]),
        y_vals_x[points_per_boundary:2*points_per_boundary],
        t_vals[points_per_boundary:2*points_per_boundary]
    ])

    # Y boundaries (y = y_min, y = y_max)
    x_vals_y = np.random.uniform(x_bounds[0], x_bounds[1], points_per_boundary * 2)

    # y = y_min boundary
    y_min_points = np.column_stack([
        x_vals_y[:points_per_boundary],
        np.full(points_per_boundary, y_bounds[0]),
        t_vals[2*points_per_boundary:3*points_per_boundary]
    ])

    # y = y_max boundary
    y_max_points = np.column_stack([
        x_vals_y[points_per_boundary:],
        np.full(points_per_boundary, y_bounds[1]),
        t_vals[3*points_per_boundary:4*points_per_boundary]
    ])

    boundary_points = np.vstack([x_min_points, x_max_points, y_min_points, y_max_points])

    return boundary_points

def train_open_system_pinn(pinn: 'OpenSystemDiffusionPINN',
                          data: Dict[str, tf.Tensor],
                          optimizer: tf.keras.optimizers.Optimizer,
                          epochs: int = 100,
                          save_dir: str = None,
                          seed: int = None) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
    """Train open system PINN with proper phase scheduling"""
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    D_history = []
    k_history = []  # Track boundary permeability
    loss_history = []

    # Multi-phase training for open system
    phase_configs = [
        {
            'name': 'Phase 1: Initial Condition Learning',
            'epochs': epochs // 6,
            'weights': {'initial': 100.0, 'boundary': 0.1, 'interior': 0.1},
            'lr_schedule': lambda epoch, total: 1e-3,
            'description': 'Focus on fitting initial condition accurately'
        },
        {
            'name': 'Phase 2: Boundary Physics Introduction',
            'epochs': epochs // 6,
            'weights': {'initial': 10.0, 'boundary': 10.0, 'interior': 1.0},
            'lr_schedule': lambda epoch, total: 5e-4,
            'description': 'Introduce Robin boundary conditions gradually'
        },
        {
            'name': 'Phase 3: Interior Physics',
            'epochs': epochs // 3,
            'weights': {'initial': 5.0, 'boundary': 5.0, 'interior': 15.0},
            'lr_schedule': lambda epoch, total: 2e-4 * (0.95 ** (epoch // 50)),
            'description': 'Balance all physics components'
        },
        {
            'name': 'Phase 4: Fine-tuning',
            'epochs': epochs // 4,
            'weights': {'initial': 1.0, 'boundary': 10.0, 'interior': 10.0},
            'lr_schedule': lambda epoch, total: 1e-4 * (0.98 ** (epoch // 25)),
            'description': 'Fine-tune with emphasis on boundary physics'
        }
    ]

    epoch_counter = 0
    convergence_history = []

    print(f"\nStarting open system training for {epochs} epochs")
    print(f"Learning parameters: D (diffusion), k (boundary permeability)")

    # Print initial parameters
    initial_D = pinn.get_diffusion_coefficient()
    initial_k = pinn.get_boundary_permeability()
    print(f"Initial D: {initial_D:.6e}")
    print(f"Initial k: {initial_k:.6e}")

    try:
        for phase_idx, phase in enumerate(phase_configs):
            print(f"\n{'='*60}")
            print(f"{phase['name']} - {phase['epochs']} epochs")
            print(f"Description: {phase['description']}")
            print(f"Loss weights: {phase['weights']}")
            print(f"{'='*60}")

            phase_start_epoch = epoch_counter

            for epoch in range(phase['epochs']):
                current_lr = phase['lr_schedule'](epoch, phase['epochs'])

                # Set learning rate
                if hasattr(optimizer, 'learning_rate'):
                    if hasattr(optimizer.learning_rate, 'assign'):
                        optimizer.learning_rate.assign(current_lr)

                # Training step
                with tf.GradientTape() as tape:
                    # Compute losses with current phase weights
                    losses = pinn.loss_fn(
                        x_data=data['X_u_train'],
                        c_data=data['u_train'],
                        x_physics=data['X_f_train'],
                        weights=phase['weights']
                    )

                    # Add interior supervision if available
                    if data['X_i_train'].shape[0] > 0:
                        interior_pred = pinn.forward_pass(data['X_i_train'])
                        interior_loss = tf.reduce_mean(tf.square(interior_pred - data['u_i_train']))
                        losses['interior_data'] = interior_loss
                        losses['total'] += 0.1 * interior_loss

                # Apply gradients
                trainable_vars = pinn.get_trainable_variables()
                gradients = tape.gradient(losses['total'], trainable_vars)

                # Gradient clipping for stability
                gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Record history
                current_D = pinn.get_diffusion_coefficient()
                current_k = pinn.get_boundary_permeability()

                D_history.append(current_D)
                k_history.append(current_k)
                loss_history.append({k: float(v.numpy()) for k, v in losses.items()})

                # Progress reporting
                if epoch_counter % 100 == 0:
                    print(f"Epoch {epoch_counter:5d}: D={current_D:.6e}, k={current_k:.6e}, "
                          f"Loss={losses['total']:.6f}, LR={current_lr:.2e}")

                    # Check convergence
                    if len(D_history) >= 50:
                        recent_D = D_history[-50:]
                        recent_k = k_history[-50:]
                        D_std = np.std(recent_D)
                        k_std = np.std(recent_k)
                        D_mean = np.mean(recent_D)
                        k_mean = np.mean(recent_k)

                        D_rel_std = D_std / D_mean if D_mean > 0 else float('inf')
                        k_rel_std = k_std / k_mean if k_mean > 0 else float('inf')

                        convergence_metric = max(D_rel_std, k_rel_std)
                        convergence_history.append(convergence_metric)

                        print(f"  D rel_std: {D_rel_std:.6f}, k rel_std: {k_rel_std:.6f}")

                epoch_counter += 1

                # Memory cleanup
                if epoch_counter % 100 == 0:
                    gc.collect()

            # Phase summary
            phase_final_D = D_history[-1]
            phase_final_k = k_history[-1]
            phase_final_loss = loss_history[-1]['total']

            print(f"\nPhase {phase_idx + 1} completed:")
            print(f"  Final D: {phase_final_D:.6e}")
            print(f"  Final k: {phase_final_k:.6e}")
            print(f"  Final Loss: {phase_final_loss:.6f}")

            # Save checkpoint
            if save_dir:
                save_open_system_checkpoint(pinn, save_dir, f"phase_{phase_idx+1}",
                                           D_history, k_history)

        # Final training summary
        final_D = D_history[-1]
        final_k = k_history[-1]
        final_loss = loss_history[-1]['total']

        print(f"\n{'='*60}")
        print("OPEN SYSTEM TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total epochs: {epoch_counter}")
        print(f"Final diffusion coefficient: {final_D:.8e}")
        print(f"Final boundary permeability: {final_k:.8e}")
        print(f"Final loss: {final_loss:.6f}")

        # Physical interpretation
        characteristic_time_diffusion = 1.0 / final_D  # Rough estimate
        characteristic_time_outflow = 1.0 / final_k

        print(f"\nPhysical Interpretation:")
        print(f"  Diffusion time scale: ~{characteristic_time_diffusion:.1f} time units")
        print(f"  Outflow time scale: ~{characteristic_time_outflow:.1f} time units")

        if characteristic_time_outflow < characteristic_time_diffusion:
            print("  System is outflow-dominated (fast boundary loss)")
        else:
            print("  System is diffusion-dominated (slow boundary loss)")

        # Save final model
        if save_dir:
            save_open_system_checkpoint(pinn, save_dir, "final", D_history, k_history)

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

    return D_history, k_history, loss_history

def save_open_system_checkpoint(pinn: 'OpenSystemDiffusionPINN', save_dir: str,
                               epoch: str, D_history: List[float], k_history: List[float]):
    """Save checkpoint with both D and k histories"""
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration including new parameters
    config_dict = {
        'hidden_layers': pinn.config.hidden_layers,
        'activation': pinn.config.activation,
        'spatial_bounds': {
            'x': [float(pinn.x_bounds[0]), float(pinn.x_bounds[1])],
            'y': [float(pinn.y_bounds[0]), float(pinn.y_bounds[1])]
        },
        'time_bounds': [float(pinn.t_bounds[0]), float(pinn.t_bounds[1])],
        'D_value': float(pinn.get_diffusion_coefficient()),
        'k_value': float(pinn.get_boundary_permeability()),
        'c_external': float(pinn.get_external_concentration()),
        'log_D_bounds': [float(pinn.log_D_min), float(pinn.log_D_max)],
        'log_k_bounds': [float(pinn.log_k_min), float(pinn.log_k_max)],
        'parameterization': 'open_system_logarithmic',
        'model_type': 'OpenSystemDiffusionPINN'
    }

    with open(os.path.join(save_dir, f'config_{epoch}.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    # Save parameter histories
    history_dict = {
        'D_history': D_history,
        'k_history': k_history,
        'final_D': float(pinn.get_diffusion_coefficient()),
        'final_k': float(pinn.get_boundary_permeability()),
        'epochs': len(D_history)
    }

    with open(os.path.join(save_dir, f'parameter_history_{epoch}.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)

    # Save network weights
    weights_dict = {f'weight_{i}': w.numpy().tolist() for i, w in enumerate(pinn.weights)}
    biases_dict = {f'bias_{i}': b.numpy().tolist() for i, b in enumerate(pinn.biases)}

    with open(os.path.join(save_dir, f'weights_{epoch}.json'), 'w') as f:
        json.dump(weights_dict, f)
    with open(os.path.join(save_dir, f'biases_{epoch}.json'), 'w') as f:
        json.dump(biases_dict, f)

    print(f"Checkpoint saved: {save_dir}/config_{epoch}.json")

# Keep some old functions for backward compatibility during transition
def train_pinn(*args, **kwargs):
    """Backward compatibility wrapper"""
    print("WARNING: train_pinn() is deprecated. Use train_open_system_pinn() instead.")
    # You could implement a wrapper here if needed
    raise NotImplementedError("Use train_open_system_pinn() for open system physics")

def create_and_initialize_pinn(*args, **kwargs):
    """Backward compatibility wrapper"""
    print("WARNING: create_and_initialize_pinn() is deprecated. Use create_open_system_pinn() instead.")
    raise NotImplementedError("Use create_open_system_pinn() for open system physics")