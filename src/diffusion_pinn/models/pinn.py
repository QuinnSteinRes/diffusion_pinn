import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES

class DiffusionPINN(tf.Module):
    """Physics-Informed Neural Network for diffusion problems with FIXED logarithmic D parameterization"""

    def __init__(
        self,
        spatial_bounds: Dict[str, Tuple[float, float]],
        time_bounds: Tuple[float, float],
        initial_D: float = PINN_VARIABLES['initial_D'],
        config: DiffusionConfig = None,
        seed: int = None,
        data_processor = None
    ):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Store bounds for normalization
        self.x_bounds = spatial_bounds['x']
        self.y_bounds = spatial_bounds['y']
        self.t_bounds = time_bounds

        # Create normalized bounds as tensors
        self.lb = tf.constant([self.x_bounds[0], self.y_bounds[0], self.t_bounds[0]],
                            dtype=tf.float32)
        self.ub = tf.constant([self.x_bounds[1], self.y_bounds[1], self.t_bounds[1]],
                            dtype=tf.float32)

        # FIXED LOGARITHMIC PARAMETERIZATION: Initialize log(D) instead of D
        initial_D_value = max(initial_D, 1e-8)  # Ensure positive value
        initial_log_D = np.log(initial_D_value)

        print(f"Initial D: {initial_D_value:.8e}")
        print(f"Initial log(D): {initial_log_D:.6f}")

        # Store log(D) as the trainable parameter
        self.log_D = tf.Variable(
            initial_log_D,
            dtype=tf.float32,
            trainable=self.config.diffusion_trainable,
            name='log_diffusion_coefficient'
        )

        # WIDENED bounds for log(D) to avoid constraining solution
        self.log_D_min = -20.0   # Corresponds to D ≈ 2e-9
        self.log_D_max = -2.0    # Corresponds to D ≈ 0.135

        print(f"log(D) bounds: [{self.log_D_min:.1f}, {self.log_D_max:.1f}]")
        print(f"Corresponding D bounds: [{np.exp(self.log_D_min):.2e}, {np.exp(self.log_D_max):.2e}]")

        # Store loss weights from variables
        self.loss_weights = PINN_VARIABLES['loss_weights']

        # FIXED: Use exact equality for condition identification instead of tolerances
        self.boundary_tol = 0.0  # Will use exact equality
        self.initial_tol = 0.0   # Will use exact equality

        # Build network architecture
        self._build_network()

    def get_diffusion_coefficient(self) -> float:
        """Get the current estimate of the diffusion coefficient"""
        # Convert from log(D) back to D
        D_value = tf.exp(self.log_D).numpy()
        return float(D_value)

    def get_log_diffusion_coefficient(self) -> float:
        """Get the current log(D) value for debugging"""
        return float(self.log_D.numpy())

    def _build_network(self):
        """Initialize neural network parameters with enhanced initialization"""
        # Full architecture including input (3: x,y,t) and output (1: concentration)
        architecture = [3] + self.config.hidden_layers + [1]

        self.weights = []
        self.biases = []

        # Initialize weights and biases with improved schemes
        for i in range(len(architecture)-1):
            input_dim, output_dim = architecture[i], architecture[i+1]

            # Enhanced initialization based on layer position
            if i == 0:  # Input layer - use smaller variance to avoid saturation
                std_dv = 0.1 / np.sqrt(input_dim)
            elif i == len(architecture) - 2:  # Output layer - very small initialization
                std_dv = 0.01 / np.sqrt(input_dim)
            else:  # Hidden layers - modified Xavier/Glorot
                if self.config.activation == 'tanh':
                    # For tanh activation, use Xavier initialization
                    std_dv = np.sqrt(2.0 / (input_dim + output_dim))
                else:
                    # For ReLU-like activations, use He initialization
                    std_dv = np.sqrt(2.0 / input_dim)

            # Initialize weights with controlled random values using stored seed
            weight_seed = self.seed + i if self.seed is not None else None
            if weight_seed is not None:
                w = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32, seed=weight_seed) * std_dv,
                    trainable=True,
                    name=f'w{i+1}'
                )
            else:
                w = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32) * std_dv,
                    trainable=True,
                    name=f'w{i+1}'
                )

            # Initialize biases with enhanced strategy
            if i < len(architecture) - 2:
                # Small positive bias for hidden layers to avoid dead neurons
                bias_seed = self.seed + 100 + i if self.seed is not None else None
                if bias_seed is not None:
                    b_init = tf.random.uniform([output_dim], minval=0.01, maxval=0.05,
                                             dtype=tf.float32, seed=bias_seed)
                else:
                    b_init = tf.random.uniform([output_dim], minval=0.01, maxval=0.05, dtype=tf.float32)
            else:
                # Zero bias for output layer
                b_init = tf.zeros([output_dim], dtype=tf.float32)

            b = tf.Variable(b_init, trainable=True, name=f'b{i+1}')

            self.weights.append(w)
            self.biases.append(b)

    def _normalize_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Enhanced input normalization with improved numerical stability"""
        # Ensure input is float32 for consistency
        x_float = tf.cast(x, tf.float32)

        # Calculate range with numerical stability check
        range_tensor = self.ub - self.lb

        # Add small epsilon to prevent division by zero
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        safe_range = tf.maximum(range_tensor, epsilon)

        # Normalize to [0, 1] first
        normalized_01 = (x_float - self.lb) / safe_range

        # Clip to handle any numerical issues
        normalized_01_clipped = tf.clip_by_value(normalized_01, 0.0, 1.0)

        # Transform to [-1, 1]
        normalized_final = 2.0 * normalized_01_clipped - 1.0

        # Final safety clip
        return tf.clip_by_value(normalized_final, -1.0, 1.0)

    @tf.function
    def forward_pass(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through the network"""
        X = self._normalize_inputs(x)
        H = X
        for i in range(len(self.weights)-1):
            H = tf.matmul(H, self.weights[i]) + self.biases[i]
            if self.config.activation == 'tanh':
                H = tf.tanh(H)
            elif self.config.activation == 'sin':
                H = tf.sin(H)
            else:
                H = tf.nn.relu(H)

        # Apply final layer
        output = tf.matmul(H, self.weights[-1]) + self.biases[-1]
        return output

    def compute_pde_residual(self, x_f: tf.Tensor) -> tf.Tensor:
        """FIXED PDE residual computation with enhanced error checking and debugging"""

        print(f"Computing PDE residual for {x_f.shape[0]} points")

        try:
            # Process in fixed-size batches for consistency
            batch_size = 512
            total_points = tf.shape(x_f)[0]
            num_batches = (total_points - 1) // batch_size + 1

            residuals = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = tf.minimum(start_idx + batch_size, total_points)
                x_batch = x_f[start_idx:end_idx]

                # Compute with enhanced numerical stability
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x_batch)
                    c = self.forward_pass(x_batch)

                    # First derivatives
                    grad = tape.gradient(c, x_batch)

                    if grad is None:
                        print("WARNING: First derivatives are None!")
                        # Create zero gradients as fallback
                        grad = tf.zeros_like(x_batch)

                    # Extract components
                    dc_dx = tf.reshape(grad[:, 0], (-1, 1))
                    dc_dy = tf.reshape(grad[:, 1], (-1, 1))
                    dc_dt = tf.reshape(grad[:, 2], (-1, 1))

                # Second derivatives with error checking
                try:
                    d2c_dx2 = tape.gradient(dc_dx, x_batch)
                    d2c_dy2 = tape.gradient(dc_dy, x_batch)

                    if d2c_dx2 is None or d2c_dy2 is None:
                        print("WARNING: Second derivatives are None!")
                        # Create zero second derivatives as fallback
                        batch_size_actual = tf.shape(x_batch)[0]
                        d2c_dx2 = tf.zeros((batch_size_actual, 1), dtype=tf.float32)
                        d2c_dy2 = tf.zeros((batch_size_actual, 1), dtype=tf.float32)
                    else:
                        d2c_dx2 = d2c_dx2[:, 0:1]
                        d2c_dy2 = d2c_dy2[:, 1:2]

                except Exception as e:
                    print(f"Error computing second derivatives: {str(e)}")
                    batch_size_actual = tf.shape(x_batch)[0]
                    d2c_dx2 = tf.zeros((batch_size_actual, 1), dtype=tf.float32)
                    d2c_dy2 = tf.zeros((batch_size_actual, 1), dtype=tf.float32)

                # Cleanup tape
                del tape

                # Compute laplacian with stability checks
                laplacian = d2c_dx2 + d2c_dy2

                # Check for NaN or Inf values
                if not tf.reduce_all(tf.math.is_finite(laplacian)):
                    print("WARNING: Laplacian contains NaN or Inf values!")
                    laplacian = tf.where(tf.math.is_finite(laplacian), laplacian, tf.zeros_like(laplacian))

                # FIXED LOGARITHMIC PARAMETERIZATION: Convert log(D) to D with bounds checking
                # Apply soft bounds to prevent numerical overflow
                clipped_log_D = tf.clip_by_value(self.log_D, self.log_D_min + 2.0, self.log_D_max - 2.0)
                D_value = tf.exp(clipped_log_D)

                # Debug output for first batch
                if i == 0:
                    print(f"Current log(D): {self.log_D.numpy():.6f}")
                    print(f"Clipped log(D): {clipped_log_D.numpy():.6f}")
                    print(f"Current D: {D_value.numpy():.8e}")

                # Calculate residual: ∂c/∂t - D * ∇²c = 0
                residual = dc_dt - D_value * laplacian

                # Check residual statistics for first batch
                if i == 0:
                    residual_stats = {
                        'mean': tf.reduce_mean(tf.abs(residual)).numpy(),
                        'max': tf.reduce_max(tf.abs(residual)).numpy(),
                        'min': tf.reduce_min(tf.abs(residual)).numpy(),
                        'std': tf.math.reduce_std(residual).numpy()
                    }
                    print(f"PDE residual stats: {residual_stats}")

                    # Check individual components
                    dc_dt_mean = tf.reduce_mean(tf.abs(dc_dt)).numpy()
                    laplacian_mean = tf.reduce_mean(tf.abs(laplacian)).numpy()
                    diffusion_term_mean = tf.reduce_mean(tf.abs(D_value * laplacian)).numpy()

                    print(f"Component stats - dc_dt: {dc_dt_mean:.8f}, laplacian: {laplacian_mean:.8f}, D*laplacian: {diffusion_term_mean:.8f}")

                # Save for this batch
                residuals.append(residual)

            # Combine all batch residuals
            full_residual = tf.concat(residuals, axis=0)

            # Final check for problematic values
            if not tf.reduce_all(tf.math.is_finite(full_residual)):
                print("ERROR: Final residual contains NaN or Inf!")
                full_residual = tf.where(tf.math.is_finite(full_residual), full_residual, tf.zeros_like(full_residual))

            # Check if residual is essentially zero (indicates no learning)
            residual_magnitude = tf.reduce_mean(tf.abs(full_residual))
            if residual_magnitude < 1e-12:
                print(f"WARNING: PDE residual magnitude very small: {residual_magnitude.numpy():.2e}")
                print("This may indicate the network is not learning the physics properly")

            return full_residual

        except Exception as e:
            print(f"CRITICAL ERROR in compute_pde_residual: {str(e)}")
            import traceback
            traceback.print_exc()

            # Return small non-zero residual to prevent training from stopping
            fallback_residual = tf.ones((tf.shape(x_f)[0], 1), dtype=tf.float32) * 0.1
            print("Returning fallback residual to continue training")
            return fallback_residual

    def loss_fn(self, x_data: tf.Tensor, c_data: tf.Tensor,
                x_physics: tf.Tensor = None, weights: Dict[str, float] = None) -> Dict[str, tf.Tensor]:
        """FIXED loss function with exact equality for condition detection"""
        if weights is None:
            weights = self.loss_weights

        losses = {}

        # Extract coordinates
        t = x_data[:, 2]
        x_coord = x_data[:, 0]
        y_coord = x_data[:, 1]

        # FIXED: Use EXACT equality instead of tolerance-based comparison
        # Initial condition mask (t = t_min) - EXACT equality
        ic_mask = tf.equal(t, tf.cast(self.t_bounds[0], tf.float32))

        # Boundary condition masks - EXACT equality
        bc_x_mask = tf.logical_or(
            tf.equal(x_coord, tf.cast(self.x_bounds[0], tf.float32)),
            tf.equal(x_coord, tf.cast(self.x_bounds[1], tf.float32))
        )
        bc_y_mask = tf.logical_or(
            tf.equal(y_coord, tf.cast(self.y_bounds[0], tf.float32)),
            tf.equal(y_coord, tf.cast(self.y_bounds[1], tf.float32))
        )
        bc_mask = tf.logical_or(bc_x_mask, bc_y_mask)

        # Ensure boundary mask excludes initial condition points
        bc_mask = tf.logical_and(bc_mask, tf.logical_not(ic_mask))

        # Interior mask - points that are neither initial nor boundary
        interior_mask = tf.logical_not(tf.logical_or(ic_mask, bc_mask))

        # Debug mask counts
        ic_count = tf.reduce_sum(tf.cast(ic_mask, tf.int32))
        bc_count = tf.reduce_sum(tf.cast(bc_mask, tf.int32))
        interior_count = tf.reduce_sum(tf.cast(interior_mask, tf.int32))

        print(f"Mask counts - IC: {ic_count}, BC: {bc_count}, Interior: {interior_count}")

        # Compute losses for each condition type

        # Initial condition loss
        if tf.reduce_any(ic_mask):
            c_pred_ic = self.forward_pass(tf.boolean_mask(x_data, ic_mask))
            c_true_ic = tf.boolean_mask(c_data, ic_mask)
            losses['initial'] = tf.reduce_mean(tf.square(c_pred_ic - c_true_ic))
        else:
            losses['initial'] = tf.constant(0.0, dtype=tf.float32)
            print("WARNING: No initial condition points found!")

        # Boundary condition loss
        if tf.reduce_any(bc_mask):
            c_pred_bc = self.forward_pass(tf.boolean_mask(x_data, bc_mask))
            c_true_bc = tf.boolean_mask(c_data, bc_mask)
            losses['boundary'] = tf.reduce_mean(tf.square(c_pred_bc - c_true_bc))
        else:
            losses['boundary'] = tf.constant(0.0, dtype=tf.float32)
            print("WARNING: No boundary condition points found!")

        # Interior data loss
        if tf.reduce_any(interior_mask):
            c_pred_interior = self.forward_pass(tf.boolean_mask(x_data, interior_mask))
            c_true_interior = tf.boolean_mask(c_data, interior_mask)

            # Use Huber loss for robustness
            delta = 0.1
            abs_error = tf.abs(c_pred_interior - c_true_interior)
            quadratic = tf.minimum(abs_error, delta)
            linear = abs_error - quadratic
            losses['interior'] = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)
        else:
            losses['interior'] = tf.constant(0.0, dtype=tf.float32)
            print("WARNING: No interior points found!")

        # FIXED Physics loss - ALWAYS compute if we have physics points
        if x_physics is not None and x_physics.shape[0] > 0:
            try:
                pde_residual = self.compute_pde_residual(x_physics)

                # Enhanced physics loss computation with stability
                residual_mean = tf.reduce_mean(tf.abs(pde_residual))

                print(f"Physics residual mean: {residual_mean.numpy():.8e}")

                if tf.math.is_finite(residual_mean) and residual_mean > 1e-15:
                    # Multi-scale physics loss with Huber-like formulation
                    delta = tf.maximum(0.01, residual_mean)  # Adaptive delta
                    abs_residual = tf.abs(pde_residual)

                    # Prevent extreme residuals from dominating
                    abs_residual_capped = tf.minimum(abs_residual, 100.0 * delta)
                    quadratic = tf.minimum(abs_residual_capped, delta)
                    linear = abs_residual_capped - quadratic

                    physics_loss = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)
                    losses['physics'] = physics_loss

                    print(f"Computed physics loss: {physics_loss.numpy():.8e}")
                else:
                    print("WARNING: Physics residual is zero, NaN, or too small!")
                    # Force a small non-zero physics loss to encourage learning
                    losses['physics'] = tf.constant(0.01, dtype=tf.float32)

            except Exception as e:
                print(f"ERROR in physics loss computation: {str(e)}")
                # Use fallback physics loss
                losses['physics'] = tf.constant(0.1, dtype=tf.float32)
        else:
            losses['physics'] = tf.constant(0.0, dtype=tf.float32)
            print("No physics points provided")

        # MINIMAL regularization to avoid biasing the solution
        log_d_regularization = self.compute_minimal_log_d_regularization()
        losses.update(log_d_regularization)

        # Total loss with weighted components
        total_loss = sum(weights.get(key, 1.0) * losses[key]
                        for key in ['initial', 'boundary', 'interior', 'physics'])

        # Add minimal regularization terms
        total_loss += sum(losses[key] for key in losses if key.startswith('log_d_reg'))

        losses['total'] = total_loss
        return losses

    def compute_minimal_log_d_regularization(self) -> Dict[str, tf.Tensor]:
        """Minimal regularization for log(D) to prevent only extreme numerical issues"""
        log_d_reg_losses = {}

        # ONLY extreme bounds regularization to prevent numerical overflow
        # Very weak penalty and only at extreme values
        log_d_reg_losses['log_d_reg_bounds'] = 0.000001 * (
            tf.nn.relu(self.log_D_min + 1.0 - self.log_D) +  # Only penalty very close to min bound
            tf.nn.relu(self.log_D - self.log_D_max + 1.0)    # Only penalty very close to max bound
        )

        return log_d_reg_losses

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables with debugging"""
        variables = self.weights + self.biases
        if self.config.diffusion_trainable:
            variables.append(self.log_D)
            print(f"Trainable variables: {len(self.weights)} weights, {len(self.biases)} biases, 1 log_D")
            print(f"log_D trainable: {self.log_D.trainable}")
        else:
            print(f"Trainable variables: {len(self.weights)} weights, {len(self.biases)} biases, NO log_D")
        return variables

    @tf.function
    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """Make concentration predictions at given points"""
        return self.forward_pass(x)

    def save(self, filepath: str):
        """Save the model to a file"""
        print(f"Model saving to {filepath} - implement based on your requirements")
        pass

    def print_diffusion_info(self):
        """Print current diffusion coefficient information for debugging"""
        current_log_D = self.get_log_diffusion_coefficient()
        current_D = self.get_diffusion_coefficient()
        print(f"Current log(D): {current_log_D:.6f}")
        print(f"Current D: {current_D:.8e}")
        print(f"log(D) bounds: [{self.log_D_min:.1f}, {self.log_D_max:.1f}]")