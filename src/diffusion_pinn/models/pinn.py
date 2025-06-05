import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES

class DiffusionPINN(tf.Module):
    """Physics-Informed Neural Network for diffusion problems with logarithmic D parameterization"""

    def __init__(
        self,
        spatial_bounds: Dict[str, Tuple[float, float]],
        time_bounds: Tuple[float, float],
        initial_D: float = PINN_VARIABLES['initial_D'],
        config: DiffusionConfig = None,
        seed: int = None
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

        # LOGARITHMIC PARAMETERIZATION: Initialize log(D) instead of D
        # This spreads small D values across a larger numerical range
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

        # Define reasonable bounds for log(D)
        self.log_D_min = -16.0   # Corresponds to D ≈ 1e-7
        self.log_D_max = -4.0    # Corresponds to D ≈ 0.018

        print(f"Log(D) bounds: [{self.log_D_min:.1f}, {self.log_D_max:.1f}]")
        print(f"Corresponding D bounds: [{np.exp(self.log_D_min):.2e}, {np.exp(self.log_D_max):.2e}]")

        # Store loss weights from variables
        self.loss_weights = PINN_VARIABLES['loss_weights']

        # Initialize tolerances for condition identification
        self.boundary_tol = 1e-6
        self.initial_tol = 1e-6

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
        """Initialize neural network parameters (keeping v0.2.14 style)"""
        # Full architecture including input (3: x,y,t) and output (1: concentration)
        architecture = [3] + self.config.hidden_layers + [1]

        self.weights = []
        self.biases = []

        # Initialize weights and biases (v0.2.14 style)
        for i in range(len(architecture)-1):
            input_dim, output_dim = architecture[i], architecture[i+1]

            # Weight initialization (keeping v0.2.14 approach)
            if self.config.initialization == 'glorot':
                std_dv = np.sqrt(2.0 / (input_dim + output_dim))
            else:  # He initialization
                std_dv = np.sqrt(2.0 / input_dim)

            # Add seed support for reproducibility
            if self.seed is not None:
                weight_seed = self.seed + i
                w = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32, seed=weight_seed) * std_dv,
                    trainable=True,
                    name=f'w{i+1}'
                )
                bias_seed = self.seed + 100 + i
                b = tf.Variable(
                    tf.random.normal([output_dim], dtype=tf.float32, seed=bias_seed) * 0.01,
                    trainable=True,
                    name=f'b{i+1}'
                )
            else:
                w = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32) * std_dv,
                    trainable=True,
                    name=f'w{i+1}'
                )
                b = tf.Variable(
                    tf.zeros([output_dim], dtype=tf.float32),
                    trainable=True,
                    name=f'b{i+1}'
                )

            self.weights.append(w)
            self.biases.append(b)

    def _normalize_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Normalize inputs to [-1, 1] (keeping v0.2.14 approach)"""
        return 2.0 * (tf.cast(x, tf.float32) - self.lb) / (self.ub - self.lb) - 1.0

    @tf.function
    def forward_pass(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through the network (keeping v0.2.14 approach)"""
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

        # Option for non-negative output (uncomment if needed)
        # return tf.nn.softplus(output)
        return output

    def identify_condition_points(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Separate points into initial and boundary conditions (keeping v0.2.14 approach)"""
        t = x[:, 2]
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        # Initial condition mask (t = t_min)
        ic_mask = tf.abs(t - self.t_bounds[0]) < self.initial_tol

        # Boundary condition masks (x = x_min/max or y = y_min/max)
        bc_x_mask = tf.logical_or(
            tf.abs(x_coord - self.x_bounds[0]) < self.boundary_tol,
            tf.abs(x_coord - self.x_bounds[1]) < self.boundary_tol
        )
        bc_y_mask = tf.logical_or(
            tf.abs(y_coord - self.y_bounds[0]) < self.boundary_tol,
            tf.abs(y_coord - self.y_bounds[1]) < self.boundary_tol
        )
        bc_mask = tf.logical_or(bc_x_mask, bc_y_mask)

        # Interior points - all points that are not boundary or initial
        interior_mask = tf.logical_not(tf.logical_or(ic_mask, bc_mask))

        # Ensure masks don't overlap
        bc_mask = tf.logical_and(bc_mask, tf.logical_not(ic_mask))

        return {
            'initial': x[ic_mask],
            'boundary': x[bc_mask],
            'interior': x[interior_mask]
        }

    @tf.function
    def compute_single_batch_residual(self, x_batch: tf.Tensor) -> tf.Tensor:
        """Compute PDE residual for a single batch with logarithmic D"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_batch)
            # Calculate function value
            c = self.forward_pass(x_batch)

            # First derivatives - get all at once
            grad = tape.gradient(c, x_batch)

            # Extract individual components with proper reshaping
            dc_dx = tf.reshape(grad[:, 0], (-1, 1))
            dc_dy = tf.reshape(grad[:, 1], (-1, 1))
            dc_dt = tf.reshape(grad[:, 2], (-1, 1))

        # Second derivatives
        d2c_dx2 = tape.gradient(dc_dx, x_batch)[:, 0:1]
        d2c_dy2 = tape.gradient(dc_dy, x_batch)[:, 1:2]

        # Cleanup
        del tape

        # LOGARITHMIC PARAMETERIZATION: Convert log(D) to D with soft constraints
        # Apply soft bounds to prevent extreme values
        constrained_log_D = tf.clip_by_value(self.log_D, self.log_D_min + 1.0, self.log_D_max - 1.0)
        D_value = tf.exp(constrained_log_D)

        # Enhanced numerical stability (keeping v0.2.14 approach)
        laplacian = d2c_dx2 + d2c_dy2
        # Apply an outlier filter to the Laplacian for numerical stability
        laplacian_mean = tf.reduce_mean(tf.abs(laplacian))
        laplacian_filtered = tf.where(
            tf.abs(laplacian) > 100.0 * laplacian_mean,
            tf.sign(laplacian) * 100.0 * laplacian_mean,
            laplacian
        )

        return dc_dt - D_value * laplacian_filtered

    def compute_pde_residual(self, x_f: tf.Tensor) -> tf.Tensor:
        """Compute PDE residual with batching for memory efficiency (keeping v0.2.14 approach)"""
        # For small inputs, just compute directly
        if tf.shape(x_f)[0] <= 1000:
            return self.compute_single_batch_residual(x_f)

        # For larger inputs, process in batches
        batch_size = 1000
        num_points = tf.shape(x_f)[0]
        num_batches = (num_points - 1) // batch_size + 1

        residuals = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, num_points)
            x_batch = x_f[start_idx:end_idx]
            batch_residual = self.compute_single_batch_residual(x_batch)
            residuals.append(batch_residual)

        # Combine all batch residuals
        return tf.concat(residuals, axis=0)

    def loss_fn(self, x_data: tf.Tensor, c_data: tf.Tensor,
                x_physics: tf.Tensor = None, weights: Dict[str, float] = None) -> Dict[str, tf.Tensor]:
        """
        Compute loss with separated components (keeping v0.2.14 approach but with log D regularization)
        """
        if weights is None:
            weights = self.loss_weights

        # Separate points by condition type
        condition_points = self.identify_condition_points(x_data)
        losses = {}

        # Compute losses for each condition type
        for condition_type in ['initial', 'boundary', 'interior']:
            points = condition_points[condition_type]
            if points.shape[0] > 0:
                # Create mask by comparing coordinates
                mask = tf.zeros(x_data.shape[0], dtype=tf.bool)
                for point in points:
                    point_match = tf.reduce_all(tf.abs(x_data - point) < self.boundary_tol, axis=1)
                    mask = tf.logical_or(mask, point_match)

                if tf.reduce_any(mask):
                    c_pred = self.forward_pass(tf.boolean_mask(x_data, mask))
                    c_true = tf.boolean_mask(c_data, mask)
                    losses[condition_type] = tf.reduce_mean(tf.square(c_pred - c_true))
                else:
                    losses[condition_type] = tf.constant(0.0, dtype=tf.float32)
            else:
                losses[condition_type] = tf.constant(0.0, dtype=tf.float32)

        # Physics loss (keeping v0.2.14 Huber loss approach)
        if self.config.use_physics_loss and x_physics is not None and x_physics.shape[0] > 0:
            pde_residual = self.compute_pde_residual(x_physics)

            # Apply Huber loss for PDE residual to reduce sensitivity to outliers
            delta = 1.0
            abs_residual = tf.abs(pde_residual)
            quadratic = tf.minimum(abs_residual, delta)
            linear = abs_residual - quadratic
            physics_loss = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)

            losses['physics'] = physics_loss
        else:
            losses['physics'] = tf.constant(0.0, dtype=tf.float32)

        # Total loss with regularization for diffusion coefficient
        total_loss = sum(weights.get(key, 1.0) * losses[key] for key in losses.keys())

        # UPDATED: Logarithmic D regularization (replaces v0.2.14 D regularization)
        # Add light regularization to keep log(D) in reasonable range
        log_d_reg = 0.001 * tf.nn.relu(self.log_D_min + 2.0 - self.log_D) + \
                    0.001 * tf.nn.relu(self.log_D - self.log_D_max + 2.0)

        total_loss = total_loss + log_d_reg
        losses['log_d_regularization'] = log_d_reg
        losses['total'] = total_loss

        return losses

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables"""
        variables = self.weights + self.biases
        if self.config.diffusion_trainable:
            variables.append(self.log_D)  # Note: now using log_D
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