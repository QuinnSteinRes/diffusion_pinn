import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES

class DiffusionPINN(tf.Module):
    """Physics-Informed Neural Network for diffusion problems"""
    def __init__(
        self,
        spatial_bounds: Dict[str, Tuple[float, float]],
        time_bounds: Tuple[float, float],
        initial_D: float = PINN_VARIABLES['initial_D'],
        config: DiffusionConfig = None
    ):
        super().__init__()
        self.config = config or DiffusionConfig()

        # Store bounds for normalization
        self.x_bounds = spatial_bounds['x']
        self.y_bounds = spatial_bounds['y']
        self.t_bounds = time_bounds

        # Create normalized bounds as tensors
        self.lb = tf.constant([self.x_bounds[0], self.y_bounds[0], self.t_bounds[0]],
                            dtype=tf.float32)
        self.ub = tf.constant([self.x_bounds[1], self.y_bounds[1], self.t_bounds[1]],
                            dtype=tf.float32)

        # Initialize diffusion coefficient with positivity constraint
        initial_D_value = max(initial_D, 1e-5)  # Ensure positive value
        self.D = tf.Variable(initial_D_value, dtype=tf.float32,
                           trainable=self.config.diffusion_trainable,
                           name='diffusion_coefficient',
                           constraint=lambda x: tf.clip_by_value(x, 1e-5, 1.0))  # Add constraint

        # Store loss weights from variables
        self.loss_weights = PINN_VARIABLES['loss_weights']

        # Initialize tolerances for condition identification
        self.boundary_tol = 1e-6
        self.initial_tol = 1e-6

        # Build network architecture
        self._build_network()

    def _build_network(self):
        """Initialize neural network parameters"""
        # Full architecture including input (3: x,y,t) and output (1: concentration)
        architecture = [3] + self.config.hidden_layers + [1]

        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(architecture)-1):
            input_dim, output_dim = architecture[i], architecture[i+1]

            # Weight initialization
            if self.config.initialization == 'glorot':
                std_dv = np.sqrt(2.0 / (input_dim + output_dim))
            else:  # He initialization
                std_dv = np.sqrt(2.0 / input_dim)

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
        """Normalize inputs to [-1, 1]"""
        #return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return 2.0 * (tf.cast(x, tf.float32) - self.lb) / (self.ub - self.lb) - 1.0


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

        # Apply softplus to ensure non-negative output (concentration is non-negative)
        output = tf.matmul(H, self.weights[-1]) + self.biases[-1]

        # Option for non-negative output (uncomment if needed)
        # return tf.nn.softplus(output)
        return output

    def identify_condition_points(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Separate points into initial and boundary conditions"""
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
        """Compute PDE residual for a single batch with improved numerical stability"""
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

        # Ensure diffusion coefficient is positive during computation
        D_value = tf.abs(self.D) + 1e-5

        # Enhanced numerical stability
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
        """Compute PDE residual with improved stability and consistent batch processing"""
        # Process in fixed-size batches for more consistency
        batch_size = 512  # Fixed size for consistent behavior
        total_points = tf.shape(x_f)[0]
        num_batches = (total_points - 1) // batch_size + 1

        residuals = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, total_points)
            x_batch = x_f[start_idx:end_idx]

            # Compute with numerical stability enhancements
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_batch)
                # Calculate function value
                c = self.forward_pass(x_batch)

                # First derivatives with intermediate normalization
                grad = tape.gradient(c, x_batch)
                # Apply softer gradient stabilization
                grad_norm = tf.reduce_mean(tf.abs(grad))
                grad_scale = tf.maximum(1.0, grad_norm / 10.0)  # Scale only if too large
                grad = grad / grad_scale

                # Extract components
                dc_dx = tf.reshape(grad[:, 0], (-1, 1))
                dc_dy = tf.reshape(grad[:, 1], (-1, 1))
                dc_dt = tf.reshape(grad[:, 2], (-1, 1))

            # Second derivatives with softer stabilization
            d2c_dx2 = tape.gradient(dc_dx, x_batch)[:, 0:1]
            d2c_dy2 = tape.gradient(dc_dy, x_batch)[:, 1:2]

            # Cleanup
            del tape

            # Robust laplacian calculation
            laplacian = d2c_dx2 + d2c_dy2
            # Apply Huber-like loss concept to the Laplacian itself
            laplacian_mean = tf.reduce_mean(tf.abs(laplacian))
            delta = 5.0 * laplacian_mean

            # Smooth capping for outliers
            laplacian_stabilized = tf.where(
                tf.abs(laplacian) > delta,
                delta * tf.tanh(laplacian / delta),
                laplacian
            )

            # Scale gradients back if we scaled them earlier
            if grad_scale > 1.0:
                dc_dt = dc_dt * grad_scale
                laplacian_stabilized = laplacian_stabilized * grad_scale

            # Ensure D is positive but not too small
            D_value = tf.maximum(1e-6, self.D)

            # Calculate residual with softer handling
            residual = dc_dt - D_value * laplacian_stabilized

            # Save for this batch
            residuals.append(residual)

        # Combine all batch residuals
        return tf.concat(residuals, axis=0)

    def loss_fn(self, x_data: tf.Tensor, c_data: tf.Tensor,
                x_physics: tf.Tensor = None, weights: Dict[str, float] = None) -> Dict[str, tf.Tensor]:
        """
        Compute loss with separated components

        Args:
            x_data: Input coordinates
            c_data: Target concentration values
            x_physics: Points for physics-informed loss
            weights: Optional dictionary of loss weights

        Returns:
            Dictionary of loss components
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

        # Physics loss
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

        # Add regularization to keep D in reasonable range
        d_reg = 0.01 * tf.square(tf.math.log(tf.abs(self.D) + 1e-6) - tf.math.log(0.01))
        total_loss = total_loss + d_reg
        losses['d_regularization'] = d_reg
        losses['total'] = total_loss

        return losses

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables"""
        variables = self.weights + self.biases
        if self.config.diffusion_trainable:
            variables.append(self.D)
        return variables

    @tf.function
    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """Make concentration predictions at given points"""
        return self.forward_pass(x)

    def get_diffusion_coefficient(self) -> float:
        """Get the current estimate of the diffusion coefficient"""
        return self.D.numpy()