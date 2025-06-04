import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES

class DiffusionPINN(tf.Module):
    """Physics-Informed Neural Network for diffusion problems with hybrid approach"""

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

        # HYBRID: Use logarithmic parameterization but with simpler bounds
        initial_D_value = max(initial_D, 1e-8)
        initial_log_D = np.log(initial_D_value)

        print(f"Initial D: {initial_D_value:.8e}")
        print(f"Initial log(D): {initial_log_D:.6f}")

        # Store log(D) as the trainable parameter with reasonable bounds
        self.log_D = tf.Variable(
            initial_log_D,
            dtype=tf.float32,
            trainable=self.config.diffusion_trainable,
            name='log_diffusion_coefficient'
        )

        # Simpler bounds than V0.2.22 - wider range but still reasonable
        self.log_D_min = -20.0   # Corresponds to D ≈ 2e-9
        self.log_D_max = -4.0    # Corresponds to D ≈ 0.018

        print(f"log(D) bounds: [{self.log_D_min:.1f}, {self.log_D_max:.1f}]")
        print(f"Corresponding D bounds: [{np.exp(self.log_D_min):.2e}, {np.exp(self.log_D_max):.2e}]")

        # Store loss weights from variables
        self.loss_weights = PINN_VARIABLES['loss_weights']

        # Tolerances for condition identification - keep from V0.2.22
        self.boundary_tol = 1e-6
        self.initial_tol = 1e-6

        # Build network architecture with enhanced initialization
        self._build_network()

    def get_diffusion_coefficient(self) -> float:
        """Get the current estimate of the diffusion coefficient"""
        # Apply soft bounds before conversion
        bounded_log_D = tf.clip_by_value(self.log_D, self.log_D_min, self.log_D_max)
        D_value = tf.exp(bounded_log_D).numpy()
        return float(D_value)

    def get_log_diffusion_coefficient(self) -> float:
        """Get the current log(D) value for debugging"""
        return float(self.log_D.numpy())

    def _build_network(self):
        """Initialize neural network parameters with enhanced initialization from V0.2.22"""
        architecture = [3] + self.config.hidden_layers + [1]

        self.weights = []
        self.biases = []

        for i in range(len(architecture)-1):
            input_dim, output_dim = architecture[i], architecture[i+1]

            # Enhanced initialization based on layer position
            if i == 0:  # Input layer
                std_dv = 0.1 / np.sqrt(input_dim)
            elif i == len(architecture) - 2:  # Output layer
                std_dv = 0.01 / np.sqrt(input_dim)
            else:  # Hidden layers
                if self.config.activation == 'tanh':
                    std_dv = np.sqrt(2.0 / (input_dim + output_dim))
                else:
                    std_dv = np.sqrt(2.0 / input_dim)

            # Initialize weights with seed control
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

            # Initialize biases
            if i < len(architecture) - 2:
                # Small positive bias for hidden layers
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
        """Enhanced input normalization from V0.2.22"""
        x_float = tf.cast(x, tf.float32)
        range_tensor = self.ub - self.lb
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        safe_range = tf.maximum(range_tensor, epsilon)
        normalized_01 = (x_float - self.lb) / safe_range
        normalized_01_clipped = tf.clip_by_value(normalized_01, 0.0, 1.0)
        normalized_final = 2.0 * normalized_01_clipped - 1.0
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

        output = tf.matmul(H, self.weights[-1]) + self.biases[-1]
        return output

    def identify_condition_points(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Separate points into initial and boundary conditions - simplified from V0.2.14"""
        t = x[:, 2]
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        # Initial condition mask (t = t_min)
        ic_mask = tf.abs(t - self.t_bounds[0]) < self.initial_tol

        # Boundary condition masks
        bc_x_mask = tf.logical_or(
            tf.abs(x_coord - self.x_bounds[0]) < self.boundary_tol,
            tf.abs(x_coord - self.x_bounds[1]) < self.boundary_tol
        )
        bc_y_mask = tf.logical_or(
            tf.abs(y_coord - self.y_bounds[0]) < self.boundary_tol,
            tf.abs(y_coord - self.y_bounds[1]) < self.boundary_tol
        )
        bc_mask = tf.logical_or(bc_x_mask, bc_y_mask)

        # Interior points - not boundary or initial
        interior_mask = tf.logical_not(tf.logical_or(ic_mask, bc_mask))

        # Ensure boundary mask excludes initial condition points
        bc_mask = tf.logical_and(bc_mask, tf.logical_not(ic_mask))

        return {
            'initial': x[ic_mask],
            'boundary': x[bc_mask],
            'interior': x[interior_mask]
        }

    @tf.function
    def compute_single_batch_residual(self, x_batch: tf.Tensor) -> tf.Tensor:
        """Compute PDE residual with logarithmic D but simpler approach"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_batch)
            c = self.forward_pass(x_batch)
            grad = tape.gradient(c, x_batch)
            dc_dx = tf.reshape(grad[:, 0], (-1, 1))
            dc_dy = tf.reshape(grad[:, 1], (-1, 1))
            dc_dt = tf.reshape(grad[:, 2], (-1, 1))

        # Second derivatives
        d2c_dx2 = tape.gradient(dc_dx, x_batch)[:, 0:1]
        d2c_dy2 = tape.gradient(dc_dy, x_batch)[:, 1:2]
        del tape

        # Convert log(D) to D with soft bounds
        bounded_log_D = tf.clip_by_value(self.log_D, self.log_D_min, self.log_D_max)
        D_value = tf.exp(bounded_log_D)

        # Simple outlier filtering from V0.2.14
        laplacian = d2c_dx2 + d2c_dy2
        laplacian_mean = tf.reduce_mean(tf.abs(laplacian))
        laplacian_filtered = tf.where(
            tf.abs(laplacian) > 100.0 * laplacian_mean,
            tf.sign(laplacian) * 100.0 * laplacian_mean,
            laplacian
        )

        return dc_dt - D_value * laplacian_filtered

    def compute_pde_residual(self, x_f: tf.Tensor) -> tf.Tensor:
        """Compute PDE residual with batching"""
        if tf.shape(x_f)[0] <= 1000:
            return self.compute_single_batch_residual(x_f)

        # Process in batches
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

        return tf.concat(residuals, axis=0)

    def loss_fn(self, x_data: tf.Tensor, c_data: tf.Tensor,
                x_physics: tf.Tensor = None, weights: Dict[str, float] = None) -> Dict[str, tf.Tensor]:
        """
        HYBRID: Use V0.2.14's simpler loss structure but with log(D) parameterization
        """
        if weights is None:
            weights = self.loss_weights

        # Separate points by condition type - simplified approach
        condition_points = self.identify_condition_points(x_data)
        losses = {}

        # Compute losses for each condition type using V0.2.14's approach
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

        # Physics loss with Huber loss from V0.2.14
        if self.config.use_physics_loss and x_physics is not None and x_physics.shape[0] > 0:
            pde_residual = self.compute_pde_residual(x_physics)
            delta = 1.0
            abs_residual = tf.abs(pde_residual)
            quadratic = tf.minimum(abs_residual, delta)
            linear = abs_residual - quadratic
            physics_loss = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)
            losses['physics'] = physics_loss
        else:
            losses['physics'] = tf.constant(0.0, dtype=tf.float32)

        # HYBRID: Simple log(D) regularization (not the complex V0.2.22 version)
        # Keep log(D) in reasonable range but don't bias toward specific values
        log_d_penalty = 0.001 * (
            tf.nn.relu(self.log_D_min + 2.0 - self.log_D) +  # Penalty near lower bound
            tf.nn.relu(self.log_D - self.log_D_max + 2.0)    # Penalty near upper bound
        )

        # Total loss
        total_loss = sum(weights.get(key, 1.0) * losses[key] for key in losses.keys())
        total_loss = total_loss + log_d_penalty

        losses['log_d_regularization'] = log_d_penalty
        losses['total'] = total_loss

        return losses

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables"""
        variables = self.weights + self.biases
        if self.config.diffusion_trainable:
            variables.append(self.log_D)
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