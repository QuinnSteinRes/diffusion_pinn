import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES

class DiffusionPINN(tf.Module):
    """
    Physics-Informed Neural Network for diffusion problems

    Clean structure with separated concerns:
    1. Network Architecture
    2. PDE Physics
    3. Loss Computation
    4. Parameter Constraints
    5. Utilities
    """

    def __init__(
        self,
        spatial_bounds: Dict[str, Tuple[float, float]],
        time_bounds: Tuple[float, float],
        initial_D: float = PINN_VARIABLES['initial_D'],
        config: DiffusionConfig = None,
        seed: int = None
    ):
        super().__init__()

        # Store configuration
        self.config = config or DiffusionConfig()
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Store domain bounds
        self.x_bounds = spatial_bounds['x']
        self.y_bounds = spatial_bounds['y']
        self.t_bounds = time_bounds

        # Create normalization bounds as tensors
        self.lb = tf.constant([self.x_bounds[0], self.y_bounds[0], self.t_bounds[0]], dtype=tf.float32)
        self.ub = tf.constant([self.x_bounds[1], self.y_bounds[1], self.t_bounds[1]], dtype=tf.float32)

        # Initialize diffusion parameter (direct D optimization)
        self._setup_diffusion_parameter(initial_D)

        # Store loss weights
        self.loss_weights = PINN_VARIABLES['loss_weights']

        # Build network architecture
        self._build_network()

        print(f"PINN initialized - D: {self.get_diffusion_coefficient():.2e}")

    def _setup_diffusion_parameter(self, initial_D: float):
        """Setup diffusion parameter - direct D optimization"""
        # Ensure positive initial value
        initial_D_value = max(initial_D, 1e-8)

        # Create trainable D parameter directly
        self.D = tf.Variable(
            initial_D_value,
            dtype=tf.float32,
            trainable=self.config.diffusion_trainable,
            name='diffusion_coefficient'
        )

    # ===================================================================
    # 1. NETWORK ARCHITECTURE
    # ===================================================================

    def _build_network(self):
        """Initialize neural network weights and biases"""
        # Network architecture: [input_dim] + hidden_layers + [output_dim]
        architecture = [3] + self.config.hidden_layers + [1]

        self.weights = []
        self.biases = []

        # Initialize each layer
        for i in range(len(architecture) - 1):
            input_dim, output_dim = architecture[i], architecture[i + 1]

            # Weight initialization
            if self.config.initialization == 'glorot':
                std_dev = np.sqrt(2.0 / (input_dim + output_dim))
            else:  # He initialization
                std_dev = np.sqrt(2.0 / input_dim)

            # Create weights with optional seed for reproducibility
            if self.seed is not None:
                weight_seed = self.seed + i
                bias_seed = self.seed + 100 + i

                weight = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32, seed=weight_seed) * std_dev,
                    trainable=True, name=f'weight_{i+1}'
                )
                bias = tf.Variable(
                    tf.random.normal([output_dim], dtype=tf.float32, seed=bias_seed) * 0.01,
                    trainable=True, name=f'bias_{i+1}'
                )
            else:
                weight = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32) * std_dev,
                    trainable=True, name=f'weight_{i+1}'
                )
                bias = tf.Variable(
                    tf.zeros([output_dim], dtype=tf.float32),
                    trainable=True, name=f'bias_{i+1}'
                )

            self.weights.append(weight)
            self.biases.append(bias)

    def _normalize_inputs(self, input_coords: tf.Tensor) -> tf.Tensor:
        """
        Normalize input coordinates to [-1, 1] range

        Args:
            input_coords: [N, 3] tensor containing [x_spatial, y_spatial, t_temporal]
        """
        return 2.0 * (tf.cast(input_coords, tf.float32) - self.lb) / (self.ub - self.lb) - 1.0

    @tf.function
    def forward_pass(self, input_coords: tf.Tensor) -> tf.Tensor:
        """Forward pass through the neural network"""
        # Normalize inputs
        X = self._normalize_inputs(input_coords)

        # Pass through hidden layers
        H = X
        for i in range(len(self.weights) - 1):
            H = tf.matmul(H, self.weights[i]) + self.biases[i]

            # Apply activation function
            if self.config.activation == 'tanh':
                H = tf.tanh(H)
            elif self.config.activation == 'sin':
                H = tf.sin(H)
            else:  # relu
                H = tf.nn.relu(H)

        # Output layer (no activation)
        output = tf.matmul(H, self.weights[-1]) + self.biases[-1]

        return output

    # ===================================================================
    # 2. PDE PHYSICS - Your frequent modification area
    # ===================================================================
    def _compute_gradients_with_tape(self, tape: tf.GradientTape, coords: tf.Tensor, u: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Compute spatial and temporal derivatives using an existing tape"""
        # First derivatives
        grad = tape.gradient(u, coords)

        # Check if gradient computation failed
        if grad is None:
            raise ValueError("First gradient computation failed - check tensor connectivity")

        du_dx = tf.reshape(grad[:, 0], (-1, 1))
        du_dy = tf.reshape(grad[:, 1], (-1, 1))
        du_dt = tf.reshape(grad[:, 2], (-1, 1))

        # Second derivatives - check for None before indexing
        d2u_dx2_full = tape.gradient(du_dx, coords)
        d2u_dy2_full = tape.gradient(du_dy, coords)

        if d2u_dx2_full is None or d2u_dy2_full is None:
            raise ValueError("Second derivative computation failed - check tape persistence and connectivity")

        d2u_dx2 = d2u_dx2_full[:, 0:1]
        d2u_dy2 = d2u_dy2_full[:, 1:2]

        return {
            'du_dx': du_dx,
            'du_dy': du_dy,
            'du_dt': du_dt,
            'd2u_dx2': d2u_dx2,
            'd2u_dy2': d2u_dy2,
            'laplacian': d2u_dx2 + d2u_dy2
        }

    def _apply_diffusion_equation(self, gradients: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Apply the diffusion PDE: du/dt = D * ∇²u

        Args:
            gradients: Dictionary of computed derivatives

        Returns:
            PDE residual
        """
        # Get current diffusion coefficient (direct D)
        D = tf.abs(self.D)  # Ensure positive D

        # Apply numerical stability to Laplacian (prevent extreme outliers)
        laplacian = gradients['laplacian']
        laplacian_mean = tf.reduce_mean(tf.abs(laplacian))
        laplacian_stable = tf.where(
            tf.abs(laplacian) > 100.0 * laplacian_mean,
            tf.sign(laplacian) * 100.0 * laplacian_mean,
            laplacian
        )

        # PDE residual: du/dt - D * ∇²u = 0
        residual = gradients['du_dt'] - D * laplacian_stable

        return residual


    def compute_pde_residual(self, physics_coords: tf.Tensor) -> tf.Tensor:
        """
        Main PDE residual computation - modify this for different physics

        Args:
            physics_coords: Collocation points [N, 3] (x, y, t)

        Returns:
            PDE residual [N, 1]
        """
        # Handle large inputs with batching (for numerical stability)
        if tf.shape(physics_coords)[0] <= 1000:
            return self._compute_single_batch_residual(physics_coords)

        # Process in batches for memory efficiency
        batch_size = 1000
        num_points = tf.shape(physics_coords)[0]
        num_batches = (num_points - 1) // batch_size + 1

        residuals = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, num_points)
            coords_batch = physics_coords[start_idx:end_idx]

            batch_residual = self._compute_single_batch_residual(coords_batch)
            residuals.append(batch_residual)

        return tf.concat(residuals, axis=0)

    def _compute_single_batch_residual(self, coords_batch: tf.Tensor) -> tf.Tensor:
        """Compute PDE residual for a single batch with proper second derivative handling"""
        coords_batch = tf.convert_to_tensor(coords_batch, dtype=tf.float32)

        # Compute first derivatives
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords_batch)

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(coords_batch)
                u = self.forward_pass(coords_batch)

            # First derivatives
            grad = tape2.gradient(u, coords_batch)
            if grad is None:
                raise ValueError("First gradient computation failed")

            du_dx = grad[:, 0:1]
            du_dy = grad[:, 1:2]
            du_dt = grad[:, 2:3]

        # Second derivatives
        d2u_dx2 = tape1.gradient(du_dx, coords_batch)
        d2u_dy2 = tape1.gradient(du_dy, coords_batch)

        if d2u_dx2 is None or d2u_dy2 is None:
            raise ValueError("Second derivative computation failed")

        d2u_dx2 = d2u_dx2[:, 0:1]
        d2u_dy2 = d2u_dy2[:, 1:2]
        laplacian = d2u_dx2 + d2u_dy2

        # Clean up tapes
        del tape1, tape2

        # Apply diffusion equation: du/dt - D * laplacian = 0
        D = tf.abs(self.D)  # Ensure positive D
        residual = du_dt - D * laplacian

        return residual


    # ===================================================================
    # 3. LOSS COMPUTATION
    # ===================================================================

    def _identify_point_types(self, coords: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Identify point types (initial, boundary, interior) from coordinates

        Args:
            coords: Input coordinates [N, 3]

        Returns:
            Dictionary of boolean masks for each point type
        """
        t = coords[:, 2]
        x_coord = coords[:, 0]
        y_coord = coords[:, 1]

        # Tolerances for boundary detection
        boundary_tol = 1e-6
        initial_tol = 1e-6

        # Initial condition: t = t_min
        initial_mask = tf.abs(t - self.t_bounds[0]) < initial_tol

        # Boundary conditions: at domain boundaries
        boundary_x = tf.logical_or(
            tf.abs(x_coord - self.x_bounds[0]) < boundary_tol,
            tf.abs(x_coord - self.x_bounds[1]) < boundary_tol
        )
        boundary_y = tf.logical_or(
            tf.abs(y_coord - self.y_bounds[0]) < boundary_tol,
            tf.abs(y_coord - self.y_bounds[1]) < boundary_tol
        )
        boundary_mask = tf.logical_or(boundary_x, boundary_y)

        # Remove initial condition points from boundary mask
        boundary_mask = tf.logical_and(boundary_mask, tf.logical_not(initial_mask))

        # Interior points: everything else
        interior_mask = tf.logical_not(tf.logical_or(initial_mask, boundary_mask))

        return {
            'initial': initial_mask,
            'boundary': boundary_mask,
            'interior': interior_mask
        }

    def compute_data_loss(self, data_coords: tf.Tensor, u_data: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Compute loss for supervised data points

        Args:
            data_coords: Data coordinates [N, 3]
            u_data: True concentration values [N, 1]

        Returns:
            Dictionary of losses by point type
        """
        # Get predictions
        u_pred = self.forward_pass(data_coords)

        # Identify point types
        point_masks = self._identify_point_types(data_coords)

        # Compute loss for each point type
        losses = {}
        for point_type, mask in point_masks.items():
            if tf.reduce_any(mask):
                pred_masked = tf.boolean_mask(u_pred, mask)
                true_masked = tf.boolean_mask(u_data, mask)
                losses[point_type] = tf.reduce_mean(tf.square(pred_masked - true_masked))
            else:
                losses[point_type] = tf.constant(0.0, dtype=tf.float32)

        return losses

    def compute_physics_loss(self, physics_coords: tf.Tensor) -> tf.Tensor:
        """
        Compute physics-informed loss from PDE residual

        Args:
            physics_coords: Collocation points [N, 3]

        Returns:
            Physics loss (scalar)
        """
        if physics_coords.shape[0] == 0:
            return tf.constant(0.0, dtype=tf.float32)

        # Get PDE residual
        residual = self.compute_pde_residual(physics_coords)

        # Apply Huber loss for robustness against outliers
        delta = 1.0
        abs_residual = tf.abs(residual)
        quadratic = tf.minimum(abs_residual, delta)
        linear = abs_residual - quadratic

        huber_loss = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)

        return huber_loss

    def loss_fn(self, x_data: tf.Tensor, c_data: tf.Tensor,
            x_physics: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Main loss function - combines all loss components

        Args:
            x_data: Supervised data coordinates [N, 3]
            c_data: True concentration values [N, 1]
            x_physics: Physics collocation points [M, 3]

        Returns:
            Dictionary containing all loss components and total loss
        """
        losses = {}

        # Data fitting losses
        data_losses = self.compute_data_loss(x_data, c_data)
        losses.update(data_losses)

        # Physics loss
        if self.config.use_physics_loss and x_physics is not None:
            losses['physics'] = self.compute_physics_loss(x_physics)
        else:
            losses['physics'] = tf.constant(0.0, dtype=tf.float32)

        # Combine with weights
        total_loss = sum(
            self.loss_weights.get(key, 1.0) * loss
            for key, loss in losses.items()
        )

        # No regularization needed for direct D optimization
        losses['total'] = total_loss

        return losses

    # ===================================================================
    # 4. PARAMETER CONSTRAINTS
    # ===================================================================

    def apply_parameter_constraints(self):
        """Apply constraints to diffusion parameter"""
        # For direct D: ensure it stays positive
        if self.D.numpy() < 0:
            self.D.assign(tf.abs(self.D))

    def get_diffusion_coefficient(self) -> float:
        """Get current diffusion coefficient"""
        return float(tf.abs(self.D).numpy())

    def get_log_diffusion_coefficient(self) -> float:
        """Get current log diffusion coefficient (only for log D version)"""
        return float(tf.math.log(tf.abs(self.D)).numpy())

    # ===================================================================
    # 5. UTILITIES
    # ===================================================================

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable parameters"""
        variables = self.weights + self.biases
        if self.config.diffusion_trainable:
            variables.append(self.D)  # Direct D parameter
        return variables

    @tf.function
    def predict(self, input_coords: tf.Tensor) -> tf.Tensor:
        """Make predictions at given coordinates"""
        return self.forward_pass(input_coords)

    def print_diagnostics(self):
        """Print current model diagnostics"""
        current_D = self.get_diffusion_coefficient()

        print(f"Current D: {current_D:.2e}")
        print(f"Network architecture: {[3] + self.config.hidden_layers + [1]}")


    def save(self, filepath: str):
        """Save model (implement based on requirements)"""
        print(f"Model saving to {filepath} - implement based on your requirements")
        pass