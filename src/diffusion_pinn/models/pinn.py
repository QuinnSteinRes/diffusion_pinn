import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES

class DiffusionPINN(tf.Module):
    """
    Physics-Informed Neural Network for diffusion problems

    Clean structure optimized for labeled point sets from processor:
    1. Network Architecture
    2. PDE Physics
    3. Loss Computation (using labeled data)
    4. Parameter Management
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

        # Store domain bounds (in original physical units)
        self.x_bounds = spatial_bounds['x']
        self.y_bounds = spatial_bounds['y']
        self.t_bounds = time_bounds

        # Create normalization bounds as tensors (for neural network input)
        self.lb = tf.constant([self.x_bounds[0], self.y_bounds[0], self.t_bounds[0]], dtype=tf.float32)
        self.ub = tf.constant([self.x_bounds[1], self.y_bounds[1], self.t_bounds[1]], dtype=tf.float32)

        # Initialize diffusion parameter
        self._setup_diffusion_parameter(initial_D)

        # Store loss weights
        self.loss_weights = PINN_VARIABLES['loss_weights']

        # Build network architecture
        self._build_network()

        print(f"PINN initialized - D: {self.get_diffusion_coefficient():.2e}")
        print(f"Domain: x={self.x_bounds}, y={self.y_bounds}, t={self.t_bounds}")

    def _setup_diffusion_parameter(self, initial_D: float):
        """Setup trainable diffusion parameter"""
        # Ensure positive initial value
        initial_D_value = max(initial_D, 1e-8)

        # Create trainable D parameter
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
        # Network architecture: [3] + hidden_layers + [1]
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

            # Create weights with reproducible initialization
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
        Normalize input coordinates to [-1, 1] range for neural network

        Args:
            input_coords: [N, 3] tensor containing [x, y, t] in original units
        """
        return 2.0 * (tf.cast(input_coords, tf.float32) - self.lb) / (self.ub - self.lb) - 1.0

    @tf.function
    def forward_pass(self, input_coords: tf.Tensor) -> tf.Tensor:
        """Forward pass through the neural network"""
        # Normalize inputs to [-1, 1] for neural network
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
    # 2. PDE PHYSICS
    # ===================================================================

    def compute_pde_residual(self, physics_coords: tf.Tensor) -> tf.Tensor:
        """
        Compute PDE residual for diffusion equation: du/dt = D * ∇²u

        Args:
            physics_coords: Collocation points [N, 3] (x, y, t)

        Returns:
            PDE residual [N, 1]
        """
        # Handle large inputs with batching for stability
        if tf.shape(physics_coords)[0] <= 1000:
            return self._compute_single_batch_residual(physics_coords)

        # Process in batches for memory efficiency
        batch_size = 1000
        num_points = tf.shape(physics_coords)[0]
        residuals = []

        for i in range(0, num_points, batch_size):
            end_idx = tf.minimum(i + batch_size, num_points)
            coords_batch = physics_coords[i:end_idx]
            batch_residual = self._compute_single_batch_residual(coords_batch)
            residuals.append(batch_residual)

        return tf.concat(residuals, axis=0)

    def _compute_single_batch_residual(self, coords_batch: tf.Tensor) -> tf.Tensor:
        """Compute PDE residual for a single batch with proper derivative handling"""
        coords_batch = tf.convert_to_tensor(coords_batch, dtype=tf.float32)

        # Compute derivatives using nested GradientTape
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

        # Apply numerical stability to prevent extreme outliers
        laplacian_mean = tf.reduce_mean(tf.abs(laplacian))
        laplacian_stable = tf.where(
            tf.abs(laplacian) > 100.0 * laplacian_mean,
            tf.sign(laplacian) * 100.0 * laplacian_mean,
            laplacian
        )

        residual = du_dt - D * laplacian_stable

        return residual

    # ===================================================================
    # 3. LOSS COMPUTATION (using labeled point sets)
    # ===================================================================

    def compute_boundary_loss(self, boundary_coords: tf.Tensor, boundary_values: tf.Tensor) -> tf.Tensor:
        """
        Compute loss for boundary/initial condition points

        Args:
            boundary_coords: Boundary coordinates [N, 3]
            boundary_values: True boundary values [N, 1]

        Returns:
            Boundary loss (scalar)
        """
        if boundary_coords.shape[0] == 0:
            return tf.constant(0.0, dtype=tf.float32)

        # Get predictions at boundary points
        boundary_pred = self.forward_pass(boundary_coords)

        # MSE loss
        boundary_loss = tf.reduce_mean(tf.square(boundary_pred - boundary_values))

        return boundary_loss

    def compute_interior_loss(self, interior_coords: tf.Tensor, interior_values: tf.Tensor) -> tf.Tensor:
        """
        Compute loss for interior supervision points

        Args:
            interior_coords: Interior coordinates [N, 3]
            interior_values: True interior values [N, 1]

        Returns:
            Interior loss (scalar)
        """
        if interior_coords.shape[0] == 0:
            return tf.constant(0.0, dtype=tf.float32)

        # Get predictions at interior points
        interior_pred = self.forward_pass(interior_coords)

        # MSE loss
        interior_loss = tf.reduce_mean(tf.square(interior_pred - interior_values))

        return interior_loss

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

    def loss_fn(self, labeled_data: Dict) -> Dict[str, tf.Tensor]:
        """
        Main loss function using labeled point sets - much cleaner than before!

        Args:
            labeled_data: Dictionary containing:
                'boundary': (coords, values) - boundary/initial conditions
                'interior': (coords, values) - interior supervision points
                'physics': coords - physics collocation points

        Returns:
            Dictionary containing all loss components and total loss
        """
        losses = {}

        # Extract labeled data
        boundary_coords, boundary_values = labeled_data['boundary']
        interior_coords, interior_values = labeled_data['interior']
        physics_coords = labeled_data['physics']

        # Convert to tensors if not already
        boundary_coords = tf.convert_to_tensor(boundary_coords, dtype=tf.float32)
        boundary_values = tf.convert_to_tensor(boundary_values, dtype=tf.float32)
        interior_coords = tf.convert_to_tensor(interior_coords, dtype=tf.float32)
        interior_values = tf.convert_to_tensor(interior_values, dtype=tf.float32)
        physics_coords = tf.convert_to_tensor(physics_coords, dtype=tf.float32)

        # Compute individual losses - no point type guessing needed!
        losses['boundary'] = self.compute_boundary_loss(boundary_coords, boundary_values)
        losses['interior'] = self.compute_interior_loss(interior_coords, interior_values)

        # Physics loss (if enabled)
        if self.config.use_physics_loss:
            losses['physics'] = self.compute_physics_loss(physics_coords)
        else:
            losses['physics'] = tf.constant(0.0, dtype=tf.float32)

        # Combine with weights
        total_loss = (
            self.loss_weights.get('boundary', 1.0) * losses['boundary'] +
            self.loss_weights.get('interior', 1.0) * losses['interior'] +
            self.loss_weights.get('physics', 1.0) * losses['physics']
        )

        losses['total'] = total_loss

        return losses

    # ===================================================================
    # 4. PARAMETER MANAGEMENT
    # ===================================================================

    def apply_parameter_constraints(self):
        """Apply constraints to diffusion parameter"""
        # Ensure diffusion coefficient stays positive
        if self.D.numpy() < 0:
            self.D.assign(tf.abs(self.D))

    def get_diffusion_coefficient(self) -> float:
        """Get current diffusion coefficient"""
        return float(tf.abs(self.D).numpy())

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable parameters"""
        variables = self.weights + self.biases
        if self.config.diffusion_trainable:
            variables.append(self.D)
        return variables

    # ===================================================================
    # 5. UTILITIES
    # ===================================================================

    @tf.function
    def predict(self, input_coords: tf.Tensor) -> tf.Tensor:
        """Make predictions at given coordinates"""
        return self.forward_pass(input_coords)

    def print_diagnostics(self):
        """Print current model diagnostics"""
        current_D = self.get_diffusion_coefficient()

        print(f"\nPINN Diagnostics:")
        print(f"  Diffusion coefficient: {current_D:.2e}")
        print(f"  Network architecture: {[3] + self.config.hidden_layers + [1]}")
        print(f"  Activation: {self.config.activation}")
        print(f"  Trainable parameters: {len(self.get_trainable_variables())}")
        print(f"  Loss weights: {self.loss_weights}")

    def save_model(self, filepath: str):
        """Save model weights and parameters"""
        # Save diffusion coefficient
        D_value = self.get_diffusion_coefficient()

        # Save network weights (implement based on your requirements)
        weights_dict = {
            'diffusion_coefficient': D_value,
            'weights': [w.numpy() for w in self.weights],
            'biases': [b.numpy() for b in self.biases],
            'config': {
                'hidden_layers': self.config.hidden_layers,
                'activation': self.config.activation,
                'spatial_bounds': self.x_bounds + self.y_bounds,
                'time_bounds': self.t_bounds
            }
        }

        # Save to file (use numpy, pickle, or HDF5 as preferred)
        np.savez(filepath, **weights_dict)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load model from saved file"""
        # Load from file and reconstruct model
        # Implementation depends on your save format
        data = np.load(filepath, allow_pickle=True)
        print(f"Model loaded from {filepath}")
        # Return reconstructed model instance
        pass