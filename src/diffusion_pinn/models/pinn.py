import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig

class DiffusionPINN(tf.Module):
    """Physics-Informed Neural Network for diffusion problems"""

    def __init__(
        self,
        spatial_bounds: Dict[str, Tuple[float, float]],
        time_bounds: Tuple[float, float],
        initial_D: float = 1.0,
        config: DiffusionConfig = None
    ):
        """
        Initialize the Diffusion PINN

        Args:
            spatial_bounds: Dictionary with 'x' and 'y' bounds as tuples
            time_bounds: Tuple of (t_min, t_max)
            initial_D: Initial guess for diffusion coefficient
            config: Network configuration
        """
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

        # Initialize diffusion coefficient
        self.D = tf.Variable(initial_D, dtype=tf.float32,
                           trainable=self.config.diffusion_trainable,
                           name='diffusion_coefficient')

        # Store loss weights
        self.loss_weights = {
            'initial': 1.0,
            'boundary': 1.0,
            'interior': 10.0,
            'physics': 5.0
        }

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
        return tf.matmul(H, self.weights[-1]) + self.biases[-1]

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
    def compute_pde_residual(self, x_f: tf.Tensor) -> tf.Tensor:
        """Compute PDE residual"""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_f)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x_f)
                c = self.forward_pass(x_f)

            dc_dxyt = tape1.gradient(c, x_f)
            dc_dt = dc_dxyt[..., 2:3]
            dc_dx = dc_dxyt[..., 0:1]
            dc_dy = dc_dxyt[..., 1:2]

        d2c_dx2 = tape2.gradient(dc_dx, x_f)[..., 0:1]
        d2c_dy2 = tape2.gradient(dc_dy, x_f)[..., 1:2]

        del tape1, tape2

        return dc_dt - self.D * (d2c_dx2 + d2c_dy2)

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
        if self.config.use_physics_loss and x_physics is not None:
            pde_residual = self.compute_pde_residual(x_physics)
            losses['physics'] = tf.reduce_mean(tf.square(pde_residual))
        else:
            losses['physics'] = tf.constant(0.0, dtype=tf.float32)

        # Total loss
        total_loss = sum(weights.get(key, 1.0) * losses[key] for key in losses.keys())
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
