import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict

from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES

class OpenSystemDiffusionPINN(tf.Module):
    """Physics-Informed Neural Network for open diffusion systems with boundary flux"""

    def __init__(
        self,
        spatial_bounds: Dict[str, Tuple[float, float]],
        time_bounds: Tuple[float, float],
        initial_D: float = PINN_VARIABLES['initial_D'],
        initial_k: float = 0.001,  # Initial boundary permeability
        c_external: float = 0.0,   # External concentration
        config: DiffusionConfig = None,
        seed: int = None,
    ):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.seed = seed

        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        # Store bounds
        self.x_bounds = spatial_bounds['x']
        self.y_bounds = spatial_bounds['y']
        self.t_bounds = time_bounds

        # Normalized bounds
        self.lb = tf.constant([self.x_bounds[0], self.y_bounds[0], self.t_bounds[0]], dtype=tf.float32)
        self.ub = tf.constant([self.x_bounds[1], self.y_bounds[1], self.t_bounds[1]], dtype=tf.float32)

        # CRITICAL: Now we learn THREE physical parameters
        # 1. Diffusion coefficient (log parameterization for stability)
        initial_D_value = max(initial_D, 1e-8)
        self.log_D = tf.Variable(
            np.log(initial_D_value),
            dtype=tf.float32,
            trainable=True,
            name='log_diffusion_coefficient'
        )

        # 2. Boundary permeability (how fast mass leaves boundaries)
        self.log_k = tf.Variable(
            np.log(max(initial_k, 1e-8)),
            dtype=tf.float32,
            trainable=True,
            name='log_boundary_permeability'
        )

        # 3. External concentration (might be trainable if unknown)
        self.c_external = tf.Variable(
            c_external,
            dtype=tf.float32,
            trainable=False,  # Usually fixed at 0
            name='external_concentration'
        )

        # Bounds for stability (much wider than before)
        self.log_D_min, self.log_D_max = -16.0, -6.0
        self.log_k_min, self.log_k_max = -20.0, -2.0

        print(f"Open System PINN initialized:")
        print(f"  Initial D: {initial_D_value:.2e}")
        print(f"  Initial k: {initial_k:.2e}")
        print(f"  External c: {c_external:.3f}")

        # Build network
        self._build_network()

        # Tolerances for boundary identification
        self.boundary_tol = 1e-6
        self.initial_tol = 1e-6

    def get_diffusion_coefficient(self) -> float:
        """Get current diffusion coefficient"""
        return float(tf.exp(self.log_D).numpy())

    def get_boundary_permeability(self) -> float:
        """Get current boundary permeability"""
        return float(tf.exp(self.log_k).numpy())

    def get_external_concentration(self) -> float:
        """Get external concentration"""
        return float(self.c_external.numpy())

    def _build_network(self):
        """Build neural network with improved initialization"""
        architecture = [3] + self.config.hidden_layers + [1]

        self.weights = []
        self.biases = []

        for i in range(len(architecture)-1):
            input_dim, output_dim = architecture[i], architecture[i+1]

            # Careful initialization for open system
            if i == 0:  # Input layer
                std_dv = 0.1 / np.sqrt(input_dim)
            elif i == len(architecture) - 2:  # Output layer
                std_dv = 0.01 / np.sqrt(input_dim)
            else:  # Hidden layers
                std_dv = np.sqrt(2.0 / (input_dim + output_dim)) if self.config.activation == 'tanh' else np.sqrt(2.0 / input_dim)

            # Initialize weights
            if self.seed is not None:
                w = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32, seed=self.seed + i) * std_dv,
                    trainable=True, name=f'w{i+1}'
                )
                b_init = tf.random.uniform([output_dim], minval=0.01, maxval=0.05, dtype=tf.float32, seed=self.seed + 100 + i) if i < len(architecture) - 2 else tf.zeros([output_dim], dtype=tf.float32)
            else:
                w = tf.Variable(
                    tf.random.normal([input_dim, output_dim], dtype=tf.float32) * std_dv,
                    trainable=True, name=f'w{i+1}'
                )
                b_init = tf.random.uniform([output_dim], minval=0.01, maxval=0.05, dtype=tf.float32) if i < len(architecture) - 2 else tf.zeros([output_dim], dtype=tf.float32)

            b = tf.Variable(b_init, trainable=True, name=f'b{i+1}')

            self.weights.append(w)
            self.biases.append(b)

    def _normalize_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Normalize inputs to [-1, 1]"""
        x_float = tf.cast(x, tf.float32)
        range_tensor = self.ub - self.lb
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        safe_range = tf.maximum(range_tensor, epsilon)
        normalized_01 = (x_float - self.lb) / safe_range
        normalized_01_clipped = tf.clip_by_value(normalized_01, 0.0, 1.0)
        return tf.clip_by_value(2.0 * normalized_01_clipped - 1.0, -1.0, 1.0)

    @tf.function
    def forward_pass(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through network"""
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

    def identify_boundary_points(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Identify boundary points by location"""
        x_coord, y_coord, t_coord = x[:, 0], x[:, 1], x[:, 2]

        # Define boundary masks
        x_min_mask = tf.abs(x_coord - self.x_bounds[0]) < self.boundary_tol
        x_max_mask = tf.abs(x_coord - self.x_bounds[1]) < self.boundary_tol
        y_min_mask = tf.abs(y_coord - self.y_bounds[0]) < self.boundary_tol
        y_max_mask = tf.abs(y_coord - self.y_bounds[1]) < self.boundary_tol

        # Initial condition mask
        initial_mask = tf.abs(t_coord - self.t_bounds[0]) < self.initial_tol

        # Boundary mask (any spatial boundary, not at initial time)
        boundary_mask = tf.logical_and(
            tf.logical_or(tf.logical_or(x_min_mask, x_max_mask), tf.logical_or(y_min_mask, y_max_mask)),
            tf.logical_not(initial_mask)
        )

        # Interior mask
        interior_mask = tf.logical_not(tf.logical_or(boundary_mask, initial_mask))

        return {
            'initial': x[initial_mask],
            'boundary': x[boundary_mask],
            'interior': x[interior_mask],
            'x_min': x[tf.logical_and(x_min_mask, tf.logical_not(initial_mask))],
            'x_max': x[tf.logical_and(x_max_mask, tf.logical_not(initial_mask))],
            'y_min': x[tf.logical_and(y_min_mask, tf.logical_not(initial_mask))],
            'y_max': x[tf.logical_and(y_max_mask, tf.logical_not(initial_mask))]
        }

    @tf.function
    def compute_interior_pde_residual(self, x_interior: tf.Tensor) -> tf.Tensor:
        """Standard diffusion PDE for interior points"""
        if tf.shape(x_interior)[0] == 0:
            return tf.constant(0.0, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_interior)
            c = self.forward_pass(x_interior)
            grad = tape.gradient(c, x_interior)
            dc_dx = grad[:, 0:1]
            dc_dy = grad[:, 1:2]
            dc_dt = grad[:, 2:3]

        d2c_dx2 = tape.gradient(dc_dx, x_interior)[:, 0:1]
        d2c_dy2 = tape.gradient(dc_dy, x_interior)[:, 1:2]
        del tape

        # Get diffusion coefficient
        D_value = tf.exp(tf.clip_by_value(self.log_D, self.log_D_min + 1.0, self.log_D_max - 1.0))

        # Standard diffusion equation in interior
        laplacian = d2c_dx2 + d2c_dy2
        return dc_dt - D_value * laplacian

    @tf.function
    def compute_boundary_flux_residual(self, x_boundary: tf.Tensor, boundary_type: str) -> tf.Tensor:
        """Robin boundary condition: -D(∂c/∂n) = k(c - c_ext)"""
        if tf.shape(x_boundary)[0] == 0:
            return tf.constant(0.0, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_boundary)
            c = self.forward_pass(x_boundary)
            grad = tape.gradient(c, x_boundary)

        # Normal gradients based on boundary type
        if boundary_type == 'x_min':
            normal_grad = -grad[:, 0:1]  # Outward normal is -x direction
        elif boundary_type == 'x_max':
            normal_grad = grad[:, 0:1]   # Outward normal is +x direction
        elif boundary_type == 'y_min':
            normal_grad = -grad[:, 1:2]  # Outward normal is -y direction
        elif boundary_type == 'y_max':
            normal_grad = grad[:, 1:2]   # Outward normal is +y direction
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")

        # Get parameters
        D_value = tf.exp(tf.clip_by_value(self.log_D, self.log_D_min + 1.0, self.log_D_max - 1.0))
        k_value = tf.exp(tf.clip_by_value(self.log_k, self.log_k_min + 1.0, self.log_k_max - 1.0))

        # Robin boundary condition: -D(∂c/∂n) = k(c - c_ext)
        diffusive_flux = -D_value * normal_grad
        convective_flux = k_value * (c - self.c_external)

        return diffusive_flux - convective_flux

    def loss_fn(self, x_data: tf.Tensor, c_data: tf.Tensor,
                x_physics: tf.Tensor = None, weights: Dict[str, float] = None) -> Dict[str, tf.Tensor]:
        """Open system loss function with proper boundary physics"""
        if weights is None:
            weights = {'initial': 10.0, 'boundary': 1.0, 'interior': 5.0}

        losses = {}

        # Separate points by type
        point_types = self.identify_boundary_points(x_data)

        # 1. Initial condition loss - fit to first image only
        initial_points = point_types['initial']
        if tf.shape(initial_points)[0] > 0:
            # Find data points corresponding to initial time
            initial_mask = tf.abs(x_data[:, 2] - self.t_bounds[0]) < self.initial_tol
            if tf.reduce_any(initial_mask):
                c_pred_initial = self.forward_pass(tf.boolean_mask(x_data, initial_mask))
                c_true_initial = tf.boolean_mask(c_data, initial_mask)
                losses['initial'] = tf.reduce_mean(tf.square(c_pred_initial - c_true_initial))
            else:
                losses['initial'] = tf.constant(0.0, dtype=tf.float32)
        else:
            losses['initial'] = tf.constant(0.0, dtype=tf.float32)

        # 2. Interior physics loss - standard diffusion PDE
        if x_physics is not None:
            physics_points = self.identify_boundary_points(x_physics)
            interior_residual = self.compute_interior_pde_residual(physics_points['interior'])
            losses['interior'] = tf.reduce_mean(tf.square(interior_residual))
        else:
            losses['interior'] = tf.constant(0.0, dtype=tf.float32)

        # 3. Boundary physics loss - Robin conditions (NOT data fitting)
        boundary_loss_total = tf.constant(0.0, dtype=tf.float32)
        boundary_types = ['x_min', 'x_max', 'y_min', 'y_max']

        if x_physics is not None:
            physics_boundaries = self.identify_boundary_points(x_physics)
            for boundary_type in boundary_types:
                boundary_points = physics_boundaries[boundary_type]
                if tf.shape(boundary_points)[0] > 0:
                    flux_residual = self.compute_boundary_flux_residual(boundary_points, boundary_type)
                    boundary_loss_total += tf.reduce_mean(tf.square(flux_residual))

        losses['boundary'] = boundary_loss_total

        # 4. Parameter regularization (minimal, just to prevent numerical issues)
        param_reg = 0.00001 * (
            tf.nn.relu(self.log_D_min + 2.0 - self.log_D) +
            tf.nn.relu(self.log_D - self.log_D_max + 2.0) +
            tf.nn.relu(self.log_k_min + 2.0 - self.log_k) +
            tf.nn.relu(self.log_k - self.log_k_max + 2.0)
        )
        losses['param_reg'] = param_reg

        # Total loss
        total_loss = (weights['initial'] * losses['initial'] +
                     weights['boundary'] * losses['boundary'] +
                     weights['interior'] * losses['interior'] +
                     losses['param_reg'])

        losses['total'] = total_loss
        return losses

    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables including physical parameters"""
        variables = self.weights + self.biases + [self.log_D, self.log_k]
        if self.c_external.trainable:
            variables.append(self.c_external)
        return variables

    @tf.function
    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """Make predictions"""
        return self.forward_pass(x)

    def print_parameters(self):
        """Print current physical parameters"""
        D = self.get_diffusion_coefficient()
        k = self.get_boundary_permeability()
        c_ext = self.get_external_concentration()
        print(f"Physical Parameters:")
        print(f"  Diffusion coefficient (D): {D:.6e}")
        print(f"  Boundary permeability (k): {k:.6e}")
        print(f"  External concentration: {c_ext:.6f}")

        # Mass loss rate estimate
        print(f"  Characteristic outflow time: {1.0/k:.1f} time units")