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
        config: DiffusionConfig = None,
        seed: int = None,
        data_processor = None  # Add this parameter for smart initialization
    ):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.seed = seed  # Store seed for consistent initialization

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

        # Smart initialization of diffusion coefficient
        #if data_processor is not None:
        #    initial_D = self._initialize_diffusion_coefficient_smartly(data_processor)

        if False:  # Disable smart initialization
            initial_D = self._initialize_diffusion_coefficient_smartly(data_processor)

        # Initialize diffusion coefficient with positivity constraint
        initial_D_value = max(initial_D, 1e-5)  # Ensure positive value
        self.D = tf.Variable(initial_D_value, dtype=tf.float32,
                           trainable=self.config.diffusion_trainable,
                           name='diffusion_coefficient',
                           constraint=lambda x: tf.clip_by_value(x, 1e-5, 1.0))

        # Store loss weights from variables
        self.loss_weights = PINN_VARIABLES['loss_weights']

        # Initialize tolerances for condition identification
        self.boundary_tol = 1e-6
        self.initial_tol = 1e-6

        # Build network architecture with enhanced initialization
        self._build_network()

    def _initialize_diffusion_coefficient_smartly(self, data_processor):
        """Initialize diffusion coefficient based on data characteristics"""
        # Estimate characteristic length and time scales from data
        domain_info = data_processor.get_domain_info()

        # Characteristic length (use the larger dimension)
        L_char = max(
            domain_info['spatial_bounds']['x'][1] - domain_info['spatial_bounds']['x'][0],
            domain_info['spatial_bounds']['y'][1] - domain_info['spatial_bounds']['y'][0]
        )

        # Characteristic time
        T_char = domain_info['time_bounds'][1] - domain_info['time_bounds'][0]

        # Estimate initial D based on diffusion scaling: L^2 ~ D*T
        D_estimate = (L_char ** 2) / T_char

        # Clamp to reasonable range based on physical expectations
        D_initial = np.clip(D_estimate, 1e-5, 1e-2)

        print(f"Smart initialization: L_char={L_char:.4f}, T_char={T_char:.4f}")
        print(f"Estimated D: {D_estimate:.6f}, Clamped D: {D_initial:.6f}")

        return D_initial

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
        """
        Enhanced input normalization with improved numerical stability

        Normalizes inputs to [-1, 1] with additional safeguards:
        - Handles edge cases where bounds might be equal
        - Ensures numerical stability with small denominators
        - Maintains consistent data types
        - Clips extreme values to prevent overflow
        """
        # Ensure input is float32 for consistency
        x_float = tf.cast(x, tf.float32)

        # Calculate range with numerical stability check
        range_tensor = self.ub - self.lb

        # Add small epsilon to prevent division by zero if any dimension has zero range
        # This can happen if all data points have the same coordinate in one dimension
        epsilon = tf.constant(1e-8, dtype=tf.float32)
        safe_range = tf.maximum(range_tensor, epsilon)

        # Normalize to [0, 1] first
        normalized_01 = (x_float - self.lb) / safe_range

        # Clip to handle any numerical issues that might push values outside [0,1]
        normalized_01_clipped = tf.clip_by_value(normalized_01, 0.0, 1.0)

        # Transform to [-1, 1]
        normalized_final = 2.0 * normalized_01_clipped - 1.0

        # Final safety clip to ensure we're exactly in [-1, 1]
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
        Enhanced loss function with additional consistency terms for stable convergence

        This adds physical realism checks beyond just data fitting and PDE residuals:
        - Mass conservation (diffusion should preserve total mass approximately)
        - Positivity constraints (concentrations can't be negative)
        - Smoothness requirements (solutions should be reasonably smooth)
        - Diffusion coefficient regularization (keep D in physically reasonable range)
        """
        if weights is None:
            weights = self.loss_weights

        # Separate points by condition type
        condition_points = self.identify_condition_points(x_data)
        losses = {}

        # Compute losses for each condition type with enhanced stability
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

                    # ENHANCED: Use Huber loss instead of MSE for robustness to outliers
                    delta = 0.1  # Huber loss parameter
                    abs_error = tf.abs(c_pred - c_true)
                    quadratic = tf.minimum(abs_error, delta)
                    linear = abs_error - quadratic
                    losses[condition_type] = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)
                else:
                    losses[condition_type] = tf.constant(0.0, dtype=tf.float32)
            else:
                losses[condition_type] = tf.constant(0.0, dtype=tf.float32)

        # Enhanced Physics loss with stability measures
        if self.config.use_physics_loss and x_physics is not None and x_physics.shape[0] > 0:
            pde_residual = self.compute_pde_residual(x_physics)

            # ENHANCED: Multi-scale physics loss - emphasize different error magnitudes
            residual_mean = tf.reduce_mean(tf.abs(pde_residual))

            # Scale-aware Huber loss - adapts to the typical residual magnitude
            delta = tf.maximum(0.1, residual_mean)
            abs_residual = tf.abs(pde_residual)
            quadratic = tf.minimum(abs_residual, delta)
            linear = abs_residual - quadratic
            physics_loss = tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)

            losses['physics'] = physics_loss
        else:
            losses['physics'] = tf.constant(0.0, dtype=tf.float32)

        # NEW: Consistency losses - ensure physical reasonableness
        consistency_losses = self.compute_consistency_losses(x_data, c_data)
        losses.update(consistency_losses)

        # NEW: Enhanced diffusion coefficient regularization
        d_reg_multiple = self.compute_diffusion_regularization()
        losses.update(d_reg_multiple)

        # Total loss with weighted components
        total_loss = sum(weights.get(key, 1.0) * losses[key] for key in ['initial', 'boundary', 'interior', 'physics'])

        # Add consistency and regularization terms with smaller weights
        consistency_weight = 0.1  # Don't let consistency terms dominate
        total_loss += consistency_weight * (
            losses.get('mass_conservation', 0.0) +
            losses.get('positivity', 0.0) +
            losses.get('smoothness', 0.0)
        )

        # Add diffusion regularization terms
        total_loss += sum(losses[key] for key in losses if key.startswith('d_reg'))

        losses['total'] = total_loss
        return losses

    def compute_consistency_losses(self, x_data: tf.Tensor, c_data: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Compute additional consistency losses for physical realism

        These losses help ensure the PINN solution makes physical sense:
        1. Mass conservation - total mass shouldn't change dramatically over time
        2. Positivity - concentrations should be non-negative
        3. Smoothness - nearby points should have similar concentrations
        """
        consistency_losses = {}

        # 1. MASS CONSERVATION CHECK
        # In diffusion, total mass should be approximately conserved
        # (allowing for boundary flux but checking for unrealistic changes)
        t_vals = tf.unique(x_data[:, 2])[0]
        if len(t_vals) > 1:
            mass_losses = []
            for i in range(min(3, len(t_vals) - 1)):  # Check first few time points
                t1, t2 = t_vals[i], t_vals[i + 1]

                # Get predictions at these times
                mask1 = tf.abs(x_data[:, 2] - t1) < 1e-6
                mask2 = tf.abs(x_data[:, 2] - t2) < 1e-6

                if tf.reduce_any(mask1) and tf.reduce_any(mask2):
                    c1 = self.forward_pass(tf.boolean_mask(x_data, mask1))
                    c2 = self.forward_pass(tf.boolean_mask(x_data, mask2))

                    # Average concentration (proxy for mass)
                    mass1 = tf.reduce_mean(c1)
                    mass2 = tf.reduce_mean(c2)

                    # Mass should change slowly (allow 50% change max)
                    relative_change = tf.abs(mass1 - mass2) / (mass1 + 1e-6)
                    mass_penalty = tf.nn.relu(relative_change - 0.5)  # Penalty if change > 50%
                    mass_losses.append(mass_penalty)

            if mass_losses:
                consistency_losses['mass_conservation'] = tf.reduce_mean(mass_losses)
            else:
                consistency_losses['mass_conservation'] = tf.constant(0.0, dtype=tf.float32)
        else:
            consistency_losses['mass_conservation'] = tf.constant(0.0, dtype=tf.float32)

        # 2. POSITIVITY CONSTRAINT
        # Concentrations should be non-negative (physical requirement)
        c_pred = self.forward_pass(x_data)
        negative_penalty = tf.reduce_mean(tf.nn.relu(-c_pred))  # Penalty for negative values
        consistency_losses['positivity'] = negative_penalty

        # 3. SMOOTHNESS CONSTRAINT
        # Nearby points should have similar concentrations (diffusion creates smooth solutions)
        if x_data.shape[0] > 10:
            # Sample random pairs of points for efficiency
            n_pairs = min(50, x_data.shape[0] // 2)  # Check up to 50 pairs
            indices = tf.range(tf.shape(x_data)[0])
            shuffled_indices = tf.random.shuffle(indices)[:2*n_pairs]

            smoothness_losses = []
            for i in range(0, len(shuffled_indices) - 1, 2):
                if i + 1 < len(shuffled_indices):
                    idx1, idx2 = shuffled_indices[i], shuffled_indices[i + 1]
                    x1, x2 = x_data[idx1], x_data[idx2]
                    c1, c2 = c_pred[idx1], c_pred[idx2]

                    # Calculate distance between points
                    spatial_dist = tf.norm(x1[:2] - x2[:2])  # x,y distance
                    temporal_dist = tf.abs(x1[2] - x2[2])    # time distance
                    total_dist = spatial_dist + 0.1 * temporal_dist + 1e-6  # Weight time less

                    # Concentration difference should be proportional to distance
                    c_diff = tf.abs(c1 - c2)

                    # Penalty if concentration changes too rapidly with distance
                    smoothness_ratio = c_diff / total_dist
                    smoothness_penalty = tf.nn.relu(smoothness_ratio - 10.0)  # Penalty if ratio > 10
                    smoothness_losses.append(smoothness_penalty)

            if smoothness_losses:
                consistency_losses['smoothness'] = tf.reduce_mean(smoothness_losses)
            else:
                consistency_losses['smoothness'] = tf.constant(0.0, dtype=tf.float32)
        else:
            consistency_losses['smoothness'] = tf.constant(0.0, dtype=tf.float32)

        return consistency_losses

    def compute_diffusion_regularization(self) -> Dict[str, tf.Tensor]:
        """
        Multiple regularization terms for diffusion coefficient

        These help keep the diffusion coefficient in physically reasonable ranges:
        1. Range regularization - keep D near expected order of magnitude
        2. Stability regularization - prevent extreme values that cause instability
        """
        d_reg_losses = {}

        # 1. RANGE REGULARIZATION
        # Keep D near expected order of magnitude (around 10^-4 for typical diffusion)
        D_target = tf.constant(0.0001, dtype=tf.float32)  # Expected order of magnitude
        d_reg_losses['d_reg_range'] = 0.01 * tf.square(
            tf.math.log(self.D + 1e-8) - tf.math.log(D_target)
        )

        # 2. STABILITY REGULARIZATION
        # Prevent extreme values that cause numerical instability
        D_min_stable = tf.constant(1e-6, dtype=tf.float32)
        D_max_stable = tf.constant(1e-2, dtype=tf.float32)

        # Penalty for going outside stable range
        d_reg_losses['d_reg_stability'] = 0.001 * (
            tf.nn.relu(D_min_stable - self.D) +  # Penalty if too small
            tf.nn.relu(self.D - D_max_stable)    # Penalty if too large
        )

        return d_reg_losses

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

    def save(self, filepath: str):
        """Save the model to a file"""
        # This is a placeholder - implement based on your needs
        print(f"Model saving to {filepath} - implement based on your requirements")
        pass