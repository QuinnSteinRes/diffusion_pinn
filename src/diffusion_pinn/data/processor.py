import pandas as pd
import numpy as np
from pyDOE import lhs
import tensorflow as tf
from typing import Dict, Tuple, Optional
import gc

class DiffusionDataProcessor:
    """
    Data processor for diffusion PINN model - Clean structure with labeled point sets

    Handles CSV data with columns: x, y, t, intensity
    Returns labeled training data that eliminates the need for point type identification in PINN
    """

    def __init__(self, inputfile: str, seed: Optional[int] = None):
        """
        Initialize data processor - keeps coordinates in original physical units

        Args:
            inputfile: Path to CSV file containing x, y, t, intensity data
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._setup_seed_management()

        print(f"Loading data from {inputfile}")
        self._load_and_validate_data(inputfile)
        self._build_solution_arrays()
        self._setup_domain_info()
        self._cleanup_memory()

        print(f"Data loaded: {len(self.x)} x {len(self.y)} x {len(self.t)} grid")

    # ===================================================================
    # 1. DATA LOADING & VALIDATION
    # ===================================================================

    def _load_and_validate_data(self, inputfile: str):
        """Load CSV data and validate format"""
        try:
            # Load data efficiently
            data = np.genfromtxt(inputfile, delimiter=',', skip_header=1, dtype=float)

            if data.shape[1] != 4:
                raise ValueError(f"CSV must have 4 columns (x, y, t, intensity), got {data.shape[1]}")

            # Extract columns
            self.x_data = data[:, 0].copy()
            self.y_data = data[:, 1].copy()
            self.t_data = data[:, 2].copy()
            self.intensity_data = data[:, 3].copy()

            del data  # Free memory immediately

            # Extract unique coordinates (keep original units)
            self.x = np.sort(np.unique(self.x_data))
            self.y = np.sort(np.unique(self.y_data))
            self.t = np.sort(np.unique(self.t_data))

            print(f"Domain: x=[{self.x.min():.3f}, {self.x.max():.3f}], y=[{self.y.min():.3f}, {self.y.max():.3f}], t=[{self.t.min():.3f}, {self.t.max():.3f}]")

        except Exception as e:
            raise ValueError(f"Error loading data from {inputfile}: {str(e)}")

    # ===================================================================
    # 2. 3D ARRAY CONSTRUCTION
    # ===================================================================

    def _build_solution_arrays(self):
        """Build 3D solution array from scattered data points"""
        print("Building 3D solution array...")

        nx, ny, nt = len(self.x), len(self.y), len(self.t)
        self.usol = np.zeros((nx, ny, nt))

        # Create index mappings for efficient lookup
        x_to_idx = {val: idx for idx, val in enumerate(self.x)}
        y_to_idx = {val: idx for idx, val in enumerate(self.y)}
        t_to_idx = {val: idx for idx, val in enumerate(self.t)}

        # Fill array in batches for memory efficiency
        batch_size = 10000
        for start_idx in range(0, len(self.t_data), batch_size):
            end_idx = min(start_idx + batch_size, len(self.t_data))

            for i in range(start_idx, end_idx):
                try:
                    xi = x_to_idx[self.x_data[i]]
                    yi = y_to_idx[self.y_data[i]]
                    ti = t_to_idx[self.t_data[i]]
                    self.usol[xi, yi, ti] = self.intensity_data[i]
                except KeyError:
                    continue  # Skip points that don't match grid

            if (start_idx // batch_size) % 10 == 0:  # Progress indicator
                progress = (end_idx / len(self.t_data)) * 100
                print(f"  Progress: {progress:.1f}%")

        # Clean up raw data arrays
        del self.x_data, self.y_data, self.t_data, self.intensity_data

    # ===================================================================
    # 3. DOMAIN SETUP
    # ===================================================================

    def _setup_domain_info(self):
        """Create meshgrids and domain boundary information"""
        # Create meshgrid
        self.X, self.Y, self.T = np.meshgrid(self.x, self.y, self.t, indexing='ij')

        # Domain bounds for PINN
        self.spatial_bounds = {
            'x': (float(self.x.min()), float(self.x.max())),
            'y': (float(self.y.min()), float(self.y.max()))
        }
        self.time_bounds = (float(self.t.min()), float(self.t.max()))

        # Full domain coordinates for testing
        self.X_u_test = np.column_stack([
            self.X.flatten(),
            self.Y.flatten(),
            self.T.flatten()
        ])

        # Flattened solution for testing
        self.u_test = self.usol.flatten('F')[:, None]

    # ===================================================================
    # 4. LABELED POINT SAMPLING
    # ===================================================================

    def sample_boundary_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample boundary points (spatial domain edges at all times)

        Returns:
            Tuple of (coordinates, values) arrays
        """
        self._set_sampling_seed(0)  # Consistent seed for boundary sampling

        boundary_coords = []
        boundary_values = []

        for t_idx, t_val in enumerate(self.t):
            # X boundaries (left and right edges)
            for x_edge_idx in [0, -1]:
                coords = np.column_stack([
                    np.full(len(self.y), self.x[x_edge_idx]),
                    self.y,
                    np.full(len(self.y), t_val)
                ])
                values = self.usol[x_edge_idx, :, t_idx][:, None]
                boundary_coords.append(coords)
                boundary_values.append(values)

            # Y boundaries (top and bottom edges, excluding corners already counted)
            for y_edge_idx in [0, -1]:
                coords = np.column_stack([
                    self.x[1:-1],  # Exclude corners to avoid double-counting
                    np.full(len(self.x) - 2, self.y[y_edge_idx]),
                    np.full(len(self.x) - 2, t_val)
                ])
                values = self.usol[1:-1, y_edge_idx, t_idx][:, None]
                boundary_coords.append(coords)
                boundary_values.append(values)

        # Combine all boundary points
        all_coords = np.vstack(boundary_coords)
        all_values = np.vstack(boundary_values)

        # Sample requested number of points
        if len(all_coords) > n_points:
            indices = np.random.choice(len(all_coords), n_points, replace=False)
            all_coords = all_coords[indices]
            all_values = all_values[indices]

        return all_coords, all_values

    def sample_interior_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample interior points (excluding spatial boundaries)

        Returns:
            Tuple of (coordinates, values) arrays
        """
        self._set_sampling_seed(1)  # Different seed for interior sampling

        interior_coords = []
        interior_values = []

        # Only use interior spatial points (exclude boundaries)
        if len(self.x) <= 2 or len(self.y) <= 2:
            print("Warning: Domain too small for interior points")
            return np.empty((0, 3)), np.empty((0, 1))

        x_interior = self.x[1:-1]
        y_interior = self.y[1:-1]

        for t_idx, t_val in enumerate(self.t):
            x_grid, y_grid = np.meshgrid(x_interior, y_interior, indexing='ij')

            coords = np.column_stack([
                x_grid.flatten(),
                y_grid.flatten(),
                np.full(x_grid.size, t_val)
            ])
            values = self.usol[1:-1, 1:-1, t_idx].flatten()[:, None]

            interior_coords.append(coords)
            interior_values.append(values)

        # Combine all interior points
        all_coords = np.vstack(interior_coords)
        all_values = np.vstack(interior_values)

        # Sample requested number of points
        if len(all_coords) > n_points:
            indices = np.random.choice(len(all_coords), n_points, replace=False)
            all_coords = all_coords[indices]
            all_values = all_values[indices]

        return all_coords, all_values

    def sample_collocation_points(self, n_points: int, temporal_density: int = 5) -> np.ndarray:
        """
        Sample physics collocation points using Latin Hypercube Sampling

        Args:
            n_points: Total number of collocation points desired
            temporal_density: Multiplier for temporal resolution (creates artificial time points)

        Returns:
            Array of coordinates for physics loss computation
        """
        self._set_sampling_seed(2)  # Different seed for physics sampling

        print(f"Generating {n_points} collocation points with temporal_density={temporal_density}")

        # Create dense temporal grid (artificial time points for physics)
        t_dense = np.linspace(self.t.min(), self.t.max(), len(self.t) * temporal_density)

        # Points per time step
        n_per_t = max(1, n_points // len(t_dense))

        collocation_points = []
        x_min, x_max = self.x.min(), self.x.max()
        y_min, y_max = self.y.min(), self.y.max()

        for i, t_val in enumerate(t_dense):
            # Different seed per time step for reproducibility
            self._set_sampling_seed(1000 + i)

            # Sample spatial points using Latin Hypercube
            xy_samples = np.column_stack([
                x_min + (x_max - x_min) * lhs(2, n_per_t)[:, 0],
                y_min + (y_max - y_min) * lhs(2, n_per_t)[:, 1]
            ])

            # Add time coordinate
            coords = np.column_stack([
                xy_samples,
                np.full(n_per_t, t_val)
            ])

            collocation_points.append(coords)

        # Combine all collocation points
        all_points = np.vstack(collocation_points)

        # Trim to exact number requested
        if len(all_points) > n_points:
            all_points = all_points[:n_points]

        print(f"Generated {len(all_points)} collocation points")
        return all_points

    def prepare_labeled_training_data(self, N_boundary: int, N_interior: int, N_collocation: int,
                                    temporal_density: int = 5) -> Dict[str, Tuple]:
        """
        Main interface: Sample all point types and return labeled training data

        Args:
            N_boundary: Number of boundary/initial condition points
            N_interior: Number of interior supervision points
            N_collocation: Number of physics collocation points
            temporal_density: Temporal density multiplier for physics points

        Returns:
            Dictionary with labeled point sets:
            {
                'boundary': (coords_array, values_array),
                'interior': (coords_array, values_array),
                'physics': coords_array,
                'test': (coords_array, values_array)
            }
        """
        print(f"Preparing labeled training data...")
        print(f"  Boundary points: {N_boundary}")
        print(f"  Interior points: {N_interior}")
        print(f"  Physics points: {N_collocation}")

        # Sample each point type
        boundary_coords, boundary_values = self.sample_boundary_points(N_boundary)
        interior_coords, interior_values = self.sample_interior_points(N_interior)
        physics_coords = self.sample_collocation_points(N_collocation, temporal_density)

        # Combine into labeled dictionary
        labeled_data = {
            'boundary': (boundary_coords, boundary_values),
            'interior': (interior_coords, interior_values),
            'physics': physics_coords,
            'test': (self.X_u_test, self.u_test)  # Full domain for testing
        }

        print(f"Labeled training data prepared successfully")
        print(f"  Boundary: {boundary_coords.shape} coords, {boundary_values.shape} values")
        print(f"  Interior: {interior_coords.shape} coords, {interior_values.shape} values")
        print(f"  Physics: {physics_coords.shape} coords")

        self._cleanup_memory()
        return labeled_data

    # ===================================================================
    # 5. SEED & MEMORY MANAGEMENT
    # ===================================================================

    def _setup_seed_management(self):
        """Initialize seed management for reproducible sampling"""
        self.base_seed = self.seed if self.seed is not None else 42

    def _set_sampling_seed(self, seed_offset: int):
        """Set numpy random seed with offset for different sampling operations"""
        if self.seed is not None:
            np.random.seed(self.base_seed + seed_offset)

    def _cleanup_memory(self):
        """Centralized memory cleanup at strategic points"""
        gc.collect()

    # ===================================================================
    # 6. UTILITIES & DIAGNOSTICS
    # ===================================================================

    def get_domain_info(self) -> Dict:
        """
        Get domain information for PINN initialization

        Returns:
            Dictionary containing spatial and temporal bounds
        """
        return {
            'spatial_bounds': self.spatial_bounds,
            'time_bounds': self.time_bounds,
            'grid_shape': (len(self.x), len(self.y), len(self.t))
        }

    def print_data_summary(self):
        """Print summary of loaded data"""
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Spatial domain: x=[{self.x.min():.3f}, {self.x.max():.3f}], y=[{self.y.min():.3f}, {self.y.max():.3f}]")
        print(f"Time domain: t=[{self.t.min():.3f}, {self.t.max():.3f}]")
        print(f"Grid resolution: {len(self.x)} x {len(self.y)} x {len(self.t)} = {self.usol.size:,} points")
        print(f"Solution range: [{self.usol.min():.3f}, {self.usol.max():.3f}]")
        if hasattr(self, 'base_seed'):
            print(f"Random seed: {self.base_seed}")
        print("="*50 + "\n")