import pandas as pd
import numpy as np
from pyDOE import lhs
import tensorflow as tf
from typing import Dict, Tuple
import gc

class DiffusionDataProcessor:
    """Data processor for diffusion PINN model - HYBRID approach"""

    def __init__(self, inputfile: str, normalize_spatial: bool = True, seed: int = None):
        """
        Initialize data processor - HYBRID: Keep V0.2.22's improvements but fix classification

        Args:
            inputfile: Path to CSV file containing x, y, t, intensity data
            normalize_spatial: If True, normalize spatial coordinates to [0,1]
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        try:
            # Read data efficiently
            data = np.genfromtxt(inputfile, delimiter=',', skip_header=1, dtype=float)

            # Extract columns and immediately delete original data
            x_data = data[:, 0].copy()
            y_data = data[:, 1].copy()
            t_data = data[:, 2].copy()
            intensity_data = data[:, 3].copy()
            del data
            gc.collect()

            # Extract unique coordinates
            self.x_raw = np.sort(np.unique(x_data))
            self.y_raw = np.sort(np.unique(y_data))
            self.t = np.sort(np.unique(t_data))

            print(f"Raw data ranges:")
            print(f"  x: [{self.x_raw.min():.6f}, {self.x_raw.max():.6f}] ({len(self.x_raw)} points)")
            print(f"  y: [{self.y_raw.min():.6f}, {self.y_raw.max():.6f}] ({len(self.y_raw)} points)")
            print(f"  t: [{self.t.min():.6f}, {self.t.max():.6f}] ({len(self.t)} points)")

            # Normalize spatial coordinates if requested
            if normalize_spatial:
                x_min, x_max = self.x_raw.min(), self.x_raw.max()
                y_min, y_max = self.y_raw.min(), self.y_raw.max()

                self.x = (self.x_raw - x_min) / (x_max - x_min)
                self.y = (self.y_raw - y_min) / (y_max - y_min)

                # Transform the data points
                x_norm = (x_data - x_min) / (x_max - x_min)
                y_norm = (y_data - y_min) / (y_max - y_min)

                del x_data, y_data
                gc.collect()

                print(f"Normalized spatial coordinates to [0,1]")
                print(f"  x_norm: [{self.x.min():.6f}, {self.x.max():.6f}]")
                print(f"  y_norm: [{self.y.min():.6f}, {self.y.max():.6f}]")
            else:
                self.x = self.x_raw
                self.y = self.y_raw
                x_norm = x_data
                y_norm = y_data

            # Initialize 3D array for solution
            nx, ny, nt = len(self.x), len(self.y), len(self.t)
            self.usol = np.zeros((nx, ny, nt))

            print(f"Creating solution array: {nx} x {ny} x {nt} = {nx*ny*nt} points")

            # Create mapping dictionaries for faster lookup
            x_indices = {val: idx for idx, val in enumerate(self.x)}
            y_indices = {val: idx for idx, val in enumerate(self.y)}
            t_indices = {val: idx for idx, val in enumerate(self.t)}

            # Fill the 3D array in batches
            batch_size = 10000
            for start_idx in range(0, len(t_data), batch_size):
                end_idx = min(start_idx + batch_size, len(t_data))
                batch_slice = slice(start_idx, end_idx)

                if normalize_spatial:
                    x_idx = [x_indices[x] for x in x_norm[batch_slice]]
                    y_idx = [y_indices[y] for y in y_norm[batch_slice]]
                else:
                    x_idx = [x_indices[x] for x in x_data[batch_slice]]
                    y_idx = [y_indices[y] for y in y_data[batch_slice]]
                t_idx = [t_indices[t] for t in t_data[batch_slice]]

                for i, (xi, yi, ti) in enumerate(zip(x_idx, y_idx, t_idx)):
                    self.usol[xi, yi, ti] = intensity_data[start_idx + i]

            # Clean up temporary arrays
            del x_norm, y_norm, t_data, intensity_data
            del x_indices, y_indices, t_indices
            gc.collect()

            # Create meshgrid
            self.X, self.Y, self.T = np.meshgrid(self.x, self.y, self.t, indexing='ij')

            # Get domain bounds - CRITICAL FIX: Use actual coordinate bounds
            self.X_u_test = np.hstack((
                self.X.flatten()[:,None],
                self.Y.flatten()[:,None],
                self.T.flatten()[:,None]
            ))

            # FIXED: Use proper bounds
            self.lb = np.array([self.x.min(), self.y.min(), self.t.min()])
            self.ub = np.array([self.x.max(), self.y.max(), self.t.max()])

            print(f"Domain bounds:")
            print(f"  Lower: [{self.lb[0]:.6f}, {self.lb[1]:.6f}, {self.lb[2]:.6f}]")
            print(f"  Upper: [{self.ub[0]:.6f}, {self.ub[1]:.6f}, {self.ub[2]:.6f}]")

            # Flatten solution
            self.u = self.usol.flatten('F')[:,None]

            print(f"Data processing completed successfully")
            print(f"Solution range: [{self.u.min():.6f}, {self.u.max():.6f}]")

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise
        finally:
            gc.collect()

    def get_boundary_and_interior_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract boundary and interior points with their values - HYBRID: Simplified but robust
        """
        try:
            all_coords = []
            all_values = []

            # Process each time step
            for t_idx, t_val in enumerate(self.t):
                # Boundary points for this time step
                # X boundaries (x=0 and x=1)
                for x_boundary_idx in [0, -1]:  # First and last x indices
                    x_boundary_coords = np.column_stack([
                        np.full(len(self.y), self.x[x_boundary_idx]),
                        self.y,
                        np.full(len(self.y), t_val)
                    ])
                    x_boundary_values = self.usol[x_boundary_idx, :, t_idx].reshape(-1, 1)

                    all_coords.append(x_boundary_coords)
                    all_values.append(x_boundary_values)

                # Y boundaries (y=0 and y=1)
                for y_boundary_idx in [0, -1]:  # First and last y indices
                    y_boundary_coords = np.column_stack([
                        self.x,
                        np.full(len(self.x), self.y[y_boundary_idx]),
                        np.full(len(self.x), t_val)
                    ])
                    y_boundary_values = self.usol[:, y_boundary_idx, t_idx].reshape(-1, 1)

                    all_coords.append(y_boundary_coords)
                    all_values.append(y_boundary_values)

                # Interior points for this time step (exclude boundaries)
                if len(self.x) > 2 and len(self.y) > 2:  # Only if we have interior points
                    interior_x_indices = range(1, len(self.x) - 1)
                    interior_y_indices = range(1, len(self.y) - 1)

                    interior_coords_list = []
                    interior_values_list = []

                    for x_idx in interior_x_indices:
                        for y_idx in interior_y_indices:
                            interior_coords_list.append([self.x[x_idx], self.y[y_idx], t_val])
                            interior_values_list.append(self.usol[x_idx, y_idx, t_idx])

                    if interior_coords_list:  # Only add if we have interior points
                        interior_coords = np.array(interior_coords_list)
                        interior_values = np.array(interior_values_list).reshape(-1, 1)

                        all_coords.append(interior_coords)
                        all_values.append(interior_values)

            # Combine all coordinates and values
            final_coords = np.vstack(all_coords)
            final_values = np.vstack(all_values)

            print(f"Extracted {len(final_coords)} boundary and interior points")

            # Verify coordinate ranges
            print(f"Point coordinate ranges:")
            print(f"  x: [{final_coords[:, 0].min():.6f}, {final_coords[:, 0].max():.6f}]")
            print(f"  y: [{final_coords[:, 1].min():.6f}, {final_coords[:, 1].max():.6f}]")
            print(f"  t: [{final_coords[:, 2].min():.6f}, {final_coords[:, 2].max():.6f}]")

            return final_coords, final_values

        except Exception as e:
            print(f"Error in boundary and interior point extraction: {str(e)}")
            raise
        finally:
            gc.collect()

    def create_deterministic_collocation_points(self, N_f: int, seed: int = None) -> np.ndarray:
        """
        Create deterministic collocation points - Keep V0.2.22's approach but simpler
        """
        if seed is not None:
            np.random.seed(seed)

        print(f"Generating {N_f} deterministic collocation points...")

        try:
            # Use scipy's Sobol sequences if available, otherwise fallback to LHS
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=3, scramble=False, seed=seed)
                points = sampler.random(N_f)
                print("Using Sobol quasi-random sequences")
            except ImportError:
                # Fallback to Latin Hypercube Sampling
                points = lhs(3, samples=N_f, criterion='maximin', random_state=seed)
                print("Using Latin Hypercube Sampling (fallback)")

            # Scale to domain bounds
            x_min, x_max = self.x.min(), self.x.max()
            y_min, y_max = self.y.min(), self.y.max()
            t_min, t_max = self.t.min(), self.t.max()

            points[:, 0] = x_min + (x_max - x_min) * points[:, 0]  # x coordinates
            points[:, 1] = y_min + (y_max - y_min) * points[:, 1]  # y coordinates
            points[:, 2] = t_min + (t_max - t_min) * points[:, 2]  # t coordinates

            print(f"Collocation points bounds:")
            print(f"  x: [{points[:, 0].min():.6f}, {points[:, 0].max():.6f}]")
            print(f"  y: [{points[:, 1].min():.6f}, {points[:, 1].max():.6f}]")
            print(f"  t: [{points[:, 2].min():.6f}, {points[:, 2].max():.6f}]")

            return points

        except Exception as e:
            print(f"Error creating collocation points: {str(e)}")
            raise

    def prepare_training_data(self, N_u: int, N_f: int, N_i: int,
                            temporal_density: int = 10, seed: int = None) -> Dict[str, tf.Tensor]:
        """
        Prepare training data - HYBRID: V0.2.22's structure but with fixed classification
        """
        if seed is not None:
            np.random.seed(seed)

        try:
            print(f"Preparing training data with N_u={N_u}, N_f={N_f}, N_i={N_i}")

            # Get boundary and interior points
            all_coords, all_values = self.get_boundary_and_interior_points()

            # CRITICAL FIX: Better point classification using exact coordinate matching
            x_coords = all_coords[:, 0]
            y_coords = all_coords[:, 1]
            t_coords = all_coords[:, 2]

            # Create masks for different types of points with tight tolerances
            tol = 1e-10  # Very tight tolerance for exact matching

            # Initial condition mask (t = t_min)
            ic_mask = np.abs(t_coords - self.t.min()) < tol

            # Boundary condition masks (x = x_min/max or y = y_min/max, but not at t_min)
            x_boundary_mask = np.logical_or(
                np.abs(x_coords - self.x.min()) < tol,
                np.abs(x_coords - self.x.max()) < tol
            )
            y_boundary_mask = np.logical_or(
                np.abs(y_coords - self.y.min()) < tol,
                np.abs(y_coords - self.y.max()) < tol
            )
            spatial_boundary_mask = np.logical_or(x_boundary_mask, y_boundary_mask)

            # Boundary points are spatial boundaries that are NOT initial conditions
            boundary_mask = np.logical_and(spatial_boundary_mask, ~ic_mask)

            # Interior points are neither boundaries nor initial conditions
            interior_mask = np.logical_and(~spatial_boundary_mask, ~ic_mask)

            # Count points for verification
            n_ic = np.sum(ic_mask)
            n_boundary = np.sum(boundary_mask)
            n_interior = np.sum(interior_mask)

            print(f"Point classification:")
            print(f"  Initial condition points: {n_ic}")
            print(f"  Boundary condition points: {n_boundary}")
            print(f"  Interior points: {n_interior}")
            print(f"  Total classified: {n_ic + n_boundary + n_interior} / {len(all_coords)}")

            # Sample points deterministically
            if seed is not None:
                np.random.seed(seed)

            # Sample boundary + initial points
            combined_mask = np.logical_or(ic_mask, boundary_mask)
            combined_indices = np.where(combined_mask)[0]
            if len(combined_indices) > N_u:
                boundary_indices = np.random.choice(combined_indices, N_u, replace=False)
                boundary_indices.sort()  # Keep deterministic order
            else:
                boundary_indices = combined_indices

            # Sample interior points
            interior_indices = np.where(interior_mask)[0]
            if len(interior_indices) > N_i:
                selected_interior_indices = np.random.choice(interior_indices, N_i, replace=False)
                selected_interior_indices.sort()  # Keep deterministic order
            else:
                selected_interior_indices = interior_indices

            X_u_train = all_coords[boundary_indices]
            u_train = all_values[boundary_indices]

            X_i_train = all_coords[selected_interior_indices]
            u_i_train = all_values[selected_interior_indices]

            print(f"Selected {len(X_u_train)} boundary/initial points, {len(X_i_train)} interior points")

            # VERIFICATION: Check that we have initial condition points
            t_check = X_u_train[:, 2]
            ic_count_in_training = np.sum(np.abs(t_check - self.t.min()) < tol)
            print(f"VERIFICATION: Training data contains {ic_count_in_training} initial condition points")

            if ic_count_in_training == 0:
                print("ERROR: No initial condition points in training data!")
                # Force include some initial condition points
                ic_indices = np.where(ic_mask)[0]
                if len(ic_indices) > 0:
                    # Add the first few IC points to boundary training data
                    n_ic_to_add = min(len(ic_indices), N_u // 4)  # Add up to 25% as IC points
                    ic_to_add = ic_indices[:n_ic_to_add]

                    X_u_train = np.vstack([X_u_train, all_coords[ic_to_add]])
                    u_train = np.vstack([u_train, all_values[ic_to_add]])

                    print(f"FIXED: Added {n_ic_to_add} initial condition points to training data")

            # Generate collocation points
            X_f_train = self.create_deterministic_collocation_points(N_f, seed=seed)

            # Convert to TensorFlow tensors
            training_data = {
                'X_u_train': tf.convert_to_tensor(X_u_train, dtype=tf.float32),
                'u_train': tf.convert_to_tensor(u_train, dtype=tf.float32),
                'X_i_train': tf.convert_to_tensor(X_i_train, dtype=tf.float32),
                'u_i_train': tf.convert_to_tensor(u_i_train, dtype=tf.float32),
                'X_f_train': tf.convert_to_tensor(X_f_train, dtype=tf.float32),
                'X_u_test': tf.convert_to_tensor(self.X_u_test, dtype=tf.float32),
                'u_test': tf.convert_to_tensor(self.u, dtype=tf.float32)
            }

            print("Training data preparation completed successfully")
            print(f"Final data shapes:")
            print(f"  X_u_train: {X_u_train.shape}")
            print(f"  X_i_train: {X_i_train.shape}")
            print(f"  X_f_train: {X_f_train.shape}")

            return training_data

        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            gc.collect()

    def get_domain_info(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Get domain information for PINN initialization
        """
        return {
            'spatial_bounds': {
                'x': (float(self.x.min()), float(self.x.max())),
                'y': (float(self.y.min()), float(self.y.max()))
            },
            'time_bounds': (float(self.t.min()), float(self.t.max()))
        }