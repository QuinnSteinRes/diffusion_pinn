import pandas as pd
import numpy as np
from pyDOE import lhs
import tensorflow as tf
from typing import Dict, Tuple
import gc

class DiffusionDataProcessor:
    """Data processor for diffusion PINN model - v0.2.14 with seed support"""

    def __init__(self, inputfile: str, normalize_spatial: bool = True, seed: int = None):
        """
        Initialize data processor

        Args:
            inputfile: Path to CSV file containing x, y, t, intensity data
            normalize_spatial: If True, normalize spatial coordinates to [0,1]
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        try:
            # Read data in chunks to reduce memory usage (keeping v0.2.14 approach)
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
            else:
                self.x = self.x_raw
                self.y = self.y_raw
                x_norm = x_data
                y_norm = y_data

            # Initialize 3D array for solution
            nx, ny, nt = len(self.x), len(self.y), len(self.t)
            self.usol = np.zeros((nx, ny, nt))

            # Create mapping dictionaries for faster lookup
            x_indices = {val: idx for idx, val in enumerate(self.x)}
            y_indices = {val: idx for idx, val in enumerate(self.y)}
            t_indices = {val: idx for idx, val in enumerate(self.t)}

            # Fill the 3D array in batches
            batch_size = 1000
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

            # Get domain bounds
            self.X_u_test = np.hstack((
                self.X.flatten()[:,None],
                self.Y.flatten()[:,None],
                self.T.flatten()[:,None]
            ))
            self.lb = self.X_u_test[0]
            self.ub = self.X_u_test[-1]

            # Flatten solution
            self.u = self.usol.flatten('F')[:,None]

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise
        finally:
            gc.collect()

    def get_boundary_and_interior_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract boundary and interior points with their values (keeping v0.2.14 approach)

        Returns:
            Tuple of (coordinates array, values array)
        """
        try:
            coords_list = []
            values_list = []

            # Process in batches to manage memory
            batch_size = max(1, len(self.t) // 4)  # Process 25% of time steps at once

            for t_start in range(0, len(self.t), batch_size):
                t_end = min(t_start + batch_size, len(self.t))
                batch_coords = []
                batch_values = []

                for t_idx in range(t_start, t_end):
                    # X boundaries
                    x_min_coords = np.hstack((
                        self.X[0,:,t_idx].flatten()[:,None],
                        self.Y[0,:,t_idx].flatten()[:,None],
                        np.ones_like(self.X[0,:,t_idx].flatten()[:,None]) * self.t[t_idx]
                    ))
                    x_max_coords = np.hstack((
                        self.X[-1,:,t_idx].flatten()[:,None],
                        self.Y[-1,:,t_idx].flatten()[:,None],
                        np.ones_like(self.X[-1,:,t_idx].flatten()[:,None]) * self.t[t_idx]
                    ))

                    # Y boundaries
                    y_min_coords = np.hstack((
                        self.X[:,0,t_idx].flatten()[:,None],
                        self.Y[:,0,t_idx].flatten()[:,None],
                        np.ones_like(self.X[:,0,t_idx].flatten()[:,None]) * self.t[t_idx]
                    ))
                    y_max_coords = np.hstack((
                        self.X[:,-1,t_idx].flatten()[:,None],
                        self.Y[:,-1,t_idx].flatten()[:,None],
                        np.ones_like(self.X[:,-1,t_idx].flatten()[:,None]) * self.t[t_idx]
                    ))

                    # Interior points
                    interior_x = self.X[1:-1,1:-1,t_idx].flatten()[:,None]
                    interior_y = self.Y[1:-1,1:-1,t_idx].flatten()[:,None]
                    interior_t = np.ones_like(interior_x) * self.t[t_idx]
                    interior_coords = np.hstack((interior_x, interior_y, interior_t))

                    # Get corresponding values
                    x_min_values = self.usol[0,:,t_idx].flatten()[:,None]
                    x_max_values = self.usol[-1,:,t_idx].flatten()[:,None]
                    y_min_values = self.usol[:,0,t_idx].flatten()[:,None]
                    y_max_values = self.usol[:,-1,t_idx].flatten()[:,None]
                    interior_values = self.usol[1:-1,1:-1,t_idx].flatten()[:,None]

                    # Append to batch lists
                    batch_coords.extend([x_min_coords, x_max_coords, y_min_coords, y_max_coords, interior_coords])
                    batch_values.extend([x_min_values, x_max_values, y_min_values, y_max_values, interior_values])

                # Stack batch results
                coords_list.append(np.vstack(batch_coords))
                values_list.append(np.vstack(batch_values))

                # Clean up batch data
                del batch_coords, batch_values
                gc.collect()

            # Combine all batches
            all_coords = np.vstack(coords_list)
            all_values = np.vstack(values_list)

            return all_coords, all_values

        except Exception as e:
            print(f"Error in boundary and interior point extraction: {str(e)}")
            raise
        finally:
            gc.collect()

    def prepare_training_data(self, N_u: int, N_f: int, N_i: int,
                            temporal_density: int = 5, seed: int = None) -> Dict[str, tf.Tensor]:
        """
        Prepare training data for the PINN - keeping v0.2.14 approach with seed support

        Args:
            N_u: Number of boundary points
            N_f: Number of collocation points
            N_i: Number of interior points with direct supervision
            temporal_density: Number of time points to generate between each frame (v0.2.14 critical parameter!)
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing training data tensors
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        try:
            print(f"Preparing training data with temporal_density={temporal_density} (v0.2.14 critical parameter)")

            # Get boundary and interior points
            all_coords, all_values = self.get_boundary_and_interior_points()

            # Separate boundary and interior points
            t = all_coords[:, 2]
            x = all_coords[:, 0]
            y = all_coords[:, 1]

            # Create masks for different types of points
            boundary_mask = np.logical_or.reduce([
                np.abs(x - self.x.min()) < 1e-6,
                np.abs(x - self.x.max()) < 1e-6,
                np.abs(y - self.y.min()) < 1e-6,
                np.abs(y - self.y.max()) < 1e-6
            ])

            interior_mask = ~boundary_mask

            # Count true values in masks
            n_boundary = np.sum(boundary_mask)
            n_interior = np.sum(interior_mask)

            print(f"Available boundary points: {n_boundary}, interior points: {n_interior}")

            # Sample points with deterministic order for reproducibility
            if seed is not None:
                np.random.seed(seed)
            boundary_indices = np.random.choice(np.where(boundary_mask)[0], min(N_u, n_boundary),
                                            replace=(N_u > n_boundary))
            interior_indices = np.random.choice(np.where(interior_mask)[0], min(N_i, n_interior),
                                            replace=(N_i > n_interior))

            # Sort indices for deterministic order
            boundary_indices.sort()
            interior_indices.sort()

            X_u_train = all_coords[boundary_indices]
            u_train = all_values[boundary_indices]

            X_i_train = all_coords[interior_indices]
            u_i_train = all_values[interior_indices]

            print(f"Selected {len(X_u_train)} boundary points, {len(X_i_train)} interior points")

            # CRITICAL v0.2.14 FEATURE: Generate dense temporal collocation points
            print(f"Generating collocation points with temporal_density={temporal_density}")
            t_dense = np.linspace(self.t.min(), self.t.max(),
                                len(self.t) * temporal_density)

            # Generate collocation points with denser temporal sampling (v0.2.14 approach)
            N_f_per_t = N_f // len(t_dense)
            X_f_train = []

            # Process collocation points in batches
            batch_size = max(1, len(t_dense) // 4)
            for t_start in range(0, len(t_dense), batch_size):
                t_end = min(t_start + batch_size, len(t_dense))
                batch_t = t_dense[t_start:t_end]

                for t_val in batch_t:
                    # Use deterministic sampling if seed provided
                    if seed is not None:
                        np.random.seed(seed + int(t_val * 1000))  # Different seed per time
                    xy_points = self.lb[0:2] + (self.ub[0:2]-self.lb[0:2])*lhs(2, N_f_per_t)
                    t_points = np.ones((N_f_per_t, 1)) * t_val
                    X_f_train.append(np.hstack((xy_points, t_points)))

                gc.collect()

            X_f_train = np.vstack(X_f_train)

            # Add boundary and interior points to collocation points (v0.2.14 approach)
            X_f_train = np.vstack((X_f_train, X_u_train, X_i_train))

            print(f"Total collocation points: {len(X_f_train)}")

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
            print(f"Data shapes: X_u_train={X_u_train.shape}, X_i_train={X_i_train.shape}, X_f_train={X_f_train.shape}")

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

        Returns:
            Dictionary containing spatial and temporal bounds
        """
        return {
            'spatial_bounds': {
                'x': (float(self.x.min()), float(self.x.max())),
                'y': (float(self.y.min()), float(self.y.max()))
            },
            'time_bounds': (float(self.t.min()), float(self.t.max()))
        }