import pandas as pd
import numpy as np
from pyDOE import lhs
import tensorflow as tf
from typing import Dict, Tuple

import numpy as np
from pyDOE import lhs
import tensorflow as tf
from typing import Dict, Tuple

class DiffusionDataProcessor:
    """Data processor for diffusion PINN model"""
    
    def __init__(self, inputfile: str, normalize_spatial: bool = True):
        """
        Initialize data processor
        
        Args:
            inputfile: Path to CSV file containing x, y, t, intensity data
            normalize_spatial: If True, normalize spatial coordinates to [0,1]
        """
        # Read data using numpy
        data = np.genfromtxt(inputfile, delimiter=',', skip_header=1, dtype=float)
        
        # Extract columns
        x_data = data[:, 0]
        y_data = data[:, 1]
        t_data = data[:, 2]
        intensity_data = data[:, 3]
        
        # Extract unique coordinates
        self.x_raw = np.sort(np.unique(x_data))
        self.y_raw = np.sort(np.unique(y_data))
        self.t = np.sort(np.unique(t_data))
        
        # Normalize spatial coordinates if requested
        if normalize_spatial:
            self.x = (self.x_raw - self.x_raw.min()) / (self.x_raw.max() - self.x_raw.min())
            self.y = (self.y_raw - self.y_raw.min()) / (self.y_raw.max() - self.y_raw.min())
            
            # Transform the data points
            x_norm = (x_data - self.x_raw.min()) / (self.x_raw.max() - self.x_raw.min())
            y_norm = (y_data - self.y_raw.min()) / (self.y_raw.max() - self.y_raw.min())
        else:
            self.x = self.x_raw
            self.y = self.y_raw
            x_norm = x_data
            y_norm = y_data
        
        # Reshape intensity data into 3D array
        nx, ny, nt = len(self.x), len(self.y), len(self.t)
        self.usol = np.zeros((nx, ny, nt))
        
        # Create mapping dictionaries for faster lookup
        x_indices = {val: idx for idx, val in enumerate(self.x)}
        y_indices = {val: idx for idx, val in enumerate(self.y)}
        t_indices = {val: idx for idx, val in enumerate(self.t)}
        
        # Fill the 3D array with intensity values
        for idx in range(len(x_data)):
            if normalize_spatial:
                i = x_indices[x_norm[idx]]
                j = y_indices[y_norm[idx]]
            else:
                i = x_indices[x_data[idx]]
                j = y_indices[y_data[idx]]
            k = t_indices[t_data[idx]]
            self.usol[i, j, k] = intensity_data[idx]
        
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
    def get_boundary_and_interior_points(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract boundary, initial, and interior points with their values
        
        Returns:
            Tuple of (coordinates array, values array)
        """
        coords_list = []
        values_list = []
        
        for t_idx in range(len(self.t)):
            # Get all boundary points for this time
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
            
            # Append to lists
            coords_list.extend([x_min_coords, x_max_coords, y_min_coords, y_max_coords, interior_coords])
            values_list.extend([x_min_values, x_max_values, y_min_values, y_max_values, interior_values])
        
        # Stack all points
        all_coords = np.vstack(coords_list)
        all_values = np.vstack(values_list)
        
        return all_coords, all_values

    def prepare_training_data(self, N_u: int, N_f: int, N_i: int, 
                            temporal_density: int = 10) -> dict[str, tf.Tensor]:
        """
        Prepare training data for the PINN
        
        Args:
            N_u: Number of boundary points
            N_f: Number of collocation points
            N_i: Number of interior points with direct supervision
            temporal_density: Number of time points to generate between each frame
            
        Returns:
            Dictionary containing training data tensors
        """
        all_coords, all_values = self.get_boundary_and_interior_points()
        
        # Separate boundary and interior points
        t = all_coords[:, 2]
        x = all_coords[:, 0]
        y = all_coords[:, 1]
        
        # Create masks for different types of points
        boundary_mask = np.logical_or.reduce([
            np.abs(x - self.x.min()) < 1e-6,  # x boundaries
            np.abs(x - self.x.max()) < 1e-6,
            np.abs(y - self.y.min()) < 1e-6,  # y boundaries
            np.abs(y - self.y.max()) < 1e-6
        ])
        
        interior_mask = ~boundary_mask
        
        # Sample points
        boundary_indices = np.random.choice(np.where(boundary_mask)[0], N_u, replace=False)
        interior_indices = np.random.choice(np.where(interior_mask)[0], N_i, replace=False)
        
        X_u_train = all_coords[boundary_indices]
        u_train = all_values[boundary_indices]
        
        X_i_train = all_coords[interior_indices]
        u_i_train = all_values[interior_indices]
        
        # Generate dense temporal collocation points
        t_dense = np.linspace(self.t.min(), self.t.max(), 
                            len(self.t) * temporal_density)
        
        # Generate collocation points with denser temporal sampling
        N_f_per_t = N_f // len(t_dense)
        X_f_train = []
        
        for t_val in t_dense:
            xy_points = self.lb[0:2] + (self.ub[0:2]-self.lb[0:2])*lhs(2, N_f_per_t)
            t_points = np.ones((N_f_per_t, 1)) * t_val
            X_f_train.append(np.hstack((xy_points, t_points)))
        
        X_f_train = np.vstack(X_f_train)
        X_f_train = np.vstack((X_f_train, X_u_train, X_i_train))
        
        return {
            'X_u_train': tf.convert_to_tensor(X_u_train, dtype=tf.float32),
            'u_train': tf.convert_to_tensor(u_train, dtype=tf.float32),
            'X_i_train': tf.convert_to_tensor(X_i_train, dtype=tf.float32),
            'u_i_train': tf.convert_to_tensor(u_i_train, dtype=tf.float32),
            'X_f_train': tf.convert_to_tensor(X_f_train, dtype=tf.float32),
            'X_u_test': tf.convert_to_tensor(self.X_u_test, dtype=tf.float32),
            'u_test': tf.convert_to_tensor(self.u, dtype=tf.float32)
        }

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
