#!/usr/bin/env python3
# Test with real data - this is the crucial test

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from diffusion_pinn.training.trainer import create_open_system_pinn
import tensorflow as tf

# Find your data file
possible_data_paths = [
    "../src/diffusion_pinn/data/intensity_time_series_spatial_temporal.csv",
    "../../src/diffusion_pinn/data/intensity_time_series_spatial_temporal.csv",
    "../defaultScripts/intensity_time_series_spatial_temporal.csv"
]

data_file = None
for path in possible_data_paths:
    if os.path.exists(path):
        data_file = path
        break

if data_file:
    print(f"Found data file: {data_file}")
    
    try:
        print("Creating open system PINN with real data...")
        
        # Small values for quick test
        pinn, data = create_open_system_pinn(
            inputfile=data_file,
            N_u=200,   # Small for quick test
            N_f=1000,  # Small for quick test
            N_i=300,   # Small for quick test
            seed=42
        )
        
        print("SUCCESS: Open system PINN created with real data!")
        print(f"  Initial D: {pinn.get_diffusion_coefficient():.6e}")
        print(f"  Initial k: {pinn.get_boundary_permeability():.6e}")
        print(f"  Data shapes:")
        print(f"    Initial conditions: {data['X_u_train'].shape}")
        print(f"    Physics points: {data['X_f_train'].shape}")
        print(f"    Interior points: {data['X_i_train'].shape}")
        
        # Test one loss computation
        print("\nTesting loss computation...")
        losses = pinn.loss_fn(
            x_data=data['X_u_train'][:50],  # Small subset for speed
            c_data=data['u_train'][:50],
            x_physics=data['X_f_train'][:100]
        )
        
        print(f"Loss computation successful!")
        print(f"  Total loss: {losses['total']:.6f}")
        print(f"  Initial loss: {losses['initial']:.6f}")
        print(f"  Boundary loss: {losses['boundary']:.6f}")
        print(f"  Interior loss: {losses['interior']:.6f}")
        
        print(f"\nREADY FOR FULL TRAINING!")
        print(f"Your open system PINN is working with real data.")
        print(f"You can now run the cluster scripts!")
        
    except Exception as e:
        print(f"Error with real data: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("Data file not found. Tried:")
    for path in possible_data_paths:
        print(f"  {path}")
    print("\nCopy your data file to one of these locations or update the path.")
