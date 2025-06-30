#!/usr/bin/env python3
# Basic test that open system model can be imported and initialized

import sys
sys.path.append('..')

try:
    from diffusion_pinn.models.pinn import OpenSystemDiffusionPINN
    from diffusion_pinn.variables import PINN_VARIABLES
    print("✓ Successfully imported OpenSystemDiffusionPINN")
    
    # Test basic initialization
    spatial_bounds = {'x': (0, 1), 'y': (0, 1)}
    time_bounds = (0, 1)
    
    pinn = OpenSystemDiffusionPINN(
        spatial_bounds=spatial_bounds,
        time_bounds=time_bounds,
        initial_D=0.0001,
        initial_k=0.001
    )
    
    print(f"✓ Model initialized successfully")
    print(f"  Initial D: {pinn.get_diffusion_coefficient():.6e}")
    print(f"  Initial k: {pinn.get_boundary_permeability():.6e}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
