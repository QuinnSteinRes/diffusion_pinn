#!/usr/bin/env python3
# Test training pipeline imports

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from diffusion_pinn.training.trainer import (
        create_open_system_pinn,
        train_open_system_pinn
    )
    print("✓ Successfully imported open system training functions")
    
    from diffusion_pinn.utils.visualization import (
        plot_open_system_parameters,
        plot_mass_conservation_analysis
    )
    print("✓ Successfully imported open system visualization functions")
    
    print("\n✓ All core components ready!")
    print("Your open system PINN is ready for testing.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nThis means you need to implement the missing modules.")
    print("Check which specific function is missing and implement it.")
    
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
