#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import tensorflow as tf

# Force reload all diffusion_pinn modules
modules_to_remove = [k for k in sys.modules.keys() if k.startswith('diffusion_pinn')]
for module in modules_to_remove:
    del sys.modules[module]

# Now continue with your existing script...
sys.path.append('src')

# Import modules
try:
    from diffusion_pinn.data.processor import DiffusionDataProcessor
    from diffusion_pinn.config import DiffusionConfig
    from diffusion_pinn.models.pinn import DiffusionPINN
    from diffusion_pinn.training.trainer import create_and_initialize_pinn, train_pinn
    from diffusion_pinn.variables import PINN_VARIABLES
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the diffusion_pinn root directory")
    sys.exit(1)

def main():
    """Main test function - follows cluster script pattern exactly"""
    print("Starting PINN training test...")

    # Use exact variables from PINN_VARIABLES
    data_file = "src/diffusion_pinn/data/intensity_time_series_spatial_temporal.csv"
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return

    print(f"Using data file: {data_file}")
    print(f"Epochs: {PINN_VARIABLES['epochs']}")
    print(f"Random seed: {PINN_VARIABLES['random_seed']}")
    print(f"Initial D: {PINN_VARIABLES['initial_D']}")

    # Set random seeds exactly like cluster script
    tf.random.set_seed(PINN_VARIABLES['random_seed'])
    np.random.seed(PINN_VARIABLES['random_seed'])

    try:
        print("Creating and initializing PINN...")

        # Create PINN with NEW parameter names
        pinn, training_data = create_and_initialize_pinn(
            inputfile=data_file,
            N_boundary=PINN_VARIABLES['N_u'],      # CHANGED: N_u → N_boundary
            N_collocation=PINN_VARIABLES['N_f'],   # CHANGED: N_f → N_collocation
            N_interior=PINN_VARIABLES['N_i'],      # CHANGED: N_i → N_interior
            initial_D=PINN_VARIABLES['initial_D'],
            seed=PINN_VARIABLES['random_seed']
        )

        print(f"PINN created - Initial D: {pinn.get_diffusion_coefficient():.6f}")

        # Create optimizer exactly like cluster script
        optimizer = tf.keras.optimizers.Adam(learning_rate=PINN_VARIABLES['learning_rate'])

        print(f"Starting training for {PINN_VARIABLES['epochs']} epochs...")
        start_time = time.time()

        # Train with NEW parameter name
        D_history, loss_history = train_pinn(
            pinn=pinn,
            labeled_data=training_data,  # CHANGED: data → labeled_data
            optimizer=optimizer,
            epochs=PINN_VARIABLES['epochs']
        )

        training_time = time.time() - start_time
        final_D = pinn.get_diffusion_coefficient()

        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final D: {final_D:.6f}")

        if len(loss_history) > 0:
            if isinstance(loss_history[-1], dict):
                final_loss = loss_history[-1].get('total', 0)
            else:
                final_loss = loss_history[-1]
            print(f"Final loss: {final_loss:.6f}")

        print(f"D history length: {len(D_history)}")
        print(f"Loss history length: {len(loss_history)}")

        if len(D_history) > 0:
            print(f"D range: {min(D_history):.6f} to {max(D_history):.6f}")

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()