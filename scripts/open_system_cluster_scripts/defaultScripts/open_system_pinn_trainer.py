#!/usr/bin/env python3
# RENAMED and UPDATED trainer for open system

import os
import sys

# Add project path
if os.path.exists('/state/partition1/home/qs8/projects/diffusion_pinn'):
    sys.path.append('/state/partition1/home/qs8/projects/diffusion_pinn')

# Import OPEN SYSTEM modules (NOT the old ones)
print("Loading open system modules...")
from diffusion_pinn.models.pinn import OpenSystemDiffusionPINN
from diffusion_pinn.data.processor import DiffusionDataProcessor
from diffusion_pinn.training.trainer import (
    create_open_system_pinn,
    train_open_system_pinn
)
from diffusion_pinn.utils.visualization import (
    plot_open_system_parameters,
    plot_mass_conservation_analysis,
    create_open_system_summary_report
)
from diffusion_pinn.variables import PINN_VARIABLES

import tensorflow as tf
import numpy as np
import argparse
import time

def main(args):
    """Main function for open system PINN training"""

    start_time = time.time()

    print("=" * 80)
    print("OPEN SYSTEM PINN TRAINING")
    print("=" * 80)
    print(f"Model: OpenSystemDiffusionPINN")
    print(f"Learning: D (diffusion) + k (boundary permeability)")
    print(f"Physics: Interior diffusion + Robin boundary conditions")
    print(f"Input: {args.input_file}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # Set seeds
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    try:
        # Create open system PINN
        print("\nCreating open system PINN...")
        pinn, data = create_open_system_pinn(
            inputfile=args.input_file,
            N_u=PINN_VARIABLES['N_u'],
            N_f=PINN_VARIABLES['N_f'],
            N_i=PINN_VARIABLES['N_i'],
            initial_D=PINN_VARIABLES['initial_D'],
            initial_k=PINN_VARIABLES['initial_k'],
            seed=args.seed
        )

        print(f"Initial D: {pinn.get_diffusion_coefficient():.6e}")
        print(f"Initial k: {pinn.get_boundary_permeability():.6e}")

        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=PINN_VARIABLES['learning_rate']
        )

        # Train the model
        print(f"\nStarting training for {args.epochs} epochs...")
        D_history, k_history, loss_history = train_open_system_pinn(
            pinn=pinn,
            data=data,
            optimizer=optimizer,
            epochs=args.epochs,
            save_dir="saved_models",
            seed=args.seed
        )

        # Final results
        final_D = pinn.get_diffusion_coefficient()
        final_k = pinn.get_boundary_permeability()

        print(f"\nTraining completed!")
        print(f"Final D: {final_D:.8e}")
        print(f"Final k: {final_k:.8e}")

        # Determine system type
        Pe_outflow = final_k / final_D if final_D > 0 else float('inf')
        system_type = "Outflow-dominated" if Pe_outflow > 1 else "Diffusion-dominated"
        print(f"System type: {system_type} (Pe = {Pe_outflow:.3f})")

        # Generate analysis plots
        try:
            print("\nGenerating analysis plots...")

            # Parameter evolution
            plot_open_system_parameters(D_history, k_history, save_dir="results")

            # Mass conservation analysis
            data_processor = DiffusionDataProcessor(args.input_file)
            plot_mass_conservation_analysis(pinn, data_processor, save_dir="results")

            # Summary report
            create_open_system_summary_report(pinn, data_processor, D_history, k_history, save_dir="results")

            print("Analysis plots completed!")

        except Exception as e:
            print(f"Warning: Error during plotting: {e}")

        # Save summary
        with open("training_summary.txt", 'w') as f:
            f.write("Open System PINN Training Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: OpenSystemDiffusionPINN\n")
            f.write(f"Final diffusion coefficient (D): {final_D:.8e}\n")
            f.write(f"Final boundary permeability (k): {final_k:.8e}\n")
            f.write(f"System type: {system_type}\n")
            f.write(f"Outflow Peclet number: {Pe_outflow:.6f}\n")
            f.write(f"Training epochs: {args.epochs}\n")
            f.write(f"Total training time: {(time.time() - start_time)/60:.2f} minutes\n")

        print(f"\nTotal runtime: {(time.time() - start_time)/60:.2f} minutes")
        return 0

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Open System PINN')
    parser.add_argument('--input-file', type=str,
                       default='intensity_time_series_spatial_temporal.csv',
                       help='Input CSV file')
    parser.add_argument('--epochs', type=int,
                       default=PINN_VARIABLES['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int,
                       default=PINN_VARIABLES['random_seed'],
                       help='Random seed')

    args = parser.parse_args()
    sys.exit(main(args))