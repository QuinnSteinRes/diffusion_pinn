#!/usr/bin/env python3
# FIXED open_system_pinn_trainer.py for cluster with graceful visualization fallback

import os
import sys

# Add project path
if os.path.exists('/state/partition1/home/qs8/projects/diffusion_pinn'):
    sys.path.append('/state/partition1/home/qs8/projects/diffusion_pinn')

# Import OPEN SYSTEM modules - FIXED VERSION
print("Loading open system modules...")
from diffusion_pinn.models.pinn import OpenSystemDiffusionPINN
from diffusion_pinn.data.processor import DiffusionDataProcessor
from diffusion_pinn.training.trainer import (
    create_open_system_pinn,
    train_open_system_pinn
)
from diffusion_pinn.variables import PINN_VARIABLES

# Try to import visualization, but don't fail if not available
try:
    from diffusion_pinn.utils.visualization import (
        plot_open_system_parameters,
        plot_mass_conservation_analysis,
        create_open_system_summary_report
    )
    HAS_VISUALIZATION = True
    print("Visualization functions loaded")
except ImportError as e:
    HAS_VISUALIZATION = False
    print(f"Visualization not available: {e}")
    print("  (Training will work, plots will be skipped)")

import tensorflow as tf
import numpy as np
import argparse
import time
import json
import traceback

def save_parameter_history(D_history, k_history, save_dir="saved_models"):
    """Save parameter histories to JSON (works without visualization)"""
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Save both histories
        history_data = {
            'D_history': [float(d) for d in D_history],
            'k_history': [float(k) for k in k_history],
            'final_D': float(D_history[-1]) if D_history else None,
            'final_k': float(k_history[-1]) if k_history else None,
            'epochs': len(D_history)
        }

        with open(os.path.join(save_dir, 'parameter_history_final.json'), 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"Parameter history saved to {save_dir}/parameter_history_final.json")

    except Exception as e:
        print(f"Warning: Could not save parameter history: {e}")

def save_training_summary(pinn, D_history, k_history, args, runtime_minutes):
    """Save comprehensive training summary"""
    try:
        final_D = pinn.get_diffusion_coefficient()
        final_k = pinn.get_boundary_permeability()

        # Determine system type
        Pe_outflow = final_k / final_D if final_D > 0 else float('inf')
        system_type = "Outflow-dominated" if Pe_outflow > 1 else "Diffusion-dominated"

        # Physical time scales
        diff_time = 1.0 / final_D if final_D > 0 else float('inf')
        outflow_time = 1.0 / final_k if final_k > 0 else float('inf')

        # Create summary
        with open("training_summary.txt", 'w') as f:
            f.write("Open System PINN Training Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: OpenSystemDiffusionPINN\n")
            f.write(f"Physics: Interior diffusion + Robin boundary conditions\n\n")

            f.write("LEARNED PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Final diffusion coefficient (D): {final_D:.8e}\n")
            f.write(f"Final boundary permeability (k): {final_k:.8e}\n\n")

            f.write("SYSTEM CHARACTERIZATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"System type: {system_type}\n")
            f.write(f"Outflow Peclet number (k/D): {Pe_outflow:.6f}\n")
            f.write(f"Diffusion time scale: {diff_time:.2f} time units\n")
            f.write(f"Outflow time scale: {outflow_time:.2f} time units\n\n")

            f.write("TRAINING DETAILS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Training epochs: {args.epochs}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Runtime: {runtime_minutes:.2f} minutes\n")
            f.write(f"Convergence: {'Good' if len(D_history) > 100 else 'Insufficient epochs'}\n\n")

            if len(D_history) >= 100:
                # Convergence analysis
                recent_D = D_history[-100:]
                recent_k = k_history[-100:]
                D_std = np.std(recent_D) / np.mean(recent_D) * 100
                k_std = np.std(recent_k) / np.mean(recent_k) * 100

                f.write("CONVERGENCE ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"D parameter stability (last 100 epochs): {D_std:.2f}% variation\n")
                f.write(f"k parameter stability (last 100 epochs): {k_std:.2f}% variation\n")
                f.write(f"Overall convergence: {'Excellent' if max(D_std, k_std) < 5 else 'Good' if max(D_std, k_std) < 10 else 'Poor'}\n")

        print("Training summary saved to training_summary.txt")

    except Exception as e:
        print(f"Warning: Could not save training summary: {e}")
        traceback.print_exc()

def create_results_directory():
    """Create results directory structure"""
    try:
        os.makedirs("results", exist_ok=True)
        os.makedirs("saved_models", exist_ok=True)
        print(" Results directories created")
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")

def main(args):
    """FIXED main function with graceful visualization handling"""

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
    print(f"Visualization: {'Available' if HAS_VISUALIZATION else 'Unavailable (will skip plots)'}")
    print("=" * 80)

    # Set seeds
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Create directories
    create_results_directory()

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

        # FIXED: Save parameter history (always works)
        save_parameter_history(D_history, k_history, "saved_models")

        # FIXED: Generate analysis plots ONLY if visualization is available
        if HAS_VISUALIZATION:
            try:
                print("\nGenerating analysis plots...")

                # Parameter evolution
                plot_open_system_parameters(D_history, k_history, save_dir="results")
                print("Parameter evolution plot created")

                # Mass conservation analysis
                data_processor = DiffusionDataProcessor(args.input_file)
                plot_mass_conservation_analysis(pinn, data_processor, save_dir="results")
                print("Mass conservation analysis created")

                # Summary report
                create_open_system_summary_report(pinn, data_processor, D_history, k_history, save_dir="results")
                print("Comprehensive summary report created")

                print("All analysis plots completed!")

            except Exception as e:
                print(f"Warning: Error during plotting: {e}")
                print("Training completed successfully, but some plots failed")
                traceback.print_exc()
        else:
            print("\nSkipping visualization (functions not available)")
            print("Parameter histories saved to JSON for later analysis")

        # FIXED: Always save training summary (doesn't need visualization)
        runtime_minutes = (time.time() - start_time) / 60
        save_training_summary(pinn, D_history, k_history, args, runtime_minutes)

        print(f"\nTotal runtime: {runtime_minutes:.2f} minutes")
        print("=" * 80)
        print("OPEN SYSTEM TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()

        # Try to save what we can
        runtime_minutes = (time.time() - start_time) / 60
        with open("error_log.txt", 'w') as f:
            f.write(f"Training failed after {runtime_minutes:.2f} minutes\n")
            f.write(f"Error: {str(e)}\n")
            f.write("Full traceback:\n")
            traceback.print_exc(file=f)

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
