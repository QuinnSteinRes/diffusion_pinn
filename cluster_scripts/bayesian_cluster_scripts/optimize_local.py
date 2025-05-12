# optimize_local.py - Modified for cluster environment

import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

# Import diffusion_pinn modules
from diffusion_pinn.data.processor import DiffusionDataProcessor
from diffusion_pinn.optimization.bayesian_opt import PINNBayesianOptimizer
from diffusion_pinn.optimization.config import OPTIMIZATION_SETTINGS
from diffusion_pinn.utils.visualization import (
    plot_solutions_and_error,
    plot_loss_history,
    plot_diffusion_convergence
)
from diffusion_pinn.utils.memory_logger import MemoryMonitor

def create_output_dirs(base_dir):
    """Create output directories"""
    dirs = {
        'results': os.path.join(base_dir, "results"),
        'plots': os.path.join(base_dir, "plots"),
        'models': os.path.join(base_dir, "models"),
        'logs': os.path.join(base_dir, "logs")
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    return dirs

def main():
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for PINN')
    parser.add_argument('--input-file', type=str,
                        default='intensity_time_series_spatial_temporal.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='optimization_output',
                        help='Base directory for output')
    parser.add_argument('--iterations', type=int,
                        default=None,
                        help='Number of optimizer iterations (overrides config)')
    parser.add_argument('--epochs', type=int,
                        default=None,
                        help='Number of training epochs per iteration (overrides config)')
    args = parser.parse_args()

    # Setup memory monitoring
    memory_monitor = MemoryMonitor(log_file='memory_usage.log')
    memory_monitor.start()

    try:
        # Use current directory as base
        base_dir = os.path.join(os.getcwd(), args.output_dir)
        dirs = create_output_dirs(base_dir)

        print("\nStarting PINN Optimization")
        print("=" * 50)

        # Load and preprocess data
        print("\nInitializing data processor...")
        data_file = args.input_file
        if not os.path.exists(data_file):
            print(f"Error: Data file not found: {data_file}")
            return

        try:
            # Override config if args provided
            opt_config = OPTIMIZATION_SETTINGS.copy()
            if args.iterations:
                opt_config['iterations_optimizer'] = args.iterations
                print(f"Overriding iterations: {args.iterations}")
            if args.epochs:
                opt_config['network_epochs'] = args.epochs
                print(f"Overriding epochs: {args.epochs}")

            data_processor = DiffusionDataProcessor(
                data_file,
                normalize_spatial=True
            )
            print("Data loaded successfully")

            # Create and run optimizer
            print("\nInitializing Bayesian optimizer...")
            optimizer = PINNBayesianOptimizer(
                data_processor=data_processor,
                opt_config=opt_config,
                save_dir=dirs['results']
            )

            print("\nStarting optimization with config:")
            print(f"Number of iterations: {opt_config['iterations_optimizer']}")
            print(f"Network epochs: {opt_config['network_epochs']}")
            print(f"Batch size: {opt_config['batchSize']}")

            results = optimizer.optimize()

            # Print results
            print("\nOptimization Results:")
            print("-" * 30)
            print("Best parameters found:")
            for param, value in results['best_parameters'].items():
                print(f"{param}: {value}")
            print(f"\nBest validation loss: {results['best_value']:.6f}")
            print(f"Total iterations run: {results['n_iterations']}")

            # Save summary to file
            with open(os.path.join(dirs['results'], 'summary.txt'), 'w') as f:
                f.write("Optimization Results\n")
                f.write("-" * 30 + "\n")
                f.write("Best parameters found:\n")
                for param, value in results['best_parameters'].items():
                    f.write(f"{param}: {value}\n")
                f.write(f"\nBest validation loss: {results['best_value']:.6f}\n")
                f.write(f"Total iterations run: {results['n_iterations']}\n")

            # Generate plots
            print("\nGenerating plots...")

            # Loss history
            plt.figure(figsize=(10, 6))
            plt.semilogy(results['all_values'])
            plt.xlabel('Iteration')
            plt.ylabel('Loss (log scale)')
            plt.title('Optimization Loss History')
            plt.grid(True)
            plt.savefig(os.path.join(dirs['plots'], 'loss_history.png'), dpi=300)
            plt.close()

            # Diffusion coefficient convergence
            if 'diffusion_values' in results:
                plt.figure(figsize=(10, 6))
                plt.plot(results['diffusion_values'])
                plt.xlabel('Iteration')
                plt.ylabel('Diffusion Coefficient')
                plt.title('Diffusion Coefficient History')
                plt.grid(True)
                plt.savefig(os.path.join(dirs['plots'], 'diffusion_convergence.png'), dpi=300)
                plt.close()

            print(f"Plots saved in: {dirs['plots']}")

            print("\nOptimization completed successfully!")
            print(f"Results saved in: {base_dir}")

        except Exception as e:
            print(f"\nError during optimization: {str(e)}")
            import traceback
            traceback.print_exc()

    finally:
        # Stop memory monitoring
        memory_monitor.stop()

if __name__ == "__main__":
    main()