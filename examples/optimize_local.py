# examples/optimize_local.py

import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from diffusion_pinn.optimization.bayesian_opt import PINNBayesianOptimizer
from diffusion_pinn.optimization.config import OPTIMIZATION_SETTINGS

from diffusion_pinn.data.processor import DiffusionDataProcessor
from diffusion_pinn.optimization.bayesian_opt import PINNBayesianOptimizer
from diffusion_pinn.optimization.config import OPTIMIZATION_SETTINGS
from diffusion_pinn.utils.visualization import (
    plot_solutions_and_error,
    plot_loss_history,
    plot_diffusion_convergence
)

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
    # Use current directory as base
    base_dir = os.path.join(os.getcwd(), "optimization_output")
    dirs = create_output_dirs(base_dir)

    print("\nStarting PINN Optimization")
    print("=" * 50)

    # Load and preprocess data
    print("\nInitializing data processor...")
    data_file = OPTIMIZATION_SETTINGS["inputFile"]
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return

    try:
        data_processor = DiffusionDataProcessor(
            data_file,
            normalize_spatial=True
        )
        print("Data loaded successfully")

        # Create and run optimizer
        print("\nInitializing Bayesian optimizer...")
        optimizer = PINNBayesianOptimizer(
            data_processor=data_processor,
            opt_config=OPTIMIZATION_SETTINGS,
            save_dir=dirs['results']
        )

        print("\nStarting optimization with config:")
        print(f"Number of iterations: {OPTIMIZATION_SETTINGS['iterations_optimizer']}")
        print(f"Network epochs: {OPTIMIZATION_SETTINGS['network_epochs']}")
        print(f"Batch size: {OPTIMIZATION_SETTINGS['batchSize']}")

        results = optimizer.optimize()

        # Print results
        print("\nOptimization Results:")
        print("-" * 30)
        print("Best parameters found:")
        for param, value in results['best_parameters'].items():
            print(f"{param}: {value}")
        print(f"\nBest validation loss: {results['best_value']:.6f}")
        print(f"Total iterations run: {results['n_iterations']}")

        # Generate plots if requested
        if OPTIMIZATION_SETTINGS["plot_results"]:
            print("\nGenerating plots...")

            # Loss history
            plot_loss_history(results['all_values'])
            plt.savefig(os.path.join(dirs['plots'], 'loss_history.png'))
            plt.close()

            # Diffusion coefficient convergence
            if 'diffusion_values' in results:
                plot_diffusion_convergence(results['diffusion_values'])
                plt.savefig(os.path.join(dirs['plots'], 'diffusion_convergence.png'))
                plt.close()

            # Load best model and plot solutions
            best_model = optimizer.load_best_model()
            t_indices = [0, len(data_processor.t)//3, 2*len(data_processor.t)//3, -1]
            plot_solutions_and_error(
                best_model,
                data_processor,
                t_indices,
                save_path=os.path.join(dirs['plots'], 'final_solutions.png')
            )

            print(f"Plots saved in: {dirs['plots']}")

        print("\nOptimization completed successfully!")
        print(f"Results saved in: {base_dir}")

    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
