#!/usr/bin/env python
# optimize_minimal.py - Stripped-down version that avoids matplotlib

# Set environment variables to limit threading and improve performance
import os
import sys
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Initialize debugging
print("Starting script execution")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Try adding diffusion_pinn path explicitly if not in sys.path
diffusion_pinn_path = "/state/partition1/home/qs8/projects/diffusion_pinn"
if diffusion_pinn_path not in sys.path:
    print(f"Adding {diffusion_pinn_path} to sys.path")
    sys.path.append(diffusion_pinn_path)

# Import basic libraries
import numpy as np
import tensorflow as tf
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path

print("Basic imports successful")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Try importing diffusion_pinn modules
from diffusion_pinn.data.processor import DiffusionDataProcessor
from diffusion_pinn.config import DiffusionConfig
from diffusion_pinn.models.pinn import DiffusionPINN
from diffusion_pinn.optimization.config import OPTIMIZATION_SETTINGS, update_config
from diffusion_pinn.variables import PINN_VARIABLES

print("diffusion_pinn modules imported successfully")

# Import skopt for Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

print("All imports successful")

# Skip visualization imports to avoid matplotlib dependency
# We'll save raw data instead of plots

def create_output_dirs(base_dir):
    """Create output directories without timestamp to keep paths predictable"""
    dirs = {
        'results': os.path.join(base_dir, "results"),
        'data': os.path.join(base_dir, "data"),
        'models': os.path.join(base_dir, "models"),
        'logs': os.path.join(base_dir, "logs"),
        'checkpoints': os.path.join(base_dir, "checkpoints")
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    return dirs

def save_config(config, filepath):
    """Save configuration to a text file"""
    with open(filepath, 'w') as f:
        f.write("Optimization Configuration:\n")
        f.write("=" * 60 + "\n\n")
        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

def create_model(layers, neurons, activation, learning_rate, config, data_processor):
    """Create a PINN model with the given parameters and configuration"""
    # Get domain information from data processor
    domain_info = data_processor.get_domain_info()

    # Create PINN configuration
    pinn_config = DiffusionConfig(
        hidden_layers=[neurons] * layers,
        activation=activation,
        initialization='glorot',
        diffusion_trainable=True,
        use_physics_loss=True
    )

    # Create PINN model
    pinn = DiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=config.get('initial_D', PINN_VARIABLES['initial_D']),
        config=pinn_config
    )

    # Create optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=config.get('gradient_clip_norm', 1.0),
    )

    return pinn, optimizer

def train_and_evaluate(pinn, optimizer, data, trial_dir, config):
    """Train and evaluate a PINN model without visualization"""
    from diffusion_pinn.training.trainer import train_pinn

    try:
        # Train model
        D_history, loss_history = train_pinn(
            pinn=pinn,
            data=data,
            optimizer=optimizer,
            epochs=config['network_epochs'],
            save_dir=trial_dir,
        )

        # Get final values
        if len(loss_history) > 0:
            if isinstance(loss_history[-1], dict):
                final_loss = loss_history[-1].get('total', float('inf'))
            else:
                final_loss = loss_history[-1]

            # Handle potential NaN/Inf values
            if np.isnan(final_loss) or np.isinf(final_loss):
                print("Warning: Invalid loss value detected")
                final_loss = float('inf')
        else:
            final_loss = float('inf')

        # Get diffusion coefficient
        if len(D_history) > 0:
            final_D = D_history[-1]
            # Check for valid D
            if np.isnan(final_D) or np.isinf(final_D):
                print("Warning: Invalid diffusion coefficient detected")
                final_D = 0.0
        else:
            final_D = 0.0

        # Save histories to numpy files - no plotting
        np.save(os.path.join(trial_dir, 'loss_history.npy'), loss_history)
        np.save(os.path.join(trial_dir, 'D_history.npy'), D_history)

        return final_loss, final_D

    except Exception as e:
        print(f"Error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('inf'), 0.0

class MinimalBayesianOptimizer:
    """Bayesian optimization for PINN hyperparameters without visualization"""

    def __init__(self, data_processor, config, save_dirs):
        self.data_processor = data_processor
        self.config = config
        self.save_dirs = save_dirs

        # Initialize tracking variables
        self.best_loss = float('inf')
        self.best_params = None
        self.best_model_path = os.path.join(save_dirs['models'], 'best_model.h5')

        # Optimization history
        self.history = {
            'losses': [],
            'diffusion_coeffs': [],
            'parameters': []
        }

        # Prepare training data
        print("Preparing training data...")
        self.training_data = self.data_processor.prepare_training_data(
            N_u=config.get('N_u', PINN_VARIABLES['N_u']),
            N_f=config.get('N_f', PINN_VARIABLES['N_f']),
            N_i=config.get('N_i', PINN_VARIABLES['N_i'])
        )

        # Define optimization space based on config
        self.dimensions = [
            Integer(
                low=config['layers_lowerBound'],
                high=config['layers_upperBound'],
                name='layers'
            ),
            Integer(
                low=config['neurons_lowerBound'],
                high=config['neurons_upperBound'],
                name='neurons'
            ),
            Categorical(
                categories=['relu', 'tanh', 'elu', 'selu'],
                name='activation'
            ),
            Real(
                low=config['learning_lowerBound'],
                high=config['learning_upperBound'],
                prior='log-uniform',
                name='learning_rate'
            )
        ]

    def objective(self, layers, neurons, activation, learning_rate):
        """Objective function for Bayesian optimization"""
        try:
            # Check for parameter validity
            if layers <= 0 or neurons <= 0 or learning_rate <= 0:
                print(f"Invalid parameters: layers={layers}, neurons={neurons}, learning_rate={learning_rate}")
                return float('inf')

            # Create trial directory
            trial_name = f"trial_l{layers}_n{neurons}_a{activation}_lr{learning_rate:.2e}"
            trial_dir = os.path.join(self.save_dirs['logs'], trial_name)
            os.makedirs(trial_dir, exist_ok=True)

            # Save trial configuration
            save_config({
                'layers': layers,
                'neurons': neurons,
                'activation': activation,
                'learning_rate': learning_rate
            }, os.path.join(trial_dir, 'config.txt'))

            print(f"\nStarting trial: {trial_name}")

            # Create model
            pinn, optimizer = create_model(
                layers, neurons, activation, learning_rate,
                self.config, self.data_processor
            )

            # Train and evaluate
            val_loss, final_D = train_and_evaluate(
                pinn, optimizer, self.training_data, trial_dir, self.config
            )

            # Update history
            self.history['losses'].append(val_loss)
            self.history['diffusion_coeffs'].append(final_D)
            self.history['parameters'].append({
                'layers': layers,
                'neurons': neurons,
                'activation': activation,
                'learning_rate': learning_rate
            })

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_params = {
                    'layers': layers,
                    'neurons': neurons,
                    'activation': activation,
                    'learning_rate': learning_rate
                }

                # Save model
                try:
                    pinn.save(self.best_model_path)
                    print(f"New best model saved with loss: {val_loss:.6f}")

                    # Save best configuration
                    save_config(
                        {**self.best_params, 'loss': val_loss, 'diffusion': final_D},
                        os.path.join(self.save_dirs['models'], 'best_config.txt')
                    )
                except Exception as e:
                    print(f"Warning: Could not save best model: {str(e)}")

            # Save checkpoint data (no plots)
            self.save_checkpoint()

            # Clean up
            del pinn, optimizer
            tf.keras.backend.clear_session()
            gc.collect()

            return val_loss

        except Exception as e:
            print(f"Error in trial: {str(e)}")
            import traceback
            traceback.print_exc()

            # Clean up
            tf.keras.backend.clear_session()
            gc.collect()

            return float('inf')

    def save_checkpoint(self):
        """Save current optimization state without plots"""
        try:
            checkpoint_path = os.path.join(
                self.save_dirs['checkpoints'],
                f'checkpoint_{len(self.history["losses"])}.npz'
            )

            np.savez(
                checkpoint_path,
                losses=np.array(self.history['losses']),
                diffusion_coeffs=np.array(self.history['diffusion_coeffs']),
                parameters=self.history['parameters'],
                best_loss=self.best_loss,
                best_params=self.best_params
            )
            
            print(f"Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def optimize(self):
        """Run the Bayesian optimization process"""
        # Decorate objective function for skopt
        @use_named_args(dimensions=self.dimensions)
        def objective_func(**params):
            return self.objective(**params)

        # Initial point
        x0 = [
            self.config['initial_layers'],
            self.config['initial_neurons'],
            self.config['initial_activation'],
            self.config['initial_learningRate']
        ]

        print("\nStarting Bayesian optimization")
        print(f"Total iterations: {self.config['iterations_optimizer']}")
        print(f"Initial point: {x0}")

        try:
            # Run optimization
            result = gp_minimize(
                func=objective_func,
                dimensions=self.dimensions,
                n_calls=self.config['iterations_optimizer'],
                n_random_starts=max(1, int(self.config['iterations_optimizer'] * self.config.get('random_starts_fraction', 0.3))),
                x0=x0,
                acq_func=self.config.get('acquisitionFunction', 'EI'),
                random_state=42,
                n_jobs=1,  # Single job to avoid memory issues
                verbose=True
            )

            # Format results
            best_params = dict(zip(
                [d.name for d in self.dimensions],
                result.x
            ))

            # Save final results
            self.save_final_results(result, best_params)

            return {
                'best_parameters': best_params,
                'best_value': result.fun,
                'all_values': result.func_vals,
                'n_iterations': len(result.func_vals),
                'diffusion_values': self.history['diffusion_coeffs'],
                'optimization_history': self.history
            }

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            import traceback
            traceback.print_exc()

            # Save what we have so far
            if self.best_params:
                self.save_final_results(None, self.best_params)

            return {
                'error': str(e),
                'best_parameters': self.best_params,
                'best_value': self.best_loss,
                'n_iterations': len(self.history['losses']),
                'optimization_history': self.history
            }

    def save_final_results(self, result, best_params):
        """Save final optimization results without visualizations"""
        # Save history
        np.savez(
            os.path.join(self.save_dirs['results'], 'optimization_history.npz'),
            losses=np.array(self.history['losses']),
            diffusion_coeffs=np.array(self.history['diffusion_coeffs']),
            parameters=self.history['parameters']
        )

        # Save results summary
        with open(os.path.join(self.save_dirs['results'], 'summary.txt'), 'w') as f:
            f.write("Optimization Results\n")
            f.write("=" * 60 + "\n\n")
            f.write("Best parameters found:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nBest validation loss: {self.best_loss:.6f}\n")
            f.write(f"Total iterations run: {len(self.history['losses'])}\n")

        # Save raw data for later plotting
        valid_indices = [i for i, loss in enumerate(self.history['losses'])
                       if not np.isnan(loss) and not np.isinf(loss)]

        if valid_indices:
            valid_losses = [self.history['losses'][i] for i in valid_indices]
            valid_ds = [self.history['diffusion_coeffs'][i] for i in valid_indices]
            valid_params = [self.history['parameters'][i] for i in valid_indices]

            # Save plot data in CSV format for later plotting
            with open(os.path.join(self.save_dirs['data'], 'plot_data.csv'), 'w') as f:
                f.write("trial,loss,diffusion,layers,neurons,activation,learning_rate\n")
                for i, idx in enumerate(valid_indices):
                    params = self.history['parameters'][idx]
                    f.write(f"{i},{valid_losses[i]},{valid_ds[i]},{params.get('layers', 0)}," +
                           f"{params.get('neurons', 0)},{params.get('activation', 'unknown')}," +
                           f"{params.get('learning_rate', 0.0)}\n")

        # Try to save skopt result
        if result:
            try:
                from skopt.utils import dump
                dump(result, os.path.join(self.save_dirs['results'], 'skopt_result.pkl'))
            except Exception as e:
                print(f"Warning: Could not save skopt result: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for PINN')
    parser.add_argument('--input-file', type=str,
                        default=OPTIMIZATION_SETTINGS['inputFile'],
                        help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='optimization_output',
                        help='Base directory for output')
    parser.add_argument('--iterations', type=int,
                        default=OPTIMIZATION_SETTINGS['iterations_optimizer'],
                        help='Number of optimizer iterations')
    parser.add_argument('--epochs', type=int,
                        default=OPTIMIZATION_SETTINGS['network_epochs'],
                        help='Number of training epochs per iteration')
    args = parser.parse_args()

    # Create a custom configuration with command-line overrides
    custom_config = update_config({
        'inputFile': args.input_file,
        'iterations_optimizer': args.iterations,
        'network_epochs': args.epochs
    })

    # Print configuration summary
    print("\nRunning with configuration:")
    print(f"Input file: {custom_config['inputFile']}")
    print(f"Iterations: {custom_config['iterations_optimizer']}")
    print(f"Epochs per trial: {custom_config['network_epochs']}")
    print(f"Parameter ranges:")
    print(f"  Layers: {custom_config['layers_lowerBound']} to {custom_config['layers_upperBound']}")
    print(f"  Neurons: {custom_config['neurons_lowerBound']} to {custom_config['neurons_upperBound']}")
    print(f"  Learning rate: {custom_config['learning_lowerBound']} to {custom_config['learning_upperBound']}")

    # Create output directories without timestamp to keep paths predictable
    dirs = create_output_dirs(args.output_dir)

    # Save the configuration
    save_config(custom_config, os.path.join(args.output_dir, 'configuration.txt'))

    try:
        print("\nStarting PINN Optimization")
        print("=" * 60)

        # Load and preprocess data
        print("\nInitializing data processor...")
        if not os.path.exists(custom_config['inputFile']):
            print(f"Error: Data file not found: {custom_config['inputFile']}")
            return

        try:
            # Apply TensorFlow memory management
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            data_processor = DiffusionDataProcessor(
                custom_config['inputFile'],
                normalize_spatial=True
            )
            print("Data loaded successfully")

            # Create and run optimizer
            print("\nInitializing Bayesian optimizer...")
            optimizer = MinimalBayesianOptimizer(
                data_processor=data_processor,
                config=custom_config,
                save_dirs=dirs
            )

            # Run optimization
            results = optimizer.optimize()

            # Print and save results
            if 'best_parameters' in results and results['best_parameters']:
                print("\nOptimization Results:")
                print("-" * 40)
                print("Best parameters found:")
                for param, value in results['best_parameters'].items():
                    print(f"{param}: {value}")
                print(f"\nBest validation loss: {results['best_value']:.6f}")
                print(f"Total iterations run: {results['n_iterations']}")
                print(f"\nResults saved in: {args.output_dir}")
            else:
                print("\nNo valid results found. Check logs for errors.")

        except Exception as e:
            print(f"\nError during optimization: {str(e)}")
            import traceback
            traceback.print_exc()

    finally:
        print(f"\nAll output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
