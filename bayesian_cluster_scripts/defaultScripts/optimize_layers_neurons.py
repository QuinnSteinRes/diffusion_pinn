#!/usr/bin/env python
# optimize_layers_neurons.py - Specialized script that only optimizes layers and neurons

# Set environment variables to limit threading and improve performance
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path

# Import diffusion_pinn modules
from diffusion_pinn.data.processor import DiffusionDataProcessor
from diffusion_pinn.config import DiffusionConfig
from diffusion_pinn.models.pinn import DiffusionPINN
from diffusion_pinn.optimization.config import OPTIMIZATION_SETTINGS, update_config
from diffusion_pinn.utils.memory_logger import MemoryMonitor
from diffusion_pinn.utils.visualization import plot_loss_history, plot_diffusion_convergence
from diffusion_pinn.variables import PINN_VARIABLES
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

def create_output_dirs(base_dir):
    """Create output directories with timestamps to avoid overwriting"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"{base_dir}_{timestamp}"

    dirs = {
        'results': os.path.join(base_dir, "results"),
        'plots': os.path.join(base_dir, "plots"),
        'models': os.path.join(base_dir, "models"),
        'logs': os.path.join(base_dir, "logs"),
        'checkpoints': os.path.join(base_dir, "checkpoints")
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    return dirs, base_dir

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

def create_model(layers, neurons, config, data_processor):
    """Create a PINN model with the given parameters and fixed configuration"""
    # Get domain information from data processor
    domain_info = data_processor.get_domain_info()

    # Fixed activation and learning rate from config
    activation = config.get('fixed_activation', 'tanh')
    learning_rate = config.get('fixed_learning_rate', 1e-4)

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

    # Create optimizer with gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=config.get('gradient_clip_norm', 1.0),
    )

    return pinn, optimizer

def train_and_evaluate(pinn, optimizer, data, trial_dir, config):
    """Train and evaluate a PINN model"""
    from diffusion_pinn.training.trainer import train_pinn

    try:
        # Train model with the updated trainer
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

        # Save histories to numpy files
        np.save(os.path.join(trial_dir, 'loss_history.npy'), loss_history)
        np.save(os.path.join(trial_dir, 'D_history.npy'), D_history)

        # Plot results for this trial
        plots_dir = os.path.join(trial_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Use the updated visualization functions
        plot_loss_history(loss_history, save_dir=plots_dir)
        plot_diffusion_convergence(D_history, save_dir=plots_dir)

        return final_loss, final_D

    except tf.errors.ResourceExhaustedError:
        print(f"Error: Out of memory in training")
        tf.keras.backend.clear_session()
        gc.collect()
        return float('inf'), 0.0
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return float('inf'), 0.0

class LayersNeuronsOptimizer:
    """Bayesian optimization for PINN architecture (layers and neurons only)"""

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

        # Define optimization space - ONLY layers and neurons
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
            )
        ]

    def objective(self, layers, neurons):
        """Objective function for Bayesian optimization"""
        try:
            # Check for parameter validity
            if layers <= 0 or neurons <= 0:
                print(f"Invalid parameters: layers={layers}, neurons={neurons}")
                return float('inf')

            # Create trial directory
            activation = self.config.get('fixed_activation', 'tanh')
            learning_rate = self.config.get('fixed_learning_rate', 1e-4)
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

            # Create model with fixed activation and learning rate
            pinn, optimizer = create_model(
                layers, neurons, self.config, self.data_processor
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
                'activation': self.config.get('fixed_activation', 'tanh'),
                'learning_rate': self.config.get('fixed_learning_rate', 1e-4)
            })

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_params = {
                    'layers': layers,
                    'neurons': neurons,
                    'activation': self.config.get('fixed_activation', 'tanh'),
                    'learning_rate': self.config.get('fixed_learning_rate', 1e-4)
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

            # Save checkpoint
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
        """Save current optimization state"""
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

            # Create checkpoint plots
            self.create_checkpoint_plots()

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def create_checkpoint_plots(self):
        """Create plots for current optimization state"""
        try:
            # Filter valid losses
            valid_indices = [i for i, loss in enumerate(self.history['losses'])
                          if not np.isnan(loss) and not np.isinf(loss)]

            if not valid_indices:
                return

            valid_losses = [self.history['losses'][i] for i in valid_indices]
            valid_ds = [self.history['diffusion_coeffs'][i] for i in valid_indices]
            valid_params = [self.history['parameters'][i] for i in valid_indices]

            # Loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(valid_losses, 'bo-')
            plt.title('Optimization Progress')
            plt.xlabel('Trial')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dirs['plots'], 'optimization_progress.png'))
            plt.close()

            # Diffusion coefficient plot
            plt.figure(figsize=(10, 6))
            plt.plot(valid_ds, 'ro-')
            plt.title('Diffusion Coefficient')
            plt.xlabel('Trial')
            plt.ylabel('D Value')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dirs['plots'], 'diffusion_progress.png'))
            plt.close()

        except Exception as e:
            print(f"Error creating checkpoint plots: {str(e)}")

    def optimize(self):
        """Run the Bayesian optimization process"""
        # Decorate objective function for skopt
        @use_named_args(dimensions=self.dimensions)
        def objective_func(**params):
            return self.objective(**params)

        # Initial point
        x0 = [
            self.config.get('initial_layers', 4),
            self.config.get('initial_neurons', 32)
        ]

        print("\nStarting Layers/Neurons optimization")
        print(f"Total iterations: {self.config['iterations_optimizer']}")
        print(f"Initial point: {x0}")
        print(f"Fixed activation: {self.config.get('fixed_activation', 'tanh')}")
        print(f"Fixed learning rate: {self.config.get('fixed_learning_rate', 1e-4)}")

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

            # Add fixed parameters
            best_params['activation'] = self.config.get('fixed_activation', 'tanh')
            best_params['learning_rate'] = self.config.get('fixed_learning_rate', 1e-4)

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
        """Save final optimization results"""
        # Save history
        np.savez(
            os.path.join(self.save_dirs['results'], 'optimization_history.npz'),
            losses=np.array(self.history['losses']),
            diffusion_coeffs=np.array(self.history['diffusion_coeffs']),
            parameters=self.history['parameters']
        )

        # Save results summary
        with open(os.path.join(self.save_dirs['results'], 'summary.txt'), 'w') as f:
            f.write("Layers and Neurons Optimization Results\n")
            f.write("=" * 60 + "\n\n")
            f.write("Best parameters found:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nBest validation loss: {self.best_loss:.6f}\n")
            f.write(f"Total iterations run: {len(self.history['losses'])}\n")

        # Create final plots
        self.create_final_plots()

        # Try to save skopt result
        if result:
            try:
                from skopt.utils import dump
                dump(result, os.path.join(self.save_dirs['results'], 'skopt_result.pkl'))
            except Exception as e:
                print(f"Warning: Could not save skopt result: {str(e)}")

    def create_final_plots(self):
        """Create final plots for the optimization results"""
        # Create the plots directory
        plots_dir = self.save_dirs['plots']
        os.makedirs(plots_dir, exist_ok=True)

        # Filter valid losses
        valid_indices = [i for i, loss in enumerate(self.history['losses'])
                       if not np.isnan(loss) and not np.isinf(loss)]

        if not valid_indices:
            print("No valid loss values to plot")
            return

        valid_losses = [self.history['losses'][i] for i in valid_indices]
        valid_ds = [self.history['diffusion_coeffs'][i] for i in valid_indices]
        valid_params = [self.history['parameters'][i] for i in valid_indices]

        # Loss history
        plt.figure(figsize=(12, 6))
        plt.plot(valid_losses, 'o-')
        plt.title('Optimization Loss History')
        plt.xlabel('Trial')
        plt.ylabel('Validation Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'optimization_loss_history.png'))
        plt.close()

        # Diffusion coefficient history
        plt.figure(figsize=(12, 6))
        plt.plot(valid_ds, 'o-')
        plt.title('Diffusion Coefficient History')
        plt.xlabel('Trial')
        plt.ylabel('Diffusion Coefficient')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'optimization_d_history.png'))
        plt.close()

        # Parameter influence plots - just layers and neurons
        param_names = ['layers', 'neurons']
        for param in param_names:
            plt.figure(figsize=(12, 6))
            param_values = [p.get(param, 0) for p in valid_params]

            plt.plot(param_values, valid_losses, 'o')
            
            # Calculate and display average losses per parameter value
            unique_vals = sorted(set(param_values))
            avg_losses = []
            
            for val in unique_vals:
                indices = [i for i, p in enumerate(param_values) if p == val]
                avg_loss = np.mean([valid_losses[i] for i in indices])
                avg_losses.append(avg_loss)
            
            plt.plot(unique_vals, avg_losses, 'r-', linewidth=2, label='Average loss')
            plt.legend()

            plt.title(f'Loss vs {param}')
            plt.xlabel(param)
            plt.ylabel('Validation Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f'param_importance_{param}.png'))
            plt.close()

        # Create a heat map of layers vs neurons
        try:
            plt.figure(figsize=(10, 8))
            
            # Extract unique values
            layer_values = sorted(set(p.get('layers', 0) for p in valid_params))
            neuron_values = sorted(set(p.get('neurons', 0) for p in valid_params))
            
            # Create matrix to hold losses
            heat_data = np.full((len(layer_values), len(neuron_values)), np.nan)
            
            # Fill in matrix
            for i, l in enumerate(layer_values):
                for j, n in enumerate(neuron_values):
                    # Find trials with this configuration
                    matching = [
                        idx for idx, p in enumerate(valid_params) 
                        if p.get('layers', 0) == l and p.get('neurons', 0) == n
                    ]
                    
                    if matching:
                        avg_loss = np.mean([valid_losses[idx] for idx in matching])
                        heat_data[i, j] = avg_loss
            
            # Create heatmap with masked NaN values
            masked_data = np.ma.masked_invalid(heat_data)
            
            # Plot heatmap
            plt.pcolormesh(neuron_values, layer_values, masked_data, cmap='viridis_r', shading='nearest')
            plt.colorbar(label='Average Loss')
            plt.xlabel('Neurons per Layer')
            plt.ylabel('Number of Layers')
            plt.title('Loss Heatmap: Layers vs Neurons')
            
            # Add numbers to cells
            for i, l in enumerate(layer_values):
                for j, n in enumerate(neuron_values):
                    if not np.isnan(heat_data[i, j]):
                        plt.text(j+0.5, i+0.5, f'{heat_data[i, j]:.2e}', 
                                 ha='center', va='center', 
                                 color='white' if heat_data[i, j] > np.nanmean(heat_data) else 'black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'layers_neurons_heatmap.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create heatmap: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Run Layers/Neurons optimization for PINN')
    parser.add_argument('--input-file', type=str,
                        default=OPTIMIZATION_SETTINGS['inputFile'],
                        help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='layerneuron_optimization',
                        help='Base directory for output')
    parser.add_argument('--iterations', type=int,
                        default=OPTIMIZATION_SETTINGS['iterations_optimizer'],
                        help='Number of optimizer iterations')
    parser.add_argument('--epochs', type=int,
                        default=OPTIMIZATION_SETTINGS['network_epochs'],
                        help='Number of training epochs per iteration')
    parser.add_argument('--activation', type=str,
                        default='tanh',
                        help='Fixed activation function to use')
    parser.add_argument('--learning-rate', type=float,
                        default=1e-4,
                        help='Fixed learning rate to use')
    args = parser.parse_args()

    # Create a custom configuration with command-line overrides
    custom_config = update_config({
        'inputFile': args.input_file,
        'iterations_optimizer': args.iterations,
        'network_epochs': args.epochs,
        'fixed_activation': args.activation,
        'fixed_learning_rate': args.learning_rate
    })

    # Print configuration summary
    print("\nRunning with configuration:")
    print(f"Input file: {custom_config['inputFile']}")
    print(f"Iterations: {custom_config['iterations_optimizer']}")
    print(f"Epochs per trial: {custom_config['network_epochs']}")
    print(f"Fixed activation: {custom_config['fixed_activation']}")
    print(f"Fixed learning rate: {custom_config['fixed_learning_rate']}")
    print(f"Parameter ranges:")
    print(f"  Layers: {custom_config['layers_lowerBound']} to {custom_config['layers_upperBound']}")
    print(f"  Neurons: {custom_config['neurons_lowerBound']} to {custom_config['neurons_upperBound']}")

    # Create output directories with timestamp
    dirs, timestamped_base_dir = create_output_dirs(args.output_dir)

    # Save the configuration
    save_config(custom_config, os.path.join(timestamped_base_dir, 'configuration.txt'))

    # Setup memory monitoring
    memory_monitor = MemoryMonitor(log_file=os.path.join(timestamped_base_dir, 'memory_usage.log'))
    memory_monitor.start()

    try:
        print("\nStarting PINN Architecture Optimization")
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
            print("\nInitializing Layers/Neurons Optimizer...")
            optimizer = LayersNeuronsOptimizer(
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
                print(f"\nResults saved in: {timestamped_base_dir}")
            else:
                print("\nNo valid results found. Check logs for errors.")

        except Exception as e:
            print(f"\nError during optimization: {str(e)}")
            import traceback
            traceback.print_exc()

    finally:
        # Stop memory monitoring
        memory_monitor.stop()
        print(f"\nAll output saved to: {timestamped_base_dir}")

if __name__ == "__main__":
    main()
