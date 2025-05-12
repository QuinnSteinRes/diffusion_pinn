#!/usr/bin/env python
# optimize_bayesian.py - Simplified script with centralized configuration

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
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
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
        initial_D=config.get('initial_D', 0.001),
        config=pinn_config
    )

    # Create optimizer with gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=config['gradient_clip_norm'],
    )

    return pinn, optimizer

def train_and_evaluate(pinn, optimizer, data, trial_dir, config):
    """Train and evaluate a PINN model"""
    from diffusion_pinn.training.trainer import train_pinn

    try:
        # Setup callbacks
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=config['earlyStop_patience'],
                mode='min',
                restore_best_weights=True,
                min_delta=1e-4
            ),
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=config['lr_reduction_factor'],
                patience=config['lr_patience'],
                min_lr=1e-8,
                verbose=1
            ),
            # Terminate on NaN
            tf.keras.callbacks.TerminateOnNaN()
        ]

        # Train model
        D_history, loss_history = train_pinn(
            pinn=pinn,
            data=data,
            optimizer=optimizer,
            epochs=config['network_epochs'],
            save_dir=trial_dir,
        )

        # Check for valid loss
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

        # Save histories
        np.save(os.path.join(trial_dir, 'loss_history.npy'), loss_history)
        np.save(os.path.join(trial_dir, 'D_history.npy'), D_history)

        # Plot results for this trial
        plots_dir = os.path.join(trial_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

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

class BayesianOptimizer:
    """Bayesian optimization for PINN hyperparameters"""

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
            N_u=config['N_u'],
            N_f=config['N_f'],
            N_i=config['N_i']
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
                n_random_starts=int(self.config['iterations_optimizer'] * self.config['random_starts_fraction']),
                x0=x0,
                acq_func=self.config['acquisitionFunction'],
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
            f.write("Optimization Results\n")
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

        # Parameter influence plots
        param_names = ['layers', 'neurons', 'learning_rate']
        for param in param_names:
            plt.figure(figsize=(12, 6))
            param_values = [p.get(param, 0) for p in valid_params]

            if param == 'learning_rate':
                plt.semilogx(param_values, valid_losses, 'o')
            else:
                plt.plot(param_values, valid_losses, 'o')

            plt.title(f'Loss vs {param}')
            plt.xlabel(param)
            plt.ylabel('Validation Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f'param_importance_{param}.png'))
            plt.close()

        # Activation function comparison (bar chart)
        plt.figure(figsize=(10, 6))
        activations = {}
        for i, loss in enumerate(valid_losses):
            act = valid_params[i].get('activation', 'unknown')
            if act not in activations:
                activations[act] = []
            activations[act].append(loss)

        act_names = list(activations.keys())
        act_mean_losses = [np.mean(activations[act]) for act in act_names]

        plt.bar(act_names, act_mean_losses)
        plt.title('Average Loss by Activation Function')
        plt.ylabel('Average Loss')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(plots_dir, 'activation_comparison.png'))
        plt.close()

def analyze_results(results_dir):
    """Analyze existing optimization results"""
    # Look for trial directories
    logs_dir = os.path.join(results_dir, "logs")
    if not os.path.exists(logs_dir):
        print(f"No logs directory found at {logs_dir}")
        return

    # Collect data from each trial
    trials = {}
    for trial_dir in os.listdir(logs_dir):
        trial_path = os.path.join(logs_dir, trial_dir)
        if not os.path.isdir(trial_path):
            continue

        loss_path = os.path.join(trial_path, "loss_history.npy")
        d_path = os.path.join(trial_path, "D_history.npy")

        if os.path.exists(loss_path) and os.path.exists(d_path):
            try:
                # Load histories
                loss_history = np.load(loss_path, allow_pickle=True)
                d_history = np.load(d_path)

                # Extract final values
                if len(loss_history) > 0:
                    if isinstance(loss_history[-1], dict):
                        final_loss = loss_history[-1].get('total', float('inf'))
                    else:
                        final_loss = float(loss_history[-1])
                else:
                    final_loss = float('inf')

                final_d = float(d_history[-1]) if len(d_history) > 0 else 0.0

                # Extract parameters from trial name
                params = {}
                parts = trial_dir.replace('trial_', '').split('_')
                for part in parts:
                    if part.startswith('l'):
                        params['layers'] = int(part[1:])
                    elif part.startswith('n'):
                        params['neurons'] = int(part[1:])
                    elif part.startswith('a'):
                        params['activation'] = part[1:]
                    elif part.startswith('lr'):
                        params['learning_rate'] = float(part[2:])

                # Store valid results
                if not np.isnan(final_loss) and not np.isinf(final_loss):
                    trials[trial_dir] = {
                        'loss': final_loss,
                        'D': final_d,
                        'params': params
                    }
            except Exception as e:
                print(f"Error processing {trial_dir}: {str(e)}")

    if not trials:
        print("No valid trials found")
        return

    # Sort by loss
    sorted_trials = sorted(trials.items(), key=lambda x: x[1]['loss'])

    # Print results
    print("\nTrial Results (sorted by loss):")
    print("-" * 80)
    for i, (trial_name, results) in enumerate(sorted_trials):
        params = results['params']
        param_str = f"l={params.get('layers', '?')} n={params.get('neurons', '?')} a={params.get('activation', '?')} lr={params.get('learning_rate', '?'):.2e}"
        print(f"{i+1}. {param_str:<40} Loss: {results['loss']:.6f}  D: {results['D']:.6f}")

    # Save summary
    plots_dir = os.path.join(results_dir, "analysis")
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(plots_dir, "summary.txt"), "w") as f:
        f.write("Optimization Results Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write("Best trial: " + sorted_trials[0][0] + "\n")
        best_params = sorted_trials[0][1]['params']
        f.write("Best configuration:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"Best loss: {sorted_trials[0][1]['loss']:.6f}\n")
        f.write(f"Diffusion coefficient: {sorted_trials[0][1]['D']:.6f}\n\n")

        f.write("All trials (sorted by loss):\n")
        f.write("-" * 80 + "\n")
        for i, (trial_name, results) in enumerate(sorted_trials):
            params = results['params']
            param_str = f"l={params.get('layers', '?')} n={params.get('neurons', '?')} a={params.get('activation', '?')} lr={params.get('learning_rate', '?'):.2e}"
            f.write(f"{i+1}. {param_str:<40} Loss: {results['loss']:.6f}  D: {results['D']:.6f}\n")

    # Create comparison plots
    plt.figure(figsize=(12, 6))
    trial_names = [name for name, _ in sorted_trials[:min(5, len(sorted_trials))]]
    losses = [results['loss'] for _, results in sorted_trials[:min(5, len(sorted_trials))]]
    plt.bar(range(len(trial_names)), losses)
    plt.xticks(range(len(trial_names)), [name.replace('trial_', '') for name in trial_names], rotation=45, ha='right')
    plt.ylabel('Loss')
    plt.title('Top Trials by Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_trials.png'))
    plt.close()

    # Load and plot best trial details
    best_trial_dir = os.path.join(logs_dir, sorted_trials[0][0])
    loss_path = os.path.join(best_trial_dir, "loss_history.npy")
    d_path = os.path.join(best_trial_dir, "D_history.npy")

    if os.path.exists(loss_path) and os.path.exists(d_path):
        loss_history = np.load(loss_path, allow_pickle=True)
        d_history = np.load(d_path)

        # Plot loss history
        plt.figure(figsize=(10, 5))
        if isinstance(loss_history[0], dict):
            # Plot individual loss components
            components = list(loss_history[0].keys())
            for component in components:
                values = [loss.get(component, 0) for loss in loss_history]
                plt.semilogy(values, label=component)
            plt.legend()
        else:
            # Plot single loss value
            plt.semilogy(loss_history)
        plt.title('Loss History for Best Trial')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'best_trial_loss.png'))
        plt.close()

        # Plot D history
        plt.figure(figsize=(10, 5))
        plt.plot(d_history)
        plt.title('Diffusion Coefficient History for Best Trial')
        plt.xlabel('Epoch')
        plt.ylabel('Diffusion Coefficient')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'best_trial_d.png'))
        plt.close()

    print(f"\nAnalysis complete. Results saved to: {plots_dir}")
    return sorted_trials[0][1]['params']

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
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results without running optimization')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from a previous optimization run')
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

    # Only analyze previous results if requested
    if args.analyze_only:
        print("Analyzing existing results...")
        analyze_results(args.output_dir)
        return

    # Create output directories with timestamp
    dirs, timestamped_base_dir = create_output_dirs(args.output_dir)

    # Save the configuration
    save_config(custom_config, os.path.join(timestamped_base_dir, 'configuration.txt'))

    # Setup memory monitoring
    memory_monitor = MemoryMonitor(log_file=os.path.join(timestamped_base_dir, 'memory_usage.log'))
    memory_monitor.start()

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
            optimizer = BayesianOptimizer(
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

            # Try to analyze partial results
            print("\nAttempting to analyze partial results...")
            analyze_results(dirs['results'])

    finally:
        # Stop memory monitoring
        memory_monitor.stop()
        print(f"\nAll output saved to: {timestamped_base_dir}")

if __name__ == "__main__":
    main()