#!/usr/bin/env python
# optimize_layers_neurons.py - Focused optimization script for layers and neurons only

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
import json

print("Basic imports successful")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Try importing diffusion_pinn modules
from diffusion_pinn.data.processor import DiffusionDataProcessor
from diffusion_pinn.config import DiffusionConfig
from diffusion_pinn.models.pinn import DiffusionPINN
from diffusion_pinn.variables import PINN_VARIABLES

print("diffusion_pinn modules imported successfully")

def create_output_dirs(base_dir):
    """Create output directories"""
    dirs = {
        'results': os.path.join(base_dir, "results"),
        'models': os.path.join(base_dir, "models"),
        'logs': os.path.join(base_dir, "logs")
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

def train_and_evaluate(pinn, data, network_epochs, save_dir):
    """Train and evaluate the PINN model"""
    from diffusion_pinn.training.trainer import train_pinn
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    # Train model
    D_history, loss_history = train_pinn(
        pinn=pinn,
        data=data,
        optimizer=optimizer,
        epochs=network_epochs,
        save_dir=save_dir
    )
    
    # Get final values
    if len(loss_history) > 0:
        if isinstance(loss_history[-1], dict):
            final_loss = loss_history[-1].get('total', float('inf'))
        else:
            final_loss = loss_history[-1]
    else:
        final_loss = float('inf')
    
    # Get diffusion coefficient
    if len(D_history) > 0:
        final_D = D_history[-1]
    else:
        final_D = 0.0
    
    return final_loss, final_D, D_history, loss_history

def create_model(layers, neurons, data_processor, initial_D=0.0001):
    """Create a PINN model with the specified number of layers and neurons"""
    # Get domain information
    domain_info = data_processor.get_domain_info()
    
    # Create PINN configuration
    config = DiffusionConfig(
        hidden_layers=[neurons] * layers,
        activation='tanh',  # Fixed to tanh as it generally works well for PDEs
        initialization='glorot',
        diffusion_trainable=True,
        use_physics_loss=True
    )
    
    # Create PINN model
    pinn = DiffusionPINN(
        spatial_bounds=domain_info['spatial_bounds'],
        time_bounds=domain_info['time_bounds'],
        initial_D=initial_D,
        config=config
    )
    
    return pinn

def main():
    parser = argparse.ArgumentParser(description='Optimize PINN layers and neurons')
    parser.add_argument('--input-file', type=str, 
                      default='intensity_time_series_spatial_temporal.csv',
                      help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, 
                      default='optimization_results',
                      help='Base directory for output')
    parser.add_argument('--epochs', type=int, default=5000,
                      help='Number of training epochs per configuration')
    parser.add_argument('--max-layers', type=int, default=5,
                      help='Maximum number of layers to test')
    parser.add_argument('--initial-d', type=float, default=0.0001,
                      help='Initial diffusion coefficient value')
    args = parser.parse_args()
    
    # Create output directories
    dirs = create_output_dirs(args.output_dir)
    
    # Configuration for grid search
    layers_range = list(range(1, args.max_layers + 1))  # From 1 to max_layers
    neurons_range = [8, 16, 32, 64, 128]  # Common neuron counts
    
    # Save configuration
    config = {
        'input_file': args.input_file,
        'epochs_per_model': args.epochs,
        'layers_range': layers_range,
        'neurons_range': neurons_range,
        'initial_diffusion': args.initial_d,
        'activation': 'tanh'  # Fixed parameter
    }
    save_config(config, os.path.join(args.output_dir, 'search_config.txt'))
    
    try:
        # Load and preprocess data
        print("\nLoading data...")
        if not os.path.exists(args.input_file):
            print(f"Error: Data file not found: {args.input_file}")
            return
        
        data_processor = DiffusionDataProcessor(
            args.input_file,
            normalize_spatial=True
        )
        
        # Prepare training data
        print("Preparing training data...")
        training_data = data_processor.prepare_training_data(
            N_u=PINN_VARIABLES['N_u'],
            N_f=PINN_VARIABLES['N_f'],
            N_i=PINN_VARIABLES['N_i']
        )
        
        # Results tracking
        results = []
        best_loss = float('inf')
        best_config = None
        
        # Start grid search
        print("\nStarting grid search for layers and neurons")
        print(f"Testing {len(layers_range)} layer configurations and {len(neurons_range)} neuron configurations")
        print(f"Total configurations to test: {len(layers_range) * len(neurons_range)}")
        
        start_time = time.time()
        
        for layers in layers_range:
            for neurons in neurons_range:
                config_name = f"layers_{layers}_neurons_{neurons}"
                print(f"\n{'='*40}")
                print(f"Testing configuration: {config_name}")
                print(f"{'='*40}")
                
                # Create trial directory
                trial_dir = os.path.join(dirs['logs'], config_name)
                os.makedirs(trial_dir, exist_ok=True)
                
                # Create model
                pinn = create_model(
                    layers=layers, 
                    neurons=neurons, 
                    data_processor=data_processor,
                    initial_D=args.initial_d
                )
                
                # Train and evaluate
                try:
                    loss, diffusion_coeff, D_history, loss_history = train_and_evaluate(
                        pinn=pinn,
                        data=training_data,
                        network_epochs=args.epochs,
                        save_dir=trial_dir
                    )
                    
                    # Save histories
                    np.save(os.path.join(trial_dir, 'loss_history.npy'), loss_history)
                    np.save(os.path.join(trial_dir, 'D_history.npy'), D_history)
                    
                    # Track results
                    result = {
                        'layers': layers,
                        'neurons': neurons,
                        'loss': float(loss),
                        'diffusion_coefficient': float(diffusion_coeff),
                        'parameter_count': sum(p.numpy().size for p in pinn.get_trainable_variables())
                    }
                    results.append(result)
                    
                    # Update best model if better
                    if loss < best_loss:
                        best_loss = loss
                        best_config = result.copy()
                        # Save best model
                        try:
                            best_model_path = os.path.join(dirs['models'], 'best_model.h5')
                            pinn.save(best_model_path)
                            print(f"New best model saved with loss: {loss:.6f}")
                        except Exception as e:
                            print(f"Warning: Could not save best model: {str(e)}")
                    
                    # Save current results
                    with open(os.path.join(dirs['results'], 'search_results.json'), 'w') as f:
                        json.dump({
                            'results': results,
                            'best_config': best_config
                        }, f, indent=2)
                    
                    print(f"Results for {config_name}:")
                    print(f"  Loss: {loss:.6f}")
                    print(f"  Diffusion coefficient: {diffusion_coeff:.6f}")
                    print(f"  Parameter count: {result['parameter_count']}")
                    
                except Exception as e:
                    print(f"Error training configuration {config_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Clean up
                del pinn
                tf.keras.backend.clear_session()
                gc.collect()
        
        # Final summary
        end_time = time.time()
        duration_mins = (end_time - start_time) / 60
        
        print("\n" + "="*60)
        print("Grid Search Complete")
        print(f"Total time: {duration_mins:.2f} minutes")
        print(f"Configurations tested: {len(results)}/{len(layers_range) * len(neurons_range)}")
        
        if best_config:
            print("\nBest Configuration:")
            print(f"  Layers: {best_config['layers']}")
            print(f"  Neurons per layer: {best_config['neurons']}")
            print(f"  Loss: {best_config['loss']:.6f}")
            print(f"  Diffusion coefficient: {best_config['diffusion_coefficient']:.6f}")
            print(f"  Parameter count: {best_config['parameter_count']}")
        
        # Create summary file
        with open(os.path.join(args.output_dir, 'optimization_summary.txt'), 'w') as f:
            f.write("Layers and Neurons Optimization Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total configurations tested: {len(results)}/{len(layers_range) * len(neurons_range)}\n")
            f.write(f"Total time: {duration_mins:.2f} minutes\n\n")
            
            if best_config:
                f.write("Best Configuration:\n")
                f.write(f"  Layers: {best_config['layers']}\n")
                f.write(f"  Neurons per layer: {best_config['neurons']}\n")
                f.write(f"  Loss: {best_config['loss']:.6f}\n")
                f.write(f"  Diffusion coefficient: {best_config['diffusion_coefficient']:.6f}\n")
                f.write(f"  Parameter count: {best_config['parameter_count']}\n\n")
            
            f.write("All Results (sorted by loss):\n")
            f.write("-" * 60 + "\n")
            
            # Sort results by loss
            sorted_results = sorted(results, key=lambda x: x['loss'])
            for i, result in enumerate(sorted_results):
                f.write(f"Rank {i+1}:\n")
                f.write(f"  Layers: {result['layers']}\n")
                f.write(f"  Neurons: {result['neurons']}\n")
                f.write(f"  Loss: {result['loss']:.6f}\n")
                f.write(f"  Diffusion: {result['diffusion_coefficient']:.6f}\n")
                f.write(f"  Parameters: {result['parameter_count']}\n")
                f.write("\n")
        
        # Generate a quick plot if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            # Convert results to arrays for plotting
            layers_arr = np.array([r['layers'] for r in results])
            neurons_arr = np.array([r['neurons'] for r in results])
            loss_arr = np.array([r['loss'] for r in results])
            
            # Create a matrix for heatmap
            unique_layers = sorted(set(layers_arr))
            unique_neurons = sorted(set(neurons_arr))
            
            loss_matrix = np.ones((len(unique_layers), len(unique_neurons))) * np.nan
            
            for i, layer in enumerate(unique_layers):
                for j, neuron in enumerate(unique_neurons):
                    matching = [(l, n, loss) for l, n, loss in zip(layers_arr, neurons_arr, loss_arr) 
                               if l == layer and n == neuron]
                    if matching:
                        loss_matrix[i, j] = matching[0][2]
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.imshow(loss_matrix, cmap='viridis_r')
            plt.colorbar(label='Loss')
            plt.xticks(range(len(unique_neurons)), unique_neurons)
            plt.yticks(range(len(unique_layers)), unique_layers)
            plt.xlabel('Neurons per Layer')
            plt.ylabel('Number of Layers')
            plt.title('Loss Landscape for Layer and Neuron Configurations')
            
            # Add text annotations
            for i in range(len(unique_layers)):
                for j in range(len(unique_neurons)):
                    if not np.isnan(loss_matrix[i, j]):
                        plt.text(j, i, f'{loss_matrix[i, j]:.4f}', 
                                ha='center', va='center', 
                                color='white' if loss_matrix[i, j] > np.nanmean(loss_matrix) else 'black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(dirs['results'], 'loss_heatmap.png'), dpi=300)
            print(f"Loss heatmap saved to {os.path.join(dirs['results'], 'loss_heatmap.png')}")
            
        except Exception as e:
            print(f"Warning: Could not generate plot: {str(e)}")
        
        print("\nOptimization complete!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
