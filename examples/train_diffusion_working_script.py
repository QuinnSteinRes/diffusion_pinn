import matplotlib
import sys
import os
from pathlib import Path
import argparse
import diffusion_pinn
import tensorflow as tf
import pandas as pd
import numpy as np

# Only use 'Agg' backend when running without display
if not sys.stdout.isatty():
    matplotlib.use('Agg')

# Import configuration and components
from diffusion_pinn import DiffusionConfig
from diffusion_pinn import DiffusionPINN, DiffusionDataProcessor
from diffusion_pinn.training import train_pinn, create_and_initialize_pinn
from diffusion_pinn.utils.visualization import (
    plot_solutions_and_error,
    plot_loss_history,
    plot_diffusion_convergence
)

def preprocess_data(input_file):
    """Preprocess the CSV data with known columns: x, y, t, intensity"""
    # Read using numpy
    data = np.genfromtxt(input_file, delimiter=',', names=True)
    
    # Verify columns
    expected_columns = ['x', 'y', 't', 'intensity']
    if not all(col in data.dtype.names for col in expected_columns):
        raise ValueError(f"CSV must contain columns: {', '.join(expected_columns)}")
    
    # Save temporary processed file
    temp_file = input_file.replace('.csv', '_processed.csv')
    
    # Save header
    with open(temp_file, 'w') as f:
        f.write('x,y,t,intensity\n')
    
    # Save data
    with open(temp_file, 'ab') as f:
        np.savetxt(f, 
                   np.column_stack((data['x'], data['y'], data['t'], data['intensity'])),
                   delimiter=',',
                   fmt='%.8f')
    
    return temp_file

def setup_directories(base_dir):
    """Create and return directory paths for outputs"""
    results_dir = os.path.join(base_dir, "results")
    plot_dir = os.path.join(base_dir, "plots")
    save_dir = os.path.join(base_dir, "saved_models")
    
    # Create directories
    for dir_path in [results_dir, plot_dir, save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
    return results_dir, plot_dir, save_dir

def save_summary(base_dir, D_final, loss_history):
    """Save summary statistics"""
    try:
        # Create summary dictionary
        summary_data = {
            'final_diffusion_coefficient': [D_final],
            'final_loss': [loss_history[-1]],
            'mean_loss': [np.mean(loss_history)],
            'min_loss': [np.min(loss_history)],
            'max_loss': [np.max(loss_history)]
        }
        
        # Save to CSV
        summary_file = os.path.join(base_dir, "training_summary.csv")
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)
            
    except Exception as e:
        print("Warning: Could not save summary statistics:")
        print(str(e))

def main(args):
    print("\nStarting training")
    print("Input file:", args.input_file)
    
    # Create configuration
    config = DiffusionConfig()
    
    # Setup directories
    results_dir, plot_dir, save_dir = setup_directories(args.output_dir)
    
    print("Output directory: {}".format(args.output_dir))
    
    # Preprocess the input data
    processed_file = preprocess_data(args.input_file)
    
    # Create and initialize PINN
    pinn, data = create_and_initialize_pinn(
        inputfile=processed_file,
        N_u=1000,
        N_f=20000,
        N_i=10000,
        initial_D=1.0
    )
    
    # Create optimizer with learning rate decay
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.95
        )
    )
    
    # Train the model
    print("Starting training...")
    D_history, loss_history = train_pinn(
        pinn=pinn,
        data=data,
        optimizer=optimizer,
        epochs=args.epochs,
        save_dir=str(save_dir)
    )
    
    # Save plots
    try:
        import matplotlib.pyplot as plt
        plot_loss_history(loss_history)
        plt.savefig(os.path.join(plot_dir, 'loss_history.png'))
        plt.close()
        
        plot_diffusion_convergence(D_history)
        plt.savefig(os.path.join(plot_dir, 'd_convergence.png'))
        plt.close()
        
        # Plot solutions
        data_processor = DiffusionDataProcessor(processed_file)
        t_indices = [0, len(data_processor.t)//3, 2*len(data_processor.t)//3, -1]
        plot_solutions_and_error(
            pinn=pinn,
            data_processor=data_processor,
            t_indices=t_indices,
            save_path=os.path.join(plot_dir, 'final_solutions.png')
        )
    except Exception as e:
        print("Warning: Error during plotting:")
        print(str(e))
    
    # Save summary
    final_D = pinn.get_diffusion_coefficient()
    save_summary(args.output_dir, final_D, loss_history)
    
    # Clean up processed file
    try:
        os.remove(processed_file)
    except:
        pass
    
    print("\nTraining completed")
    print("Final diffusion coefficient: {}".format(final_D))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PINN model for diffusion problem')
    parser.add_argument('--input-file', type=str, default='intensity_time_series_spatial_temporal.csv',
                      help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default='.',
                      help='Base directory for output')
    parser.add_argument('--epochs', type=int, default=60000,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    main(args)
