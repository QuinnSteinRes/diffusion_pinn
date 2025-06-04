#!/usr/bin/env python3
# Hybrid pinn_trainer.py combining V0.2.14 stability with V0.2.22 improvements

import os
import sys
import gc
import time
import signal
import argparse
import traceback
from pathlib import Path
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

# Configure TensorFlow for optimal performance
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Use non-interactive backend when no display
if not sys.stdout.isatty():
    matplotlib.use('Agg')

# Add package to path
if os.path.exists('/state/partition1/home/qs8/projects/diffusion_pinn'):
    sys.path.append('/state/partition1/home/qs8/projects/diffusion_pinn')

# Import diffusion_pinn modules
print("Importing diffusion_pinn...")
import diffusion_pinn
from diffusion_pinn import DiffusionConfig
from diffusion_pinn import DiffusionPINN, DiffusionDataProcessor
from diffusion_pinn.training import train_pinn, create_and_initialize_pinn
from diffusion_pinn.utils.visualization import (
    plot_solutions_and_error,
    plot_loss_history,
    plot_diffusion_convergence
)
from diffusion_pinn.variables import PINN_VARIABLES

print("diffusion_pinn location:", diffusion_pinn.__file__)

class MemoryMonitor:
    """Simple memory monitoring for debugging"""
    def __init__(self, log_file='memory_usage.log', interval=10):
        self.log_file = log_file
        self.interval = interval
        self.is_running = False
        self.monitor_thread = None

        with open(self.log_file, 'w') as f:
            f.write("timestamp,rss_mb,vms_mb,available_mb,system_percent\n")

    def _log_memory(self):
        process = psutil.Process(os.getpid())
        while self.is_running:
            try:
                mem_info = process.memory_info()
                sys_mem = psutil.virtual_memory()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                with open(self.log_file, 'a') as f:
                    f.write(f"{timestamp},{mem_info.rss/1024/1024:.2f},"
                           f"{mem_info.vms/1024/1024:.2f},"
                           f"{sys_mem.available/1024/1024:.2f},"
                           f"{sys_mem.percent}\n")

                if mem_info.rss > 10 * 1024 * 1024 * 1024:  # Over 10GB
                    gc.collect()

                time.sleep(self.interval)
            except Exception as e:
                print(f"Memory monitoring error: {str(e)}")
                time.sleep(self.interval)

    def start(self):
        import threading
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._log_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Memory monitoring started, logging to {self.log_file}")

    def stop(self):
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            print("Memory monitoring stopped")

def setup_signal_handlers(log_file='crash_log.txt'):
    """Setup signal handlers for debugging crashes"""
    def signal_handler(sig, frame):
        signal_name = signal.Signals(sig).name
        with open(log_file, 'a') as f:
            f.write(f"\n--- Received signal {signal_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write("Stack trace:\n")
            traceback.print_stack(frame, file=f)

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            f.write(f"\nMemory: RSS={mem_info.rss/(1024*1024):.1f}MB, VMS={mem_info.vms/(1024*1024):.1f}MB\n")

        if sig in (signal.SIGTERM, signal.SIGINT):
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        signal.signal(signal.SIGSEGV, signal_handler)
    except:
        pass

    print(f"Signal handlers set up, logging to {log_file}")

def preprocess_data(input_file):
    """Preprocess the CSV data"""
    try:
        print(f"Preprocessing data file: {input_file}")
        data = np.genfromtxt(input_file, delimiter=',', names=True)

        expected_columns = ['x', 'y', 't', 'intensity']
        if not all(col in data.dtype.names for col in expected_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(expected_columns)}")

        temp_file = input_file.replace('.csv', '_processed.csv')

        with open(temp_file, 'w') as f:
            f.write('x,y,t,intensity\n')

        batch_size = 20000
        total_rows = len(data)

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = data[start_idx:end_idx]

            with open(temp_file, 'ab') as f:
                np.savetxt(f,
                           np.column_stack((batch['x'], batch['y'], batch['t'], batch['intensity'])),
                           delimiter=',',
                           fmt='%.8f')

            del batch
            gc.collect()

        print(f"Data preprocessing complete: {temp_file}")
        return temp_file

    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        traceback.print_exc()
        raise

def setup_directories(base_dir):
    """Create output directories"""
    results_dir = os.path.join(base_dir, "results")
    save_dir = os.path.join(base_dir, "saved_models")
    log_dir = os.path.join(base_dir, "tensorboard_data")

    for dir_path in [results_dir, save_dir, log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    return results_dir, save_dir, log_dir

def save_summary(base_dir, D_final, loss_history, converged=False):
    """Save summary statistics"""
    try:
        total_losses = [loss.get('total', 0) for loss in loss_history]

        summary_data = {
            'final_diffusion_coefficient': [D_final],
            'converged': [converged],
            'final_loss': [total_losses[-1] if total_losses else None],
            'mean_loss': [np.mean(total_losses) if total_losses else None],
            'min_loss': [np.min(total_losses) if total_losses else None],
            'max_loss': [np.max(total_losses) if total_losses else None],
            'total_epochs': [len(loss_history)]
        }

        summary_file = os.path.join(base_dir, "training_summary.csv")
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)

        with open(os.path.join(base_dir, "training_summary.txt"), 'w') as f:
            f.write("Training Summary\n")
            f.write("===============\n\n")
            f.write(f"Final diffusion coefficient: {D_final:.8f}\n")
            f.write(f"Converged: {converged}\n")
            if total_losses:
                f.write(f"Final loss: {total_losses[-1]:.8f}\n")
                f.write(f"Mean loss: {np.mean(total_losses):.8f}\n")
                f.write(f"Min loss: {np.min(total_losses):.8f}\n")
                f.write(f"Max loss: {np.max(total_losses):.8f}\n")
            f.write(f"Total epochs: {len(loss_history)}\n")

        print(f"Summary saved to {summary_file}")

    except Exception as e:
        print(f"Warning: Could not save summary statistics: {str(e)}")
        traceback.print_exc()

def check_convergence(D_history, threshold=0.01, window=100):
    """Check if diffusion coefficient has converged"""
    if len(D_history) < window:
        return False

    recent_values = D_history[-window:]
    mean_value = np.mean(recent_values)
    std_value = np.std(recent_values)

    is_converged = (std_value / mean_value < threshold) if mean_value > 0 else False

    print(f"Convergence check - Mean D: {mean_value:.6f}, Std Dev: {std_value:.6f}")
    print(f"Relative variation: {(std_value/mean_value):.6f} (threshold: {threshold})")
    print(f"Convergence status: {is_converged}")

    return is_converged

def main(args):
    """Main training function with hybrid approach"""
    start_time = time.time()
    print("\n" + "="*50)
    print(f"Starting HYBRID PINN training - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print("="*50 + "\n")

    # Set random seeds
    if args.seed is not None:
        print(f"Setting random seeds to {args.seed}")
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    else:
        default_seed = PINN_VARIABLES['random_seed']
        print(f"Using default seed from variables.py: {default_seed}")
        tf.random.set_seed(default_seed)
        np.random.seed(default_seed)
        args.seed = default_seed

    # Set up signal handlers and memory monitoring
    setup_signal_handlers(os.path.join(args.output_dir, 'crash_log.txt'))

    memory_monitor = MemoryMonitor(
        log_file=os.path.join(args.output_dir, 'memory_usage.log'),
        interval=20
    )
    memory_monitor.start()

    try:
        # Setup directories
        results_dir, save_dir, log_dir = setup_directories(args.output_dir)
        gc.collect()

        # Preprocess the input data
        processed_file = preprocess_data(args.input_file)

        # Create and initialize PINN with hybrid approach
        print("\nInitializing HYBRID PINN model...")
        pinn, data = create_and_initialize_pinn(
            inputfile=processed_file,
            N_u=PINN_VARIABLES['N_u'],
            N_f=PINN_VARIABLES['N_f'],
            N_i=PINN_VARIABLES['N_i'],
            initial_D=PINN_VARIABLES['initial_D'],
            seed=args.seed
        )

        print(f"PINN initialized with log(D) parameterization")
        print(f"Initial D: {pinn.get_diffusion_coefficient():.8e}")
        print(f"Initial log(D): {pinn.get_log_diffusion_coefficient():.6f}")

        # Create optimizer
        print("Creating optimizer...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=PINN_VARIABLES['learning_rate']
        )

        # Train the model using hybrid approach
        print("\n" + "="*50)
        print(f"Starting HYBRID training - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training for {args.epochs} epochs using two-phase approach with log(D)")
        print("="*50 + "\n")

        try:
            D_history, loss_history = train_pinn(
                pinn=pinn,
                data=data,
                optimizer=optimizer,
                epochs=args.epochs,
                save_dir=str(save_dir),
                seed=args.seed
            )

            if not D_history or not loss_history:
                raise RuntimeError("Training failed - empty history")

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()

            if 'pinn' in locals() and hasattr(pinn, 'get_diffusion_coefficient'):
                D_final = pinn.get_diffusion_coefficient()
                print(f"Final diffusion coefficient: {D_final:.6f}")
                save_summary(args.output_dir, D_final, [], converged=False)

            return 1

        # Clean up training data
        del data
        gc.collect()

        # Check convergence
        final_D = pinn.get_diffusion_coefficient()
        final_log_D = pinn.get_log_diffusion_coefficient()
        converged = check_convergence(D_history) if len(D_history) > 100 else False

        # Generate and save plots
        try:
            import matplotlib.pyplot as plt

            print("Generating plots...")

            # Plot loss history
            plot_loss_history(loss_history, save_dir=results_dir)
            plt.close('all')

            # Plot diffusion coefficient convergence
            plot_diffusion_convergence(D_history, save_dir=results_dir)
            plt.close('all')

            # Plot solutions at different time points
            print("Creating solution plots...")
            data_processor = DiffusionDataProcessor(processed_file, seed=args.seed)

            # Select time indices for plotting
            n_times = len(data_processor.t)
            if n_times >= 4:
                t_indices = [0, n_times//3, 2*n_times//3, -1]
            elif n_times >= 2:
                t_indices = [0, -1]
            else:
                t_indices = [0]

            plot_solutions_and_error(
                pinn=pinn,
                data_processor=data_processor,
                t_indices=t_indices,
                save_path=os.path.join(results_dir, 'final_solutions.png')
            )
            plt.close('all')

            print("Plots generated successfully")

        except Exception as e:
            print(f"Warning: Error during plotting: {str(e)}")
            traceback.print_exc()

        # Save summary statistics
        save_summary(args.output_dir, final_D, loss_history, converged)

        # Check for extreme diffusion values
        if final_D < 1e-8 or final_D > 1e-1:
            with open(os.path.join(args.output_dir, "warning.txt"), 'w') as f:
                f.write(f"WARNING: Final diffusion coefficient {final_D} may be outside expected range\n")
                f.write("This may indicate a convergence issue or data quality problem.\n")
                f.write(f"Final log(D): {final_log_D:.6f}\n")

        # Training completion summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"HYBRID TRAINING COMPLETED - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {elapsed_time/60:.2f} minutes")
        print(f"Final diffusion coefficient: {final_D:.8e}")
        print(f"Final log(D): {final_log_D:.6f}")
        print(f"Convergence status: {converged}")
        print(f"Total epochs: {len(D_history)}")

        # Convergence analysis
        if len(D_history) >= 100:
            recent_std = np.std(D_history[-100:])
            recent_mean = np.mean(D_history[-100:])
            rel_variation = recent_std / recent_mean if recent_mean > 0 else float('inf')
            print(f"Final relative variation: {rel_variation:.6f}")

        print("="*50 + "\n")

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        print("Stopping memory monitoring...")
        memory_monitor.stop()

        # Clean up processed file
        try:
            if 'processed_file' in locals() and os.path.exists(processed_file):
                os.remove(processed_file)
                print(f"Removed temporary file: {processed_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {str(e)}")

        # Final garbage collection
        gc.collect()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HYBRID PINN model for diffusion problem')
    parser.add_argument('--input-file', type=str, default=os.path.join(os.path.dirname(__file__), 'intensity_time_series_spatial_temporal.csv'),
                      help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default='.',
                      help='Base directory for output')
    parser.add_argument('--epochs', type=int, default=PINN_VARIABLES['epochs'],
                      help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=PINN_VARIABLES['random_seed'],
                      help='Random seed for reproducibility')

    args = parser.parse_args()
    sys.exit(main(args))