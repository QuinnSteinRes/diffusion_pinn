#!/usr/bin/env python3
# Improved pinn_trainer.py with memory management, stability fixes, and version logging

import os
import sys
import gc
import time
import signal
import argparse
import traceback
import subprocess
from pathlib import Path
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

# Configure TensorFlow to grow GPU memory allocation (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")

# Configure CPU threading limits to avoid resource contention
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# When running without a display, use non-interactive backend
if not sys.stdout.isatty():
    matplotlib.use('Agg')

# Add the package to path if needed
if os.path.exists('/state/partition1/home/qs8/projects/diffusion_pinn'):
    sys.path.append('/state/partition1/home/qs8/projects/diffusion_pinn')

# Now import diffusion_pinn modules
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
print("train_pinn location:", train_pinn.__code__.co_filename)

def log_version_and_environment():
    """Log comprehensive version and environment information"""
    print("\n" + "=" * 80)
    print("PINN DIFFUSION COEFFICIENT ESTIMATION - VERSION INFORMATION")
    print("=" * 80)

    # Basic run information
    import datetime
    print(f"Start Time: {datetime.datetime.now()}")
    print(f"Job ID: {os.environ.get('JOB_ID', 'LOCAL')}")
    print(f"Hostname: {os.environ.get('HOSTNAME', 'unknown')}")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    print(f"Working Directory: {os.getcwd()}")

    print(f"\nSOFTWARE VERSIONS:")
    print("-" * 40)

    # Package versions
    try:
        print(f"diffusion_pinn version: {getattr(diffusion_pinn, '__version__', 'unknown')}")
        print(f"diffusion_pinn location: {diffusion_pinn.__file__}")
    except Exception as e:
        print(f"diffusion_pinn version: error - {e}")

    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")

    try:
        print(f"TensorFlow version: {tf.__version__}")
    except Exception as e:
        print(f"TensorFlow version: error - {e}")

    try:
        print(f"NumPy version: {np.__version__}")
    except Exception as e:
        print(f"NumPy version: error - {e}")

    try:
        print(f"Pandas version: {pd.__version__}")
    except Exception as e:
        print(f"Pandas version: error - {e}")

    print(f"\nGIT REPOSITORY STATUS:")
    print("-" * 40)

    # Git information
    git_repo_path = "/state/partition1/home/qs8/projects/diffusion_pinn"
    try:
        if os.path.exists(os.path.join(git_repo_path, '.git')):
            # Get commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  cwd=git_repo_path,
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            if result.returncode == 0:
                print(f"Git commit: {result.stdout.strip()}")
            else:
                print(f"Git commit: command failed - {result.stderr}")

            # Get branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                  cwd=git_repo_path,
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            if result.returncode == 0:
                print(f"Git branch: {result.stdout.strip()}")

            # Get commit message
            result = subprocess.run(['git', 'log', '-1', '--pretty=format:%s'],
                                  cwd=git_repo_path,
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            if result.returncode == 0:
                print(f"Commit message: {result.stdout.strip()}")

            # Get commit date
            result = subprocess.run(['git', 'log', '-1', '--pretty=format:%ci'],
                                  cwd=git_repo_path,
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            if result.returncode == 0:
                print(f"Commit date: {result.stdout.strip()}")

            # Check working directory status
            result = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD', '--'],
                                  cwd=git_repo_path,
                                  capture_output=True,
                                  timeout=10)
            if result.returncode == 0:
                print("Working directory: clean")
            else:
                print("Working directory: has uncommitted changes")
                # Show modified files
                result = subprocess.run(['git', 'diff', '--name-only'],
                                      cwd=git_repo_path,
                                      capture_output=True,
                                      text=True,
                                      timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    print("Modified files:")
                    for file in result.stdout.strip().split('\n'):
                        print(f"  {file}")

            # Get recent commits
            print("\nRecent commits:")
            result = subprocess.run(['git', 'log', '--oneline', '-5'],
                                  cwd=git_repo_path,
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    print(f"  {line}")

        else:
            print(f"Git repository: not found at {git_repo_path}")

    except subprocess.TimeoutExpired:
        print("Git commands: timed out")
    except Exception as e:
        print(f"Git status: error - {e}")

    print(f"\nRUNTIME ENVIRONMENT:")
    print("-" * 40)
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    print(f"PATH (first 3 entries): {':'.join(os.environ.get('PATH', '').split(':')[:3])}")

    # System information
    print(f"\nSYSTEM INFORMATION:")
    print("-" * 40)
    try:
        mem = psutil.virtual_memory()
        print(f"Total memory: {mem.total / (1024**3):.1f} GB")
        print(f"Available memory: {mem.available / (1024**3):.1f} GB")
        print(f"CPU count: {psutil.cpu_count()}")
    except Exception as e:
        print(f"System info error: {e}")

    print(f"\nPINN CONFIGURATION:")
    print("-" * 40)
    try:
        for key, value in sorted(PINN_VARIABLES.items()):
            print(f"{key}: {value}")
    except Exception as e:
        print(f"PINN variables error: {e}")

    print("=" * 80)
    print("END VERSION INFORMATION - STARTING TRAINING")
    print("=" * 80 + "\n")

# [Rest of the existing code remains the same - MemoryMonitor, setup_signal_handlers, etc.]

class MemoryMonitor:
    def __init__(self, log_file='memory_usage.log', interval=10):
        self.log_file = log_file
        self.interval = interval
        self.is_running = False
        self.monitor_thread = None

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("timestamp,rss_mb,vms_mb,available_mb,system_percent\n")

    def _log_memory(self):
        process = psutil.Process(os.getpid())
        while self.is_running:
            try:
                # Process memory
                mem_info = process.memory_info()

                # System memory
                sys_mem = psutil.virtual_memory()

                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                with open(self.log_file, 'a') as f:
                    f.write(f"{timestamp},{mem_info.rss/1024/1024:.2f},"
                           f"{mem_info.vms/1024/1024:.2f},"
                           f"{sys_mem.available/1024/1024:.2f},"
                           f"{sys_mem.percent}\n")

                # If memory usage is high, log more details and force GC
                if mem_info.rss > 10 * 1024 * 1024 * 1024:  # Over 10GB
                    gc.collect()
                    with open(f"{self.log_file}.alert", 'a') as f:
                        f.write(f"HIGH MEMORY ALERT: {timestamp} - "
                               f"RSS: {mem_info.rss/1024/1024:.2f}MB\n")

                time.sleep(self.interval)
            except Exception as e:
                print(f"Memory monitoring error: {str(e)}")
                with open(f"{self.log_file}.error", 'a') as f:
                    f.write(f"Error: {str(e)}\n")
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

# Setup signal handlers for better error reporting
def setup_signal_handlers(log_file='crash_log.txt'):
    def signal_handler(sig, frame):
        signal_name = signal.Signals(sig).name
        with open(log_file, 'a') as f:
            f.write(f"\n--- Received signal {signal_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write("Stack trace:\n")
            traceback.print_stack(frame, file=f)

            # Log memory info
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            f.write(f"\nMemory: RSS={mem_info.rss/(1024*1024):.1f}MB, VMS={mem_info.vms/(1024*1024):.1f}MB\n")

            # Log system memory
            f.write("\nSystem memory:\n")
            system_mem = psutil.virtual_memory()
            f.write(f"Total: {system_mem.total/(1024*1024):.1f}MB, "
                   f"Available: {system_mem.available/(1024*1024):.1f}MB, "
                   f"Used: {system_mem.used/(1024*1024):.1f}MB, "
                   f"Percent: {system_mem.percent}%\n")

        # For SIGTERM and SIGINT, exit after logging
        if sig in (signal.SIGTERM, signal.SIGINT):
            sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Can't catch SIGSEGV reliably in Python, but try anyway
    try:
        signal.signal(signal.SIGSEGV, signal_handler)
    except:
        pass

    print(f"Signal handlers set up, logging to {log_file}")

def preprocess_data(input_file):
    """Preprocess the CSV data with known columns: x, y, t, intensity"""
    try:
        print(f"Preprocessing data file: {input_file}")
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

        # Save data in batches to reduce memory usage
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

            # Clean up batch to free memory
            del batch
            gc.collect()

        print(f"Data preprocessing complete: {temp_file}")
        return temp_file

    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        traceback.print_exc()
        raise

def setup_directories(base_dir):
    """Create and return directory paths for outputs"""
    results_dir = os.path.join(base_dir, "results")
    save_dir = os.path.join(base_dir, "saved_models")
    log_dir = os.path.join(base_dir, "tensorboard_data")

    # Create directories
    for dir_path in [results_dir, save_dir, log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    return results_dir, save_dir, log_dir

def save_summary(base_dir, D_final, loss_history, converged=False):
    """Save summary statistics"""
    try:
        # Create summary dictionary
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

        # Save to CSV
        summary_file = os.path.join(base_dir, "training_summary.csv")
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)

        # Also save as text for easier viewing
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

def check_convergence(D_history, threshold=0.001, window=100):
    """Check if diffusion coefficient has converged"""
    if len(D_history) < window:
        return False

    recent_values = D_history[-window:]
    mean_value = np.mean(recent_values)
    std_value = np.std(recent_values)

    # Converged if standard deviation is small relative to mean
    is_converged = (std_value / mean_value < threshold)

    # Print convergence metrics
    print(f"Convergence check - Mean D: {mean_value:.6f}, Std Dev: {std_value:.6f}")
    print(f"Relative variation: {(std_value/mean_value):.6f} (threshold: {threshold})")
    print(f"Convergence status: {is_converged}")

    return is_converged

def main(args):
    """Main training function with improved error handling and memory management"""
    start_time = time.time()

    # Log version information FIRST
    log_version_and_environment()

    print("\n" + "="*50)
    print(f"Starting PINN training - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print("="*50 + "\n")

    if args.seed is not None:
        print(f"Setting random seeds to {args.seed}")
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    else:
        # Use default seed from PINN_VARIABLES
        default_seed = PINN_VARIABLES['random_seed']
        print(f"Using default seed from variables.py: {default_seed}")
        tf.random.set_seed(default_seed)
        np.random.seed(default_seed)

    # Set up signal handlers for crashes
    setup_signal_handlers(os.path.join(args.output_dir, 'crash_log.txt'))

    # Initialize memory monitor
    memory_monitor = MemoryMonitor(
        log_file=os.path.join(args.output_dir, 'memory_usage.log'),
        interval=20  # Check every 20 seconds
    )
    memory_monitor.start()

    try:
        # Setup directories
        results_dir, save_dir, log_dir = setup_directories(args.output_dir)

        # Force garbage collection before starting
        gc.collect()

        # Preprocess the input data
        processed_file = preprocess_data(args.input_file)

        # Create and initialize PINN
        print("\nInitializing PINN model...")
        pinn, data = create_and_initialize_pinn(
            inputfile=processed_file,
            N_u=PINN_VARIABLES['N_u'],
            N_f=PINN_VARIABLES['N_f'],
            N_i=PINN_VARIABLES['N_i'],
            initial_D=PINN_VARIABLES['initial_D'],
            seed=args.seed
        )

        # Create optimizer with learning rate decay
        print("Creating optimizer with learning rate decay...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=PINN_VARIABLES['learning_rate']
        )

        # Train the model
        print("\n" + "="*50)
        print(f"Starting training - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training for {args.epochs} epochs")
        print(f"Initial D = {pinn.get_diffusion_coefficient():.6f}")
        print("="*50 + "\n")

        # Use a try-except block for training to catch any errors
        try:
            D_history, loss_history = train_pinn(
                pinn=pinn,
                data=data,
                optimizer=optimizer,
                epochs=args.epochs,
                save_dir=str(save_dir)
            )

            # Check if training was successful
            if not D_history or not loss_history:
                raise RuntimeError("Training failed - empty history")

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()

            # Try to salvage what we can
            if 'pinn' in locals() and hasattr(pinn, 'get_diffusion_coefficient'):
                D_final = pinn.get_diffusion_coefficient()
                print(f"Final diffusion coefficient: {D_final:.6f}")
                save_summary(args.output_dir, D_final, [], converged=False)

            return 1

        # Clean up training data to free memory
        del data
        gc.collect()

        # Check convergence
        final_D = pinn.get_diffusion_coefficient()
        converged = check_convergence(D_history) if len(D_history) > 100 else False

        # Generate and save plots
        try:
            import matplotlib.pyplot as plt

            # Plot loss history
            plot_loss_history(loss_history, save_dir=results_dir)
            plt.close()

            # Plot diffusion coefficient convergence
            plot_diffusion_convergence(D_history, save_dir=results_dir)
            plt.close()

            # Plot solutions
            data_processor = DiffusionDataProcessor(processed_file)
            t_indices = [0, len(data_processor.t)//3, 2*len(data_processor.t)//3, -1]
            plot_solutions_and_error(
                pinn=pinn,
                data_processor=data_processor,
                t_indices=t_indices,
                save_path=os.path.join(results_dir, 'final_solutions.png')
            )
        except Exception as e:
            print(f"Warning: Error during plotting: {str(e)}")
            traceback.print_exc()

        # Save summary statistics
        save_summary(args.output_dir, final_D, loss_history, converged)

        # Special handling for extreme diffusion values
        if final_D < 1e-6 or final_D > 1e-2:
            with open(os.path.join(args.output_dir, "warning.txt"), 'w') as f:
                f.write(f"WARNING: Final diffusion coefficient {final_D} is outside expected range (1e-6 - 1e-2)\n")
                f.write("This may indicate a convergence issue or an issue with the data.\n")

        # Training completion
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"Training completed - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {elapsed_time/60:.2f} minutes")
        print(f"Final diffusion coefficient: {final_D:.8f}")
        print(f"Convergence status: {converged}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        traceback.print_exc()
        return 1

    finally:
        # Clean up
        print("Stopping memory monitoring...")
        memory_monitor.stop()

        # Clean up processed file
        try:
            if 'processed_file' in locals() and os.path.exists(processed_file):
                os.remove(processed_file)
                print(f"Removed temporary file: {processed_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {str(e)}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PINN model for diffusion problem')
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
