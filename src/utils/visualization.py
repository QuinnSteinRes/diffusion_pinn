# visualization.py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List, Dict
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Core plotting functions
def plot_solutions_and_error(pinn, data_processor, t_indices, save_path=None):
    """Original detailed solution plotting function"""
    [Your existing implementation]

def plot_loss_history(losses, save_dir='results', save_data=True):
    """
    Enhanced loss history plotting that combines both implementations

    Args:
        losses: Either List[Dict] (from original) or Dict (from plot_results)
        save_dir: Directory to save plots and data
        save_data: Whether to also save CSV data
    """
    os.makedirs(save_dir, exist_ok=True)

    # Handle both input formats
    if isinstance(losses, list):
        # Convert list of dicts to dataframe format
        epochs = range(len(losses))
        df = pd.DataFrame({'epoch': epochs})
        components = ['initial', 'boundary', 'interior', 'physics', 'total']
        for component in components:
            try:
                df[component] = [loss.get(component, np.nan) for loss in losses]
            except Exception as e:
                print(f"Warning: Could not process {component} loss: {str(e)}")
    else:
        # Handle dict format from plot_results version
        epochs = list(range(len(next(iter(losses.values())))))
        df = pd.DataFrame({'epoch': epochs})
        for loss_name, values in losses.items():
            df[loss_name] = values

    # Save data if requested
    if save_data:
        df.to_csv(os.path.join(save_dir, 'loss_data.csv'), index=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in df.columns:
        if column != 'epoch':
            ax.semilogy(df['epoch'], df[column], label=column)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(True)

    # Save plot
    fig.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_diffusion_convergence(d_history, save_dir='results', save_data=True):
    """
    Enhanced diffusion coefficient convergence plotting
    Combines functionality from both implementations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save data if requested
    if save_data:
        d_df = pd.DataFrame({
            'iteration': range(len(d_history)),
            'value': d_history
        })
        d_df.to_csv(os.path.join(save_dir, 'd_history.csv'), index=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(d_history)), d_history, 'b-', label='Predicted D')
    ax.axhline(y=d_history[-1], color='r', linestyle='--', label='Final Value')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diffusion Coefficient')
    ax.set_title('Convergence of Diffusion Coefficient')
    ax.legend()
    ax.grid(True)

    # Save plot
    fig.savefig(os.path.join(save_dir, 'd_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)