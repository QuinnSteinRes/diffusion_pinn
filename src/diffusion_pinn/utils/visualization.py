# Standard library imports
import os
from typing import List, Dict, Union, Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_solutions_and_error(pinn: 'DiffusionPINN',
                           data_processor: 'DiffusionDataProcessor',
                           t_indices: List[int],
                           save_path: str = None) -> None:
    """
    Plot true vs predicted solutions and error at specified time points

    Args:
        pinn: Trained PINN model
        data_processor: Data processor containing the true solution
        t_indices: List of time indices to plot
        save_path: Optional path to save the figure
    """
    try:
        fig, axes = plt.subplots(3, len(t_indices), figsize=(5*len(t_indices), 12))

        # Get meshgrid for plotting
        Y, X = np.meshgrid(data_processor.y, data_processor.x)
        t_vals = data_processor.t.flatten()

        # Pre-calculate all solutions and errors for global min/max
        solutions_true = []
        solutions_pred = []
        errors = []

        for t_idx in t_indices:
            t_val = t_vals[t_idx]

            # Create input points grid
            input_points = np.hstack([
                X.flatten()[:,None],
                Y.flatten()[:,None],
                np.ones_like(X.flatten()[:,None]) * t_val
            ])

            # Get predictions and reshape to match grid
            pred = pinn.predict(input_points)
            pred_reshaped = pred.numpy().reshape(X.shape)

            # Get true solution for this time step
            true = data_processor.usol[:,:,t_idx]

            # Calculate error
            error = np.abs(pred_reshaped - true)

            solutions_true.append(true)
            solutions_pred.append(pred_reshaped)
            errors.append(error)

        # Get global min/max values
        vmin_solution = min(np.min(solutions_true), np.min(solutions_pred))
        vmax_solution = max(np.max(solutions_true), np.max(solutions_pred))
        vmax_error = np.max(errors)

        for idx, t_idx in enumerate(t_indices):
            t_val = t_vals[t_idx]

            # Plot true solution
            im1 = axes[0,idx].pcolormesh(X, Y, solutions_true[idx],
                                        vmin=vmin_solution, vmax=vmax_solution,
                                        shading='auto')
            axes[0,idx].set_title(f't = {t_val:.3f} (True)')
            if idx == len(t_indices)-1:
                divider = make_axes_locatable(axes[0,idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax)

            # Plot predicted solution
            im2 = axes[1,idx].pcolormesh(X, Y, solutions_pred[idx],
                                        vmin=vmin_solution, vmax=vmax_solution,
                                        shading='auto')
            axes[1,idx].set_title(f't = {t_val:.3f} (Predicted)')
            if idx == len(t_indices)-1:
                divider = make_axes_locatable(axes[1,idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im2, cax=cax)

            # Plot error
            im3 = axes[2,idx].pcolormesh(X, Y, errors[idx],
                                        vmin=0, vmax=vmax_error,
                                        shading='auto')
            axes[2,idx].set_title(f't = {t_val:.3f} (Error)')
            if idx == len(t_indices)-1:
                divider = make_axes_locatable(axes[2,idx])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im3, cax=cax)

            # Set equal aspect ratio and add labels
            for ax_row in axes:
                ax_row[idx].set_aspect('equal')
                ax_row[idx].set_xlabel('x')
                ax_row[idx].set_ylabel('y')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"Error in plot_solutions_and_error: {str(e)}")
        raise

def plot_loss_history(losses: Union[List[Dict], Dict],
                     save_dir: str = 'results',
                     save_data: bool = True) -> None:
    """
    Enhanced loss history plotting that handles both list and dict formats

    Args:
        losses: Either List[Dict] or Dict containing loss values
        save_dir: Directory to save plots and data
        save_data: Whether to also save CSV data
    """
    try:
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
            # Handle dict format
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

    except Exception as e:
        print(f"Error in plot_loss_history: {str(e)}")
        raise

def plot_diffusion_convergence(d_history: List[float],
                             save_dir: str = 'results',
                             save_data: bool = True) -> None:
    """
    Plot diffusion coefficient convergence during training

    Args:
        d_history: List of diffusion coefficients during training
        save_dir: Directory to save plots and data
        save_data: Whether to also save CSV data
    """
    try:
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

    except Exception as e:
        print(f"Error in plot_diffusion_convergence: {str(e)}")
        raise