import matplotlib
matplotlib.use('TkAgg')  # Do this before importing plt
import matplotlib.pyplot as plt
from typing import List, Dict

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
        plt.savefig(save_path)
    plt.show()

def plot_diffusion_convergence(D_history: List[float], save_path: str = None) -> None:
    """
    Plot convergence of diffusion coefficient
    
    Args:
        D_history: List of diffusion coefficients during training
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 5))
    plt.plot(D_history, 'b-', label='Predicted D')
    plt.axhline(y=D_history[-1], color='r', linestyle='--', label='Final Value')
    plt.xlabel('Iteration')
    plt.ylabel('Diffusion Coefficient')
    plt.title('Convergence of Diffusion Coefficient')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_loss_history(loss_history: List[Dict[str, float]]) -> None:
    """
    Plot training loss history
    
    Args:
        loss_history: List of dictionaries containing loss components
    """
    if not loss_history:
        print("Error: loss_history is empty")
        return
        
    print(f"Number of epochs recorded: {len(loss_history)}")
    print("Keys in loss dictionary:", loss_history[0].keys())
    
    plt.figure(figsize=(12, 6))
    
    epochs = range(len(loss_history))
    components = ['initial', 'boundary', 'interior', 'physics', 'total']
    
    for component in components:
        try:
            values = [loss[component] for loss in loss_history]
            plt.semilogy(epochs, values, label=component)
            print(f"Successfully plotted {component} loss")
        except KeyError:
            print(f"Warning: Could not find {component} loss in history")
        except Exception as e:
            print(f"Error plotting {component} loss: {str(e)}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    plt.show()
