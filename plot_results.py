import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be called before importing plt
import matplotlib.pyplot as plt
import os

def plot_loss_history(losses, save_dir='plots'):
    """
    Plot training loss history for cluster environment
    
    Args:
        losses (dict): Dictionary containing different types of losses
        save_dir (str): Directory to save plots (will be created if doesn't exist)
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert losses to DataFrame
    epochs = list(range(len(next(iter(losses.values())))))
    df = pd.DataFrame({'epoch': epochs})
    for loss_name, values in losses.items():
        df[loss_name] = values
    
    # Save loss data
    df.to_csv(os.path.join(save_dir, 'loss_data.csv'), index=False)
    
    # Create loss history plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in df.columns:
        if column != 'epoch':
            ax.plot(df['epoch'], df[column], label=column)
    
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    fig.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_d_convergence(d_history, save_dir='plots'):
    """
    Plot D parameter convergence for cluster environment
    
    Args:
        d_history (list): History of D values during training
        save_dir (str): Directory to save plots (will be created if doesn't exist)
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save D history data
    d_df = pd.DataFrame({
        'iteration': range(len(d_history)),
        'value': d_history
    })
    d_df.to_csv(os.path.join(save_dir, 'd_history.csv'), index=False)
    
    # Create D convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d_df['iteration'], d_df['value'], 'b-', label='Predicted D')
    ax.axhline(y=d_df['value'].iloc[-1], color='r', linestyle='--', label='Final Value')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diffusion Coefficient')
    ax.set_title('Convergence of Diffusion Coefficient')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    fig.savefig(os.path.join(save_dir, 'd_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    try:
        # This part is for when running plot_results.py directly
        df = pd.read_csv('plots/loss_data.csv')
        d_df = pd.read_csv('plots/d_history.csv')
        
        losses = {}
        for col in df.columns:
            if col != 'epoch':
                losses[col] = df[col].tolist()
                
        plot_loss_history(losses)
        plot_d_convergence(d_df['value'].tolist())
        print("\nPlotting completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()