import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt

def plot_data():
    # Plot loss history
    print("Loading loss data...")
    loss_data = pd.read_csv('plots/loss_data.csv')
    print("Loss data columns:", loss_data.columns.tolist())
    
    # Create loss history plot
    print("Creating loss history plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in loss_data.columns[1:]:  # Skip epoch column
        ax.plot(loss_data['epoch'], loss_data[column], label=column)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(True)
    fig.savefig('plots/loss_history.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Loss history plot saved")
    
    # Plot D history
    print("\nLoading D history data...")
    d_data = pd.read_csv('plots/d_history.csv')
    print("D history data shape:", d_data.shape)
    
    # Create D history plot
    print("Creating D history plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d_data['iteration'], d_data['value'], 'b-', label='Predicted D')
    ax.axhline(y=d_data['value'].iloc[-1], color='r', linestyle='--', label='Final Value')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diffusion Coefficient')
    ax.set_title('Convergence of Diffusion Coefficient')
    ax.legend()
    ax.grid(True)
    fig.savefig('plots/d_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("D history plot saved")

if __name__ == "__main__":
    try:
        plot_data()
        print("\nPlotting completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
