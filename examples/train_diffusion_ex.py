import tensorflow as tf
from diffusion_pinn import DiffusionConfig
from diffusion_pinn import (
    DiffusionPINN,
    DiffusionDataProcessor
)
from diffusion_pinn.training import train_pinn, create_and_initialize_pinn
from diffusion_pinn.utils.visualization import (
    plot_solutions_and_error,
    plot_loss_history,
    plot_diffusion_convergence
)

def main():
    # Create configuration
    config = DiffusionConfig()

    # File paths using the configuration
    input_file = config.get_data_file("intensity_time_series_spatial_temporal.csv")
    save_dir = config.get_saved_models_dir()

    # Create and initialize PINN
    pinn, data = create_and_initialize_pinn(
        inputfile=str(input_file),  # Convert Path to string
        N_u=1000,  # boundary points
        N_f=20000,  # collocation points
        N_i=10000,  # interior supervision points
        initial_D=1.0  # initial guess for diffusion coefficient
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
        epochs=20000,
        save_dir=str(save_dir)  # Convert Path to string
    )

    # Plot results
    print("\nTraining completed. Plotting results...")

    # Plot loss history
    plot_loss_history(loss_history)

    # Plot solutions at different time points
    data_processor = DiffusionDataProcessor(str(input_file))  # Convert Path to string
    t_indices = [0, len(data_processor.t)//3, 2*len(data_processor.t)//3, -1]

    # Create results directory if it doesn't exist
    results_dir = config.get_project_root() / "results"
    results_dir.mkdir(exist_ok=True)

    plot_solutions_and_error(
        pinn=pinn,
        data_processor=data_processor,
        t_indices=t_indices,
        save_path=str(results_dir / 'final_solutions.png')
    )

    # Plot diffusion coefficient convergence
    plot_diffusion_convergence(
        D_history=D_history,
        save_path=str(results_dir / 'D_convergence.png')
    )

    print("\nFinal diffusion coefficient:", pinn.get_diffusion_coefficient())

if __name__ == "__main__":
    main()