# Additional functions for src/diffusion_pinn/utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict

def plot_open_system_parameters(D_history: List[float], k_history: List[float],
                               save_dir: str = 'results', save_data: bool = True) -> None:
    """Plot evolution of both D and k parameters for open system"""
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Save data if requested
        if save_data:
            import pandas as pd
            param_df = pd.DataFrame({
                'epoch': range(len(D_history)),
                'D_value': D_history,
                'k_value': k_history
            })
            param_df.to_csv(os.path.join(save_dir, 'parameter_history.csv'), index=False)

        # Create subplot for both parameters
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        epochs = range(len(D_history))

        # Plot diffusion coefficient
        ax1.semilogy(epochs, D_history, 'b-', linewidth=2, label='Diffusion Coefficient (D)')
        ax1.set_ylabel('D (log scale)', fontsize=12)
        ax1.set_title('Open System Parameter Evolution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot boundary permeability
        ax2.semilogy(epochs, k_history, 'r-', linewidth=2, label='Boundary Permeability (k)')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('k (log scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add final values as text
        final_D = D_history[-1]
        final_k = k_history[-1]

        ax1.text(0.02, 0.95, f'Final D = {final_D:.2e}', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"), fontsize=10)
        ax2.text(0.02, 0.95, f'Final k = {final_k:.2e}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"), fontsize=10)

        # Add time scale comparison
        diff_time = 1.0 / final_D if final_D > 0 else float('inf')
        outflow_time = 1.0 / final_k if final_k > 0 else float('inf')

        ax2.text(0.02, 0.02, f'Diffusion time scale ≈ {diff_time:.1f}\nOutflow time scale ≈ {outflow_time:.1f}',
                transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
                fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'open_system_parameters.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Open system parameter plot saved to {save_dir}/open_system_parameters.png")

    except Exception as e:
        print(f"Error in plot_open_system_parameters: {str(e)}")
        raise

def plot_mass_conservation_analysis(pinn: 'OpenSystemDiffusionPINN',
                                   data_processor: 'DiffusionDataProcessor',
                                   save_dir: str = 'results') -> None:
    """Analyze and plot mass conservation in the open system"""
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Calculate actual mass over time from data
        actual_masses = []
        for t_idx in range(len(data_processor.t)):
            mass = np.sum(data_processor.usol[:, :, t_idx])
            actual_masses.append(mass)

        # Predict mass over time using PINN
        n_t_points = len(data_processor.t)
        predicted_masses = []

        for t_idx in range(n_t_points):
            t_val = data_processor.t[t_idx]
            # Create grid for this time
            X_grid, Y_grid = np.meshgrid(data_processor.x, data_processor.y, indexing='ij')
            points = np.column_stack([
                X_grid.flatten(),
                Y_grid.flatten(),
                np.full(X_grid.size, t_val)
            ])

            # Predict concentrations
            import tensorflow as tf
            points_tf = tf.convert_to_tensor(points, dtype=tf.float32)
            c_pred = pinn.predict(points_tf).numpy()

            # Calculate total predicted mass
            predicted_mass = np.sum(c_pred.reshape(X_grid.shape))
            predicted_masses.append(predicted_mass)

        # Create mass conservation plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        time_points = data_processor.t

        # Plot actual vs predicted mass
        ax1.plot(time_points, actual_masses, 'b-', linewidth=2, label='Actual Mass (from data)')
        ax1.plot(time_points, predicted_masses, 'r--', linewidth=2, label='PINN Predicted Mass')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Total Mass')
        ax1.set_title('Mass Conservation: Open System Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calculate mass loss rates
        dt = time_points[1] - time_points[0]
        actual_loss_rate = np.gradient(actual_masses, dt)
        predicted_loss_rate = np.gradient(predicted_masses, dt)

        # Plot mass loss rates
        ax2.plot(time_points, actual_loss_rate, 'b-', linewidth=2, label='Actual Mass Loss Rate')
        ax2.plot(time_points, predicted_loss_rate, 'r--', linewidth=2, label='PINN Predicted Loss Rate')
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('dM/dt')
        ax2.set_title('Mass Loss Rate Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add statistics
        total_actual_loss = actual_masses[0] - actual_masses[-1]
        total_predicted_loss = predicted_masses[0] - predicted_masses[-1]
        loss_error = abs(total_predicted_loss - total_actual_loss) / total_actual_loss * 100

        ax1.text(0.02, 0.98, f'Actual total loss: {total_actual_loss:.3f}\n'
                             f'PINN predicted loss: {total_predicted_loss:.3f}\n'
                             f'Relative error: {loss_error:.1f}%',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"), fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mass_conservation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Mass conservation analysis saved to {save_dir}/mass_conservation_analysis.png")

        # Save numerical data
        import pandas as pd
        mass_data = pd.DataFrame({
            'time': time_points,
            'actual_mass': actual_masses,
            'predicted_mass': predicted_masses,
            'actual_loss_rate': actual_loss_rate,
            'predicted_loss_rate': predicted_loss_rate
        })
        mass_data.to_csv(os.path.join(save_dir, 'mass_conservation_data.csv'), index=False)

    except Exception as e:
        print(f"Error in plot_mass_conservation_analysis: {str(e)}")
        raise

def plot_boundary_flux_analysis(pinn: 'OpenSystemDiffusionPINN',
                               data_processor: 'DiffusionDataProcessor',
                               save_dir: str = 'results') -> None:
    """Analyze and visualize boundary fluxes"""
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Sample boundary points at different times
        n_boundary_points = 100
        n_time_points = 10

        time_indices = np.linspace(0, len(data_processor.t)-1, n_time_points, dtype=int)
        time_values = data_processor.t[time_indices]

        # Get domain bounds
        x_min, x_max = data_processor.x.min(), data_processor.x.max()
        y_min, y_max = data_processor.y.min(), data_processor.y.max()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        import tensorflow as tf

        for i, t_val in enumerate(time_values):
            # X boundaries
            y_boundary = np.linspace(y_min, y_max, n_boundary_points)

            # x = x_min boundary
            x_min_points = np.column_stack([
                np.full(n_boundary_points, x_min),
                y_boundary,
                np.full(n_boundary_points, t_val)
            ])

            x_min_tf = tf.convert_to_tensor(x_min_points, dtype=tf.float32)
            flux_x_min = pinn.compute_boundary_flux_residual(x_min_tf, 'x_min').numpy()

            # x = x_max boundary
            x_max_points = np.column_stack([
                np.full(n_boundary_points, x_max),
                y_boundary,
                np.full(n_boundary_points, t_val)
            ])

            x_max_tf = tf.convert_to_tensor(x_max_points, dtype=tf.float32)
            flux_x_max = pinn.compute_boundary_flux_residual(x_max_tf, 'x_max').numpy()

            # Y boundaries
            x_boundary = np.linspace(x_min, x_max, n_boundary_points)

            # y = y_min boundary
            y_min_points = np.column_stack([
                x_boundary,
                np.full(n_boundary_points, y_min),
                np.full(n_boundary_points, t_val)
            ])

            y_min_tf = tf.convert_to_tensor(y_min_points, dtype=tf.float32)
            flux_y_min = pinn.compute_boundary_flux_residual(y_min_tf, 'y_min').numpy()

            # y = y_max boundary
            y_max_points = np.column_stack([
                x_boundary,
                np.full(n_boundary_points, y_max),
                np.full(n_boundary_points, t_val)
            ])

            y_max_tf = tf.convert_to_tensor(y_max_points, dtype=tf.float32)
            flux_y_max = pinn.compute_boundary_flux_residual(y_max_tf, 'y_max').numpy()

            # Plot fluxes
            color = plt.cm.viridis(i / (n_time_points - 1))

            ax1.plot(y_boundary, flux_x_min.flatten(), color=color, alpha=0.7, label=f't={t_val:.2f}' if i < 5 else "")
            ax1.set_title('Flux at x = x_min boundary')
            ax1.set_xlabel('y coordinate')
            ax1.set_ylabel('Flux residual')

            ax2.plot(y_boundary, flux_x_max.flatten(), color=color, alpha=0.7)
            ax2.set_title('Flux at x = x_max boundary')
            ax2.set_xlabel('y coordinate')
            ax2.set_ylabel('Flux residual')

            ax3.plot(x_boundary, flux_y_min.flatten(), color=color, alpha=0.7)
            ax3.set_title('Flux at y = y_min boundary')
            ax3.set_xlabel('x coordinate')
            ax3.set_ylabel('Flux residual')

            ax4.plot(x_boundary, flux_y_max.flatten(), color=color, alpha=0.7)
            ax4.set_title('Flux at y = y_max boundary')
            ax4.set_xlabel('x coordinate')
            ax4.set_ylabel('Flux residual')

        # Add legend to first subplot
        ax1.legend(fontsize=8)

        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=time_values[0], vmax=time_values[-1]))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4], orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Time')

        plt.suptitle(f'Boundary Flux Analysis\nD = {pinn.get_diffusion_coefficient():.2e}, k = {pinn.get_boundary_permeability():.2e}',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'boundary_flux_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Boundary flux analysis saved to {save_dir}/boundary_flux_analysis.png")

    except Exception as e:
        print(f"Error in plot_boundary_flux_analysis: {str(e)}")
        raise

def create_open_system_summary_report(pinn: 'OpenSystemDiffusionPINN',
                                     data_processor: 'DiffusionDataProcessor',
                                     D_history: List[float],
                                     k_history: List[float],
                                     save_dir: str = 'results') -> None:
    """Create comprehensive summary report for open system results"""
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Get final parameters
        final_D = pinn.get_diffusion_coefficient()
        final_k = pinn.get_boundary_permeability()
        final_c_ext = pinn.get_external_concentration()

        # Calculate characteristic time scales
        diff_time = 1.0 / final_D if final_D > 0 else float('inf')
        outflow_time = 1.0 / final_k if final_k > 0 else float('inf')

        # Mass loss analysis from data
        initial_mass = np.sum(data_processor.usol[:, :, 0])
        final_mass = np.sum(data_processor.usol[:, :, -1])
        total_time = data_processor.t[-1] - data_processor.t[0]
        mass_loss_rate = (initial_mass - final_mass) / total_time

        # Theoretical mass loss rate from Robin boundary condition
        # Rough estimate: k * perimeter * average_concentration
        perimeter = 2 * ((data_processor.x.max() - data_processor.x.min()) +
                        (data_processor.y.max() - data_processor.y.min()))
        avg_concentration = np.mean(data_processor.usol)
        theoretical_loss_rate = final_k * perimeter * avg_concentration

        # Create summary report
        report_path = os.path.join(save_dir, 'open_system_summary.txt')

        with open(report_path, 'w') as f:
            f.write("OPEN SYSTEM DIFFUSION ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write("LEARNED PHYSICAL PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Diffusion coefficient (D):      {final_D:.6e}\n")
            f.write(f"Boundary permeability (k):      {final_k:.6e}\n")
            f.write(f"External concentration:         {final_c_ext:.6f}\n\n")

            f.write("TIME SCALES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Diffusion time scale:           {diff_time:.2f} time units\n")
            f.write(f"Outflow time scale:             {outflow_time:.2f} time units\n")

            if outflow_time < diff_time:
                f.write("System is OUTFLOW-DOMINATED (fast boundary loss)\n\n")
            else:
                f.write("System is DIFFUSION-DOMINATED (slow boundary loss)\n\n")

            f.write("MASS BALANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Initial total mass:             {initial_mass:.6f}\n")
            f.write(f"Final total mass:               {final_mass:.6f}\n")
            f.write(f"Total mass lost:                {initial_mass - final_mass:.6f}\n")
            f.write(f"Relative mass loss:             {(initial_mass - final_mass)/initial_mass*100:.1f}%\n")
            f.write(f"Observed loss rate:             {mass_loss_rate:.6f} units/time\n")
            f.write(f"Theoretical loss rate (k*P*c):  {theoretical_loss_rate:.6f} units/time\n")
            f.write(f"Loss rate agreement:            {abs(mass_loss_rate - theoretical_loss_rate)/abs(mass_loss_rate)*100:.1f}% error\n\n")

            f.write("CONVERGENCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            if len(D_history) >= 100:
                recent_D = D_history[-100:]
                recent_k = k_history[-100:]
                D_rel_std = np.std(recent_D) / np.mean(recent_D)
                k_rel_std = np.std(recent_k) / np.mean(recent_k)

                f.write(f"D convergence (rel std):        {D_rel_std:.6f}\n")
                f.write(f"k convergence (rel std):        {k_rel_std:.6f}\n")
                f.write(f"Overall convergence:            {'Good' if max(D_rel_std, k_rel_std) < 0.05 else 'Poor'}\n\n")

            f.write("PHYSICAL INTERPRETATION:\n")
            f.write("-" * 40 + "\n")

            # Estimate Peclet number analog for outflow
            Pe_outflow = final_k / final_D if final_D > 0 else float('inf')
            f.write(f"Outflow Peclet number (k/D):    {Pe_outflow:.3f}\n")

            if Pe_outflow > 1:
                f.write("Boundary outflow dominates over diffusion\n")
            elif Pe_outflow < 0.1:
                f.write("Diffusion dominates over boundary outflow\n")
            else:
                f.write("Balanced diffusion-outflow system\n")

            f.write(f"\nHalf-life due to outflow only:  {np.log(2)/final_k:.2f} time units\n")
            f.write(f"Diffusion penetration length:   {np.sqrt(final_D):.4f} spatial units\n\n")

            f.write("BOUNDARY CONDITION VALIDATION:\n")
            f.write("-" * 40 + "\n")
            f.write("Robin boundary condition: -D(∂c/∂n) = k(c - c_ext)\n")
            f.write(f"This model assumes:\n")
            f.write(f"  - Linear relationship between concentration and flux\n")
            f.write(f"  - Uniform boundary permeability (k = {final_k:.2e})\n")
            f.write(f"  - Constant external concentration (c_ext = {final_c_ext:.3f})\n\n")

            f.write("RECOMMENDED VALIDATION STEPS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Check boundary flux residuals are small (< 0.01)\n")
            f.write("2. Verify mass conservation matches data trend\n")
            f.write("3. Compare with independent diffusion coefficient measurements\n")
            f.write("4. Validate boundary permeability with independent experiments\n")
            f.write("5. Test model predictions on held-out time points\n\n")

            f.write("COMPARISON WITH CLOSED SYSTEM:\n")
            f.write("-" * 40 + "\n")
            f.write("If this were modeled as a closed system:\n")
            f.write("  - Mass would be artificially conserved\n")
            f.write("  - D would be biased to compensate for missing outflow\n")
            f.write("  - Boundary conditions would be incorrectly fit to data\n")
            f.write("  - Physical interpretation would be meaningless\n\n")

            f.write("MODEL LIMITATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("- Assumes isotropic diffusion (D_x = D_y)\n")
            f.write("- Assumes uniform boundary permeability\n")
            f.write("- Neglects convection/advection effects\n")
            f.write("- Linear Robin boundary condition may be approximate\n")
            f.write("- Does not account for source/sink terms in interior\n")

        print(f"Open system summary report saved to {report_path}")

    except Exception as e:
        print(f"Error creating open system summary report: {str(e)}")
        raise