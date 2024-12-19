#!/usr/bin/env python
# coding: utf-8

import matplotlib
# Set the backend to 'Agg' if running from terminal
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

def I(x, y, Lx, Ly):
    """Initial condition function."""
    x0, y0 = Lx/2, Ly/2
    sigma = 0.1
    amplitude = 10
    return amplitude * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def f(x, y, t):
    """Source term function."""
    return 0

def solver_FE_2D_neumann(I, f, a, Lx, Ly, T, Nx, Ny):
    """2D Finite Element solver with Neumann boundary conditions."""
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = 0.25 * min(dx**2, dy**2) / a
    Nt = int(T / dt) + 1
    t = np.linspace(0, T, Nt)
    
    nGhost = 1
    x = np.linspace(0, Lx, Nx+2*nGhost)
    y = np.linspace(0, Ly, Ny+2*nGhost)
    
    Fx = a * dt / dx**2
    Fy = a * dt / dy**2
    u = np.zeros((Nt, Nx+2*nGhost, Ny+2*nGhost))
    
    # Set initial condition for real nodes
    initial_total_mass = 0
    for j in range(1, Ny+1):
        for i in range(1, Nx+1):
            u[0, i, j] = I(x[i], y[j], Lx, Ly)
            initial_total_mass += u[0, i, j]
    
    print(f"Initial total mass: {initial_total_mass}")
    
    for n in range(Nt - 1):
        # Apply Neumann boundary conditions
        u[n, 0, :] = u[n, 1, :]   # Left boundary
        u[n, -1, :] = u[n, -2, :] # Right boundary
        u[n, :, 0] = u[n, :, 1]   # Bottom boundary
        u[n, :, -1] = u[n, :, -2] # Top boundary
        
        # Update solution
        u[n+1, 1:-1, 1:-1] = u[n, 1:-1, 1:-1] + \
                             Fx * (u[n, 2:, 1:-1] - 2*u[n, 1:-1, 1:-1] + u[n, :-2, 1:-1]) + \
                             Fy * (u[n, 1:-1, 2:] - 2*u[n, 1:-1, 1:-1] + u[n, 1:-1, :-2]) + \
                             dt * f(x[1:-1], y[1:-1], t[n])
        
        if (n + 1) % 200 == 0 or n == Nt - 2:
            total_mass = np.sum(u[n+1, 1:Nx+1, 1:Ny+1])
            print(f"Time step {n+1}, Total mass: {total_mass}")
    
    return u, x, y, t


import matplotlib
import os
import sys

def is_wsl():
    """Check if running in Windows Subsystem for Linux"""
    return 'microsoft-standard' in os.uname().release.lower() or \
           'WSL' in os.uname().release

def is_running_in_jupyter():
    """Check if code is running in Jupyter notebook"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        return False
    except ImportError:
        return False

# Set appropriate backend
if is_wsl() or (not is_running_in_jupyter()):
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode
else:
    import matplotlib.pyplot as plt
    plt.ion()  # Turn on interactive mode

import numpy as np
import scipy.io

def plot_results(u, x, y, t, Lx, Ly, output_path='diffusion_results.png'):
    """Plot and save the results."""
    try:
        # Explicit array conversions
        u = np.asarray(u, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        
        # Calculate mean with explicit array type
        u_avg = np.asarray(np.mean(u[:, 1:-1, 1:-1], axis=(1, 2)), dtype=np.float64)
        
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle('2D Diffusion Equation Solution (Neumann BC)')
        
        # Plot solution at different time steps
        times = [0, len(t)//4, 2*len(t)//4, -1]
        vmin = float(np.min(u))  # Explicit float conversion
        vmax = float(np.max(u))  # Explicit float conversion
        
        for i, time in enumerate(times):
            ax = fig.add_subplot(3, 2, i+1)
            # Convert the slice to array explicitly
            plot_data = np.asarray(u[time], dtype=np.float64).T
            
            # Create extent array explicitly
            extent = [0.0, float(Lx), 0.0, float(Ly)]
            
            im = ax.imshow(plot_data, 
                          origin='lower', 
                          extent=extent,
                          aspect='equal', 
                          cmap='jet', 
                          vmin=vmin, 
                          vmax=vmax)
            
            ax.set_title(f't = {float(t[time]):.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(im, ax=ax, label='u')
        
        # Plot average u over time
        ax_avg = fig.add_subplot(3, 2, (5, 6))
        ax_avg.plot(np.asarray(t, dtype=np.float64), 
                   np.asarray(u_avg, dtype=np.float64))
        ax_avg.set_xlabel('Time')
        ax_avg.set_ylabel('Average u')
        ax_avg.set_title('Average Concentration Over Time')
        
        plt.tight_layout()
        
        # Save with explicit DPI and bbox_inches
        fig.savefig(output_path, 
                   bbox_inches='tight', 
                   dpi=300, 
                   format='png')
        
        print(f"Plot saved successfully to {output_path}")
        
    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        print(f"Debug info - Shapes: u:{u.shape}, x:{x.shape}, y:{y.shape}, t:{t.shape}")
        print(f"Debug info - Types: u:{u.dtype}, x:{x.dtype}, y:{y.dtype}, t:{t.dtype}")
        
    finally:
        plt.close('all')

def main():
    # Parameters
    a = 0.01
    Lx = Ly = 1
    T = 1
    Nx = Ny = 44
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Solve the equation
    print("Solving diffusion equation...")
    u, x, y, t = solver_FE_2D_neumann(I, f, a, Lx, Ly, T, Nx, Ny)
    
    # Save results
    output_mat = os.path.join(results_dir, "2D_u_curveGhostCell.mat")
    output_plot = os.path.join(results_dir, "diffusion_results.png")
    
    print("Saving MAT file...")
    scipy.io.savemat(output_mat, {"usol": u, "x": x, "y": y, "t": t})
    
    print("Generating plot...")
    plot_results(u, x, y, t, Lx, Ly, output_path=output_plot)
    
    print("\nAll operations completed:")
    print(f"- MAT file saved to: {output_mat}")
    print(f"- Plot saved to: {output_plot}")

if __name__ == "__main__":
    main()