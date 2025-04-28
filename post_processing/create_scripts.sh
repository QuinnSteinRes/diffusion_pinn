#!/bin/bash
# This script creates all the necessary files for D value analysis with proper syntax

cat > d_extract.py << 'EOL'
#!/usr/bin/env python3
# Script to extract D values from d_history.csv files with downsampling
import os
import pandas as pd
import numpy as np

def extract_d_values(downsample_factor=100):
    """
    Extract D values from d_history.csv files in each run's results directory
    with downsampling to reduce data points
    
    Args:
        downsample_factor: Keep only every Nth point (default: 100)
    """
    print("Starting D value extraction with downsampling...")
    print(f"Downsample factor: {downsample_factor} (keeping 1/{downsample_factor} points)")
    
    # Dictionary to store D values from each run
    all_d_values = {}
    
    # Check for run directories
    max_run_to_check = 10
    max_length = 0
    
    # Loop through possible run directories
    for run_num in range(1, max_run_to_check + 1):
        run_name = f"Run_{run_num}"
        csv_path = os.path.join(f"run_{run_num}", "results", "d_history.csv")
        
        if os.path.exists(csv_path):
            print(f"Found d_history.csv for {run_name}")
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
                
                # Most common format has 'D' column 
                if 'D' in df.columns:
                    # Apply downsampling - take every Nth row
                    d_values = df['D'].values[::downsample_factor]
                    all_d_values[run_name] = d_values
                    max_length = max(max_length, len(d_values))
                    print(f"  Extracted {len(d_values)} D values after downsampling")
                    print(f"  (Original data had {len(df)} points)")
                else:
                    # Try to find any column that might contain D values
                    # Usually the second column in most formats
                    if df.shape[1] >= 2:
                        d_values = df.iloc[::downsample_factor, 1].values  # Second column, downsampled
                        all_d_values[run_name] = d_values
                        max_length = max(max_length, len(d_values))
                        print(f"  Extracted {len(d_values)} D values from column 2 after downsampling")
                        print(f"  (Original data had {len(df)} points)")
                    else:
                        print(f"  Warning: Could not identify D values in {csv_path}")
            
            except Exception as e:
                print(f"  Error reading {csv_path}: {e}")
        
    # If we found any D values, save them to a combined CSV
    if all_d_values:
        # Create a combined DataFrame, padding shorter series with NaN
        combined_df = pd.DataFrame()
        
        for run_name, values in all_d_values.items():
            # Pad with NaN if needed
            padded = np.pad(values, (0, max_length - len(values)), 
                           mode='constant', constant_values=np.nan)
            combined_df[run_name] = padded
        
        # Save to CSV
        output_file = "d_value_history.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved {len(all_d_values)} runs to {output_file}")
        print(f"Each run has {max_length} epochs of data (downsampled)")
        
        return True
    else:
        print("\nNo D values were found. Check your directory structure.")
        return False

if __name__ == "__main__":
    # You can adjust the downsample factor here
    # Higher values mean fewer points (e.g., 100 keeps only 1 out of every 100 points)
    extract_d_values(downsample_factor=100)
EOL

cat > d_plt.py << 'EOL'
#!/usr/bin/env python3
# Script to plot and analyze downsampled D values
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_d_value(d_pinn, roi_dimensions=(299, 99), pixel_size_mm=0.0513, time_step_seconds=60):
    """Convert a normalized D value to physical units"""
    # Get characteristic length (use largest dimension)
    L_pixels = max(roi_dimensions)
    L_mm = L_pixels * pixel_size_mm
    
    # Convert from normalized units to physical units
    D_mm2s = d_pinn * L_mm**2 / time_step_seconds
    D_mm2min = D_mm2s * 60
    D_m2s = D_mm2s * 1e-6
    D_m2s_e10 = D_m2s * 1e10  # In units of 10^-10 m^2/s
    
    # Convert to pixel units
    D_pixels_per_s = d_pinn * L_pixels**2 / time_step_seconds
    D_pixels_per_min = D_pixels_per_s * 60
    
    return {
        'D_pinn': d_pinn,
        'D_m2s': D_m2s,
        'D_m2s_e10': D_m2s_e10,
        'D_mm2s': D_mm2s,
        'D_mm2min': D_mm2min,
        'D_pixels_per_s': D_pixels_per_s,
        'D_pixels_per_min': D_pixels_per_min
    }

def plot_d_values(
    csv_file="d_value_history.csv", 
    roi_dimensions=(299, 99), 
    pixel_size_mm=0.0513, 
    time_step_seconds=60,
    reference_range=(0.000071, 0.0001032),  # Expected range in PINN units
    downsample_factor=100,  # The factor used in data extraction
    further_simplify=False  # Option to further simplify the plot
):
    """Create an interactive plot of D values with unit conversion"""
    print(f"Plotting downsampled D values from {csv_file}...")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found!")
        return False
    
    # Read the data
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            print(f"Error: No data found in {csv_file}")
            return False
            
        print(f"Found {len(df.columns)} runs in the dataset")
        print(f"Each run has {len(df)} data points (already downsampled by factor of {downsample_factor})")
        
        # Further simplify if requested
        if further_simplify and len(df) > 500:
            simplify_factor = len(df) // 500  # Aim for ~500 points
            df = df.iloc[::simplify_factor].copy()
            print(f"Further simplified to {len(df)} points")
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return False
    
    # Create a figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            "PINN Diffusion Coefficient (Normalized)",
            "Converted Physical Units (10^-10 m^2/s)"
        )
    )
    
    # Generate x-axis values (epochs), accounting for downsampling
    epochs = np.arange(0, len(df) * downsample_factor, downsample_factor)
    
    # Add each run to the plot
    for col in df.columns:
        # Add original D values to top plot
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=df[col].values,
                name=col,
                mode='lines',
                connectgaps=False,  # Don't connect across NaN values
                line=dict(width=1.5)  # Thinner lines for cleaner plot
            ),
            row=1, col=1
        )
        
        # Convert to physical units for bottom plot
        d_converted = np.zeros_like(df[col].values)
        for i, d in enumerate(df[col].values):
            if not np.isnan(d):
                result = analyze_d_value(
                    d, roi_dimensions, pixel_size_mm, time_step_seconds
                )
                d_converted[i] = result['D_m2s_e10']
        
        # Add converted values to bottom plot
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=d_converted,
                name=f"{col} (10^-10 m^2/s)",
                mode='lines',
                connectgaps=False,
                line=dict(width=1.5)  # Thinner lines for cleaner plot
            ),
            row=2, col=1
        )
    
    # Add reference range if provided
    if reference_range:
        min_val, max_val = reference_range
        color = 'rgba(0, 128, 0, 0.7)'  # Green
        
        # Convert reference range to physical units
        min_phys = analyze_d_value(min_val, roi_dimensions, pixel_size_mm, time_step_seconds)['D_m2s_e10']
        max_phys = analyze_d_value(max_val, roi_dimensions, pixel_size_mm, time_step_seconds)['D_m2s_e10']
        
        # Add to top plot
        fig.add_trace(
            go.Scatter(
                x=[epochs[0], epochs[-1]],
                y=[min_val, min_val],
                mode='lines',
                line=dict(color=color, width=1.5, dash='dash'),
                name="Expected range (min)",
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[epochs[0], epochs[-1]],
                y=[max_val, max_val],
                mode='lines',
                line=dict(color=color, width=1.5, dash='dash'),
                name="Expected range (max)",
            ),
            row=1, col=1
        )
        
        # Add filled area
        fig.add_trace(
            go.Scatter(
                x=[epochs[0], epochs[-1], epochs[-1], epochs[0]],
                y=[min_val, min_val, max_val, max_val],
                fill='toself',
                fillcolor='rgba(0, 128, 0, 0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                name="Expected range",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add to bottom plot
        fig.add_trace(
            go.Scatter(
                x=[epochs[0], epochs[-1]],
                y=[min_phys, min_phys],
                mode='lines',
                line=dict(color=color, width=1.5, dash='dash'),
                name="Expected range (min)",
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[epochs[0], epochs[-1]],
                y=[max_phys, max_phys],
                mode='lines',
                line=dict(color=color, width=1.5, dash='dash'),
                name="Expected range (max)",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add filled area
        fig.add_trace(
            go.Scatter(
                x=[epochs[0], epochs[-1], epochs[-1], epochs[0]],
                y=[min_phys, min_phys, max_phys, max_phys],
                fill='toself',
                fillcolor='rgba(0, 128, 0, 0.1)',
                line=dict(color='rgba(0,0,0,0)'),
                name="Expected range",
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout for a cleaner look
    fig.update_layout(
        height=800,
        title_text=f"Diffusion Coefficient Convergence Analysis (Downsampled {downsample_factor}x)",
        legend_title="Runs",
        plot_bgcolor='rgba(240, 240, 240, 0.9)',  # Lighter background
        legend=dict(
            font=dict(size=10),  # Smaller legend text
            borderwidth=1  # Add border to legend
        )
    )
    
    # Add scale type buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.1,
                showactive=True,
                buttons=[
                    dict(
                        label="Linear Scale",
                        method="relayout",
                        args=[{"yaxis.type": "linear", "yaxis2.type": "linear"}]
                    ),
                    dict(
                        label="Log Scale",
                        method="relayout",
                        args=[{"yaxis.type": "log", "yaxis2.type": "log"}]
                    )
                ]
            )
        ]
    )
    
    # Update axis labels and grid
    fig.update_yaxes(title_text="Normalized D Value", row=1, col=1, gridcolor='rgba(200, 200, 200, 0.3)')
    fig.update_yaxes(title_text="D Value (10^-10 m^2/s)", row=2, col=1, gridcolor='rgba(200, 200, 200, 0.3)')
    fig.update_xaxes(title_text="Epoch", row=2, col=1, gridcolor='rgba(200, 200, 200, 0.3)')
    
    # Add analysis parameters as annotation to bottom plot
    param_text = (
        f"Analysis Parameters: ROI {roi_dimensions[0]}x{roi_dimensions[1]} px | "
        f"Pixel Size {pixel_size_mm:.4f} mm | Time Step {time_step_seconds} s | "
        f"Downsampled {downsample_factor}x"
    )
    fig.layout.annotations[1].text = f"Converted Physical Units (10^-10 m^2/s)<br><span style='font-size:10px'>{param_text}</span>"
    
    # Save the figure to HTML
    html_file = "d_value_history_interactive.html"
    fig.write_html(html_file, auto_open=False)
    print(f"Saved interactive plot to {html_file}")
    
    # Also save as an image
    png_file = "d_value_history.png"
    try:
        fig.write_image(png_file)
        print(f"Saved static image to {png_file}")
    except Exception as e:
        print(f"Note: Could not save PNG (requires kaleido package): {e}")
    
    # Display the figure
    fig.show()
    
    # Analyze final D values
    analyze_final_values(df, roi_dimensions, pixel_size_mm, time_step_seconds, downsample_factor)
    
    return True

def analyze_final_values(df, roi_dimensions, pixel_size_mm, time_step_seconds, downsample_factor):
    """Analyze the final D value for each run"""
    print("\n=== Final D Value Analysis ===")
    
    final_values = []
    
    for col in df.columns:
        # Get the last non-NaN value
        valid_values = df[col].dropna().values
        if len(valid_values) > 0:
            final_d = valid_values[-1]
            results = analyze_d_value(final_d, roi_dimensions, pixel_size_mm, time_step_seconds)
            
            print(f"\n== {col} ==")
            print(f"Final Epoch: ~{len(df) * downsample_factor}")
            print(f"Normalized D: {results['D_pinn']:.6f}")
            print(f"D in 10^-10 m^2/s: {results['D_m2s_e10']:.6f}")
            print(f"D in mm^2/min: {results['D_mm2min']:.6f}")
            
            final_values.append(results)
    
    # Calculate average and standard deviation if we have multiple runs
    if len(final_values) > 1:
        print("\n== AVERAGE ACROSS ALL RUNS ==")
        avg_d_pinn = np.mean([r['D_pinn'] for r in final_values])
        avg_results = analyze_d_value(avg_d_pinn, roi_dimensions, pixel_size_mm, time_step_seconds)
        
        print(f"Normalized D: {avg_results['D_pinn']:.6f}")
        print(f"D in 10^-10 m^2/s: {avg_results['D_m2s_e10']:.6f}")
        print(f"D in mm^2/min: {avg_results['D_mm2min']:.6f}")
        
        # Calculate standard deviation
        std_d_pinn = np.std([r['D_pinn'] for r in final_values])
        print(f"\nStandard Deviation:")
        print(f"Normalized D: {std_d_pinn:.6f}")
        print(f"Relative std: {(std_d_pinn/avg_d_pinn)*100:.2f}%")

if __name__ == "__main__":
    # You can adjust these parameters as needed
    plot_d_values(
        csv_file="d_value_history.csv",
        roi_dimensions=(299, 99),
        pixel_size_mm=0.0513,
        time_step_seconds=60,
        reference_range=(0.000071, 0.0001032),
        downsample_factor=100, # This should match what was used in d_extract.py
        further_simplify=True   # Set to True to further simplify the plot
    )
EOL

cat > run_d.sh << 'EOL'
#!/bin/bash
# Script to run D value extraction and plotting with downsampling

# Configurable parameters
DOWNSAMPLE_FACTOR=100  # Keep only 1 out of every 100 points

echo "====================================================="
echo "PINN Diffusion Coefficient Analysis (Downsampled ${DOWNSAMPLE_FACTOR}x)"
echo "====================================================="

echo -e "\n>>> Step 1: Extracting D values with ${DOWNSAMPLE_FACTOR}x downsampling..."
python d_extract.py

echo -e "\n>>> Step 2: Creating D value plots and analysis..."
python d_plt.py

echo -e "\n>>> Analysis complete!"
echo "Check the current directory for:"
echo "- d_value_history_interactive.html (interactive plot)"
echo "- d_value_history.png (static image)"
echo ""
echo "Note: The data has been downsampled by a factor of ${DOWNSAMPLE_FACTOR}."
echo "If you want a different level of downsampling, edit the DOWNSAMPLE_FACTOR"
echo "value at the top of this script and run it again."
EOL

# Make the scripts executable
chmod +x d_extract.py d_plt.py run_d.sh

echo "All scripts have been created successfully!"
echo ""
echo "To change the downsampling factor:"
echo "1. Edit the DOWNSAMPLE_FACTOR in run_d.sh"
echo ""
echo "To run the analysis, use:"
echo "./run_d.sh"
