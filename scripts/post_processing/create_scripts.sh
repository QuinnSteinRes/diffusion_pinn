#!/bin/bash
# This script creates all the necessary files for D value analysis with proper syntax

cat > d_extract.py << 'ENDOFEXTRACT'
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

                # Most common format has 'value' column or column 1 for D values
                if 'value' in df.columns:
                    # Apply downsampling - take every Nth row
                    d_values = df['value'].values[::downsample_factor]
                    all_d_values[run_name] = d_values
                    max_length = max(max_length, len(d_values))
                    print(f"  Extracted {len(d_values)} D values after downsampling")
                    print(f"  (Original data had {len(df)} points)")
                # Try 'D' column if available
                elif 'D' in df.columns:
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
ENDOFEXTRACT

cat > d_plt.py << 'ENDOFPLOT'
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
ENDOFPLOT

# NEW SCRIPT: d_summary.py for extracting and summarizing convergence status and final D values
cat > d_summary.py << 'ENDOFSUMMARY'
#!/usr/bin/env python3
# Script to extract and summarize convergence status and final diffusion coefficients
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

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

    return {
        'D_pinn': d_pinn,
        'D_m2s': D_m2s,
        'D_m2s_e10': D_m2s_e10,
        'D_mm2min': D_mm2min
    }

def extract_run_summaries(max_runs=10):
    """
    Extract summary information from each run and compile into a single file

    Args:
        max_runs: Maximum number of runs to check
    """
    print("Extracting run summaries...")
    summary_data = []

    # Loop through run directories
    for run_num in range(1, max_runs + 1):
        run_name = f"Run {run_num}"
        run_dir = f"run_{run_num}"

        if not os.path.exists(run_dir):
            continue

        print(f"Checking {run_name} in {run_dir}...")

        # Data collection for this run
        run_data = {
            'run_number': run_num,
            'run_name': run_name,
            'converged': 'Unknown',
            'final_diffusion': None,
            'final_diffusion_physical': None,
            'total_epochs': None,
            'final_loss': None,
            'training_time_minutes': None,
            'notes': []
        }

        # Check for summary file first
        summary_txt = os.path.join(run_dir, "training_summary.txt")
        summary_csv = os.path.join(run_dir, "training_summary.csv")

        if os.path.exists(summary_txt):
            print(f"  Found summary text file")
            with open(summary_txt, 'r') as f:
                content = f.read()

                # Extract data using simple parsing
                if "Final diffusion coefficient:" in content:
                    try:
                        d_line = [line for line in content.split('\n') if "Final diffusion coefficient:" in line][0]
                        run_data['final_diffusion'] = float(d_line.split(':')[1].strip())
                    except Exception as e:
                        run_data['notes'].append(f"Error extracting D value: {str(e)}")

                if "Converged:" in content:
                    try:
                        conv_line = [line for line in content.split('\n') if "Converged:" in line][0]
                        run_data['converged'] = conv_line.split(':')[1].strip()
                    except Exception as e:
                        run_data['notes'].append(f"Error extracting convergence: {str(e)}")

                if "Final loss:" in content:
                    try:
                        loss_line = [line for line in content.split('\n') if "Final loss:" in line][0]
                        run_data['final_loss'] = float(loss_line.split(':')[1].strip())
                    except Exception as e:
                        run_data['notes'].append(f"Error extracting loss: {str(e)}")

                if "Total epochs:" in content:
                    try:
                        epochs_line = [line for line in content.split('\n') if "Total epochs:" in line][0]
                        run_data['total_epochs'] = int(epochs_line.split(':')[1].strip())
                    except Exception as e:
                        run_data['notes'].append(f"Error extracting epochs: {str(e)}")

        elif os.path.exists(summary_csv):
            print(f"  Found summary CSV file")
            try:
                df = pd.read_csv(summary_csv)
                if len(df) > 0:
                    row = df.iloc[0]
                    if 'final_diffusion_coefficient' in row:
                        run_data['final_diffusion'] = row['final_diffusion_coefficient']
                    if 'converged' in row:
                        run_data['converged'] = str(row['converged'])
                    if 'final_loss' in row:
                        run_data['final_loss'] = row['final_loss']
                    if 'total_epochs' in row:
                        run_data['total_epochs'] = row['total_epochs']
            except Exception as e:
                run_data['notes'].append(f"Error reading summary CSV: {str(e)}")

        # If we couldn't find summary files, look for D history file
        if run_data['final_diffusion'] is None:
            d_history_file = os.path.join(run_dir, "results", "d_history.csv")

            if os.path.exists(d_history_file):
                print(f"  Found D history file")
                try:
                    df = pd.read_csv(d_history_file)
                    if 'value' in df.columns and len(df) > 0:
                        run_data['final_diffusion'] = df['value'].values[-1]
                        run_data['total_epochs'] = len(df)
                    elif 'D' in df.columns and len(df) > 0:
                        run_data['final_diffusion'] = df['D'].values[-1]
                        run_data['total_epochs'] = len(df)
                    elif df.shape[1] >= 2 and len(df) > 0:
                        # Assume second column is D
                        run_data['final_diffusion'] = df.iloc[-1, 1]
                        run_data['total_epochs'] = len(df)
                except Exception as e:
                    run_data['notes'].append(f"Error reading D history: {str(e)}")

        # Check execution log for training time
        execution_log = os.path.join(run_dir, "execution.log")
        if os.path.exists(execution_log):
            try:
                with open(execution_log, 'r') as f:
                    log_content = f.read()
                    if "Starting Python execution at" in log_content:
                        start_line = [line for line in log_content.split('\n') if "Starting Python execution at" in line][0]
                        start_time_str = start_line.replace("Starting Python execution at", "").strip()
                        start_time = datetime.strptime(start_time_str, "%a %b %d %H:%M:%S %Y")

                        # Look for file modification time as end time
                        end_time = datetime.fromtimestamp(os.path.getmtime(execution_log))

                        # Calculate duration in minutes
                        duration = (end_time - start_time).total_seconds() / 60
                        run_data['training_time_minutes'] = round(duration, 2)
            except Exception as e:
                print(f"  Error parsing execution log: {str(e)}")

        # Check for warnings
        warning_file = os.path.join(run_dir, "warning.txt")
        if os.path.exists(warning_file):
            try:
                with open(warning_file, 'r') as f:
                    warning_content = f.read()
                    run_data['notes'].append(f"WARNING: {warning_content.split('\n')[0]}")
            except Exception:
                pass

        # Calculate physical D values
        if run_data['final_diffusion'] is not None:
            try:
                physical_values = analyze_d_value(run_data['final_diffusion'])
                run_data['final_diffusion_physical'] = {
                    '10^-10_m2_per_s': physical_values['D_m2s_e10'],
                    'mm2_per_min': physical_values['D_mm2min']
                }
            except Exception as e:
                run_data['notes'].append(f"Error calculating physical values: {str(e)}")

        # Add consolidated notes
        run_data['notes'] = '; '.join(run_data['notes']) if run_data['notes'] else None

        # Add to summary data
        summary_data.append(run_data)
        print(f"  Extracted summary for Run {run_num}")

    if not summary_data:
        print("No runs found!")
        return False

    # Create summary file
    create_summary_file(summary_data)
    return True

def create_summary_file(summary_data):
    """Create a summary file with all the extracted information"""
    if not summary_data:
        return

    # Create a nicely formatted text file
    with open("run_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"PINN DIFFUSION COEFFICIENT ANALYSIS SUMMARY\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Write each run's data
        for run in summary_data:
            f.write(f"RUN {run['run_number']} SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Convergence Status: {run['converged']}\n")

            if run['final_diffusion'] is not None:
                f.write(f"Final Diffusion Coefficient (normalized): {run['final_diffusion']:.8f}\n")

                if run['final_diffusion_physical'] is not None:
                    f.write(f"Physical Units:\n")
                    f.write(f"  - 10^-10 m²/s: {run['final_diffusion_physical']['10^-10_m2_per_s']:.6f}\n")
                    f.write(f"  - mm²/min: {run['final_diffusion_physical']['mm2_per_min']:.6f}\n")
            else:
                f.write("Final Diffusion Coefficient: Not available\n")

            if run['total_epochs'] is not None:
                f.write(f"Total Epochs: {run['total_epochs']}\n")

            if run['final_loss'] is not None:
                f.write(f"Final Loss: {run['final_loss']:.8f}\n")

            if run['training_time_minutes'] is not None:
                f.write(f"Training Time: {run['training_time_minutes']:.2f} minutes\n")

            if run['notes']:
                f.write(f"Notes: {run['notes']}\n")

            f.write("\n\n")

        # Add summary statistics if we have multiple runs
        if len(summary_data) > 1:
            d_values = [run['final_diffusion'] for run in summary_data if run['final_diffusion'] is not None]

            if d_values:
                f.write("=" * 80 + "\n")
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Runs Analyzed: {len(summary_data)}\n")
                f.write(f"Runs with D Values: {len(d_values)}\n\n")

                avg_d = np.mean(d_values)
                std_d = np.std(d_values)
                min_d = np.min(d_values)
                max_d = np.max(d_values)

                f.write(f"Diffusion Coefficient Statistics (normalized):\n")
                f.write(f"  - Average: {avg_d:.8f}\n")
                f.write(f"  - Std Dev: {std_d:.8f} ({(std_d/avg_d)*100:.2f}%)\n")
                f.write(f"  - Min: {min_d:.8f}\n")
                f.write(f"  - Max: {max_d:.8f}\n\n")

                # Convert to physical units
                physical_values = analyze_d_value(avg_d)
                f.write(f"Physical Units (Average):\n")
                f.write(f"  - 10^-10 m²/s: {physical_values['D_m2s_e10']:.6f}\n")
                f.write(f"  - mm²/min: {physical_values['D_mm2min']:.6f}\n")

                # Convergence stats
                converged = [run for run in summary_data if str(run['converged']).lower() == 'true']
                not_converged = [run for run in summary_data if str(run['converged']).lower() == 'false']
                unknown = [run for run in summary_data if run['converged'] == 'Unknown']

                f.write(f"\nConvergence Statistics:\n")
                f.write(f"  - Converged: {len(converged)}/{len(summary_data)}\n")
                f.write(f"  - Not Converged: {len(not_converged)}/{len(summary_data)}\n")
                f.write(f"  - Unknown: {len(unknown)}/{len(summary_data)}\n")

    # Also save as JSON for programmatic access
    with open("run_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary saved to run_summary.txt and run_summary.json")

if __name__ == "__main__":
    extract_run_summaries(max_runs=10)
ENDOFSUMMARY

cat > run_d.sh << 'ENDOFRUNSCRIPT'
#!/bin/bash
# Script to run D value extraction and plotting with downsampling

# Configurable parameters
DOWNSAMPLE_FACTOR=100  # Keep only 1 out of every 100 points

echo "====================================================="
echo "PINN Diffusion Coefficient Analysis (Downsampled ${DOWNSAMPLE_FACTOR}x)"
echo "====================================================="

echo -e "\n>>> Step 1: Extracting run summaries (convergence status, final D values)..."
python d_summary.py

echo -e "\n>>> Step 2: Extracting D values with ${DOWNSAMPLE_FACTOR}x downsampling..."
python d_extract.py

echo -e "\n>>> Step 3: Creating D value plots and analysis..."
python d_plt.py

echo -e "\n>>> Analysis complete!"
echo "Check the current directory for:"
echo "- run_summary.txt (plain text summary of all runs)"
echo "- run_summary.json (machine-readable summary)"
echo "- d_value_history_interactive.html (interactive plot)"
echo "- d_value_history.png (static image)"
echo ""
echo "Note: The data has been downsampled by a factor of ${DOWNSAMPLE_FACTOR}."
echo "If you want a different level of downsampling, edit the DOWNSAMPLE_FACTOR"
echo "value at the top of this script and run it again."
ENDOFRUNSCRIPT

# Make the scripts executable
chmod +x d_extract.py d_plt.py d_summary.py run_d.sh

echo "All scripts have been created successfully!"
echo ""
echo "To change the downsampling factor:"
echo "1. Edit the DOWNSAMPLE_FACTOR in run_d.sh"
echo ""
echo "To run the analysis, use:"
echo "./run_d.sh"
echo ""
echo "The new summary functionality will create:"
echo "- run_summary.txt: Plain text summary of all runs with convergence status and final D values"
echo "- run_summary.json: Machine-readable JSON summary for further processing"