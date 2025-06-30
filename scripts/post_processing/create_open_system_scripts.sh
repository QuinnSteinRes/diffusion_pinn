#!/bin/bash
# create_open_system_scripts.sh
# Post-processing for open system PINN with boundary flux

cat > extract_open_system_parameters.py << 'ENDOFEXTRACT'
#!/usr/bin/env python3
# Extract both D and k parameters from open system runs

import os
import pandas as pd
import numpy as np
import json

def extract_open_system_parameters(downsample_factor=100):
    """
    Extract D and k values from open system parameter_history.json files
    """
    print("Starting open system parameter extraction...")
    print(f"Downsample factor: {downsample_factor}")

    all_D_values = {}
    all_k_values = {}
    max_length = 0

    # Check for run directories
    max_run_to_check = 10

    for run_num in range(1, max_run_to_check + 1):
        run_name = f"Run_{run_num}"

        # Look for the new JSON format first
        json_path = os.path.join(f"run_{run_num}", "saved_models", "parameter_history_final.json")

        if os.path.exists(json_path):
            print(f"Found parameter history for {run_name}")

            try:
                with open(json_path, 'r') as f:
                    history_data = json.load(f)

                if 'D_history' in history_data and 'k_history' in history_data:
                    # Apply downsampling
                    D_values = np.array(history_data['D_history'])[::downsample_factor]
                    k_values = np.array(history_data['k_history'])[::downsample_factor]

                    all_D_values[run_name] = D_values
                    all_k_values[run_name] = k_values
                    max_length = max(max_length, len(D_values))

                    print(f"  Extracted {len(D_values)} parameter points after downsampling")
                    print(f"  Final D: {history_data.get('final_D', 'unknown'):.6e}")
                    print(f"  Final k: {history_data.get('final_k', 'unknown'):.6e}")
                else:
                    print(f"  Warning: JSON missing D_history or k_history")

            except Exception as e:
                print(f"  Error reading {json_path}: {e}")

        # Fallback: look for old CSV format
        else:
            csv_path = os.path.join(f"run_{run_num}", "results", "parameter_history.csv")
            if os.path.exists(csv_path):
                print(f"Found CSV parameter history for {run_name}")
                try:
                    df = pd.read_csv(csv_path)
                    if 'D_value' in df.columns and 'k_value' in df.columns:
                        D_values = df['D_value'].values[::downsample_factor]
                        k_values = df['k_value'].values[::downsample_factor]

                        all_D_values[run_name] = D_values
                        all_k_values[run_name] = k_values
                        max_length = max(max_length, len(D_values))

                        print(f"  Extracted {len(D_values)} parameter points from CSV")
                except Exception as e:
                    print(f"  Error reading {csv_path}: {e}")

    # Save combined results
    if all_D_values and all_k_values:
        # Create combined DataFrames
        combined_D_df = pd.DataFrame()
        combined_k_df = pd.DataFrame()

        for run_name in all_D_values.keys():
            # Pad with NaN if needed
            D_padded = np.pad(all_D_values[run_name], (0, max_length - len(all_D_values[run_name])),
                             mode='constant', constant_values=np.nan)
            k_padded = np.pad(all_k_values[run_name], (0, max_length - len(all_k_values[run_name])),
                             mode='constant', constant_values=np.nan)

            combined_D_df[run_name] = D_padded
            combined_k_df[run_name] = k_padded

        # Save to CSV
        combined_D_df.to_csv("D_parameter_history.csv", index=False)
        combined_k_df.to_csv("k_parameter_history.csv", index=False)

        print(f"\nSuccessfully saved {len(all_D_values)} runs:")
        print(f"  D parameters: D_parameter_history.csv")
        print(f"  k parameters: k_parameter_history.csv")
        print(f"  Each run has {max_length} epochs of data (downsampled)")

        return True
    else:
        print("\nNo open system parameter histories found.")
        print("Make sure you're using the new OpenSystemDiffusionPINN model.")
        return False

if __name__ == "__main__":
    extract_open_system_parameters(downsample_factor=100)
ENDOFEXTRACT

cat > plot_open_system_analysis.py << 'ENDOFPLOT'
#!/usr/bin/env python3
# Comprehensive analysis and plotting for open system results

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def analyze_parameter_value(D_value, k_value, roi_dimensions=(299, 99),
                           pixel_size_mm=0.0513, time_step_seconds=60):
    """Convert normalized parameters to physical units and analyze"""
    # Get characteristic length
    L_pixels = max(roi_dimensions)
    L_mm = L_pixels * pixel_size_mm

    # Convert D from normalized to physical units
    D_mm2s = D_value * L_mm**2 / time_step_seconds
    D_mm2min = D_mm2s * 60
    D_m2s = D_mm2s * 1e-6
    D_m2s_e10 = D_m2s * 1e10

    # Convert k (boundary permeability) - units are 1/time in normalized coordinates
    k_per_second = k_value / time_step_seconds
    k_per_minute = k_per_second * 60

    # Physical interpretation
    diffusion_time_scale = 1.0 / D_value if D_value > 0 else float('inf')
    outflow_time_scale = 1.0 / k_value if k_value > 0 else float('inf')

    # Peclet number analog for outflow vs diffusion
    Pe_outflow = k_value / D_value if D_value > 0 else float('inf')

    return {
        'D_normalized': D_value,
        'k_normalized': k_value,
        'D_m2s_e10': D_m2s_e10,
        'D_mm2min': D_mm2min,
        'k_per_minute': k_per_minute,
        'diffusion_time_scale': diffusion_time_scale,
        'outflow_time_scale': outflow_time_scale,
        'Pe_outflow': Pe_outflow,
        'system_type': 'outflow_dominated' if Pe_outflow > 1 else 'diffusion_dominated'
    }

def plot_open_system_results(
    D_csv_file="D_parameter_history.csv",
    k_csv_file="k_parameter_history.csv",
    roi_dimensions=(299, 99),
    pixel_size_mm=0.0513,
    time_step_seconds=60,
    downsample_factor=100
):
    """Create comprehensive analysis of open system results"""

    print(f"Analyzing open system results...")

    # Check if files exist
    if not os.path.exists(D_csv_file) or not os.path.exists(k_csv_file):
        print(f"Error: Parameter files not found!")
        print(f"Expected: {D_csv_file} and {k_csv_file}")
        return False

    # Read data
    try:
        D_df = pd.read_csv(D_csv_file)
        k_df = pd.read_csv(k_csv_file)

        if D_df.empty or k_df.empty:
            print("Error: Empty parameter files")
            return False

        print(f"Found {len(D_df.columns)} runs in the dataset")
        print(f"Each run has {len(D_df)} data points")

    except Exception as e:
        print(f"Error reading parameter files: {e}")
        return False

    # Generate epoch array
    epochs = np.arange(0, len(D_df) * downsample_factor, downsample_factor)

    # Create comprehensive figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Diffusion Coefficient Evolution (D)",
            "Boundary Permeability Evolution (k)",
            "Physical Units: D (10^-10 m²/s)",
            "Physical Units: k (1/min)",
            "Time Scale Comparison",
            "Outflow Peclet Number (k/D)"
        )
    )

    # Process each run
    final_D_values = []
    final_k_values = []

    for col in D_df.columns:
        if col in k_df.columns:
            D_values = D_df[col].dropna().values
            k_values = k_df[col].dropna().values

            if len(D_values) > 0 and len(k_values) > 0:
                min_length = min(len(D_values), len(k_values))
                D_values = D_values[:min_length]
                k_values = k_values[:min_length]
                epoch_subset = epochs[:min_length]

                # Plot normalized parameters
                fig.add_trace(
                    go.Scatter(x=epoch_subset, y=D_values, name=f"{col} (D)",
                              mode='lines', line=dict(width=1.5)),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=epoch_subset, y=k_values, name=f"{col} (k)",
                              mode='lines', line=dict(width=1.5), showlegend=False),
                    row=1, col=2
                )

                # Convert to physical units
                D_physical = []
                k_physical = []
                time_scales_D = []
                time_scales_k = []
                pe_numbers = []

                for D_val, k_val in zip(D_values, k_values):
                    if not (np.isnan(D_val) or np.isnan(k_val)):
                        result = analyze_parameter_value(D_val, k_val, roi_dimensions,
                                                       pixel_size_mm, time_step_seconds)
                        D_physical.append(result['D_m2s_e10'])
                        k_physical.append(result['k_per_minute'])
                        time_scales_D.append(result['diffusion_time_scale'])
                        time_scales_k.append(result['outflow_time_scale'])
                        pe_numbers.append(result['Pe_outflow'])

                # Plot physical units
                if D_physical and k_physical:
                    fig.add_trace(
                        go.Scatter(x=epoch_subset[:len(D_physical)], y=D_physical,
                                  name=f"{col} (D phys)", mode='lines', line=dict(width=1.5),
                                  showlegend=False),
                        row=2, col=1
                    )

                    fig.add_trace(
                        go.Scatter(x=epoch_subset[:len(k_physical)], y=k_physical,
                                  name=f"{col} (k phys)", mode='lines', line=dict(width=1.5),
                                  showlegend=False),
                        row=2, col=2
                    )

                    # Time scales
                    fig.add_trace(
                        go.Scatter(x=epoch_subset[:len(time_scales_D)], y=time_scales_D,
                                  name=f"{col} (diff time)", mode='lines', line=dict(width=1.5),
                                  showlegend=False),
                        row=3, col=1
                    )

                    fig.add_trace(
                        go.Scatter(x=epoch_subset[:len(time_scales_k)], y=time_scales_k,
                                  name=f"{col} (outflow time)", mode='lines', line=dict(dash='dash', width=1.5),
                                  showlegend=False),
                        row=3, col=1
                    )

                    # Peclet numbers
                    fig.add_trace(
                        go.Scatter(x=epoch_subset[:len(pe_numbers)], y=pe_numbers,
                                  name=f"{col} (Pe)", mode='lines', line=dict(width=1.5),
                                  showlegend=False),
                        row=3, col=2
                    )

                # Store final values
                final_D_values.append(D_values[-1])
                final_k_values.append(k_values[-1])

    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.5, row=3, col=2)
    fig.add_annotation(x=0.5, y=1.0, text="Pe=1 (transition)", showarrow=False,
                      xref="x6", yref="y6", textangle=0)

    # Update layout
    fig.update_layout(
        height=1000,
        title_text=f"Open System PINN Analysis (Downsampled {downsample_factor}x)<br>" +
                   f"Learning both D (diffusion) and k (boundary permeability)",
        showlegend=True
    )

    # Update axis labels
    fig.update_yaxes(title_text="D (normalized)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="k (normalized)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="D (10^-10 m²/s)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="k (1/min)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="Time Scale", type="log", row=3, col=1)
    fig.update_yaxes(title_text="Pe_outflow", type="log", row=3, col=2)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=2)

    # Save interactive plot
    html_file = "open_system_analysis_interactive.html"
    fig.write_html(html_file, auto_open=False)
    print(f"Saved interactive analysis to {html_file}")

    # Save static image
    try:
        png_file = "open_system_analysis.png"
        fig.write_image(png_file, width=1400, height=1000)
        print(f"Saved static image to {png_file}")
    except Exception as e:
        print(f"Note: Could not save PNG: {e}")

    # Analysis summary
    analyze_final_open_system_values(final_D_values, final_k_values, roi_dimensions,
                                   pixel_size_mm, time_step_seconds)

    return True

def analyze_final_open_system_values(D_values, k_values, roi_dimensions,
                                   pixel_size_mm, time_step_seconds):
    """Analyze final parameter values across all runs"""
    print("\n" + "="*60)
    print("FINAL OPEN SYSTEM PARAMETER ANALYSIS")
    print("="*60)

    if not D_values or not k_values:
        print("No parameter values to analyze")
        return

    # Statistics
    D_mean, D_std = np.mean(D_values), np.std(D_values)
    k_mean, k_std = np.mean(k_values), np.std(k_values)

    print(f"\nDiffusion Coefficient (D):")
    print(f"  Mean: {D_mean:.6e} ± {D_std:.6e}")
    print(f"  Range: [{np.min(D_values):.6e}, {np.max(D_values):.6e}]")
    print(f"  Coefficient of variation: {D_std/D_mean*100:.2f}%")

    print(f"\nBoundary Permeability (k):")
    print(f"  Mean: {k_mean:.6e} ± {k_std:.6e}")
    print(f"  Range: [{np.min(k_values):.6e}, {np.max(k_values):.6e}]")
    print(f"  Coefficient of variation: {k_std/k_mean*100:.2f}%")

    # Physical interpretation of mean values
    result = analyze_parameter_value(D_mean, k_mean, roi_dimensions,
                                   pixel_size_mm, time_step_seconds)

    print(f"\nPhysical Interpretation (Mean Values):")
    print(f"  D in 10^-10 m²/s: {result['D_m2s_e10']:.6f}")
    print(f"  D in mm²/min: {result['D_mm2min']:.6f}")
    print(f"  k in 1/min: {result['k_per_minute']:.6f}")
    print(f"  Diffusion time scale: {result['diffusion_time_scale']:.2f} time units")
    print(f"  Outflow time scale: {result['outflow_time_scale']:.2f} time units")
    print(f"  Outflow Peclet number: {result['Pe_outflow']:.3f}")
    print(f"  System type: {result['system_type'].replace('_', ' ').title()}")

    # System characterization
    print(f"\nSystem Characterization:")
    if result['Pe_outflow'] > 10:
        print("  STRONGLY outflow-dominated: Mass loss through boundaries >> diffusive mixing")
    elif result['Pe_outflow'] > 1:
        print("  MODERATELY outflow-dominated: Boundary losses significant vs diffusion")
    elif result['Pe_outflow'] > 0.1:
        print("  BALANCED system: Diffusion and outflow roughly comparable")
    else:
        print("  DIFFUSION-dominated: Internal mixing >> boundary losses")

    # Half-life calculation
    half_life_outflow = np.log(2) / k_mean if k_mean > 0 else float('inf')
    print(f"  Half-life due to outflow: {half_life_outflow:.2f} time units")

    # Consistency check
    print(f"\nConsistency Analysis:")
    D_cv = D_std / D_mean * 100
    k_cv = k_std / k_mean * 100

    if D_cv < 10 and k_cv < 10:
        print("  EXCELLENT: Both parameters show good consistency across runs")
    elif D_cv < 20 and k_cv < 20:
        print("  GOOD: Parameters reasonably consistent across runs")
    else:
        print("  WARNING: High variability suggests convergence issues or insufficient training")

if __name__ == "__main__":
    plot_open_system_results()
ENDOFPLOT

cat > create_open_system_summary.py << 'ENDOFSUMMARY'
#!/usr/bin/env python3
# Extract summary information for open system runs

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

def analyze_parameter_value(D_value, k_value, roi_dimensions=(299, 99),
                           pixel_size_mm=0.0513, time_step_seconds=60):
    """Convert normalized parameters to physical units"""
    L_pixels = max(roi_dimensions)
    L_mm = L_pixels * pixel_size_mm

    D_mm2s = D_value * L_mm**2 / time_step_seconds
    D_m2s_e10 = D_mm2s * 1e-6 * 1e10
    k_per_minute = k_value / time_step_seconds * 60

    return {
        'D_normalized': D_value,
        'k_normalized': k_value,
        'D_m2s_e10': D_m2s_e10,
        'k_per_minute': k_per_minute,
        'Pe_outflow': k_value / D_value if D_value > 0 else float('inf')
    }

def extract_open_system_summaries(max_runs=10):
    """Extract summary information from open system runs"""
    print("Extracting open system run summaries...")
    summary_data = []

    for run_num in range(1, max_runs + 1):
        run_name = f"Run {run_num}"
        run_dir = f"run_{run_num}"

        if not os.path.exists(run_dir):
            continue

        print(f"Checking {run_name} in {run_dir}...")

        run_data = {
            'run_number': run_num,
            'run_name': run_name,
            'converged': 'Unknown',
            'final_D': None,
            'final_k': None,
            'final_D_physical': None,
            'final_k_physical': None,
            'system_type': 'Unknown',
            'total_epochs': None,
            'final_loss': None,
            'training_time_minutes': None,
            'notes': []
        }

        # Check for open system summary files
        summary_txt = os.path.join(run_dir, "training_summary.txt")

        if os.path.exists(summary_txt):
            print(f"  Found open system summary")
            with open(summary_txt, 'r') as f:
                content = f.read()

                # Extract D and k values
                for line in content.split('\n'):
                    if "Final diffusion coefficient (D):" in line:
                        try:
                            run_data['final_D'] = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif "Final boundary permeability (k):" in line:
                        try:
                            run_data['final_k'] = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif "Converged:" in line:
                        try:
                            run_data['converged'] = line.split(':')[1].strip()
                        except:
                            pass
                    elif "Final loss:" in line:
                        try:
                            run_data['final_loss'] = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif "Total epochs:" in line:
                        try:
                            run_data['total_epochs'] = int(line.split(':')[1].strip())
                        except:
                            pass

        # Check for parameter history JSON
        param_json = os.path.join(run_dir, "saved_models", "parameter_history_final.json")
        if os.path.exists(param_json):
            print(f"  Found parameter history JSON")
            try:
                with open(param_json, 'r') as f:
                    param_data = json.load(f)

                if 'final_D' in param_data:
                    run_data['final_D'] = param_data['final_D']
                if 'final_k' in param_data:
                    run_data['final_k'] = param_data['final_k']
                if 'epochs' in param_data:
                    run_data['total_epochs'] = param_data['epochs']

            except Exception as e:
                run_data['notes'].append(f"Error reading parameter JSON: {str(e)}")

        # Calculate physical values and system type
        if run_data['final_D'] is not None and run_data['final_k'] is not None:
            try:
                physical_values = analyze_parameter_value(run_data['final_D'], run_data['final_k'])
                run_data['final_D_physical'] = physical_values['D_m2s_e10']
                run_data['final_k_physical'] = physical_values['k_per_minute']

                if physical_values['Pe_outflow'] > 1:
                    run_data['system_type'] = 'Outflow-dominated'
                else:
                    run_data['system_type'] = 'Diffusion-dominated'

            except Exception as e:
                run_data['notes'].append(f"Error calculating physical values: {str(e)}")

        # Check execution log for timing
        execution_log = os.path.join(run_dir, "execution.log")
        if os.path.exists(execution_log):
            try:
                with open(execution_log, 'r') as f:
                    log_content = f.read()

                # Look for start and end times
                start_time = None
                end_time = None

                for line in log_content.split('\n'):
                    if "Starting OPEN SYSTEM training" in line and "Starting" in line:
                        try:
                            # Extract time from log
                            time_str = line.split(' - ')[-1]
                            start_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    elif "OPEN SYSTEM training completed" in line:
                        try:
                            time_str = line.split(' - ')[-1]
                            end_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                        except:
                            pass

                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds() / 60
                    run_data['training_time_minutes'] = round(duration, 2)

            except Exception as e:
                print(f"  Error parsing execution log: {str(e)}")

        # Add consolidated notes
        run_data['notes'] = '; '.join(run_data['notes']) if run_data['notes'] else None

        summary_data.append(run_data)
        print(f"  Extracted summary for Run {run_num}")

    if not summary_data:
        print("No open system runs found!")
        return False

    # Create summary files
    create_open_system_summary_file(summary_data)
    return True

def create_open_system_summary_file(summary_data):
    """Create comprehensive summary file for open system runs"""
    if not summary_data:
        return

    # Create detailed text summary
    with open("open_system_run_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OPEN SYSTEM PINN ANALYSIS SUMMARY\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("MODEL TYPE: Open System with Robin Boundary Conditions\n")
        f.write("PDE: ∂c/∂t = D∇²c (interior)\n")
        f.write("BC:  -D(∂c/∂n) = k(c - c_ext) (boundaries)\n")
        f.write("PARAMETERS: D (diffusion coefficient), k (boundary permeability)\n\n")

        # Individual run summaries
        for run in summary_data:
            f.write(f"RUN {run['run_number']} SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Convergence Status: {run['converged']}\n")

            if run['final_D'] is not None and run['final_k'] is not None:
                f.write(f"Final Diffusion Coefficient (D): {run['final_D']:.8e}\n")
                f.write(f"Final Boundary Permeability (k): {run['final_k']:.8e}\n")

                if run['final_D_physical'] is not None:
                    f.write(f"Physical Units:\n")
                    f.write(f"  D: {run['final_D_physical']:.6f} × 10^-10 m²/s\n")
                    f.write(f"  k: {run['final_k_physical']:.6f} 1/min\n")

                f.write(f"System Type: {run['system_type']}\n")
            else:
                f.write("Parameters: Not available\n")

            if run['total_epochs'] is not None:
                f.write(f"Total Epochs: {run['total_epochs']}\n")

            if run['final_loss'] is not None:
                f.write(f"Final Loss: {run['final_loss']:.8f}\n")

            if run['training_time_minutes'] is not None:
                f.write(f"Training Time: {run['training_time_minutes']:.2f} minutes\n")

            if run['notes']:
                f.write(f"Notes: {run['notes']}\n")

            f.write("\n\n")

        # Overall statistics
        valid_runs = [run for run in summary_data if run['final_D'] is not None and run['final_k'] is not None]

        if len(valid_runs) > 1:
            f.write("=" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Runs Analyzed: {len(summary_data)}\n")
            f.write(f"Runs with Valid Parameters: {len(valid_runs)}\n\n")

            # Parameter statistics
            D_values = [run['final_D'] for run in valid_runs]
            k_values = [run['final_k'] for run in valid_runs]

            D_mean, D_std = np.mean(D_values), np.std(D_values)
            k_mean, k_std = np.mean(k_values), np.std(k_values)

            f.write("Diffusion Coefficient (D) Statistics:\n")
            f.write(f"  Mean: {D_mean:.8e}\n")
            f.write(f"  Std Dev: {D_std:.8e} ({(D_std/D_mean)*100:.2f}%)\n")
            f.write(f"  Range: [{np.min(D_values):.8e}, {np.max(D_values):.8e}]\n\n")

            f.write("Boundary Permeability (k) Statistics:\n")
            f.write(f"  Mean: {k_mean:.8e}\n")
            f.write(f"  Std Dev: {k_std:.8e} ({(k_std/k_mean)*100:.2f}%)\n")
            f.write(f"  Range: [{np.min(k_values):.8e}, {np.max(k_values):.8e}]\n\n")

            # Physical interpretation
            from analyze_parameter_value import analyze_parameter_value  # Local function
            try:
                physical = analyze_parameter_value(D_mean, k_mean)
                f.write("Physical Interpretation (Mean Values):\n")
                f.write(f"  D: {physical['D_m2s_e10']:.6f} × 10^-10 m²/s\n")
                f.write(f"  k: {physical['k_per_minute']:.6f} 1/min\n")
                f.write(f"  Outflow Peclet Number: {physical['Pe_outflow']:.3f}\n")

                if physical['Pe_outflow'] > 1:
                    f.write("  System: Outflow-dominated (boundary losses > diffusive mixing)\n")
                else:
                    f.write("  System: Diffusion-dominated (internal mixing > boundary losses)\n")
            except:
                pass

            # Convergence statistics
            converged_runs = [run for run in valid_runs if str(run['converged']).lower() == 'true']
            f.write(f"\nConvergence Statistics:\n")
            f.write(f"  Converged: {len(converged_runs)}/{len(valid_runs)}\n")
            f.write(f"  Convergence Rate: {len(converged_runs)/len(valid_runs)*100:.1f}%\n")

            # System type distribution
            outflow_dominated = len([run for run in valid_runs if run['system_type'] == 'Outflow-dominated'])
            diffusion_dominated = len([run for run in valid_runs if run['system_type'] == 'Diffusion-dominated'])
            f.write(f"\nSystem Type Distribution:\n")
            f.write(f"  Outflow-dominated: {outflow_dominated}/{len(valid_runs)}\n")
            f.write(f"  Diffusion-dominated: {diffusion_dominated}/{len(valid_runs)}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("VALIDATION RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Check mass conservation plots match data trends\n")
        f.write("2. Verify boundary flux residuals are small\n")
        f.write("3. Compare D values with independent measurements\n")
        f.write("4. Validate k values with boundary transport experiments\n")
        f.write("5. Test model predictions on held-out data\n")
        f.write("6. Check parameter consistency across different seeds\n")

    # Save as JSON for programmatic access
    with open("open_system_run_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)

    print("Open system summary saved to:")
    print("  - open_system_run_summary.txt (human readable)")
    print("  - open_system_run_summary.json (machine readable)")

if __name__ == "__main__":
    extract_open_system_summaries(max_runs=10)
ENDOFSUMMARY

cat > run_open_system_analysis.sh << 'ENDOFRUNSCRIPT'
#!/bin/bash
# Master script to run complete open system analysis

# Configuration
DOWNSAMPLE_FACTOR=100

echo "================================================================="
echo "OPEN SYSTEM PINN ANALYSIS PIPELINE"
echo "================================================================="
echo "Model: OpenSystemDiffusionPINN with Robin Boundary Conditions"
echo "Parameters: D (diffusion) + k (boundary permeability)"
echo "Downsampling: ${DOWNSAMPLE_FACTOR}x"
echo ""

echo ">>> Step 1: Extracting run summaries (D, k, convergence status)..."
python create_open_system_summary.py

echo ""
echo ">>> Step 2: Extracting parameter histories with downsampling..."
python extract_open_system_parameters.py

echo ""
echo ">>> Step 3: Creating comprehensive analysis plots..."
python plot_open_system_analysis.py

echo ""
echo ">>> Analysis complete!"
echo ""
echo "Generated files:"
echo "- open_system_run_summary.txt          (detailed text summary)"
echo "- open_system_run_summary.json         (machine-readable summary)"
echo "- D_parameter_history.csv              (diffusion coefficient evolution)"
echo "- k_parameter_history.csv              (boundary permeability evolution)"
echo "- open_system_analysis_interactive.html (interactive plots)"
echo "- open_system_analysis.png             (static summary plot)"
echo ""
echo "Key differences from closed system analysis:"
echo "1. Now tracks TWO physical parameters (D and k)"
echo "2. Provides system characterization (outflow vs diffusion dominated)"
echo "3. Includes time scale analysis and Peclet numbers"
echo "4. Validates mass conservation instead of assuming it"
echo "5. Parameters have actual physical meaning"
echo ""
echo "Next steps:"
echo "1. Check that mass conservation plots match your data"
echo "2. Validate D values against independent measurements"
echo "3. Compare k values with boundary transport properties"
echo "4. Verify the system type makes physical sense"
ENDOFRUNSCRIPT

# Make scripts executable
chmod +x extract_open_system_parameters.py
chmod +x plot_open_system_analysis.py
chmod +x create_open_system_summary.py
chmod +x run_open_system_analysis.sh

echo "Open system post-processing scripts created successfully!"
echo ""
echo "CRITICAL DIFFERENCES from your old scripts:"
echo "1. Extracts TWO parameters (D and k) instead of just D"
echo "2. Provides physical interpretation of boundary transport"
echo "3. Characterizes system as outflow vs diffusion dominated"
echo "4. Validates mass conservation instead of assuming it"
echo "5. All parameters now have actual physical meaning"
echo ""
echo "To run the analysis:"
echo "./run_open_system_analysis.sh"
echo ""
echo "WARNING: These scripts will NOT work with your old closed-system results!"
echo "You must retrain using OpenSystemDiffusionPINN first."