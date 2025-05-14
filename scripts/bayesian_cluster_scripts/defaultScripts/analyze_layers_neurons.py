#!/usr/bin/env python3
# analyze_layers_neurons.py - Script to analyze results from multiple layer/neuron optimization runs

import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze layer/neuron optimization results')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base directory containing run_X directories')
    parser.add_argument('--output-file', type=str, default='layers_neurons_analysis.csv',
                        help='Output CSV file for combined results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots if matplotlib is available')
    return parser.parse_args()

def find_result_files(base_dir):
    """Find all search_results.json files in run directories"""
    result_files = []
    
    # Check for run_X directories
    for i in range(1, 20):  # Check up to run_20
        run_dir = os.path.join(base_dir, f"run_{i}")
        if os.path.isdir(run_dir):
            result_file = os.path.join(run_dir, "optimization_results", "results", "search_results.json")
            if os.path.isfile(result_file):
                result_files.append((i, result_file))
    
    return result_files

def load_result_file(run_id, file_path):
    """Load a single result file and attach run_id"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if 'results' in data:
            # Add run_id to each result
            for result in data['results']:
                result['run_id'] = run_id
            
            return data['results']
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def analyze_results(all_results):
    """Analyze aggregated results across runs"""
    if not all_results:
        print("No results found!")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # Create configuration identifier
    df['config'] = df.apply(lambda row: f"{row['layers']}x{row['neurons']}", axis=1)
    
    # Group by configuration
    grouped = df.groupby('config').agg({
        'layers': 'first',
        'neurons': 'first',
        'loss': ['mean', 'std', 'min', 'count'],
        'diffusion_coefficient': ['mean', 'std', 'min', 'max'],
        'parameter_count': 'first'
    }).reset_index()
    
    # Sort by average loss
    sorted_results = grouped.sort_values(('loss', 'mean'))
    
    return sorted_results

def generate_reports(results_df, output_file):
    """Generate reports from the analysis"""
    if results_df is None:
        return
    
    # Save to CSV
    results_df.to_csv(output_file)
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\nTop 5 configurations by mean loss:")
    print("=" * 80)
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        config = row['config']
        layers = row[('layers', 'first')]
        neurons = row[('neurons', 'first')]
        mean_loss = row[('loss', 'mean')]
        std_loss = row[('loss', 'std')]
        count = row[('loss', 'count')]
        mean_diff = row[('diffusion_coefficient', 'mean')]
        std_diff = row[('diffusion_coefficient', 'std')]
        params = row[('parameter_count', 'first')]
        
        print(f"Rank {i+1}: {config} ({count} runs)")
        print(f"  Layers: {layers}, Neurons: {neurons}, Parameters: {params}")
        print(f"  Mean Loss: {mean_loss:.6f} ± {std_loss:.6f}")
        print(f"  Mean Diffusion: {mean_diff:.6f} ± {std_diff:.6f}")
        print("-" * 80)
    
    # Generate text report
    with open(output_file.replace('.csv', '.txt'), 'w') as f:
        f.write("Layer/Neuron Optimization Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total configurations: {len(results_df.index)}\n")
        f.write(f"Total runs analyzed: {results_df[('loss', 'count')].sum()}\n\n")
        
        f.write("Top configurations by mean loss:\n")
        f.write("-" * 80 + "\n")
        
        for i in range(min(10, len(results_df))):
            row = results_df.iloc[i]
            config = row['config']
            layers = row[('layers', 'first')]
            neurons = row[('neurons', 'first')]
            mean_loss = row[('loss', 'mean')]
            std_loss = row[('loss', 'std')]
            count = row[('loss', 'count')]
            mean_diff = row[('diffusion_coefficient', 'mean')]
            std_diff = row[('diffusion_coefficient', 'std')]
            params = row[('parameter_count', 'first')]
            
            f.write(f"Rank {i+1}: {config} ({count} runs)\n")
            f.write(f"  Layers: {layers}, Neurons: {neurons}, Parameters: {params}\n")
            f.write(f"  Mean Loss: {mean_loss:.6f} ± {std_loss:.6f}\n")
            f.write(f"  Mean Diffusion: {mean_diff:.6f} ± {std_diff:.6f}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Detailed report saved to {output_file.replace('.csv', '.txt')}")

def generate_plots(results_df, output_prefix):
    """Generate visualization plots if matplotlib is available"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract data for plotting
        plot_data = pd.DataFrame({
            'layers': results_df[('layers', 'first')],
            'neurons': results_df[('neurons', 'first')],
            'mean_loss': results_df[('loss', 'mean')],
            'std_loss': results_df[('loss', 'std')],
            'count': results_df[('loss', 'count')],
            'mean_diffusion': results_df[('diffusion_coefficient', 'mean')],
            'params': results_df[('parameter_count', 'first')]
        })
        
        # 1. Heat map of loss vs layers/neurons
        plt.figure(figsize=(12, 10))
        
        # Create pivoted data for heatmap
        pivot_data = plot_data.pivot_table(
            index='layers', 
            columns='neurons', 
            values='mean_loss'
        )
        
        # Create the heatmap
        ax = sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis_r',
                         linewidths=.5, cbar_kws={'label': 'Mean Loss'})
        
        plt.title('Mean Loss by Network Architecture', fontsize=16)
        plt.xlabel('Neurons per Layer', fontsize=14)
        plt.ylabel('Number of Layers', fontsize=14)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_loss_heatmap.png", dpi=300)
        plt.close()
        
        # 2. Model size vs performance plot
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with size based on number of runs
        scatter = plt.scatter(
            plot_data['params'], 
            plot_data['mean_loss'],
            s=plot_data['count'] * 30,  # Size based on count
            c=plot_data['mean_diffusion'],  # Color based on diffusion coefficient
            alpha=0.7,
            cmap='plasma',
            edgecolors='black'
        )
        
        # Add color bar for diffusion coefficient
        cbar = plt.colorbar(scatter)
        cbar.set_label('Mean Diffusion Coefficient', fontsize=12)
        
        # Add annotations for top configurations
        for i in range(min(5, len(plot_data))):
            plt.annotate(
                f"{plot_data['layers'].iloc[i]}x{plot_data['neurons'].iloc[i]}",
                (plot_data['params'].iloc[i], plot_data['mean_loss'].iloc[i]),
                xytext=(10, 5),
                textcoords='offset points',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
            )
        
        plt.title('Model Size vs Performance', fontsize=16)
        plt.xlabel('Number of Parameters', fontsize=14)
        plt.ylabel('Mean Loss', fontsize=14)
        plt.xscale('log')  # Log scale for parameters
        plt.yscale('log')  # Log scale for loss
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_size_vs_performance.png", dpi=300)
        plt.close()
        
        # 3. Correlation between diffusion coefficient and loss
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(
            plot_data['mean_loss'],
            plot_data['mean_diffusion'],
            s=plot_data['count'] * 30,
            c=plot_data['params'],
            alpha=0.7,
            cmap='viridis',
            edgecolors='black'
        )
        
        # Add color bar for parameters
        cbar = plt.colorbar()
        cbar.set_label('Number of Parameters', fontsize=12)
        
        plt.title('Relationship between Loss and Diffusion Coefficient', fontsize=16)
        plt.xlabel('Mean Loss', fontsize=14)
        plt.ylabel('Mean Diffusion Coefficient', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_loss_vs_diffusion.png", dpi=300)
        plt.close()
        
        print(f"Plots saved with prefix: {output_prefix}")
        
    except ImportError:
        print("Matplotlib or seaborn not available. Skipping plots.")
    except Exception as e:
        print(f"Error generating plots: {e}")

def main():
    args = parse_arguments()
    
    print(f"Analyzing layer/neuron optimization results in {args.base_dir}")
    
    # Find result files
    result_files = find_result_files(args.base_dir)
    print(f"Found {len(result_files)} result files")
    
    if not result_files:
        print("No result files found. Make sure the runs have completed.")
        return
    
    # Load and combine all results
    all_results = []
    for run_id, file_path in result_files:
        print(f"Loading results from run_{run_id}: {file_path}")
        results = load_result_file(run_id, file_path)
        all_results.extend(results)
        print(f"  Found {len(results)} configurations")
    
    print(f"Total configurations loaded: {len(all_results)}")
    
    # Analyze results
    results_df = analyze_results(all_results)
    if results_df is None:
        return
    
    # Generate reports
    generate_reports(results_df, args.output_file)
    
    # Generate plots if requested
    if args.plot:
        output_prefix = os.path.splitext(args.output_file)[0]
        generate_plots(results_df, output_prefix)

if __name__ == "__main__":
    main()
