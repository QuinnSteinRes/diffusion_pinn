#!/bin/bash

# Add basic error handling
set -e

# Working directory
WORKDIR=$PWD

# Configuration
NUM_RUNS=${1:-1}       # Number of parallel runs to submit (default: 1)
MAX_LAYERS=${2:-5}     # Maximum number of layers to test (default: 5)
EPOCHS=${3:-5000}      # Number of epochs per configuration (default: 5000)

echo "Starting Layer/Neuron Optimization"
echo "Working directory: $WORKDIR"
echo "Number of runs: $NUM_RUNS"
echo "Max layers: $MAX_LAYERS"
echo "Epochs: $EPOCHS"

# Base seed for reproducibility (prime number for better distribution)
BASE_SEED=42

# Make sure defaultScripts directory exists
if [ ! -d "$WORKDIR/defaultScripts" ]; then
    echo "Error: defaultScripts directory not found!"
    exit 1
fi

# Create run directories and submit jobs
for casei in $(seq 1 $NUM_RUNS)
do
    echo "Processing: run_$casei"

    # Generate a unique seed for each case
    CASE_SEED=$((BASE_SEED + casei * 97))  # Multiply by a prime for better distribution
    echo "Using seed $CASE_SEED for run_$casei"

    # Verify directory exists
    if [ ! -d "run_$casei" ]; then
        echo "Creating directory run_$casei"
        mkdir -p "run_$casei"
    fi

    # Copy required files
    cp "$WORKDIR/defaultScripts/optimize_layers_neurons.py" "run_$casei/"
    cp "$WORKDIR/defaultScripts/runCase_layers_neurons.sh" "run_$casei/runCase.sh"

    # Copy data file
    cp "$WORKDIR/defaultScripts/intensity_time_series_spatial_temporal.csv" "run_$casei/"

    # Add permissions
    chmod +x "run_$casei"/*.py
    chmod +x "run_$casei"/*.sh

    # Change to run directory
    cd "run_$casei"

    # Update job name in runCase script
    sed -i "s/CHARCASE/layers_${casei}/g" runCase.sh

    # Submit job with customized parameters including seed
    echo "Submitting job for run_$casei with max_layers=$MAX_LAYERS, epochs=$EPOCHS, seed=$CASE_SEED"

    # Update the command in runCase.sh
    COMMAND_LINE="python optimize_layers_neurons.py \\\\\n    --epochs $EPOCHS \\\\\n    --max-layers $MAX_LAYERS \\\\\n    --output-dir optimization_results \\\\\n    --seed $CASE_SEED"

    # Replace the existing python command line
    sed -i "s|python optimize_layers_neurons\.py.*$SEED|$COMMAND_LINE|" runCase.sh

    # Submit the job
    qsub runCase.sh

    # Return to main directory
    cd "$WORKDIR"

    # Add small delay between submissions
    sleep 1
done

echo "All jobs submitted"

cat << EOF

To analyze results after all jobs complete, run:
python analyze_layers_neurons.py

This will combine results from all runs and identify the best configurations.
EOF