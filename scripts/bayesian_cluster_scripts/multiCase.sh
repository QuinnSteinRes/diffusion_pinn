#!/bin/bash

# Add basic error handling
set -e

# Working directory
WORKDIR=$PWD

# Configuration
OPTIMIZATION_TYPE=${1:-"full"}  # Options: full, layers_neurons
NUM_RUNS=${2:-1}                # Number of parallel runs to submit

echo "Starting Bayesian optimization runs"
echo "Working directory: $WORKDIR"
echo "Optimization type: $OPTIMIZATION_TYPE"
echo "Number of runs: $NUM_RUNS"

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

    # Copy required files based on optimization type
    if [ "$OPTIMIZATION_TYPE" == "layers_neurons" ]; then
        # Copy specialized script for layers/neurons optimization
        cp "$WORKDIR/defaultScripts/optimize_layers_neurons.py" "run_$casei/"
        cp "$WORKDIR/defaultScripts/runCase_layers_neurons.sh" "run_$casei/runCase.sh"
        JOB_NAME="layeropt_${casei}"
    else
        # Copy full Bayesian optimization script
        cp "$WORKDIR/defaultScripts/optimize_minimal.py" "run_$casei/"
        cp "$WORKDIR/defaultScripts/runCase.sh" "run_$casei/"
        JOB_NAME="bayOp_${casei}"
    fi

    # Copy data file and other dependencies
    cp "$WORKDIR/defaultScripts/intensity_time_series_spatial_temporal.csv" "run_$casei/"

    # Add permissions
    chmod +x "run_$casei"/*.py
    chmod +x "run_$casei"/*.sh

    # Change to run directory
    cd "run_$casei"

    # Update job name in runCase script
    sed -i "s/CHARCASE/$JOB_NAME/g" runCase.sh

    # Add or update seed parameter
    if grep -q "\-\-seed" runCase.sh; then
        # Update existing seed
        sed -i "s/\-\-seed [0-9]*/--seed $CASE_SEED/g" runCase.sh
    else
        # Add seed parameter to python command (for different script types)
        sed -i "s/python optimize_minimal\.py/python optimize_minimal.py --seed $CASE_SEED/g" runCase.sh
        sed -i "s/python optimize_layers_neurons\.py/python optimize_layers_neurons.py --seed $CASE_SEED/g" runCase.sh
    fi

    # Submit job
    echo "Submitting job $JOB_NAME for run_$casei with seed $CASE_SEED"
    qsub runCase.sh

    # Return to main directory
    cd "$WORKDIR"

    # Add small delay between submissions
    sleep 1
done

echo "All jobs submitted"