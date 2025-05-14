#!/bin/bash

# Add basic error handling
set -e

# Working directory
WORKDIR=$PWD

# Array of case numbers
cases=(1)

# Check if we're in the correct directory
echo "Starting multi-case submission"
echo "Working directory: $WORKDIR"

# Base seed for reproducibility (prime number for better distribution)
BASE_SEED=42

for casei in "${cases[@]}"
do
    echo "Processing: run_$casei"

    # Generate a unique seed for each case
    CASE_SEED=$((BASE_SEED + casei * 97))  # Multiply by a prime for better distribution

    # Verify directory exists
    if [ ! -d "run_$casei" ]; then
        echo "Creating directory run_$casei"
        cp -r $WORKDIR/defaultScripts run_$casei
    fi

    cd "run_$casei"

    # Update the runCase.sh script with the seed
    sed -i "s/CHARCASE/caseX_${casei}/g" runCase.sh

    # Check if the file contains the --seed parameter, add if missing
    if grep -q "\-\-seed" runCase.sh; then
        # Update existing seed
        sed -i "s/\-\-seed [0-9]*/--seed $CASE_SEED/g" runCase.sh
    else
        # Add seed parameter to python command
        sed -i "s/python pinn_trainer\.py/python pinn_trainer.py --seed $CASE_SEED/g" runCase.sh
    fi

    echo "Using seed $CASE_SEED for run_$casei"

    # Submit job
    echo "Submitting job for run_$casei"
    qsub runCase.sh

    cd "$WORKDIR"

    # Add small delay between submissions
    sleep 1
done

echo "All jobs submitted"