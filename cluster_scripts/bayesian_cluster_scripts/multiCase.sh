#!/bin/bash

# Add basic error handling
set -e

# Working directory
WORKDIR=$PWD

# Array of case numbers - can be expanded for multiple runs with different parameters
cases=(1)

echo "Starting Bayesian optimization runs"
echo "Working directory: $WORKDIR"

for casei in "${cases[@]}"
do
    echo "Processing: run_$casei"

    # Verify directory exists
    if [ ! -d "run_$casei" ]; then
        echo "Creating directory run_$casei"
        mkdir -p run_$casei
        cp -r $WORKDIR/defaultScripts/* run_$casei/
    fi

    cd "run_$casei"

    # Prepare runCase script - replace placeholder with actual case name
    sed -i "s/CHARCASE/bayesOpt_${casei}/g" runCase.sh
    chmod +x runCase.sh

    # Submit job
    echo "Submitting Bayesian optimization job for run_$casei"
    qsub runCase.sh

    cd "$WORKDIR"

    # Add small delay between submissions
    sleep 1
done

echo "All jobs submitted"
