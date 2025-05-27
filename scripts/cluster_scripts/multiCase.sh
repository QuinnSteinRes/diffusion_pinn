#!/bin/bash

# Add basic error handling
set -e

# Working directory
WORKDIR=$PWD

# cases=(1 2 3 4 5 6 7 8 9 10)

# Array of case numbers
cases=(1)

# Check if we're in the correct directory
echo "Starting multi-case submission"
echo "Working directory: $WORKDIR"

for casei in "${cases[@]}"
do
    echo "Processing: run_$casei"

    # Verify directory exists
    if [ ! -d "run_$casei" ]; then
        echo "Creating directory run_$casei"
        cp -r $WORKDIR/defaultScripts run_$casei
    fi

    cd "run_$casei"

    # Update the case name in runCase.sh without changing the seed
    sed -i "s/CHARCASE/caseX_${casei}/g" runCase.sh

    # Remove any existing seed parameter to use the default from variables.py
    if grep -q "\-\-seed" runCase.sh; then
        # Remove the seed parameter entirely
        sed -i "s/--seed [0-9]*//g" runCase.sh
    fi

    echo "Using default seed from variables.py for run_$casei"

    # Submit job
    echo "Submitting job for run_$casei"
    qsub runCase.sh

    cd "$WORKDIR"

    # Add small delay between submissions
    sleep 1
done

echo "All jobs submitted"