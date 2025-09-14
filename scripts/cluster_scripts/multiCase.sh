#!/bin/bash
# Fixed multiCase.sh - FORCES variables.py configuration, no overrides
# Seed robustness testing for PINN diffusion coefficient

set -e

# Configuration
WORKDIR=$PWD
NUM_RUNS=${1:-10}

echo "PINN Seed Robustness Test (FORCED variables.py Configuration)"
echo "============================================================="
echo "Working directory: $WORKDIR"
echo "Number of runs: $NUM_RUNS"
echo "ALL PARAMETERS FORCED FROM variables.py - NO OVERRIDES"
echo "============================================================="

# Verify defaultScripts exists
if [ ! -d "$WORKDIR/defaultScripts" ]; then
    echo "Error: defaultScripts directory not found!"
    echo "Expected: $WORKDIR/defaultScripts"
    exit 1
fi

# Verify data file exists
DATA_FILE="$WORKDIR/defaultScripts/intensity_time_series_spatial_temporal.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found!"
    echo "Expected: $DATA_FILE"
    echo "Please copy your data file to the defaultScripts directory."
    exit 1
fi

# Create a test log
TEST_LOG="$WORKDIR/seed_test_log.txt"
echo "PINN Seed Robustness Test (FORCED variables.py) - $(date)" > $TEST_LOG
echo "Number of runs: $NUM_RUNS" >> $TEST_LOG
echo "Configuration: ALL FROM variables.py - NO OVERRIDES" >> $TEST_LOG
echo "========================================" >> $TEST_LOG

# Process each run
for i in $(seq 1 $NUM_RUNS)
do
    RUN_DIR="run_$i"

    echo "Processing: $RUN_DIR (using variables.py configuration)"
    echo "Run $i: variables.py config" >> $TEST_LOG

    # Create or clean run directory
    if [ -d "$RUN_DIR" ]; then
        echo "  Cleaning existing directory $RUN_DIR"
        rm -rf "$RUN_DIR"
    fi

    echo "  Copying defaultScripts to $RUN_DIR"
    cp -r "$WORKDIR/defaultScripts" "$RUN_DIR"

    cd "$RUN_DIR"

    # Update job name for tracking
    JOB_NAME="pinn_${i}"
    sed -i "s/CHARCASE/$JOB_NAME/g" runCase.sh

    # CRITICAL: Remove any parameter overrides and force clean command
    if grep -q "python pinn_trainer\.py" runCase.sh; then
        # Replace with clean command - NO ARGUMENTS
        sed -i "s/python pinn_trainer\.py.*/python pinn_trainer.py/" runCase.sh
        echo "  [OK] Set clean Python command (no parameter overrides)"
    else
        echo "  [ERROR] Could not find Python command in runCase.sh"
        echo "  Current runCase.sh content:"
        grep -n "python" runCase.sh || echo "  No Python commands found!"
        cd "$WORKDIR"
        continue
    fi

    # Verify the command is clean
    if grep -q "python pinn_trainer.py$" runCase.sh; then
        echo "  [OK] Clean command verified (no parameter overrides)"
    else
        echo "  [WARNING] Command may have parameter overrides:"
        grep "python pinn_trainer.py" runCase.sh
    fi

    # Ensure data file is present
    if [ ! -f "intensity_time_series_spatial_temporal.csv" ]; then
        echo "  Warning: Data file missing, copying from defaultScripts"
        cp "$DATA_FILE" .
    fi

    # Submit job
    echo "  Submitting job $JOB_NAME (variables.py config)"
    qsub runCase.sh

    cd "$WORKDIR"

    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "All $NUM_RUNS jobs submitted successfully!"
echo "Configuration: ALL FROM variables.py (epochs: from PINN_VARIABLES['epochs'])"
echo ""
echo "Key features of this version:"
echo "- NO parameter overrides - everything from variables.py"
echo "- NO --epochs arguments"
echo "- NO --seed arguments"
echo "- Consistent configuration across all runs"
echo "- Seed comes from PINN_VARIABLES['random_seed']"
echo ""
echo "Monitor progress with:"
echo "qstat -u $USER"
echo ""
echo "After completion, run post-processing:"
echo "./create_scripts.sh && ./run_d.sh"