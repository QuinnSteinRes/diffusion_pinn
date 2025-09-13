#!/bin/bash
# Updated multiCase.sh - Compatible with new PINN labeled data interface
# Seed robustness testing for PINN diffusion coefficient - ASCII only

set -e

# Configuration
WORKDIR=$PWD
NUM_RUNS=${1:-10}
SEED_MODE=${2:-"random"}  # Options: "random", "fixed", "sequential"
BASE_SEED=${3:-42}

echo "PINN Seed Robustness Test (Updated for New Interface)"
echo "===================================================="
echo "Working directory: $WORKDIR"
echo "Number of runs: $NUM_RUNS"
echo "Seed mode: $SEED_MODE"

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

# Generate seeds based on mode
declare -a SEEDS
case $SEED_MODE in
    "random")
        echo "Generating random seeds..."
        for i in $(seq 1 $NUM_RUNS); do
            SEEDS[$i]=$RANDOM
        done
        ;;
    "sequential")
        echo "Generating sequential seeds starting from $BASE_SEED..."
        for i in $(seq 1 $NUM_RUNS); do
            SEEDS[$i]=$((BASE_SEED + i - 1))
        done
        ;;
    "fixed")
        echo "Using predetermined seed list..."
        FIXED_SEEDS=(42 55 71 89 107 127 149 173 199 227 251 277 307 337 367 397 431 463 499 541)
        for i in $(seq 1 $NUM_RUNS); do
            if [ $i -le ${#FIXED_SEEDS[@]} ]; then
                SEEDS[$i]=${FIXED_SEEDS[$((i-1))]}
            else
                # If we need more runs than fixed seeds, generate random ones
                SEEDS[$i]=$RANDOM
            fi
        done
        ;;
    *)
        echo "Error: Unknown seed mode '$SEED_MODE'"
        echo "Valid modes: random, fixed, sequential"
        exit 1
        ;;
esac

# Log the seeds being used
echo "Seeds for this test: ${SEEDS[@]}"
echo ""

# Create a test log
TEST_LOG="$WORKDIR/seed_test_log.txt"
echo "PINN Seed Robustness Test (New Interface) - $(date)" > $TEST_LOG
echo "Seed mode: $SEED_MODE" >> $TEST_LOG
echo "Number of runs: $NUM_RUNS" >> $TEST_LOG
echo "Seeds: ${SEEDS[@]}" >> $TEST_LOG
echo "New interface: labeled data structure" >> $TEST_LOG
echo "========================================" >> $TEST_LOG

# Process each run
for i in $(seq 1 $NUM_RUNS)
do
    SEED=${SEEDS[$i]}
    RUN_DIR="run_$i"

    echo "Processing: $RUN_DIR with seed $SEED"
    echo "Run $i: seed $SEED" >> $TEST_LOG

    # Create or clean run directory
    if [ -d "$RUN_DIR" ]; then
        echo "  Cleaning existing directory $RUN_DIR"
        rm -rf "$RUN_DIR"
    fi

    echo "  Copying defaultScripts to $RUN_DIR"
    cp -r "$WORKDIR/defaultScripts" "$RUN_DIR"

    cd "$RUN_DIR"

    # Update job name to include seed for tracking
    JOB_NAME="pinn_${i}_s${SEED}"
    sed -i "s/CHARCASE/$JOB_NAME/g" runCase.sh

    # Update the Python command with proper arguments for new interface
    # Remove any existing seed arguments first
    sed -i 's/--seed [0-9]*//g' runCase.sh
    sed -i 's/--epochs [0-9]*//g' runCase.sh

    # Add proper arguments with new interface
    if grep -q "python pinn_trainer\.py" runCase.sh; then
        # Update existing Python command
        sed -i "s/python pinn_trainer\.py.*/python pinn_trainer.py --seed $SEED --epochs 100 --output-dir \./" runCase.sh
        echo "  [OK] Updated Python command with seed $SEED and new interface"
    else
        echo "  [ERROR] Warning: Could not find Python command in runCase.sh"
        echo "  Current runCase.sh content:"
        grep -n "python" runCase.sh || echo "  No Python commands found!"
    fi

    # Verify the command was set correctly
    if grep -q "python pinn_trainer.py --seed $SEED" runCase.sh; then
        echo "  [OK] Seed $SEED correctly set in runCase.sh"
    else
        echo "  [WARNING] Warning: Seed may not be set correctly"
        echo "  Current Python command:"
        grep "python pinn_trainer.py" runCase.sh || echo "  No Python command found!"
    fi

    # Ensure data file is present
    if [ ! -f "intensity_time_series_spatial_temporal.csv" ]; then
        echo "  Warning: Data file missing, copying from defaultScripts"
        cp "$DATA_FILE" .
    fi

    # Submit job
    echo "  Submitting job $JOB_NAME"
    qsub runCase.sh

    cd "$WORKDIR"

    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "All $NUM_RUNS jobs submitted successfully!"
echo "Seeds used: ${SEEDS[@]}"
echo ""
echo "Key updates for new interface:"
echo "- Using labeled data structure (boundary, interior, physics points)"
echo "- Updated parameter names (N_boundary, N_interior, N_collocation)"
echo "- Improved memory management and error handling"
echo "- Enhanced convergence checking"
echo ""
echo "Monitor progress with:"
echo "qstat -u $USER"
echo ""
echo "After completion, run post-processing:"
echo "./create_scripts.sh && ./run_d.sh"