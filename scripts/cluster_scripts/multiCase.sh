#!/bin/bash
# Seed robustness testing for PINN diffusion coefficient
# This script tests whether your PINN gives consistent but varied results across different seeds

set -e

# Configuration
WORKDIR=$PWD
NUM_RUNS=${1:-10}
SEED_MODE=${2:-"random"}  # Options: "random", "fixed", "sequential"
BASE_SEED=${3:-42}

echo "PINN Seed Robustness Test"
echo "========================"
echo "Working directory: $WORKDIR"
echo "Number of runs: $NUM_RUNS"
echo "Seed mode: $SEED_MODE"

# Verify defaultScripts exists
if [ ! -d "$WORKDIR/defaultScripts" ]; then
    echo "Error: defaultScripts directory not found!"
    echo "Expected: $WORKDIR/defaultScripts"
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
echo "PINN Seed Robustness Test - $(date)" > $TEST_LOG
echo "Seed mode: $SEED_MODE" >> $TEST_LOG
echo "Number of runs: $NUM_RUNS" >> $TEST_LOG
echo "Seeds: ${SEEDS[@]}" >> $TEST_LOG
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
    JOB_NAME="run10_${i}_s${SEED}"
    sed -i "s/CHARCASE/$JOB_NAME/g" runCase.sh

    # Add or update seed parameter in the Python command
    if grep -q "\-\-seed" runCase.sh; then
        # Update existing seed parameter
        sed -i "s/--seed [0-9]*/--seed $SEED/g" runCase.sh
        echo "  Updated existing seed parameter to $SEED"
    else
        # Add seed parameter to Python command
        sed -i "s/python pinn_trainer\.py/python pinn_trainer.py --seed $SEED/g" runCase.sh
        echo "  Added seed parameter $SEED to Python command"
    fi

    # Verify the seed was set correctly
    if grep -q "python pinn_trainer.py --seed $SEED" runCase.sh; then
        echo "  \u2713 Seed $SEED correctly set in runCase.sh"
    else
        echo "  \u2717 Warning: Seed may not be set correctly"
        echo "  Current Python command:"
        grep "python pinn_trainer.py" runCase.sh || echo "  No Python command found!"
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

