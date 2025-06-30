#!/bin/bash
# Open system multi-case runner with proper seed management

set -e

WORKDIR=$PWD
NUM_RUNS=${1:-5}
SEED_MODE=${2:-"sequential"}
BASE_SEED=${3:-42}

echo "OPEN SYSTEM PINN Multi-Case Runner"
echo "=================================="
echo "Model: OpenSystemDiffusionPINN with Robin Boundaries"
echo "Parameters: D (diffusion) + k (boundary permeability)"
echo "Working directory: $WORKDIR"
echo "Number of runs: $NUM_RUNS"
echo "Seed mode: $SEED_MODE"

# Verify we have open system scripts
if [ ! -d "$WORKDIR/defaultScripts" ]; then
    echo "Error: defaultScripts directory not found!"
    exit 1
fi

if [ ! -f "$WORKDIR/defaultScripts/open_system_pinn_trainer.py" ]; then
    echo "Error: open_system_pinn_trainer.py not found!"
    echo "Make sure you're using the NEW open system scripts."
    exit 1
fi

# Generate seeds
declare -a SEEDS
case $SEED_MODE in
    "sequential")
        echo "Generating sequential seeds starting from $BASE_SEED..."
        for i in $(seq 1 $NUM_RUNS); do
            SEEDS[$i]=$((BASE_SEED + i - 1))
        done
        ;;
    "random")
        echo "Generating random seeds..."
        for i in $(seq 1 $NUM_RUNS); do
            SEEDS[$i]=$RANDOM
        done
        ;;
    "fixed")
        echo "Using predetermined high-quality seeds..."
        FIXED_SEEDS=(42 137 251 389 523 677 809 967 1103 1249)
        for i in $(seq 1 $NUM_RUNS); do
            if [ $i -le ${#FIXED_SEEDS[@]} ]; then
                SEEDS[$i]=${FIXED_SEEDS[$((i-1))]}
            else
                SEEDS[$i]=$((42 + i * 97))  # Prime-based fallback
            fi
        done
        ;;
    *)
        echo "Error: Unknown seed mode '$SEED_MODE'"
        exit 1
        ;;
esac

echo "Seeds for this test: ${SEEDS[@]}"

# Create test log
TEST_LOG="$WORKDIR/open_system_test_log.txt"
echo "Open System PINN Test - $(date)" > $TEST_LOG
echo "Model: OpenSystemDiffusionPINN" >> $TEST_LOG
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

    # Create run directory
    if [ -d "$RUN_DIR" ]; then
        echo "  Cleaning existing directory $RUN_DIR"
        rm -rf "$RUN_DIR"
    fi

    echo "  Copying scripts to $RUN_DIR"
    cp -r "$WORKDIR/defaultScripts" "$RUN_DIR"

    cd "$RUN_DIR"

    # Update job name
    JOB_NAME="opensys_${i}_s${SEED}"
    sed -i "s/CHARCASE/$JOB_NAME/g" runCase.sh

    # Set seed in the training script
    if grep -q "\-\-seed" runCase.sh; then
        sed -i "s/--seed [0-9]*/--seed $SEED/g" runCase.sh
        echo "  Updated seed parameter to $SEED"
    else
        # Add seed parameter to Python command
        sed -i "s/python open_system_pinn_trainer\.py/python open_system_pinn_trainer.py --seed $SEED/g" runCase.sh
        echo "  Added seed parameter $SEED"
    fi

    # Verify the script is using open system model
    if ! grep -q "OpenSystemDiffusionPINN" runCase.sh && ! grep -q "open_system_pinn_trainer.py" runCase.sh; then
        echo "  ERROR: Script not configured for open system!"
        echo "  Check that you're using the correct trainer script."
        cd "$WORKDIR"
        continue
    fi

    echo "  Submitting open system job $JOB_NAME"
    qsub runCase.sh

    cd "$WORKDIR"
    sleep 2
done

echo ""
echo "All $NUM_RUNS OPEN SYSTEM jobs submitted successfully!"
echo "Seeds used: ${SEEDS[@]}"
echo ""
echo "IMPORTANT: These jobs will learn TWO parameters:"
echo "  - D (diffusion coefficient)"
echo "  - k (boundary permeability)"
echo ""
echo "After completion, use the open system post-processing:"
echo "  ./create_open_system_scripts.sh"
echo "  ./run_open_system_analysis.sh"