#!/bin/bash
# Simplified multiCase.sh for seed robustness testing

set -e

WORKDIR=$PWD
SEED_LIST=(42 55 71 89 107 127 149 173 199 227)
NUM_RUNS=${1:-10}

echo "Starting test - $NUM_RUNS runs"

if [ ! -d "$WORKDIR/defaultScripts" ]; then
    echo "Error: defaultScripts directory not found!"
    exit 1
fi

for i in $(seq 1 $NUM_RUNS)
do
    SEED=${SEED_LIST[$((i-1))]}

    echo "Processing run_$i with seed $SEED"

    if [ ! -d "run_$i" ]; then
        cp -r $WORKDIR/defaultScripts run_$i
    fi

    cd "run_$i"

    # Set job name and seed
    sed -i "s/CHARCASE/cmap_sml_${i}_s${SEED}/g" runCase.sh

    if grep -q "\-\-seed" runCase.sh; then
        sed -i "s/--seed [0-9]*/--seed $SEED/g" runCase.sh
    else
        sed -i "s/python pinn_trainer\.py/python pinn_trainer.py --seed $SEED/g" runCase.sh
    fi

    echo "  Submitting job for run_$i (seed $SEED)"
    qsub runCase.sh
    cd "$WORKDIR"
    sleep 1
done

echo "All $NUM_RUNS jobs submitted with seeds: ${SEED_LIST[@]:0:$NUM_RUNS}"
