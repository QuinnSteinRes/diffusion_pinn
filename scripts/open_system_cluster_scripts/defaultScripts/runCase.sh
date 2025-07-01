#!/bin/bash
#$ -S /bin/sh
#$ -cwd
#$ -j y
#$ -N CHARCASE
#$ -pe mpich 1
#$ -P WolframGroup

set -e

# Source bashrc
source ~/.bashrc

# Enhanced environment for open system
export TF_CPP_MIN_LOG_LEVEL=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0

echo "Starting OPEN SYSTEM PINN job at $(date)"
echo "Job ID: $JOB_ID"
echo "Node: $HOSTNAME"

# Source environment
conda activate tbnn

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

echo "Environment check:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Add project to path
export PYTHONPATH="/state/partition1/home/qs8/projects/diffusion_pinn${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

# Case setup
caseFolder=$PWD
caseFolderName=$(echo $caseFolder | awk -F "/" '{print $(NF-1)""$NF}')

echo "Case folder: $caseFolder"
echo "Case folder name: $caseFolderName"

# Work in tmp
cd /tmp
if [ ! -d $USER ]; then
    mkdir $USER
fi
cd $USER
if [ -d $caseFolderName ]; then
    rm -rf $caseFolderName
fi
mkdir $caseFolderName

# Copy files
rsync -a $caseFolder/ ./$caseFolderName/
cd $caseFolderName

echo "Contents of working directory:"
ls -la

# Monitor memory
(
    while true; do
        echo "$(date): Memory: $(free -m | grep Mem | awk '{print $3}')MB" >> memory_monitor.log
        sleep 30
    done
) &
MONITOR_PID=$!

# Run OPEN SYSTEM training
echo "Starting OPEN SYSTEM PINN training at $(date)" > execution.log
echo "Model: OpenSystemDiffusionPINN" >> execution.log
echo "Learning: D (diffusion) + k (boundary permeability)" >> execution.log

{
    python open_system_pinn_trainer.py --epochs 15000
    exit_code=$?
    echo "Training completed with exit code: $exit_code" >> execution.log
    echo "Model type: OpenSystemDiffusionPINN" >> execution.log
} >> logRun 2>&1

# Stop monitoring
kill $MONITOR_PID || true

# Check for successful completion
if [ -f "training_summary.txt" ]; then
    echo "Training summary found - checking results..." >> execution.log

    # Check if it's actually open system results
    if grep -q "boundary permeability" training_summary.txt; then
        echo "SUCCESS: Open system training completed" >> execution.log
    else
        echo "WARNING: Training completed but no boundary permeability found" >> execution.log
    fi
else
    echo "WARNING: No training summary found" >> execution.log
fi

# Sync back
cd ..
echo "Syncing results back to $caseFolder"
rsync -uavz --exclude="*.core" ./$caseFolderName/ $caseFolder/

# Cleanup
rm -rf $caseFolderName

echo "OPEN SYSTEM job completed at $(date)" >> $caseFolder/execution.log
