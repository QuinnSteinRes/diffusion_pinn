#!/bin/bash
# Updated runCase.sh script with better error handling and memory management

#$ -S /bin/sh
#$ -cwd
#$ -j y
#$ -N CHARCASE
#$ -pe mpich 1
#$ -P WolframGroup

# Better error handling
set -e

# Source bashrc
source ~/.bashrc

# Enable core dumps for debugging
ulimit -c unlimited

# Set environment variables for better error reporting
export TF_CPP_MIN_LOG_LEVEL=0  # Show all TensorFlow logs
export PYTHONUNBUFFERED=1      # Ensure Python output is not buffered
export OMP_NUM_THREADS=2       # Limit OpenMP threads
export MKL_NUM_THREADS=2       # Limit MKL threads

# Set memory limits (90% of available RAM on compute nodes)
# Assuming nodes have 128GB RAM
export MALLOC_ARENA_MAX=2      # Limit number of memory pools
export MALLOC_TRIM_THRESHOLD_=0  # Ensure memory is returned to OS when freed

# Source TF envs
conda activate tbnn

# Debug: Print environment after activation
echo "After conda activate:"
conda info
echo "Python location:"
which python
echo "Python version:"
python --version
echo "PYTHONPATH:"
echo $PYTHONPATH
echo "Conda environment:"
echo $CONDA_DEFAULT_ENV

# Add the correct absolute path to PYTHONPATH
export PYTHONPATH="/state/partition1/home/qs8/projects/diffusion_pinn${PYTHONPATH:+:$PYTHONPATH}"
echo "Final PYTHONPATH:"
echo $PYTHONPATH

# Case folder setup
caseFolder=$PWD
caseFolderName=$(echo $caseFolder | awk -F "/" '{print $(NF-1)""$NF}')

# Calculate seed from case name or job ID for reproducibility
SEED=$(($(echo "$caseFolderName" | cksum | cut -d ' ' -f 1) % 10000))
echo "Using random seed: $SEED"

echo "Case folder: $caseFolder"
echo "Case folder name: $caseFolderName"

# Go to tmp folder
cd /tmp
if [ ! -d $USER ]; then
    mkdir $USER
fi
cd $USER
if [ -d $caseFolderName ]; then
    rm -rf $caseFolderName
fi
mkdir $caseFolderName

# Copy required files
rsync -a $caseFolder/ ./$caseFolderName/
cd $caseFolderName

# Monitor memory usage in background
(
    while true; do
        echo "$(date): Memory usage: $(free -m | grep Mem | awk '{print $3}')MB" >> memory_monitor.log
        sleep 10
    done
) &
MONITOR_PID=$!

# Run with error handling and pass seed
echo "Starting Python execution at $(date)" > execution.log
{
    python pinn_trainer.py --seed $SEED #--epochs 20000
    exit_code=$?
    echo "Python exit code: $exit_code" >> execution.log
} >> logRun 2>&1

# Kill the background monitor
kill $MONITOR_PID || true

# Cleanup and sync back
cd ..
echo "Syncing results back to $caseFolder"
rsync -uavz --exclude="*.core" ./$caseFolderName/ $caseFolder/

# If any core dumps were generated, compress and save them
if ls ./$caseFolderName/*.core 1> /dev/null 2>&1; then
    echo "Core dumps found, compressing..." >> $caseFolder/error_log.txt
    tar -czf $caseFolder/core_dumps.tar.gz ./$caseFolderName/*.core
    echo "Core dumps compressed and saved" >> $caseFolder/error_log.txt
fi

# Cleanup
rm -rf $caseFolderName