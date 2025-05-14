#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -j y
#$ -N CHARCASE
#$ -pe mpich 1
#$ -P WolframGroup

# Source bashrc
source ~/.bashrc

# Set environment variables for better performance
export TF_CPP_MIN_LOG_LEVEL=0  # Show all TensorFlow logs
export PYTHONUNBUFFERED=1      # Ensure Python output is not buffered
export OMP_NUM_THREADS=2       # Limit OpenMP threads
export MKL_NUM_THREADS=2       # Limit MKL threads
export MALLOC_ARENA_MAX=2      # Limit number of memory pools
export MALLOC_TRIM_THRESHOLD_=0  # Ensure memory is returned to OS when freed

# Debug: Print initial state
echo "Initial conda info:"
conda info
echo "Initial Python path:"
which python

# Source TF environment
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

# Fix the GLIBCXX issue by setting proper library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Unset LD_PRELOAD as it might be causing conflicts
unset LD_PRELOAD

# Verify the available GLIBCXX versions for debugging
echo "Checking available GLIBCXX versions:"
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX

# Add the correct absolute path to PYTHONPATH
export PYTHONPATH="/state/partition1/home/qs8/projects/diffusion_pinn${PYTHONPATH:+:$PYTHONPATH}"
echo "Final PYTHONPATH:"
echo $PYTHONPATH

# Case folder setup
caseFolder=$PWD
caseFolderName=$(echo $caseFolder | awk -F "/" '{print $(NF-1)""$NF}')

echo "Case folder: $caseFolder"
echo "Case folder name: $caseFolderName"

# Go to tmp folder for execution
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

echo "Contents of temporary directory:"
ls -la ./$caseFolderName/

# Go to case folder and run
cd $caseFolderName

# Run the layers and neurons optimization script
echo "Starting optimization script at $(date)" > execution.log
python optimize_layers_neurons.py \
    --epochs 5000 \
    --max-layers 5 \
    --output-dir optimization_results >> execution.log 2>&1 || \
    echo "Script failed with exit code $?" >> execution.log

# Check if logRun is still empty or very small
if [ ! -s execution.log ] || [ $(wc -c < execution.log) -lt 100 ]; then
    echo "WARNING: execution log file is empty or very small. Creating debug info." > $caseFolder/empty_log_debug.txt
    echo "Python path:" >> $caseFolder/empty_log_debug.txt
    python -c "import sys; print(sys.path)" >> $caseFolder/empty_log_debug.txt 2>&1
    echo "Directory contents:" >> $caseFolder/empty_log_debug.txt
    ls -la >> $caseFolder/empty_log_debug.txt 2>&1
    echo "Environment:" >> $caseFolder/empty_log_debug.txt
    env >> $caseFolder/empty_log_debug.txt 2>&1
fi

# Cleanup and sync back
cd ..
echo "Syncing results back to $caseFolder"
rsync -uavz ./$caseFolderName/ $caseFolder/
rm -rf $caseFolderName

echo "Job completed at $(date)" >> $caseFolder/execution.log