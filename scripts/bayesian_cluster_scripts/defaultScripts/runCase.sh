#$ -S /bin/sh
#$ -cwd
#$ -j y
#$ -N CHARCASE
#$ -pe mpich 1
#$ -P OzelGroup
##$ -P WolframGroup

# Source bashrc
source ~/.bashrc

# Calculate seed from job ID for reproducibility
SEED=$(($(echo "$JOB_ID" | cksum | cut -d ' ' -f 1) % 10000))
echo "Using random seed: $SEED"

# Debug: Print initial state
echo "Initial conda info:"
conda info
echo "Initial Python path:"
which python

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
echo "Listing pip packages:"
pip list

# Add the correct absolute path to PYTHONPATH and fix C++ library issues
export PYTHONPATH="/state/partition1/home/qs8/projects/diffusion_pinn${PYTHONPATH:+:$PYTHONPATH}"
echo "Final PYTHONPATH:"
echo $PYTHONPATH

# Try to fix the libstdc++ issue by ignoring system libraries
export CONDA_OVERRIDE_GLIBC=2.35  # This might help with some library conflicts
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"  # Force using conda's libraries

# Case folder setup
caseFolder=$PWD
caseFolderName=$(echo $caseFolder | awk -F "/" '{print $(NF-1)""$NF}')

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

echo "Contents of temporary directory:"
ls -la ./$caseFolderName/

# Go to case folder and run
cd $caseFolderName

# Run the minimal version that avoids matplotlib with seed
echo "Starting optimization script at $(date)" > logRun
python optimize_minimal.py --seed $SEED >> logRun 2>&1 || echo "Script failed with exit code $?" >> logRun

# Check if logRun is still empty or very small
if [ ! -s logRun ] || [ $(wc -c < logRun) -lt 100 ]; then
    echo "WARNING: logRun file is empty or very small. Creating debug info." > $caseFolder/empty_log_debug.txt
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