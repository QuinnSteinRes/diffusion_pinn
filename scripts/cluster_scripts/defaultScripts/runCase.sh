#!/bin/bash
# Updated runCase.sh script with version tracking and better error handling

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
export MALLOC_ARENA_MAX=2      # Limit number of memory pools
export MALLOC_TRIM_THRESHOLD_=0  # Ensure memory is returned to OS when freed

# Source TF envs
conda activate tbnn

# Add the correct absolute path to PYTHONPATH
export PYTHONPATH="/state/partition1/home/qs8/projects/diffusion_pinn${PYTHONPATH:+:$PYTHONPATH}"

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
cd $caseFolderName

# ============================================================================
# VERSION AND GIT TRACKING - START
# ============================================================================
echo "============================================================================" > logRun
echo "PINN DIFFUSION COEFFICIENT ESTIMATION RUN" >> logRun
echo "Start Time: $(date)" >> logRun
echo "Job ID: ${JOB_ID:-LOCAL}" >> logRun
echo "Node: $(hostname)" >> logRun
echo "============================================================================" >> logRun

# Capture diffusion_pinn package version
echo "" >> logRun
echo "SOFTWARE VERSIONS:" >> logRun
echo "----------------" >> logRun

# Get Python package version
python -c "
try:
    import diffusion_pinn
    print('diffusion_pinn version: ' + getattr(diffusion_pinn, '__version__', 'unknown'))
except ImportError:
    print('diffusion_pinn version: package not found')
except Exception as e:
    print('diffusion_pinn version: error - ' + str(e))
" >> logRun 2>&1

# Get Python version
echo "Python version: $(python --version 2>&1)" >> logRun

# Get TensorFlow version
python -c "
try:
    import tensorflow as tf
    print('TensorFlow version: ' + tf.__version__)
except ImportError:
    print('TensorFlow version: not available')
except Exception as e:
    print('TensorFlow version: error - ' + str(e))
" >> logRun 2>&1

# Get NumPy version
python -c "
try:
    import numpy as np
    print('NumPy version: ' + np.__version__)
except ImportError:
    print('NumPy version: not available')
except Exception as e:
    print('NumPy version: error - ' + str(e))
" >> logRun 2>&1

echo "" >> logRun
echo "GIT REPOSITORY STATUS:" >> logRun
echo "--------------------" >> logRun

# Capture git information from the diffusion_pinn project
DIFFUSION_PINN_PATH="/state/partition1/home/qs8/projects/diffusion_pinn"
if [ -d "$DIFFUSION_PINN_PATH/.git" ]; then
    cd "$DIFFUSION_PINN_PATH"

    # Get current commit hash
    echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')" >> "$caseFolderName/logRun"

    # Get current branch
    echo "Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')" >> "$caseFolderName/logRun"

    # Get commit message
    echo "Commit message: $(git log -1 --pretty=format:'%s' 2>/dev/null || echo 'unknown')" >> "$caseFolderName/logRun"

    # Get commit date
    echo "Commit date: $(git log -1 --pretty=format:'%ci' 2>/dev/null || echo 'unknown')" >> "$caseFolderName/logRun"

    # Check if working directory is clean
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "Working directory: clean" >> "$caseFolderName/logRun"
    else
        echo "Working directory: has uncommitted changes" >> "$caseFolderName/logRun"
        echo "Modified files:" >> "$caseFolderName/logRun"
        git diff --name-only 2>/dev/null >> "$caseFolderName/logRun" || echo "  Could not list modified files" >> "$caseFolderName/logRun"
    fi

    # Get last few commits for context
    echo "" >> "$caseFolderName/logRun"
    echo "Recent commits:" >> "$caseFolderName/logRun"
    git log --oneline -5 2>/dev/null >> "$caseFolderName/logRun" || echo "  Could not retrieve commit history" >> "$caseFolderName/logRun"

    cd /tmp/$USER/$caseFolderName
else
    echo "Git repository: not found at $DIFFUSION_PINN_PATH" >> logRun
fi

echo "" >> logRun
echo "RUNTIME ENVIRONMENT:" >> logRun
echo "------------------" >> logRun
echo "PYTHONPATH: $PYTHONPATH" >> logRun
echo "Conda environment: $CONDA_DEFAULT_ENV" >> logRun
echo "Working directory: $(pwd)" >> logRun

# Get PINN configuration from variables.py
echo "" >> logRun
echo "PINN CONFIGURATION:" >> logRun
echo "-----------------" >> logRun
python -c "
try:
    from diffusion_pinn.variables import PINN_VARIABLES
    for key, value in sorted(PINN_VARIABLES.items()):
        print(f'{key}: {value}')
except ImportError:
    print('Could not import PINN_VARIABLES')
except Exception as e:
    print(f'Error reading PINN_VARIABLES: {e}')
" >> logRun 2>&1

echo "" >> logRun
echo "============================================================================" >> logRun
echo "TRAINING LOG:" >> logRun
echo "============================================================================" >> logRun
# ============================================================================
# VERSION AND GIT TRACKING - END
# ============================================================================

# Monitor memory usage in background
(
    while true; do
        echo "$(date): Memory usage: $(free -m | grep Mem | awk '{print $3}')MB" >> memory_monitor.log
        sleep 10
    done
) &
MONITOR_PID=$!

# Run with error handling
echo "Starting Python execution at $(date)" >> logRun
{
    python pinn_trainer.py
    exit_code=$?
    echo "Python exit code: $exit_code" >> logRun
    echo "Training completed at: $(date)" >> logRun
} >> logRun 2>&1

# Kill the background monitor
kill $MONITOR_PID || true

# Add final summary to log
echo "" >> logRun
echo "============================================================================" >> logRun
echo "RUN COMPLETED: $(date)" >> logRun
echo "Total runtime: $SECONDS seconds" >> logRun
echo "============================================================================" >> logRun

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