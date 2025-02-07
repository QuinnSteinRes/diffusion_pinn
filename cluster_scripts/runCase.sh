#$ -S /bin/sh
#$ -cwd
#$ -j y
#$ -N CHARCASE
#$ -pe mpich 1
#$ -P OzelGroup

# Source bashrc
source ~/.bashrc

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

# Add the correct absolute path to PYTHONPATH
export PYTHONPATH="/state/partition1/home/qs8/projects/diffusion_pinn${PYTHONPATH:+:$PYTHONPATH}"
echo "Final PYTHONPATH:"
echo $PYTHONPATH

# Case folder setup
caseFolder=$PWD
#defaultScriptsDir="$(dirname $(dirname $caseFolder))/defaultScripts"
caseFolderName=$(echo $caseFolder | awk -F "/" '{print $(NF-1)""$NF}')

echo "Case folder: $caseFolder"
#echo "Default scripts directory: $defaultScriptsDir"
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
#cp "$defaultScriptsDir/train_diffusion_batch_new.py" "./$caseFolderName/"
rsync -a $caseFolder/ ./$caseFolderName/

echo "Contents of temporary directory:"
ls -la ./$caseFolderName/

# Go to case folder and run
cd $caseFolderName
#python train_diffusion_batch_new.py 1> logRun
python pinn_trainer.py 1> logRun

# Cleanup and sync back
cd ..
rsync -uavz ./$caseFolderName/ $caseFolder/
rm -rf $caseFolderName
