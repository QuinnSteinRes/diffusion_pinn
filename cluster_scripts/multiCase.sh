#!/bin/bash

# Add basic error handling
set -e

# Working directory 
WORKDIR=$PWD

# Array of case numbers
#cases=(2 3 4 5 6 7 8 9 10)

cases=(1)

# Check if we're in the correct directory
#if [ ! -d "run_1" ]; then
#    echo "Error: Must be run from the 'runs' directory containing run_X folders"
#    exit 1
#fi

echo "Starting multi-case submission"
echo "Working directory: $WORKDIR"

for casei in "${cases[@]}"
do
    echo "Processing: run_$casei"
    
    # Verify directory exists
    if [ ! -d "run_$casei" ]; then
        #echo "Warning: Missing directory run_$casei"
	echo "Creating directory run_$casei"
	#mkdir run_$casei
	cp -r $WORKDIR/defaultScripts run_$casei
        #continue
    fi
    
    cd "run_$casei"
    
    # Copy and prepare runCase script
    #cp ../../defaultScripts/runCaseNew.sh ./runCase.sh
    #cp ../defaultScripts/runCaseNew.sh ./runCase.sh
    sed -i "s/CHARCASE/casev2_${casei}/g" runCase.sh
    #chmod +x runCase.sh
    
    # Submit job
    echo "Submitting job for run_$casei"
    #qsub -N "caseX_${casei}" runCase.sh
    qsub runCase.sh
    
    cd "$WORKDIR"
    
    # Add small delay between submissions
    sleep 1
done

echo "All jobs submitted"
