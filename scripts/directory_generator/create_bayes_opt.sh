#!/bin/bash
# create_bayes_opt.sh
# Script to create a new Bayesian optimization directory for PINN
# Author: Quinn Stein (modified by [your name])

# Display help message if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [case_name] [options]"
    echo "Creates a new directory with scripts for PINN Bayesian optimization"
    echo ""
    echo "Arguments:"
    echo "  case_name          Name of the case directory (default: YYYYMMDD_BayesOpt)"
    echo ""
    echo "Options:"
    echo "  -s, --source DIR   Source directory for scripts (default: ~/projects/diffusion_pinn/scripts)"
    echo "  -d, --dest DIR     Parent directory for the new case (default: ~/projects/pinnRuns)"
    echo "  -r, --runs N       Number of parallel runs to prepare (default: 1)"
    echo "  -i, --iter N       Number of Bayesian optimization iterations (default: 20)"
    echo "  -e, --epochs N     Number of epochs per iteration (default: 10000)"
    echo "  -l, --layers       Use focused layer/neuron optimization instead of full Bayesian opt"
    echo "  -h, --help         Display this help message and exit"
    exit 0
fi

# Default values
SOURCE_DIR="$HOME/projects/diffusion_pinn/scripts"
POST_PROCESS_DIR="$HOME/projects/diffusion_pinn/scripts/post_processing"
DEST_PARENT="$HOME/projects/pinnRuns"
CURRENT_DATE=$(date +"%Y%m%d")
CASE_NAME="${CURRENT_DATE}_BayesOpt"
NUM_RUNS=1
ITERATIONS=20
EPOCHS=10000
USE_LAYERS_OPT=false

# Process command line arguments
if [ -n "$1" ] && [[ ! "$1" == -* ]]; then
    CASE_NAME="$1"
    shift
fi

# Process additional options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--source)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -d|--dest)
            DEST_PARENT="$2"
            shift 2
            ;;
        -r|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -i|--iter)
            ITERATIONS="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--layers)
            USE_LAYERS_OPT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine which script directories to use
SCRIPT_SRC="$SOURCE_DIR/bayesian_cluster_scripts"

# Check if source directory exists
if [ ! -d "$SCRIPT_SRC" ]; then
    echo "Error: Source directory $SCRIPT_SRC does not exist!"
    echo "Make sure the diffusion_pinn project has the scripts directory structure:"
    echo "  $SOURCE_DIR/bayesian_cluster_scripts"
    exit 1
fi

# Check if source contains required files
if [ ! -d "$SCRIPT_SRC/defaultScripts" ]; then
    echo "Error: Source directory $SCRIPT_SRC does not contain required defaultScripts/ directory!"
    exit 1
fi

# Set up destination directory
DEST_DIR="$DEST_PARENT/$CASE_NAME"

# Check if destination directory already exists
if [ -d "$DEST_DIR" ]; then
    read -p "Directory $DEST_DIR already exists. Overwrite? (y/N): " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy files
echo "Creating new Bayesian optimization directory in $DEST_DIR"
echo "Copying default scripts..."

# Copy the defaultScripts directory
cp -r "$SCRIPT_SRC/defaultScripts" "$DEST_DIR/"

# Copy the appropriate multiCase script
if [ "$USE_LAYERS_OPT" = true ]; then
    # Use layer neuron optimization script
    cp "$SCRIPT_SRC/run_layers_neurons.sh" "$DEST_DIR/multiCase.sh"
    echo "Using layer/neuron focused optimization"
else
    # Use standard Bayesian optimization script
    cp "$SCRIPT_SRC/multiCase.sh" "$DEST_DIR/"
    echo "Using full Bayesian optimization"
fi

# Add post-processing script
echo "Adding post-processing script..."
if [ -f "$POST_PROCESS_DIR/create_scripts.sh" ]; then
    cp "$POST_PROCESS_DIR/create_scripts.sh" "$DEST_DIR/"
    chmod +x "$DEST_DIR/create_scripts.sh"
    echo "Post-processing script added successfully."
else
    echo "Warning: create_scripts.sh not found in $POST_PROCESS_DIR"
    echo "Expected post-processing script at: $POST_PROCESS_DIR/create_scripts.sh"
fi

# Make sure scripts are executable
chmod +x "$DEST_DIR/multiCase.sh"
find "$DEST_DIR/defaultScripts" -name "*.sh" -exec chmod +x {} \;
find "$DEST_DIR/defaultScripts" -name "*.py" -exec chmod +x {} \;

# Add data files
DATA_DIR="$HOME/projects/diffusion_pinn/src/diffusion_pinn/data"
echo "Copying data files..."
if [ -f "$DATA_DIR/intensity_time_series_spatial_temporal.csv" ]; then
    cp "$DATA_DIR/intensity_time_series_spatial_temporal.csv" "$DEST_DIR/defaultScripts/"
    echo "Data file copied successfully."
else
    echo "Warning: Data file not found in $DATA_DIR"
    echo "You may need to manually copy the data file to $DEST_DIR/defaultScripts/"
fi

# Modify scripts based on parameters
if [ "$USE_LAYERS_OPT" = true ]; then
    # Update the run_layers_neurons.sh with custom parameters
    sed -i "s/NUM_RUNS=\${1:-1}/NUM_RUNS=\${1:-$NUM_RUNS}/g" "$DEST_DIR/multiCase.sh"
    sed -i "s/EPOCHS=\${3:-5000}/EPOCHS=\${3:-$EPOCHS}/g" "$DEST_DIR/multiCase.sh"
else
    # Modify multiCase.sh to use the specified number of runs
    if [ "$NUM_RUNS" -ne 1 ]; then
        echo "Configuring for $NUM_RUNS runs..."
        sed -i "s/NUM_RUNS=\${2:-1}/NUM_RUNS=\${2:-$NUM_RUNS}/g" "$DEST_DIR/multiCase.sh"
    fi
    
    # Update optimization parameters in the Python scripts
    OPTIMIZE_SCRIPT="$DEST_DIR/defaultScripts/optimize_minimal.py"
    if [ -f "$OPTIMIZE_SCRIPT" ]; then
        echo "Updating optimization parameters in script..."
        sed -i "s/default=OPTIMIZATION_SETTINGS\['iterations_optimizer'\]/default=$ITERATIONS/g" "$OPTIMIZE_SCRIPT"
        sed -i "s/default=OPTIMIZATION_SETTINGS\['network_epochs'\]/default=$EPOCHS/g" "$OPTIMIZE_SCRIPT"
    fi
fi

# Create a README file with usage instructions
cat > "$DEST_DIR/README.md" << EOF
# Bayesian Optimization for PINN

This directory contains scripts for Bayesian optimization of Physics-Informed Neural Networks (PINNs) 
for the diffusion coefficient problem.

## Configuration

- Number of parallel runs: $NUM_RUNS
- Number of iterations: $ITERATIONS
- Epochs per iteration: $EPOCHS
- Optimization type: $([ "$USE_LAYERS_OPT" = true ] && echo "Layer/Neuron Grid Search" || echo "Full Bayesian Optimization")

## Running the optimization

To run all optimization processes in parallel:

```bash
cd $DEST_DIR
./multiCase.sh
```

## Post-processing results

After the runs complete, use the post-processing script to analyze results:

```bash
./create_scripts.sh
./run_d.sh
```

The post-processing will:
1. Extract diffusion coefficient history
2. Create interactive plots
3. Summarize convergence status and final values

## Output

Each run will create its own directory with:
- Optimization logs and results
- Best model parameters
- Diffusion coefficient history
- Loss history

The combined analysis will provide statistics across all runs.
EOF

echo "Successfully created Bayesian optimization directory: $DEST_DIR"
echo ""
echo "Directory structure:"
ls -la "$DEST_DIR"
echo ""
echo "To run the Bayesian optimization:"
echo "cd $DEST_DIR"
echo "./multiCase.sh"
echo ""
echo "For post-processing after runs complete:"
echo "./create_scripts.sh"
echo "./run_d.sh"
