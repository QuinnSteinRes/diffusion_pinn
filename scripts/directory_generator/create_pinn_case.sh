#!/bin/bash
# Updated create_pinn_case.sh - Compatible with new PINN labeled data interface
# Script to create a new PINN test case directory with updated scripts
# Author: Quinn Stein

# Display help message if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [case_name] [options]"
    echo "Creates a new directory with updated scripts for PINN testing"
    echo ""
    echo "Arguments:"
    echo "  case_name          Name of the case directory (default: YYYYMMDD_Test)"
    echo ""
    echo "Options:"
    echo "  -s, --source DIR   Source directory for scripts (default: ~/projects/diffusion_pinn/scripts)"
    echo "  -b, --bayes        Use Bayesian optimization scripts instead of standard PINN"
    echo "  -d, --dest DIR     Parent directory for the new case (default: ~/projects/pinnRuns)"
    echo "  -r, --runs N       Number of runs to prepare (default: 1)"
    echo "  -h, --help         Display this help message and exit"
    echo ""
    echo "New interface features:"
    echo "  - Labeled data structure (boundary, interior, physics points)"
    echo "  - Updated parameter names and interfaces"
    echo "  - Improved memory management and error handling"
    echo "  - Enhanced convergence analysis"
    exit 0
fi

# Default values
SOURCE_DIR="$HOME/projects/diffusion_pinn/scripts"
POST_PROCESS_DIR="$HOME/projects/diffusion_pinn/scripts/post_processing"
DEST_PARENT="$HOME/projects/pinnRuns"
CURRENT_DATE=$(date +"%Y%m%d")
CASE_NAME="${CURRENT_DATE}_Test"
USE_BAYESIAN=false
NUM_RUNS=1

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
        -b|--bayes)
            USE_BAYESIAN=true
            shift
            ;;
        -r|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine which script directories to use based on bayes flag
if [ "$USE_BAYESIAN" = true ]; then
    SCRIPT_SRC="$SOURCE_DIR/bayesian_cluster_scripts"
    echo "Using Bayesian optimization scripts"
else
    SCRIPT_SRC="$SOURCE_DIR/cluster_scripts"
    echo "Using standard PINN scripts (updated for new interface)"
fi

# Check if source directory exists
if [ ! -d "$SCRIPT_SRC" ]; then
    echo "Error: Source directory $SCRIPT_SRC does not exist!"
    echo "Make sure the diffusion_pinn project has the scripts directory structure:"
    echo "  $SOURCE_DIR/cluster_scripts"
    echo "  $SOURCE_DIR/bayesian_cluster_scripts"
    exit 1
fi

# Check if source contains required files
if [ ! -d "$SCRIPT_SRC/defaultScripts" ] || [ ! -f "$SCRIPT_SRC/multiCase.sh" ]; then
    echo "Error: Source directory $SCRIPT_SRC does not contain required files!"
    echo "Expected: defaultScripts/ directory and multiCase.sh file"
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
echo "Creating new PINN test case in $DEST_DIR"
echo "Copying updated scripts for new interface..."
cp -r "$SCRIPT_SRC/defaultScripts" "$DEST_DIR/"
cp "$SCRIPT_SRC/multiCase.sh" "$DEST_DIR/"

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

# Modify multiCase.sh to use the specified number of runs
if [ "$NUM_RUNS" -ne 1 ]; then
    echo "Configuring for $NUM_RUNS runs..."
    sed -i "s/NUM_RUNS=\${1:-10}/NUM_RUNS=\${1:-$NUM_RUNS}/g" "$DEST_DIR/multiCase.sh"
fi

# Create a README with new interface information
cat > "$DEST_DIR/README.md" << 'EOF'
# PINN Test Case - Updated Interface

This directory contains updated PINN scripts with the new labeled data interface.

## Key Updates

### New Interface Features:
- **Labeled Data Structure**: Clear separation of boundary, interior, and physics points
- **Updated Parameter Names**:
  - `N_u` → `N_boundary` (boundary/initial condition points)
  - `N_i` → `N_interior` (interior supervision points)
  - `N_f` → `N_collocation` (physics collocation points)
- **Improved Memory Management**: Better handling of large datasets
- **Enhanced Error Handling**: More robust training with better diagnostics
- **Cleaner Loss Computation**: No more point type guessing

### Training Process:
1. Data processor creates labeled point sets
2. PINN trains on clearly defined point types
3. Enhanced convergence checking and diagnostics
4. Improved visualization and analysis

## Usage

### Run Single Test:
```bash
cd defaultScripts
qsub runCase.sh
```

### Run Multiple Seeds:
```bash
./multiCase.sh [num_runs] [seed_mode] [base_seed]
```

### Post-Processing:
```bash
./create_scripts.sh  # Create analysis scripts
./run_d.sh          # Run full analysis
```

## File Structure

- `defaultScripts/`: Updated training scripts
  - `pinn_trainer.py`: Main trainer with new interface
  - `runCase.sh`: Cluster submission script
  - `intensity_time_series_spatial_temporal.csv`: Training data
- `multiCase.sh`: Multi-run submission script
- `create_scripts.sh`: Post-processing script generator

## New Interface Benefits

1. **Clearer Code**: No ambiguity about point types
2. **Better Performance**: Optimized memory usage
3. **Easier Debugging**: Clear separation of loss components
4. **More Robust**: Better error handling and recovery
5. **Enhanced Analysis**: Improved convergence checking

EOF

echo "Successfully created PINN test case directory: $DEST_DIR"
echo ""
echo "Directory structure:"
ls -la "$DEST_DIR"
echo ""
echo "New interface features:"
echo "✓ Labeled data structure (boundary, interior, physics)"
echo "✓ Updated parameter names and interfaces"
echo "✓ Improved memory management"
echo "✓ Enhanced error handling and diagnostics"
echo "✓ Better convergence analysis"
echo ""
echo "To run the test case:"
echo "cd $DEST_DIR"
echo "./multiCase.sh"
echo ""
echo "For post-processing after runs complete:"
echo "./create_scripts.sh"
echo "./run_d.sh"
echo ""
echo "See README.md for detailed information about the new interface."