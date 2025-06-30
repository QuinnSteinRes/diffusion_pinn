#!/bin/bash
# create_open_system_case.sh
# Creates test cases for OPEN SYSTEM PINN with boundary flux

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [case_name] [options]"
    echo "Creates a new directory for OPEN SYSTEM PINN testing"
    echo ""
    echo "Arguments:"
    echo "  case_name          Name of the case directory (default: YYYYMMDD_OpenSystem)"
    echo ""
    echo "Options:"
    echo "  -s, --source DIR   Source directory for scripts (default: ~/projects/diffusion_pinn/scripts)"
    echo "  -d, --dest DIR     Parent directory for the new case (default: ~/projects/pinnRuns)"
    echo "  -r, --runs N       Number of runs to prepare (default: 5)"
    echo "  -e, --epochs N     Number of epochs per run (default: 15000)"
    echo "  -h, --help         Display this help message and exit"
    echo ""
    echo "IMPORTANT: This creates OPEN SYSTEM cases that learn both D and k parameters"
    echo "Do NOT use this for closed system problems!"
    exit 0
fi

# Default values
SOURCE_DIR="$HOME/projects/diffusion_pinn/scripts"
POST_PROCESS_DIR="$HOME/projects/diffusion_pinn/scripts/post_processing"
DEST_PARENT="$HOME/projects/pinnRuns"
CURRENT_DATE=$(date +"%Y%m%d")
CASE_NAME="${CURRENT_DATE}_OpenSystem"
NUM_RUNS=5
EPOCHS=15000

# Process command line arguments
if [ -n "$1" ] && [[ ! "$1" == -* ]]; then
    CASE_NAME="$1"
    shift
fi

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
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use open system scripts
SCRIPT_SRC="$SOURCE_DIR/open_system_scripts"

# Check if source directory exists
if [ ! -d "$SCRIPT_SRC" ]; then
    echo "Error: Open system script directory $SCRIPT_SRC does not exist!"
    echo "Expected structure:"
    echo "  $SOURCE_DIR/open_system_scripts/defaultScripts/"
    echo "  $SOURCE_DIR/open_system_scripts/multiCase.sh"
    echo ""
    echo "You need to create this directory with the new open system scripts."
    exit 1
fi

# Set up destination
DEST_DIR="$DEST_PARENT/$CASE_NAME"

if [ -d "$DEST_DIR" ]; then
    read -p "Directory $DEST_DIR already exists. Overwrite? (y/N): " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
fi

mkdir -p "$DEST_DIR"

echo "Creating OPEN SYSTEM PINN test case in $DEST_DIR"
echo "Model: OpenSystemDiffusionPINN with Robin boundary conditions"
echo "Parameters: D (diffusion) + k (boundary permeability)"
echo "Number of runs: $NUM_RUNS"
echo "Epochs per run: $EPOCHS"

# Copy open system scripts
echo "Copying open system scripts..."
cp -r "$SCRIPT_SRC/defaultScripts" "$DEST_DIR/"
cp "$SCRIPT_SRC/multiCase.sh" "$DEST_DIR/"

# Add post-processing for open system
echo "Adding open system post-processing..."
if [ -f "$POST_PROCESS_DIR/create_scripts.sh" ]; then
    # Use the new open system post-processing
    cp "$POST_PROCESS_DIR/create_scripts.sh" "$DEST_DIR/create_open_system_scripts.sh"
    chmod +x "$DEST_DIR/create_open_system_scripts.sh"
    echo "Open system post-processing script added."
else
    echo "Warning: Post-processing script not found at $POST_PROCESS_DIR"
fi

# Make scripts executable
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
    echo "You may need to manually copy the data file."
fi

# Configure for multiple runs with different seeds
if [ "$NUM_RUNS" -ne 1 ]; then
    echo "Configuring for $NUM_RUNS runs with different seeds..."
    sed -i "s/NUM_RUNS=\${1:-1}/NUM_RUNS=\${1:-$NUM_RUNS}/g" "$DEST_DIR/multiCase.sh"
fi

# Update epoch count in scripts
echo "Setting epochs to $EPOCHS..."
if [ -f "$DEST_DIR/defaultScripts/pinn_trainer.py" ]; then
    sed -i "s/default=PINN_VARIABLES\['epochs'\]/default=$EPOCHS/g" "$DEST_DIR/defaultScripts/pinn_trainer.py"
fi

# Create a README for the case
cat > "$DEST_DIR/README.md" << EOF
# Open System PINN Case: $CASE_NAME

This directory contains scripts for training an **Open System Physics-Informed Neural Network** for diffusion problems with boundary flux.

## Model Type: OpenSystemDiffusionPINN

### Physics:
- **Interior**: ∂c/∂t = D∇²c (standard diffusion)
- **Boundaries**: -D(∂c/∂n) = k(c - c_ext) (Robin boundary condition)
- **Initial**: c(x,y,0) = data (fitted to first image)

### Parameters Learned:
- **D**: Diffusion coefficient
- **k**: Boundary permeability (how fast mass leaves through boundaries)

### Key Differences from Closed System:
- Accounts for mass loss through boundaries
- Learns boundary transport properties
- Physically meaningful parameters
- Proper mass conservation

## Configuration

- **Runs**: $NUM_RUNS (with different random seeds)
- **Epochs per run**: $EPOCHS
- **Model**: OpenSystemDiffusionPINN
- **Generated**: $(date)

## Running the Analysis

1. **Train the models**:
   \`\`\`bash
   ./multiCase.sh
   \`\`\`

2. **Post-process results** (after training completes):
   \`\`\`bash
   ./create_open_system_scripts.sh
   ./run_open_system_analysis.sh
   \`\`\`

## Expected Outputs

Each run will create:
- Diffusion coefficient (D) evolution
- Boundary permeability (k) evolution
- Mass conservation analysis
- Boundary flux analysis
- Physical parameter interpretation

## Validation Checklist

- [ ] Mass conservation plots match data trend
- [ ] D values are physically reasonable (10^-8 to 10^-3)
- [ ] k values make sense for boundary properties
- [ ] System type (outflow vs diffusion dominated) is reasonable
- [ ] Parameters converge consistently across seeds

## Important Notes

- This is an **OPEN SYSTEM** model - do not use for closed systems
- Mass loss through boundaries is physically modeled, not ignored
- Both D and k have physical meaning and units
- Results are not comparable to closed system PINN results
EOF

echo ""
echo "Successfully created OPEN SYSTEM PINN case: $DEST_DIR"
echo ""
echo "Directory structure:"
ls -la "$DEST_DIR"
echo ""
echo "To run the open system analysis:"
echo "cd $DEST_DIR"
echo "./multiCase.sh"
echo ""
echo "For post-processing after training:"
echo "./create_open_system_scripts.sh"
echo "./run_open_system_analysis.sh"
echo ""
echo "CRITICAL: This uses OpenSystemDiffusionPINN - completely different from your old model!"
echo "Results will NOT be comparable to any previous closed-system analysis."