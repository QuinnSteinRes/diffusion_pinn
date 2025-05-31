#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN Data Flow & Connection Analyzer
Traces how data flows through your PINN system from input CSV to final diffusion coefficient
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

class PINNDataFlowAnalyzer:
    """Analyzes data flow and connections in PINN project"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.file_contents = {}
        self.data_flow = {}
        self.connections = {}
        
    def analyze_complete_flow(self):
        """Analyze the complete data flow from input to output"""
        print("PINN DATA FLOW ANALYSIS")
        print("=" * 80)
        
        # Load all relevant files
        self._load_project_files()
        
        # Trace the complete data journey
        print("\nCOMPLETE DATA JOURNEY:")
        print("=" * 50)
        
        # 1. Input Data Analysis
        self._analyze_input_data()
        
        # 2. Data Processing Pipeline
        self._analyze_data_processing()
        
        # 3. Model Architecture & Flow
        self._analyze_model_flow()
        
        # 4. Training Process
        self._analyze_training_flow()
        
        # 5. Output Generation
        self._analyze_output_flow()
        
        # 6. File Connections Map
        self._analyze_file_connections()
        
        # 7. Critical Data Transformation Points
        self._analyze_transformation_points()
        
    def _load_project_files(self):
        """Load contents of all relevant files"""
        key_files = [
            "src/diffusion_pinn/data/processor.py",
            "src/diffusion_pinn/models/pinn.py", 
            "src/diffusion_pinn/training/trainer.py",
            "src/diffusion_pinn/config.py",
            "src/diffusion_pinn/variables.py",
            "src/diffusion_pinn/__init__.py",
            "scripts/cluster_scripts/defaultScripts/pinn_trainer.py",
            "scripts/bayesian_cluster_scripts/defaultScripts/optimize_minimal.py"
        ]
        
        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        self.file_contents[file_path] = f.read()
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    def _analyze_input_data(self):
        """Analyze how input data enters the system"""
        print("\nSTEP 1: INPUT DATA ENTRY")
        print("-" * 30)
        
        # Find CSV input references
        csv_references = []
        for file_path, content in self.file_contents.items():
            if 'intensity_time_series_spatial_temporal.csv' in content:
                csv_references.append(file_path)
        
        print("Input File: intensity_time_series_spatial_temporal.csv")
        print("   Structure: x, y, t, intensity columns")
        print("   Content: 5 time-series images of diffusion process")
        print(f"   Referenced in: {len(csv_references)} files")
        
        for file_path in csv_references:
            print(f"      \u2022 {file_path}")
        
        # Analyze data loading patterns
        loading_patterns = self._find_data_loading_patterns()
        if loading_patterns:
            print("\n   Data Loading Methods:")
            for pattern in loading_patterns:
                print(f"      \u2022 {pattern}")
    
    def _find_data_loading_patterns(self):
        """Find how data is loaded from CSV"""
        patterns = []
        
        for file_path, content in self.file_contents.items():
            # Look for pandas/numpy data loading
            if 'pd.read_csv' in content:
                patterns.append(f"{file_path}: pandas.read_csv()")
            if 'np.genfromtxt' in content:
                patterns.append(f"{file_path}: numpy.genfromtxt()")
            if 'np.loadtxt' in content:
                patterns.append(f"{file_path}: numpy.loadtxt()")
        
        return patterns
    
    def _analyze_data_processing(self):
        """Analyze data processing pipeline"""
        print("\nSTEP 2: DATA PROCESSING PIPELINE")
        print("-" * 30)
        
        # Focus on DiffusionDataProcessor
        processor_file = "src/diffusion_pinn/data/processor.py"
        if processor_file in self.file_contents:
            content = self.file_contents[processor_file]
            
            print("DiffusionDataProcessor Class:")
            
            # Extract key methods and their purposes
            methods = self._extract_class_methods(content, "DiffusionDataProcessor")
            
            # Show data processing flow
            processing_steps = [
                ("__init__", "Load CSV -> Extract x,y,t,intensity -> Create 3D solution array"),
                ("get_boundary_and_interior_points", "Separate boundary vs interior data points"),
                ("create_deterministic_collocation_points", "Generate physics-informed sampling points"),
                ("prepare_training_data", "Create training tensors for PINN"),
                ("get_domain_info", "Extract spatial/temporal bounds")
            ]
            
            print("\n   Processing Flow:")
            for i, (method, description) in enumerate(processing_steps, 1):
                if any(method in m['name'] for m in methods):
                    print(f"      {i}. {method}()")
                    print(f"         -> {description}")
                    
                    # Show data transformations
                    if method == "__init__":
                        print("         Data Shape Transformations:")
                        print("            CSV rows -> x_raw, y_raw, t arrays")
                        print("            -> Normalized x, y coordinates [0,1]")
                        print("            -> 3D array usol[nx, ny, nt]")
                        print("            -> Meshgrid X, Y, T")
                    
                    elif method == "prepare_training_data":
                        print("         Training Data Creation:")
                        print("            -> X_u_train: boundary condition points")
                        print("            -> X_i_train: interior supervision points") 
                        print("            -> X_f_train: physics collocation points")
                        print("            -> u_train, u_i_train: target values")
    
    def _analyze_model_flow(self):
        """Analyze model architecture and data flow"""
        print("\nSTEP 3: NEURAL NETWORK MODEL FLOW")
        print("-" * 30)
        
        pinn_file = "src/diffusion_pinn/models/pinn.py"
        if pinn_file in self.file_contents:
            content = self.file_contents[pinn_file]
            
            print("DiffusionPINN Class:")
            
            # Extract network architecture details
            if 'hidden_layers' in content:
                print("   Network Architecture:")
                print("      Input: [x, y, t] coordinates (3D)")
                print("      Hidden Layers: Configurable (default from variables.py)")
                print("      Output: concentration prediction (1D)")
                print("      Activation: tanh (optimal for PDEs)")
            
            # Show forward pass flow
            print("\n   Forward Pass Flow:")
            forward_steps = [
                "Input [x,y,t] -> _normalize_inputs() -> [-1,1] range",
                "Normalized input -> Neural network layers",
                "Hidden layers with tanh activation",
                "Final layer -> concentration prediction c(x,y,t)"
            ]
            
            for i, step in enumerate(forward_steps, 1):
                print(f"      {i}. {step}")
            
            # Show loss computation flow
            print("\n   Loss Computation Flow:")
            loss_steps = [
                "Data Loss: |c_pred - c_true|^2 on boundary/interior points",
                "Physics Loss: PDE residual dc/dt - D*nabla^2*c = 0",
                "Consistency Loss: Mass conservation, positivity constraints",
                "Regularization: Diffusion coefficient bounds",
                "Total Loss: Weighted combination of all terms"
            ]
            
            for i, step in enumerate(loss_steps, 1):
                print(f"      {i}. {step}")
    
    def _analyze_training_flow(self):
        """Analyze training process flow"""
        print("\nSTEP 4: TRAINING PROCESS FLOW")  
        print("-" * 30)
        
        trainer_file = "src/diffusion_pinn/training/trainer.py"
        if trainer_file in self.file_contents:
            content = self.file_contents[trainer_file]
            
            print("Training Pipeline:")
            
            # Show training phases
            if 'deterministic_train_pinn' in content:
                print("\n   Three-Phase Training Schedule:")
                phases = [
                    ("Phase 1: Physics Learning", "Focus on PDE constraints", "25% of epochs"),
                    ("Phase 2: Data Fitting", "Balance physics + data", "50% of epochs"), 
                    ("Phase 3: Fine Tuning", "Emphasize data accuracy", "25% of epochs")
                ]
                
                for i, (phase, description, duration) in enumerate(phases, 1):
                    print(f"      {i}. {phase} ({duration})")
                    print(f"         -> {description}")
            
            # Show gradient flow
            print("\n   Training Loop Flow:")
            training_steps = [
                "Forward pass: x -> NN -> c_pred",
                "Compute losses: data + physics + consistency",
                "Backward pass: compute gradients",
                "Gradient clipping for stability",
                "Update network weights + diffusion coefficient D",
                "Apply constraints: D in [1e-6, 1e-2]"
            ]
            
            for i, step in enumerate(training_steps, 1):
                print(f"      {i}. {step}")
    
    def _analyze_output_flow(self):
        """Analyze how outputs are generated"""
        print("\nSTEP 5: OUTPUT GENERATION")
        print("-" * 30)
        
        print("Final Outputs:")
        
        outputs = [
            ("Diffusion Coefficient D", "Primary target - learned parameter", "pinn.get_diffusion_coefficient()"),
            ("Loss History", "Training convergence tracking", "loss_history list"),
            ("D History", "D value evolution", "D_history list"),
            ("Model Predictions", "c(x,y,t) at any point", "pinn.predict(x)"),
            ("Visualization Plots", "Solution fields, convergence", "utils/visualization.py")
        ]
        
        for output, description, source in outputs:
            print(f"   {output}")
            print(f"      Purpose: {description}")
            print(f"      Source: {source}")
    
    def _analyze_file_connections(self):
        """Analyze how files are connected"""
        print("\nSTEP 6: FILE CONNECTION MAP")
        print("-" * 30)
        
        # Build import graph
        imports = {}
        for file_path, content in self.file_contents.items():
            imports[file_path] = self._extract_imports(content)
        
        # Show main execution flows
        print("Main Execution Flows:")
        
        flows = [
            {
                "name": "Standard Training Flow",
                "entry": "scripts/cluster_scripts/defaultScripts/pinn_trainer.py",
                "flow": [
                    "pinn_trainer.py (main script)",
                    "-> diffusion_pinn.__init__ (imports)",
                    "-> DiffusionDataProcessor (data loading)",
                    "-> DiffusionPINN (model creation)", 
                    "-> train_pinn (training logic)",
                    "-> visualization utilities (plotting)"
                ]
            },
            {
                "name": "Bayesian Optimization Flow", 
                "entry": "scripts/bayesian_cluster_scripts/defaultScripts/optimize_minimal.py",
                "flow": [
                    "optimize_minimal.py (main script)",
                    "-> PINNBayesianOptimizer (hyperparameter search)",
                    "-> DiffusionDataProcessor (data prep)",
                    "-> DiffusionPINN (model variants)",
                    "-> train_pinn (evaluation)",
                    "-> skopt.gp_minimize (optimization)"
                ]
            }
        ]
        
        for flow_info in flows:
            print(f"\n   {flow_info['name']}:")
            print(f"      Entry Point: {flow_info['entry']}")
            for step in flow_info['flow']:
                print(f"      {step}")
    
    def _analyze_transformation_points(self):
        """Analyze critical data transformation points"""
        print("\nSTEP 7: CRITICAL DATA TRANSFORMATION POINTS")
        print("-" * 30)
        
        print("Points Where Data Shape/Meaning Changes:")
        
        transformations = [
            {
                "location": "DiffusionDataProcessor.__init__",
                "input": "CSV file (N rows x 4 cols)",
                "output": "3D solution array usol[nx, ny, nt]",
                "critical": "Spatial coordinate normalization [0,1]"
            },
            {
                "location": "DiffusionDataProcessor.prepare_training_data", 
                "input": "3D solution array + domain bounds",
                "output": "TensorFlow tensors (X_u_train, u_train, X_f_train)",
                "critical": "Random sampling for training points"
            },
            {
                "location": "DiffusionPINN._normalize_inputs",
                "input": "Physical coordinates [x,y,t]", 
                "output": "Normalized coordinates [-1,1]",
                "critical": "Input scaling for neural network"
            },
            {
                "location": "DiffusionPINN.forward_pass",
                "input": "Normalized coordinates [-1,1]",
                "output": "Concentration predictions c(x,y,t)",
                "critical": "Neural network prediction"
            },
            {
                "location": "Training loop gradient computation",
                "input": "Loss values (scalar)",
                "output": "Parameter updates (weights + D)",
                "critical": "Diffusion coefficient learning"
            }
        ]
        
        for i, transform in enumerate(transformations, 1):
            print(f"\n   {i}. {transform['location']}")
            print(f"      Input: {transform['input']}")
            print(f"      Output: {transform['output']}")
            print(f"      Critical: {transform['critical']}")
        
        # Show data dimensions through the pipeline
        print(f"\nData Dimensions Through Pipeline:")
        dimensions = [
            ("CSV Input", "~149,500 rows (299x99x5 points)"),
            ("3D Array", "usol[299, 99, 5] = spatial x time"),
            ("Training Tensors", "X_u_train[1000,3], X_f_train[15000,3]"),
            ("Network Input", "Batch x 3 (normalized [x,y,t])"),
            ("Network Output", "Batch x 1 (concentration)"),
            ("Final D", "Single scalar (diffusion coefficient)")
        ]
        
        for stage, dimension in dimensions:
            print(f"   \u2022 {stage}: {dimension}")
    
    def _extract_class_methods(self, content: str, class_name: str):
        """Extract methods from a specific class"""
        try:
            tree = ast.parse(content)
            methods = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                'name': item.name,
                                'line': item.lineno,
                                'args': [arg.arg for arg in item.args.args]
                            })
            
            return methods
        except:
            return []
    
    def _extract_imports(self, content: str):
        """Extract import statements from file content"""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass
        return imports
    
    def generate_flow_diagram(self):
        """Generate ASCII flow diagram"""
        print("\nDATA FLOW DIAGRAM")
        print("-" * 30)
        
        diagram = """
INPUT DATA
intensity_time_series_spatial_temporal.csv
    |
    v
DATA PROCESSOR (DiffusionDataProcessor)
\u2022 Load CSV -> Extract x,y,t,intensity
\u2022 Normalize coordinates -> [0,1] range  
\u2022 Create 3D solution array usol[nx,ny,nt]
\u2022 Generate training points (boundary, interior, collocation)
    |
    v
NEURAL NETWORK (DiffusionPINN) 
\u2022 Input: [x,y,t] coordinates
\u2022 Architecture: [3] -> hidden_layers -> [1]
\u2022 Output: concentration c(x,y,t)
\u2022 Learnable parameter: diffusion coefficient D
    |
    v
TRAINING LOOP (train_pinn)
\u2022 Compute losses: data + physics + consistency
\u2022 Update network weights + diffusion coefficient
\u2022 Three-phase training schedule
    |
    v
OUTPUTS
\u2022 Final diffusion coefficient D
\u2022 Loss history (convergence tracking)
\u2022 Model predictions c(x,y,t)
\u2022 Visualization plots
"""
        print(diagram)
    
    def identify_reproducibility_risks(self):
        """Identify potential reproducibility issues in data flow"""
        print("\nREPRODUCIBILITY RISK POINTS")
        print("-" * 30)
        
        risks = [
            {
                "location": "DiffusionDataProcessor.prepare_training_data",
                "risk": "Random sampling of training points",
                "impact": "Different training sets -> different convergence",
                "solution": "Use deterministic sampling with fixed seed"
            },
            {
                "location": "DiffusionPINN._build_network", 
                "risk": "Random weight initialization",
                "impact": "Different starting points -> different solutions",
                "solution": "Set tf.random.set_seed before initialization"
            },
            {
                "location": "Training gradient computation",
                "risk": "Floating point operations order",
                "impact": "Slight numerical differences accumulate",
                "solution": "Use deterministic operations where possible"
            },
            {
                "location": "Bayesian optimization (gp_minimize)",
                "risk": "Random hyperparameter search",
                "impact": "Different parameter combinations tested",
                "solution": "Set random_state parameter"
            }
        ]
        
        print("High-Risk Points for Non-Determinism:")
        for i, risk in enumerate(risks, 1):
            print(f"\n   {i}. {risk['location']}")
            print(f"      Risk: {risk['risk']}")
            print(f"      Impact: {risk['impact']}")
            print(f"      Solution: {risk['solution']}")

def main():
    """Run the complete PINN data flow analysis"""
    analyzer = PINNDataFlowAnalyzer()
    
    # Run complete analysis
    analyzer.analyze_complete_flow()
    
    # Generate flow diagram
    analyzer.generate_flow_diagram()
    
    # Identify reproducibility risks
    analyzer.identify_reproducibility_risks()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR REPRODUCIBILITY:")
    print("=" * 80)
    
    next_steps = [
        "1. Check seed handling in DiffusionDataProcessor.prepare_training_data",
        "2. Verify DiffusionPINN weight initialization uses consistent seeds",
        "3. Ensure training loop uses deterministic operations", 
        "4. Add random_state to Bayesian optimization calls",
        "5. Test with identical seeds across parallel runs"
    ]
    
    for step in next_steps:
        print(f"   {step}")

if __name__ == "__main__":
    main()
