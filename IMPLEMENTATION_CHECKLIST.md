# Open System Implementation Checklist

## Phase 1: Clean Slate (Delete Wrong Physics)
- [ ] `rm -rf src/diffusion_pinn/optimization/`
- [ ] `rm -rf scripts/bayesian_cluster_scripts/`
- [ ] `rm -rf scripts/cluster_scripts/`
- [ ] Commit deletions: "Remove closed-system and optimization code"

## Phase 2: Core Physics (Models & Training)
- [ ] Replace `src/diffusion_pinn/models/pinn.py`
- [ ] Replace `src/diffusion_pinn/training/trainer.py`
- [ ] Update `src/diffusion_pinn/variables.py`
- [ ] Update `src/diffusion_pinn/__init__.py`
- [ ] Commit: "Implement OpenSystemDiffusionPINN with Robin boundaries"

## Phase 3: Cluster Integration
- [ ] Create `scripts/open_system_cluster_scripts/`
- [ ] Add new multiCase.sh
- [ ] Add new runCase.sh
- [ ] Add open_system_pinn_trainer.py
- [ ] Commit: "Add open system cluster scripts"

## Phase 4: Visualization & Analysis
- [ ] Update `src/diffusion_pinn/utils/visualization.py`
- [ ] Create new post-processing scripts
- [ ] Update directory generators
- [ ] Commit: "Add open system analysis tools"

## Phase 5: Testing
- [ ] Create small test case
- [ ] Run 1-2 short jobs (100 epochs)
- [ ] Verify two parameters learned
- [ ] Check mass conservation plots
- [ ] Commit: "Verify open system implementation works"

## Phase 6: Documentation
- [ ] Update main README.md
- [ ] Update requirements.txt if needed
- [ ] Add migration guide
- [ ] Commit: "Update documentation for open system"

