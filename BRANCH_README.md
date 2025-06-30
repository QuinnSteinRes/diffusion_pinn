# Open System Physics Branch

## Purpose
Convert from closed-system PINN (physically incorrect) to open-system PINN with Robin boundary conditions (physically correct).

## Key Changes
- Replace `DiffusionPINN` with `OpenSystemDiffusionPINN`
- Add boundary flux physics: -D(∂c/∂n) = k(c - c_ext)
- Learn TWO parameters: D (diffusion) + k (boundary permeability)
- Proper mass conservation validation (not assumption)

## Files Being Replaced
- `src/diffusion_pinn/models/pinn.py` - Complete rewrite for Robin boundaries
- `src/diffusion_pinn/training/trainer.py` - Open system training
- `scripts/cluster_scripts/` - DELETE entire directory
- `scripts/bayesian_cluster_scripts/` - DELETE entire directory (optimizing wrong physics)
- `src/diffusion_pinn/optimization/` - DELETE entire directory

## Files Being Added
- `scripts/open_system_cluster_scripts/` - New cluster scripts
- Enhanced visualization for two-parameter analysis
- Mass conservation validation tools

## Status
- [ ] Delete wrong physics files
- [ ] Implement OpenSystemDiffusionPINN
- [ ] Update training pipeline
- [ ] Create new cluster scripts
- [ ] Update post-processing
- [ ] Test on small dataset
- [ ] Validate against known results

## Testing Plan
1. Small test case (100 epochs) to verify no crashes
2. Check that two parameters (D, k) are learned
3. Verify mass conservation analysis works
4. Compare results across different seeds
5. Physical sanity check on parameter values

## Validation Criteria
- Mass balance matches data trend (not artificially conserved)
- D values in physically reasonable range (10^-8 to 10^-3)
- k values make sense for boundary transport
- Parameters converge consistently across seeds
- System type (outflow vs diffusion dominated) is reasonable

