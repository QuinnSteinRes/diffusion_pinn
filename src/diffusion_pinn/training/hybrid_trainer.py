#!/usr/bin/env python3
"""
hybrid_trainer.py - Implementation of hybrid training strategies
Add this to your diffusion_pinn/training/ directory
"""

import tensorflow as tf
import numpy as np
import gc
from typing import Dict, List, Tuple
from ..variables import PINN_VARIABLES

def hybrid_train_pinn_v1(pinn, data, optimizer, epochs=10000, save_dir=None, seed=None):
    """
    Hybrid v1: v0.2.19 3-phase training + v0.2.26 log(D) bounds
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Modify bounds to v0.2.26 style
    original_bounds = (pinn.log_D_min, pinn.log_D_max)
    pinn.log_D_min = -16.0
    pinn.log_D_max = -4.0  # Tighter upper bound like v0.2.26

    print(f"Using v0.2.26 bounds: [{pinn.log_D_min}, {pinn.log_D_max}]")

    # Use v0.2.19 3-phase approach
    phase_configs = [
        {
            'name': 'Phase 1: Physics Learning',
            'epochs': epochs // 3,
            'weights': {'initial': 2.0, 'boundary': 2.0, 'interior': 0.5, 'physics': 10.0},
            'lr_schedule': lambda epoch, total: 1e-3 * (0.95 ** (epoch // 100)),
            'regularization': 0.001
        },
        {
            'name': 'Phase 2: Balanced Learning',
            'epochs': epochs // 3,
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 5.0, 'physics': 3.0},
            'lr_schedule': lambda epoch, total: 5e-4 * (0.98 ** (epoch // 50)),
            'regularization': 0.0005
        },
        {
            'name': 'Phase 3: Data Fitting',
            'epochs': epochs // 3,
            'weights': {'initial': 0.5, 'boundary': 0.5, 'interior': 10.0, 'physics': 1.0},
            'lr_schedule': lambda epoch, total: 1e-4 * (0.99 ** (epoch // 25)),
            'regularization': 0.0001
        }
    ]

    D_history, loss_history = run_phased_training(
        pinn, data, optimizer, phase_configs, "Hybrid_v1"
    )

    # Restore original bounds
    pinn.log_D_min, pinn.log_D_max = original_bounds
    return D_history, loss_history

def hybrid_train_pinn_v2(pinn, data, optimizer, epochs=10000, save_dir=None, seed=None):
    """
    Hybrid v2: Pure v0.2.19 approach (3-phase + wider bounds)
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Use v0.2.19 bounds
    original_bounds = (pinn.log_D_min, pinn.log_D_max)
    pinn.log_D_min = -16.0
    pinn.log_D_max = -6.0  # Wider upper bound like v0.2.19

    print(f"Using v0.2.19 bounds: [{pinn.log_D_min}, {pinn.log_D_max}]")

    # Exact v0.2.19 phase progression
    phase_configs = [
        {
            'name': 'Phase 1: Physics Learning (v0.2.19)',
            'epochs': epochs // 4,  # v0.2.19 uses //4 for first phase
            'weights': {'initial': 2.0, 'boundary': 2.0, 'interior': 0.5, 'physics': 10.0},
            'lr_schedule': lambda epoch, total: 1e-3 * (0.95 ** (epoch // 100)),
            'regularization': 0.001
        },
        {
            'name': 'Phase 2: Data Fitting (v0.2.19)',
            'epochs': epochs // 2,   # v0.2.19 uses //2 for middle phase
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 5.0, 'physics': 3.0},
            'lr_schedule': lambda epoch, total: 5e-4 * (0.98 ** (epoch // 50)),
            'regularization': 0.0005
        },
        {
            'name': 'Phase 3: Fine Tuning (v0.2.19)',
            'epochs': epochs // 4,   # v0.2.19 uses //4 for final phase
            'weights': {'initial': 0.5, 'boundary': 0.5, 'interior': 10.0, 'physics': 1.0},
            'lr_schedule': lambda epoch, total: 1e-4 * (0.99 ** (epoch // 25)),
            'regularization': 0.0001
        }
    ]

    D_history, loss_history = run_phased_training(
        pinn, data, optimizer, phase_configs, "Hybrid_v2_Pure_v019"
    )

    # Restore original bounds
    pinn.log_D_min, pinn.log_D_max = original_bounds
    return D_history, loss_history

def v026_with_enhanced_interior(pinn, data, optimizer, epochs=10000, save_dir=None, seed=None):
    """
    v0.2.26 approach but with progressive interior weighting
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    print("Using v0.2.26 training with progressive interior focus")

    # Progressive interior weighting (v0.2.26 style but with progression)
    phase_configs = [
        {
            'name': 'Phase 1: Physics + Light Interior',
            'epochs': epochs // 2,
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 5.0, 'physics': 8.0},
            'lr_schedule': lambda epoch, total: 1e-3 * (0.95 ** (epoch // 100)),
            'regularization': 0.001
        },
        {
            'name': 'Phase 2: Strong Interior (v0.2.26)',
            'epochs': epochs // 2,
            'weights': {'initial': 1.0, 'boundary': 1.0, 'interior': 10.0, 'physics': 8.0},
            'lr_schedule': lambda epoch, total: 5e-4 * (0.98 ** (epoch // 50)),
            'regularization': 0.0005
        }
    ]

    D_history, loss_history = run_phased_training(
        pinn, data, optimizer, phase_configs, "v026_Enhanced"
    )

    return D_history, loss_history

def run_phased_training(pinn, data, optimizer, phase_configs, experiment_name):
    """
    Common phased training implementation
    """
    D_history = []
    loss_history = []
    epoch_counter = 0

    print(f"\n{'='*70}")
    print(f"STARTING {experiment_name}")
    print(f"{'='*70}")

    # Log initial state
    initial_D = pinn.get_diffusion_coefficient()
    initial_log_D = pinn.get_log_diffusion_coefficient()
    print(f"Initial D: {initial_D:.8e}, log(D): {initial_log_D:.6f}")

    try:
        for phase_idx, phase in enumerate(phase_configs):
            print(f"\n{'-'*50}")
            print(f"{phase['name']} - {phase['epochs']} epochs")
            print(f"Weights: {phase['weights']}")
            print(f"{'-'*50}")

            phase_start_epoch = epoch_counter

            for epoch in range(phase['epochs']):
                # Update learning rate
                current_lr = phase['lr_schedule'](epoch, phase['epochs'])
                if hasattr(optimizer, 'learning_rate'):
                    if hasattr(optimizer.learning_rate, 'assign'):
                        optimizer.learning_rate.assign(current_lr)

                # Training step
                with tf.GradientTape() as tape:
                    # Main losses
                    losses = pinn.loss_fn(
                        x_data=data['X_u_train'],
                        c_data=data['u_train'],
                        x_physics=data['X_f_train'],
                        weights=phase['weights']
                    )

                    # Interior loss with v0.2.19 Huber approach
                    interior_loss = compute_v019_interior_loss(pinn, data)

                    # L2 regularization
                    l2_loss = phase['regularization'] * sum(
                        tf.reduce_sum(tf.square(w)) for w in pinn.weights
                    )

                    # CRITICAL: Use v0.2.19 unbiased log(D) regularization
                    log_d_reg = compute_v019_log_d_reg(pinn)

                    # Total loss
                    total_loss = (
                        losses['total'] +
                        phase['weights']['interior'] * interior_loss +
                        l2_loss +
                        log_d_reg
                    )

                # Apply gradients
                trainable_vars = pinn.get_trainable_variables()
                gradients = tape.gradient(total_loss, trainable_vars)
                gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                # Record history
                current_D = pinn.get_diffusion_coefficient()
                current_log_D = pinn.get_log_diffusion_coefficient()

                D_history.append(current_D)
                loss_history.append({
                    'total': float(total_loss.numpy()),
                    'physics': float(losses['physics'].numpy()),
                    'interior': float(interior_loss.numpy()),
                    'initial': float(losses.get('initial', 0.0).numpy()),
                    'boundary': float(losses.get('boundary', 0.0).numpy()),
                    'l2_reg': float(l2_loss.numpy()),
                    'log_d_reg': float(log_d_reg.numpy())
                })

                # Progress reporting
                if epoch_counter % 200 == 0:
                    print(f"Epoch {epoch_counter:5d}: D={current_D:.6e}, log(D)={current_log_D:.6f}, "
                          f"Total={total_loss:.4f}, Physics={losses['physics']:.4f}, "
                          f"Interior={interior_loss:.4f}")

                epoch_counter += 1

                # Memory cleanup
                if epoch_counter % 100 == 0:
                    gc.collect()

            # Phase summary
            phase_D = D_history[-1]
            phase_log_D = pinn.get_log_diffusion_coefficient()
            phase_loss = loss_history[-1]['total']

            print(f"\nPhase {phase_idx + 1} Summary:")
            print(f"  D: {phase_D:.6e}")
            print(f"  log(D): {phase_log_D:.6f}")
            print(f"  Loss: {phase_loss:.6f}")
            print(f"  Epochs: {phase_start_epoch} to {epoch_counter-1}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

    # Final analysis
    final_D = D_history[-1] if D_history else 0
    final_log_D = pinn.get_log_diffusion_coefficient() if hasattr(pinn, 'get_log_diffusion_coefficient') else 0

    print(f"\n{'='*70}")
    print(f"{experiment_name} COMPLETED")
    print(f"{'='*70}")
    print(f"Final D: {final_D:.8e}")
    print(f"Final log(D): {final_log_D:.6f}")

    # Test prediction quality
    analyze_final_performance(pinn, D_history, experiment_name)

    return D_history, loss_history

def compute_v019_interior_loss(pinn, data):
    """Compute interior loss using v0.2.19 Huber approach"""
    if data['X_i_train'].shape[0] == 0:
        return tf.constant(0.0, dtype=tf.float32)

    c_pred = pinn.forward_pass(data['X_i_train'])
    c_true = data['u_i_train']

    # v0.2.19 Huber loss
    raw_errors = c_pred - c_true
    delta = 0.1
    abs_errors = tf.abs(raw_errors)
    quadratic = tf.minimum(abs_errors, delta)
    linear = abs_errors - quadratic

    return tf.reduce_mean(0.5 * quadratic * quadratic + delta * linear)

def compute_v019_log_d_reg(pinn):
    """Compute v0.2.19 style unbiased log(D) regularization"""
    # UNBIASED: Only prevent extreme numerical overflow
    return 0.00001 * (
        tf.nn.relu(pinn.log_D_min + 3.0 - pinn.log_D) +
        tf.nn.relu(pinn.log_D - pinn.log_D_max + 3.0)
    )

def analyze_final_performance(pinn, D_history, experiment_name):
    """Analyze both D convergence and field prediction quality"""

    # Test field prediction
    test_points = tf.constant([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 1.0],
        [0.5, 0.5, 2.5],
        [0.75, 0.75, 4.0],
        [1.0, 1.0, 5.0]
    ], dtype=tf.float32)

    test_preds = pinn.predict(test_points)
    pred_range = tf.reduce_max(test_preds) - tf.reduce_min(test_preds)
    pred_std = tf.math.reduce_std(test_preds)

    print(f"\nField Prediction Analysis:")
    print(f"  Prediction range: {pred_range.numpy():.6f}")
    print(f"  Prediction std: {pred_std.numpy():.6f}")
    print(f"  Field quality: {'Good' if pred_range > 1e-3 else 'Poor (uniform)'}")

    # Test D convergence
    if len(D_history) >= 100:
        recent_d = D_history[-100:]
        d_std = np.std(recent_d)
        d_mean = np.mean(recent_d)
        convergence_metric = d_std / d_mean if d_mean > 0 else float('inf')

        print(f"\nD Convergence Analysis:")
        print(f"  Final D: {D_history[-1]:.6e}")
        print(f"  D std (last 100): {d_std:.6e}")
        print(f"  D convergence metric: {convergence_metric:.6f}")
        print(f"  D converged: {'Yes' if convergence_metric < 0.01 else 'No'}")

    # Overall assessment
    field_good = pred_range > 1e-3
    d_converged = len(D_history) >= 100 and (np.std(D_history[-100:]) / np.mean(D_history[-100:]) < 0.01)

    print(f"\n{experiment_name} ASSESSMENT:")
    print(f"  Field Prediction: {'GOOD' if field_good else 'POOR'}")
    print(f"  D Convergence: {'GOOD' if d_converged else 'POOR'}")

    if field_good and d_converged:
        print(f"   SUCCESS: Both objectives achieved!")
    elif field_good:
        print(f"    PARTIAL: Good fields, poor D convergence")
    elif d_converged:
        print(f"   PARTIAL: Good D convergence, poor fields")
    else:
        print(f"   FAILURE: Neither objective achieved")

# Export the training functions
__all__ = [
    'hybrid_train_pinn_v1',
    'hybrid_train_pinn_v2',
    'v026_with_enhanced_interior',
    'run_phased_training'
]