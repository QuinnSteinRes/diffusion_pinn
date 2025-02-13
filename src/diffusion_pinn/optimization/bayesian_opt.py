# Standard library imports
import os
from typing import Dict, List, Optional, Tuple

# Third-party imports
import tensorflow as tf
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

# Local imports
from ..models.pinn import DiffusionPINN
from ..config import DiffusionConfig
from ..variables import PINN_VARIABLES
from ..training.trainer import train_pinn
from .config import OPTIMIZATION_SETTINGS

class PINNBayesianOptimizer:
    """Bayesian optimization for PINN hyperparameters"""

    def __init__(
        self,
        data_processor: 'DiffusionDataProcessor',
        opt_config: Dict = None,
        save_dir: str = 'optimization_results'
    ):
        """Initialize optimizer with configuration"""
        self.data_processor = data_processor
        self.config = opt_config or OPTIMIZATION_SETTINGS
        self.save_dir = save_dir

        # Create output directories
        self.model_dir = os.path.join(save_dir, 'models')
        self.log_dir = os.path.join(save_dir, 'logs')
        for dir_path in [self.model_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Define optimization space
        self.dimensions = [
            Integer(
                low=self.config['layers_lowerBound'],
                high=self.config['layers_upperBound'],
                name='layers'
            ),
            Integer(
                low=self.config['neurons_lowerBound'],
                high=self.config['neurons_upperBound'],
                name='neurons'
            ),
            Categorical(
                categories=['relu', 'tanh', 'elu', 'selu'],
                name='activation'
            ),
            Real(
                low=self.config['learning_lowerBound'],
                high=self.config['learning_upperBound'],
                prior='log-uniform',
                name='learning_rate'
            )
        ]

        # Best model tracking
        self.best_loss = float('inf')
        self.best_model_path = os.path.join(self.model_dir, 'best_model.h5')
        self.best_config_path = os.path.join(self.model_dir, 'best_config.txt')

        # Optimization history
        self.history = {
            'losses': [],
            'diffusion_coeffs': [],
            'parameters': []
        }

        # Prepare training data
        print("Preparing training data...")
        self.training_data = self.data_processor.prepare_training_data(
            N_u=self.config.get('N_u', PINN_VARIABLES['N_u']),
            N_f=self.config.get('N_f', PINN_VARIABLES['N_f']),
            N_i=self.config.get('N_i', PINN_VARIABLES['N_i'])
        )

    def create_model(self, layers: int, neurons: int,
                    activation: str, learning_rate: float) -> Tuple[DiffusionPINN, tf.keras.optimizers.Optimizer]:
        """Create PINN model with given hyperparameters"""
        config = DiffusionConfig(
            hidden_layers=[neurons] * layers,
            activation=activation,
            initialization='glorot',
            diffusion_trainable=True,
            use_physics_loss=True
        )

        domain_info = self.data_processor.get_domain_info()

        pinn = DiffusionPINN(
            spatial_bounds=domain_info['spatial_bounds'],
            time_bounds=domain_info['time_bounds'],
            initial_D=PINN_VARIABLES['initial_D'],
            config=config
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        return pinn, optimizer

    def train_and_evaluate(self, pinn: DiffusionPINN,
                          optimizer: tf.keras.optimizers.Optimizer,
                          trial_dir: str) -> Tuple[float, float]:
        """Train model and compute validation loss"""
        try:
            # Setup callbacks
            tf_callback = tf.keras.callbacks.TensorBoard(
                log_dir=trial_dir,
                histogram_freq=0,
                write_graph=True
            )
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('earlyStop_patience', 5),
                mode='min',
                restore_best_weights=True
            )

            # Train model
            D_history, loss_history = train_pinn(
                pinn=pinn,
                data=self.training_data,
                optimizer=optimizer,
                epochs=self.config['network_epochs'],
                save_dir=trial_dir
            )

            final_loss = loss_history[-1]['total']
            final_D = D_history[-1]

            # Save histories
            np.save(os.path.join(trial_dir, 'loss_history.npy'), loss_history)
            np.save(os.path.join(trial_dir, 'D_history.npy'), D_history)

            return final_loss, final_D

        except Exception as e:
            print(f"Error in training: {str(e)}")
            return float('inf'), 0.0

    def objective_impl(self, layers: int, neurons: int,
                      activation: str, learning_rate: float) -> float:
        """Implementation of the objective function"""
        try:
            trial_name = f"trial_l{layers}_n{neurons}_a{activation}_lr{learning_rate:.2e}"
            trial_dir = os.path.join(self.log_dir, trial_name)
            os.makedirs(trial_dir, exist_ok=True)

            print(f"\nStarting trial: {trial_name}")
            pinn, optimizer = self.create_model(
                layers, neurons, activation, learning_rate
            )

            val_loss, final_D = self.train_and_evaluate(
                pinn, optimizer, trial_dir
            )

            # Update history
            self.history['losses'].append(val_loss)
            self.history['diffusion_coeffs'].append(final_D)
            self.history['parameters'].append({
                'layers': layers,
                'neurons': neurons,
                'activation': activation,
                'learning_rate': learning_rate
            })

            # Save if best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                pinn.save(self.best_model_path)

                with open(self.best_config_path, 'w') as f:
                    f.write(f"Best model configuration:\n")
                    f.write(f"Layers: {layers}\n")
                    f.write(f"Neurons per layer: {neurons}\n")
                    f.write(f"Activation function: {activation}\n")
                    f.write(f"Learning rate: {learning_rate}\n")
                    f.write(f"Validation loss: {val_loss}\n")
                    f.write(f"Final diffusion coefficient: {final_D}\n")

            return val_loss

        except Exception as e:
            print(f"Error in trial: {str(e)}")
            return float('inf')

    def optimize(self) -> Dict:
        """Run Bayesian optimization"""
        # Create the decorated objective function
        @use_named_args(dimensions=self.dimensions)
        def objective(**params):
            return self.objective_impl(**params)

        # Initial point
        x0 = [
            self.config['initial_layers'],
            self.config['initial_neurons'],
            self.config['initial_activation'],
            self.config['initial_learningRate']
        ]

        print("\nStarting Bayesian optimization")
        print(f"Total iterations: {self.config['iterations_optimizer']}")
        print(f"Initial point: {x0}")

        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_calls=self.config['iterations_optimizer'],
            n_random_starts=self.config['iterations_optimizer'] // 3,
            x0=x0,
            acq_func=self.config['acquisitionFunction'],
            random_state=42
        )

        # Format results
        best_params = dict(zip(
            [d.name for d in self.dimensions],
            result.x
        ))

        # Save optimization history
        self.save_history()

        return {
            'best_parameters': best_params,
            'best_value': result.fun,
            'all_values': result.func_vals,
            'n_iterations': len(result.func_vals),
            'diffusion_values': self.history['diffusion_coeffs'],
            'optimization_history': self.history
        }

    def load_best_model(self) -> Optional[DiffusionPINN]:
        """Load the best model found during optimization"""
        if os.path.exists(self.best_model_path):
            return tf.keras.models.load_model(self.best_model_path)
        return None

    def save_history(self, filename: str = 'optimization_history.npz'):
        """Save optimization history to file"""
        save_path = os.path.join(self.save_dir, filename)
        np.savez(
            save_path,
            losses=self.history['losses'],
            diffusion_coeffs=self.history['diffusion_coeffs'],
            parameters=self.history['parameters']
        )