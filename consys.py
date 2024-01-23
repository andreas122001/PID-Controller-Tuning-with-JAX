import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from controller import AbstractController, DefaultController, NeuralController
from plant import AbstractPlant, BathtubPlant, CournotPlant, RobotArmPlant
import matplotlib.pyplot as plt
from config import config

class ConSys:
    def __init__(self, controller: AbstractController, 
                        plant: AbstractPlant, debug=False) -> None:
        self.controller = controller
        self.plant = plant
        if not debug:
            self._step_once = jax.jit(self._step_once) # should be jitted anyway, why not

    def train(self, epochs, 
                learning_rate=0.05, 
                steps_per_epoch=10, 
                noise_range=[-.1,.1], 
                enable_jit=False,
                enable_param_logging=False):
        
        # Noise function
        gen_noise = lambda n: np.random.uniform(*noise_range, n)

        params = self.controller.init_params()
        gradfunc = jax.value_and_grad(self._simulate_one_epoch)

        if enable_jit:
            print("Compiling simulation...")
            gradfunc = jax.jit(gradfunc, static_argnames=['steps'])

        mse_log = [0]*(epochs)
        param_log = [0]*(epochs) if enable_param_logging else [0]
        for i in tqdm(range(epochs)):
            
            # Initialize state, error and noise vector
            D = gen_noise(steps_per_epoch)
            state, error = self.plant.reset()
            
            # Compute gradients and error, and update parameters
            mse, gradients = gradfunc(params, state, error, D, steps_per_epoch)
            params = self.controller.update_params(params, learning_rate, gradients)

            print(f"\nError: {mse},\nParameters: {params},\nGradients: {gradients}")

            # Log current loss (and parameters)
            mse_log[i] = mse
            if enable_param_logging:
                param_log[i] = params

        return params, mse_log, param_log


    def _simulate_one_epoch(self, params, state, error, noise_vector, steps):
        """Main differentiable simulation function. Also jittable."""

        # Initialize error and history
        err_hist = jnp.array([error, error])
        
        # Simulate PID and accumulate error
        for t in range(steps):
            state, error, err_hist = self._step_once(
                params, state, error, err_hist, noise_vector[t]
            )
        
        # Compute MSE from error history
        return jnp.mean(jnp.array([e**2 for e in err_hist]))

    def _step_once(self, params, state, error, err_hist, noise):
        """Performs one step of PID simulation and updates state and error."""

        # Update controller with current error and error history
        signal = self.controller.step(params, error, err_hist)
        # Update plant with signal from controller
        new_state, new_error = self.plant.step(state, signal, noise)

        # Add error to history and return
        err_hist = jnp.append(err_hist, new_error)
        return new_state, new_error, err_hist

    def test(self, params, steps,
                noise_range=[-.1,.1]):
        """Test a system using tuned parameters. Returns accumulated simulation state and error."""

        # Initialize variables
        state, error = self.plant.reset()
        err_hist = jnp.array([error, error])
        D = np.random.uniform(*noise_range, steps)

        state_log = [0]*steps

        for t in tqdm(range(steps)):
            state, error, err_hist = self._step_once(
                params, state, error, err_hist, D[t])
            state_log[t] = state[0]

        return state_log, err_hist