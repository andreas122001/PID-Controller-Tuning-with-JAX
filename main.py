import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from controller import AbstractController, DefaultController, NeuralController
from plant import AbstractPlant, BathtubPlant
import matplotlib.pyplot as plt

h_params = {
    'plant_name': 'bathtub',
    'controller_name': 'default',
    'nn_hidden_layers': [2,2],
    'nn_activation': ['Tanh'],
    'nn_init_weight_range': [-10,10],
    'nn_init_bias_range': [-10,10],
    'epochs': 10,
    'runs_per_epoch': 10,
    'learning_rate': 0.01,
    'noise_range': [-0.05, 0.05],
    'bathtub_area': 100,
    'bathtub_drain_area': 1,
    'bathtub_init_height': 5,
    'cournot_max_price': 100,
    'cournot_margin_cost': 0.1,
    'param1': 0,
    'param2': 0,
    'enable_jit': True
}

class ConSys:
    def __init__(self, controller: AbstractController, plant: AbstractPlant) -> None:
        self.controller = controller
        self.plant = plant

    def run(self, epochs, lr=0.05, steps_per_epoch=10):
        x_mse = []

        params = self.controller.init_params()
        gradfunc = jax.value_and_grad(self.simulate)
        gradfunc = jax.jit(gradfunc, static_argnames=['steps'],)
        for _ in tqdm(range(epochs)):
            D = np.random.uniform(-0.1, 0.1, steps_per_epoch)#jnp.zeros(steps)
            target, state = self.plant.reset()
            
            mse, gradients = gradfunc(params, state, target, D, steps_per_epoch, )
            params = self.controller.update_params(params, lr, gradients)

            print("\n", mse, gradients)
            x_mse.append(mse)

        steps_per_epoch = 250
        x_state = []
        target, state = self.plant.reset()
        err = target - state
        err_hist = jnp.array([err])
        D = np.random.uniform(-0.05, 0.05, steps_per_epoch)
        for t in range(steps_per_epoch):
            x_state.append(state)
            U = self.controller.step(params, err, err_hist) # calc control signal
            state = self.plant.step(state, U, D[t]) # calc plant output
            err = target - state
            err_hist = jnp.append(err_hist, err)
        plt.plot(x_mse)
        plt.show()
        plt.plot(x_state)
        plt.show()
        print(params)

    def simulate(self, params, state, target, noise_vector, steps):
        """Main differentiable simulation function. Also jittable."""

        # Initialize error and history
        error = target - state
        err_hist = jnp.array([error])
        
        for t in range(steps):
            # Perform one step
            state, error, err_hist = self._step_once(
                params, state, error, err_hist, target, noise_vector[t]
            )
            
        return jnp.sum(jnp.array([e**2 for e in err_hist]))

    def _step_once(self, params, state, error, err_hist, target, noise):
        """Performs one step of PID simulation"""
        # Update plant and controller
        signal = controller.step(params, error, err_hist)
        new_state = plant.step(state, signal, noise)

        # Calculate error
        new_error = target - new_state
        err_hist = jnp.append(err_hist, new_error)
        return new_state, new_error, err_hist

if __name__=="__main__":
    plant = BathtubPlant(5.0)
    # controller = NeuralController(3,1,[],['linear'])
    controller = DefaultController()
    system = ConSys(controller, plant)
    system.run(epochs=100)
    # print(system.simulate([1,0,0],2000))

