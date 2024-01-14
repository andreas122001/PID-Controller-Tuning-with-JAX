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
}

class ConSys:
    def __init__(self, controller: AbstractController, plant: AbstractPlant) -> None:
        self.controller = controller
        self.plant = plant
        self.timesteps = 5

    def run(self, epochs):
        x_mse = []
        steps = 50

        D = np.random.rand(steps) * 0.001 * 2 - 0.001#jnp.zeros(steps)
        state = self.plant.reset()
        target = self.plant.target

        lr = 0.09
        params = self.controller.init_params()
        gradfunc = jax.value_and_grad(self.simulate)
        for _ in tqdm(range(epochs)):
            err_hist = jnp.array([])
            mse, gradients = gradfunc(params, state, D, target, err_hist, steps, )
            params = self.controller.update_params(params, lr, gradients)

            print("\n", mse, gradients)
            x_mse.append(mse)

        x_state = []
        state = self.plant.reset()
        err = target - state
        err_hist = jnp.array([err])
        for t in range(250):
            x_state.append(state)
            U = self.controller.step(params, err, err_hist) # calc control signal
            state = self.plant.step(state, U, 0.0) # calc plant output
            err = target - state
            err_hist = jnp.append(err_hist, err)
        plt.plot(x_mse)
        plt.show()
        plt.plot(x_state)
        plt.show()

    def simulate(self, params, state, D, target, err_hist, timesteps=20):

        # Initialize plant/controller
        # D = jnp.zeros(timesteps)#np.random.rand(timesteps) * 0.001 * 2 - 0.001

        # Simulate t timesteps
        # U = 0.0 # TODO: initialize: but how...?
        err = target - state
        err_hist = jnp.append(err_hist, err)
        for t in range(timesteps):
            U = self.controller.step(params, err, err_hist) # calc control signal
            state = self.plant.step(state, U, D[t]) # calc plant output
            err = target - state
            err_hist = jnp.append(err_hist, err)
            # print(state)
        return jnp.mean(jnp.array([e**2 for e in err_hist])) # mean squared error

    def step(self, params, err, err_hist, d):
        err = self.plant.target - state
        err_hist = jnp.append(err_hist, err)
        U = self.controller.step(params, err, err_hist) # calc control signal
        state = self.plant.step(state, U, d) # calc plant output

if __name__=="__main__":
    plant = BathtubPlant(5.0)
    controller = NeuralController(3,1,[5,5,3],['relu','relu','relu','sigmoid'])
    # controller = DefaultController()
    system = ConSys(controller, plant)
    system.run(epochs=200)
    # print(system.simulate([1,0,0],2000))

