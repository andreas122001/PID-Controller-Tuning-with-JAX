import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from controller import AbstractController, DefaultController, NeuralController
from plant import AbstractPlant, BathtubPlant
import matplotlib.pyplot as plt
from config import h_params

# Read config from file
# with open("./config.json") as f:
#     h_params = json.load(f)

class ConSys:
    def __init__(self, controller: AbstractController, 
                        plant: AbstractPlant) -> None:
        self.controller = controller
        self.plant = plant
        self._step_once = jax.jit(self._step_once) # should be jitted anyway

    def train(self, epochs, 
                learning_rate=0.05, 
                steps_per_epoch=10, 
                noise_range=[-.1,.1], 
                enable_jit=False):
        
        # Noise function
        gen_noise = lambda n: np.random.uniform(*noise_range, n)

        params = self.controller.init_params()
        gradfunc = jax.value_and_grad(self._simulate_one_epoch)

        if enable_jit:
            print("Compiling simulation...")
            gradfunc = jax.jit(gradfunc, static_argnames=['steps'])

        x_mse = [0]*epochs
        log_params = [0]*epochs
        for i in tqdm(range(epochs)):
            log_params[i] = params

            D = gen_noise(steps_per_epoch) #jnp.zeros(steps)
            target, state = self.plant.reset()
            
            mse, gradients = gradfunc(params, state, target, D, steps_per_epoch, )
            params = self.controller.update_params(params, learning_rate, gradients)

            # print("\n", mse, gradients, end="\r")
            x_mse[i] = mse

        steps_per_epoch = 250
        log_state = [0]*steps_per_epoch
        target, state = self.plant.reset()
        err = target - state
        err_hist = jnp.array([err])
        D = gen_noise(steps_per_epoch)
        for t in tqdm(range(steps_per_epoch)):
            state, _, err_hist = self._step_once(
                params, state, err, err_hist, target, D[t])
            log_state[t] = state
        # for t in range(steps_per_epoch):
        #     U = self.controller.step(params, err, err_hist) # calc control signal
        #     state = self.plant.step(state, U, D[t]) # calc plant output
        #     err = target - state
        #     err_hist = jnp.append(err_hist, err)
        plt.plot(x_mse)
        plt.title("Error")
        plt.show()
        plt.plot(log_state)
        plt.title("State")
        # plt.plot(U_state)
        plt.show()
        plt.plot(np.array(log_params)[:,0])
        plt.plot(np.array(log_params)[:,1])
        plt.plot(np.array(log_params)[:,2])
        plt.title("Parameters")
        plt.show()

        return params, x_mse

    def _simulate_one_epoch(self, params, state, target, noise_vector, steps):
        """Main differentiable simulation function. Also jittable."""

        # Initialize error and history
        error = target - state
        err_hist = jnp.array([error])
        
        for t in range(steps):
            # Perform one step
            state, error, err_hist = self._step_once(
                params, state, error, err_hist, noise_vector[t]
            )
            
        return jnp.sum(jnp.array([e**2 for e in err_hist]))

    def _step_once(self, params, state, error, err_hist, noise):
        """Performs one step of PID simulation"""
        # Update plant and controller
        signal = self.controller.step(params, error, err_hist)
        new_state, new_error = self.plant.step(state, signal, noise)

        # Add error to history
        err_hist = jnp.append(err_hist, new_error)
        return new_state, new_error, err_hist

    def sim_and_plot(self, params, steps):
        # Log loss
        # Log parameters
        pass

if __name__=="__main__":
    plant = BathtubPlant(**h_params['plant']['bathtub'])
    controller = NeuralController(3,1,[3],['linear','linear'])
    # controller = DefaultController()
    system = ConSys(controller, plant)
    params, mse = system.train(**h_params['system'])
    print(params)
    # print(system.simulate([1,0,0],2000))

