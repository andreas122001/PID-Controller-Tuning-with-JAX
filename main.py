import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from controller import AbstractController, DefaultController, NeuralController, create_controller
from plant import create_plant
import matplotlib.pyplot as plt
from config import config
from consys import ConSys


if __name__=="__main__":
    # Create plant and controller
    plant = create_plant(config['plant'])
    controller = create_controller(config['controller'])

    # Disable JIT if we want to debug, because it is annoying
    if config['system']['debug']:
        config['system']['enable_jit'] = False

    # Initialize and train system 
    system = ConSys(controller, plant, debug=config['system']['debug'])
    # params = controller.init_params()
    params, mse_log = system.train(**config['system']['train'])
    
    # Just some interesting logging/plotting
    print(f"Tuned Parameters: {params}")
    print("Logging MSE")
    plt.figure("Mean Squared Error (MSE)")
    plt.title("Mean Squared Error (MSE)")
    plt.xlabel("Timesteps")
    plt.ylabel("MSE")
    plt.plot(mse_log)
    plt.tight_layout()
    plt.show()

    # Test the system
    print("Testing parameters")
    test_steps = config['system']['test_steps']
    state_log, mse_log = system.test(params, test_steps)
    
    print("Test state over time")
    plt.figure("Simulation")
    plt.title(f"Simulation state over {test_steps} steps")
    plt.xlabel("Timesteps")
    plt.ylabel("State value")
    plt.ylim((plant.TARGET-0.5, plant.TARGET+0.5)) # to give a normalized scale (easy to compare)
    plt.plot(np.array(state_log), label="value")
    # plt.plot(state_log, label="value")
    # plt.plot(np.array(state_log)[:,1], label="velocity")
    # plt.plot(np.array(mse_log), label="error")
    plt.plot([plant.TARGET]*len(state_log), label="target")
    plt.legend()
    plt.tight_layout()
    plt.show()
