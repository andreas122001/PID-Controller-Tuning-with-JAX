import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from controller import AbstractController, DefaultController, NeuralController
from plant import AbstractPlant, BathtubPlant, CournotPlant, RobotArmPlant
import matplotlib.pyplot as plt
from config import h_params
from consys import ConSys


if __name__=="__main__":
    plant_name = h_params['plant']['name']
    if plant_name == 'bathtub':
        plant = BathtubPlant(**h_params['plant']['bathtub'])
    elif plant_name == 'cournot':
        plant = CournotPlant(**h_params['plant']['cournot'])
    elif plant_name == 'robot':
        plant = RobotArmPlant(**h_params['plant']['robot'])
    else:
        raise ValueError(f"Invalid plant name given: '{plant_name}'.")
    
    controller_name = h_params['controller']['name']
    if controller_name == "default":
        controller = DefaultController(**h_params['controller']['default'])
    elif controller_name == "neural":
        controller = NeuralController(**h_params['controller']['neural'])
    else:
        raise ValueError(f"Invalid controller name given: '{controller_name}'")

    # Initialize and train system 
    system = ConSys(controller, plant)
    params, mse_log = system.train(**h_params['system'])
    
    # Just some interesting logging/plotting
    print(f"Tuned Parameters: {params}")
    print("Logging MSE")
    plt.title("Mean Squared Error (MSE)")
    plt.xlabel("Timesteps")
    plt.ylabel("MSE")
    plt.plot(mse_log)
    plt.tight_layout()
    plt.show()

    # Test the system
    print("Testing parameters")
    test_steps = 100
    state_log, mse_log = system.test(params, test_steps)
    
    print("Test state over time")
    plt.title(f"Simulation state over {test_steps} steps")
    plt.xlabel("Timesteps")
    plt.ylabel("Current state")
    plt.ylim((4.5, 5.5)) # fit to this interval
    plt.plot(state_log)
    plt.tight_layout()
    plt.show()

