import numpy as np
from controller import create_controller
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
    params = controller.init_params()

    params, mse_log, param_log = system.train(**config['system']['train'])
    param_log = np.array(param_log)

    # Just some interesting logging/plotting
    print(f"Tuned Parameters: {params}")
    plt.figure(f"Loss (MSE) [{config['plant']['name']}-{config['controller']['name']}]")
    plt.title("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(mse_log)
    plt.tight_layout()

    # ugly hardcoding but works
    if config['controller']['name'] == "default" and config['system']['train']['enable_param_logging']: 
        plt.figure(f"Control Parameters [{config['plant']['name']}]")
        plt.title("Control Parameters")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.plot(param_log[:,0], label="$K_p$")
        plt.plot(param_log[:,1], label="$K_i$")
        plt.plot(param_log[:,2], label="$K_d$")
        plt.legend()
        plt.tight_layout()
    plt.show()


    # Test the system
    if config['system']['do_test']:
        print("Testing parameters")
        test_steps = config['system']['test_steps']
        state_log, mse_log = system.test(params, test_steps)
        
        plt.figure(f"Simulation - [{config['plant']['name']}-{config['controller']['name']}]")
        plt.title(f"State over {test_steps} steps")
        plt.xlabel("Timesteps")
        plt.ylabel("Value")
        plt.ylim((plant.TARGET-1.0, plant.TARGET+1.0)) # to give a normalized scale (easy to compare)
        plt.plot(np.array(state_log), label="value")
        plt.plot([plant.TARGET]*len(state_log), label="target")
        plt.legend()
        plt.tight_layout()
        plt.show()
