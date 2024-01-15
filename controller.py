from jax._src.basearray import Array as Array
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod

class AbstractController(ABC):
    def __init__(self) -> None:
        super().__init__()

    def step(self, params, e, err_hist,) -> jax.Array:
        """Entry function for performing one step of controller simulation."""
        # Calcualte PID-error values
        de = (e - err_hist[-1])
        ie = jnp.sum(jnp.array(err_hist))
        x = jnp.array([e,ie,de])

        # Perform controller calculation
        return self._step_function(params, x)
    
    @abstractmethod
    def _step_function(params, x) -> jax.Array:
        pass

    @abstractmethod
    def init_params(self):
        pass

    def update_params(self, params, lr, gradients):
        # Use PyTrees to update general parameter structures 
        return jax.tree_map(
            lambda param, grad: param - lr*grad, params, gradients
        )

class DefaultController(AbstractController):
    def _step_function(self, params, x):
        return jnp.dot(params,x) # Lin.Alg. version of default PID formula

    def init_params(self):
        return np.array([1,1,1], dtype=float)

class StandardController(AbstractController):
    pass

class NeuralController(AbstractController):
    a_func_map = {
        'relu': lambda x: jnp.maximum(x, 0),
        'sigmoid': lambda x: 1 / (1 + jnp.exp(-x)),
        'tanh': lambda x: jnp.tanh(x),
        'leaky_relu': lambda x: jnp.maximum(x, 0.01*x),
        'softmax': lambda x: jnp.exp(x - jnp.max(x)) / jnp.exp(x - jnp.max(x)).sum(axis=0), # assumes multi-variable input
        'linear': lambda x: x,
    }
        
    def __init__(self, inputs, outputs, hidden_layers, a_funcs) -> None:
        super().__init__()
        if len(a_funcs) != len(hidden_layers) + 1:
            raise Exception(f"Expected {len(hidden_layers) + 1} activation functions, but got {len(a_funcs)}.")
        
        # Class constants
        self.activation_functions = a_funcs
        self.layers = np.concatenate(([inputs], hidden_layers, [outputs])).astype(int)
        
    def _step_function(self, params, x) -> jax.Array:
        for (layer), a_func_name in zip(params, self.activation_functions):
            a = self.a_func_map[a_func_name.lower()]
            x = jnp.dot(x, layer['weights']) + layer['biases']
            x = a(x.flatten())
        return x.flatten()[0]
    
    def init_params(self):
        sender = self.layers[0]
        params = []
        for receiver in self.layers[1:]:
            params.append({
                'weights': np.random.uniform(-1, 1, (sender, receiver)),
                'biases': np.random.uniform(-1, 1, (1, receiver))
            })
            sender = receiver
        return params
