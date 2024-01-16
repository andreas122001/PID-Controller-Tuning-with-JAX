from jax._src.basearray import Array as Array
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from typing import Tuple

class AbstractPlant(ABC):
    # class State:
    #     value = 0
    def __init__(self, state0, target) -> None:
        self.STATE0 = state0 # Initial state
        self.TARGET = target # Target state
        self.ERROR0 = self._error(state0, target) # Initial error

    def step(self, state, U, D) -> Tuple[jax.Array, jax.Array]:
        """Performs a transition on the input state and returns the new state and the error of the new state as a tuple."""
        state = self._step_function(state, U, D) # Calculate new state 
        error = self._error(state, self.TARGET) # Calculate new error
        return state, error

    @abstractmethod
    def _step_function(self, state, U, D) -> Tuple[jax.Array, jax.Array]:
        """Plant specific step function. This method should be overridden with per-plant state transition logic."""
        pass

    def _error(self, state, target):
        return target - state

    def reset(self) -> Tuple[jax.Array, jax.Array]:
        """Returns tuple of initial state and target state."""
        return self.STATE0, self.ERROR0


class BathtubPlant(AbstractPlant):
    def __init__(self, target, 
                area=100, 
                cross_section=1, 
                gravity=9.81) -> None:
        super().__init__(target, target) # state0 = target
        self.AREA = area
        self.CROSS_SECTION_DRAIN = cross_section
        self.GRAVITY = gravity
    
    def _step_function(self, state, U, D) -> jax.Array:
        V = jnp.sqrt(2*self.GRAVITY*state)
        Q = V*self.CROSS_SECTION_DRAIN
        dB = U + D - Q
        dH = dB / self.AREA

        new_state = jnp.maximum(0, state + dH) # we can't have negative water...

        return new_state


class CournotPlant(AbstractPlant):
    def __init__(self, state0, target) -> None:
        super().__init__(state0, target)

    def _step_function(self, state, U, D) -> Tuple[Array, Array]:
        return super()._step_function(state, U, D)

class RobotArmPlant(AbstractPlant):
    def __init__(self, angle0, target, 
                    dt=.01,
                    mass=3,
                    length=1,
                    gravity=9.81,
                ) -> None:
        super().__init__(
            jnp.array([angle0, 0], dtype=float),
            target
        )
        self.dt = dt
        self.MASS = mass
        self.LENGTH = length
        self.GRAVITY = gravity

    def _step_function(self, state, U, D) -> Array:

        T = U*100 - self.LENGTH * self.MASS * self.GRAVITY * jnp.cos(state[0])
        I = (self.MASS * jnp.power(self.LENGTH, 2)) / 3
        a = T / I

        new_w = state[1] + a*self.dt
        new_ø = state[0] + new_w*self.dt + 0.5*a*jnp.power(self.dt, 2)
        # new_ø = jnp.fmod(new_ø + jnp.pi, 2*jnp.pi) - jnp.pi
        # new_ø = jnp.fmod(new_ø - jnp.pi, 2*jnp.pi) + jnp.pi
        new_ø = jnp.maximum(new_ø, -jnp.pi)
        new_ø = jnp.minimum(new_ø, jnp.pi)


        state = jnp.array([new_ø, new_w])

        return state
    
    def _error(self, state, target):
        return target - state[0]
        # return jnp.mod(target - state[0], jnp.pi)

class HyperparamTuner(AbstractPlant):
    pass