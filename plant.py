from jax._src.basearray import Array as Array
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from typing import Tuple

class AbstractPlant(ABC):
    def __init__(self, target) -> None:
        self.TARGET = target
        self.reset()

    def step(self, state, U, D) -> jax.Array:
        return self._step_function(state, U, D)

    @abstractmethod
    def _step_function(self, state, U, D) -> jax.Array:
        pass

    @abstractmethod
    def reset(self) -> Tuple[float, float]:
        """Returns tuple of state and target, (state, target)"""
        pass


class BathtubPlant(AbstractPlant):
    def __init__(self, target, 
                area=100, 
                cross_section=1, 
                gravity=9.81) -> None:
        super().__init__(target)
        self.AREA = area
        self.CROSS_SECTION_DRAIN = cross_section
        self.GRAVITY = gravity
    
    def _step_function(self, state, U, D) -> jax.Array:
        V = jnp.sqrt(2*self.GRAVITY*state)
        Q = V*self.CROSS_SECTION_DRAIN
        dB = U + D - Q
        dH = dB / self.AREA

        new_state = jnp.maximum(0, state + dH) # we can't have negative water...
        error = self.TARGET - new_state

        return new_state, error
    
    def reset(self) -> Tuple[float, float]:
        return self.TARGET, self.TARGET

class CournotCompetitionPlant(AbstractPlant):
    pass

class RobotArmPlant(AbstractPlant):
    def __init__(self, target, state0, resistance) -> None:
        super().__init__(target)
        self.STATE0 = state0
        self.RESISTANCE = resistance

    def _step_function(self, state, U, D) -> Array:
        
        state['acceleration'] += jnp.maximum(U-self.RESISTANCE, 0)
        state['speed'] += state['acceleration']
        state = state['position'] + state['speed'] + D

        error = self.TARGET - state.position
        return state, error

class HyperparamTuner(AbstractPlant):
    pass