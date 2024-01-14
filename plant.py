import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod
from typing import Tuple

class AbstractPlant(ABC):
    def __init__(self, target) -> None:
        self.target = target
        self.reset()

    def step(self, state, U, D):
        return self._step_function(state, U, D)

    @abstractmethod
    def _step_function(self, state, U, D):
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
    
    def _step_function(self, state, U, D):
        V = jnp.sqrt(2*self.GRAVITY*state)
        Q = V*self.CROSS_SECTION_DRAIN
        dB = U + D - Q
        dH = dB / self.AREA

        new_state = jnp.maximum(0, state + dH) # we can't have negative water...
        self.current_state = new_state

        return new_state
    
    def reset(self):
        self.current_state = self.target
        return self.target, self.target

class CournotCompetitionPlant(AbstractPlant):
    pass

class HyperparamTuner(AbstractPlant):
    pass