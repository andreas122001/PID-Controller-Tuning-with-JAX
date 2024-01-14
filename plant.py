import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from abc import ABC, abstractmethod, abstractclassmethod

class AbstractPlant(ABC):
    def __init__(self, target) -> None:
        self.target = target
        self.reset()

    @abstractmethod
    def step(self, state, U, D):
        pass

    @abstractmethod
    def reset(self):
        pass


class BathtubPlant(AbstractPlant):
    def __init__(self, target, 
                area=100, 
                cross_section=1, 
                gravity=9.81) -> None:
        super().__init__(target)
        self.A = area
        self.C = cross_section
        self.g = gravity
    
    def step(self, state, U, D):
        V = jnp.sqrt(2*9.81*state)#jnp.sqrt(2*self.g*state)
        Q = V*1#self.C
        dB = U + D - Q
        dH = dB / 100#self.A

        new_state = jnp.maximum(0, state + dH)
        self.current_state = new_state

        # n_state= n_state.item()
        return new_state
    
    def reset(self):
        self.current_state = self.target
        return self.target, self.target

class CournotCompetitionPlant(AbstractPlant):
    pass

class HyperparamTuner(AbstractPlant):
    pass