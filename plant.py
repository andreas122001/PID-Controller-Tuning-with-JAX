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
    def __init__(self, state0, target,
                    p_max=10,
                    margin_cost=0.1
                 ) -> None:
        super().__init__(state0, target)

    def _step_function(self, state, U, D) -> Tuple[Array, Array]:
        q1 = q1 + U
        q2 = q2 + D

        q = q1 + q2

        p = self.p_max - q

        p1 = q1*jnp.floor(p - self.margin_cost)

        return jnp.array([q1, p1])
    
    def _error(self, state, target):
        return super()._error(state[1], target)


class RobotArmPlant(AbstractPlant):
    def __init__(self, angle0, target, 
                    dt=.01,
                    mass=1,
                    length=50,
                    gravity=9.81,
                    multiplier=50,
                    coulomb_fric=0.4,
                    viscous_fric=0.5,
                    interval=[-0.5*jnp.pi, 0.5*jnp.pi],
                ) -> None:
        super().__init__(
            jnp.array([angle0, 0], dtype=float),
            target
        )
        self.dt = dt
        self.MASS = mass
        self.LENGTH = length
        self.GRAVITY = gravity 
        self.MULTIPLIER = multiplier
        self.INTERVAL = interval
        self.C_FRIC = coulomb_fric
        self.V_FRIC = viscous_fric

    def _step_function(self, state, U, D) -> Array:

        ø = state[0]
        w = state[1]

        T = self.MULTIPLIER*U - self.LENGTH * self.MASS * self.GRAVITY * jnp.cos(ø)
        T -= self.C_FRIC*jnp.copysign(1, w) + self.V_FRIC*w # Friction

        I = (self.MASS * jnp.power(self.LENGTH, 2)) / 3

        a = T / I

        new_w = w + a*(self.dt)
        new_ø = ø + w*(self.dt)
        
        # Check position against boundaries
        boundary_neg =  jnp.copysign(1, new_ø - self.INTERVAL[0])
        boundary_pos = -jnp.copysign(1, new_ø - self.INTERVAL[1]) # copy sign to preserve 0 as positive

        # Reflect speed if boundary is breached
        new_w = boundary_pos*boundary_neg * new_w

        # Fix position to boundary
        new_ø = jnp.maximum(new_ø, self.INTERVAL[0])
        new_ø = jnp.minimum(new_ø, self.INTERVAL[1])

        return jnp.array([new_ø, new_w])
    
    def _error(self, state, target):
        return target - state[0]
        # return jnp.mod(target - state[0], jnp.pi)

class HyperparamTuner(AbstractPlant):
    pass