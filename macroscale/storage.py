import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class State:
    x: npt.NDArray[np.float64]      # state
    lam: npt.NDArray[np.float64]    # co-state
    t: float                        # time

@dataclass
class Control:
    u: float                        # optimal heading - control
    v: npt.NDArray[np.float64]      # velocity

@dataclass 
class Diagnostic:
    hull_grad : npt.NDArray[np.float64]
    wind_grad : npt.NDArray[np.float64]
    alpha: float = 0.0  # wind direction
    theta: float = 0.0  # hull selected
    v_ref: float = 0.0 # wind speed at current position

@dataclass
class Record:
    state: State
    control: Control
    diagnostic: Diagnostic


