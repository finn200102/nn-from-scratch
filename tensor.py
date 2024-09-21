import numpy as np
from enum import Enum


class Operations(Enum):
    plus = "+"

class Tensor:
    def __init__(self, value, history={}):
        self.value = np.array(value, dtype=np.float64)
        self.grads = np.zeros_like(self.value)
        self.history = history

    def __add__(self, other):
        history = {"value1": self,
                   "value2": other,
                   "operation": Operations.plus}
        return Tensor(self.value + other.value, history=history)

