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

    def differentiation(self):
        if self.history["operation"] == Operations.plus:
            return {"diff1": np.eye(len(self.history["value1"].value)),
                    "diff2": np.eye(len(self.history["value2"].value))}

    def gradient(self, root):
        if self.history:
            differentiation = self.differentiation()
            if "value1" in self.history:
                if self == root:
                    grad1 = differentiation["diff1"] @ np.ones_like(self.grads)
                else:
                    grad1 = differentiation["diff1"] @ self.grads
                self.history["value1"].grads += grad1
            if "value" in self.history:
                if self == root:
                    grad2 = differentiation["diff2"] @ np.ones_like(self.grads)
                else:
                    grad2 = differentiation["diff2"] @ self.grads
                self.history["value2"].grads += grad2


