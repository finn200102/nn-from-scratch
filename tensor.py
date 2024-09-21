import numpy as np
from enum import Enum


class Operations(Enum):
    plus = "+"
    matmul = "@"
    mul = "*"

class Tensor:
    def __init__(self, value, history={}):
        self.value = np.array(value, dtype=np.float64)
        self.grads = np.zeros(self.value.shape)
        self.history = history

    def __add__(self, other):
        history = {"value1": self,
                   "value2": other,
                   "operation": Operations.plus}
        return Tensor(self.value + other.value, history=history)

    def __mul__(self, other):
        # elementwise multiplication
        history = {"value1": self,
                   "value2": other,
                   "operation": Operations.mul}
        return Tensor(self.value * other.value, history=history)

    def __matmul__(self, other):
        history = {"value1": self,
                   "value2": other,
                   "operation": Operations.matmul}
        return Tensor(self.value @ other.value, history=history)

    def differentiation(self):
        if self.history["operation"] == Operations.plus:
            return {"diff1": np.eye(len(np.atleast_1d(
                self.history["value1"].value))),
                    "diff2": np.eye(len(np.atleast_1d(
                        self.history["value2"].value)))}

        if self.history["operation"] == Operations.mul:
            return {"diff1": self.history["value2"].value,
                    "diff2": self.history["value1"].value}

        if self.history["operation"] == Operations.matmul:
            A = self.history["value1"].value
            x = self.history["value2"].value
            m, n = A.shape

            # Gradient with respect to A (self.history["value1"])
            dy_dA = np.zeros((m, n, m))
            for i in range(m):
                dy_dA[:, :, i] = np.outer(np.eye(m)[i], x)

            # Gradient with respect to x (self.history["value2"])
            dy_dx = A.T
            return {"diff1": dy_dA, "diff2": dy_dx}

    def scalar_or_matmul(self, a, b):
        if np.isscalar(a) or np.isscalar(b) or a.shape == () or b.shape == ():
            print("scalar")
            print(a)
            print(b)
            return a * b  # Scalar product
        else:
            print("matmul")
            print(a)
            print(b)
            return a @ b  # Matrix multiplication

    def gradient(self, root):
        if self.history:
            differentiation = self.differentiation()
            if "value1" in self.history:
                if self == root:
                    grad1 = self.scalar_or_matmul(differentiation["diff1"], np.ones_like(self.grads))
                else:
                    grad1 = self.scalar_or_matmul(differentiation["diff1"], self.grads)
                print("---")
                print(grad1) 
                print(self.history["value1"].grads)

                self.history["value1"].grads += grad1#np.reshape(grad1, self.history["value1"].grads.shape)
                print(self.history["value1"].grads)
            if "value2" in self.history:
                if self == root:
                    grad2 = self.scalar_or_matmul(differentiation["diff2"], np.ones_like(self.grads))
                else:
                    grad2 = self.scalar_or_matmul(differentiation["diff2"], self.grads)
                print("-+-")
                print(grad2)
                print(self.history["value1"].grads)

                self.history["value2"].grads +=grad2 #np.reshape(grad2, self.history["value2"].grads.shape)
                print(self.history["value2"].grads)


