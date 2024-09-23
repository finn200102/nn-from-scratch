import numpy as np
from enum import Enum


class Operations(Enum):
    plus = "+"
    minus = "-"
    matmul = "@"
    mul = "*"
    exp = "exp"

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

    def __sub__(self, other):
        history = {"value1": self,
                   "value2": other,
                   "operation": Operations.minus}
        return Tensor(self.value - other.value, history=history)

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

    def exp(self):
        # exponential elementwise
        history = {"value1": self,
                   "operation": Operations.exp}
        return Tensor(np.exp(self.value), history=history)

    def differentiation(self):
        if self.history["operation"] == Operations.plus:
            return {
                    "diff1": np.squeeze(np.eye(len(np.atleast_1d(self.history["value1"].value)))),
                    "diff2": np.squeeze(np.eye(len(np.atleast_1d(self.history["value2"].value))))
}

        if self.history["operation"] == Operations.minus:
            return {
                    "diff1": np.squeeze(np.eye(len(np.atleast_1d(self.history["value1"].value)))),
                    "diff2": - np.squeeze(np.eye(len(np.atleast_1d(self.history["value2"].value))))
}

        if self.history["operation"] == Operations.mul:
            return {"diff1": self.history["value2"].value,
                    "diff2": self.history["value1"].value}

        if self.history["operation"] == Operations.exp:

            A = self.history["value1"].value
            exp_A = np.exp(A)

            if A.ndim == 0:  # Scalar input0
                return {"diff1": exp_A}
            elif A.ndim == 1:  # Vector input
                return {"diff1": np.diag(exp_A)}
            else:  # Matrix input
                m, n = A.shape
                dy_dA = np.zeros((m, n, m*n))
                i, j = np.indices((m, n))
                flat_index = i * n + j
                dy_dA[i, j, flat_index] = exp_A
                return {"diff1": dy_dA}

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
            return a * b  # Scalar product
        elif a.ndim == 3 and b.ndim == 2:
            # Handle the case for exp of a matrix
            return np.sum(a * b.flatten(), axis=2)
        else:
            return a @ b  # Matrix multiplication

    def gradient(self, root):
        if self.history:
            differentiation = self.differentiation()
            if "value1" in self.history:
                if self == root:
                    grad1 = self.scalar_or_matmul(differentiation["diff1"], np.ones_like(self.grads))
                else:
                    grad1 = self.scalar_or_matmul(differentiation["diff1"], self.grads)

                print("+++")
                print(differentiation["diff1"])
                print(grad1)
                print(np.ones_like(self.grads))
                print(self.history["value1"].grads)
                self.history["value1"].grads += grad1#np.reshape(grad1, self.history["value1"].grads.shape)
            if "value2" in self.history:
                if self == root:
                    grad2 = self.scalar_or_matmul(differentiation["diff2"], np.ones_like(self.grads))
                else:
                    grad2 = self.scalar_or_matmul(differentiation["diff2"], self.grads)

                print("---")
                print(differentiation["diff2"])
                self.history["value2"].grads +=grad2 #np.reshape(grad2, self.history["value2"].grads.shape)


