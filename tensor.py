import numpy as np
from enum import Enum


class Operations(Enum):
    plus = "+"
    minus = "-"
    matmul = "@"
    mul = "*"
    div = "/"
    exp = "exp"
    log = "log"
    sum = "sum"
    soft = "soft"
    relu = "relu"

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

    def __truediv__(self, other):
        # elementwise division
        history = {"value1": self,
                   "value2": other,
                   "operation": Operations.div}
        return Tensor(self.value / other.value, history=history) 

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

    def log(self):
        # exponential elementwise
        history = {"value1": self,
                   "operation": Operations.log}
        return Tensor(np.log(self.value), history=history)

    def softmax(self):
        history = {"value1": self,
                   "operation": Operations.soft}
        shiftx = self.value - np.max(self.value)
        exps = np.exp(shiftx)
        return Tensor(exps / np.sum(exps), history=history)

    def relu(self):
        history = {"value1": self,
                   "operation": Operations.relu}
        return Tensor(np.maximum(0, self.value),
                      history)

    def sum(self):
        # only for vectors
        history = {"value1": self,
                   "operation": Operations.sum}
        return Tensor(np.sum(self.value), history=history)

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
#            return {"diff1": self.history["value2"].value,
#                    "diff2": self.history["value1"].value}
            a = self.history["value1"].value
            b = self.history["value2"].value
            if a.ndim == 0 and b.ndim == 0:
                return {"diff1": b,
                        "diff2": a}
            if a.ndim == 1 and b.ndim == 1:
                return {"diff1": np.diag(b),
                        "diff2": np.diag(a)}

        if self.history["operation"] == Operations.div:
            return {"diff1": 1 / self.history["value2"].value,
                    "diff2": - self.history["value1"].value / (self.history["value2"].value ** 2)}


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

        if self.history["operation"] == Operations.soft:
            A = self.history["value1"].value
            if A.ndim == 0:  # Scalar input
                return {"diff1": 1}
            elif A.ndim == 1:  # Vector input
                softmax_A = stablesoftmax(A)
                diag_softmax = np.diag(softmax_A)
                outer_softmax = np.outer(softmax_A, softmax_A)
                return {"diff1": diag_softmax - outer_softmax}

        if self.history["operation"] == Operations.relu:
            A = self.history["value1"].value
            if A.ndim == 0:
                if A > 0:
                    return np.array(1)
                else:
                    return np.array(0)
            elif A.ndim == 1:
                diag = np.where(A > 0, 1, 0)
                return {"diff1": np.diag(diag)}


        if self.history["operation"] == Operations.log:
            A = self.history["value1"].value
            log_A = np.log(A)

            if A.ndim == 0:  # Scalar input
                return {"diff1": 1 / A}
            elif A.ndim == 1:  # Vector input
                return {"diff1": np.diag(1 / A)}
            else:  # Matrix input
                m, n = A.shape
                dy_dA = np.zeros((m, n, m*n))
                i, j = np.indices((m, n))
                flat_index = i * n + j
                dy_dA[i, j, flat_index] = 1 / A
                return {"diff1": dy_dA}



        if self.history["operation"] == Operations.sum:
            return {"diff1": np.ones_like(self.history["value1"].value)}

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


               # print("------11111")
               # print(differentiation["diff1"])
               # print(self.grads)
               # print(grad1)
                self.history["value1"].grads += grad1#np.reshape(grad1, self.history["value1"].grads.shape)
            if "value2" in self.history:
                if self == root:
                    grad2 = self.scalar_or_matmul(differentiation["diff2"], np.ones_like(self.grads))
                else:
                    grad2 = self.scalar_or_matmul(differentiation["diff2"], self.grads)

               # print("----22222")
               # print(differentiation["diff2"])
               # print(self.grads)
                #print(grad2)
                self.history["value2"].grads +=grad2 #np.reshape(grad2, self.history["value2"].grads.shape)

    def backprop(self):
        root = self
        # Topological sort of the nodes
        def topological_sort(node, visited, sorted_nodes):
            if node in visited:
                return
            visited.add(node)
            if "value1" in node.history:
                topological_sort(node.history["value1"], visited, sorted_nodes)
            if "value2" in node.history:
                topological_sort(node.history["value2"], visited, sorted_nodes)
            sorted_nodes.append(node)

        visited = set()
        sorted_nodes = []
        topological_sort(self, visited, sorted_nodes)

        # Reverse the sorted nodes to process in reverse order
        sorted_nodes.reverse()

        # Compute gradients in reverse order
        for node in sorted_nodes:
            node.gradient(root)


    def clear(self):
        # Topological sort of the nodes
        def topological_sort(node, visited, sorted_nodes):
            if node in visited:
                return
            visited.add(node)
            if "value1" in node.history:
                topological_sort(node.history["value1"], visited, sorted_nodes)
            if "value2" in node.history:
                topological_sort(node.history["value2"], visited, sorted_nodes)
            sorted_nodes.append(node)

        visited = set()
        sorted_nodes = []
        topological_sort(self, visited, sorted_nodes)

        # Reverse the sorted nodes to process in reverse order
   # sorted_nodes.reverse()

        # Compute gradients in reverse order
        for node in sorted_nodes:
            node.grads = np.zeros(node.value.shape)


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
