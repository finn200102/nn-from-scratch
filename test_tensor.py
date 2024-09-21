from tensor import Tensor
import numpy as np


def test_addition():
    a = Tensor(np.array([1, 2]))
    b = Tensor(np.array([3, 4]))
    c = a + b
    c.gradient(c)
    assert np.allclose(a.grads, np.array([1, 1], dtype=np.float64)), f"""
    The gradient {a.grads} is not {np.array([1, 1], dtype=np.float64)}"""

    A = Tensor(np.array([[1, 2], [3, 4]]))
    B = Tensor(np.array([[1, 2], [3, 4]]))
    C = A + B
    C.gradient(C)
    assert np.allclose(A.grads, np.array([[1, 1], [1, 1]], 
                                         dtype=np.float64)), f"""
    The gradient {A.grads} is not {np.array([[1, 1], [1, 1]], 
                                         dtype=np.float64)}"""

    e = Tensor(np.array(1))
    f = Tensor(np.array(1))
    g = e + f
    g.gradient(g)
    assert np.allclose(a.grads, np.array(1, dtype=np.float64)), f"""
    The gradient {a.grads} is not {np.array(1, dtype=np.float64)}"""


def test_matmul():
    A = Tensor(np.array([[1, 2], [3, 4]]))
    x = Tensor(np.array([1, 2]))
    y = A @ x
    y.gradient(y)
    assert np.allclose(A.grads, np.array([[1, 2], [1, 2]], 
                                         dtype=np.float64)), f"""
    The gradient {A.grads} is not {np.array([[1, 2], [1, 2]], dtype=np.float64)}"""

    assert np.allclose(x.grads, np.array([4, 6], 
                                         dtype=np.float64)), f"""
    The gradient {x.grads} is not {np.array([4, 7], dtype=np.float64)}"""
    A = Tensor(np.array([[1, 2], [1,3], [1,2]]))
    print("---------++++")
    print(A.value)
    x = Tensor(np.array([1, 2]))
    y = A @ x
    print(y.value)
    y.gradient(y)

def test_mul():
    a = Tensor(np.array([1, 2]))
    b = Tensor(2)
    c = a * b
    c.gradient(c)
    A = Tensor(np.array([[1, 2], [1,3], [1,2]]))


#test_addition()
test_matmul()
#test_mul()
