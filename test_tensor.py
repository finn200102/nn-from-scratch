from tensor import Tensor
import numpy as np
import time


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
    A = Tensor(np.array([[1, 2, 3], [1,3, 3], [1,2,3], [1,2,3]]))
    x = Tensor(np.array([1, 2, 3]))
    y = A @ x
    y.gradient(y)

def test_mul():
    a = Tensor(np.array([1, 2]))
    b = Tensor(2)
    c = a * b
    c.gradient(c)
    A = Tensor(np.array([[1, 2], [1,3], [1,2]]))

def test_sub():
    a = Tensor(np.array([1, 2]))
    b = Tensor(np.array([2, 2]))
    c = a - b
    c.gradient(c)
    assert np.allclose(b.grads, np.array([-1, -1], 
                                         dtype=np.float64)), f"""
    The gradient {b.grads} is not {np.array([-1, -1], 
    dtype=np.float64)}"""

def test_exp():
    a = Tensor(np.array([1, 2]))
    c = a.exp()
    c.gradient(c)

    print(c.value)
    print(a.grads)
    A = Tensor(np.array([[1, 2],[3, 4]]))
    C = A.exp()
    C.gradient(C)
    print(C.value)
    print(A.grads)
    T = 0
    for i in range(10):
        t1 = time.time()
        A = Tensor(np.random.uniform(0, 0.20, size=(100, 700)))
        b = Tensor(np.random.uniform(0, 0.20, size=(700, 1)))
        c = A @ b
        z = c.exp()
        B = Tensor(np.random.uniform(0, 0.20, size=(50, 100)))
        d = B @ z 
        d.gradient(d)
        B.gradient(B)
        c.gradient(c)
        z.gradient(z)
        A.gradient(A)
        b.gradient(b)
        t2 = time.time()
        t = t2-t1
        T += t
#        print(t)
    print(T)


#test_addition()
#test_matmul()
#test_mul()
#test_sub()
test_exp()
