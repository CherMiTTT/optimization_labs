import sympy as sym
import numpy as np
from numpy import linalg as LA


def f(x: np.ndarray) -> np.double:
    return 2 * x[0] * x[0] + x[0] * x[1] + x[1] * x[1]

def grad_f(x: np.ndarray) -> np.ndarray:
    x1, x2 = sym.symbols('x1, x2')
    grad_x1 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x1)
    grad_x2 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x2)
    grad_x1 = grad_x1.subs([(x1, x[0]), (x2, x[1])])
    grad_x2 = grad_x2.subs([(x1, x[0]), (x2, x[1])])
    return np.array([grad_x1, grad_x2], dtype=np.double)

def grad_descend(x_0: np.ndarray, eps: np.double, t_0: np.double):
    k = 0 #номер итерации
    x_k = x_0
    x_next = x_k

    while 1:
        print("k =", k)
        grad = grad_f(x_k)
        
        if LA.norm(x_k) < eps:
            return (x_k, f(x_k))
        t_k = t_0
        while 1:
            x_next = x_k - t_k * grad
            if f(x_next) - f(x_k) >= 0:
                t_k /= 2
            else:
                break
        
        if LA.norm(x_next - x_k) < eps and np.fabs(f(x_next) - f(x_k)) < eps:
            return (x_next, f(x_next))
        else:
            x_k = x_next
            k += 1


print("Grad descend:")            
result_1 = grad_descend(np.array([3, 3]), np.double(0.01), np.double(0.5))
print("x* = ", result_1[0], "\nf(x*) = ", result_1[1])