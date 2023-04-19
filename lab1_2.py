import sympy as sym
import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize_scalar

def f(x: np.ndarray) -> np.double:
    return 2 * x[0] * x[0] + x[0] * x[1] + x[1] * x[1]

def grad_f(x: np.ndarray) -> np.ndarray:
    x1, x2 = sym.symbols('x1, x2')
    grad_x1 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x1)
    grad_x2 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x2)
    grad_x1 = grad_x1.subs([(x1, x[0]), (x2, x[1])])
    grad_x2 = grad_x2.subs([(x1, x[0]), (x2, x[1])])
    return np.array([grad_x1, grad_x2], dtype=np.double)
    
def hessian_f(x: np.ndarray) -> np.ndarray:
    x1, x2 = sym.symbols('x1, x2')
    h1_1 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x1, 2)
    h1_1 = h1_1.subs([(x1, x[0]), (x2, x[1])])
    h1_2 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x1, x2)
    h1_2 = h1_2.subs([(x1, x[0]), (x2, x[1])])
    h2_1 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x2, x1)
    h2_1 = h2_1.subs([(x1, x[0]), (x2, x[1])])
    h2_2 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x2, 2)
    h1_2 = h1_2.subs([(x1, x[0]), (x2, x[1])])
    return np.array([[h1_1, h1_2], [h2_1, h2_2]], dtype=np.double)

def grad_descend(x_0: np.ndarray, eps: np.double, t_0: np.double):
    k = 0 #номер итерации
    x_k = x_0
    x_next = x_k

    while 1:
        print("k =", k)
        grad = grad_f(x_k)
        
        if LA.norm(grad) < eps:
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
        
def fastest_grad_descend(x_0: np.ndarray, eps: np.double):
    k = 0 #номер итерации
    x_k = x_0
    x_next = x_k

    while 1:
        print("k =", k)
        grad = grad_f(x_k)
        if LA.norm(x_k) < eps:
            return (x_k, f(x_k))
        
        t_next = sym.symbols('t_next')
        expr = f(x_k - t_next * grad)
        lambd = sym.lambdify(t_next, expr, "scipy")
        t_k = minimize_scalar(lambd, bracket=[-10, 10], method="golden").x
        print("t_k = ", t_k)
        x_next = x_k - t_k * grad
        if LA.norm(x_next - x_k) < eps and np.fabs(f(x_next) - f(x_k)) < eps:
            return (x_next, f(x_next))
        else:
            x_k = x_next
            k += 1

def newton(x_0: np.ndarray, eps: np.double, t_0: np.double):
    k = 0 #номер итерации
    x_k = x_0
    x_next = x_k
    
    while 1:
        d_k = 0; t_k = 0
        print("k =", k)
        grad = grad_f(x_k)
        if LA.norm(grad) < eps:
            return (x_k, f(x_k))
        hessian = hessian_f(x_k)
        inv_hessian = LA.inv(hessian)
        if np.all(LA.eigvals(inv_hessian) > 0):
            d_k = -1*np.dot(inv_hessian, grad)
            t_k = 1
        else:
            d_k = -1*grad
            t_k = t_0
            x_temp = x_k+t_k*d_k
            while f(x_temp)>=f(x_k):
                t_k = -1*t_k/2
                x_temp = x_k+t_k*d_k
        x_next = x_k+t_k*d_k
        if LA.norm(x_next - x_k) < eps and np.fabs(f(x_next) - f(x_k)) < eps:
            return (x_next, f(x_next))
        else:
            x_k = x_next
            k += 1
        

def newton_rafson(x_0: np.ndarray, eps: np.double):
    k = 0 #номер итерации
    x_k = x_0
    x_next = x_k
    
    while 1:
        d_k = 0; t_k = 0
        print("k =", k)
        grad = grad_f(x_k)
        if LA.norm(grad) < eps:
            return (x_k, f(x_k))
        hessian = hessian_f(x_k)
        inv_hessian = LA.inv(hessian)
        if np.all(LA.eigvals(inv_hessian) > 0):
            d_k = -1*np.dot(inv_hessian, grad)
        else:
            d_k = -1*grad
        t_next = sym.symbols('t_next')
        expr = f(x_k + t_next * d_k)
        lambd = sym.lambdify(t_next, expr, "scipy")
        t_k = minimize_scalar(lambd, bracket=[-10, 10], method="golden").x
        x_next = x_k+t_k*d_k
        if LA.norm(x_next - x_k) < eps and np.fabs(f(x_next) - f(x_k)) < eps:
            return (x_next, f(x_next))
        else:
            x_k = x_next
            k += 1
        

x_0 = np.array([0.5, 1])
eps = np.double(0.01)
t_0 = np.double(0.5)

print("Grad descend:")            
result_1 = grad_descend(x_0, eps, t_0)
print("x* = ", result_1[0], "\nf(x*) = ", result_1[1])
print("Fastest grad descend:")
result_2 = fastest_grad_descend(x_0, eps)
print("x* = ", result_2[0], "\nf(x*) = ", result_2[1])
print("Newton:")            
result_3 = newton(x_0, eps, t_0)
print("x* = ", result_3[0], "\nf(x*) = ", result_3[1])
print("Newton-Rafson:")
result_4 = newton_rafson(x_0, eps)
print("x* = ", result_4[0], "\nf(x*) = ", result_4[1])
