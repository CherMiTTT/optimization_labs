import sympy as sym
import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
function = type(lambda: 0)

x_1, x_2 = sym.symbols('x1, x2')

def f(x: np.array):
    fun = f_sym()
    return fun.subs([(x_1, x[0]), (x_2, x[1])])

def f_sym():
    fun = x_1 * x_1 + x_2 * x_2
    return fun

def g_1_sym(): # ограничение = 0
    g_1 = x_1 - 1
    return g_1

def g_2_sym(): # ограничение <= 0
    g_2 = x_1 + x_2 - 2
    return g_2

def P(r_k: np.double):
    g_1 = g_1_sym()

    g_2 = g_2_sym()
    g_2_plus = sym.Max(0, g_2)

    fun = (r_k / 2) * (g_1**2 + g_2_plus**2)
    return fun

def F(r_k: np.double):
    return f_sym() + P(r_k)

def grad_f(expr: sym.core.add.Add, x: np.ndarray) -> np.ndarray:
    x1, x2 = sym.symbols('x1, x2')
    grad_x1 = sym.diff(expr, x1)
    grad_x2 = sym.diff(expr, x2)
    grad_x1 = grad_x1.subs([(x1, x[0]), (x2, x[1])])
    grad_x2 = grad_x2.subs([(x1, x[0]), (x2, x[1])])
    return np.array([grad_x1, grad_x2], dtype=np.double)

def fastest_grad_descend(F_r_k: sym.core.add.Add, x_0: np.ndarray, eps: np.double):
    k = 0 #номер итерации
    x_k = x_0
    x_next = x_k
    print("F = ", F_r_k)

    while 1:
        grad = grad_f(F_r_k, x_k)
        if LA.norm(x_k) < eps:
            return (x_k)
        x1, x2 = sym.symbols('x1, x2')
        t_next = sym.symbols('t_next')
        expr = F_r_k.subs([(x1, x_next[0] - t_next * grad[0]), (x2, x_next[1]- t_next * grad[1])])
        lambd = sym.lambdify(t_next, expr, "scipy")
        t_k = minimize_scalar(lambd, bracket=[-10, 10], method="golden").x
        x_next = x_k - t_k * grad
        f_next = F_r_k.subs([(x1, x_next[0]), (x2, x_next[1])])
        print("x_next = ", x_next)
        print("x_k = ", x_k)
        print("f_next = ", f_next)
        f_k = F_r_k.subs([(x1, x_k[0]), (x2, x_k[1])])
        print("f_k = ", f_k)
        if LA.norm(x_next - x_k) < eps and np.fabs(float(f_next) - float(f_k)) < eps:
            return (x_next)
        else:
            x_k = x_next
            k += 1

def external_penalties():
    r_k = 1
    x_k: np.array = [2, 1]
    m = 1
    p = 2

    k = 1 # номер итерации
    eps = 0.001

    while True:
        K = k

        # expr = sym.lambdify([(x_1, x_2)], F(r_k), "scipy")
        # x_min =  minimize(expr, x_k).x
        x_min = fastest_grad_descend(F(r_k), x_k, eps)

        print("x_min = ",x_min)

        print(P(r_k).subs([(x_1, x_min[0]), (x_2, x_min[1])]))
        if P(r_k).subs([(x_1, x_min[0]), (x_2, x_min[1])]) < eps:
            return (x_min, f(x_min))
        else:
            r_k = K * r_k
            x_k = x_min
            k = k + 1


print(external_penalties())
