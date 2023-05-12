import sympy as sym
import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize

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

    fun = r_k / 2 * (g_1**2 + g_2_plus**2)
    return fun

def F(r_k: np.double):
    return f_sym() + P(r_k)

def external_penalties():
    r_k = 1
    x_k: np.array = [2, 1]
    m = 1
    p = 2

    k = 1 # номер итерации
    eps = 0.0000001

    while True:
        K = k

        expr = sym.lambdify([(x_1, x_2)], F(r_k), "scipy")
        x_min = minimize(expr, x_k, tol=eps).x

        if P(r_k).subs([(x_1, x_min[0]), (x_2, x_min[1])]) < eps:
            print("k = ", k)
            return (x_min, f(x_min))
        else:
            r_k = K * r_k
            x_k = x_min
            k = k + 1

print(external_penalties())
