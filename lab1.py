from typing import Tuple
import sympy as sym

def f(x: Tuple[float, float]) -> float:
    return 2 * x[0] * x[0] + x[0] * x[1] + x[1] * x[1]

def grad_f(x: Tuple[float, float]) -> Tuple[float, float]:
    x1, x2 = sym.symbols('x1, x2')
    grad_x1 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x1)
    grad_x2 = sym.diff(2 * x1 * x1 + x1 * x2 + x2 * x2, x2)
    grad_x1 = grad_x1.subs([(x1, x[0]), (x2, x[1])])
    grad_x2 = grad_x2.subs([(x1, x[0]), (x2, x[1])])
    return (grad_x1, grad_x2)

print(grad_f((1, 1)))