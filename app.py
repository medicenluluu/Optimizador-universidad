import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import re
st.set_page_config(page_title="Optimizador Web", layout="wide")
def parse_function(func_str, variables):
"""Convierte un string a una función simbólica con soporte para
multiplicación implícita."""
try:
# Reemplazar '^' por '**'
func_str = func_str.replace('^', '**')
# Insertar * entre número y variable (ej: 3x1 -> 3*x1)
func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
# Parsear
expr = sp.sympify(func_str)
return expr
except Exception:
return None

def compute_gradient(expr, variables):
return [sp.diff(expr, var) for var in variables]
def compute_hessian(expr, variables):
n = len(variables)
return sp.Matrix(n, n, lambda i, j: sp.diff(sp.diff(expr,
variables[i]), variables[j]))
def evaluate_func(expr, variables, point):
subs = {var: val for var, val in zip(variables, point)}
return float(expr.subs(subs))
def evaluate_grad(grad_exprs, variables, point):
subs = {var: val for var, val in zip(variables, point)}
return np.array([float(g.subs(subs)) for g in grad_exprs])

def evaluate_hessian(hessian_expr, variables, point):
subs = {var: val for var, val in zip(variables, point)}
return np.array(hessian_expr.subs(subs)).astype(np.float64)
def line_search_wolfe(expr, grad_exprs, variables, x_k, p_k,
alpha_init, c1, c2, max_ls_iter=20):
alpha = alpha_init
f_k = evaluate_func(expr, variables, x_k)
g_k = evaluate_grad(grad_exprs, variables, x_k)
dir_deriv = np.dot(g_k, p_k)
if dir_deriv >= 0: return None
alpha_min = 0
alpha_max = np.inf
for _ in range(max_ls_iter):
x_new = x_k + alpha * p_k
f_new = evaluate_func(expr, variables, x_new)
if f_new > f_k + c1 * alpha * dir_deriv:
alpha_max = alpha
alpha = (alpha_min + alpha_max) / 2
else:
g_new = evaluate_grad(grad_exprs, variables, x_new)
dir_deriv_new = np.dot(g_new, p_k)
if abs(dir_deriv_new) > c2 * abs(dir_deriv):
alpha_min = alpha
alpha = alpha * 2 if alpha_max == np.inf else

(alpha_min + alpha_max) / 2

else:
return alpha

return alpha_min if alpha_min > 0 else alpha_init * 0.1
def optimize_gradient_descent(expr, vars_sym, x0, tol, max_iter,
alpha_init, c1, c2):
grad_exprs = compute_gradient(expr, vars_sym)
x = np.array(x0, dtype=float)
history = []
errors = []
f_val = evaluate_func(expr, vars_sym, x)
g_val = evaluate_grad(grad_exprs, vars_sym, x)
error = np.linalg.norm(g_val)
history.append((x.copy(), f_val))
errors.append(error)
iteration = 0
while error > tol and iteration < max_iter:

p = -g_val
# Aplicar alpha_init en la primera iteración o usar búsqueda

de línea

alpha = alpha_init if iteration == 0 else

line_search_wolfe(expr, grad_exprs, vars_sym, x, p, alpha_init, c1,
c2)

if alpha is None or alpha < 1e-10: break
x = x + alpha * p
f_val = evaluate_func(expr, vars_sym, x)
g_val = evaluate_grad(grad_exprs, vars_sym, x)
error = np.linalg.norm(g_val)
history.append((x.copy(), f_val))
errors.append(error)
iteration += 1
return x, f_val, iteration, error, history, errors
# ... (El resto de las funciones: optimize_conjugate_gradient,
optimize_newton,
# show_sidebar, etc., permanecen iguales, asegurando la consistencia)
...
def main():
if 'page' not in st.session_state: st.session_state.page = "login"
# ... resto del flujo de la app ...
if __name__ == "__main__":
main()
