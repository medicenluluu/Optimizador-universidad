import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import re

st.set_page_config(page_title="Optimizador Web", layout="wide")

def parse_function(func_str, variables):
    """Convierte un string a una función simbólica con soporte para multiplicación implícita."""
    try:
        # Reemplazar '^' por '**'
        func_str = func_str.replace('^', '**')
        # Insertar * entre número y variable (ej: 3x1 -> 3*x1)
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        return sp.sympify(func_str)
    except Exception:
        return None

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    n = len(variables)
    return sp.Matrix(n, n, lambda i, j: sp.diff(sp.diff(expr, variables[i]), variables[j]))

def evaluate_func(expr, variables, point):
    subs = {var: val for var, val in zip(variables, point)}
    return float(expr.subs(subs))

def evaluate_grad(grad_exprs, variables, point):
    subs = {var: val for var, val in zip(variables, point)}
    return np.array([float(g.subs(subs)) for g in grad_exprs])

def evaluate_hessian(hessian_expr, variables, point):
    subs = {var: val for var, val in zip(variables, point)}
    return np.array(hessian_expr.subs(subs)).astype(np.float64)

def optimize_gradient_descent(expr, vars_sym, x0, tol, max_iter, alpha_init):
    """Método del Gradiente con PASO FIJO."""
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
        # Dirección de máximo descenso
        p = -g_val 
        
        # PASO FIJO: Se usa alpha_init directamente
        x = x + alpha_init * p
        
        f_val = evaluate_func(expr, vars_sym, x)
        g_val = evaluate_grad(grad_exprs, vars_sym, x)
        error = np.linalg.norm(g_val)
        
        history.append((x.copy(), f_val))
        errors.append(error)
        iteration += 1
        
    return x, f_val, iteration, error, history, errors

# --- Resto de las funciones (Conjugado/Newton/Interfaz) ---
# Nota: Debes actualizar la llamada en show_results_page para no pasar c1 y c2 a esta función
# Ejemplo: res_x, ... = optimize_gradient_descent(cfg['expr'], cfg['vars_sym'], cfg['x0'], cfg['tol'], cfg['max_iter'], cfg['alpha_init'])

def init_session_state():
    if 'page' not in st.session_state: st.session_state.page = "login"
    if 'score' not in st.session_state: st.session_state.score = 0
    if 'user_name' not in st.session_state: st.session_state.user_name = ""
    if 'run_completed' not in st.session_state: st.session_state.run_completed = False

def show_results_page():
    cfg = st.session_state.config
    # ... (Tu lógica de visualización aquí)
    if not st.session_state.run_completed:
        if cfg['method'] == "Gradiente (Steepest Descent)":
            # Llamada corregida sin c1 y c2
            res_x, res_f, res_iter, res_err, history, errors = optimize_gradient_descent(
                cfg['expr'], cfg['vars_sym'], cfg['x0'], cfg['tol'], cfg['max_iter'], cfg['alpha_init']
            )
        # ... resto del código ...
