import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re

# Configuración de página
st.set_page_config(page_title="Optimizador Web", layout="wide")

def parse_function(func_str, vars_list):
    try:
        func_str = func_str.replace('^', '**')
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        expr = sp.sympify(func_str)
        return expr
    except Exception:
        return None

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def run_gradient_descent(expr, vars_sym, x0, alpha, max_iter):
    history = []
    
    # Precompilar funciones
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    
    curr_x = np.array(x0, dtype=float)
    
    for k in range(max_iter + 1):
        # Evaluar f(x)
        f_val = f_lambdified(*curr_x) if len(vars_sym) > 1 else f_lambdified(curr_x[0])
        
        # Evaluar gradiente
        grad_val = np.array([g(*curr_x) for g in grad_lambdified])
        
        # Registrar
        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val)}
        for i, val in enumerate(curr_x):
            entry[f'x_{i+1}'] = val
        history.append(entry)
        
        # Actualización: x_{k+1} = x_k - alpha * grad
        if k < max_iter:
            curr_x = curr_x - alpha * grad_val
            
    return pd.DataFrame(history)

# Estado de navegación
if 'page' not in st.session_state: st.session_state.page = "config"

st.title("⚙️ Optimizador de Descenso de Gradiente")

n_vars = st.number_input("Número de variables (n)", min_value=1, max_value=10, value=1)
vars_names = [f"x{i+1}" for i in range(n_vars)]

func_input = st.text_input(f"Función f({', '.join(vars_names)})", value="x1**4 - 3*x1**3 + 2")
start_point = st.text_input(f"Punto inicial (x1, ...)", value="0.5")
alpha = st.number_input("Tamaño del paso (alfa)", value=0.01, format="%.4f")
max_iter = st.number_input("Iteraciones", value=10)

if st.button("Ejecutar"):
    vars_sym = sp.symbols(' '.join(vars_names))
    if n_vars == 1: vars_sym = [vars_sym]
    expr = parse_function(func_input, vars_sym)
    
    try:
        clean_str = re.sub(r'[^0-9.,-]', '', start_point)
        x0 = [float(i) for i in clean_str.split(',') if i.strip()]
        
        if expr is not None and len(x0) == n_vars:
            results = run_gradient_descent(expr, vars_sym, x0, alpha, int(max_iter))
            st.dataframe(results)
        else:
            st.error("Error en las dimensiones o la función.")
    except Exception as e:
        st.error(f"Error: {e}")
