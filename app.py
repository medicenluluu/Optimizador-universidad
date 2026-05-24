import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re
from scipy.optimize import minimize

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

def compute_hessian(expr, variables):
    return sp.hessian(expr, variables)

def optimize_wrapper(method, expr, vars_sym, x0, tol, max_iter, alpha):
    # Aseguramos que vars_sym sea siempre una lista (iterable)
    if not isinstance(vars_sym, (list, tuple)):
        vars_sym = [vars_sym]
        
    f = sp.lambdify(vars_sym, expr, 'numpy')
    
    # Adaptación para que funcione con 1 o más variables
    func = lambda x: f(*x) if len(vars_sym) > 1 else f(x[0])
    
    grad_exprs = compute_gradient(expr, vars_sym)
    grad = lambda x: np.array([float(g.subs({v: val for v, val in zip(vars_sym, x)})) for g in grad_exprs])
    
    hess_expr = compute_hessian(expr, vars_sym)
    hess = lambda x: np.array([[float(h.subs({v: val for v, val in zip(vars_sym, x)})) for h in row] for row in hess_expr])

    history = []
    def callback(xk):
        x_list = xk.tolist() if hasattr(xk, 'tolist') else [xk]
        entry = {'Iteración': len(history), 'f(x)': func(x_list), '||∇f||': np.linalg.norm(grad(x_list))}
        for i, val in enumerate(x_list):
            entry[f'x{i+1}'] = val
        history.append(entry)

    method_map = {
        "Método del Gradiente": "CG",
        "Método del Gradiente Conjugado": "CG",
        "Método de Newton": "Newton-CG"
    }

    # Nota: Scipy maneja internamente el paso óptimo en CG/Newton, 
    # pero aquí se expone el alfa por si deseas futuras personalizaciones manuales.
    if method == "Método de Newton":
        minimize(func, x0, method=method_map[method], jac=grad, hess=hess, callback=callback, options={'maxiter': max_iter, 'xtol': tol})
    else:
        minimize(func, x0, method=method_map[method], jac=grad, callback=callback, options={'maxiter': max_iter, 'gtol': tol})
        
    return pd.DataFrame(history)

# Estado de navegación
if 'page' not in st.session_state: st.session_state.page = "login"

if st.session_state.page == "login":
    st.title("👤 Registro de Usuario")
    name = st.text_input("Nombre de usuario:")
    if st.button("Ingresar"):
        if name:
            st.session_state.user_name = name
            st.session_state.page = "config"
            st.rerun()

elif st.session_state.page == "config":
    st.title(f"⚙️ Configuración - {st.session_state.user_name}")
    
    n_vars = st.number_input("Número de variables (n)", min_value=1, max_value=10, value=2)
    vars_names = [f"x{i+1}" for i in range(n_vars)]
    
    func_input = st.text_input(f"Función f({', '.join(vars_names)})", value=" + ".join([f"{v}^2" for v in vars_names]))
    start_point = st.text_input(f"Punto inicial ({', '.join(vars_names)})", value=", ".join(["1.0"] * n_vars))
    
    method = st.selectbox("Método de optimización:", 
                          ["Método del Gradiente", "Método del Gradiente Conjugado", "Método de Newton"])
    
    alpha = st.number_input("Tamaño del paso (alfa)", value=0.01, format="%.4f")
    max_iter = st.number_input("Iteraciones Máximas", value=50)
    tol = st.number_input("Tolerancia", value=1e-5, format="%.1e")
    
    if st.button("Ejecutar"):
        syms_str = ' '.join(vars_names)
        vars_sym = sp.symbols(syms_str)
        if n_vars == 1:
            vars_sym = [vars_sym]
            
        expr = parse_function(func_input, vars_sym)
        
        try:
            raw_points = start_point.replace('(', '').replace(')', '').split(',')
            x0 = [float(i.strip()) for i in raw_points if i.strip()]
            
            if expr is not None and len(x0) == n_vars:
                st.session_state.results = optimize_wrapper(method, expr, vars_sym, x0, tol, max_iter, alpha)
                st.session_state.page = "results"
                st.rerun()
            else:
                st.error(f"Error: Asegúrate de ingresar una función válida y {n_vars} coordenadas numéricas.")
        except Exception as e:
            st.error(f"Error al procesar los datos: {str(e)}")

elif st.session_state.page == "results":
    st.title("📊 Resultados")
    if st.button("Volver"):
        st.session_state.page = "config"
        st.rerun()
    st.dataframe(st.session_state.results)
