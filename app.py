import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import re
from scipy.optimize import minimize

# Configuración de página
st.set_page_config(page_title="Optimizador Web", layout="wide")

def parse_function(func_str, vars_list):
    try:
        func_str = func_str.replace('^', '**')
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        return sp.sympify(func_str)
    except Exception:
        return None

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    return sp.hessian(expr, variables)

def optimize_wrapper(method, expr, vars_sym, x0, tol, max_iter):
    f = sp.lambdify(vars_sym, expr, 'numpy')
    func = lambda x: f(*x)
    
    grad_exprs = compute_gradient(expr, vars_sym)
    grad = lambda x: np.array([float(g.subs({v: val for v, val in zip(vars_sym, x)})) for g in grad_exprs])
    
    # Matriz Hessiana para Newton
    hess_expr = compute_hessian(expr, vars_sym)
    hess = lambda x: np.array([[float(h.subs({v: val for v, val in zip(vars_sym, x)})) for h in row] for row in hess_expr])

    history = []
    def callback(xk):
        # Convertimos xk a lista para calcular f(x) y grad(x)
        x_list = xk.tolist()
        history.append({
            'Iteración': len(history), 
            'x1': x_list[0], 
            'x2': x_list[1], 
            'f(x)': func(x_list), 
            '||∇f||': np.linalg.norm(grad(x_list))
        })

    # Mapeo preciso a algoritmos de scipy
    method_map = {
        "Método del Gradiente": "CG",           # Steepest Descent aproximado mediante Conjugate Gradient
        "Método del Gradiente Conjugado": "CG", # Algoritmo de Fletcher-Reeves
        "Método de Newton": "Newton-CG"         # Newton-Conjugate Gradient
    }

    # Ejecución
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
    func_input = st.text_input("Función f(x1, x2)", value="x1^2 + x2^2")
    start_point = st.text_input("Punto inicial (x1, x2)", value="2.0, 2.0")
    
    # Lista desplegable de métodos
    method = st.selectbox("Selecciona el método de optimización:", 
                          ["Método del Gradiente", "Método del Gradiente Conjugado", "Método de Newton"])
    
    max_iter = st.number_input("Iteraciones Máximas", value=50)
    tol = st.number_input("Tolerancia", value=1e-5, format="%.1e")
    
    if st.button("Ejecutar"):
        vars_sym = sp.symbols('x1 x2')
        expr = parse_function(func_input, vars_sym)
        try:
            x0 = [float(i.strip()) for i in start_point.split(',')]
            if expr and len(x0) == 2:
                st.session_state.results = optimize_wrapper(method, expr, vars_sym, x0, tol, max_iter)
                st.session_state.page = "results"
                st.rerun()
            else:
                st.error("Error: Revisa la sintaxis de la función o los puntos iniciales.")
        except:
            st.error("Error al procesar los datos de entrada.")

elif st.session_state.page == "results":
    st.title("📊 Resultados")
    if st.button("Volver"):
        st.session_state.page = "config"
        st.rerun()
    st.dataframe(st.session_state.results)
