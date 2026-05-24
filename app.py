import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import re
from scipy.optimize import minimize

# Configuración de página
st.set_page_config(page_title="Optimizador Web Pro", layout="wide")

# Funciones de lógica matemática
def parse_function(func_str, variables):
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

def evaluate_func(expr, variables, point):
    subs = {var: val for var, val in zip(variables, point)}
    return float(expr.subs(subs))

def optimize_wrapper(method, expr, vars_sym, x0, tol, max_iter, alpha):
    # Función lambda para scipy
    f = sp.lambdify(vars_sym, expr, 'numpy')
    func = lambda x: f(*x)
    
    # Gradiente simbólico para scipy
    grad_exprs = compute_gradient(expr, vars_sym)
    grad = lambda x: np.array([float(g.subs({v: val for v, val in zip(vars_sym, x)})) for g in grad_exprs])
    
    history = []
    def callback(xk):
        f_val = func(xk)
        g_val = grad(xk)
        history.append({
            'Iteración': len(history),
            'x1': xk[0],
            'x2': xk[1],
            'f(x)': f_val,
            '||∇f||': np.linalg.norm(g_val)
        })

    # Mapeo de métodos
    method_map = {
        "Método del Gradiente": "CG", # Simplificación para la demo
        "Método del Gradiente Conjugado": "CG",
        "Método de Newton": "Newton-CG"
    }

    res = minimize(func, x0, method=method_map.get(method, "BFGS"), jac=grad, callback=callback, options={'maxiter': max_iter, 'gtol': tol})
    return pd.DataFrame(history)

# Lógica de navegación
if 'page' not in st.session_state: st.session_state.page = "login"

# Página 1: Login
if st.session_state.page == "login":
    st.title("👤 Registro de Usuario")
    name = st.text_input("Nombre de usuario:")
    if st.button("Ingresar"):
        if name:
            st.session_state.user_name = name
            st.session_state.page = "config"
            st.rerun()

# Página 2: Configuración
elif st.session_state.page == "config":
    st.title(f"⚙️ Configuración - {st.session_state.user_name}")
    func_input = st.text_input("Función f(x1, x2)", value="x1^2 + x2^2")
    start_point = st.text_input("Punto inicial (x1, x2)", value="2.0, 2.0")
    
    method = st.selectbox("Selecciona el método de optimización:", 
                          ["Método del Gradiente", "Método del Gradiente Conjugado", "Método de Newton"])
    
    col1, col2 = st.columns(2)
    alpha = col1.number_input("Paso Fijo (α)", value=0.1, format="%.4f")
    max_iter = col2.number_input("Iteraciones", value=50)
    tol = col1.number_input("Tolerancia", value=1e-5, format="%.1e")
    
    if st.button("Ejecutar"):
        vars_sym = sp.symbols('x1 x2')
        expr = parse_function(func_input, vars_sym)
        x0 = [float(i.strip()) for i in start_point.split(',')]
        if expr:
            st.session_state.results = optimize_wrapper(method, expr, vars_sym, x0, tol, max_iter, alpha)
            st.session_state.expr = expr
            st.session_state.vars_sym = vars_sym
            st.session_state.page = "results"
            st.rerun()

# Página 3: Resultados
elif st.session_state.page == "results":
    st.title("📊 Resultados")
    if st.button("Volver"):
        st.session_state.page = "config"
        st.rerun()
        
    df = st.session_state.results
    st.dataframe(df)
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots()
        ax.plot(df['Iteración'], df['||∇f||'], marker='o')
        ax.set_yscale('log')
        ax.set_title("Convergencia")
        st.pyplot(fig)
        
    with col_b:
        f_lambda = sp.lambdify(st.session_state.vars_sym, st.session_state.expr, 'numpy')
        X, Y = np.meshgrid(np.linspace(df['x1'].min()-2, df['x1'].max()+2, 30), 
                           np.linspace(df['x2'].min()-2, df['x2'].max()+2, 30))
        Z = f_lambda(X, Y)
        fig2, ax2 = plt.subplots()
        ax2.contour(X, Y, Z, levels=15)
        ax2.plot(df['x1'], df['x2'], 'r-x')
        st.pyplot(fig2)
        
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
    ax3.plot(df['x1'], df['x2'], df['f(x)'], 'r-o')
    st.pyplot(fig3)
