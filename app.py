import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import re

# Configuración inicial de la página
st.set_page_config(page_title="Optimizador Web", layout="wide")

# --- Funciones de Procesamiento ---
def parse_function(func_str, variables):
    try:
        func_str = func_str.replace('^', '**')
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        return sp.sympify(func_str)
    except Exception:
        return None

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def evaluate_func(expr, variables, point):
    subs = {var: val for var, val in zip(variables, point)}
    return float(expr.subs(subs))

def evaluate_grad(grad_exprs, variables, point):
    subs = {var: val for var, val in zip(variables, point)}
    return np.array([float(g.subs(subs)) for g in grad_exprs])

# --- Método del Gradiente (Paso Fijo) ---
def optimize_gradient_descent(expr, vars_sym, x0, tol, max_iter, alpha):
    grad_exprs = compute_gradient(expr, vars_sym)
    x = np.array(x0, dtype=float)
    history = []
    errors = []
    
    for i in range(max_iter):
        f_val = evaluate_func(expr, vars_sym, x)
        g_val = evaluate_grad(grad_exprs, vars_sym, x)
        error = np.linalg.norm(g_val)
        
        history.append({'Iteración': i, 'x': x.copy(), 'f(x)': f_val, '||∇f||': error})
        errors.append(error)
        
        if error < tol:
            break
            
        x = x - alpha * g_val
        
    return pd.DataFrame(history)

# --- Interfaz de Usuario ---
def main():
    if 'user_name' not in st.session_state:
        st.title("👤 Registro de Usuario")
        name = st.text_input("Por favor, ingresa tu nombre:")
        if st.button("Comenzar"):
            if name:
                st.session_state.user_name = name
                st.rerun()
        return

    st.title(f"🚀 Bienvenido {st.session_state.user_name} - Optimizador Matemático")
    
    with st.sidebar:
        st.header("Configuración")
        alpha = st.number_input("Tamaño de paso (α)", value=0.01, format="%.4f")
        max_iter = st.number_input("Iteraciones Máximas", value=100)
        tol = st.number_input("Tolerancia", value=1e-6, format="%.6f")
        if st.button("Cerrar Sesión"):
            del st.session_state.user_name
            st.rerun()

    func_input = st.text_input("Función f(x1, x2)", value="x1^2 + x2^2")
    start_point = st.text_input("Punto inicial (x1, x2)", value="1.0, 1.0")

    if st.button("Ejecutar Optimización"):
        vars_sym = sp.symbols('x1 x2')
        expr = parse_function(func_input, vars_sym)
        x0 = [float(i) for i in start_point.split(',')]
        
        if expr:
            df_results = optimize_gradient_descent(expr, vars_sym, x0, tol, max_iter, alpha)
            
            st.success("¡Cálculo finalizado!")
            st.subheader("Tabla de Iteraciones")
            st.dataframe(df_results.style.format({'f(x)': '{:.6e}', '||∇f||': '{:.6e}'}))
            
            # Gráfica
            fig, ax = plt.subplots()
            ax.plot(df_results['Iteración'], df_results['||∇f||'], marker='o')
            ax.set_yscale('log')
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Norma del Gradiente")
            st.pyplot(fig)
        else:
            st.error("Error en la sintaxis de la función.")

if __name__ == "__main__":
    main()
