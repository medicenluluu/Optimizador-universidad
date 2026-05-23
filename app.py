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
        # Soporte para multiplicación implícita (ej: 3x1 -> 3*x1)
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
    history = [(x.copy(), evaluate_func(expr, vars_sym, x))]
    errors = []
    
    for _ in range(max_iter):
        g_val = evaluate_grad(grad_exprs, vars_sym, x)
        error = np.linalg.norm(g_val)
        errors.append(error)
        
        if error < tol:
            break
            
        x = x - alpha * g_val # Paso fijo: x = x - alpha * gradiente
        history.append((x.copy(), evaluate_func(expr, vars_sym, x)))
        
    return x, evaluate_func(expr, vars_sym, x), len(history), errors[-1] if errors else 0, history, errors

# --- Interfaz de Usuario ---
def main():
    if 'page' not in st.session_state: st.session_state.page = "config"
    
    st.title("🚀 Optimizador Matemático (Paso Fijo)")
    
    with st.sidebar:
        st.header("Configuración")
        method = st.selectbox("Método", ["Gradiente (Paso Fijo)"])
        alpha = st.number_input("Tamaño de paso (α)", value=0.01, format="%.4f")
        max_iter = st.number_input("Iteraciones", value=100)
        tol = st.number_input("Tolerancia", value=1e-6, format="%.6f")

    func_input = st.text_input("Función f(x1, x2)", value="x1^2 + x2^2")
    start_point = st.text_input("Punto inicial (x1, x2)", value="1.0, 1.0")

    if st.button("Ejecutar Optimización"):
        vars_sym = sp.symbols('x1 x2')
        expr = parse_function(func_input, vars_sym)
        x0 = [float(i) for i in start_point.split(',')]
        
        if expr:
            res_x, res_f, res_iter, res_err, _, errors = optimize_gradient_descent(
                expr, vars_sym, x0, tol, max_iter, alpha
            )
            
            st.success("¡Cálculo finalizado!")
            st.write(f"**Mínimo en:** {res_x}")
            st.write(f"**Valor de la función:** {res_f}")
            
            # Gráfica
            fig, ax = plt.subplots()
            ax.plot(errors)
            ax.set_yscale('log')
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Norma del Gradiente")
            st.pyplot(fig)
        else:
            st.error("Error en la sintaxis de la función.")

if __name__ == "__main__":
    main()
