import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import re

# Configuración de página
st.set_page_config(page_title="Optimizador Web Pro", layout="wide")

# Lógica matemática robusta
def parse_function(func_str, vars_list):
    try:
        func_str = func_str.replace('^', '**')
        # Soporte para multiplicación implícita
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        expr = sp.sympify(func_str)
        # Asegurar que solo usa las variables permitidas
        if not expr.free_symbols.issubset(set(vars_list)):
            return None
        return expr
    except:
        return None

def get_gradient(expr, vars_list):
    return [sp.diff(expr, v) for v in vars_list]

def optimize_manual(expr, vars_list, x0, tol, max_iter, alpha):
    grad_exprs = get_gradient(expr, vars_list)
    x = np.array(x0, dtype=float)
    history = []
    
    for i in range(max_iter):
        subs = {v: x[j] for j, v in enumerate(vars_list)}
        f_val = float(expr.subs(subs))
        g_val = np.array([float(g.subs(subs)) for g in grad_exprs])
        norm_g = np.linalg.norm(g_val)
        
        history.append({'Iteración': i, 'f(x)': f_val, '||∇f||': norm_g, 'x': x.copy()})
        
        if norm_g < tol: break
        x = x - alpha * g_val
    return pd.DataFrame(history)

# Navegación
if 'page' not in st.session_state: st.session_state.page = "login"

if st.session_state.page == "login":
    st.title("👤 Registro")
    name = st.text_input("Nombre:")
    if st.button("Ingresar") and name:
        st.session_state.user_name = name
        st.session_state.page = "config"
        st.rerun()

elif st.session_state.page == "config":
    st.title(f"Configuración - {st.session_state.user_name}")
    func_input = st.text_input("Función f(x1, x2)", "x1**2 + x2**2")
    x0_input = st.text_input("Punto inicial (x1, x2)", "1.0, 1.0")
    alpha = st.number_input("Paso (alpha)", 0.01)
    
    if st.button("Ejecutar"):
        vars_list = sp.symbols('x1 x2')
        expr = parse_function(func_input, vars_list)
        x0 = [float(i) for i in x0_input.split(',')]
        
        if expr:
            st.session_state.df = optimize_manual(expr, vars_list, x0, 1e-6, 100, alpha)
            st.session_state.page = "results"
            st.rerun()
        else:
            st.error("Error en la sintaxis. Asegúrate de usar x1 y x2.")

elif st.session_state.page == "results":
    st.title("Resultados")
    st.dataframe(st.session_state.df)
    if st.button("Volver"):
        st.session_state.page = "config"
        st.rerun()
