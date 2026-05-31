import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(page_title="Calculadora Optimizadora", layout="wide", initial_sidebar_state="expanded")

# --- Inyección de CSS para Personalización de Estilo Avanzada ---
st.markdown(
    """
    <style>
    /* (Se mantiene todo tu CSS original intacto) */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #F0F7FF !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background-color: #E0F2FE !important; border-right: 1px solid #CBD5E1 !important; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family: 'Times New Roman', Times, serif !important; }
    p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p { font-family: 'Times New Roman', Times, serif !important; font-size: 16px !important; color: #1E293B !important;  }
    h1, h2, h3, h4 { font-family: 'Times New Roman', Times, serif !important; color: #0F172A !important; font-weight: bold !important; }
    h1 { font-size: 32px !important; margin-bottom: 15px !important; }
    h2 { font-size: 24px !important; margin-bottom: 12px !important; }
    h3 { font-size: 20px !important; margin-top: 20px !important;}
    h4 { font-size: 18px !important; margin-bottom: 10px !important;}
    [data-testid="stForm"], .stFormCreator { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 12px !important; padding: 24px !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important; }
    .instructions-box { background-color: #FFFFFF; border-left: 5px solid #1E3A8A; padding: 20px 25px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 25px; margin-top: 10px; }
    .instructions-box ol { margin-bottom: 0; padding-left: 20px; }
    .instructions-box li { font-size: 15.5px !important; margin-bottom: 6px; color: #334155 !important; }
    .instructions-box code { background-color: #F1F5F9; color: #0F172A; padding: 2px 6px; border-radius: 4px; font-family: monospace !important; }
    .method-card { background-color: #FFFFFF !important; padding: 14px; border-radius: 8px; border: 1px solid #CBD5E1; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02); }
    .method-card strong { color: #1E3A8A !important; font-size: 15px !important; display: block; margin-bottom: 4px; }
    .method-card span { font-size: 13.5px !important; color: #334155 !important; line-height: 1.3 !important; display: block; }
    input, select, textarea, [data-baseweb="select"], [data-baseweb="input"] { background-color: #FFFFFF !important; color: #0F172A !important; font-family: 'Times New Roman', Times, serif !important; font-size: 16px !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; }
    [data-testid="stWidgetLabel"] *, input *, select * { color: #0F172A !important; }
    div[data-baseweb="popover"], div[data-baseweb="popover"] *, div[role="listbox"], div[role="listbox"] *, ul[role="listbox"], ul[role="listbox"] *, li[role="option"], li[role="option"] * { background-color: #FFFFFF !important; color: #0F172A !important; font-family: 'Times New Roman', Times, serif !important; }
    li[role="option"]:hover, li[role="option"]:hover *, div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li:hover * { background-color: #E0F2FE !important; color: #1E3A8A !important; }
    .stButton > button, [data-testid="stForm"] button, button[kind="primaryFormSubmit"] { background-color: #1E3A8A !important; color: #FFFFFF !important; font-family: 'Times New Roman', Times, serif !important; font-size: 16px !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; padding: 8px 24px !important; width: 100% !important; transition: all 0.2s ease-in-out !important; box-shadow: 0 2px 4px rgba(30, 58, 138, 0.2) !important; }
    .stButton > button *, [data-testid="stForm"] button * { color: #FFFFFF !important; }
    .stButton > button:hover, [data-testid="stForm"] button:hover, button[kind="primaryFormSubmit"]:hover { background-color: #1D4ED8 !important; color: #FFFFFF !important; transform: translateY(-1px) !important; box-shadow: 0 4px 6px rgba(30, 58, 138, 0.3) !important; cursor: pointer; }
    .stButton > button:hover *, [data-testid="stForm"] button:hover * { color: #FFFFFF !important; }
    [data-testid="stDataFrame"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 10px !important; padding: 10px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important; }
    [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Funciones Matemáticas ---
def parse_function(func_str, vars_list):
    try:
        # MEJORA: Traducciones comunes para cálculo
        func_str = func_str.replace('^', '**')
        func_str = func_str.replace('ln', 'log') # Sympy usa log para logaritmo natural
        # Convertir e^algo o e**algo a exp(algo) de forma básica
        func_str = re.sub(r'e\*\*\((.*?)\)', r'exp(\1)', func_str)
        func_str = re.sub(r'e\*\*(.*?)(\s|\+|-|\*|\/|$)', r'exp(\1)\2', func_str)
        
        # Reemplazar números seguidos directamente de letras (ej: 4x1 -> 4*x1)
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        expr = sp.sympify(func_str)
        return expr
    except Exception as e:
        st.error(f"Error al analizar la función: {e}")
        return None

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    return sp.hessian(expr, variables)

# MEJORA: Agregado parámetro de tolerancia (tol)
def run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, max_iter, tol=1e-3):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    
    curr_x = np.array(x0, dtype=float)
    
    for k in range(max_iter + 1):
        f_val = f_lambdified(*curr_x) if len(vars_sym) > 1 else f_lambdified(curr_x[0])
        grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
        
        # Calcular Error Relativo para inciso (iii)
        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'x_{i+1}'] for i in range(len(curr_x))])
            # Precisión relativa: ||x_new - x_old|| / ||x_old||
            denom = np.linalg.norm(prev_x)
            rel_error = np.linalg.norm(curr_x - prev_x) / (denom if denom != 0 else 1e-8)

        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val), 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x):
            entry[f'x_{i+1}'] = val
        for i, val in enumerate(grad_val):
            entry[f'g_{i+1}'] = val
        history.append(entry)
        
        # Criterio de Parada por Precisión Relativa (Inciso iii)
        if k > 0 and rel_error < tol:
            break
            
        if k < max_iter:
            if alpha_type == "Fijo":
                alpha = alpha_val
            else:
                alpha = wolfe_params['alpha_init']
                c1 = wolfe_params['c1']
                rho = wolfe_params['rho']
                direction = -grad_val
                for _ in range(50):
                    new_x = curr_x + alpha * direction
                    f_new = f_lambdified(*new_x) if len(vars_sym) > 1 else f_lambdified(new_x[0])
                    armijo_cumple = f_new <= (f_val + c1 * alpha * np.dot(grad_val, direction))
                    if not armijo_cumple:
                        alpha *= rho 
                    else:
                        break 
            
            curr_x = curr_x - alpha * grad_val
            
    return pd.DataFrame(history)

def run_new_ton_method(expr, vars_sym, x0, max_iter, tol=1e-3):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    hess_expr = compute_hessian(expr, variables=vars_sym)
    grad_lambdified = sp.lambdify(vars_sym, grad_exprs, 'numpy')
    hess_lambdified = sp.lambdify(vars_sym, hess_expr, 'numpy')
    curr_x = np.array(x0, dtype=float)
    for k in range(max_iter + 1):
        f_val = f_lambdified(*curr_x) if len(vars_sym) > 1 else f_lambdified(curr_x[0])
        grad_val = np.array(grad_lambdified(*curr_x)) if len(vars_sym) > 1 else np.array([grad_lambdified(curr_x[0])])
        hess_val = np.array(hess_lambdified(*curr_x)) if len(vars_sym) > 1 else np.array([[hess_lambdified(curr_x[0])]])
        
        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'x_{i+1}'] for i in range(len(curr_x))])
            rel_error = np.linalg.norm(curr_x - prev_x) / (np.linalg.norm(prev_x) + 1e-8)
            
        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val), 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x): entry[f'x_{i+1}'] = val
        history.append(entry)
        
        if k > 0 and rel_error < tol: break
        
        if k < max_iter:
            try: curr_x = curr_x - np.linalg.inv(hess_val).dot(grad_val)
            except: break
    return pd.DataFrame(history)

def run_conjugate_gradient(expr, vars_sym, x0, alpha_type, alpha_val, max_iter, tol=1e-3):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    curr_x = np.array(x0, dtype=float)
    grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
    p = -grad_val.copy()
    
    for k in range(max_iter + 1):
        f_val = f_lambdified(*curr_x) if len(vars_sym) > 1 else f_lambdified(curr_x[0])
        grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
        
        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'x_{i+1}'] for i in range(len(curr_x))])
            rel_error = np.linalg.norm(curr_x - prev_x) / (np.linalg.norm(prev_x) + 1e-8)
            
        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val), 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x): entry[f'x_{i+1}'] = val
        history.append(entry)
        
        if k > 0 and (rel_error < tol or np.linalg.norm(grad_val) < 1e-6): break
        
        if k < max_iter:
            if np.dot(grad_val, p) >= 0: p = -grad_val
            if alpha_type == "Fijo": alpha = alpha_val
            else:
                alpha = alpha_val 
                for _ in range(50):
                    new_x = curr_x + alpha * p
                    f_new = f_lambdified(*new_x) if len(vars_sym) > 1 else f_lambdified(new_x[0])
                    if f_new <= f_val + 1e-4 * alpha * np.dot(grad_val, p): break
                    alpha *= 0.5
            next_x = curr_x + alpha * p
            grad_next_val = np.array([g(*next_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](next_x[0])])
            denom = np.dot(grad_val, grad_val)
            beta = 0.0 if denom < 1e-12 else np.dot(grad_next_val, grad_next_val) / denom
            p = -grad_next_val + beta * p
            curr_x = next_x
    return pd.DataFrame(history)

# --- Interfaces de Usuario ---
def login_page():
    st.title("Bienvenido al Optimizador Web 🚀")
    with st.form("login_form"):
        username = st.text_input("Nombre de usuario:")
        submitted = st.form_submit_button("Ingresar")
        if submitted and username.strip() != "":
            st.session_state['username'] = username
            st.rerun()

def main_app():
    with st.sidebar:
        st.write(f"👤 Usuario: **{st.session_state['username']}**")
        st.markdown("<hr style='margin: 12px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        st.markdown("### 💡 Diccionario de Métodos")
        st.markdown("<div class='method-card'><strong>📉 Método del Gradiente</strong><span>Paso a paso hacia la menor inclinación.</span></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 12px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        if st.button("Cerrar sesión"):
            st.session_state.pop('username')
            st.rerun()

    st.markdown("""
    <div class="instructions-box">
        <h4>📖 Para resolver tu problema:</h4>
        <ol>
            <li><strong>Inciso a)</strong> Usa <code>log(1+x1**2) + cos(3*x1)</code> (Sympy usa log para ln).</li>
            <li><strong>Inciso b)</strong> Usa <code>exp(-x1**2) + 0.2*x1**4 - x1</code>.</li>
            <li>En "Precisión Relativa", usa <strong>0.001</strong> para que se detenga al 0.1%.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.title("⚙️ Calculadora de Optimización")
    
    st.markdown("### 1. Definición del Problema")
    col1, col2, col3 = st.columns([1, 2, 1]) 
    
    with col1:
        n_vars = st.number_input("Variables (n)", min_value=1, max_value=10, value=1)
        vars_names = [f"x{i+1}" for i in range(n_vars)]
    with col2:
        func_input = st.text_input(f"Función f({', '.join(vars_names)})", value="log(1+x1**2) + cos(3*x1)")
    with col3:
        start_point = st.text_input("Punto inicial (x0)", value="0.5")

    st.markdown("### 2. Parámetros del Algoritmo")
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        method = st.selectbox("Método de optimización:", ["Método del Gradiente", "Método de Newton", "Método del Gradiente Conjugado"])
        # MEJORA: Ingreso de la tolerancia exigida en el problema
        col_m1_1, col_m1_2 = st.columns(2)
        with col_m1_1:
            max_iter = st.number_input("Iteraciones máximas", value=100, min_value=1)
        with col_m1_2:
            tolerancia = st.number_input("Tolerancia / Precisión", value=0.001, format="%.4f", help="Ej: 0.001 = 0.1% de error relativo")
    
    alpha_type = "Fijo"
    alpha_val = 0.05
    wolfe_params = {'alpha_init': 1.0, 'c1': 1e-4, 'rho': 0.5, 'use_curvature': False, 'theta': 0.9}
    
    with col_m2:
        if method == "Método del Gradiente":
            alpha_type = st.radio("Tamaño de paso (alfa):", ["Fijo", "Wolfe (Armijo)"], horizontal=True)
            if alpha_type == "Fijo":
                alpha_val = st.number_input("Valor de alfa:", value=0.05, format="%.4f")
            else:
                wolfe_params['alpha_init'] = st.number_input("Alfa inicial:", value=1.0)
        else:
            st.info("Configuraciones avanzadas automáticas para este método.")

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        execute = st.button("▶ Ejecutar Optimización")

    # --- EJECUCIÓN ---
    if execute:
        st.markdown("### 3. Resultados y Gráficos")
        vars_sym = sp.symbols(' '.join(vars_names))
        if n_vars == 1: vars_sym = [vars_sym]
        expr = parse_function(func_input, vars_sym)
        
        try:
            clean_str = re.sub(r'[^0-9.,-]', '', start_point)
            x0 = [float(i) for i in clean_str.split(',') if i.strip()]
            
            if expr is not None and len(x0) == n_vars:
                grad_exprs = compute_gradient(expr, vars_sym)
                
                # --- RECUPERADO: Mostrar Gradiente Analítico y Evaluación Inicial ---
                st.markdown("#### Fórmulas Analíticas Calculadas:")
                st.latex(r"f(" + ", ".join(vars_names) + r") = " + sp.latex(expr))
                
                grad_latex_elements = [rf"\frac{{\partial f}}{{\partial {v}}} = {sp.latex(g)}" for v, g in zip(vars_names, grad_exprs)]
                st.latex(r"\nabla f = \begin{bmatrix} " + r" \\ ".join(grad_latex_elements) + r" \end{bmatrix}")
                
                # Evaluar gradiente en el punto inicial x0
                grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
                curr_x = np.array(x0, dtype=float)
                grad_at_x0 = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
                
                st.markdown("#### Evaluación en el Punto Inicial:")
                st.latex(r"x_0 = \begin{bmatrix} " + r" \\ ".join([f"{val:.4f}" for val in x0]) + r" \end{bmatrix}")
                st.latex(r"\nabla f(x_0) = \begin{bmatrix} " + r" \\ ".join([f"{val:.4f}" for val in grad_at_x0]) + r" \end{bmatrix}")
                # ----------------------------------------------------------------------

                # Ejecutar algoritmo
                if method == "Método del Gradiente":
                    results = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, int(max_iter), tol=tolerancia)
                elif method == "Método de Newton":
                    results = run_new_ton_method(expr, vars_sym, x0, int(max_iter), tol=tolerancia)
                else:
                    results = run_conjugate_gradient(expr, vars_sym, x0, "Fijo", alpha_val, int(max_iter), tol=tolerancia)
                
                st.success(f"Optimización completada. Se alcanzó el criterio de parada en {len(results)-1} iteraciones.")
                
                # --- GRÁFICOS (Incisos i, iv y v) ---
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("#### (i) y (iv) Trayectoria en f(x)")
                    if n_vars == 1:
                        # Crear el gráfico en 1D
                        f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
                        x_hist = results['x_1'].values
                        f_hist = results['f(x)'].values
                        
                        # Definir rango dinámico para el eje X
                        margin = max(1.0, (max(x_hist) - min(x_hist)) * 0.5)
                        x_range = np.linspace(min(x_hist) - margin, max(x_hist) + margin, 500)
                        y_range = [f_lambdified(val) for val in x_range]
                        
                        fig = go.Figure()
                        # Grafico de la función
                        fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='f(x)', line=dict(color='#1E3A8A', width=2)))
                        # Puntos de las iteraciones
                        fig.add_trace(go.Scatter(x=x_hist, y=f_hist, mode='markers+lines', name='Iteraciones', 
                                                 marker=dict(color='#EF4444', size=8, symbol='circle'),
                                                 line=dict(color='#EF4444', width=1, dash='dot')))
                        fig.update_layout(title="Comportamiento del Algoritmo", xaxis_title="x", yaxis_title="f(x)", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("La visualización en 2D/3D está disponible solo para funciones de 1 variable en esta versión.")

                with col_g2:
                    st.markdown("#### (v) Análisis de Convergencia")
                    # Gráfico para analizar convergencia (Error Relativo vs Iteración)
                    iter_vals = results['Iteración'].values[1:] # Omitimos la 0 para error relativo
                    err_vals = results['Error Rel. (%)'].values[1:]
                    
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(x=iter_vals, y=err_vals, mode='lines+markers', name='Error (%)', line=dict(color='#10B981')))
                    fig_conv.update_layout(title="Evolución del Error Relativo", xaxis_title="Iteración (k)", yaxis_title="Error Relativo (%)", yaxis_type="log", template="plotly_white")
                    st.plotly_chart(fig_conv, use_container_width=True)

                st.markdown("#### Tabla del Historial de Iteraciones")
                st.dataframe(results, use_container_width=True)
            else:
                st.error("Error: Revisa que el punto inicial tenga la misma cantidad de variables.")
        except Exception as e:
            st.error(f"Se encontró un error al procesar los datos: {e}")

if 'username' not in st.session_state: login_page()
else: main_app()
