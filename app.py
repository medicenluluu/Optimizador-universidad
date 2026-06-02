import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re
import plotly.graph_objects as go

# 1. Configuración de página
st.set_page_config(page_title="Calculadora de Optimización", layout="wide", initial_sidebar_state="expanded")

# --- BLOQUE DE LOGIN ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A; font-family: \"Times New Roman\", Times, serif;'>⚙️ Calculadora de Optimización Académica</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-family: \"Times New Roman\", Times, serif;'>Bienvenido</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-family: \"Times New Roman\", Times, serif;'>Por favor, ingresa tu nombre para comenzar a usar la calculadora.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name_input = st.text_input("Tu nombre:")
        if st.button("Entrar a la aplicación", use_container_width=True):
            if name_input.strip():
                st.session_state.user_name = name_input.strip()
                st.rerun() 
            else:
                st.warning("Por favor, escribe un nombre primero.")
    st.stop()
# --- FIN BLOQUE DE LOGIN ---

# 2. Inyección de CSS (Diseño Académico Conservado + Potenciado)
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background-color: #F0F7FF !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background-color: #E0F2FE !important; border-right: 1px solid #CBD5E1 !important; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family: 'Times New Roman', Times, serif !important; }
    p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p { font-family: 'Times New Roman', Times, serif !important; font-size: 16px !important; color: #1E293B !important; }
    h1, h2, h3, h4 { font-family: 'Times New Roman', Times, serif !important; color: #0F172A !important; font-weight: bold !important; }
    h1 { font-size: 32px !important; margin-bottom: 15px !important; }
    h2 { font-size: 24px !important; margin-bottom: 12px !important; }
    h3 { font-size: 20px !important; margin-top: 20px !important;}
    h4 { font-size: 18px !important; margin-bottom: 10px !important;}
    
    .instructions-box { background-color: #FFFFFF; border-left: 5px solid #1E3A8A; padding: 20px 25px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 25px; margin-top: 10px; }
    .instructions-box ol { margin-bottom: 0; padding-left: 20px; }
    .instructions-box li { font-size: 15.5px !important; margin-bottom: 6px; color: #334155 !important; }
    .method-card { background-color: #FFFFFF !important; padding: 14px; border-radius: 8px; border: 1px solid #CBD5E1; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02); }
    .method-card strong { color: #1E3A8A !important; font-size: 15px !important; display: block; margin-bottom: 4px; }
    .method-card span { font-size: 13.5px !important; color: #334155 !important; line-height: 1.3 !important; display: block; }
    
    .stButton > button, [data-testid="stForm"] button { background-color: #1E3A8A !important; color: #FFFFFF !important; font-family: 'Times New Roman', Times, serif !important; font-size: 16px !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; padding: 8px 24px !important; transition: all 0.2s ease-in-out !important; box-shadow: 0 2px 4px rgba(30, 58, 138, 0.2) !important; }
    .stButton > button:hover { background-color: #1D4ED8 !important; transform: translateY(-1px) !important; box-shadow: 0 4px 6px rgba(30, 58, 138, 0.3) !important; cursor: pointer; }
    
    .metric-card { background: #FFFFFF; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #CBD5E1; text-align: center;}
    .metric-title { font-size: 14px; color: #64748B; text-transform: uppercase; font-weight: bold;}
    .metric-value { font-size: 24px; color: #1E3A8A; font-weight: bold;}
    .insight-box { background-color: #F8FAFC; border: 1px solid #38BDF8; border-left: 6px solid #0284C7; padding: 20px; border-radius: 8px; margin-top: 15px; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .insight-box h4 { margin-top: 0; color: #0369A1 !important; display: flex; align-items: center; gap: 8px; }
    .insight-box ul { margin-bottom: 0; }
    .insight-box li { margin-bottom: 8px; }
    
    [data-testid="stDataFrame"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 10px !important; padding: 10px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important; }
    
    /* Estilo para las opciones de radio del Sidebar (haciéndolas parecer menús) */
    div.row-widget.stRadio > div { background-color: transparent; }
    div.row-widget.stRadio > div > label { font-size: 18px !important; font-weight: bold !important; padding: 10px 0 !important; color: #1E3A8A !important; border-bottom: 1px solid #CBD5E1; cursor: pointer; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Funciones de Matemáticas ---
def parse_function(func_str, vars_list):
    try:
        func_str = func_str.replace('^', '**').replace('ln', 'log')
        func_str = re.sub(r'e\*\*\((.*?)\)', r'exp(\1)', func_str)
        func_str = re.sub(r'e\*\*(.*?)(\s|\+|-|\*|\/|$)', r'exp(\1)\2', func_str)
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        return sp.sympify(func_str)
    except Exception:
        return None

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    return sp.hessian(expr, variables)

def calcular_error(curr_x, prev_x, norm_type):
    if norm_type == "L_infinito (Máximo)":
        num, den = np.linalg.norm(curr_x - prev_x, ord=np.inf), np.linalg.norm(curr_x, ord=np.inf)
    elif norm_type == "L1 (Manhattan)":
        num, den = np.linalg.norm(curr_x - prev_x, ord=1), np.linalg.norm(curr_x, ord=1)
    else: 
        num, den = np.linalg.norm(curr_x - prev_x), np.linalg.norm(curr_x)
    return num / (den if den != 0 else 1e-8)

def evaluate_func_safe(f_lambdified, x, vars_sym):
    try:
        val = f_lambdified(*x) if len(vars_sym) > 1 else f_lambdified(x[0])
        return 1e9 if np.isnan(val) or np.isinf(val) else val
    except Exception:
        return 1e9

# --- Algoritmos de Optimización ---
def run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, max_iter, tol, norm_type):
    history, backtrack_log = [], []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    
    curr_x = np.array(x0, dtype=float)
    for k in range(max_iter + 1):
        f_val = evaluate_func_safe(f_lambdified, curr_x, vars_sym)
        grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
        
        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'{v}'] for v in vars_sym])
            rel_error = calcular_error(curr_x, prev_x, norm_type)

        entry = {'Iteración': k, 'C(x)': f_val, '||∇C(x)||': np.linalg.norm(grad_val), 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x): entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val): entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)
        
        if k > 0 and rel_error < tol: break
        if k == max_iter: break
        
        direction = -grad_val
        alpha = alpha_val
        if alpha_type == "Wolfe (Armijo)":
            alpha, c1, rho = wolfe_params['alpha_init'], wolfe_params['c1'], wolfe_params['rho']
            for intento in range(50):
                new_x = curr_x + alpha * direction
                f_new = evaluate_func_safe(f_lambdified, new_x, vars_sym)
                limite_armijo = f_val + c1 * alpha * np.dot(grad_val, direction)
                armijo_cumple = f_new <= limite_armijo
                if k < 3: 
                    backtrack_log.append({'Iteración k': k, 'Intento': intento+1, 'Alfa (α)': alpha, 'C(x + αd)': f_new, 'Cota de Armijo': limite_armijo, 'Cumple': armijo_cumple})
                if not armijo_cumple: alpha *= rho 
                else: break 
        curr_x = curr_x - alpha * grad_val
    return pd.DataFrame(history), pd.DataFrame(backtrack_log), grad_exprs

def run_newton_method(expr, vars_sym, x0, max_iter, tol, norm_type):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    hess_expr = compute_hessian(expr, variables=vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    hess_lambdified = sp.lambdify(vars_sym, hess_expr, 'numpy')
    
    curr_x = np.array(x0, dtype=float)
    for k in range(max_iter + 1):
        f_val = evaluate_func_safe(f_lambdified, curr_x, vars_sym)
        grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
        try:
            hess_val = np.array(hess_lambdified(*curr_x), dtype=float) if len(vars_sym) > 1 else np.array([[hess_lambdified(curr_x[0])]], dtype=float)
        except:
            hess_val = np.eye(len(vars_sym))
        
        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'{v}'] for v in vars_sym])
            rel_error = calcular_error(curr_x, prev_x, norm_type)
            
        entry = {'Iteración': k, 'C(x)': f_val, '||∇C(x)||': np.linalg.norm(grad_val), 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x): entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val): entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)
        
        if k > 0 and rel_error < tol: break
        if k < max_iter:
            try: curr_x = curr_x - np.linalg.inv(hess_val).dot(grad_val)
            except np.linalg.LinAlgError: break 
    return pd.DataFrame(history)

def run_conjugate_gradient(expr, vars_sym, x0, alpha_type, alpha_val, max_iter, tol, norm_type):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    
    curr_x = np.array(x0, dtype=float)
    grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
    p = -grad_val.copy()
    
    for k in range(max_iter + 1):
        f_val = evaluate_func_safe(f_lambdified, curr_x, vars_sym)
        grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
        norm_grad = np.linalg.norm(grad_val)
        
        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'{v}'] for v in vars_sym])
            rel_error = calcular_error(curr_x, prev_x, norm_type)
            
        entry = {'Iteración': k, 'C(x)': f_val, '||∇C(x)||': norm_grad, 'Error Rel. (%)': rel_error * 100}
        for i, val in enumerate(curr_x): entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val): entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)
        
        if k > 0 and (rel_error < tol or norm_grad < 1e-6): break 
        if k == max_iter: break

        if np.dot(grad_val, p) >= 0: p = -grad_val
            
        alpha = alpha_val
        if alpha_type != "Fijo":
            c1, rho = 1e-4, 0.5
            for _ in range(50):
                new_x = curr_x + alpha * p
                if evaluate_func_safe(f_lambdified, new_x, vars_sym) <= f_val + c1 * alpha * np.dot(grad_val, p): break
                alpha *= rho
            
        next_x = curr_x + alpha * p
        grad_next_val = np.array([g(*next_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](next_x[0])])
        denom = np.dot(grad_val, grad_val)
        beta = 0.0 if denom < 1e-12 else np.dot(grad_next_val, grad_next_val) / denom
        p = -grad_next_val + beta * p
        curr_x = next_x
            
    return pd.DataFrame(history)

# --- Interfaz de Usuario Principal (Múltiples Páginas Simuladas) ---
def main_app():
    # Inicializar el Estado de la Sesión para persistencia entre "Páginas"
    defaults = {
        'page_nav': "1. Problema",
        'vars_input': 'x, y', 'func_input': 'x**2 + y**2 - x*y', 'start_point': '4, 4',
        'method': 'Método del Gradiente', 'max_iter': 20, 
        'alpha_type_gc': 'Fijo', 'alpha_val_gc': 0.1,
        'alpha_type_grad': 'Wolfe (Armijo)', 'alpha_val_grad': 0.1,
        'wolfe_alpha_init': 1.0, 'wolfe_rho': 0.5, 'wolfe_c1': 0.1, 
        'tolerancia': 0.001, 'norm_type': 'L_infinito (Máximo)', 'show_exam_mode': True,
        'calc_results': None, 'calc_bt_log': None, 'calc_grad_exprs': None, 'has_results': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Menú Lateral
    with st.sidebar:
        st.write(f"👤 Usuario: **{st.session_state.user_name}**")
        st.markdown("<hr style='margin: 12px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        
        st.markdown("### 🧭 Menú de Páginas")
        opciones_paginas = ["1. Problema", "2. Parámetros", "3. Resultados"]
        indice_actual = opciones_paginas.index(st.session_state.page_nav)
        
        # Este Radio actúa como el router/enrutador de páginas
        st.session_state.page_nav = st.radio("Navegación", opciones_paginas, index=indice_actual, label_visibility="collapsed")
        
        st.markdown("<hr style='margin: 12px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        st.markdown("""
        <div class="instructions-box" style="padding: 15px; margin-bottom: 15px;">
            <h4 style="font-size: 16px;">📖 Guía Rápida</h4>
            <ol style="padding-left: 15px;">
                <li style="font-size: 14px !important;">Define las <b>Variables</b> y la <b>Función</b>.</li>
                <li style="font-size: 14px !important;">Configura el <b>Algoritmo</b>.</li>
                <li style="font-size: 14px !important;">Analiza los <b>Resultados</b>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Cerrar Sesión"):
            st.session_state.clear()
            st.rerun()

    # Título Fijo
    st.title("⚙️ Calculadora de Optimización Académica")

    # ==========================================
    # PÁGINA 1: DEFINICIÓN DEL PROBLEMA
    # ==========================================
    if st.session_state.page_nav == "1. Problema":
        st.markdown("### Página 1: Definición Matemática del Problema")
        st.info("Ingresa los datos fundamentales de tu problema matemático. Los datos se guardarán al pasar de página.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.session_state.vars_input = st.text_input("Variables (separadas por coma)", value=st.session_state.vars_input)
        with col2:
            st.session_state.func_input = st.text_input("Función Objetivo C(x)", value=st.session_state.func_input)
        with col3:
            st.session_state.start_point = st.text_input("Punto inicial (x0, y0...)", value=st.session_state.start_point)

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("Siguiente: Configurar Parámetros ➡️", use_container_width=True):
                st.session_state.page_nav = "2. Parámetros"
                st.rerun()

    # ==========================================
    # PÁGINA 2: PARÁMETROS DEL ALGORITMO
    # ==========================================
    elif st.session_state.page_nav == "2. Parámetros":
        st.markdown("### Página 2: Configuración del Algoritmo")
        st.info("Selecciona el método de optimización y configura los tamaños de paso. Presiona 'Resolver' cuando estés listo.")
        
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
        with col_m1:
            opciones_metodo = ["Método del Gradiente", "Método de Newton", "Método del Gradiente Conjugado"]
            st.session_state.method = st.selectbox("Método de optimización:", opciones_metodo, index=opciones_metodo.index(st.session_state.method))
            st.session_state.max_iter = st.number_input("Iteraciones máximas", value=st.session_state.max_iter, min_value=1)
        
        with col_m2:
            if st.session_state.method == "Método de Newton":
                st.success("El Método de Newton calcula su propio paso dinámicamente usando la Matriz Hessiana.")
                alpha_type = "Newton"
                alpha_val = None
                wolfe_params = None
            elif st.session_state.method == "Método del Gradiente Conjugado":
                opciones_gc = ["Fijo", "Búsqueda de línea (Armijo)"]
                st.session_state.alpha_type_gc = st.radio("Cálculo de tamaño de paso (α) GC:", opciones_gc, horizontal=True, index=opciones_gc.index(st.session_state.alpha_type_gc))
                st.session_state.alpha_val_gc = st.number_input("Valor de alfa (GC):", value=st.session_state.alpha_val_gc, format="%.4f")
                alpha_type = st.session_state.alpha_type_gc
                alpha_val = st.session_state.alpha_val_gc
                wolfe_params = None
            else:
                opciones_grad = ["Fijo", "Wolfe (Armijo)"]
                st.session_state.alpha_type_grad = st.radio("Cálculo del paso (alfa):", opciones_grad, index=opciones_grad.index(st.session_state.alpha_type_grad), horizontal=True)
                if st.session_state.alpha_type_grad == "Fijo":
                    st.session_state.alpha_val_grad = st.number_input("Valor de alfa:", value=st.session_state.alpha_val_grad, format="%.4f")
                    alpha_type = "Fijo"
                    alpha_val = st.session_state.alpha_val_grad
                    wolfe_params = None
                else:
                    alpha_type = "Wolfe (Armijo)"
                    alpha_val = 0.0 
                    st.session_state.wolfe_alpha_init = st.number_input("Alfa inicial (α_0):", value=st.session_state.wolfe_alpha_init, format="%.4f")
                    st.session_state.wolfe_rho = st.number_input("Rho (ρ - reducción):", value=st.session_state.wolfe_rho, format="%.4f")
                    st.session_state.wolfe_c1 = st.number_input("Beta (β - Armijo):", value=st.session_state.wolfe_c1, format="%.4f")
                    wolfe_params = {'alpha_init': st.session_state.wolfe_alpha_init, 'rho': st.session_state.wolfe_rho, 'c1': st.session_state.wolfe_c1}
        
        with col_m3:
            st.session_state.tolerancia = st.number_input("Tolerancia", value=st.session_state.tolerancia, format="%.4f")
            opciones_norma = ["L_infinito (Máximo)", "L2 (Euclidiana)", "L1 (Manhattan)"]
            st.session_state.norm_type = st.selectbox("Norma para Error Relativo:", opciones_norma, index=opciones_norma.index(st.session_state.norm_type))
            if st.session_state.method == "Método del Gradiente":
                st.session_state.show_exam_mode = st.checkbox("🔍 Mostrar Detalles en Modo Examen", value=st.session_state.show_exam_mode)

        st.markdown("<br><hr style='margin: 12px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        colA, colB, colC = st.columns([1, 1.5, 1])
        with colA:
            if st.button("⬅️ Volver a Problema", use_container_width=True):
                st.session_state.page_nav = "1. Problema"
                st.rerun()
        with colB:
            if st.button("▶ Resolver Problema y Ver Resultados", use_container_width=True):
                # Validar y calcular
                vars_names = [v.strip() for v in st.session_state.vars_input.split(',')]
                vars_sym = sp.symbols(' '.join(vars_names))
                if len(vars_names) == 1: vars_sym = [vars_sym]
                expr = parse_function(st.session_state.func_input, vars_sym)
                
                try:
                    clean_str = re.sub(r'[^0-9.,-]', '', st.session_state.start_point)
                    x0 = [float(i) for i in clean_str.split(',') if i.strip()]
                    
                    if expr is not None and len(x0) == len(vars_names):
                        with st.spinner("Realizando cálculos..."):
                            if st.session_state.method == "Método del Gradiente":
                                results, bt_log, grad_exprs = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, int(st.session_state.max_iter), st.session_state.tolerancia, st.session_state.norm_type)
                            elif st.session_state.method == "Método de Newton":
                                results = run_newton_method(expr, vars_sym, x0, int(st.session_state.max_iter), st.session_state.tolerancia, st.session_state.norm_type)
                                bt_log = pd.DataFrame()
                                grad_exprs = compute_gradient(expr, vars_sym)
                            else:
                                results = run_conjugate_gradient(expr, vars_sym, x0, alpha_type, alpha_val, int(st.session_state.max_iter), st.session_state.tolerancia, st.session_state.norm_type)
                                bt_log = pd.DataFrame()
                                grad_exprs = compute_gradient(expr, vars_sym)

                        # Guardar cálculos en Session State (grad_exprs se guarda como texto para evitar errores de memoria)
                        st.session_state.calc_results = results
                        st.session_state.calc_bt_log = bt_log
                        st.session_state.calc_grad_exprs = [str(g) for g in grad_exprs]
                        st.session_state.has_results = True
                        
                        # Saltar a la página 3
                        st.session_state.page_nav = "3. Resultados"
                        st.rerun()
                    else:
                        st.error("Error de Dimensiones: Asegúrate de que el punto inicial coincida con la cantidad de variables.")
                except Exception as e:
                    st.error(f"Se encontró un error matemático o de sintaxis: {e}")

    # ==========================================
    # PÁGINA 3: RESULTADOS Y ANÁLISIS
    # ==========================================
    elif st.session_state.page_nav == "3. Resultados":
        st.markdown("### Página 3: Resultados y Análisis Finales")
        
        if not st.session_state.has_results:
            st.warning("⚠️ Aún no hay resultados calculados. Por favor, configura el problema y los parámetros primero.")
            colA, colB = st.columns([1, 4])
            with colA:
                if st.button("⬅️ Ir a Parámetros"):
                    st.session_state.page_nav = "2. Parámetros"
                    st.rerun()
        else:
            # Recuperar datos guardados
            results = st.session_state.calc_results
            bt_log = st.session_state.calc_bt_log
            grad_exprs = [sp.sympify(g) for g in st.session_state.calc_grad_exprs]
            
            vars_names = [v.strip() for v in st.session_state.vars_input.split(',')]
            vars_sym = sp.symbols(' '.join(vars_names))
            if len(vars_names) == 1: vars_sym = [vars_sym]
            expr = parse_function(st.session_state.func_input, vars_sym)
            clean_str = re.sub(r'[^0-9.,-]', '', st.session_state.start_point)
            x0 = [float(i) for i in clean_str.split(',') if i.strip()]
            
            # Extraer variables auxiliares
            method = st.session_state.method
            max_iter = st.session_state.max_iter
            norm_type = st.session_state.norm_type
            show_exam_mode = st.session_state.show_exam_mode
            
            if method == "Método del Gradiente":
                alpha_type = st.session_state.alpha_type_grad
            elif method == "Método del Gradiente Conjugado":
                alpha_type = st.session_state.alpha_type_gc
            else:
                alpha_type = "Newton"

            # --- KPIs ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            final_row = results.iloc[-1]
            kpi1.markdown(f"<div class='metric-card'><div class='metric-title'>Iteraciones Tomadas</div><div class='metric-value'>{int(final_row['Iteración'])}</div></div>", unsafe_allow_html=True)
            kpi2.markdown(f"<div class='metric-card'><div class='metric-title'>Valor Final C(x)</div><div class='metric-value'>{final_row['C(x)']:.4f}</div></div>", unsafe_allow_html=True)
            kpi3.markdown(f"<div class='metric-card'><div class='metric-title'>Error Relativo Final</div><div class='metric-value'>{final_row['Error Rel. (%)']:.4f}%</div></div>", unsafe_allow_html=True)
            kpi4.markdown(f"<div class='metric-card'><div class='metric-title'>Norma del Gradiente</div><div class='metric-value'>{final_row['||∇C(x)||']:.4e}</div></div>", unsafe_allow_html=True)
            st.write("")

            # --- MOTOR DE INSIGHTS ---
            final_iter = int(results.iloc[-1]['Iteración'])
            mejor_cx = results.iloc[-1]['C(x)']
            inicial_cx = results.iloc[0]['C(x)']
            mejora_total = inicial_cx - mejor_cx
            grad_final = results.iloc[-1]['||∇C(x)||']
            
            insight_html = "<div class='insight-box'><h4>💡 Análisis de Inteligencia Matemática</h4><ul>"
            if final_iter < max_iter: insight_html += f"<li><b>Velocidad de Convergencia:</b> El método logró converger de manera anticipada en la iteración <b>{final_iter}</b> cumpliendo con la tolerancia. Esto indica que el paso (alfa) y el método seleccionado fueron muy eficientes para este problema.</li>"
            else: insight_html += f"<li><b>Límite Alcanzado:</b> El proceso alcanzó el límite máximo de <b>{max_iter} iteraciones</b> antes de llegar a la tolerancia estricta. Considera aumentar las iteraciones o modificar el tamaño de paso si notas que avanza muy lento.</li>"
            
            if mejora_total > 0: insight_html += f"<li><b>Calidad de Optimización:</b> Excelente. El valor de la función se redujo en <b>{mejora_total:.4f}</b> unidades respecto a tu punto de partida original.</li>"
            elif mejora_total < 0: insight_html += f"<li><b>Advertencia de Divergencia:</b> El valor de C(x) <i>aumentó</i>. Esto suele ocurrir cuando el tamaño de paso es muy grande y el algoritmo 'salta' por encima del valle. ¡Prueba reduciendo el valor de Alfa!</li>"
            
            if grad_final < 1e-2: insight_html += f"<li><b>Confirmación de Óptimo:</b> La norma del gradiente final es sumamente cercana a cero (<code>{grad_final:.2e}</code>). Matemáticamente, esto nos garantiza que estás parado casi exactamente sobre un punto crítico (mínimo local).</li>"
            else: insight_html += f"<li><b>Fuerza del Gradiente:</b> La magnitud del gradiente aún es de <code>{grad_final:.4f}</code>. Significa que la función aún tiene cierta pendiente y no hemos tocado el 'fondo' exacto del valle matemático.</li>"
            insight_html += "</ul></div>"
            st.markdown(insight_html, unsafe_allow_html=True)

            # --- TABS SUB-SECCIONES DE RESULTADOS ---
            tab_graficos, tab_datos, tab_examen = st.tabs(["📈 Gráficos Interactivos", "🗄️ Tabla de Iteraciones", "📚 Modo Examen (Detalle)"])

            with tab_graficos:
                col_plot1, col_plot2 = st.columns(2)
                with col_plot1:
                    st.subheader("Evolución del Error Relativo")
                    if len(results) > 1:
                        fig_err = go.Figure()
                        fig_err.add_trace(go.Scatter(x=results['Iteración'][1:], y=results['Error Rel. (%)'][1:], mode='lines+markers', line=dict(color='#1E3A8A', width=2), marker=dict(color='#EF4444', size=6)))
                        fig_err.update_layout(xaxis_title="Iteración (k)", yaxis_title="Error Relativo (%)", yaxis_type="log", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_err, use_container_width=True)
                    else:
                        st.info("Convergencia alcanzada en el primer intento.")

                with col_plot2:
                    st.subheader("Comportamiento / Trayectoria")
                    if len(vars_names) == 2:
                        f_lamb = sp.lambdify(vars_sym, expr, 'numpy')
                        x_vals = results[f'{vars_names[0]}'].values
                        y_vals = results[f'{vars_names[1]}'].values
                        
                        margin = max(1.0, max(abs(max(x_vals)-min(x_vals)), abs(max(y_vals)-min(y_vals))) * 0.5)
                        x_range = np.linspace(min(x_vals) - margin, max(x_vals) + margin, 100)
                        y_range = np.linspace(min(y_vals) - margin, max(y_vals) + margin, 100)
                        X, Y = np.meshgrid(x_range, y_range)
                        
                        try:
                            Z = f_lamb(X, Y)
                            fig_cont = go.Figure(data=go.Contour(z=Z, x=x_range, y=y_range, colorscale='Blues', opacity=0.7))
                            fig_cont.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', marker=dict(size=8, color='#EF4444'), line=dict(color='#EF4444', width=2), name="Trayectoria"))
                            fig_cont.update_layout(xaxis_title=vars_names[0], yaxis_title=vars_names[1], template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_cont, use_container_width=True)
                        except:
                            st.warning("La función ingresada es demasiado compleja para renderizar el fondo de contorno topográfico de forma segura.")
                    elif len(vars_names) == 1:
                        f_lamb = sp.lambdify(vars_sym, expr, 'numpy')
                        x_vals = results[f'{vars_names[0]}'].values
                        margin = max(1.0, (max(x_vals) - min(x_vals)) * 0.5)
                        x_range = np.linspace(min(x_vals) - margin, max(x_vals) + margin, 200)
                        y_range = [f_lamb(v) for v in x_range]
                        
                        fig_1d = go.Figure()
                        fig_1d.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='C(x)', line=dict(color='#1E3A8A', width=2)))
                        fig_1d.add_trace(go.Scatter(x=x_vals, y=results['C(x)'].values, mode='markers+lines', name='Iteraciones', marker=dict(color='#EF4444', size=8)))
                        fig_1d.update_layout(xaxis_title=f"{vars_names[0]}", yaxis_title="C(x)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_1d, use_container_width=True)
                    else:
                        st.info("La representación espacial gráfica está disponible solo para 1 o 2 variables.")

            with tab_datos:
                st.subheader("Historial Completo")
                st.dataframe(results, use_container_width=True)
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Exportar Resultados a CSV (Excel)", data=csv, file_name='optimizacion_historial.csv', mime='text/csv')

            with tab_examen:
                if method == "Método del Gradiente" and show_exam_mode:
                    st.info("📚 **Detalle Académico (Iteración 0 a 1)**")
                    st.markdown("### (a) Una iteración de descenso por gradiente")
                    
                    st.markdown("**1. Gradiente de $C(" + ", ".join(vars_names) + ")$**")
                    st.latex(r"\nabla C = \left( " + r", \quad ".join([sp.latex(g) for g in grad_exprs]) + r" \right)")
                    
                    x0_str = ", ".join([f"{v:g}" for v in x0])
                    x0_tuple = f"({x0_str})" if len(x0) > 1 else f"{x0[0]:g}"
                    
                    g0_vals = np.array([results.iloc[0][f'g_{v}'] for v in vars_sym])
                    g0_str = ", ".join([f"{v:g}" for v in g0_vals])
                    g0_tuple = f"({g0_str})" if len(g0_vals) > 1 else f"{g0_vals[0]:g}"
                    
                    st.markdown(f"Evaluando en ${x0_tuple}$:")
                    st.latex(rf"\nabla C{x0_tuple} = {g0_tuple}")
                    
                    d0_vals = -g0_vals
                    d0_str = ", ".join([f"{v:g}" for v in d0_vals])
                    d0_tuple = f"({d0_str})" if len(d0_vals) > 1 else f"{d0_vals[0]:g}"
                    
                    st.markdown("La dirección de descenso es")
                    st.latex(rf"d^{{(0)}} = -\nabla C(x^{{(0)}}) = {d0_tuple}")
                    
                    if alpha_type == "Wolfe (Armijo)" and not bt_log.empty:
                        f0_val = results.iloc[0]['C(x)']
                        dot_g0_d0 = np.dot(g0_vals, d0_vals)
                        row_0 = bt_log.iloc[0]
                        a0 = row_0['Alfa (α)']
                        
                        st.markdown(f"**2. Probar $\\alpha_0 = {a0}$ (Condición de Armijo)**")
                        cand_0 = x0 + a0 * d0_vals
                        cand_0_str = ", ".join([f"{v:g}" for v in cand_0])
                        cand_0_tuple = f"({cand_0_str})" if len(cand_0) > 1 else f"{cand_0[0]:g}"
                        
                        st.markdown("Punto candidato:")
                        st.latex(rf"x^{{(0)}} + \alpha_0 d^{{(0)}} = {cand_0_tuple}")
                        
                        f_cand_0 = row_0['C(x + αd)']
                        st.markdown("Valor de la función en el candidato:")
                        st.latex(rf"C{cand_0_tuple} = {f_cand_0:g}")
                        
                        st.markdown("**3. Backtracking (Búsqueda)**")
                        st.dataframe(bt_log, use_container_width=True)
                        
                    if len(results) > 1:
                        st.markdown(f"### (b) Error porcentual de la iteración usando norma")
                        norm_symbol = norm_type.split('_')[0].replace("L", "")
                        
                        st.latex(r"E = \frac{\|x^{(1)} - x^{(0)}\|_{" + norm_symbol + r"}}{\|x^{(1)}\|_{" + norm_symbol + r"}} \times 100\%")
                        x1_vals = np.array([results.iloc[1][f'{v}'] for v in vars_sym])
                        diff_vals = x1_vals - x0
                        
                        ord_val = np.inf if "infinito" in norm_type else (1 if "L1" in norm_type else 2)
                        num_val = np.linalg.norm(diff_vals, ord=ord_val)
                        den_val = np.linalg.norm(x1_vals, ord=ord_val)
                        E_val = (num_val / den_val) * 100 if den_val != 0 else 0
                        
                        st.markdown("Resultado del Error:")
                        st.latex(rf"\boxed{{E = {E_val:g}\%}}")
                else:
                    st.info("Para ver el Modo Examen detallado debes marcar la casilla en los parámetros.")

            st.markdown("<hr>", unsafe_allow_html=True)
            if st.button("🔄 Volver al Inicio (Nueva Optimización)", use_container_width=True):
                st.session_state.page_nav = "1. Problema"
                st.rerun()

if __name__ == "__main__":
    main_app()
