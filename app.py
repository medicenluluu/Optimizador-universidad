import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re
import matplotlib.pyplot as plt

# 1. Configuración de página
st.set_page_config(page_title="Calculadora de Optimización", layout="wide", initial_sidebar_state="expanded")

# --- BLOQUE DE LOGIN ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.title("⚙️ Calculadora de Optimización Académica")
    st.markdown("### Bienvenido")
    st.write("Por favor, ingresa tu nombre para comenzar a usar la calculadora.")
    
    name_input = st.text_input("Tu nombre:")
    if st.button("Entrar a la aplicación"):
        if name_input:
            st.session_state.user_name = name_input
            st.rerun() 
        else:
            st.warning("Por favor, escribe un nombre primero.")
    st.stop() # Detiene la ejecución hasta que el usuario se identifique
# --- FIN BLOQUE DE LOGIN ---

# 2. Inyección de CSS (Diseño conservado)
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
    .stButton > button:hover *, [data-testid="stButton"] button:hover * { color: #FFFFFF !important; }
    [data-testid="stDataFrame"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 10px !important; padding: 10px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important; }
    [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True

)

# 3. Resto de tu lógica de la aplicación
with st.sidebar:
    st.write(f"👤 Usuario: {st.session_state.user_name}")

# --- Funciones de Matemáticas ---
def parse_function(func_str, vars_list):
    try:
        func_str = func_str.replace('^', '**')
        func_str = func_str.replace('ln', 'log')
        func_str = re.sub(r'e\*\*\((.*?)\)', r'exp(\1)', func_str)
        func_str = re.sub(r'e\*\*(.*?)(\s|\+|-|\*|\/|$)', r'exp(\1)\2', func_str)
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        expr = sp.sympify(func_str)
        return expr
    except Exception:
        return None

def compute_gradient(expr, variables):
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    return sp.hessian(expr, variables)

def calcular_error(curr_x, prev_x, norm_type):
    if norm_type == "L_infinito (Máximo)":
        num = np.linalg.norm(curr_x - prev_x, ord=np.inf)
        den = np.linalg.norm(curr_x, ord=np.inf)
    elif norm_type == "L1 (Manhattan)":
        num = np.linalg.norm(curr_x - prev_x, ord=1)
        den = np.linalg.norm(curr_x, ord=1)
    else: 
        num = np.linalg.norm(curr_x - prev_x)
        den = np.linalg.norm(curr_x)
    return num / (den if den != 0 else 1e-8)

def evaluate_func_safe(f_lambdified, x, vars_sym):
    try:
        val = f_lambdified(*x) if len(vars_sym) > 1 else f_lambdified(x[0])
        if np.isnan(val) or np.isinf(val):
            return 1e9
        return val
    except Exception:
        return 1e9

# --- Algoritmos de Optimización ---
def run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, max_iter, tol, norm_type):
    history = []
    backtrack_log = [] 
    
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
        for i, val in enumerate(curr_x):
            entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val):
            entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)
        
        if k > 0 and rel_error < tol:
            break
        
        if k < max_iter:
            direction = -grad_val
            alpha = alpha_val
            
            if alpha_type == "Wolfe (Armijo)":
                alpha = wolfe_params['alpha_init']
                c1 = wolfe_params['c1']
                rho = wolfe_params['rho']
                
                for intento in range(50):
                    new_x = curr_x + alpha * direction
                    f_new = evaluate_func_safe(f_lambdified, new_x, vars_sym)
                    
                    limite_armijo = f_val + c1 * alpha * np.dot(grad_val, direction)
                    armijo_cumple = f_new <= limite_armijo
                    
                    if k < 3: 
                        backtrack_log.append({
                            'Iteración k': k, 'Intento': intento+1, 'Alfa (α)': alpha, 
                            'C(x + αd)': f_new, 'Cota de Armijo': limite_armijo, 'Cumple': armijo_cumple
                        })
                    
                    if not armijo_cumple:
                        alpha *= rho 
                    else:
                        break 
            
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
        for i, val in enumerate(curr_x):
            entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val):
            entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)
        
        if k > 0 and rel_error < tol: 
            break
            
        if k < max_iter:
            try:
                curr_x = curr_x - np.linalg.inv(hess_val).dot(grad_val)
            except np.linalg.LinAlgError:
                break # Evitar singularidad
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
        for i, val in enumerate(curr_x):
            entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val):
            entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)
        
        if k > 0 and (rel_error < tol or norm_grad < 1e-6):
            break 
            
        if k < max_iter:
            if np.dot(grad_val, p) >= 0:
                p = -grad_val
                
            if alpha_type == "Fijo":
                alpha = alpha_val
            else:
                alpha = alpha_val 
                c1 = 1e-4
                rho = 0.5
                for _ in range(50):
                    new_x = curr_x + alpha * p
                    f_new = evaluate_func_safe(f_lambdified, new_x, vars_sym)
                    if f_new <= f_val + c1 * alpha * np.dot(grad_val, p):
                        break
                    alpha *= rho
                
            next_x = curr_x + alpha * p
            grad_next_val = np.array([g(*next_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](next_x[0])])
            
            denom = np.dot(grad_val, grad_val)
            if denom < 1e-12:
                beta = 0.0
            else:
                beta = np.dot(grad_next_val, grad_next_val) / denom
                
            p = -grad_next_val + beta * p
            curr_x = next_x
            
    return pd.DataFrame(history)

# --- Interfaz de Usuario ---
def main_app():
    with st.sidebar:
        st.markdown("<hr style='margin: 12px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        st.markdown("### 💡 Diccionario de Métodos")
        st.markdown("""<div class="method-card"><strong>📉 Método del Gradiente</strong><span>Fácil de usar para buscar mínimos locales moviéndose en la dirección del gradiente negativo.</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="method-card"><strong>🚀 Método de Newton</strong><span>Utiliza la pendiente y la curvatura (Hessiana) para determinar un paso ultra rápido, ideal para soluciones cercanas al óptimo.</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="method-card"><strong>🎯 Gradiente Conjugado</strong><span>Optimiza usando direcciones ortogonales. Evita repetir caminos explorados para avanzar con gran precisión.</span></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="instructions-box">
        <h4>📖 Resuelve tus Guías de Estudio</h4>
        <ol>
            <li><strong>Variables Flexibles:</strong> Escribe <code>x, y</code> o <code>x1, x2</code> dependiendo de tu problema.</li>
            <li><strong>Manejo de logaritmos:</strong> Usa <code>ln()</code> o <code>log()</code> sin problema.</li>
            <li><strong>Reporte Detallado:</strong> Selecciona Método del Gradiente + Armijo para ver el paso a paso exacto como en los exámenes universitarios.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.title("⚙️ Calculadora de Optimización Académica")
    
    st.markdown("### 1. Definición del Problema")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        vars_input = st.text_input("Variables (separadas por coma)", value="x, y")
        vars_names = [v.strip() for v in vars_input.split(',')]
    
    with col2:
        func_input = st.text_input(f"Función C({', '.join(vars_names)})", value="ln(x**2 + y**2) - 2*x*y")
        
    with col3:
        start_point = st.text_input("Punto inicial (x0, y0)", value="-1, 0")

    st.markdown("### 2. Parámetros del Algoritmo")
    
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    
    with col_m1:
        method = st.selectbox("Método de optimización:", ["Método del Gradiente", "Método de Newton", "Método del Gradiente Conjugado"])
        max_iter = st.number_input("Iteraciones máximas", value=10, min_value=1)
        
    with col_m2:
        if method == "Método de Newton":
            st.info("El Método de Newton calcula su propio paso dinámicamente con la Matriz Hessiana.")
            alpha_type = "Newton"
        elif method == "Método del Gradiente Conjugado":
            alpha_type = st.radio("Cálculo del tamaño de paso (alfa) para GC:", ["Fijo", "Búsqueda de línea (Armijo)"], horizontal=True)
            if alpha_type == "Fijo":
                cg_alpha_val = st.number_input("Valor de alfa (GC):", value=0.01, format="%.4f")
            else:
                cg_alpha_val = st.number_input("Alfa inicial para búsqueda (GC):", value=1.0, format="%.4f")
        else: # Método del Gradiente
            alpha_type = st.radio("Cálculo del paso (alfa):", ["Fijo", "Wolfe (Armijo)"], index=1, horizontal=True)
            if alpha_type == "Fijo":
                alpha_val = st.number_input("Valor de alfa:", value=0.01, format="%.4f")
            else:
                alpha_val = 0.0 
                wolfe_params = {}
                w_c1, w_c2 = st.columns(2)
                with w_c1:
                    wolfe_params['alpha_init'] = st.number_input("Alfa inicial (α_0):", value=0.5, format="%.4f")
                    wolfe_params['rho'] = st.number_input("Rho (ρ - reducción):", value=0.5, format="%.4f")
                with w_c2:
                    wolfe_params['c1'] = st.number_input("Beta (β - Armijo):", value=0.25, format="%.4f")
                    wolfe_sigma = st.number_input("Sigma (σ - Curvatura):", value=0.5, format="%.4f", help="Para la 2da condición de Wolfe")
                
    with col_m3:
        tolerancia = st.number_input("Tolerancia", value=0.001, format="%.4f")
        norm_type = st.selectbox("Norma para Error Relativo:", ["L_infinito (Máximo)", "L2 (Euclidiana)", "L1 (Manhattan)"])
        if method == "Método del Gradiente":
            show_exam_mode = st.checkbox("🔍 Mostrar Detalles en Modo Examen", value=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        execute = st.button("▶ Resolver Problema")

    if execute:
        st.markdown("### 3. Resultados y Análisis")
        vars_sym = sp.symbols(' '.join(vars_names))
        if len(vars_names) == 1: vars_sym = [vars_sym]
        expr = parse_function(func_input, vars_sym)
        
        try:
            clean_str = re.sub(r'[^0-9.,-]', '', start_point)
            x0 = [float(i) for i in clean_str.split(',') if i.strip()]
            
            if expr is not None and len(x0) == len(vars_names):
                
                # Ejecutar algoritmo
                if method == "Método del Gradiente":
                    results, bt_log, grad_exprs = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params if alpha_type=="Wolfe (Armijo)" else None, int(max_iter), tolerancia, norm_type)
                elif method == "Método de Newton":
                    results = run_newton_method(expr, vars_sym, x0, int(max_iter), tolerancia, norm_type)
                else:
                    results = run_conjugate_gradient(expr, vars_sym, x0, alpha_type, cg_alpha_val if alpha_type == "Fijo" else cg_alpha_val, int(max_iter), tolerancia, norm_type)

                # --- MODO EXAMEN (Solo si es Gradiente) ---
                if method == "Método del Gradiente" and show_exam_mode:
                    st.info("📚 **Detalle Académico (Iteración 0 a 1)**")
                    
                    st.markdown("### (a) Una iteración de descenso por gradiente con Backtracking")
                    
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
                        st.markdown(f"**2. Probar $\\alpha_0 = {a0}$**")
                        
                        cand_0 = x0 + a0 * d0_vals
                        cand_0_str = ", ".join([f"{v:g}" for v in cand_0])
                        cand_0_tuple = f"({cand_0_str})" if len(cand_0) > 1 else f"{cand_0[0]:g}"
                        
                        st.markdown("Punto candidato:")
                        st.latex(rf"x^{{(0)}} + \alpha_0 d^{{(0)}} = {x0_tuple} + {a0}{d0_tuple} = {cand_0_tuple}")
                        
                        f_cand_0 = row_0['C(x + αd)']
                        st.markdown("Valor de la función:")
                        st.latex(rf"C{x0_tuple} = {f0_val:g}")
                        st.latex(rf"C{cand_0_tuple} = {f_cand_0:g}")
                        
                        st.markdown("**3. Condición de Armijo**")
                        c1_val = wolfe_params['c1']
                        
                        if c1_val == 0.25:
                            beta_tex = r"\frac14"
                        elif c1_val == 0.5:
                            beta_tex = r"\frac12"
                        else:
                            beta_tex = f"{c1_val:g}"

                        st.markdown(f"Con $\\beta = {beta_tex}$")
                        st.latex(r"C(x^{(0)} + \alpha d^{(0)}) \le C(x^{(0)}) + \beta \alpha \nabla C(x^{(0)})^T d^{(0)}")
                        
                        st.markdown("Calculamos")
                        st.latex(rf"\nabla C(x^{{(0)}})^T d^{{(0)}} = {g0_tuple} \cdot {d0_tuple} = {dot_g0_d0:g}")
                        
                        rhs_0 = f0_val + c1_val * a0 * dot_g0_d0
                        st.markdown("Entonces el lado derecho vale")
                        st.latex(rf"{f0_val:g} + {beta_tex}({a0})({dot_g0_d0:g}) = {rhs_0:g}")
                        
                        st.markdown("Debemos verificar")
                        st.latex(rf"{f_cand_0:g} \le {rhs_0:g},")
                        
                        if row_0['Cumple']:
                            st.markdown("lo cual es verdadero.")
                            st.markdown("Por tanto el paso aceptado es")
                            st.latex(rf"\boxed{{\alpha_1 = {a0}}}")
                            st.markdown("y el nuevo iterado es")
                            st.latex(rf"\boxed{{x^{{(1)}} = {cand_0_tuple}}}")
                        else:
                            st.markdown("lo cual es falso.\nPor tanto, se rechaza $\\alpha = " + str(a0) + "$.")
                            
                            st.markdown("**4. Backtracking**")
                            rho_val = wolfe_params['rho']
                            
                            for i in range(1, len(bt_log)):
                                row_i = bt_log.iloc[i]
                                a_i = row_i['Alfa (α)']
                                cand_i = x0 + a_i * d0_vals
                                cand_i_str = ", ".join([f"{v:g}" for v in cand_i])
                                cand_i_tuple = f"({cand_i_str})" if len(cand_i) > 1 else f"{cand_i[0]:g}"
                                
                                prev_a = bt_log.iloc[i-1]['Alfa (α)']
                                st.latex(rf"\alpha = {rho_val}({prev_a}) = {a_i}.")
                                st.markdown("Nuevo punto:")
                                st.latex(rf"{x0_tuple} + {a_i}{d0_tuple} = {cand_i_tuple}.")
                                
                                f_cand_i = row_i['C(x + αd)']
                                st.markdown("Valor de la función:")
                                st.latex(rf"C{cand_i_tuple} = {f_cand_i:g}")
                                
                                rhs_i = f0_val + c1_val * a_i * dot_g0_d0
                                st.markdown("Condición de Armijo:")
                                st.latex(rf"{f0_val:g} + {beta_tex}({a_i})({dot_g0_d0:g}) = {rhs_i:g}.")
                                
                                st.markdown("Verificación:")
                                st.latex(rf"{f_cand_i:g} \le {rhs_i:g},")
                                
                                if row_i['Cumple']:
                                    st.markdown("verdadero.")
                                    st.markdown("Por tanto el paso aceptado es")
                                    st.latex(rf"\boxed{{\alpha_1 = {a_i}}}.")
                                    st.markdown("y el nuevo iterado es")
                                    st.latex(rf"\boxed{{x^{{(1)}} = {cand_i_tuple}}}.")
                                    break
                                else:
                                    st.markdown("falso.")
                                    
                    if len(results) > 1:
                        st.markdown(f"### (b) Error porcentual de la primera iteración usando $\\| \cdot \\|_{{\\text{{{norm_type.split(' ')[0]}}}}}$")
                        
                        # Corrección aquí: Extraemos la parte del símbolo de norma de forma segura
                        norm_symbol = norm_type.split('_')[0].replace("L", "")
                        
                        st.markdown("Usamos el error relativo aproximado")
                        st.latex(
                            r"E = \frac{\|x^{(1)} - x^{(0)}\|_{" + norm_symbol + r"}}{\|x^{(1)}\|_{" + norm_symbol + r"}} \times 100\%"
                        )

                        x1_vals = np.array([results.iloc[1][f'{v}'] for v in vars_sym])
                        diff_vals = x1_vals - x0
                        diff_str = ", ".join([f"{v:g}" for v in diff_vals])
                        diff_tuple = f"({diff_str})" if len(diff_vals) > 1 else f"{diff_vals[0]:g}"
                        x1_str = ", ".join([f"{v:g}" for v in x1_vals])
                        x1_tuple = f"({x1_str})" if len(x1_vals) > 1 else f"{x1_vals[0]:g}"

                        st.markdown("Calculamos")
                        st.latex(rf"x^{{(1)}} - x^{{(0)}} = {x1_tuple} - {x0_tuple} = {diff_tuple}")

                        ord_val = np.inf if "infinito" in norm_type else (1 if "L1" in norm_type else 2)
                        num_val = np.linalg.norm(diff_vals, ord=ord_val)
                        den_val = np.linalg.norm(x1_vals, ord=ord_val)

                        st.markdown("Entonces")
                        st.latex(rf"\|x^{{(1)}} - x^{{(0)}}\|_{{{norm_symbol}}} = {num_val:g}")
                        st.markdown("Además,")
                        st.latex(rf"\|x^{{(1)}}\|_{{{norm_symbol}}} = {den_val:g}")

                        E_val = (num_val / den_val) * 100 if den_val != 0 else 0
                        st.markdown("Por tanto")
                        st.latex(rf"E = \frac{{{num_val:g}}}{{{den_val:g}}}\times100 = {E_val:g}\%")
                        st.latex(rf"\boxed{{E = {E_val:g}\%}}")
                
                # --- GRÁFICOS ---
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("#### (i) y (iv) Trayectoria en C(x)")
                    if len(vars_names) == 1:
                        f_lambdified_plot = sp.lambdify(vars_sym, expr, 'numpy')
                        x_hist = results[f'{vars_names[0]}'].values
                        f_hist = results['C(x)'].values
                        margin = max(1.0, (max(x_hist) - min(x_hist)) * 0.5)
                        x_range = np.linspace(min(x_hist) - margin, max(x_hist) + margin, 500)
                        y_range = [f_lambdified_plot(val) for val in x_range]
                        
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.plot(x_range, y_range, label='C(x)', color='#1E3A8A', linewidth=2)
                        ax.plot(x_hist, f_hist, label='Iteraciones', color='#EF4444', marker='o', linestyle=':', markersize=6)
                        ax.set_title("Comportamiento del Algoritmo")
                        ax.set_xlabel(f"{vars_names[0]}")
                        ax.set_ylabel("C(x)")
                        ax.grid(True, linestyle='--', alpha=0.6)
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("La gráfica de trayectoria en 2D está disponible solo para funciones de 1 variable.")

                with col_g2:
                    st.markdown("#### (v) Análisis de Convergencia")
                    if len(results) > 1:
                        iter_vals = results['Iteración'].values[1:] 
                        err_vals = results['Error Rel. (%)'].values[1:]
                        fig_conv, ax_conv = plt.subplots(figsize=(7, 5))
                        ax_conv.plot(iter_vals, err_vals, label='Error (%)', color='#10B981', marker='s', linestyle='-')
                        ax_conv.set_title("Evolución del Error Relativo")
                        ax_conv.set_xlabel("Iteración (k)")
                        ax_conv.set_ylabel("Error Relativo (%)")
                        ax_conv.set_yscale('log')
                        ax_conv.grid(True, linestyle='--', alpha=0.6)
                        ax_conv.legend()
                        st.pyplot(fig_conv, use_container_width=True)
                    else:
                        st.info("El algoritmo convergió en el primer intento.")

                st.markdown("#### Tabla General del Historial de Iteraciones")
                st.dataframe(results, use_container_width=True)
                
            else:
                st.error("Error: Asegúrate que la cantidad de valores en el punto inicial coincida con las variables.")
        except Exception as e:
            st.error(f"Se encontró un error matemático/sintáctico: {e}")

if __name__ == "__main__":
    main_app()
