import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Calculadora Optimizadora", layout="wide", initial_sidebar_state="expanded")

# --- Inyección de CSS (Tu diseño original conservado) ---
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
    # Permite elegir la norma requerida en problemas académicos
    if norm_type == "L_infinito (Máximo)":
        num = np.linalg.norm(curr_x - prev_x, ord=np.inf)
        den = np.linalg.norm(prev_x, ord=np.inf)
    elif norm_type == "L1 (Manhattan)":
        num = np.linalg.norm(curr_x - prev_x, ord=1)
        den = np.linalg.norm(prev_x, ord=1)
    else: # L2 Euclidiana
        num = np.linalg.norm(curr_x - prev_x)
        den = np.linalg.norm(prev_x)
    return num / (den if den != 0 else 1e-8)

def evaluate_func_safe(f_lambdified, x, vars_sym):
    # Protege contra evaluación de log(0) o raíces negativas usuales en estos problemas
    try:
        val = f_lambdified(*x) if len(vars_sym) > 1 else f_lambdified(x[0])
        if np.isnan(val) or np.isinf(val):
            return 1e9 # Penalización alta si sale del dominio
        return val
    except Exception:
        return 1e9

# --- Algoritmos Modificados para retornar logs detallados ---
def run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, max_iter, tol, norm_type):
    history = []
    backtrack_log = [] # Para guardar el detalle del backtracking (Ítem a)
    
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

        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val), 'Error Rel. (%)': rel_error * 100}
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
                
                # Búsqueda de línea con registro
                for intento in range(50):
                    new_x = curr_x + alpha * direction
                    f_new = evaluate_func_safe(f_lambdified, new_x, vars_sym)
                    
                    limite_armijo = f_val + c1 * alpha * np.dot(grad_val, direction)
                    armijo_cumple = f_new <= limite_armijo
                    
                    # Guardar log solo de las primeras iteraciones para reporte
                    if k < 3: 
                        backtrack_log.append({
                            'Iteración k': k, 'Intento': intento+1, 'Alfa (α)': alpha, 
                            'f(x + αd)': f_new, 'Cota Armijo': limite_armijo, 'Cumple': armijo_cumple
                        })
                    
                    if not armijo_cumple:
                        alpha *= rho 
                    else:
                        break 
            
            curr_x = curr_x - alpha * grad_val
            
    return pd.DataFrame(history), pd.DataFrame(backtrack_log), grad_exprs

# --- Interfaces de Usuario ---
def main_app():
    with st.sidebar:
        st.write("👤 Usuario: **Invitado**")
        st.markdown("<hr style='margin: 12px 0; border-color: #CBD5E1;'>", unsafe_allow_html=True)
        st.markdown("### 💡 Diccionario de Métodos")
        st.markdown("""<div class="method-card"><strong>📉 Método del Gradiente</strong><span>Útil para encontrar mínimos locales moviéndose en la dirección del gradiente negativo.</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="method-card"><strong>🚀 Búsqueda de Línea (Armijo)</strong><span>Ajusta dinámicamente el tamaño del paso para garantizar un descenso suficiente (Primera Condición de Wolfe).</span></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="instructions-box">
        <h4>📖 Resuelve tus Guías de Estudio</h4>
        <ol>
            <li><strong>Variables Flexibles:</strong> Escribe <code>x, y</code> o <code>x1, x2</code> según tu problema.</li>
            <li><strong>Manejo de logaritmos:</strong> Usa <code>ln()</code> o <code>log()</code> sin problema.</li>
            <li><strong>Reporte Detallado:</strong> Ideal para copiar el paso a paso en tus pruebas (incluyendo Backtracking y Condiciones de Wolfe).</li>
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
        func_input = st.text_input(f"Función f({', '.join(vars_names)})", value="ln(x**2 + y**2) - 2*x*y")
        
    with col3:
        start_point = st.text_input("Punto inicial (x0, y0)", value="-1, 0")

    st.markdown("### 2. Parámetros del Algoritmo")
    
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    
    with col_m1:
        method = st.selectbox("Método:", ["Método del Gradiente"])
        max_iter = st.number_input("Iteraciones máximas", value=10, min_value=1)
        
    with col_m2:
        alpha_type = st.radio("Cálculo del paso (alfa):", ["Fijo", "Wolfe (Armijo)"], index=1, horizontal=True)
        if alpha_type == "Fijo":
            alpha_val = st.number_input("Valor de alfa:", value=0.01, format="%.4f")
        else:
            alpha_val = 0.0 # Placeholder
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
        show_exam_mode = st.checkbox("🔍 Mostrar Desglose Modo Examen", value=True)

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
                
                # --- Ejecución del Algoritmo ---
                if method == "Método del Gradiente":
                    results, bt_log, grad_exprs = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params if alpha_type=="Wolfe (Armijo)" else None, int(max_iter), tolerancia, norm_type)
                
                # --- MODO EXAMEN: DESGLOSE PASO A PASO ---
                if show_exam_mode:
                    st.info("📚 **Desglose Académico (Iteración 0 a 1)**")
                    
                    st.markdown("**1. Gradiente Analítico:**")
                    grad_latex = [rf"\frac{{\partial f}}{{\partial {v}}} = {sp.latex(g)}" for v, g in zip(vars_names, grad_exprs)]
                    st.latex(r"\nabla f = \begin{bmatrix} " + r" \\ ".join(grad_latex) + r" \end{bmatrix}")
                    
                    x0_str = ", ".join([f"{v:.4f}" for v in x0])
                    grad_x0_str = ", ".join([f"{results.iloc[0][f'g_{v}']:.4f}" for v in vars_sym])
                    st.markdown(f"**Evaluación inicial:** $\\nabla f({x0_str}) = [{grad_x0_str}]^T$")
                    
                    # Análisis del inciso (a)
                    if alpha_type == "Wolfe (Armijo)" and not bt_log.empty:
                        st.markdown("**2. Búsqueda de Línea Backtracking (Resolución inciso a):**")
                        st.dataframe(bt_log[bt_log['Iteración k'] == 0].drop(columns=['Iteración k']), use_container_width=True)
                        alpha_final = bt_log[bt_log['Iteración k'] == 0].iloc[-1]['Alfa (α)']
                        st.success(f"**Paso aceptado:** $\\alpha_0 = {alpha_final}$")
                    
                    # Análisis del inciso (b)
                    if len(results) > 1:
                        st.markdown(f"**3. Error Relativo (Resolución inciso b):**")
                        err_0 = results.iloc[1]['Error Rel. (%)']
                        st.markdown(f"Usando la norma seleccionada ({norm_type}), el error en la primera iteración es: **{err_0:.4f}%**")
                    
                    # Análisis del inciso (c) - Condición de Wolfe
                    if len(results) > 1 and alpha_type == "Wolfe (Armijo)":
                        st.markdown("**4. Verificación de 2da Condición de Wolfe (Curvatura) (Resolución inciso c):**")
                        st.latex(r"\nabla C(\mathbf{x}^{(k)} + \alpha_m \mathbf{d}^{(k)})^T \mathbf{d}^{(k)} \ge \sigma \nabla C(\mathbf{x}^{(k)})^T \mathbf{d}^{(k)}")
                        
                        g_k = np.array([results.iloc[0][f'g_{v}'] for v in vars_sym])
                        d_k = -g_k
                        g_k1 = np.array([results.iloc[1][f'g_{v}'] for v in vars_sym])
                        
                        lado_izq = np.dot(g_k1, d_k)
                        lado_der = wolfe_sigma * np.dot(g_k, d_k)
                        
                        st.latex(rf"{lado_izq:.6f} \ge {wolfe_sigma} \times ({np.dot(g_k, d_k):.6f})")
                        st.latex(rf"{lado_izq:.6f} \ge {lado_der:.6f}")
                        
                        if lado_izq >= lado_der:
                            st.success("✅ **La segunda condición de Wolfe SÍ se cumple.**")
                        else:
                            st.error("❌ **La segunda condición de Wolfe NO se cumple.** (Es común que Armijo simple no la satisfaga sin cálculo de curvatura estricto).")
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                
                # --- Tabla General ---
                st.markdown("#### Tabla del Historial de Iteraciones General")
                st.dataframe(results, use_container_width=True)
                
            else:
                st.error("Error: Asegúrate que la cantidad de valores en el punto inicial coincida con las variables.")
        except Exception as e:
            st.error(f"Se encontró un error matemático/sintáctico: {e}")

if __name__ == "__main__":
    main_app()
