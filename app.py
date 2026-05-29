import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re

# Configuración de página
st.set_page_config(page_title="Calculadora Optimizadora", layout="wide")

# --- Inyección de CSS para Personalización de Estilo Avanzada ---
st.markdown(
    """
    <style>
    /* 1. Fondo general de la aplicación */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F0F7FF !important; /* Baby Blue optimizado, más limpio y luminoso */
    }
    
    /* 2. Fondo de la barra lateral (sidebar) */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #E0F2FE !important; /* Contraste sutil con el fondo principal */
        border-right: 1px solid #CBD5E1 !important;
    }

    /* 3. Tipografía Times New Roman para la estructura base */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* 4. Textos Generales, Etiquetas y Párrafos */
    p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important;
        color: #1E293B !important; 
    }
    
    /* 5. Títulos con jerarquía visual fuerte y color destacado */
    h1, h2, h3, h4 {
        font-family: 'Times New Roman', Times, serif !important;
        color: #0F172A !important;
        font-weight: bold !important;
    }
    h1 { font-size: 32px !important; margin-bottom: 15px !important; }
    h2 { font-size: 24px !important; margin-bottom: 12px !important; }
    h3 { font-size: 20px !important; margin-top: 20px !important;}
    h4 { font-size: 18px !important; margin-bottom: 10px !important;}

    /* 6. Diseño tipo "Tarjeta" para formularios y bloques de entrada */
    [data-testid="stForm"], .stFormCreator {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }

    /* NUEVO: Caja de Instrucciones Elegante */
    .instructions-box {
        background-color: #FFFFFF;
        border-left: 5px solid #1E3A8A; /* Línea decorativa lateral */
        padding: 20px 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        margin-top: 10px;
    }
    .instructions-box ol {
        margin-bottom: 0;
        padding-left: 20px;
    }
    .instructions-box li {
        font-size: 15.5px !important;
        margin-bottom: 6px;
        color: #334155 !important;
    }
    .instructions-box code {
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace !important;
    }

    /* 7. Inputs de texto, números y selectores */
    input, select, textarea, [data-baseweb="select"], [data-baseweb="input"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important; 
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    [data-testid="stWidgetLabel"] *, input *, select * {
        color: #0F172A !important;
    }

    /* 8. Botones espectaculares con alto contraste */
    .stButton > button, [data-testid="stForm"] button, button[kind="primaryFormSubmit"] {
        background-color: #1E3A8A !important; 
        color: #FFFFFF !important; 
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        width: 100% !important; /* Adaptado al ancho de columna */
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 4px rgba(30, 58, 138, 0.2) !important;
    }

    .stButton > button *, [data-testid="stForm"] button * {
        color: #FFFFFF !important;
    }

    .stButton > button:hover, [data-testid="stForm"] button:hover, button[kind="primaryFormSubmit"]:hover {
        background-color: #1D4ED8 !important; 
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px rgba(30, 58, 138, 0.3) !important;
        cursor: pointer;
    }
    
    .stButton > button:hover *, [data-testid="stForm"] button:hover * {
        color: #FFFFFF !important;
    }

    /* 9. Tablas y Dataframes limpios y elegantes */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }

    /* 10. Corrección iconos barra lateral */
    [data-testid="collapsedControl"] *, 
    [data-testid="stSidebarCollapseButton"] button * {
        font-size: 0 !important;
        color: transparent !important;
    }
    [data-testid="collapsedControl"]::after {
        content: "→" !important; font-size: 22px !important; color: #1E293B !important; font-family: 'Times New Roman', Times, serif !important; font-weight: bold !important; display: block !important; text-align: center !important;
    }
    [data-testid="stSidebarCollapseButton"] button::after {
        content: "←" !important; font-size: 22px !important; color: #1E293B !important; font-family: 'Times New Roman', Times, serif !important; font-weight: bold !important; display: block !important; text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Funciones Matemáticas ---
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

def run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, max_iter):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    
    curr_x = np.array(x0, dtype=float)
    
    for k in range(max_iter + 1):
        f_val = f_lambdified(*curr_x) if len(vars_sym) > 1 else f_lambdified(curr_x[0])
        grad_val = np.array([g(*curr_x) for g in grad_lambdified])
        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val)}
        for i, val in enumerate(curr_x):
            entry[f'x_{i+1}'] = val
        history.append(entry)
        
        if k < max_iter:
            if alpha_type == "Fijo":
                alpha = alpha_val
            else:
                alpha = wolfe_params['alpha_init']
                c1 = wolfe_params['c1']
                rho = wolfe_params['rho']
                use_curv = wolfe_params.get('use_curvature', False)
                theta = wolfe_params.get('theta', 0.9)
                
                direction = -grad_val
                
                for _ in range(50):
                    new_x = curr_x + alpha * direction
                    f_new = f_lambdified(*new_x) if len(vars_sym) > 1 else f_lambdified(new_x[0])
                    
                    armijo_cumple = f_new <= (f_val + c1 * alpha * np.dot(grad_val, direction))
                    
                    if not armijo_cumple:
                        alpha *= rho 
                    else:
                        if use_curv:
                            grad_new_val = np.array([g(*new_x) for g in grad_lambdified])
                            curv_cumple = np.dot(grad_new_val, direction) >= theta * np.dot(grad_val, direction)
                            
                            if not curv_cumple:
                                alpha *= 1.5 
                            else:
                                break 
                        else:
                            break 
            
            curr_x = curr_x - alpha * grad_val
            
    return pd.DataFrame(history)

def run_newton_method(expr, vars_sym, x0, max_iter):
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
        
        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val)}
        for i, val in enumerate(curr_x):
            entry[f'x_{i+1}'] = val
        history.append(entry)
        
        if k < max_iter:
            try:
                curr_x = curr_x - np.linalg.inv(hess_val).dot(grad_val)
            except:
                break
    return pd.DataFrame(history)

def run_conjugate_gradient(expr, vars_sym, x0, max_iter):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]
    
    curr_x = np.array(x0, dtype=float)
    r = -np.array([g(*curr_x) for g in grad_lambdified])
    p = r.copy()
    
    for k in range(max_iter + 1):
        f_val = f_lambdified(*curr_x) if len(vars_sym) > 1 else f_lambdified(curr_x[0])
        grad_val = -r 
        entry = {'Iteración': k, 'f(x)': f_val, '||∇f(x)||': np.linalg.norm(grad_val)}
        for i, val in enumerate(curr_x):
            entry[f'x_{i+1}'] = val
        history.append(entry)
        
        if k < max_iter:
            alpha = 0.01 
            curr_x = curr_x + alpha * p
            new_r = -np.array([g(*curr_x) for g in grad_lambdified])
            beta = np.dot(new_r, new_r) / np.dot(r, r)
            p = new_r + beta * p
            r = new_r
    return pd.DataFrame(history)


# --- Interfaces de Usuario ---
def login_page():
    st.title("Bienvenido al Optimizador Web 🚀")
    st.markdown("Por favor, ingresa tu nombre de usuario para continuar.")
    
    with st.form("login_form"):
        username = st.text_input("Nombre de usuario:")
        submitted = st.form_submit_button("Ingresar")
        
        if submitted:
            if username.strip() != "":
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("Por favor, ingresa un nombre válido.")

def main_app():
    # Barra lateral
    with st.sidebar:
        st.write(f"👤 Usuario: **{st.session_state['username']}**")
        if st.button("Cerrar sesión"):
            st.session_state.pop('username')
            st.rerun()

    # --- NUEVO: Cuadro de Instrucciones ---
    st.markdown("""
    <div class="instructions-box">
        <h4>📖 Guía Paso a Paso</h4>
        <ol>
            <li><strong>Configura el problema:</strong> Indica cuántas variables tiene tu función.</li>
            <li><strong>Ingresa tu función:</strong> Usa variables como <code>x1</code>, <code>x2</code>. Ej: <code>x1**4 - 3*x1**3 + 2</code>.</li>
            <li><strong>Punto inicial:</strong> Define desde dónde arranca el algoritmo (separado por comas si hay más de 1 variable).</li>
            <li><strong>Ajusta el algoritmo:</strong> Selecciona el método, su tolerancia (alfa) y las iteraciones máximas.</li>
            <li><strong>Visualiza:</strong> Clic en <em>Ejecutar</em> para ver la evolución matemática iteración por iteración.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.title("⚙️ Calculadora")
    
    # --- REDUCCIÓN DE TAMAÑOS (Uso de Columnas) ---
    st.markdown("### 1. Definición del Problema")
    
    col1, col2, col3 = st.columns([1, 2, 1]) # Columnas proporcionadas
    
    with col1:
        n_vars = st.number_input("Variables (n)", min_value=1, max_value=10, value=1)
        vars_names = [f"x{i+1}" for i in range(n_vars)]
    
    with col2:
        func_input = st.text_input(f"Función f({', '.join(vars_names)})", value="x1**4 - 3*x1**3 + 2")
        
    with col3:
        start_point = st.text_input("Punto inicial (x0)", value="0.5")

    st.markdown("### 2. Parámetros del Algoritmo")
    
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        method = st.selectbox("Método de optimización:", ["Método del Gradiente", "Método de Newton", "Método del Gradiente Conjugado"])
        max_iter = st.number_input("Número de iteraciones", value=10, min_value=1)
    
    alpha_type = "Fijo"
    alpha_val = 0.01
    wolfe_params = {'alpha_init': 1.0, 'c1': 1e-4, 'rho': 0.5, 'use_curvature': False, 'theta': 0.9}

    with col_m2:
        if method == "Método del Gradiente":
            alpha_type = st.radio("Cálculo del tamaño de paso (alfa):", ["Fijo", "Wolfe (Armijo)"], horizontal=True)
            
            if alpha_type == "Fijo":
                alpha_val = st.number_input("Valor de alfa:", value=0.01, format="%.4f")
            elif alpha_type == "Wolfe (Armijo)":
                c_w1, c_w2 = st.columns(2)
                with c_w1:
                    wolfe_params['alpha_init'] = st.number_input("Alfa inicial:", value=1.0, format="%.4f")
                    wolfe_params['rho'] = st.number_input("Rho (reducción):", value=0.5, format="%.4f")
                with c_w2:
                    wolfe_params['c1'] = st.number_input("C1 (Armijo):", value=1e-4, format="%.4e")
                    calc_curv = st.selectbox("¿Condición curvatura?", ["No", "Sí"])
                    
                if calc_curv == "Sí":
                    wolfe_params['use_curvature'] = True
                    wolfe_params['theta'] = st.number_input("Theta:", value=0.9, format="%.4f")
        else:
            st.info(f"El {method} no requiere configuración adicional de tamaño de paso aquí.")

    st.markdown("<br>", unsafe_allow_html=True) # Espaciado estético

    # Botón centrado usando columnas vacías
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        execute = st.button("▶ Ejecutar Optimización")

    # --- EJECUCIÓN ---
    if execute:
        st.markdown("### 3. Resultados Iterativos")
        vars_sym = sp.symbols(' '.join(vars_names))
        if n_vars == 1: vars_sym = [vars_sym]
        expr = parse_function(func_input, vars_sym)
        
        try:
            clean_str = re.sub(r'[^0-9.,-]', '', start_point)
            x0 = [float(i) for i in clean_str.split(',') if i.strip()]
            
            if expr is not None and len(x0) == n_vars:
                if method == "Método del Gradiente":
                    results = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, int(max_iter))
                elif method == "Método de Newton":
                    results = run_newton_method(expr, vars_sym, x0, int(max_iter))
                else:
                    results = run_conjugate_gradient(expr, vars_sym, x0, int(max_iter))
                
                st.success("Cálculo completado exitosamente.")
                st.dataframe(results, use_container_width=True)
            else:
                st.error("Error en las dimensiones o la función. Revisa que el punto inicial tenga la misma cantidad de variables.")
        except Exception as e:
            st.error(f"Se encontró un error al procesar los datos: {e}")

# Manejo de estado de la aplicación
if 'username' not in st.session_state:
    login_page()
else:
    main_app()
