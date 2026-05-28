import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re

# Configuración de página
st.set_page_config(page_title="Calculadora", layout="wide")

# --- Inyección de CSS para Personalización de Estilo Avanzada (Estética Armónica y Profesional) ---
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

    /* 4. Textos Generales, Etiquetas y Párrafos (Times New Roman 16px en Gris Oscuro para elegancia) */
    p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important;
        color: #1E293B !important; /* Gris pizarra muy oscuro para mejor legibilidad que el negro puro */
    }
    
    /* 5. Títulos con jerarquía visual fuerte y color destacado */
    h1, h2, h3 {
        font-family: 'Times New Roman', Times, serif !important;
        color: #0F172A !important; /* Casi negro para máximo contraste */
        font-weight: bold !important;
    }
    h1 { font-size: 32px !important; margin-bottom: 15px !important; }
    h2 { font-size: 24px !important; margin-bottom: 12px !important; }
    h3 { font-size: 20px !important; }

    /* 6. Diseño tipo "Tarjeta" para formularios y bloques de entrada */
    [data-testid="stForm"], .stFormCreator {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }

    /* 7. Inputs de texto, números y selectores (Fondo blanco, texto negro y legibilidad garantizada) */
    input, select, textarea, [data-baseweb="select"], [data-baseweb="input"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important; /* Texto negro legible */
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    /* Asegurar que el texto dentro de los campos activos o deseleccionados sea negro */
    [data-testid="stWidgetLabel"] *, input *, select * {
        color: #0F172A !important;
    }

    /* 8. Botones espectaculares con alto contraste (Azul marino con letras blancas) */
    .stButton > button, [data-testid="stForm"] button, button[kind="primaryFormSubmit"] {
        background-color: #1E3A8A !important; /* Azul marino profundo */
        color: #FFFFFF !important; /* Texto blanco de alto contraste */
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        width: auto !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 4px rgba(30, 58, 138, 0.2) !important;
    }

    /* FORZAR que las letras dentro de los botones sean siempre blancas (solución a tu problema) */
    .stButton > button *, [data-testid="stForm"] button * {
        color: #FFFFFF !important;
    }

    /* Efecto Hover para los botones */
    .stButton > button:hover, [data-testid="stForm"] button:hover, button[kind="primaryFormSubmit"]:hover {
        background-color: #1D4ED8 !important; /* Azul un poco más claro al posar el cursor */
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
    
    /* Pequeño hack para corregir mensajes de alerta */
    [data-testid="stNotification"] {
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Funciones Matemáticas (Mantenidas de tu código original) ---

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
            else: # Condiciones de Wolfe (Armijo y opcionalmente Curvatura)
                alpha = wolfe_params['alpha_init']
                c1 = wolfe_params['c1']
                rho = wolfe_params['rho']
                use_curv = wolfe_params.get('use_curvature', False)
                theta = wolfe_params.get('theta', 0.9)
                
                direction = -grad_val
                
                # Búsqueda de línea con límite de iteraciones para evitar bucles infinitos
                for _ in range(50):
                    new_x = curr_x + alpha * direction
                    f_new = f_lambdified(*new_x) if len(vars_sym) > 1 else f_lambdified(new_x[0])
                    
                    # 1. Condición de Armijo (descenso suficiente)
                    armijo_cumple = f_new <= (f_val + c1 * alpha * np.dot(grad_val, direction))
                    
                    if not armijo_cumple:
                        alpha *= rho # Reducir paso si no cumple Armijo
                    else:
                        if use_curv:
                            # 2. Condición de Curvatura
                            grad_new_val = np.array([g(*new_x) for g in grad_lambdified])
                            curv_cumple = np.dot(grad_new_val, direction) >= theta * np.dot(grad_val, direction)
                            
                            if not curv_cumple:
                                alpha *= 1.5 # Incrementar el paso si es demasiado corto
                            else:
                                break # Cumple ambas condiciones
                        else:
                            break # Solo evaluamos Armijo, así que terminamos aquí
            
            curr_x = curr_x - alpha * grad_val
            
    return pd.DataFrame(history)

def run_newton_method(expr, vars_sym, x0, max_iter):
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    hess_expr = compute_hessian(expr, vars_sym)
    
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

# --- Interfaces de Usuario (Páginas) ---

def login_page():
    st.title("Bienvenido al Optimizador Web 🚀")
    st.markdown("Por favor, ingresa tu nombre de usuario para continuar.")
    
    # Usamos un formulario para presionar "Enter" fácilmente
    with st.form("login_form"):
        username = st.text_input("Nombre de usuario:")
        submitted = st.form_submit_button("Ingresar")
        
        if submitted:
            if username.strip() != "":
                # Guardamos el usuario en el estado de la sesión
                st.session_state['username'] = username
                st.rerun() # Recarga la app para mostrar la siguiente página
            else:
                st.error("Por favor, ingresa un nombre válido.")

def main_app():
    # Botón lateral para cerrar sesión
    with st.sidebar:
        st.write(f"👤 Usuario: **{st.session_state['username']}**")
        if st.button("Cerrar sesión"):
            st.session_state.pop('username')
            st.rerun()

    st.title("⚙️ Optimizador Web")
    st.markdown(f"Hola **{st.session_state['username']}**, configura tu algoritmo a continuación:")

    # Todo el código original de la interfaz del optimizador
    n_vars = st.number_input("Número de variables (n)", min_value=1, max_value=10, value=1)
    vars_names = [f"x{i+1}" for i in range(n_vars)]
    func_input = st.text_input(f"Función f({', '.join(vars_names)})", value="x1**4 - 3*x1**3 + 2")
    start_point = st.text_input(f"Punto inicial (separado por comas)", value="0.5")
    method = st.selectbox("Selecciona el método:", ["Método del Gradiente", "Método de Newton", "Método del Gradiente Conjugado"])

    alpha_type = "Fijo"
    alpha_val = 0.01
    wolfe_params = {'alpha_init': 1.0, 'c1': 1e-4, 'rho': 0.5, 'use_curvature': False, 'theta': 0.9}

    if method == "Método del Gradiente":
        alpha_type = st.radio("Tipo de alfa:", ["Fijo", "Wolfe (Armijo)"])
        if alpha_type == "Fijo":
            alpha_val = st.number_input("Tamaño del paso (alfa):", value=0.01, format="%.4f")
        elif alpha_type == "Wolfe (Armijo)":
            wolfe_params['alpha_init'] = st.number_input("Alfa inicial:", value=1.0, format="%.4f")
            wolfe_params['rho'] = st.number_input("Rho (factor de reducción):", value=0.5, format="%.4f")
            wolfe_params['c1'] = st.number_input("C1 (constante Armijo):", value=1e-4, format="%.4e")
            
            # --- NUEVO: Lista desplegable para curvatura ---
            calc_curv = st.selectbox("¿Evaluar condición de curvatura?", ["No", "Sí"])
            if calc_curv == "Sí":
                wolfe_params['use_curvature'] = True
                wolfe_params['theta'] = st.number_input("Parámetro Theta (Curvatura):", value=0.9, format="%.4f", min_value=0.0, max_value=1.0)

    max_iter = st.number_input("Iteraciones", value=10)

    if st.button("Ejecutar"):
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
                st.dataframe(results)
            else:
                st.error("Error en las dimensiones o la función. Revisa que el punto inicial tenga la misma cantidad de variables.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Lógica principal de enrutamiento ---

if 'username' not in st.session_state:
    login_page()
else:
    main_app()
