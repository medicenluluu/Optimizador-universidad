import streamlit as st          # Librería principal para crear la interfaz web
import numpy as np              # Para cálculos numéricos, vectores y matrices
import pandas as pd             # Para el manejo de datos en tablas (DataFrames)
import sympy as sp              # Para cálculo simbólico (derivadas, gradientes, hessianas)
import re                       # Para expresiones regulares (limpieza de textos)
import matplotlib.pyplot as plt # Para la creación de gráficos 2D
import plotly.graph_objects as go # Para la creación de gráficos 3D interactivos
from plotly.subplots import make_subplots # Para crear gráficos con múltiples ejes (eje Y secundario)

st.set_page_config(page_title="Calculadora de Optimización", layout="wide", initial_sidebar_state="expanded")

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.title("Calculadora")
    st.divider() # Línea separadora
    st.subheader("Bienvenido")
    name_input = st.text_input("Tu nombre:")
    if st.button("Entrar a la aplicación"):
        if name_input: # Si escribió algo, se guarda y se recarga la página
            st.session_state.user_name = name_input
            st.rerun()
        else: # Si está vacío, muestra una advertencia
            st.warning("Por favor, escribe un nombre primero.")
    st.stop()

st.markdown(
    """
    <style>
    /* =========================
       FONDO GENERAL
    ========================= */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    /* =========================
       SIDEBAR
    ========================= */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        border-right: 1px solid #FFFFFF !important;
    }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    /* =========================
       TIPOGRAFÍA
    ========================= */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: Helvetica, Arial, sans-serif !important;
    }
    p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p {
        font-family: Helvetica, Arial, sans-serif !important;
        font-size: 16px !important;
        color: #000000 !important;
    }
    h1, h2, h3, h4 {
        font-family: Helvetica, Arial, sans-serif !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    h1 { font-size: 32px !important; margin-bottom: 15px !important; }
    h2 { font-size: 24px !important; margin-bottom: 12px !important; }
    h3 { font-size: 20px !important; margin-top: 20px !important; }
    h4 { font-size: 18px !important; margin-bottom: 10px !important; }
    /* =========================
       FORMULARIOS Y CAJAS
    ========================= */
    [data-testid="stForm"], .stFormCreator {
        background-color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: none !important;
    }
    /* =========================
       CAJA DE INSTRUCCIONES
    ========================= */
    .instructions-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px 25px;
        margin-bottom: 25px;
        margin-top: 10px;
    }
    .instructions-box ol { margin-bottom: 0; padding-left: 20px; }
    .instructions-box li { color: #000000 !important; }
    .instructions-box code {
        background-color: #F3F4F6;
        color: #000000;
        padding: 2px 6px;
        border-radius: 4px;
    }
    /* =========================
       TARJETAS SIDEBAR
    ========================= */
    .method-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: none;
    }
    .method-card strong { color: #000000 !important; display: block; margin-bottom: 4px; }
    .method-card span { color: #4B5563 !important; }
    
    /* =========================
       INPUTS, SELECTBOXES Y CAJAS DE NÚMEROS
    ========================= */
    input, textarea, 
    [data-baseweb="select"] > div, 
    [data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stWidgetLabel"] *, input *, select *, [data-baseweb="select"] span { 
        color: #000000 !important; 
    }
    
    /* Flecha del menú desplegable */
    [data-baseweb="select"] svg { 
        fill: #000000 !important; 
    }

    /* =========================
       BOTONES DE SUMAR/RESTAR (+/-)
    ========================= */
    [data-testid="stNumberInputStepUp"],
    [data-testid="stNumberInputStepDown"],
    div[data-baseweb="input"] button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stNumberInputStepUp"] svg,
    [data-testid="stNumberInputStepDown"] svg,
    div[data-baseweb="input"] button svg {
        fill: currentColor !important; /* Para que el ícono tome el color de la letra */
    }

    /* Efecto al pasar el mouse o hacer clic en + / - */
    [data-testid="stNumberInputStepUp"]:hover,
    [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stNumberInputStepUp"]:active,
    [data-testid="stNumberInputStepDown"]:active,
    div[data-baseweb="input"] button:hover,
    div[data-baseweb="input"] button:active {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }

    /* =========================
       DROPDOWNS (MENÚS DESPLEGABLES - LA LISTA INTERNA)
    ========================= */
    div[data-baseweb="popover"], div[data-baseweb="popover"] *,
    div[role="listbox"], div[role="listbox"] *, ul[role="listbox"],
    ul[role="listbox"] *, li[role="option"], li[role="option"] * {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Efecto al pasar el mouse por las opciones de la lista */
    li[role="option"]:hover, li[role="option"]:hover *,
    div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li:hover * {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }

    /* =========================
       BOTONES PRINCIPALES
    ========================= */
    .stButton > button, [data-testid="stForm"] button, button[kind="primaryFormSubmit"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        width: 100% !important;
        padding: 10px 24px !important;
        transition: 0.2s ease !important;
    }
    
    /* Efecto hover en botones principales */
    .stButton > button:hover, [data-testid="stForm"] button:hover, button[kind="primaryFormSubmit"]:hover,
    .stButton > button:active {
        background-color: #000000 !important; 
        color: #FFFFFF !important; 
        border: 1px solid #000000 !important;
        transform: translateY(-1px);
    }
    .stButton > button:hover *, [data-testid="stForm"] button:hover * { 
        color: #FFFFFF !important; 
    }

    /* =========================
       TABLAS
    ========================= */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        padding: 10px !important;
        box-shadow: none !important;
    }
    /* =========================
       MÉTRICAS / ALERTAS
    ========================= */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }
    /* =========================
       CUADRO DE DIÁLOGO (POPUP) BLANCO
    ========================= */
    [data-testid="stDialog"], 
    div[data-baseweb="modal"], 
    div[role="dialog"], 
    .st-emotion-cache-12w0qpk {
        background-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True # Permite que Streamlit renderice este HTML/CSS
)

with st.sidebar:
    st.markdown(f"**Usuario:** {st.session_state.user_name}")

def parse_function(func_str, vars_list):
    """
    Toma la función escrita por el usuario (string) y la convierte en un
    objeto matemático manipulable por la librería SymPy.
    """
    try:
        func_str = func_str.replace('^', '**') # Potencias: x^2 pasa a x**2
        func_str = func_str.replace('ln', 'log') # Logaritmos naturales
        func_str = re.sub(r'e\\\((.*?)\)', r'exp(\1)', func_str)
        func_str = re.sub(r'e\\(.?)(\s|\+|-|\|\/|$)', r'exp(\1)\2', func_str)
        func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
        expr = sp.sympify(func_str)
        return expr
    except Exception:
        return None # Si hay un error de sintaxis, retorna None

def compute_gradient(expr, variables):
    """ Calcula el gradiente (vector de derivadas parciales) de la función. """
    return [sp.diff(expr, var) for var in variables]

def compute_hessian(expr, variables):
    """ Calcula la matriz Hessiana (matriz de segundas derivadas parciales). """
    return sp.hessian(expr, variables)

def calcular_error(curr_x, prev_x, norm_type):
    """
    Calcula el error relativo entre la iteración actual (curr_x) y la anterior (prev_x).
    Permite elegir el tipo de norma matemática para medir esta "distancia".
    """
    if norm_type == "L_infinito (Máximo)":
        num = np.linalg.norm(curr_x - prev_x, ord=np.inf)
        den = np.linalg.norm(curr_x, ord=np.inf)
    elif norm_type == "L1 (Manhattan)":
        num = np.linalg.norm(curr_x - prev_x, ord=1)
        den = np.linalg.norm(curr_x, ord=1)
    else: # Norma Euclidiana por defecto (L2)
        num = np.linalg.norm(curr_x - prev_x)
        den = np.linalg.norm(curr_x)
    return num / (den if den != 0 else 1e-8)

def evaluate_func_safe(f_lambdified, x, vars_sym):
    """
    Evalúa matemáticamente un punto "x" en la función.
    Si el resultado da error (ej. logaritmo de un negativo) o infinito, retorna un valor altísimo (1e9)
    para penalizar ese punto y que el algoritmo no colapse.
    """
    try:
        val = f_lambdified(*x) if len(vars_sym) > 1 else f_lambdified(x[0])
        if np.isnan(val) or np.isinf(val):
            return 1e9
        return val
    except Exception:
        return 1e9

def run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params, max_iter, tol, norm_type):
    """
    MÉTODO DEL GRADIENTE (Descenso más pronunciado).
    Se mueve iterativamente en la dirección opuesta al gradiente para buscar un mínimo.
    """
    history = []      # Guarda la tabla de iteraciones
    backtrack_log = [] # Guarda los intentos fallidos en la búsqueda de Armijo

    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]

    curr_x = np.array(x0, dtype=float) # Punto inicial

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

        if k > 0 and rel_error < tol:
            break

        if k < max_iter:
            direction = -grad_val # La dirección de descenso es el gradiente negativo
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
                        alpha *= rho # Reduce alpha multiplicándolo por rho (ej. a la mitad)
                    else:
                        break # Si cumple, salimos del bucle de intentos

            curr_x = curr_x - alpha * grad_val

    return pd.DataFrame(history), pd.DataFrame(backtrack_log), grad_exprs

def run_newton_method(expr, vars_sym, x0, max_iter, tol, norm_type):
    """
    MÉTODO DE NEWTON.
    Utiliza información de segunda derivada (Hessiana) para hacer pasos mucho más precisos.
    Protección: Si la Hessiana no es definida positiva (sus autovalores no son todos positivos)
    o no es invertible, se reemplaza por la matriz Identidad para evitar direcciones de ascenso o errores.
    """
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

        rel_error = 0.0
        if k > 0:
            prev_x = np.array([history[-1][f'{v}'] for v in vars_sym])
            rel_error = calcular_error(curr_x, prev_x, norm_type)

        # Determinar qué acción tomará para la próxima iteración
        accion_str = ""
        hess_inv = None

        if k < max_iter:
            try:
                # Calculamos la matriz Hessiana evaluada en el punto actual
                hess_val = np.array(hess_lambdified(*curr_x), dtype=float) if len(vars_sym) > 1 else np.array([[hess_lambdified(curr_x[0])]], dtype=float)
                
                # Verificamos si la matriz es definida positiva a través de sus autovalores
                eigvals = np.linalg.eigvals(hess_val)
                
                if np.any(eigvals <= 1e-8):
                    # Si tiene autovalores negativos o cercanos a cero, falla la prueba de positividad
                    hess_inv = np.eye(len(vars_sym))
                    accion_str = "Falla matriz positiva -> Uso Matriz Identidad"
                else:
                    # Si es definida positiva, podemos invertirla de manera segura
                    hess_inv = np.linalg.inv(hess_val)
                    accion_str = "Hessiana OK -> Newton Estándar"
                    
            except Exception:
                # Si ocurre cualquier error algebraico (ej. no invertible), usamos Identidad
                hess_inv = np.eye(len(vars_sym))
                accion_str = "Error al invertir -> Uso Matriz Identidad"

        if k > 0 and rel_error < tol:
            accion_str = "Óptimo alcanzado"

        # Guardar en el historial
        entry = {
            'Iteración': k, 
            'C(x)': f_val, 
            '||∇C(x)||': np.linalg.norm(grad_val), 
            'Error Rel. (%)': rel_error * 100,
            'Acción para sig. paso': accion_str
        }
        
        for i, val in enumerate(curr_x): entry[f'{vars_sym[i]}'] = val
        for i, val in enumerate(grad_val): entry[f'g_{vars_sym[i]}'] = val
        history.append(entry)

        if k > 0 and rel_error < tol:
            break

        # Dar el paso si no hemos terminado
        if k < max_iter and hess_inv is not None:
            curr_x = curr_x - hess_inv.dot(grad_val)

    return pd.DataFrame(history)

def run_conjugate_gradient(expr, vars_sym, x0, alpha_type, alpha_val, max_iter, tol, norm_type):
    """
    MÉTODO DEL GRADIENTE CONJUGADO.
    En lugar de ir siempre por el gradiente directo, crea direcciones "conjugadas" que evitan
    deshacer el trabajo de las iteraciones anteriores (como pasa en el zigzag del gradiente normal).
    """
    history = []
    f_lambdified = sp.lambdify(vars_sym, expr, 'numpy')
    grad_exprs = compute_gradient(expr, vars_sym)
    grad_lambdified = [sp.lambdify(vars_sym, g, 'numpy') for g in grad_exprs]

    curr_x = np.array(x0, dtype=float)
    grad_val = np.array([g(*curr_x) for g in grad_lambdified]) if len(vars_sym) > 1 else np.array([grad_lambdified[0](curr_x[0])])
    p = -grad_val.copy() # Dirección inicial 'p' es el gradiente negativo

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

@st.dialog("📖 Definición del Método")
def mostrar_metodo():
    # Esta línea fuerza el color blanco específicamente cuando se abre el cuadro
    st.markdown("""<style>div[role="dialog"] { background-color: #FFFFFF !important; }</style>""", unsafe_allow_html=True)
    
    metodo = st.session_state.metodo_info
    if metodo == "gradiente":
        st.subheader("Método del Gradiente")
        st.write("""
        Busca mínimos moviéndose en la dirección opuesta al gradiente.
        • Fácil de implementar.
        • Utiliza derivadas de primer orden.
        • Puede requerir muchas iteraciones.
        """)
    elif metodo == "newton":
        st.subheader("Método de Newton")
        st.write("""
        Utiliza gradiente y Hessiana para aproximar rápidamente el óptimo.
        • Convergencia rápida.
        • Usa derivadas de segundo orden.
        • **Mejora:** Si la matriz Hessiana no es definida positiva o no es invertible, la aplicación utilizará automáticamente una Matriz Identidad para proteger el algoritmo.
        """)
    elif metodo == "conjugado":
        st.subheader("Gradiente Conjugado")
        st.write("""
        Genera direcciones conjugadas para evitar recorrer caminos repetidos.
        • Más eficiente que gradiente clásico.
        • Muy útil en problemas grandes.
        • No requiere Hessiana completa.
        """)
    if st.button("Cerrar"):
        del st.session_state["metodo_info"]
        st.rerun()

def main_app():
    """
    Función principal que renderiza el núcleo de la aplicación:
    formularios, configuraciones, ejecución y visualización de resultados.
    """
    with st.sidebar:
        st.markdown("## 💡 Diccionario")
        st.markdown("### de Métodos")
        if st.button("📉 Método del Gradiente"):
            st.session_state.metodo_info = "gradiente"
            mostrar_metodo() # Se invoca explícitamente para abrir el dialog (en Streamlit 1.35+)
        if st.button("🚀 Método de Newton"):
            st.session_state.metodo_info = "newton"
            mostrar_metodo()
        if st.button("🎯 Gradiente Conjugado"):
            st.session_state.metodo_info = "conjugado"
            mostrar_metodo()

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

    st.title(" Calculadora de Optimización Académica")

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
            st.info("Calcula su propio paso. Si la Hessiana falla (no definida positiva), usará Matriz Identidad.")
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
                w_c1, w_c2 = st.columns(2) # Sub-columnas para los parámetros avanzados de Wolfe
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

    st.markdown("<br>", unsafe_allow_html=True) # Espacio en blanco HTML
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        execute = st.button("▶️ Resolver Problema")

    if execute:
        st.markdown("### 3. Resultados y Análisis")

        vars_sym = sp.symbols(' '.join(vars_names))
        if len(vars_names) == 1: vars_sym = [vars_sym]
        expr = parse_function(func_input, vars_sym)

        try:
            clean_str = re.sub(r'[^0-9.,-]', '', start_point)
            x0 = [float(i) for i in clean_str.split(',') if i.strip()]

            if expr is not None and len(x0) == len(vars_names):
                if method == "Método del Gradiente":
                    results, bt_log, grad_exprs = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params if alpha_type=="Wolfe (Armijo)" else None, int(max_iter), tolerancia, norm_type)
                elif method == "Método de Newton":
                    results = run_newton_method(expr, vars_sym, x0, int(max_iter), tolerancia, norm_type)
                else:
                    results = run_conjugate_gradient(expr, vars_sym, x0, alpha_type, cg_alpha_val if alpha_type == "Fijo" else cg_alpha_val, int(max_iter), tolerancia, norm_type)

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    st.markdown("#### Trayectoria y Curvas de Nivel")
                    f_lambdified_plot = sp.lambdify(vars_sym, expr, 'numpy')
                    if len(vars_names) == 1:
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

                    elif len(vars_names) == 2:
                        x_hist = results[f'{vars_names[0]}'].values
                        y_hist = results[f'{vars_names[1]}'].values
                        z_hist = results['C(x)'].values

                        margin_x = max(1.0, (max(x_hist) - min(x_hist)) * 0.5)
                        margin_y = max(1.0, (max(y_hist) - min(y_hist)) * 0.5)
                        x_range = np.linspace(min(x_hist) - margin_x, max(x_hist) + margin_x, 50)
                        y_range = np.linspace(min(y_hist) - margin_y, max(y_hist) + margin_y, 50)
                        X, Y = np.meshgrid(x_range, y_range)
                        Z = np.zeros_like(X)

                        for i in range(X.shape[0]):
                            for j in range(X.shape[1]):
                                try:
                                    val = f_lambdified_plot(X[i,j], Y[i,j])
                                    Z[i,j] = val if not np.isnan(val) and not np.isinf(val) else np.nan
                                except:
                                    Z[i,j] = np.nan

                        fig_3d = go.Figure()
                        fig_3d.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale='Viridis',
                            opacity=0.8,
                            name='Superficie',
                            showscale=False
                        ))
                        fig_3d.add_trace(go.Scatter3d(
                            x=x_hist, y=y_hist, z=z_hist,
                            mode='lines+markers',
                            marker=dict(size=4, color='red', symbol='circle'),
                            line=dict(color='red', width=4),
                            name='Trayectoria'
                        ))
                        fig_3d.update_layout(
                            title="Vista 3D: Superficie y Trayectoria",
                            scene=dict(
                                xaxis_title=f"{vars_names[0]}",
                                yaxis_title=f"{vars_names[1]}",
                                zaxis_title="C(x,y)"
                            ),
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

                        fig_contour = go.Figure()
                        fig_contour.add_trace(go.Contour(
                            x=x_range, y=y_range, z=Z,
                            colorscale='Viridis',
                            contours=dict(showlabels=True),
                            name='Curvas de Nivel',
                            colorbar=dict(title="C(x,y)")
                        ))
                        fig_contour.add_trace(go.Scatter(
                            x=x_hist, y=y_hist,
                            mode='lines+markers',
                            marker=dict(size=6, color='red', symbol='circle'),
                            line=dict(color='red', width=2),
                            name='Trayectoria'
                        ))
                        fig_contour.update_layout(
                            title="Vista Topográfica: Curvas de Nivel",
                            xaxis_title=f"{vars_names[0]}",
                            yaxis_title=f"{vars_names[1]}",
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        st.plotly_chart(fig_contour, use_container_width=True)

                    else:
                        st.info("La gráfica de trayectoria interactiva solo está disponible para 1 o 2 variables.")

                with col_g2:
                    st.markdown("#### Análisis de Convergencia")
                    if len(results) > 0:
                        iter_vals = results['Iteración'].values
                        grad_norms = results['||∇C(x)||'].values
                        f_vals = results['C(x)'].values

                        fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_conv.add_trace(
                            go.Scatter(
                                x=iter_vals,
                                y=grad_norms,
                                name="||∇f(x_k)||",
                                mode="lines+markers",
                                line=dict(color="#2563EB", width=3), # Azul vibrante
                                marker=dict(size=6)
                            ),
                            secondary_y=False,
                        )
                        fig_conv.add_trace(
                            go.Scatter(
                                x=iter_vals,
                                y=f_vals,
                                name="f(x_k)",
                                mode="lines+markers",
                                line=dict(color="#F97316", width=3, dash="dot"), # Naranja punteado
                                marker=dict(size=6)
                            ),
                            secondary_y=True,
                        )

                        fig_conv.update_layout(
                            title="Convergencia",
                            xaxis_title="Iteración",
                            legend=dict(
                                orientation="h",   # Leyenda horizontal en la parte superior
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            margin=dict(l=0, r=0, b=0, t=60)
                        )
                        fig_conv.update_yaxes(
                            title_text="||∇f(x_k)|| (escala log)",
                            type="log",
                            secondary_y=False
                        )
                        fig_conv.update_yaxes(
                            title_text="f(x_k)",
                            secondary_y=True
                        )
                        st.plotly_chart(fig_conv, use_container_width=True)
                    else:
                        st.info("El algoritmo no generó iteraciones válidas.")

                st.markdown("#### Tabla General del Historial de Iteraciones")
                st.dataframe(results, use_container_width=True)

            else:
                st.error("Error: Asegúrate que la cantidad de valores en el punto inicial coincida con las variables.")
        except Exception as e:
            st.error(f"Se encontró un error matemático/sintáctico: {e}")

if __name__ == "__main__":
    main_app()
