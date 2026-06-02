import streamlit as st
import numpy as np
import sympy as sp

# 1. Configuración de página
st.set_page_config(
    page_title="Calculadora de Optimización", 
    page_icon="✨", 
    layout="wide"
)

# 2. CSS Estético (Baby Blue & Rounded UI)
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

        .stApp { 
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%) !important; 
            font-family: 'Nunito', sans-serif !important;
        }
        
        .hero-card {
            background: rgba(255, 255, 255, 0.9);
            padding: 3rem;
            border-radius: 30px;
            text-align: center;
            box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .hero-title { color: #0f172a !important; font-size: 3rem !important; margin-bottom: 1rem; }
        .hero-subtitle { color: #334155 !important; font-size: 1.25rem !important; }
        
        .card {
            background-color: #FFFFFF !important;
            padding: 2rem !important;
            border-radius: 25px !important;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05) !important;
            margin-bottom: 20px;
        }

        div.stButton > button {
            background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
            color: white !important;
            border-radius: 50px !important;
            padding: 10px 30px !important;
            font-weight: 700 !important;
            border: none !important;
            width: 100% !important;
        }

        .stTextInput > div > div > input {
            border-radius: 15px !important;
            border: 2px solid #bae6fd !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# --- LOGIN ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.markdown("""
    <div class="hero-card">
        <h1 class="hero-title">Calculadora de Optimización</h1>
        <p class="hero-subtitle">Resuelve problemas mediante Gradiente, Newton y Gradiente Conjugado.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name_input = st.text_input("Ingresa tu nombre para comenzar:")
        if st.button("🚀 Comenzar"):
            if name_input:
                st.session_state.user_name = name_input
                st.rerun()
            else:
                st.warning("Por favor, ingresa tu nombre.")
    st.stop()

# --- APP PRINCIPAL ---
st.title(f"¡Bienvenido, {st.session_state.user_name}! 🎓")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        vars_input = st.text_input("Variables (ej: x, y)", "x, y")
        func_input = st.text_input("Función C(x, y)", "x**2 + y**2")
    
    with col2:
        start_point = st.text_input("Punto inicial (ej: 1, 1)", "1, 1")
        method = st.selectbox("Método de resolución", ["Gradiente", "Newton", "Gradiente Conjugado"])
    
    if st.button("▶️ Resolver Problema"):
        st.write("---")
        st.info(f"Procesando optimización con **{method}**...")
        # Lógica de cálculo aquí
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.sidebar.markdown(f"### 👤 Usuario: {st.session_state.user_name}")
if st.sidebar.button("Cerrar sesión"):
    del st.session_state.user_name
    st.rerun()


# --- Funciones de Matemáticas (Intactas) ---
def parse_function(func_str, vars_list):
    try:
        func_str = func_str.replace('^', '**')
        func_str = func_str.replace('ln', 'log')
        func_str = re.sub(r'e\\\((.*?)\)', r'exp(\1)', func_str)
        func_str = re.sub(r'e\\(.?)(\s|\+|-|\|\/|$)', r'exp(\1)\2', func_str)
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

# --- Algoritmos de Optimización (Intactos) ---
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

# --- Interfaz de Usuario Principal ---
def main_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Hola, **{st.session_state.user_name}**")
        st.markdown("<hr style='margin: 12px 0; border-color: rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        
        with st.expander("💡 Diccionario de Métodos", expanded=True):
            st.markdown("""<div class="method-card"><strong>📉 Método del Gradiente</strong><span>Fácil de usar para buscar mínimos locales moviéndose en la dirección del gradiente negativo.</span></div>""", unsafe_allow_html=True)
            st.markdown("""<div class="method-card"><strong>🚀 Método de Newton</strong><span>Utiliza la pendiente y la curvatura (Hessiana) para determinar un paso rápido, ideal para soluciones cercanas al óptimo.</span></div>""", unsafe_allow_html=True)
            st.markdown("""<div class="method-card"><strong>🎯 Gradiente Conjugado</strong><span>Optimiza usando direcciones ortogonales. Evita repetir caminos explorados para avanzar con gran precisión.</span></div>""", unsafe_allow_html=True)

        if st.button("🚪 Cambiar Usuario"):
            st.session_state.user_name = ""
            st.rerun()

    # Contenido Principal
    st.markdown("""
    <div class="instructions-box">
        <h4>📖 Guía de Uso Académico</h4>
        <ol>
            <li><strong>Variables Flexibles:</strong> Escribe <code>x, y</code> o <code>x1, x2</code> dependiendo de tu problema.</li>
            <li><strong>Manejo de Funciones:</strong> Usa <code>ln()</code> o <code>log()</code> sin problema. Usa <code>^</code> para exponentes.</li>
            <li><strong>Reporte Detallado:</strong> Selecciona "Método del Gradiente" + "Wolfe (Armijo)" y activa el "Modo Examen" para ver el paso a paso matemático exacto.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.title("⚙️ Calculadora de Optimización")
    st.markdown("### 1. Definición del Problema")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        vars_input = st.text_input("Variables (separadas por coma)", value="x, y")
        vars_names = [v.strip() for v in vars_input.split(',')]
    
    with col2:
        func_input = st.text_input(f"Función C({', '.join(vars_names)}) a minimizar", value="ln(x^2 + y^2) - 2*x*y")
        
    with col3:
        start_point = st.text_input("Punto inicial (x0, y0)", value="-1, 0")

    st.markdown("<hr style='border: 1px dashed #cbd5e1; margin: 25px 0;'>", unsafe_allow_html=True)
    
    st.markdown("### 2. Parámetros del Algoritmo")
    
    col_m1, col_m2, col_m3 = st.columns([1, 1.2, 1])
    
    with col_m1:
        method = st.selectbox("Método de optimización:", ["Método del Gradiente", "Método de Newton", "Método del Gradiente Conjugado"])
        max_iter = st.number_input("Iteraciones máximas", value=10, min_value=1)
        
    with col_m2:
        if method == "Método de Newton":
            st.info("El Método de Newton calcula su propio paso dinámicamente usando la Matriz Hessiana.")
            alpha_type = "Newton"
        elif method == "Método del Gradiente Conjugado":
            alpha_type = st.radio("Cálculo del tamaño de paso (alfa):", ["Fijo", "Búsqueda de línea (Armijo)"], horizontal=True)
            if alpha_type == "Fijo":
                cg_alpha_val = st.number_input("Valor de alfa (GC):", value=0.01, format="%.4f")
            else:
                cg_alpha_val = st.number_input("Alfa inicial para búsqueda:", value=1.0, format="%.4f")
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
                    wolfe_sigma = st.number_input("Sigma (σ):", value=0.5, format="%.4f", help="Para la 2da condición de Wolfe")
                
    with col_m3:
        tolerancia = st.number_input("Tolerancia de parada", value=0.001, format="%.4f")
        norm_type = st.selectbox("Norma para Error Relativo:", ["L_infinito (Máximo)", "L2 (Euclidiana)", "L1 (Manhattan)"])
        if method == "Método del Gradiente":
            st.markdown("<br>", unsafe_allow_html=True)
            show_exam_mode = st.checkbox("🔍 Mostrar Detalles Académicos (Modo Examen)", value=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        execute = st.button("▶️ Iniciar Optimización")
    st.markdown('</div>', unsafe_allow_html=True) # Cierra el glass-panel

    if execute:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("### 📊 3. Resultados y Análisis")
        vars_sym = sp.symbols(' '.join(vars_names))
        if len(vars_names) == 1: vars_sym = [vars_sym]
        expr = parse_function(func_input, vars_sym)
        
        try:
            clean_str = re.sub(r'[^0-9.,-]', '', start_point)
            x0 = [float(i) for i in clean_str.split(',') if i.strip()]
            
            if expr is not None and len(x0) == len(vars_names):
                
                # --- EJECUTAR ALGORITMO ---
                if method == "Método del Gradiente":
                    results, bt_log, grad_exprs = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params if alpha_type=="Wolfe (Armijo)" else None, int(max_iter), tolerancia, norm_type)
                elif method == "Método de Newton":
                    results = run_newton_method(expr, vars_sym, x0, int(max_iter), tolerancia, norm_type)
                else:
                    results = run_conjugate_gradient(expr, vars_sym, x0, alpha_type, cg_alpha_val if alpha_type == "Fijo" else cg_alpha_val, int(max_iter), tolerancia, norm_type)

                # --- MODO EXAMEN ---
                if method == "Método del Gradiente" and show_exam_mode:
                    with st.expander("📚 Desarrollo Paso a Paso (Iteración 0 a 1) - MODO EXAMEN", expanded=True):
                        st.markdown("#### (a) Una iteración de descenso por gradiente con Backtracking")
                        
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
                        
                        st.markdown("**La dirección de descenso es:**")
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
                            
                            st.markdown("Calculamos:")
                            st.latex(rf"\nabla C(x^{{(0)}})^T d^{{(0)}} = {g0_tuple} \cdot {d0_tuple} = {dot_g0_d0:g}")
                            
                            rhs_0 = f0_val + c1_val * a0 * dot_g0_d0
                            st.markdown("Entonces el lado derecho vale:")
                            st.latex(rf"{f0_val:g} + {beta_tex}({a0})({dot_g0_d0:g}) = {rhs_0:g}")
                            
                            st.markdown("Debemos verificar:")
                            st.latex(rf"{f_cand_0:g} \le {rhs_0:g}")
                            
                            if row_0['Cumple']:
                                st.success("Lo cual es verdadero. Condición cumplida.")
                                st.markdown("Por tanto el paso aceptado es:")
                                st.latex(rf"\boxed{{\alpha_1 = {a0}}}")
                                st.markdown("Y el nuevo iterado es:")
                                st.latex(rf"\boxed{{x^{{(1)}} = {cand_0_tuple}}}")
                            else:
                                st.error(f"Lo cual es **falso**. Por tanto, se rechaza $\\alpha = {a0}$.")
                                
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
                                    st.latex(rf"{f_cand_i:g} \le {rhs_i:g}")
                                    
                                    if row_i['Cumple']:
                                        st.success("¡Verdadero!")
                                        st.markdown("Por tanto el paso aceptado es:")
                                        st.latex(rf"\boxed{{\alpha_1 = {a_i}}}")
                                        st.markdown("Y el nuevo iterado es:")
                                        st.latex(rf"\boxed{{x^{{(1)}} = {cand_i_tuple}}}")
                                        break
                                    else:
                                        st.error("Falso. Se sigue iterando.")
                                        
                        if len(results) > 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                            norm_symbol = norm_type.split('_')[0].replace("L", "")
                            st.markdown(f"#### (b) Error porcentual de la primera iteración usando $\\| \cdot \\|_{{\\text{{{norm_symbol}}}}}$")
                            
                            st.markdown("Usamos el error relativo aproximado:")
                            st.latex(
                                r"E = \frac{\|x^{(1)} - x^{(0)}\|_{" + norm_symbol + r"}}{\|x^{(1)}\|_{" + norm_symbol + r"}} \times 100\%"
                            )

                            x1_vals = np.array([results.iloc[1][f'{v}'] for v in vars_sym])
                            diff_vals = x1_vals - x0
                            diff_str = ", ".join([f"{v:g}" for v in diff_vals])
                            diff_tuple = f"({diff_str})" if len(diff_vals) > 1 else f"{diff_vals[0]:g}"
                            x1_str = ", ".join([f"{v:g}" for v in x1_vals])
                            x1_tuple = f"({x1_str})" if len(x1_vals) > 1 else f"{x1_vals[0]:g}"

                            st.markdown("Calculamos:")
                            st.latex(rf"x^{{(1)}} - x^{{(0)}} = {x1_tuple} - {x0_tuple} = {diff_tuple}")

                            ord_val = np.inf if "infinito" in norm_type else (1 if "L1" in norm_type else 2)
                            num_val = np.linalg.norm(diff_vals, ord=ord_val)
                            den_val = np.linalg.norm(x1_vals, ord=ord_val)

                            st.markdown("Entonces:")
                            st.latex(rf"\|x^{{(1)}} - x^{{(0)}}\|_{{{norm_symbol}}} = {num_val:g}")
                            st.markdown("Además:")
                            st.latex(rf"\|x^{{(1)}}\|_{{{norm_symbol}}} = {den_val:g}")

                            E_val = (num_val / den_val) * 100 if den_val != 0 else 0
                            st.markdown("Por tanto:")
                            st.latex(rf"E = \frac{{{num_val:g}}}{{{den_val:g}}}\times100 = {E_val:g}\%")
                            st.latex(rf"\boxed{{E = {E_val:g}\%}}")

                # --- GRÁFICOS ESTÉTICOS ---
                st.markdown("<br>", unsafe_allow_html=True)
                col_g1, col_g2 = st.columns(2)
                
                # Estilo general para matplotlib
                plt.style.use('default')
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = ['Nunito', 'Arial']
                
                with col_g1:
                    st.markdown("#### 📈 Trayectoria de la Función")
                    if len(vars_names) == 1:
                        f_lambdified_plot = sp.lambdify(vars_sym, expr, 'numpy')
                        x_hist = results[f'{vars_names[0]}'].values
                        f_hist = results['C(x)'].values
                        margin = max(1.0, (max(x_hist) - min(x_hist)) * 0.5)
                        x_range = np.linspace(min(x_hist) - margin, max(x_hist) + margin, 500)
                        y_range = [f_lambdified_plot(val) for val in x_range]
                        
                        fig, ax = plt.subplots(figsize=(7, 5))
                        fig.patch.set_facecolor('#ffffff')
                        ax.set_facecolor('#f8fafc')
                        ax.plot(x_range, y_range, label='C(x)', color='#0284c7', linewidth=2.5)
                        ax.plot(x_hist, f_hist, label='Iteraciones', color='#f43f5e', marker='o', linestyle=':', markersize=8, markerfacecolor='#ffffff', markeredgewidth=2)
                        
                        ax.set_title("Comportamiento del Algoritmo", fontsize=14, fontweight='bold', color='#0f172a', pad=15)
                        ax.set_xlabel(f"Variable {vars_names[0]}", color='#475569')
                        ax.set_ylabel("Valor C(x)", color='#475569')
                        
                        # Quitar bordes para un look más moderno
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_color('#cbd5e1')
                        ax.spines['left'].set_color('#cbd5e1')
                        ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
                        ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')
                        
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("💡 *La gráfica 2D está disponible solo para funciones de 1 variable.*")

                with col_g2:
                    st.markdown("#### 📉 Análisis de Convergencia")
                    if len(results) > 1:
                        iter_vals = results['Iteración'].values[1:] 
                        err_vals = results['Error Rel. (%)'].values[1:]
                        
                        fig_conv, ax_conv = plt.subplots(figsize=(7, 5))
                        fig_conv.patch.set_facecolor('#ffffff')
                        ax_conv.set_facecolor('#f8fafc')
                        
                        ax_conv.plot(iter_vals, err_vals, label='Error Relativo (%)', color='#10b981', marker='s', linestyle='-', linewidth=2.5, markersize=7, markerfacecolor='#ffffff', markeredgewidth=2)
                        ax_conv.set_title("Evolución del Error", fontsize=14, fontweight='bold', color='#0f172a', pad=15)
                        ax_conv.set_xlabel("Número de Iteración (k)", color='#475569')
                        ax_conv.set_ylabel("Error (%) - Escala Logarítmica", color='#475569')
                        ax_conv.set_yscale('log')
                        
                        # Estilos modernos
                        ax_conv.spines['top'].set_visible(False)
                        ax_conv.spines['right'].set_visible(False)
                        ax_conv.spines['bottom'].set_color('#cbd5e1')
                        ax_conv.spines['left'].set_color('#cbd5e1')
                        ax_conv.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
                        ax_conv.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')
                        
                        st.pyplot(fig_conv, use_container_width=True)
                    else:
                        st.success("✨ ¡El algoritmo convergió de manera exacta en el primer intento!")

                st.markdown("<br>#### 📋 Historial Completo de Iteraciones", unsafe_allow_html=True)
                # Aplicamos estilo al dataframe de pandas
                st.dataframe(results.style.format(precision=6), use_container_width=True)
                
            else:
                st.error("⚠️ Error: Asegúrate que la cantidad de valores en el punto inicial coincida exactamente con las variables.")
        except Exception as e:
            st.error(f"⚠️ Se encontró un error matemático/sintáctico en tu función o parámetros: {e}")
            
        st.markdown('</div>', unsafe_allow_html=True) # Cierra el glass-panel final

if __name__ == "__main__":
    main_app()
