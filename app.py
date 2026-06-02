import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re
import plotly.graph_objects as go

# 1. Configuración de página
st.set_page_config(page_title="OptiCalc Pro", layout="wide", initial_sidebar_state="expanded")

# --- BLOQUE DE LOGIN ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>⚙️ OptiCalc Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Plataforma Avanzada de Optimización Numérica</h3>", unsafe_allow_html=True)
    st.write("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👋 Por favor, identifícate para iniciar la sesión de estudio.")
        name_input = st.text_input("Ingresa tu nombre o matrícula:")
        if st.button("Ingresar a la Plataforma", use_container_width=True):
            if name_input.strip():
                st.session_state.user_name = name_input.strip()
                st.rerun() 
            else:
                st.warning("El nombre es obligatorio.")
    st.stop()
# --- FIN BLOQUE DE LOGIN ---

# 2. Inyección de CSS (Diseño mejorado)
st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC !important; }
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
    h1, h2, h3 { color: #0F172A !important; font-family: 'Inter', sans-serif !important; font-weight: 700 !important; }
    .metric-card { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E2E8F0; text-align: center;}
    .metric-title { font-size: 14px; color: #64748B; text-transform: uppercase; font-weight: 600;}
    .metric-value { font-size: 24px; color: #1E3A8A; font-weight: bold;}
    .method-card { background-color: #F1F5F9 !important; padding: 14px; border-radius: 8px; border-left: 4px solid #3B82F6; margin-bottom: 12px; }
    .method-card strong { color: #1E40AF !important; font-size: 15px !important; display: block; margin-bottom: 4px; }
    .stButton > button { background-color: #2563EB !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; border: none !important; transition: all 0.3s ease !important; }
    .stButton > button:hover { background-color: #1D4ED8 !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important; }
    [data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; border: 1px solid #E2E8F0 !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Funciones de Matemáticas ---
@st.cache_data
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

# --- Interfaz de Usuario Principal ---
def main_app():
    with st.sidebar:
        st.markdown(f"### 👤 Usuario: {st.session_state.user_name}")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 💡 Algoritmos Disponibles")
        st.markdown("""<div class="method-card"><strong>📉 Método del Gradiente</strong><span>Ideal para funciones convexas simples.</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="method-card"><strong>🚀 Método de Newton</strong><span>Utiliza la Hessiana para convergencia cuadrática.</span></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="method-card"><strong>🎯 Gradiente Conjugado</strong><span>Optimiza usando direcciones conjugadas.</span></div>""", unsafe_allow_html=True)
        if st.button("Cerrar Sesión"):
            st.session_state.user_name = ""
            st.rerun()

    st.title("⚙️ OptiCalc Pro")
    st.markdown("Calculadora Avanzada de Optimización Matemática con Trazado Interactivo.")
    
    with st.expander("📝 Configuración del Problema Matemático", expanded=True):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            vars_input = st.text_input("Variables (ej. x, y)", value="x, y")
            vars_names = [v.strip() for v in vars_input.split(',')]
        with col2:
            func_input = st.text_input("Función Objetivo C(x)", value="x**2 + y**2 - x*y")
        with col3:
            start_point = st.text_input("Punto Inicial (x0)", value="4, 4")

        st.markdown("---")
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
        with col_m1:
            method = st.selectbox("Método de Optimización:", ["Método del Gradiente", "Método de Newton", "Método del Gradiente Conjugado"])
            max_iter = st.number_input("Iteraciones Máximas", value=20, min_value=1)
        with col_m2:
            if method == "Método de Newton":
                st.info("El paso es calculado dinámicamente con la Matriz Hessiana inversa.")
                alpha_type = "Newton"
                alpha_val = None
            elif method == "Método del Gradiente Conjugado":
                alpha_type = st.radio("Tamaño de paso (α):", ["Fijo", "Búsqueda (Armijo)"], horizontal=True)
                alpha_val = st.number_input("Valor de α:", value=0.1, format="%.4f")
            else:
                alpha_type = st.radio("Cálculo de Paso:", ["Fijo", "Wolfe (Armijo)"], index=1, horizontal=True)
                if alpha_type == "Fijo":
                    alpha_val = st.number_input("Valor de α:", value=0.1, format="%.4f")
                else:
                    alpha_val = 0.0 
                    wolfe_params = {
                        'alpha_init': st.number_input("α inicial:", value=1.0, format="%.4f"),
                        'rho': st.number_input("Factor ρ:", value=0.5, format="%.4f"),
                        'c1': st.number_input("Factor c1 (Armijo):", value=0.1, format="%.4f")
                    }
        with col_m3:
            tolerancia = st.number_input("Tolerancia (Criterio de Parada)", value=0.001, format="%.4f")
            norm_type = st.selectbox("Norma para Error Relativo:", ["L_infinito (Máximo)", "L2 (Euclidiana)", "L1 (Manhattan)"])

    if st.button("🚀 Ejecutar Optimización", use_container_width=True):
        vars_sym = sp.symbols(' '.join(vars_names))
        if len(vars_names) == 1: vars_sym = [vars_sym]
        expr = parse_function(func_input, vars_sym)
        
        try:
            clean_str = re.sub(r'[^0-9.,-]', '', start_point)
            x0 = [float(i) for i in clean_str.split(',') if i.strip()]
            
            if expr is not None and len(x0) == len(vars_names):
                with st.spinner("Optimizando y generando gráficos interactivos..."):
                    if method == "Método del Gradiente":
                        results, bt_log, grad_exprs = run_gradient_descent(expr, vars_sym, x0, alpha_type, alpha_val, wolfe_params if alpha_type=="Wolfe (Armijo)" else None, int(max_iter), tolerancia, norm_type)
                    elif method == "Método de Newton":
                        results = run_newton_method(expr, vars_sym, x0, int(max_iter), tolerancia, norm_type)
                    else:
                        results = run_conjugate_gradient(expr, vars_sym, x0, alpha_type, alpha_val, int(max_iter), tolerancia, norm_type)

                # DASHBOARD KPIs
                st.markdown("### 📊 Resumen de Resultados")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                final_row = results.iloc[-1]
                
                kpi1.markdown(f"<div class='metric-card'><div class='metric-title'>Iteraciones</div><div class='metric-value'>{int(final_row['Iteración'])}</div></div>", unsafe_allow_html=True)
                kpi2.markdown(f"<div class='metric-card'><div class='metric-title'>Valor Final C(x)</div><div class='metric-value'>{final_row['C(x)']:.4f}</div></div>", unsafe_allow_html=True)
                kpi3.markdown(f"<div class='metric-card'><div class='metric-title'>Error Relativo Final</div><div class='metric-value'>{final_row['Error Rel. (%)']:.4f}%</div></div>", unsafe_allow_html=True)
                kpi4.markdown(f"<div class='metric-card'><div class='metric-title'>Norma del Gradiente</div><div class='metric-value'>{final_row['||∇C(x)||']:.4f}</div></div>", unsafe_allow_html=True)
                st.write("")

                # SISTEMA DE PESTAÑAS (TABS)
                tab_graficos, tab_datos, tab_examen = st.tabs(["📈 Gráficos Interactivos", "🗄️ Tabla de Datos", "📚 Modo Examen (Detalle)"])

                with tab_graficos:
                    col_plot1, col_plot2 = st.columns(2)
                    
                    with col_plot1:
                        st.subheader("Convergencia del Error")
                        if len(results) > 1:
                            fig_err = go.Figure()
                            fig_err.add_trace(go.Scatter(x=results['Iteración'][1:], y=results['Error Rel. (%)'][1:], mode='lines+markers', line=dict(color='#10B981', width=3)))
                            fig_err.update_layout(title="Reducción del Error por Iteración", xaxis_title="Iteración", yaxis_title="Error Relativo (%)", yaxis_type="log", template="plotly_white")
                            st.plotly_chart(fig_err, use_container_width=True)
                        else:
                            st.info("Convergencia en un solo paso.")

                    with col_plot2:
                        st.subheader("Trayectoria de Optimización")
                        if len(vars_names) == 2:
                            # Gráfico de Contorno Interactivo con Plotly
                            f_lamb = sp.lambdify(vars_sym, expr, 'numpy')
                            x_vals = results[f'{vars_names[0]}'].values
                            y_vals = results[f'{vars_names[1]}'].values
                            
                            margin = max(1.0, max(abs(max(x_vals)-min(x_vals)), abs(max(y_vals)-min(y_vals))) * 0.5)
                            x_range = np.linspace(min(x_vals) - margin, max(x_vals) + margin, 100)
                            y_range = np.linspace(min(y_vals) - margin, max(y_vals) + margin, 100)
                            X, Y = np.meshgrid(x_range, y_range)
                            
                            try:
                                Z = f_lamb(X, Y)
                                fig_cont = go.Figure(data=go.Contour(z=Z, x=x_range, y=y_range, colorscale='Blues', opacity=0.8))
                                fig_cont.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', marker=dict(size=8, color='red'), line=dict(color='red', width=2), name="Camino"))
                                fig_cont.update_layout(title="Mapa de Contorno Interactivo", xaxis_title=vars_names[0], yaxis_title=vars_names[1], template="plotly_white")
                                st.plotly_chart(fig_cont, use_container_width=True)
                            except:
                                st.warning("La función es demasiado compleja para renderizar el fondo de contorno de forma segura.")
                        elif len(vars_names) == 1:
                            f_lamb = sp.lambdify(vars_sym, expr, 'numpy')
                            x_vals = results[f'{vars_names[0]}'].values
                            margin = max(1.0, (max(x_vals) - min(x_vals)) * 0.5)
                            x_range = np.linspace(min(x_vals) - margin, max(x_vals) + margin, 200)
                            y_range = [f_lamb(v) for v in x_range]
                            
                            fig_1d = go.Figure()
                            fig_1d.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='C(x)', line=dict(color='#1E3A8A')))
                            fig_1d.add_trace(go.Scatter(x=x_vals, y=results['C(x)'].values, mode='markers+lines', name='Pasos', marker=dict(color='red', size=8)))
                            fig_1d.update_layout(title="Evaluación de C(x)", template="plotly_white")
                            st.plotly_chart(fig_1d, use_container_width=True)
                        else:
                            st.info("La representación espacial gráfica está disponible solo para 1 o 2 variables.")

                with tab_datos:
                    st.subheader("Historial de Iteraciones")
                    st.dataframe(results, use_container_width=True)
                    # Añadir botón de descarga CSV
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Descargar Tabla (CSV)", data=csv, file_name='optimizacion_historial.csv', mime='text/csv')

                with tab_examen:
                    if method == "Método del Gradiente" and alpha_type == "Wolfe (Armijo)":
                        st.subheader("📝 Resolución Paso a Paso (Primera Iteración)")
                        st.markdown("**1. Gradiente Analítico**")
                        st.latex(r"\nabla C = \left( " + r", \quad ".join([sp.latex(g) for g in grad_exprs]) + r" \right)")
                        
                        g0_vals = np.array([results.iloc[0][f'g_{v}'] for v in vars_sym])
                        g0_tuple = f"({', '.join([f'{v:g}' for v in g0_vals])})"
                        st.markdown(f"Evaluando en $x_0$: $\\nabla C = {g0_tuple}$")
                        
                        st.markdown("**2. Dirección de Descenso**")
                        st.latex(rf"d^{{(0)}} = -\nabla C(x^{{(0)}}) = {tuple(-g0_vals) if len(g0_vals)>1 else -g0_vals[0]}")
                        
                        if not bt_log.empty:
                            st.markdown("**3. Búsqueda de Armijo**")
                            st.dataframe(bt_log)
                            st.success("Se muestra la tabla de búsqueda para encontrar el $\\alpha$ que cumple la condición.")
                    else:
                        st.info("El Modo Examen detallado está habilitado configurando: Método del Gradiente + Wolfe (Armijo).")

            else:
                st.error("Error de Dimensiones: El punto inicial no coincide con el número de variables.")
        except Exception as e:
            st.error(f"Se encontró un error matemático: {e}")

if __name__ == "__main__":
    main_app()
