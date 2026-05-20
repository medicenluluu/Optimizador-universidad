import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

# Configuración inicial de la página web
st.set_page_config(page_title="Optimizador Multivariable", layout="wide")

# ==========================================
# GESTIÓN DEL ESTADO DE SESIÓN (PANTALLA 0 - BIENVENIDA)
# ==========================================
if "username" not in st.session_state:
    st.session_state.username = None

# Paso 0: Pantalla de Ingreso de Nombre (Se muestra si no hay un nombre guardado)
if st.session_state.username is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #4F46E5;'>¡Bienvenido al Optimizador Numérico!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.1em;'>Aplicación para encontrar el mínimo de una función usando métodos avanzados de optimización.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center; margin-top:0;'>Identificación</h3>", unsafe_allow_html=True)
            nombre = st.text_input("Ingresa tu nombre para comenzar:", placeholder="Ej. Juan Pérez")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Ingresar", use_container_width=True, type="primary"):
                if nombre.strip() != "":
                    st.session_state.username = nombre.strip()
                    st.rerun() # Recarga la app para aplicar el inicio de sesión
                else:
                    st.error("Por favor, ingresa tu nombre antes de hacer clic en Ingresar.")
                    
    st.markdown("<p style='text-align: center; color: #9CA3AF; margin-top: 50px;'>Trabajo Grupal - Optimización Numérica</p>", unsafe_allow_html=True)
    st.stop() # Detiene la carga del resto del script hasta que el usuario inicie sesión

# ==========================================
# CÓDIGO DEL OPTIMIZADOR (Pantallas de Trabajo)
# ==========================================

# Menú lateral para saludar al usuario y opción de cerrar sesión
st.sidebar.markdown(f"### 👤 Usuario: {st.session_state.username}")
if st.sidebar.button("Cerrar Sesión 🚪", type="secondary", use_container_width=True):
    st.session_state.username = None
    st.rerun()

# --- FUNCIONES MATEMÁTICAS ---

def get_symbols_and_func(expr_str, num_vars):
    """Parsea el string de la función y genera símbolos en SymPy."""
    # Reemplazar notación común de potencias de JavaScript/humana a la de Python
    expr_cleaned = expr_str.replace('^', '**')
    # Crear símbolos dinámicamente: x1, x2, ..., xN
    symbols = sp.symbols(f'x1:{num_vars + 1}')
    # Compilar expresión sympy
    expr = sp.sympify(expr_cleaned)
    return symbols, expr

def evaluate_func(expr, symbols, point):
    """Evalúa la función en un punto numérico."""
    subs_dict = {sym: val for sym, val in zip(symbols, point)}
    return float(expr.subs(subs_dict))

def get_gradient(expr, symbols, point):
    """Calcula el gradiente analítico en un punto."""
    grad = []
    subs_dict = {sym: val for sym, val in zip(symbols, point)}
    for sym in symbols:
        deriv = sp.diff(expr, sym)
        grad.append(float(deriv.subs(subs_dict)))
    return np.array(grad, dtype=float)

def get_hessian(expr, symbols, point):
    """Calcula la matriz Hessiana analítica en un punto."""
    n = len(symbols)
    hessian = np.zeros((n, n))
    subs_dict = {sym: val for sym, val in zip(symbols, point)}
    for i in range(n):
        for j in range(n):
            deriv = sp.diff(sp.diff(expr, symbols[i]), symbols[j])
            hessian[i, j] = float(deriv.subs(subs_dict))
    return hessian

# --- BÚSQUEDA DE LÍNEA: CONDICIONES DE WOLFE ---

def line_search_wolfe(expr, symbols, x, pk, c1=1e-4, c2=0.9):
    """
    Búsqueda de línea con Condiciones Fuertes de Wolfe usando Backtracking adaptativo.
    """
    alpha = 1.0
    alpha_min = 1e-10
    factor = 0.5
    
    fx = evaluate_func(expr, symbols, x)
    grad_x = get_gradient(expr, symbols, x)
    m1 = np.dot(grad_x, pk)
    
    for _ in range(100):
        x_next = x + alpha * pk
        fx_next = evaluate_func(expr, symbols, x_next)
        
        # 1. Condición de Armijo (Suficiente Descenso)
        if fx_next <= fx + c1 * alpha * m1:
            grad_next = get_gradient(expr, symbols, x_next)
            m2 = np.dot(grad_next, pk)
            
            # 2. Condición de Curvatura Fuerte
            if abs(m2) <= c2 * abs(m1):
                return alpha
            
        alpha *= factor
        if alpha < alpha_min:
            return alpha_min
            
    return alpha

# --- ALGORITMOS DE OPTIMIZACIÓN ---

def optimize(method, expr, symbols, x0, max_iter, tol, c1, c2):
    x = np.array(x0, dtype=float)
    n = len(x)
    history = []
    path = [x.copy()]
    
    # Evaluar punto de partida
    fx = evaluate_func(expr, symbols, x)
    grad = get_gradient(expr, symbols, x)
    err = np.linalg.norm(grad)
    # Guardamos también el gradiente en history (índice 5) para el cálculo de beta en GC
    history.append((0, x.copy(), fx, err, 0.0, grad.copy()))
    
    # Inicialización para la dirección inicial de búsqueda
    pk = -grad
    
    for k in range(1, max_iter + 1):
        if err < tol:
            break
            
        # Determinar dirección de búsqueda (pk) según el método seleccionado
        if method == "Descenso de Gradiente":
            pk = -grad
            
        elif method == "Gradiente Conjugado":
            if k == 1:
                pk = -grad
            else:
                # Recuperar el gradiente de la iteración previa
                grad_prev = history[-2][5]
                pk_prev = pk # Dirección previa
                
                denom_fr = np.dot(grad_prev, grad_prev)
                
                # Calcular Beta (β) usando únicamente Fletcher-Reeves
                if denom_fr > 1e-15:
                    beta = np.dot(grad, grad) / denom_fr
                else:
                    beta = 0.0
                
                # Nueva dirección conjugada
                pk = -grad + beta * pk_prev
                
                # Heurística de reinicio: Si la dirección calculada no es de descenso estricto, reiniciamos al gradiente
                if np.dot(pk, grad) >= 0:
                    pk = -grad
                    
        elif method == "Newton":
            hess = get_hessian(expr, symbols, x)
            try:
                # Intentar resolver el sistema lineal para la dirección H * pk = -g
                pk = np.linalg.solve(hess, -grad)
                # Garantizar descenso si no es definida positiva
                if np.dot(pk, grad) >= 0:
                    pk = -grad
            except np.linalg.LinAlgError:
                # Fallback seguro a descenso de gradiente si la Hessiana es singular o indefinida
                pk = -grad
        
        # Búsqueda de línea con Condiciones de Wolfe
        alpha = line_search_wolfe(expr, symbols, x, pk, c1, c2)
        
        # Actualización de la posición
        x_next = x + alpha * pk
        fx_next = evaluate_func(expr, symbols, x_next)
        grad_next = get_gradient(expr, symbols, x_next)
        err_next = np.linalg.norm(grad_next)
        
        history.append((k, x_next.copy(), fx_next, err_next, alpha, grad_next.copy()))
        path.append(x_next.copy())
        
        # Siguiente iteración
        x = x_next
        grad = grad_next
        err = err_next
        
    return x, fx, k, err, history, np.array(path)

# ==========================================
# INTERFAZ DE CONFIGURACIÓN DEL USUARIO
# ==========================================

st.title("🧮 Optimizador Multivariable")
st.write(f"¡Hola **{st.session_state.username}**! Configura los parámetros en el panel izquierdo y haz clic en 'Ejecutar Optimización'.")

# Panel lateral con todas las opciones solicitadas por tu profesor
st.sidebar.header("⚙️ Parámetros de Entrada")

num_vars = st.sidebar.number_input("Número de variables (N)", min_value=1, max_value=10, value=2, step=1)
method = st.sidebar.selectbox("Método de optimización", ["Descenso de Gradiente", "Gradiente Conjugado", "Newton"])

# Mensaje estático indicando el uso de Fletcher-Reeves cuando se selecciona Gradiente Conjugado
if method == "Gradiente Conjugado":
    st.sidebar.info("Gradiente Conjugado implementado con la fórmula clásica de **Fletcher-Reeves**.")

func_str = st.sidebar.text_input("Función objetivo f(x)", value="100*(x2 - x1^2)^2 + (1 - x1)^2")

# Generar puntos de partida dinámicos para N variables
st.sidebar.markdown("### Punto inicial x0")
x0 = []
cols_x0 = st.sidebar.columns(num_vars)
for i in range(num_vars):
    val = cols_x0[i].number_input(f"x{i+1}", value=-1.2 if i==0 else 1.0)
    x0.append(val)

max_iter = st.sidebar.number_input("Iteraciones máximas", min_value=1, max_value=2000, value=200, step=10)
tol = st.sidebar.number_input("Tolerancia (Convergencia)", value=1e-5, format="%.2e")

st.sidebar.markdown("### 🔍 Parámetros de Wolfe")
c1 = st.sidebar.slider("c1 (Armijo - Descenso)", min_value=1e-5, max_value=1e-1, value=1e-4, format="%.5f")
c2 = st.sidebar.slider("c2 (Curvatura)", min_value=1e-2, max_value=0.99, value=0.9 if method != "Newton" else 0.1)

# Botón para iniciar los cálculos
if st.sidebar.button("🚀 Ejecutar Optimización", type="primary", use_container_width=True):
    try:
        # Validación matemática inicial
        symbols, expr = get_symbols_and_func(func_str, num_vars)
        
        st.info("Calculando optimización... por favor espera.")
        
        # Ejecutar el cálculo en segundo plano
        x_min, f_min, total_iter, final_err, history, path = optimize(
            method, expr, symbols, x0, max_iter, tol, c1, c2
        )
        
        st.success("¡Optimización completada con éxito!")
        
        # --- DISEÑO Y PRESENTACIÓN DE RESULTADOS ---
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Valor Mínimo f(x*)", f"{f_min:.6e}")
        col_m2.metric("Iteraciones", f"{total_iter}")
        col_m3.metric("Gradiente Final (Error)", f"{final_err:.6e}")
        col_m4.metric("Estado de parada", "Convergencia" if final_err < tol else "Máx Iteraciones")
        
        # Mostrar las coordenadas del mínimo encontrado de forma ordenada
        st.markdown("### 📍 Punto Mínimo Encontrado ($x^*$)")
        pt_dict = {f"Variable x{i+1}": [val] for i, val in enumerate(x_min)}
        st.table(pd.DataFrame(pt_dict))
        
        # PESTAÑAS DE VISUALIZACIÓN DE DATOS (Convergencia, Tabla y Gráficas Espaciales)
        tab1, tab2, tab3 = st.tabs(["📉 Convergencia", "📊 Tabla de Iteraciones", "🗺️ Análisis Espacial (2D/3D)"])
        
        with tab1:
            st.subheader("Gráfico de Convergencia")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            iters = [h[0] for h in history]
            errors = [h[3] for h in history]
            
            ax.plot(iters, errors, color="#4F46E5", linewidth=2.5, marker='o', markersize=4, label='||∇f(x)||')
            ax.set_yscale('log')
            ax.set_xlabel('Número de Iteración (k)', fontsize=11)
            ax.set_ylabel('Norma del Gradiente (Error)', fontsize=11)
            ax.set_title('Convergencia del Algoritmo (Escala Semilogarítmica)', fontsize=12, fontweight='bold')
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Historial detallado de iteraciones")
            # Construir la estructura de la tabla de iteraciones solicitada
            rows = []
            for h in history:
                row = {
                    "Iteración (k)": h[0],
                    "Punto x(k)": [round(val, 6) for val in h[1]],
                    "f(x)": h[2],
                    "||∇f(x)|| (Error)": h[3],
                    "Paso (α)": h[4]
                }
                rows.append(row)
            df_hist = pd.DataFrame(rows)
            st.dataframe(df_hist, use_container_width=True)
            
        with tab3:
            # Gráficos de curvas de nivel y 3D (disponibles solo para optimizaciones en 2D)
            if num_vars == 2:
                st.subheader("Representación Topográfica y de Superficie")
                
                xs = path[:, 0]
                ys = path[:, 1]
                
                margin_x = max((xs.max() - xs.min()) * 0.3, 1.0)
                margin_y = max((ys.max() - ys.min()) * 0.3, 1.0)
                
                x_lim = np.linspace(xs.min() - margin_x, xs.max() + margin_x, 100)
                y_lim = np.linspace(ys.min() - margin_y, ys.max() + margin_y, 100)
                
                X, Y = np.meshgrid(x_lim, y_lim)
                Z = np.zeros_like(X)
                
                for i in range(len(x_lim)):
                    for j in range(len(y_lim)):
                        Z[j, i] = evaluate_func(expr, symbols, [X[j, i], Y[j, i]])
                
                # Renderizado de los gráficos usando matplotlib
                fig_spatial, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Gráfica 1: Curvas de nivel cenitales
                contour = ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
                ax1.clabel(contour, inline=True, fontsize=8)
                ax1.plot(xs, ys, color='red', marker='o', markersize=4, linewidth=1.5, label='Trayectoria')
                ax1.scatter(x0[0], x0[1], color='blue', s=80, zorder=5, label='Inicio (x0)')
                ax1.scatter(x_min[0], x_min[1], color='gold', marker='*', s=200, zorder=5, label='Mínimo (x*)')
                ax1.set_title("Trayectoria en Curvas de Nivel", fontsize=12, fontweight='bold')
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")
                ax1.legend()
                ax1.grid(True, linestyle=":", alpha=0.6)
                
                # Gráfica 2: Superficie en perspectiva 3D
                ax2 = fig_spatial.add_subplot(1, 2, 2, projection='3d')
                surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)
                ax2.plot(xs, ys, [evaluate_func(expr, symbols, p) for p in path], color='red', marker='o', markersize=2, linewidth=2, zorder=10)
                ax2.set_title("Superficie f(x1, x2) en 3D", fontsize=12, fontweight='bold')
                ax2.set_xlabel("x1")
                ax2.set_ylabel("x2")
                ax2.set_zlabel("f(x)")
                fig_spatial.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
                
                st.pyplot(fig_spatial)
            else:
                st.info("Las visualizaciones espaciales 2D/3D están limitadas exclusivamente para funciones de 2 variables.")
                
    except Exception as e:
        st.error(f"Ocurrió un error en la evaluación matemática o computacional: {e}")
        st.warning("Verifica que la función esté bien escrita (ej: usar x1, x2 y operadores explícitos como '*' para multiplicaciones).")
