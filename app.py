import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

st.set_page_config(page_title="Optimizador de Funciones", layout="wide")

def parse_function(func_str, variables):
    """Convierte un string a una función simbólica de sympy."""
    try:
        # Reemplazar '^' por '**' para que sympy lo entienda
        func_str = func_str.replace('^', '**')
        # Parsear la expresión
        expr = sp.sympify(func_str)
        return expr
    except Exception as e:
        return None

def compute_gradient(expr, variables):
    """Calcula el gradiente simbólico."""
    grad = [sp.diff(expr, var) for var in variables]
    return grad

def compute_hessian(expr, variables):
    """Calcula la matriz Hessiana simbólica."""
    n = len(variables)
    hessian = sp.Matrix(n, n, lambda i, j: sp.diff(sp.diff(expr, variables[i]), variables[j]))
    return hessian

def evaluate_func(expr, variables, point):
    """Evalúa la función en un punto dado."""
    subs = {var: val for var, val in zip(variables, point)}
    return float(expr.subs(subs))

def evaluate_grad(grad_exprs, variables, point):
    """Evalúa el gradiente en un punto dado."""
    subs = {var: val for var, val in zip(variables, point)}
    return np.array([float(g.subs(subs)) for g in grad_exprs])

def evaluate_hessian(hessian_expr, variables, point):
    """Evalúa la matriz Hessiana en un punto dado."""
    subs = {var: val for var, val in zip(variables, point)}
    # Convertir sympy Matrix a numpy array
    return np.array(hessian_expr.subs(subs)).astype(np.float64)

def line_search_wolfe(expr, grad_exprs, variables, x_k, p_k, alpha_init, c1, c2, max_ls_iter=20):
    """
    Búsqueda de línea con condiciones fuerte de Wolfe.
    x_k: punto actual
    p_k: dirección de búsqueda
    alpha_init: tamaño de paso inicial (alfa)
    c1: parámetro de Armijo (ro)
    c2: parámetro de curvatura (teta)
    """
    alpha = alpha_init
    f_k = evaluate_func(expr, variables, x_k)
    g_k = evaluate_grad(grad_exprs, variables, x_k)
    
    # Derivada direccional inicial
    dir_deriv = np.dot(g_k, p_k)
    
    if dir_deriv >= 0:
        return None # No es dirección de descenso
        
    alpha_min = 0
    alpha_max = np.inf
    
    for _ in range(max_ls_iter):
        x_new = x_k + alpha * p_k
        f_new = evaluate_func(expr, variables, x_new)
        
        # 1. Condición de Armijo (Wolfe 1)
        if f_new > f_k + c1 * alpha * dir_deriv:
            alpha_max = alpha
            alpha = (alpha_min + alpha_max) / 2
            continue
            
        g_new = evaluate_grad(grad_exprs, variables, x_new)
        dir_deriv_new = np.dot(g_new, p_k)
        
        # 2. Condición Fuerte de Curvatura (Wolfe 2 fuerte)
        if abs(dir_deriv_new) > c2 * abs(dir_deriv):
            # Si se cumple Armijo pero no curvatura, necesitamos un alpha más grande
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha = alpha * 2
            else:
                alpha = (alpha_min + alpha_max) / 2
            continue
            
        # Si cumple ambas, retornar alpha
        return alpha
        
    # Si no converge, retornar el mejor alpha encontrado (normalmente alpha_min)
    return alpha_min if alpha_min > 0 else alpha_init * 0.1

def optimize_gradient_descent(expr, vars_sym, x0, tol, max_iter, alpha_init, c1, c2):
    """Método del Gradiente (Steepest Descent)."""
    grad_exprs = compute_gradient(expr, vars_sym)
    
    x = np.array(x0, dtype=float)
    history = []
    errors = []
    
    f_val = evaluate_func(expr, vars_sym, x)
    g_val = evaluate_grad(grad_exprs, vars_sym, x)
    error = np.linalg.norm(g_val)
    
    history.append((x.copy(), f_val))
    errors.append(error)
    
    iteration = 0
    while error > tol and iteration < max_iter:
        p = -g_val # Dirección de máximo descenso
        
        # Búsqueda de línea
        alpha = line_search_wolfe(expr, grad_exprs, vars_sym, x, p, alpha_init, c1, c2)
        if alpha is None or alpha < 1e-10:
            st.warning(f"Búsqueda de línea falló o paso muy pequeño en iteración {iteration}.")
            break
            
        x = x + alpha * p
        f_val = evaluate_func(expr, vars_sym, x)
        g_val = evaluate_grad(grad_exprs, vars_sym, x)
        error = np.linalg.norm(g_val)
        
        history.append((x.copy(), f_val))
        errors.append(error)
        iteration += 1
        
    return x, f_val, iteration, error, history, errors

def optimize_conjugate_gradient(expr, vars_sym, x0, tol, max_iter, alpha_init, c1, c2):
    """Método del Gradiente Conjugado (Fletcher-Reeves)."""
    grad_exprs = compute_gradient(expr, vars_sym)
    
    x = np.array(x0, dtype=float)
    history = []
    errors = []
    
    f_val = evaluate_func(expr, vars_sym, x)
    g_val = evaluate_grad(grad_exprs, vars_sym, x)
    error = np.linalg.norm(g_val)
    
    p = -g_val
    
    history.append((x.copy(), f_val))
    errors.append(error)
    
    iteration = 0
    while error > tol and iteration < max_iter:
        alpha = line_search_wolfe(expr, grad_exprs, vars_sym, x, p, alpha_init, c1, c2)
        if alpha is None or alpha < 1e-10:
             st.warning(f"Búsqueda de línea falló o paso muy pequeño en iteración {iteration}. Reiniciando dirección.")
             p = -g_val # Reinicio a steepest descent
             alpha = line_search_wolfe(expr, grad_exprs, vars_sym, x, p, alpha_init, c1, c2)
             if alpha is None or alpha < 1e-10: break
             
        x_new = x + alpha * p
        g_new = evaluate_grad(grad_exprs, vars_sym, x_new)
        
        # Fletcher-Reeves beta
        beta = np.dot(g_new, g_new) / (np.dot(g_val, g_val) + 1e-10)
        p_new = -g_new + beta * p
        
        # Asegurar dirección de descenso
        if np.dot(g_new, p_new) >= 0:
            p_new = -g_new
            
        x = x_new
        g_val = g_new
        p = p_new
        
        f_val = evaluate_func(expr, vars_sym, x)
        error = np.linalg.norm(g_val)
        
        history.append((x.copy(), f_val))
        errors.append(error)
        iteration += 1
        
    return x, f_val, iteration, error, history, errors

def optimize_newton(expr, vars_sym, x0, tol, max_iter, alpha_init, c1, c2):
    """Método de Newton con Modificación de Hessiana."""
    grad_exprs = compute_gradient(expr, vars_sym)
    hess_expr = compute_hessian(expr, vars_sym)
    
    x = np.array(x0, dtype=float)
    history = []
    errors = []
    
    f_val = evaluate_func(expr, vars_sym, x)
    g_val = evaluate_grad(grad_exprs, vars_sym, x)
    error = np.linalg.norm(g_val)
    
    history.append((x.copy(), f_val))
    errors.append(error)
    
    iteration = 0
    while error > tol and iteration < max_iter:
        H = evaluate_hessian(hess_expr, vars_sym, x)
        
        # Asegurar que H sea definida positiva (Regularización simple)
        try:
            # Intento resolver H * p = -g
            p = np.linalg.solve(H, -g_val)
            
            # Verificar si es dirección de descenso
            if np.dot(g_val, p) >= 0:
                raise np.linalg.LinAlgError("Dirección no de descenso")
        except np.linalg.LinAlgError:
            # Si no es definida positiva o no es descenso, usar gradiente
            p = -g_val
            
        alpha = line_search_wolfe(expr, grad_exprs, vars_sym, x, p, alpha_init, c1, c2)
        if alpha is None or alpha < 1e-10:
             st.warning(f"Búsqueda de línea falló en iteración {iteration}.")
             break
             
        x = x + alpha * p
        f_val = evaluate_func(expr, vars_sym, x)
        g_val = evaluate_grad(grad_exprs, vars_sym, x)
        error = np.linalg.norm(g_val)
        
        history.append((x.copy(), f_val))
        errors.append(error)
        iteration += 1
        
    return x, f_val, iteration, error, history, errors

def main():
    st.title("🚀 Optimizador Matemático Avanzado")
    st.markdown("Encuentra el mínimo de funciones multidimensionales utilizando métodos avanzados con condiciones de Wolfe.")
    
    # --- VALOR AGREGADO: Gamificación / Bienvenida ---
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""

    with st.sidebar:
        st.header("👤 Perfil de Usuario")
        name_input = st.text_input("Ingresa tu nombre para comenzar:", value=st.session_state.user_name)
        if name_input != st.session_state.user_name:
            st.session_state.user_name = name_input
        
        if st.session_state.user_name:
            st.success(f"¡Hola, {st.session_state.user_name}! Listo para optimizar.")
            st.metric(label="🌟 Puntos Acumulados", value=st.session_state.score)
        else:
            st.info("Ingresa tu nombre para acumular puntos.")
            
        st.divider()

        # --- PARÁMETROS DE ENTRADA ---
        st.header("⚙️ Configuración")
        
        num_vars = st.number_input("Número de variables (n)", min_value=1, max_value=5, value=2)
        
        method = st.selectbox(
            "Método de Optimización",
            ("Gradiente (Steepest Descent)", "Gradiente Conjugado", "Newton")
        )
        
        st.divider()
        st.subheader("Parámetros del Algoritmo")
        max_iter = st.number_input("Número máximo de iteraciones", min_value=10, max_value=10000, value=100)
        tol = st.number_input("Tolerancia de convergencia", min_value=1e-10, max_value=1.0, value=1e-6, format="%.6f")
        
        st.divider()
        st.subheader("Condiciones de Wolfe")
        st.markdown("Ajusta los parámetros de búsqueda de línea.")
        alpha_init = st.number_input("Alfa inicial (Tamaño de paso)", min_value=0.01, max_value=10.0, value=1.0)
        c1 = st.slider("Rho (Armijo / Wolfe 1)", min_value=0.0001, max_value=0.5, value=1e-4, format="%.4f")
        # c2 debe ser mayor que c1 y menor que 1
        c2 = st.slider("Theta (Curvatura / Wolfe 2)", min_value=c1, max_value=0.99, value=0.9, format="%.4f")

    # --- ÁREA PRINCIPAL ---
    # Variables simbólicas basadas en n
    var_names = [f'x{i+1}' for i in range(num_vars)]
    vars_sym = sp.symbols(' '.join(var_names))
    if num_vars == 1:
        vars_sym = (vars_sym,) # Asegurar que sea tupla si es 1 var

    st.subheader("📝 Definición del Problema")
    st.markdown(f"**Variables disponibles:** `{'`, `'.join(var_names)}`")
    
    col_func, col_start = st.columns([2, 1])
    with col_func:
        func_input = st.text_input(
            "Función Objetivo f(x)", 
            value="100*(x2 - x1^2)^2 + (1 - x1)^2" if num_vars == 2 else "+".join([f"{v}^2" for v in var_names]),
            help="Usa sintaxis matemática estándar. Ejemplo: x1^2 + sin(x2)"
        )
    
    with col_start:
        st.markdown("**Punto de Partida (separado por comas)**")
        start_point_str = st.text_input(f"x0 ∈ ℝ^{num_vars}", value="-1.2, 1.0" if num_vars==2 else ",".join(["1.0"]*num_vars))
    
    # Botón de ejecución
    if st.button("🚀 Iniciar Optimización", type="primary", use_container_width=True):
        if not st.session_state.user_name:
            st.error("⚠️ Por favor, ingresa tu nombre en la barra lateral antes de ejecutar.")
            return

        # 1. Parsear función
        expr = parse_function(func_input, vars_sym)
        if expr is None:
            st.error("❌ Error al procesar la función. Revisa la sintaxis matemática.")
            return
            
        # 2. Parsear punto inicial
        try:
            x0 = [float(x.strip()) for x in start_point_str.split(',')]
            if len(x0) != num_vars:
                st.error(f"❌ El punto de partida debe tener exactamente {num_vars} componentes.")
                return
        except ValueError:
            st.error("❌ Formato de punto de partida inválido. Usa números separados por comas.")
            return

        # --- VALOR AGREGADO: Puntaje por función válida ---
        st.session_state.score += 10
        st.success("✅ Función válida. ¡+10 Puntos!")

        # 3. Ejecutar Optimización
        with st.spinner(f"Ejecutando método de {method}..."):
            if method == "Gradiente (Steepest Descent)":
                res_x, res_f, res_iter, res_err, history, errors = optimize_gradient_descent(
                    expr, vars_sym, x0, tol, max_iter, alpha_init, c1, c2
                )
            elif method == "Gradiente Conjugado":
                 res_x, res_f, res_iter, res_err, history, errors = optimize_conjugate_gradient(
                    expr, vars_sym, x0, tol, max_iter, alpha_init, c1, c2
                )
            else: # Newton
                 res_x, res_f, res_iter, res_err, history, errors = optimize_newton(
                    expr, vars_sym, x0, tol, max_iter, alpha_init, c1, c2
                )

        # --- RESULTADOS ---
        st.divider()
        st.header("📊 Resultados de la Optimización")
        
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        col1.metric("Iteraciones Realizadas", res_iter)
        col2.metric("Valor Mínimo f(x*)", f"{res_f:.6e}")
        col3.metric("Error Final ||∇f||", f"{res_err:.6e}")
        
        # Punto mínimo encontrado
        st.subheader("📍 Punto Mínimo Encontrado (x*)")
        formatted_x = [f"{var_names[i]} = {val:.6f}" for i, val in enumerate(res_x)]
        st.code("\n".join(formatted_x), language="text")
        
        # Gráfico de Convergencia
        st.subheader("📉 Gráfico de Convergencia")
        
        # Utilizamos pandas para estructurar los datos del gráfico
        df_errors = pd.DataFrame({
            'Iteración': range(len(errors)), 
            'Error': errors
        })
        
        # Creamos el gráfico con matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_errors['Iteración'], df_errors['Error'], marker='o', linestyle='-', color='#1f77b4', label='Error (Norma del Gradiente)')
        
        ax.set_title("Error vs Número de Iteraciones", fontsize=14)
        ax.set_xlabel("Iteración (k)", fontsize=12)
        ax.set_ylabel("Error ||∇f(x_k)||", fontsize=12)
        ax.set_yscale("log") # Escala logarítmica para ver la convergencia
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend()
        
        # Mostramos el gráfico de matplotlib en Streamlit
        st.pyplot(fig)

        # Si convergió, dar más puntos
        if res_err <= tol:
            st.session_state.score += 50
            st.balloons()
            st.success(f"¡Convergencia exitosa! Criterio de parada alcanzado. ¡+50 Puntos para {st.session_state.user_name}!")
        else:
            st.warning("Se alcanzó el número máximo de iteraciones sin lograr la tolerancia deseada.")

if __name__ == "__main__":
    main()
