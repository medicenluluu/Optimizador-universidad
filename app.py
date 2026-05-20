import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, sympify, lambdify

st.set_page_config(page_title="Optimizador Avanzado", layout="wide")

def evaluar_f(func, x):
    return func(x)

def obtener_gradiente(func, x):
    h = 1e-6
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_forward = np.array(x, dtype=float)
        x_backward = np.array(x, dtype=float)
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (evaluar_f(func, x_forward) - evaluar_f(func, x_backward)) / (2 * h)
    return grad

def obtener_hessiana(func, x):
    h = 1e-4
    n = len(x)
    H = np.zeros((n, n))
    fx = evaluar_f(func, x)
    for i in range(n):
        for j in range(i + 1):
            x_ij = np.array(x, dtype=float)
            x_ij[i] += h; x_ij[j] += h
            x_i = np.array(x, dtype=float)
            x_i[i] += h
            x_j = np.array(x, dtype=float)
            x_j[j] += h
            d2f = (evaluar_f(func, x_ij) - evaluar_f(func, x_i) - evaluar_f(func, x_j) + fx) / (h * h)
            H[i, j] = d2f
            H[j, i] = d2f
    return H

def busqueda_linea_wolfe(func, x, p, grad, c1=0.0001, c2=0.9):
    alpha = 1.0
    fx = evaluar_f(func, x)
    dir_deriv = np.dot(grad, p)
    if dir_deriv >= 0:
        return 1e-4
    rho = 0.5
    max_iter = 20
    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = evaluar_f(func, x_new)
        if fx_new <= fx + c1 * alpha * dir_deriv:
            grad_new = obtener_gradiente(func, x_new)
            dir_deriv_new = np.dot(grad_new, p)
            if dir_deriv_new >= c2 * dir_deriv:
                return alpha
            if alpha < 1e-3:
                return alpha
        alpha *= rho
    return alpha

def ejecutar_optimizacion(func, x0, metodo, max_iter, tol, c1, c2):
    x = np.array(x0, dtype=float)
    n = len(x)
    historial_error = []
    historial_trayectoria = [x.copy()]
    historial_f = []
    historial_alpha = [0.0]
    
    iteracion = 0
    grad = obtener_gradiente(func, x)
    error = np.linalg.norm(grad)
    p_prev = None
    grad_prev = None
    
    historial_error.append(error)
    historial_f.append(evaluar_f(func, x))
    
    while iteracion < max_iter and error > tol:
        if metodo == 'Descenso de Gradiente':
            p = -grad
        elif metodo == 'Gradiente Conjugado':
            if iteracion == 0 or iteracion % n == 0:
                p = -grad
            else:
                beta = np.dot(grad, grad) / np.dot(grad_prev, grad_prev)
                p = -grad + beta * p_prev
                if np.dot(p, grad) >= 0:
                    p = -grad
        elif metodo == 'Newton':
            H = obtener_hessiana(func, x)
            try:
                p = np.linalg.solve(H, -grad)
                if np.dot(p, grad) >= 0:
                    raise ValueError
            except:
                p = -grad
                
        p_prev = p.copy()
        grad_prev = grad.copy()
        
        alpha = busqueda_linea_wolfe(func, x, p, grad, c1, c2)
        x = x + alpha * p
        grad = obtener_gradiente(func, x)
        error = np.linalg.norm(grad)
        
        historial_error.append(error)
        historial_trayectoria.append(x.copy())
        historial_f.append(evaluar_f(func, x))
        historial_alpha.append(alpha)
        iteracion += 1
        
    return {
        'x_final': x, 'f_final': evaluar_f(func, x), 'iteraciones': iteracion,
        'error_final': error, 'historial_error': historial_error,
        'trayectoria': historial_trayectoria, 'historial_f': historial_f,
        'historial_alpha': historial_alpha
    }

def graficar_resultados(func, resultados):
    trayectoria = np.array(resultados['trayectoria'])
    hist_error = resultados['historial_error']
    n_vars = trayectoria.shape[1]
    
    if n_vars == 2:
        fig = plt.figure(figsize=(18, 5))
        
        # Gráfico Convergencia
        ax1 = fig.add_subplot(131)
        ax1.plot(hist_error, color='blue', marker='o', markersize=3)
        ax1.set_yscale('log')
        ax1.set_title('Convergencia: Error vs Iteraciones')
        ax1.set_xlabel('Iteración k')
        ax1.set_ylabel('||∇f|| (log)')
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        
        # Preparar Malla
        x_vals = trayectoria[:, 0]
        y_vals = trayectoria[:, 1]
        margen_x = max((max(x_vals) - min(x_vals)) * 0.4, 1.0)
        margen_y = max((max(y_vals) - min(y_vals)) * 0.4, 1.0)
        
        X, Y = np.meshgrid(
            np.linspace(min(x_vals) - margen_x, max(x_vals) + margen_x, 50),
            np.linspace(min(y_vals) - margen_y, max(y_vals) + margen_y, 50)
        )
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func([X[i, j], Y[i, j]])
                
        # Gráfico Contornos
        ax2 = fig.add_subplot(132)
        ax2.contour(X, Y, Z, levels=30, cmap='viridis')
        ax2.plot(x_vals, y_vals, 'r.-', label='Trayectoria', markersize=5)
        ax2.plot(x_vals[-1], y_vals[-1], 'y*', markersize=15, label='Mínimo')
        ax2.set_title('Curvas de Nivel')
        ax2.legend()
        
        # Gráfico 3D
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax3.set_title('Superficie 3D')
        
    else:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(hist_error, color='blue', marker='o')
        plt.yscale('log')
        plt.title('Convergencia: Error vs Iteraciones')
        plt.xlabel('Iteración k')
        plt.ylabel('||∇f|| (log)')
        plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    return fig

st.title("📈 Optimizador de Funciones - Proyecto Grupal")
st.markdown("Encuentra el mínimo de una función usando métodos numéricos y condiciones de Wolfe.")

with st.sidebar:
    st.header("1. Parámetros de Entrada")
    func_str = st.text_input("Función f(x)", value="100 * (x2 - x1**2)**2 + (1 - x1)**2")
    st.caption("Usa x1, x2... y ** para exponentes (Ej. x1**2)")
    
    x0_str = st.text_input("Punto inicial (separado por comas)", value="-1.2, 1.0")
    
    metodo = st.selectbox("Método de Optimización", 
                         ['Descenso de Gradiente', 'Gradiente Conjugado', 'Newton'])
    
    max_iter = st.number_input("Iteraciones Máximas", min_value=10, max_value=5000, value=1000)
    tol = st.number_input("Tolerancia", format="%e", value=1e-5)
    
    st.subheader("Condiciones de Wolfe")
    c1 = st.number_input("c1 (Armijo)", format="%f", value=0.0001)
    c2 = st.number_input("c2 (Curvatura)", format="%f", value=0.9)
    
    ejecutar = st.button("🚀 Ejecutar Optimización", use_container_width=True)

if ejecutar:
    try:
        # Pre-procesamiento de entradas
        func_str_clean = func_str.replace('^', '**')
        x0 = [float(val.strip()) for val in x0_str.split(',')]
        num_vars = len(x0)
        
        # Procesar con Sympy
        with st.spinner('Analizando función matemática...'):
            syms = symbols(f'x1:{num_vars + 1}')
            expr = sympify(func_str_clean)
            func_lambda = lambdify([syms], expr, "numpy")
            
            def funcion_objetivo(x_array):
                return float(func_lambda(x_array))
                
            # Validar que la función es evaluable en x0
            funcion_objetivo(x0)
            
        with st.spinner('Ejecutando algoritmo de optimización...'):
            resultados = ejecutar_optimizacion(
                funcion_objetivo, x0, metodo, max_iter, tol, c1, c2
            )
            
        # Mostrar Resultados
        st.success("¡Optimización completada con éxito!")
        
        st.subheader("Resultados")
        col1, col2, col3, col4 = st.columns(4)
        x_min_str = "[" + ", ".join([f"{v:.4f}" for v in resultados['x_final']]) + "]"
        col1.metric("Punto Mínimo x*", x_min_str)
        col2.metric("Valor f(x*)", f"{resultados['f_final']:.6f}")
        col3.metric("Iteraciones", resultados['iteraciones'])
        col4.metric("Error Final ||∇f||", f"{resultados['error_final']:.2e}")
        
        # Mostrar Gráficas
        st.subheader("Análisis Gráfico")
        fig = graficar_resultados(funcion_objetivo, resultados)
        st.pyplot(fig)
        
        # Mostrar Tabla usando Pandas
        st.subheader("Tabla de Iteraciones")
        df = pd.DataFrame({
            'Iteración': range(len(resultados['historial_f'])),
            'x(k)': [str(np.round(x, 4)) for x in resultados['trayectoria']],
            'f(x)': resultados['historial_f'],
            '||∇f||': resultados['historial_error'],
            'Paso (α)': resultados['historial_alpha']
        })
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error al evaluar la función o ejecutar el algoritmo. Revisa la sintaxis. Detalle: {e}")
        