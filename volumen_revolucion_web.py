import ast
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import quad

st.set_page_config(page_title='Laboratorio de volúmenes de revolución', layout='wide')
st.title('Laboratorio interactivo: volúmenes de revolución')
st.caption('Método de discos y arandelas · Revolución alrededor de la recta horizontal y = k')

ALLOWED = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'sqrt': np.sqrt,
    'exp': np.exp, 'log': np.log, 'abs': np.abs, 'pi': np.pi, 'e': np.e,
}
OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}


def parse_expression(expr):
    tree = ast.parse(expr.replace('^', '**'), mode='eval')

    def compute(node, x):
        if isinstance(node, ast.Expression):
            return compute(node.body, x)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in ('x', 'pi', 'e'):
            return x if node.id == 'x' else ALLOWED[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](compute(node.left, x), compute(node.right, x))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = compute(node.operand, x)
            return value if isinstance(node.op, ast.UAdd) else -value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ALLOWED
            and callable(ALLOWED[node.func.id])
            and len(node.args) == 1
            and not node.keywords
        ):
            return ALLOWED[node.func.id](compute(node.args[0], x))
        raise ValueError('Expresión no admitida. Utilice x, operaciones y funciones matemáticas usuales.')

    def function(x):
        with np.errstate(all='ignore'):
            result = np.asarray(compute(tree, x), dtype=float)
        return np.broadcast_to(result, np.shape(x))

    return function


with st.sidebar:
    st.header('Datos de entrada')
    f_text = st.text_input('Función f(x)', 'x**2 + 1')
    g_text = st.text_input('Función g(x)', 'x')
    k = st.number_input('Eje de revolución y = k', value=-2.0, format='%.3f')
    a = st.number_input('Límite inferior a', value=0.0, format='%.3f')
    b = st.number_input('Límite superior b', value=3.0, format='%.3f')
    angle = st.slider('Ángulo de revolución (grados)', 5, 360, 288, 5)
    st.caption('Ejemplos: x**2, sqrt(x), sin(x), exp(x). También puede usar ^ para potencias.')

try:
    if not a < b:
        raise ValueError('El límite inferior debe ser menor que el superior.')
    if b - a > 1e5:
        raise ValueError('El intervalo es demasiado grande para esta visualización.')

    f, g = parse_expression(f_text), parse_expression(g_text)
    x = np.linspace(a, b, 1500)
    yf, yg = f(x), g(x)
    if not (np.all(np.isfinite(yf)) and np.all(np.isfinite(yg))):
        raise ValueError('Las funciones deben tener valores reales y finitos en todo el intervalo.')

    upper, lower = np.maximum(yf, yg), np.minimum(yf, yg)
    below = bool(np.all(k <= lower + 1e-8))
    above = bool(np.all(k >= upper - 1e-8))
    if not (below or above):
        raise ValueError('El eje de revolución atraviesa la región. Debe quedar fuera de ella.')

    def radii(t):
        ft, gt = f(t), g(t)
        top, bot = np.maximum(ft, gt), np.minimum(ft, gt)
        return (top - k, bot - k) if below else (k - bot, k - top)

    def integrand(t):
        R, r = radii(t)
        return float(np.pi * (R**2 - r**2))

    volume, error = quad(integrand, a, b, limit=200)
    if not np.isfinite(volume):
        raise ValueError('No se obtuvo un volumen finito.')

    # La expresión cambia de signo según el orden de las curvas y el lado del eje.
    f_above_g = bool(np.all(yf >= yg - 1e-8))
    g_above_f = bool(np.all(yg >= yf - 1e-8))
    if f_above_g and below:
        terms = r'(f(x)-k)^2-(g(x)-k)^2'
    elif g_above_f and below:
        terms = r'(g(x)-k)^2-(f(x)-k)^2'
    elif f_above_g and above:
        terms = r'(g(x)-k)^2-(f(x)-k)^2'
    elif g_above_f and above:
        terms = r'(f(x)-k)^2-(g(x)-k)^2'
    else:
        # Si las curvas se cruzan, el valor absoluto preserva el volumen positivo.
        terms = r'\left|(f(x)-k)^2-(g(x)-k)^2\right|'

    formula = (
        rf'V=\pi\int_{{{a:g}}}^{{{b:g}}}\left[{terms}\right]\,dx'
        rf'\approx {volume:.6f}'
    )
    st.latex(formula)

    col2d, col3d = st.columns(2)
    with col2d:
        st.subheader('Región generatriz (2D)')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x, y=yf, mode='lines', name=f'f(x) = {f_text}',
            line=dict(color='#C0392B', width=3),
        ))
        fig2.add_trace(go.Scatter(
            x=x, y=yg, mode='lines', name=f'g(x) = {g_text}',
            line=dict(color='#2674C7', width=3),
            fill='tonexty', fillcolor='rgba(230,182,107,0.45)',
        ))
        fig2.add_hline(y=k, line_color='navy', line_dash='dash', annotation_text=f'y = {k:g}')
        for edge in (a, b):
            fig2.add_trace(go.Scatter(
                x=[edge, edge], y=[float(f(edge)), float(g(edge))],
                mode='lines', line=dict(color='#A66A27', width=2), showlegend=False,
            ))
        fig2.update_layout(
            xaxis_title='x', yaxis_title='y', height=520,
            margin=dict(l=25, r=15, t=30, b=30),
            legend=dict(orientation='h', y=-0.2),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3d:
        st.subheader('Sólido de revolución (3D)')
        u = np.linspace(a, b, 100)
        theta = np.linspace(0, np.deg2rad(angle), 90)
        U, T = np.meshgrid(u, theta)
        R, r = radii(U)
        fig3 = go.Figure()
        fig3.add_trace(go.Surface(
            x=U, y=k + R * np.cos(T), z=R * np.sin(T),
            colorscale=[[0, '#4169E1'], [1, '#4169E1']],
            opacity=0.55, showscale=False, name='Superficie exterior',
        ))
        fig3.add_trace(go.Surface(
            x=U, y=k + r * np.cos(T), z=r * np.sin(T),
            colorscale=[[0, '#FF8C00'], [1, '#FF8C00']],
            opacity=0.82, showscale=False, name='Superficie interior',
        ))
        rho = np.linspace(0, 1, 25)[:, None]
        tt = theta[None, :]
        for edge in (a, b):
            R0, r0 = radii(edge)
            radius = r0 + rho * (R0 - r0)
            xx = np.full_like(radius * np.cos(tt), edge)
            fig3.add_trace(go.Surface(
                x=xx, y=k + radius * np.cos(tt), z=radius * np.sin(tt),
                colorscale=[[0, '#708090'], [1, '#708090']],
                opacity=0.6, showscale=False,
            ))
        fig3.add_trace(go.Scatter3d(
            x=[a, b], y=[k, k], z=[0, 0], mode='lines',
            line=dict(color='crimson', width=5, dash='dash'),
            name='Eje de revolución',
        ))
        radius_max = max(float(np.max(radii(u)[0])), 0.1)
        length = b - a
        scale = max(length, 2 * radius_max)
        visual_ratio = dict(
            x=max(length / scale, 0.25),
            y=max(2 * radius_max / scale, 0.25),
            z=max(2 * radius_max / scale, 0.25),
        )
        fig3.update_layout(
            height=590,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(title='x', visible=False, range=[a - 0.08 * length, b + 0.08 * length]),
                yaxis=dict(title='y', visible=False, range=[k - 1.12 * radius_max, k + 1.12 * radius_max]),
                zaxis=dict(title='z', visible=False, range=[-1.12 * radius_max, 1.12 * radius_max]),
                aspectmode='manual',
                aspectratio=visual_ratio,
                camera=dict(
                    eye=dict(x=1.8, y=1.35, z=0.85),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(type='orthographic'),
                ),
                dragmode='orbit',
            ),
            showlegend=False,
            uirevision='laboratorio-volumen-3d',
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.info('Puede rotar el sólido 3D con el ratón y modificar el ángulo desde el panel lateral.')
except Exception as exc:
    st.error(f'No se pudo actualizar el laboratorio: {exc}')
