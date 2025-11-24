import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.patches as mpatches

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="SmartPot Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para mejorar la apariencia
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. GENERACIÓN DE DATOS Y ENTRENAMIENTO (CON CACHÉ)
# ==========================================
# Usamos @st.cache_resource para que el entrenamiento solo ocurra UNA vez
# y no cada vez que mueves un slider. Esto hace la web muy rápida.

@st.cache_resource
def load_and_train_models():
    # --- Generación de Datos Sintéticos ---
    np.random.seed(42)
    synthetic_data = []
    for _ in range(500):
        temp = np.random.uniform(15, 40)
        hum = np.random.uniform(30, 95)

        # Lógica: Ventilador ON si Temp > 28 OR Humedad > 75 (con ruido)
        fan = 1 if (temp > 28 or hum > 75) else 0
        if np.random.rand() > 0.95: fan = 1 - fan

        ph = np.random.uniform(4.0, 9.0)
        tds = np.random.uniform(200, 900)

        # Lógica: Reductor ON si pH > 6.5 (con ruido)
        reducer = 1 if ph > 6.5 else 0
        if np.random.rand() > 0.95: reducer = 1 - reducer

        synthetic_data.append([ph, tds, temp, hum, reducer, fan])

    df_synth = pd.DataFrame(synthetic_data, columns=['pH', 'TDS', 'DHT_temp', 'DHT_humidity', 'pH_reducer', 'ex_fan'])

    # --- MODELO A: Regresión Logística (Ventilador) ---
    X_log = df_synth[['DHT_temp', 'DHT_humidity']]
    y_log = df_synth['ex_fan']
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_log, y_train_log)
    y_pred_log = log_reg.predict(X_test_log)

    # --- MODELO B: Red Neuronal (pH) ---
    X_nn = df_synth[['pH', 'TDS']]
    y_nn = df_synth['pH_reducer']
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_nn_scaled = scaler.fit_transform(X_train_nn)
    X_test_nn_scaled = scaler.transform(X_test_nn)

    nn_model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=2000, random_state=42)
    nn_model.fit(X_train_nn_scaled, y_train_nn)
    y_pred_nn = nn_model.predict(X_test_nn_scaled)

    # Métricas
    metrics = {
        "log_acc": accuracy_score(y_test_log, y_pred_log),
        "nn_acc": accuracy_score(y_test_nn, y_pred_nn),
        "y_test_log": y_test_log, "y_pred_log": y_pred_log,
        "y_test_nn": y_test_nn, "y_pred_nn": y_pred_nn
    }

    return log_reg, nn_model, scaler, df_synth, metrics

def draw_neural_network_fixed(ax, layer_sizes, input_labels=None, output_labels=None):
    """
    Dibuja la arquitectura de una red neuronal usando matplotlib,
    con corrección para evitar que los títulos de las capas se superpongan.
    """
    left, right, bottom, top = .1, .9, .1, .8  # Bajé un poco el 'top' para dar espacio a los títulos alternados
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Colores y estilos
    node_color = '#FFCCCC'  # Color rosado claro
    edge_color = '#FF0000'  # Borde rojo
    arrow_color = '#4682B4'  # Azul acero para flechas
    text_color = '#333333'

    # Función auxiliar para calcular posiciones
    def get_node_position(layer_idx, node_idx, layer_size):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        x = left + layer_idx * h_spacing
        y = layer_top - node_idx * v_spacing
        return x, y

    # 1. Dibujar conexiones (flechas) - Se dibujan primero para quedar al fondo
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                x1, y1 = get_node_position(n, m, layer_size_a)
                x2, y2 = get_node_position(n + 1, o, layer_size_b)
                # Ajustar inicio y fin de flecha para que no queden dentro del círculo
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx ** 2 + dy ** 2)
                shrink = v_spacing / 3.  # Radio aproximado del círculo

                # Factor de corrección para que la flecha toque el borde
                correction = 0.8

                ax.annotate("",
                            xy=(x2 - dx * (shrink * correction / dist), y2 - dy * (shrink * correction / dist)),
                            xytext=(x1 + dx * (shrink * correction / dist), y1 + dy * (shrink * correction / dist)),
                            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.2, shrinkA=0, shrinkB=0), zorder=1)

    # 2. Dibujar nodos (círculos)
    for n, layer_size in enumerate(layer_sizes):
        for m in range(layer_size):
            x, y = get_node_position(n, m, layer_size)
            # El radio depende del espaciado vertical
            radius = v_spacing / 2.8
            circle = mpatches.Circle((x, y), radius, edgecolor=edge_color, facecolor=node_color, lw=1.5, zorder=4)
            ax.add_artist(circle)
            # Numerar los nodos
            ax.text(x, y, str(m + 1), ha='center', va='center', fontsize=10, fontweight='bold', color=text_color,
                    zorder=5)

    # 3. Etiquetas de capas (CORREGIDO: Altura alternada)
    layer_names = ["Capa de Entrada", "Capa Oculta 1", "Capa Oculta 2", "Capa de Salida"]
    for i, name in enumerate(layer_names):
        x_pos = left + i * h_spacing
        # AQUÍ ESTÁ LA CORRECCIÓN: Alternamos la altura (y_pos)
        # Si el índice 'i' es par (0, 2), lo ponemos más abajo. Si es impar (1, 3), más arriba.
        y_offset = 0.05 if i % 2 == 0 else 0.12
        y_pos = top + y_offset

        ax.text(x_pos, y_pos, name, ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color)

    # 4. Etiquetas de Entrada laterales
    if input_labels:
        layer_size = layer_sizes[0]
        for i, label in enumerate(input_labels):
            x, y = get_node_position(0, i, layer_size)
            ax.text(x - 0.05, y, label, ha='right', va='center', fontweight='bold', fontsize=11, color=text_color)

    # 5. Etiquetas de Salida laterales
    if output_labels:
        layer_size = layer_sizes[-1]
        for i, label in enumerate(output_labels):
            x, y = get_node_position(len(layer_sizes) - 1, i, layer_size)
            ax.text(x + 0.05, y, label, ha='left', va='center', fontweight='bold', fontsize=11, color=text_color)

    # Ajustes finales del gráfico
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    # Expandir los límites para asegurar que los textos de arriba se vean
    ax.set_ylim(bottom - 0.1, top + 0.2)

# Cargar modelos
log_reg, nn_model, scaler, df_synth, metrics = load_and_train_models()

# ==========================================
# 2. INTERFAZ GRÁFICA WEB
# ==========================================

st.title("🌱 SmartPot - Dashboard de Control IoT")
st.markdown(
    "Sistema inteligente basado en **Regresión Logística** y **Redes Neuronales** para el control de invernaderos.")

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("🎛️ Panel de Control")
    st.info("Ajusta los sensores para ver la reacción de la IA.")

    st.subheader("Variables Ambientales")
    temp = st.slider("🌡️ Temperatura (°C)", 10.0, 40.0, 25.5, 0.1)
    hum = st.slider("💧 Humedad (%)", 30, 95, 60, 1)

    st.markdown("---")

    st.subheader("Variables del Agua")
    ph = st.slider("⚗️ pH del Agua", 4.0, 9.0, 7.0, 0.01)
    tds = st.slider("🧂 TDS (ppm)", 200, 900, 500, 10)

    st.caption("Modelo entrenado con 500 muestras sintéticas.")

# --- LÓGICA DE PREDICCIÓN EN TIEMPO REAL ---
# Preparar datos
in_log = pd.DataFrame([[temp, hum]], columns=['DHT_temp', 'DHT_humidity'])
in_nn = pd.DataFrame([[ph, tds]], columns=['pH', 'TDS'])

# Predecir Ventilador
fan_act = log_reg.predict(in_log)[0]
fan_prob = log_reg.predict_proba(in_log)[0][1]

# Predecir pH
in_nn_scaled = scaler.transform(in_nn)
ph_act = nn_model.predict(in_nn_scaled)[0]

# --- PANEL PRINCIPAL DE RESULTADOS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("💨 Control Ambiental (Fan)")
    if fan_act == 1:
        st.error(f"**VENTILADOR ENCENDIDO**\n\nProbabilidad de activación: {fan_prob * 100:.1f}%")
    else:
        st.success(f"**VENTILADOR APAGADO**\n\nProbabilidad de activación: {fan_prob * 100:.1f}%")

with col2:
    st.subheader("🧪 Control Químico (pH)")
    if ph_act == 1:
        st.error("**DOSIFICAR REDUCTOR**\n\nEl pH está por encima del umbral seguro.")
    else:
        st.success("**NIVEL ESTABLE**\n\nNo se requieren químicos en este momento.")

st.markdown("---")

# --- PESTAÑAS DE VISUALIZACIÓN ---
tab1, tab2 = st.tabs(["📊 Dinámica del Modelo", "📈 Evaluación de Precisión"])

with tab1:
    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        st.markdown("#### Distribución Temp vs Humedad")
        # Gráfico Scatter
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        scatter = ax1.scatter(df_synth['DHT_temp'], df_synth['DHT_humidity'],
                              c=df_synth['ex_fan'], cmap='coolwarm', alpha=0.5, edgecolors='none')

        # Agregar el punto actual seleccionado por el usuario
        ax1.scatter([temp], [hum], color='lime', s=200, marker='*', edgecolors='black', label='Tu Selección')

        ax1.set_xlabel("Temperatura (°C)")
        ax1.set_ylabel("Humedad (%)")
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig1)
        st.caption("Puntos rojos: Ventilador ON histórico. Estrella verde: Tu configuración actual.")

    with col_graph2:
        st.markdown("#### Curva de Aprendizaje (Red Neuronal)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(nn_model.loss_curve_, color='#673AB7', linewidth=2)
        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Error (Loss)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig2)

with tab2:
    col_mat1, col_mat2 = st.columns(2)

    with col_mat1:
        st.markdown(f"#### Matriz Regresión Logística (Acc: {metrics['log_acc']:.2f})")
        fig3, ax3 = plt.subplots()
        cm_log = confusion_matrix(metrics['y_test_log'], metrics['y_pred_log'])
        disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Apagado", "Encendido"])
        disp_log.plot(ax=ax3, cmap='Blues', colorbar=False)
        st.pyplot(fig3)

    with col_mat2:
        st.markdown(f"#### Matriz Red Neuronal (Acc: {metrics['nn_acc']:.2f})")
        fig4, ax4 = plt.subplots()
        cm_nn = confusion_matrix(metrics['y_test_nn'], metrics['y_pred_nn'])
        disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=["Normal", "Dosificar"])
        disp_nn.plot(ax=ax4, cmap='Purples', colorbar=False)
        st.pyplot(fig4)

st.markdown("---")
st.subheader("🧠 Arquitectura Exacta de la Red Neuronal (Modelo pH)")
st.markdown("""
Este diagrama representa fielmente el modelo `MLPClassifier` entrenado en el código:
- **Entrada (2 neuronas):** Reciben los valores normalizados de pH y TDS.
- **Capa Oculta 1 (5 neuronas):** Procesa las entradas iniciales.
- **Capa Oculta 2 (5 neuronas):** Procesa la información de la primera capa oculta para capturar patrones complejos.
- **Salida (1 neurona):** Determina la probabilidad de activar el reductor (Dosificar).
""")

# Definir la estructura exacta de tu modelo: [Entradas, Oculta1, Oculta2, Salida]
# Esto coincide con: X_nn (2 variables) y hidden_layer_sizes=(5, 5) y salida binaria (1)
rna_structure_exacta = [2, 5, 5, 1]
input_names_exactos = ["pH", "TDS"]
output_names_exactos = ["Salida\n(Dosificar)"]

# Crear la figura
fig_rna_fixed, ax_rna_fixed = plt.subplots(figsize=(9, 6)) # Un poco más ancho para dar espacio
draw_neural_network_fixed(ax_rna_fixed, rna_structure_exacta, input_labels=input_names_exactos, output_labels=output_names_exactos)

# Mostrar en Streamlit
st.pyplot(fig_rna_fixed)
st.caption("Figura 5: Diagrama generado dinámicamente de la arquitectura 2-5-5-1 utilizada en este dashboard.")