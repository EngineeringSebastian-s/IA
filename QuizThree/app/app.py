import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from io import StringIO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SmartPot Dashboard", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS (Para que se vea elegante) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #f0f2f6;
        color: black;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. PREPARACIÓN DE DATOS Y MODELOS (CON CACHÉ) ---
# Usamos @st.cache_resource para que ESTO NO SE REPITA cada vez que mueves un slider.
# Solo se ejecuta una vez al arrancar la app.
@st.cache_resource
def entrenar_modelos():
    # --- Cargar / Simular Datos ---
    try:
        df = pd.read_csv("data/IoTData --Raw--.csv")
    except FileNotFoundError:
        # Fallback (Tu simulación original)
        csv_data = """
        "id","timestamp","pH","TDS","water_level","DHT_temp","DHT_humidity","water_temp","pH_reducer","add_water","nutrients_adder","humidifier","ex_fan"
        "1","2023-11-26 10:57:52","7","500","0","25.5","60","20","ON",,"OFF","OFF","ON"
        """
        df = pd.read_csv(StringIO(csv_data))

    # --- Limpieza ---
    cols_actuadores = ["pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan"]
    existing_cols = [c for c in cols_actuadores if c in df.columns]
    if existing_cols:
        df[existing_cols] = df[existing_cols].fillna("OFF")
        mapping = {'ON': 1, 'OFF': 0}
        for col in existing_cols:
            df[col] = df[col].map(mapping)

    # --- Aumento de Datos (Simulación Sintética) ---
    np.random.seed(42)
    synthetic_data = []
    for _ in range(100):
        temp = np.random.uniform(20, 35)
        hum = np.random.uniform(40, 90)
        fan = 1 if (temp > 28 or hum > 75) else 0

        ph = np.random.uniform(5.0, 8.0)
        tds = np.random.uniform(300, 800)
        reducer = 1 if ph > 6.2 else 0

        synthetic_data.append([ph, tds, temp, hum, reducer, fan])

    df_synth = pd.DataFrame(synthetic_data, columns=['pH', 'TDS', 'DHT_temp', 'DHT_humidity', 'pH_reducer', 'ex_fan'])

    # --- Entrenamiento ---

    # Modelo 1: Regresión Logística
    X_log = df_synth[['DHT_temp', 'DHT_humidity']]
    y_log = df_synth['ex_fan']
    log_reg = LogisticRegression()
    log_reg.fit(X_log, y_log)

    # Modelo 2: Red Neuronal
    X_nn = df_synth[['pH', 'TDS']]
    y_nn = df_synth['pH_reducer']
    scaler = StandardScaler()
    X_nn_scaled = scaler.fit_transform(X_nn)
    nn_model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42)
    nn_model.fit(X_nn_scaled, y_nn)

    return log_reg, nn_model, scaler, df_synth


# Cargar los modelos entrenados
log_reg, nn_model, scaler, df_synth = entrenar_modelos()

# --- INTERFAZ GRÁFICA (FRONTEND) ---

st.title("🌱 SmartPot Dashboard - Control Inteligente")

# Crear layout de 2 columnas (Izquierda Controles, Derecha Gráficos)
col_izq, col_der = st.columns([1, 1.5], gap="large")

with col_izq:
    st.subheader("Panel de Control Manual (Simulación)")

    with st.container(border=True):
        # 1. Sliders (Replican tus tk.Scale)
        t = st.slider("🌡️ Temperatura (°C):", 10.0, 40.0, 25.5, step=0.1)
        h = st.slider("💧 Humedad (%):", 0, 100, 60, step=1)
        ph = st.slider("⚗️ pH del Agua:", 0.0, 14.0, 7.0, step=0.01)
        tds = st.slider("🧂 TDS (ppm):", 0, 1000, 500, step=10)

        st.markdown("---")

        # Botón Analizar
        # En Streamlit el botón devuelve True si se presiona
        boton_analizar = st.button("ANALIZAR DATOS", type="primary")

    # Resultados (Se muestran debajo de los controles)
    st.subheader("Decisión de la IA")

    # Lógica de predicción
    # Usamos dataframes con nombres de columnas correctos para evitar warnings de sklearn
    input_log_df = pd.DataFrame([[t, h]], columns=['DHT_temp', 'DHT_humidity'])
    input_nn_df = pd.DataFrame([[ph, tds]], columns=['pH', 'TDS'])

    # Predecir
    fan_pred = log_reg.predict(input_log_df)[0]
    fan_prob = log_reg.predict_proba(input_log_df)[0][1]

    input_nn_scaled = scaler.transform(input_nn_df)
    ph_pred = nn_model.predict(input_nn_scaled)[0]

    # Mostrar Resultados visualmente
    with st.container(border=True):
        if fan_pred == 1:
            st.error(f"💨 Ventilador: ACTIVADO ({fan_prob * 100:.1f}%)")
        else:
            st.success(f"💨 Ventilador: APAGADO ({fan_prob * 100:.1f}%)")

        if ph_pred == 1:
            st.error("🧪 Reductor pH: DOSIFICAR")
        else:
            st.success("🧪 Reductor pH: ESTABLE")

with col_der:
    st.subheader("Visualización de Modelos")

    # Configurar estilo de gráficas
    plt.style.use('ggplot')

    # Crear Figura (2 subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    # --- Gráfico 1: Scatter (Regresión Logística) ---
    scatter = ax1.scatter(df_synth['DHT_temp'], df_synth['DHT_humidity'],
                          c=df_synth['ex_fan'], cmap='coolwarm', alpha=0.7, edgecolors='k')

    # Destacar el punto actual seleccionado por el usuario
    ax1.scatter([t], [h], color='yellow', s=200, marker='*', label='Tu Selección', edgecolors='black')

    ax1.set_title("Regresión Logística: Zona de Activación")
    ax1.set_xlabel("Temperatura (°C)")
    ax1.set_ylabel("Humedad (%)")
    ax1.legend(*scatter.legend_elements(), title="Ventilador")

    # --- Gráfico 2: Curva de Pérdida (Red Neuronal) ---
    ax2.plot(nn_model.loss_curve_, color='purple', linewidth=2)
    ax2.set_title("Red Neuronal: Curva de Aprendizaje")
    ax2.set_xlabel("Iteraciones")
    ax2.set_ylabel("Error (Loss)")

    plt.tight_layout()

    # Mostrar en Streamlit
    st.pyplot(fig)